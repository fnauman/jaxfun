"""Regression coverage for the PCF zero-net-flux MRI scout diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax
from production import sweep
from production.adapters import config_from_spec
from production.problem_spec import load_spec
from production.profiles import (
    MULTIPLANE_PROFILE_CHANNELS,
    pcf_multiplane_profiles,
    write_pcf_multiplane_h5,
)

ROOT = Path(__file__).resolve().parents[2]
SCOUT = ROOT / "production" / "runs" / "pcf_mri_znf_scout_v1.json"


def _solver(family: str = "L") -> PlaneCouetteMRIShearpyJax:
    return PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        family=family,
        dt=1.0e-3,
        background_b=(0.0, 0.0, 0.0),
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.1,
        magnetic_seed="sinusoidal_bz_x",
        solenoidal_velocity_seed=True,
    )


def test_znf_seed_is_solenoidal_and_has_zero_mean_vertical_flux() -> None:
    solver = _solver()
    state = solver.initial_state()
    diagnostics = solver.diagnostics(state)
    magnetic = solver._backward_B(solver.update_B_from_A(state.A), padded=False)
    x = np.asarray(solver.X[0][:, 0, 0])

    assert float(diagnostics["divL2"]) < 1.0e-12
    assert float(diagnostics["divB_L2"]) < 1.0e-12
    assert float(diagnostics["mean_bz"]) == pytest.approx(0.0, abs=1.0e-12)
    assert np.max(np.abs(np.asarray(magnetic[0]))) < 1.0e-12
    assert np.max(np.abs(np.asarray(magnetic[1]))) < 1.0e-12
    assert np.asarray(magnetic[2][:, 0, 0]) == pytest.approx(
        0.1 * np.sin(np.pi * x), abs=5.0e-5
    )
    assert bool(jnp.isfinite(diagnostics["channel_kz1_total_rms"]))


def test_chebyshev_weighted_seed_can_hide_zero_physical_flux() -> None:
    """The ZNF sine seed cannot distinguish physical and Chebyshev measures.

    Use an asymmetric-in-measure represented field with exact zero physical
    flux: Ay=(x^3-x)/3, hence Bz=x^2-1/3 and integral Bz dx=0. The obsolete
    Gauss-Chebyshev diagnostic reports pi/12 instead.
    """
    solver = _solver(family="C")
    initial = solver.initial_state()
    x, y, z = solver.X
    ay = (x**3 - x) / 3.0 + 0.0 * y + 0.0 * z
    zeros = jnp.zeros_like(ay)
    state = type(initial)(
        flow=initial.flow,
        A=solver._A_state_from_physical((zeros, ay, zeros)),
    )

    diagnostics = solver.diagnostics(state)
    B = solver.update_B_from_A(state.A)
    bz = jnp.real(solver._backward_B(B, padded=False)[2])
    wx = solver.TC.basespaces[0].quadrature_weights()
    wy = solver.TC.basespaces[1].integration_weights()
    wz = solver.TC.basespaces[2].integration_weights()
    volume = (
        (solver.domain[0][1] - solver.domain[0][0])
        * (solver.domain[1][1] - solver.domain[1][0])
        * (solver.domain[2][1] - solver.domain[2][0])
    )
    obsolete_weighted_mean = (
        jnp.sum(bz * wx[:, None, None] * wy[None, :, None] * wz[None, None, :]) / volume
    )

    assert float(obsolete_weighted_mean) == pytest.approx(np.pi / 12.0, abs=1e-12)
    assert abs(float(diagnostics["mean_bz"])) < 1.0e-14
    assert abs(float(diagnostics["mean_bz_trace"])) < 1.0e-14
    assert float(diagnostics["mean_b_trace_mismatch_linf"]) < 1.0e-14


@pytest.mark.parametrize("family", ("C", "L"))
def test_znf_flux_and_electric_health_diagnostics(family: str) -> None:
    solver = _solver(family)
    state = solver.initial_state()
    diagnostics = solver.diagnostics(state)

    for key in (
        "electric_ideal_l2",
        "electric_resistive_l2",
        "electric_total_l2",
        "divergence_e_l2",
        "divergence_a_l2",
        "divergence_e_ideal_l2",
        "divergence_e_resistive_l2",
        "divergence_a_wall_linf",
        "electric_ideal_wall_tangential_linf",
        "electric_resistive_wall_tangential_linf",
        "electric_wall_tangential_linf",
        "faraday_mean_by_tendency",
        "faraday_mean_bz_tendency",
        "mean_bx_trace",
        "mean_by_trace",
        "mean_bz_trace",
        "mean_b_trace_mismatch_linf",
    ):
        assert key in diagnostics
        assert bool(jnp.isfinite(diagnostics[key]))

    ideal, resistive, electric = solver.electric_field_parts(state)
    for ei, er, et in zip(ideal, resistive, electric, strict=True):
        assert np.asarray(et) == pytest.approx(np.asarray(ei + er), abs=1.0e-14)

    assert float(diagnostics["mean_bx_trace"]) == pytest.approx(0.0, abs=1.0e-14)
    assert float(diagnostics["mean_by_trace"]) == pytest.approx(0.0, abs=1.0e-14)
    assert float(diagnostics["mean_bz_trace"]) == pytest.approx(0.0, abs=1.0e-14)
    assert float(diagnostics["mean_b_trace_mismatch_linf"]) < 1.0e-13
    assert float(diagnostics["electric_ideal_wall_tangential_linf"]) < 1.0e-13
    # The lower bound is tied to this test's nonzero seed amplitude and eta;
    # it ensures the unprojected resistive wall residual remains observable.
    assert float(diagnostics["electric_resistive_wall_tangential_linf"]) > 1.0e-6
    assert float(diagnostics["electric_wall_tangential_linf"]) == pytest.approx(
        float(diagnostics["electric_resistive_wall_tangential_linf"]),
        rel=1.0e-10,
    )
    assert abs(float(diagnostics["faraday_mean_by_tendency"])) < 1.0e-13
    assert abs(float(diagnostics["faraday_mean_bz_tendency"])) < 1.0e-13


@pytest.mark.parametrize("family", ("C", "L"))
def test_znf_flux_remains_machine_zero_after_a_step(family: str) -> None:
    solver = _solver(family)
    state = solver.step(solver.initial_state())
    diagnostics = solver.diagnostics(state)

    for key in ("mean_bx_trace", "mean_by_trace", "mean_bz_trace"):
        assert abs(float(diagnostics[key])) < 1.0e-13
    assert float(diagnostics["mean_b_trace_mismatch_linf"]) < 1.0e-12


def test_multiplane_v2_shapes_and_append_layout(tmp_path: Path) -> None:
    solver = _solver()
    profiles = pcf_multiplane_profiles(solver, solver.initial_state())

    assert profiles["channels"] == MULTIPLANE_PROFILE_CHANNELS
    assert profiles["z_profile"].shape == (33, 8)
    assert profiles["xy"].shape == (33, 9, 8)
    assert profiles["xz"].shape == (33, 9, 8)
    assert profiles["yz"].shape == (33, 8, 8)
    assert all(
        np.all(np.isfinite(profiles[name])) for name in ("z_profile", "xy", "xz", "yz")
    )
    assert all(np.isrealobj(profiles[name]) for name in ("z_profile", "xy", "xz", "yz"))

    path = tmp_path / "profiles" / "multiplane_v2.h5"
    write_pcf_multiplane_h5(path, profiles=profiles, t=0.0, tstep=0)
    write_pcf_multiplane_h5(path, profiles=profiles, t=0.1, tstep=10)

    with h5py.File(path, "r") as handle:
        group = handle["multiplane_profiles"]
        assert handle["time"][...].tolist() == [0.0, 0.1]
        assert handle["tstep"][...].tolist() == [0, 10]
        assert group["z_profile"].shape == (2, 33, 8)
        assert group["xy"].shape == (2, 33, 9, 8)
        assert group["xz"].shape == (2, 33, 9, 8)
        assert group["yz"].shape == (2, 33, 8, 8)
        assert all(
            group[name].dtype.kind == "f" for name in ("z_profile", "xy", "xz", "yz")
        )
        assert handle["meta"].attrs["format_version"] == 2
        assert json.loads(group.attrs["channels_json"]) == list(
            MULTIPLANE_PROFILE_CHANNELS
        )


def test_scout_spec_is_wired_to_vector_potential_sources() -> None:
    spec = load_spec(SCOUT)
    effective = config_from_spec(spec, resolution_tier="smoke")

    assert effective.spec["nondimensional_groups"]["B0"] == 0.0
    assert effective.spec["nondimensional_groups"]["Omega"] == pytest.approx(2.0 / 3.0)
    assert effective.spec["nondimensional_groups"]["S"] == 1.0
    assert effective.spec["initial_condition"]["magnetic_seed"] == "sinusoidal_bz_x"
    assert effective.spec["initial_condition"]["solenoidal_velocity_seed"] is True
    assert effective.metadata["solver_source_files"] == [
        "examples/pcf_mhd_jax.py",
        "examples/pcf_mhd_mri_shearpy_jax.py",
    ]


def test_set_execute_dispatches_a_one_point_sweep(tmp_path: Path, monkeypatch) -> None:
    observed = {}

    def fake_execute(base, grid, out, **kwargs):
        observed.update(base=base, grid=grid, out=out, kwargs=kwargs)
        return {
            "points": 1,
            "completed": 1,
            "failed": 0,
            "skipped": 0,
            "index_path": str(tmp_path / "sweep_index.json"),
        }

    monkeypatch.setattr(sweep, "execute_sweep", fake_execute)
    result = sweep.main(
        [
            "--base",
            str(SCOUT),
            "--out",
            str(tmp_path),
            "--set",
            "Rm_h=6000",
            "--execute",
            "--resolution-tier",
            "smoke",
            "--steps",
            "2",
            "--checkpoint-every",
            "10",
            "--snapshot-every",
            "20",
            "--profiles-every",
            "5",
            "--diagnostics-every",
            "2",
        ]
    )

    assert result == 0
    assert observed["grid"] == {"Rm_h": [6000]}
    assert observed["kwargs"] == {
        "execute": True,
        "resolution_tier": "smoke",
        "steps": 2,
        "checkpoint_every": 10,
        "snapshot_every": 20,
        "profiles_every": 5,
        "diagnostics_every": 2,
        "wandb": False,
    }
