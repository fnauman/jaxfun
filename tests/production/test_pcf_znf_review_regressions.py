"""Regressions for review feedback on the PCF ZNF diagnostics PR."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax
from production.frontier import (
    FRONTIER_SCHEMA_VERSION,
    _payload_hash,
    execute_frontier_sweep,
)
from production.problem_spec import load_spec
from production.profiles import (
    MULTIPLANE_PROFILE_CHANNELS,
    pcf_multiplane_profiles,
    truncate_pcf_multiplane_h5,
    write_pcf_multiplane_h5,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTIER_BASE = ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"


def test_state_from_physical_preserves_raw_transverse_transforms() -> None:
    solver = PlaneCouetteFluctuationJax(N=(9, 8, 8), family="L")
    x, _, z = solver.X
    wall = 1.0 - x**2
    zeros = jnp.zeros(solver.TD.num_quad_points)
    physical = (
        zeros + 0.01 * wall * jnp.cos(z),
        zeros + 0.02 * wall * jnp.sin(z),
        zeros + 0.03 * wall * jnp.cos(2.0 * z),
    )

    state = solver.state_from_physical(physical)
    expected = (
        solver.TB.mask_nyquist(solver.TB.forward(physical[0])),
        solver.TD.mask_nyquist(solver.TD.forward(physical[1])),
        solver.TD.mask_nyquist(solver.TD.forward(physical[2])),
    )

    for actual, raw_transform in zip(state.u, expected, strict=True):
        np.testing.assert_allclose(actual, raw_transform, rtol=0.0, atol=0.0)


def test_opt_in_scout_velocity_seed_is_solenoidal() -> None:
    solver = PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        family="L",
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.0,
        background_b=(0.0, 0.0, 0.0),
        solenoidal_velocity_seed=True,
    )
    state = solver.initial_state()
    assert float(solver.divergence_l2(state.flow)) < 1.0e-12


def test_base_vector_potential_mhd_profiles_do_not_require_mri_helpers() -> None:
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        family="L",
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.05,
    )
    state = solver.initial_state()
    profiles = pcf_multiplane_profiles(solver, state)

    assert profiles["z_profile"].shape == (33, 8)
    assert all(
        np.all(np.isfinite(profiles[name])) for name in ("z_profile", "xy", "xz", "yz")
    )


def test_profile_curl_channels_reuse_solver_vorticity_and_current() -> None:
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        family="L",
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.05,
    )
    state = solver.initial_state()
    profiles = pcf_multiplane_profiles(solver, state)
    omega = solver.velocity_vorticity_physical(state.flow.u)
    magnetic = solver.update_B_from_A(state.A)
    current = solver._backward_J(solver.update_J_from_B(magnetic))

    for offset, field in enumerate(omega):
        index = MULTIPLANE_PROFILE_CHANNELS.index("omega_x") + offset
        np.testing.assert_allclose(
            profiles["xy"][index], np.mean(np.asarray(field), axis=2), atol=1.0e-12
        )
    for offset, field in enumerate(current):
        index = MULTIPLANE_PROFILE_CHANNELS.index("j_x") + offset
        np.testing.assert_allclose(
            profiles["xy"][index], np.mean(np.asarray(field), axis=2), atol=1.0e-12
        )


def test_profile_emf_includes_base_shear_velocity() -> None:
    solver = PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        family="L",
        shear_rate=1.0,
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.0,
        background_b=(0.0, 0.0, 0.1),
    )
    state = solver.initial_state()
    profiles = pcf_multiplane_profiles(solver, state)
    total_velocity = solver.total_velocity_physical(state.flow)
    expected_emf_x = np.mean(np.asarray(total_velocity[1]) * 0.1, axis=2)
    index = MULTIPLANE_PROFILE_CHANNELS.index("emf_x")

    np.testing.assert_allclose(profiles["xy"][index], expected_emf_x, atol=1.0e-12)
    assert np.max(np.abs(expected_emf_x)) > 0.05


def test_profile_writer_replaces_retraversed_steps_and_truncates_resume_tail(
    tmp_path: Path,
) -> None:
    solver = PlaneCouetteMHDJax(N=(9, 8, 8), family="L")
    profiles = pcf_multiplane_profiles(solver, solver.initial_state())
    path = tmp_path / "profiles.h5"
    for step in (10, 20, 30):
        write_pcf_multiplane_h5(path, profiles=profiles, t=step / 10, tstep=step)

    write_pcf_multiplane_h5(path, profiles=profiles, t=2.1, tstep=20)
    with h5py.File(path, "r") as handle:
        assert handle["tstep"][...].tolist() == [10, 20]
        assert handle["time"][...].tolist() == [1.0, 2.1]

    write_pcf_multiplane_h5(path, profiles=profiles, t=3.0, tstep=30)
    truncate_pcf_multiplane_h5(path, after_tstep=20)
    with h5py.File(path, "r") as handle:
        assert handle["tstep"][...].tolist() == [10, 20]
        lengths = [handle[name].shape[0] for name in ("time", "tstep")]
        group = handle["multiplane_profiles"]
        lengths.extend(group[name].shape[0] for name in ("z_profile", "xy", "xz", "yz"))
        assert lengths == [2, 2, 2, 2, 2, 2]


def test_default_frontier_hash_matches_pre_cadence_lineage(tmp_path: Path) -> None:
    def classified_runner(*, config_path, **_kwargs):
        spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
        scientific_class = (
            "decayed" if spec["nondimensional_groups"]["Rm"] < 600 else "growing"
        )
        slope = -0.1 if scientific_class == "decayed" else 0.1
        return {
            "execution": {"status": "completed"},
            "classification": {
                "scientific_class": scientific_class,
                "reason": "test classification",
                "fit": {"slope": slope, "stderr": 0.01},
            },
        }

    summary = execute_frontier_sweep(
        FRONTIER_BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path,
        abs_tolerance=25.0,
        max_refinements=0,
        runner=classified_runner,
    )
    record = json.loads(Path(summary["lineage_path"]).read_text(encoding="utf-8"))[0]
    base_spec = load_spec(FRONTIER_BASE)
    legacy_hash = _payload_hash(
        {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "base_spec_hash": base_spec["spec_hash"],
            "axis": "Rm_h",
            "bounds": [400.0, 800.0],
            "fixed_overrides": {},
            "abs_tolerance": 25.0,
            "rel_tolerance": 0.0,
            "confidence_z": 1.96,
            "max_refinements": 0,
            "resolution_tier": None,
            "steps": None,
        }
    )

    assert record["request_hash"] == legacy_hash


def test_explicit_profile_cadence_changes_frontier_request_hash(tmp_path: Path) -> None:
    def classified_runner(**_kwargs):
        return {
            "execution": {"status": "completed"},
            "classification": {
                "scientific_class": "decayed",
                "reason": "test classification",
                "fit": {"slope": -0.1, "stderr": 0.01},
            },
        }

    first = execute_frontier_sweep(
        FRONTIER_BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path / "default",
        abs_tolerance=25.0,
        max_refinements=0,
        runner=classified_runner,
    )
    second = execute_frontier_sweep(
        FRONTIER_BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path / "profiles",
        abs_tolerance=25.0,
        max_refinements=0,
        profiles_every=10,
        runner=classified_runner,
    )
    first_record = json.loads(Path(first["lineage_path"]).read_text())[0]
    second_record = json.loads(Path(second["lineage_path"]).read_text())[0]

    assert first_record["request_hash"] != second_record["request_hash"]
