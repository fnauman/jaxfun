"""FJ-03: curl / vector-potential PCF-MRI production oracle."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from production.oracles import (
    ProductionOracleNotImplementedError,
    load_resume_checkpoint,
    run_supported_spec,
)

# Solenoidal ceiling for the vector-potential (B = B0 + curl A) family. div B is
# analytically zero, so the oracle holds it at roundoff for the whole horizon
# (measured ~1e-18 GPU / ~1e-16 CPU, growing only linearly with roundoff
# accumulation). The gate sits a few orders above that floor: tight enough that a
# regression into the primitive-`b` finite regime (div B ~ 1e-4..1e-2) is caught
# with margin, loose enough not to flake on benign roundoff growth. The invariant
# that matters is that div B does NOT grow past this ceiling anywhere on the
# horizon -- checked over the whole time series, not just the final scalar.
SOLENOIDAL_CEIL = 1e-12


def _max_div_b_over_horizon(out):
    return max(
        row["divergence_b_l2"] for row in out["time_series"] if "divergence_b_l2" in row
    )


# Final-state scalars that must agree between a straight run and a
# checkpoint+resume continuation of the same spec (growth-family scalars are
# excluded: they are measured from the evolved baseline, which differs by
# construction on a resume).
_CONTINUATION_KEYS = (
    "kinetic_energy",
    "magnetic_energy",
    "total_energy",
    "divergence_u_l2",
    "divergence_b_l2",
    "reynolds_stress",
    "maxwell_stress_xy",
    "total_stress",
    "mean_bx",
    "mean_by",
    "mean_bz",
    "mag_energy_mean",
    "mag_energy_fluct",
)


def _vp_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vector_potential_smoke",
        "spec_hash": "vp-smoke-spec-hash",
        "numerics_contract_version": 2,
        "precision": "float64",
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {
            "type": "gpu_generated_saturated_dns",
            "divergence_b_guard_l2": SOLENOIDAL_CEIL,
        },
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "conducting"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 2e-2,
            "eta_mag": 2e-2,
            "Re": 50.0,
            "Rm": 50.0,
            "Pm": 1.0,
            "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1e-3, "final_time": 0.01},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {
            "artifact_id": "pcf_mri_vector_potential_smoke",
            "regeneration_command": "test-only spec; no committed golden",
        },
    }
    spec["nondimensional_groups"].update(groups)
    spec["forcing"]["B0"] = spec["nondimensional_groups"][
        "B0"
    ]  # keep sources consistent
    return spec


class _GuardProbeSolver:
    dt = 1.0e-3

    def __init__(self):
        self.initial = object()

    def initial_state(self):
        return self.initial


def _guard_probe_scalars(divergence_b_l2):
    return {"divergence_b_l2": divergence_b_l2, "total_energy": 1.0}


def test_vector_potential_oracle_is_solenoidal_by_construction():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    assert sc["divergence_b_guard_l2"] == SOLENOIDAL_CEIL
    # B = curl A -> div B = 0 to roundoff (the invariant the primitive path lacks).
    # It must hold at the solenoidal ceiling at the final step and, more
    # importantly, must not grow past it anywhere on the horizon.
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert _max_div_b_over_horizon(out) < SOLENOIDAL_CEIL
    for key in (
        "kinetic_energy",
        "magnetic_energy",
        "total_stress",
        "alpha_Sh",
        "growth_rate",
    ):
        assert key in sc
    assert len(out["time_series"]) >= 2


@pytest.mark.parametrize("magnetic_bc", ["conducting", "insulating"])
def test_vector_potential_guard_rejects_unsafe_initial_state_before_stepping(
    monkeypatch, magnetic_bc
):
    import production.oracles as oracles

    spec = _vp_spec()
    spec["boundary_conditions"]["magnetic"]["type"] = magnetic_bc
    solver = _GuardProbeSolver()
    monkeypatch.setattr(oracles, "_curl_solver_from_spec", lambda _spec: solver)
    monkeypatch.setattr(
        oracles,
        "_pcf_curl_scalars",
        lambda _solver, _state: _guard_probe_scalars(2.0 * SOLENOIDAL_CEIL),
    )

    def must_not_step(*_args, **_kwargs):  # pragma: no cover - guard must win
        raise AssertionError("unsafe initial state reached the fixed-step driver")

    monkeypatch.setattr(oracles, "_solve_with_optional_checkpoints", must_not_step)
    with pytest.raises(FloatingPointError, match="tstep=0"):
        oracles._run_pcf_vector_potential_mhd_saturation(spec, steps=1)


def test_vector_potential_guard_reaches_fixed_step_block_endpoints(monkeypatch):
    import production.oracles as oracles

    spec = _vp_spec()
    solver = _GuardProbeSolver()
    monkeypatch.setattr(oracles, "_curl_solver_from_spec", lambda _spec: solver)
    monkeypatch.setattr(
        oracles,
        "_pcf_curl_scalars",
        lambda _solver, _state: _guard_probe_scalars(0.0),
    )

    def unsafe_fixed_block(*_args, **kwargs):
        limit = kwargs["magnetic_divergence_limit"]
        assert limit == SOLENOIDAL_CEIL
        oracles._raise_on_divergence_drift(
            None,
            None,
            t=solver.dt,
            tstep=1,
            diagnostics={"divB_L2": 2.0 * limit},
            magnetic_limit=limit,
        )

    monkeypatch.setattr(oracles, "_solve_with_optional_checkpoints", unsafe_fixed_block)
    with pytest.raises(FloatingPointError, match="tstep=1"):
        oracles._run_pcf_vector_potential_mhd_saturation(spec, steps=1)


def test_vector_potential_guard_treats_limit_as_inclusive_ceiling():
    import production.oracles as oracles

    oracles._raise_on_divergence_drift(
        None,
        None,
        t=1.0,
        tstep=1,
        diagnostics={"divB_L2": SOLENOIDAL_CEIL},
        magnetic_limit=SOLENOIDAL_CEIL,
    )
    with pytest.raises(FloatingPointError, match="divergence guard failed"):
        oracles._raise_on_divergence_drift(
            None,
            None,
            t=1.0,
            tstep=1,
            diagnostics={"divB_L2": 2.0 * SOLENOIDAL_CEIL},
            magnetic_limit=SOLENOIDAL_CEIL,
        )


def test_vector_potential_guard_rejects_checkpoint_before_write(tmp_path):
    import production.oracles as oracles
    from jaxfun.io import run_with_cadence

    class UnsafeCheckpointSolver:
        dt = 1.0

        @staticmethod
        def solve(state, steps):
            return int(state) + int(steps)

        def solve_with_cadence(
            self,
            state,
            steps,
            cadence,
            *,
            block_size=1,
            on_diagnostics=None,
            on_snapshot=None,
            on_checkpoint=None,
            should_stop=None,
            t0=0.0,
            tstep0=0,
        ):
            return run_with_cadence(
                self.solve,
                state,
                steps=steps,
                dt=self.dt,
                cadence=cadence,
                block_size=block_size,
                diagnostics=self.diagnostics,
                on_diagnostics=on_diagnostics,
                on_snapshot=on_snapshot,
                on_checkpoint=on_checkpoint,
                should_stop=should_stop,
                t0=t0,
                tstep0=tstep0,
            )

        @staticmethod
        def diagnostics(_state):
            return {"divB_L2": 2.0 * SOLENOIDAL_CEIL}

    with pytest.raises(FloatingPointError, match="tstep=1"):
        oracles._solve_with_optional_checkpoints(
            UnsafeCheckpointSolver(),
            0,
            1,
            spec={
                "problem_id": "unit",
                "spec_hash": "hash",
                "golden": {"artifact_id": "unit"},
            },
            out_dir=tmp_path,
            checkpoint_every=1,
            snapshot_every=None,
            diagnostics_every=None,
            state_kind="unit",
            checkpoint_bank=True,
            magnetic_divergence_limit=SOLENOIDAL_CEIL,
        )

    assert not (tmp_path / "checkpoints" / "checkpoints.h5").exists()
    assert not (tmp_path / "checkpoints" / "bank").exists()


def test_vector_potential_guard_rejects_snapshot_before_write(tmp_path):
    import production.oracles as oracles

    class UnsafeSnapshotSolver:
        dt = 1.0

        def solve_with_cadence(self, state, *_args, **callbacks):
            callbacks["on_snapshot"](1.0, 1, state)
            return state

        @staticmethod
        def diagnostics(_state):
            return {"divB_L2": 2.0 * SOLENOIDAL_CEIL}

    with pytest.raises(FloatingPointError, match="tstep=1"):
        oracles._solve_with_optional_checkpoints(
            UnsafeSnapshotSolver(),
            0,
            1,
            spec={
                "problem_id": "unit",
                "spec_hash": "hash",
                "golden": {"artifact_id": "unit"},
            },
            out_dir=tmp_path,
            checkpoint_every=None,
            snapshot_every=1,
            diagnostics_every=None,
            state_kind="unit",
            magnetic_divergence_limit=SOLENOIDAL_CEIL,
        )

    assert not (tmp_path / "snapshots").exists()


def test_vector_potential_guard_reaches_adaptive_block_endpoints(monkeypatch):
    import production.oracles as oracles

    spec = _vp_spec()
    spec["time"]["adaptive_cfl"] = {"check_every": 1}
    solver = _GuardProbeSolver()
    monkeypatch.setattr(oracles, "_curl_solver_from_spec", lambda _spec: solver)
    monkeypatch.setattr(
        oracles,
        "_pcf_curl_scalars",
        lambda _solver, _state: _guard_probe_scalars(0.0),
    )

    def unsafe_adaptive_block(*_args, **kwargs):
        limit = kwargs["magnetic_divergence_limit"]
        assert limit == SOLENOIDAL_CEIL
        oracles._raise_on_divergence_drift(
            None,
            None,
            t=solver.dt,
            tstep=1,
            diagnostics={"divB_L2": 2.0 * limit},
            magnetic_limit=limit,
        )

    monkeypatch.setattr(oracles, "_run_vp_adaptive_blocks", unsafe_adaptive_block)
    with pytest.raises(FloatingPointError, match="tstep=1"):
        oracles._run_pcf_vector_potential_mhd_saturation(spec, steps=1)


def test_vector_potential_guard_certifies_returned_final_state(monkeypatch):
    import production.oracles as oracles

    spec = _vp_spec()
    solver = _GuardProbeSolver()
    final_state = object()
    monkeypatch.setattr(oracles, "_curl_solver_from_spec", lambda _spec: solver)

    def state_scalars(_solver, state):
        divergence = 2.0 * SOLENOIDAL_CEIL if state is final_state else 0.0
        return _guard_probe_scalars(divergence)

    def unchecked_fixed_driver(*_args, **kwargs):
        assert kwargs["magnetic_divergence_limit"] == SOLENOIDAL_CEIL
        return final_state

    monkeypatch.setattr(oracles, "_pcf_curl_scalars", state_scalars)
    monkeypatch.setattr(
        oracles, "_solve_with_optional_checkpoints", unchecked_fixed_driver
    )
    with pytest.raises(FloatingPointError, match="tstep=1"):
        oracles._run_pcf_vector_potential_mhd_saturation(spec, steps=1)


def test_vector_potential_oracle_is_znf_safe():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(B0=0.0), steps=3)
    sc = out["scalars"]
    # ZNF: no net-flux alpha, but the shear-normalized alpha is present and finite
    assert "transport_alpha" not in sc
    assert "alpha_Sh" in sc

    assert math.isfinite(sc["alpha_Sh"])


def test_vector_potential_emits_flux_diagnostics():
    """FJ-04: the ZNF curl workhorse must expose mean flux + mean/fluct split."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=3)
    sc = out["scalars"]
    for key in (
        "mean_bx",
        "mean_by",
        "mean_bz",
        "mag_energy_mean",
        "mag_energy_fluct",
        "flux_drift_bx",
        "flux_drift_by",
        "flux_drift_bz",
        "mean_bx_trace",
        "mean_by_trace",
        "mean_bz_trace",
        "mean_b_trace_mismatch_linf",
        "electric_ideal_l2",
        "electric_resistive_l2",
        "electric_total_l2",
        "divergence_e_l2",
        "divergence_a_l2",
        "divergence_e_ideal_l2",
        "divergence_e_resistive_l2",
        "electric_wall_tangential_linf",
        "faraday_mean_by_tendency",
        "faraday_mean_bz_tendency",
    ):
        assert key in sc

    runtime_keys = {
        "electric_ideal_l2",
        "electric_resistive_l2",
        "electric_total_l2",
        "divergence_e_l2",
        "electric_wall_tangential_linf",
        "divergence_e_ideal_l2",
        "divergence_e_resistive_l2",
        "faraday_mean_by_tendency",
        "faraday_mean_bz_tendency",
        "mean_b_trace_mismatch_linf",
    }
    assert out["time_series"]
    assert all(runtime_keys <= row.keys() for row in out["time_series"])


def test_vector_potential_energy_split_identity_and_total_field_means():
    """Total-field magnetic semantics (review round 3, blocker 1).

    The split and the component means are TOTAL-field quantities: a net-flux run
    reports mean_bz == B0 (matching shearpy), and
    mag_energy_mean + mag_energy_fluct == magnetic_energy_total exactly, in the
    family's integral_abs2 convention.
    """
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=2)
    sc = out["scalars"]
    assert sc["energy_convention"] == "integral_abs2"
    assert sc["box_volume"] == pytest.approx(2.0 * 4.0 * 1.0)
    # Net-flux run: the physical mean field includes the imposed B0.
    assert sc["mean_bz"] == pytest.approx(0.05, rel=1e-6)
    split = sc["mag_energy_mean"] + sc["mag_energy_fluct"]
    assert sc["magnetic_energy_total"] == pytest.approx(split, rel=1e-9)


def test_vector_potential_znf_split_reduces_to_perturbation_energy():
    """At B0=0 the total field is the perturbation, so the split identity
    coincides with `magnetic_energy` (the original factor-2 regression)."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(B0=0.0), steps=2)
    sc = out["scalars"]
    split = sc["mag_energy_mean"] + sc["mag_energy_fluct"]
    assert sc["magnetic_energy"] == pytest.approx(split, rel=1e-9)
    assert abs(sc["mean_bz"]) < 1e-12


def test_vector_potential_checkpoint_resume_matches_straight_run(tmp_path):
    """FJ-03/FJ-05: MHDState checkpoint + resume reproduces the straight run."""
    jax.config.update("jax_enable_x64", True)
    spec = _vp_spec()

    straight_dir = tmp_path / "straight"
    straight = run_supported_spec(
        spec, steps=4, out_dir=straight_dir, checkpoint_every=2
    )

    parent_dir = tmp_path / "parent"
    run_supported_spec(spec, steps=2, out_dir=parent_dir, checkpoint_every=2)
    record = load_resume_checkpoint(parent_dir)
    assert record.tstep == 2
    resumed = run_supported_spec(spec, steps=4, resume_checkpoint=record)

    for key in _CONTINUATION_KEYS:
        assert np.isclose(
            resumed["scalars"][key],
            straight["scalars"][key],
            rtol=1e-10,
            atol=1e-14,
        ), key
    # The resumed series starts from the checkpointed time, not t=0.
    assert resumed["time_series"][0]["t"] == pytest.approx(2 * spec["time"]["dt"])
    # Both checkpoint files carry the curl state_kind.
    assert str(record.attrs["state_kind"]) == "pcf_vector_potential_mhd_saturation"
    assert (straight_dir / "checkpoints" / "checkpoints.h5").exists()


def test_vector_potential_resume_rejects_spec_hash_mismatch(tmp_path):
    jax.config.update("jax_enable_x64", True)
    parent_dir = tmp_path / "parent"
    run_supported_spec(_vp_spec(), steps=2, out_dir=parent_dir, checkpoint_every=2)
    record = load_resume_checkpoint(parent_dir)

    other = _vp_spec()
    other["spec_hash"] = "a-different-spec-hash"
    with pytest.raises(ValueError, match="spec_hash"):
        run_supported_spec(other, steps=4, resume_checkpoint=record)


def test_vector_potential_quench_continues_with_new_physics(tmp_path):
    """FJ-05: a quench (changed eta/Rm) continues from the parent MHDState."""
    jax.config.update("jax_enable_x64", True)
    parent_dir = tmp_path / "parent"
    parent = run_supported_spec(
        _vp_spec(), steps=2, out_dir=parent_dir, checkpoint_every=2
    )
    record = load_resume_checkpoint(parent_dir)

    child = _vp_spec(Rm=40.0, eta_mag=2.5e-2, Pm=0.8)
    child["spec_hash"] = "vp-smoke-quench-child-hash"
    streamed = []
    out = run_supported_spec(
        child,
        additional_time=2 * child["time"]["dt"],
        resume_checkpoint=record,
        quench=True,
        diagnostics_every=1,
        on_row=streamed.append,
    )
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    assert math.isfinite(sc["magnetic_energy"]) and sc["magnetic_energy"] > 0.0
    # The curl representation stays solenoidal across a quench (changed eta/Rm).
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert _max_div_b_over_horizon(out) < SOLENOIDAL_CEIL
    # FJ-05 baseline: the first series row is the loaded parent state at the
    # checkpoint time, not a fresh seed at t=0.
    first = out["time_series"][0]
    assert first["t"] == pytest.approx(2 * child["time"]["dt"])
    assert out["time_series"][-1]["t"] == pytest.approx(4 * child["time"]["dt"])
    assert out["run_horizon"] == {
        "final_time": pytest.approx(4 * child["time"]["dt"]),
        "final_step": 4,
    }
    assert streamed
    assert all(isinstance(row["tstep"], int) for row in streamed)
    assert first["magnetic_energy"] == pytest.approx(
        parent["scalars"]["magnetic_energy"], rel=1e-10
    )


def test_vector_potential_snapshot_writes_fields(tmp_path):
    jax.config.update("jax_enable_x64", True)
    out_dir = tmp_path / "run"
    run_supported_spec(_vp_spec(), steps=2, out_dir=out_dir, snapshot_every=2)
    snapshot_index = out_dir / "snapshots" / "snapshots.h5"
    assert snapshot_index.exists()
    assert (out_dir / "snapshots" / "steps" / "snapshot_00000002.h5").exists()
    assert (out_dir / "snapshots" / "manifest.json").exists()

    import h5py

    with h5py.File(out_dir / "snapshots" / "steps" / "snapshot_00000002.h5", "r") as h5:
        names = set()
        h5.visit(names.add)
        joined = " ".join(names)
        for field in ("u_x", "u_y", "u_z", "b_x", "b_y", "b_z"):
            assert field in joined


@pytest.mark.parametrize("integrator", ["analytic", "linear_eigenproblem"])
def test_vector_potential_rejects_non_time_stepping_integrator_early(
    integrator,
) -> None:
    spec = _vp_spec()
    spec["time"]["integrator"] = integrator

    with pytest.raises(
        ProductionOracleNotImplementedError,
        match=r"PCF KMM production DNS requires a time-stepping integrator",
    ):
        run_supported_spec(spec, steps=0)
