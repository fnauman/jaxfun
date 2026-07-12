"""Adaptive-CFL production stepping (chunked dt adaptation).

The driver measures the explicit CFL from the family health scalars between
compiled blocks and rebuilds the implicit factorizations via
``solver.set_dt`` when the step leaves the configured band.  Physics
contracts (solenoidality, finiteness) must hold across dt changes, elapsed
time is accounted exactly, and every adaptation is recorded.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from production.adaptive import (
    AdaptiveCFLConfig,
    _proposed_dt,
    adaptive_cfl_from_spec,
    run_adaptive_cfl,
)
from production.oracles import _adaptive_elapsed_target, run_supported_spec

SOLENOIDAL_CEIL = 1.0e-12


def _vp_spec(time_block):
    return {
        "problem_id": "vp_adaptive_smoke",
        "spec_hash": "vp-adaptive-smoke-hash",
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "conducting"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 2.0e-2,
            "eta_mag": 2.0e-2,
            "B0": 0.05,
        },
        "time": time_block,
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {"artifact_id": "vp_adaptive_smoke", "regeneration_command": "test"},
    }


def test_config_validation_rejects_nonsense():
    with pytest.raises(ValueError, match="target"):
        AdaptiveCFLConfig(target=1.5)
    with pytest.raises(ValueError, match="dt_min"):
        AdaptiveCFLConfig(dt_min=1e-2, dt_max=1e-3)
    with pytest.raises(ValueError, match="growth_cap"):
        AdaptiveCFLConfig(growth_cap=0.9)
    assert adaptive_cfl_from_spec({"time": {}}) is None
    assert adaptive_cfl_from_spec({"time": {"adaptive_cfl": False}}) is None
    assert adaptive_cfl_from_spec({"time": {"adaptive_cfl": True}}) is not None
    assert adaptive_cfl_from_spec({"time": {"adaptive_cfl": {}}}) == (
        AdaptiveCFLConfig()
    )
    with pytest.raises(ValueError, match="boolean or an object"):
        adaptive_cfl_from_spec({"time": {"adaptive_cfl": None}})


def test_proposed_dt_shrinks_grows_and_holds():
    config = AdaptiveCFLConfig(
        target=0.4, safety=0.9, dt_min=1e-6, dt_max=1.0, growth_cap=1.5
    )
    # Over target: immediate shrink toward safety * dt_at_target.
    new = _proposed_dt(1.0e-2, 0.8, config)
    assert new == pytest.approx(0.9 * 1.0e-2 * 0.4 / 0.8)
    # Inside the hysteresis band: hold.
    assert _proposed_dt(1.0e-2, 0.3, config) is None
    # Far below target: damped growth, capped by growth_cap.
    new = _proposed_dt(1.0e-2, 0.05, config)
    assert new == pytest.approx(1.5e-2)
    # dt_max cap engages.
    config2 = AdaptiveCFLConfig(target=0.4, dt_min=1e-6, dt_max=1.2e-2)
    assert _proposed_dt(1.0e-2, 0.05, config2) == pytest.approx(1.2e-2)
    # Nonfinite CFL: hold rather than act on garbage.
    assert _proposed_dt(1.0e-2, float("nan"), config) is None


def test_adaptive_horizon_uses_final_time_without_initial_dt_quantization():
    spec = {"time": {"dt": 0.3, "final_time": 1.0}}
    assert _adaptive_elapsed_target(spec, steps=None, t0=0.0) == pytest.approx(1.0)
    assert _adaptive_elapsed_target(spec, steps=None, t0=0.4) == pytest.approx(0.6)
    assert _adaptive_elapsed_target(spec, steps=2, t0=0.0) == pytest.approx(0.6)


def test_adaptive_driver_aborts_after_a_newly_unsafe_state():
    class Solver:
        def __init__(self):
            self.dt = 1.0
            self.solve_dts = []

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, state, steps):
            self.solve_dts.append(self.dt)
            return 1.2

    solver = Solver()
    config = AdaptiveCFLConfig(
        target=0.5,
        safety=0.9,
        dt_min=0.1,
        dt_max=2.0,
        check_every=1,
    )
    with pytest.raises(RuntimeError, match="completed block cannot be repaired"):
        run_adaptive_cfl(
            solver,
            0.4,
            elapsed_target=1.5,
            config=config,
            health_scalars_fn=lambda current_solver, state: {
                "cfl_total": float(state) * current_solver.dt
            },
        )
    assert solver.solve_dts == [pytest.approx(1.0)]


def test_adaptive_driver_shrinks_a_safe_over_target_state():
    class Solver:
        def __init__(self):
            self.dt = 1.0
            self.solve_dts = []

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, _state, _steps):
            self.solve_dts.append(self.dt)
            return 0.8

    solver = Solver()
    _, record = run_adaptive_cfl(
        solver,
        0.4,
        elapsed_target=1.5625,
        config=AdaptiveCFLConfig(
            target=0.5, safety=0.9, dt_min=0.1, dt_max=2.0, check_every=1
        ),
        health_scalars_fn=lambda current_solver, state: {
            "cfl_total": float(state) * current_solver.dt
        },
    )
    assert solver.solve_dts == [pytest.approx(1.0), pytest.approx(0.5625)]
    cfl_change = next(
        change for change in record["dt_changes"] if change["reason"] == "cfl"
    )
    assert cfl_change["cfl_total"] == pytest.approx(0.8)
    assert cfl_change["cfl_total_projected"] == pytest.approx(0.45)


def test_adaptive_reporting_excludes_a_preflight_dt_that_was_never_used():
    class Solver:
        def __init__(self):
            self.dt = 1.0
            self.solve_dts = []

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, state, _steps):
            self.solve_dts.append(self.dt)
            return state

    solver = Solver()
    _, record = run_adaptive_cfl(
        solver,
        1.0,
        elapsed_target=0.45,
        config=AdaptiveCFLConfig(
            target=0.5, safety=0.9, dt_min=0.1, dt_max=2.0, check_every=2
        ),
        health_scalars_fn=lambda current_solver, state: {
            "cfl_total": 2.0 * float(state) * current_solver.dt
        },
    )
    assert solver.solve_dts == [pytest.approx(0.225)]
    assert record["dt_min_used"] == pytest.approx(0.225)
    assert record["dt_max_used"] == pytest.approx(0.225)


def test_final_time_redistribution_never_steps_below_dt_min():
    class Solver:
        def __init__(self):
            self.dt = 1.0
            self.solve_calls = []

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, state, steps):
            self.solve_calls.append((self.dt, steps))
            return state

    solver = Solver()
    _, record = run_adaptive_cfl(
        solver,
        None,
        elapsed_target=1.1,
        config=AdaptiveCFLConfig(target=0.5, dt_min=0.4, dt_max=2.0, check_every=2),
        health_scalars_fn=lambda current_solver, _state: {
            "cfl_total": 0.3 * current_solver.dt
        },
    )
    assert solver.solve_calls == [(pytest.approx(0.55), 2)]
    assert record["elapsed_time"] == pytest.approx(1.1)
    assert record["dt_min_used"] == pytest.approx(0.55)
    assert record["dt_last_used"] == pytest.approx(0.55)
    assert record["dt_final"] == pytest.approx(1.0)
    assert solver.dt == pytest.approx(1.0)
    change = record["dt_changes"][-1]
    assert change["reason"] == "final_time_redistribution"
    assert change["cfl_total_projected"] == pytest.approx(0.165)


def test_final_clip_projects_cfl_from_old_dt_and_restores_controller_dt():
    class Solver:
        def __init__(self):
            self.dt = 1.0

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, state, _steps):
            return state

    solver = Solver()
    _, record = run_adaptive_cfl(
        solver,
        None,
        elapsed_target=1.5,
        config=AdaptiveCFLConfig(target=0.5, dt_min=0.1, dt_max=2.0, check_every=1),
        health_scalars_fn=lambda current_solver, _state: {
            "cfl_total": 0.3 * current_solver.dt
        },
    )
    clip = next(
        change
        for change in record["dt_changes"]
        if change["reason"] == "final_time_clip"
    )
    assert clip["cfl_total"] == pytest.approx(0.3)
    assert clip["cfl_total_projected"] == pytest.approx(0.15)
    assert record["dt_last_used"] == pytest.approx(0.5)
    assert record["dt_final"] == pytest.approx(1.0)
    assert solver.dt == pytest.approx(1.0)


def test_exact_horizon_shorter_than_dt_min_is_rejected_before_solving():
    class Solver:
        dt = 1.0

        def set_dt(self, dt):  # pragma: no cover - must not be reached
            self.dt = float(dt)

        def solve(self, state, steps):  # pragma: no cover - must not be reached
            raise AssertionError("sub-floor horizon advanced a solver step")

    with pytest.raises(ValueError, match="smaller than dt_min"):
        run_adaptive_cfl(
            Solver(),
            None,
            elapsed_target=0.05,
            config=AdaptiveCFLConfig(dt_min=0.1, dt_max=2.0),
            health_scalars_fn=lambda _solver, _state: {"cfl_total": 0.2},
        )


@pytest.mark.parametrize("cadence", ["checkpoint_every", "snapshot_every"])
def test_adaptive_cadence_flags_are_rejected_explicitly(tmp_path, cadence):
    spec = _vp_spec(
        {
            "integrator": "IMEXRK222",
            "dt": 1.0e-3,
            "final_time": 0.01,
            "adaptive_cfl": {},
        }
    )
    with pytest.raises(NotImplementedError, match="does not yet support"):
        run_supported_spec(spec, steps=1, out_dir=tmp_path, **{cadence: 1})


def test_adaptive_driver_refuses_an_unsafe_dt_min_before_solving():
    class Solver:
        dt = 0.1

        def set_dt(self, dt):
            self.dt = float(dt)

        def solve(self, state, steps):  # pragma: no cover - must not be reached
            raise AssertionError("unsafe dt_min advanced a solver step")

    with pytest.raises(RuntimeError, match="dt_min"):
        run_adaptive_cfl(
            Solver(),
            None,
            elapsed_target=0.1,
            config=AdaptiveCFLConfig(target=0.5, dt_min=0.1, dt_max=1.0, check_every=1),
            health_scalars_fn=lambda _solver, _state: {"cfl_total": 2.0},
        )


@pytest.mark.integration
def test_adaptive_run_shrinks_an_unsafe_dt_before_the_first_block():
    """Start with a dt whose rotation CFL exceeds 1 (the health-gate abort
    threshold): the driver must shrink it *before* advancing any block --
    otherwise the production resolution gate would abort mid-run -- land
    exactly on the requested elapsed time, and hold the solenoidal contract
    across the operator rebuilds."""
    jax.config.update("jax_enable_x64", True)
    # 2*Omega*dt = 1.6 at dt=1.2 -> cfl_total > 1 on the *initial* state.
    spec = _vp_spec(
        {
            "integrator": "IMEXRK222",
            "dt": 1.2,
            "final_time": 24.0,
            "adaptive_cfl": {
                "target": 0.15,
                "check_every": 5,
                "dt_min": 1e-5,
                "dt_max": 2.0,
                "safety": 0.9,
            },
        }
    )
    out = run_supported_spec(spec, steps=8, diagnostics_every=5)
    sc = out["scalars"]
    assert sc["n_dt_changes"] >= 1
    assert sc["dt_final"] < 1.2
    assert sc["dt_min_used"] < 1.2
    assert sc["dt_max_used"] < 1.2  # pre-flight dt was never advanced
    assert sc["adaptive_cfl_target"] == pytest.approx(0.15)
    # The unsafe initial dt must never evolve a block: every *observed* CFL
    # (all measured on evolved states or the pre-flight initial state) stays
    # in the safe band after the pre-flight shrink.
    cfls = [row["cfl_total"] for row in out["time_series"] if "cfl_total" in row]
    assert max(cfls) <= 0.15 * 1.05
    # The run lands exactly on the requested elapsed time (8 steps * dt=1.2).
    assert out["time_series"][-1]["t"] == pytest.approx(8 * 1.2, rel=1e-12)
    assert sc["adaptive_final_step_clipped"] in (True, False)
    assert sc["adaptive_steps_taken"] > 8  # shrunk dt -> more, smaller steps
    assert (
        max(
            row["divergence_b_l2"]
            for row in out["time_series"]
            if "divergence_b_l2" in row
        )
        < SOLENOIDAL_CEIL
    )


@pytest.mark.integration
def test_adaptive_run_grows_an_undersized_dt_and_keeps_the_final_time():
    jax.config.update("jax_enable_x64", True)
    spec = _vp_spec(
        {
            "integrator": "IMEXRK222",
            "dt": 1.0e-3,
            "final_time": 0.1,
            "adaptive_cfl": {
                "target": 0.3,
                "check_every": 5,
                "dt_min": 1e-5,
                "dt_max": 4.0e-3,
                "safety": 0.9,
                "growth_cap": 2.0,
            },
        }
    )
    out = run_supported_spec(spec, steps=25, diagnostics_every=5)
    sc = out["scalars"]
    assert sc["n_dt_changes"] >= 1
    assert sc["dt_max_used"] == pytest.approx(4.0e-3)  # capped by dt_max
    assert sc["cfl_total_max_observed"] < 0.3
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    # A grown dt takes fewer steps but must NOT overshoot the requested
    # elapsed time (25 steps * dt=1e-3): the final block/step is clipped.
    assert sc["adaptive_steps_taken"] < 25
    assert out["time_series"][-1]["t"] == pytest.approx(0.025, rel=1e-12)
    assert sc["adaptive_final_step_clipped"] is True
    assert sc["dt_final"] == pytest.approx(sc["dt_max_used"])
    assert sc["dt_last_used"] == pytest.approx(out["time_series"][-1]["dt"])
    assert sc["dt_min_used"] <= sc["dt_last_used"] < sc["dt_final"]


@pytest.mark.integration
def test_tc_vp_set_dt_recombination_preserves_the_contract():
    """The TC vector-potential family rebuilds its per-mode LU from cached
    mass/physics parts on set_dt; the solenoidal witness and the insulating
    matching rows must hold across the change (including the dt-dependent
    trapped-flux Faraday row)."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=20,
        Ntheta=2,
        Nz=8,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="insulating",
    )
    state, _ = solver.seed_linear_eigenmode(m=0, kz_mode=1, amp=1e-6)
    state = solver.solve(state, 10)
    solver.set_dt(5e-4)
    import dataclasses

    state = dataclasses.replace(state, have_old=False)  # AB2 bootstrap
    state = solver.solve(state, 10)
    diag = solver.diagnostics(state)
    assert float(diag["divb_l2"]) < SOLENOIDAL_CEIL
    assert float(diag["insulating_bc_residual"]) < SOLENOIDAL_CEIL
    assert np.isfinite(float(diag["E"]))
