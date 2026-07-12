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
)
from production.oracles import run_supported_spec

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
    assert adaptive_cfl_from_spec({"time": {"adaptive_cfl": True}}) is not None


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


def test_adaptive_run_shrinks_an_oversized_dt_and_stays_solenoidal():
    """Start with a dt whose rotation CFL breaks the target: the driver must
    shrink it, the recorded history must say so, and the vector-potential
    solenoidal contract must hold across the operator rebuilds."""
    jax.config.update("jax_enable_x64", True)
    # 2*Omega*dt = 0.27 at dt=0.2 -> cfl_total ~ 0.27 > target 0.15.
    spec = _vp_spec(
        {
            "integrator": "IMEXRK222",
            "dt": 0.2,
            "final_time": 24.0,
            "adaptive_cfl": {
                "target": 0.15,
                "check_every": 5,
                "dt_min": 1e-5,
                "dt_max": 0.5,
                "safety": 0.9,
            },
        }
    )
    out = run_supported_spec(spec, steps=30, diagnostics_every=5)
    sc = out["scalars"]
    assert sc["n_dt_changes"] >= 1
    assert sc["dt_final"] < 0.2
    assert sc["dt_min_used"] < 0.2
    assert sc["adaptive_cfl_target"] == pytest.approx(0.15)
    dts = [row["dt"] for row in out["time_series"] if "dt" in row]
    assert min(dts) < 0.2
    cfls = [row["cfl_total"] for row in out["time_series"] if "cfl_total" in row]
    # After adaptation the measured CFL sits at/below the target band.
    assert cfls[-1] <= 0.15 * 1.05
    assert (
        max(
            row["divergence_b_l2"]
            for row in out["time_series"]
            if "divergence_b_l2" in row
        )
        < SOLENOIDAL_CEIL
    )


def test_adaptive_run_grows_an_undersized_dt_up_to_the_cap():
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
    assert sc["dt_final"] == pytest.approx(4.0e-3)  # capped by dt_max
    assert sc["cfl_total_max_observed"] < 0.3
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL


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
