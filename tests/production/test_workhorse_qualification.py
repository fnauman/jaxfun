"""Review round 3, blocker 4: CPU-side nonlinear qualification of the workhorse.

A longer nonlinear run of the vector-potential family must hold the physics
contract end to end: solenoidality at roundoff, a closing shearing-box energy
budget, sane CFL/spectral health, and silent guards. The campaign-scale GPU
benchmark artifacts remain tracked in production/KNOWN_ISSUES.md (KI-3).
"""

from __future__ import annotations

import math

import jax
import pytest

from production.oracles import run_supported_spec

pytestmark = pytest.mark.slow

# Solenoidal ceiling for the vector-potential (B = B0 + curl A) family: div B is
# analytically zero and must stay at roundoff for the whole horizon (measured
# ~2.5e-16 max over this 300-step CPU run). The gate is a few orders above that
# floor so a regression into the primitive-`b` finite regime (~1e-4..1e-2) is
# caught while benign roundoff accumulation is not flagged. See
# tests/production/test_vector_potential_oracle.py for the full rationale.
SOLENOIDAL_CEIL = 1e-12


def _vp_spec():
    return {
        "problem_id": "pcf_mri_vp_qualification",
        "spec_hash": "vp-qualification-hash",
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
            "nu": 5e-3,
            "eta_mag": 5e-3,
            "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 2e-3, "final_time": 0.6},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {"artifact_id": "vp-qualification", "regeneration_command": "test"},
    }


def test_long_nonlinear_curl_run_holds_the_physics_contract():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=300, diagnostics_every=10)
    sc = out["scalars"]

    # Solenoidal by construction, for the whole horizon: div B must hold at the
    # roundoff ceiling at the final step AND must not grow past it at any recorded
    # step -- the primitive-`b` failure mode is div B drifting into the finite
    # regime over the run, so a final-only check would miss a mid-horizon blowup.
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert max(row["divergence_b_l2"] for row in out["time_series"]) < SOLENOIDAL_CEIL
    assert sc["divergence_u_l2"] < 1e-4

    # The shearing-box energy budget closes over the nonlinear horizon.
    assert sc["energy_budget_residual"] < 1e-2

    # Health block: stable and resolved.
    assert sc["cfl_total"] < 1.0
    assert sc["spectral_tail_max"] < 1e-2
    assert 0.0 < sc["mode_occupancy"] <= 1.0

    # The run produced a usable statistical record.
    assert len(out["time_series"]) >= 30
    assert math.isfinite(sc["correlation_time_total_stress"])
    for key in ("stationarity_relative_change", "growth_rate", "alpha_Sh"):
        assert math.isfinite(sc[key]), key
