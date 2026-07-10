"""FJ-03: curl / vector-potential PCF-MRI production oracle."""

from __future__ import annotations

import jax
import pytest

from production.oracles import run_supported_spec


def _vp_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vector_potential_smoke",
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0, "Omega": 2.0 / 3.0, "nu": 2e-2, "eta_mag": 2e-2,
            "Re": 50.0, "Rm": 50.0, "Pm": 1.0, "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1e-3, "final_time": 0.01},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
    }
    spec["nondimensional_groups"].update(groups)
    spec["forcing"]["B0"] = spec["nondimensional_groups"]["B0"]  # keep sources consistent
    return spec


def test_vector_potential_oracle_is_solenoidal_by_construction():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    # B = curl A -> div B = 0 to roundoff (the invariant the primitive path lacks)
    assert sc["divergence_b_l2"] < 1e-8
    for key in ("kinetic_energy", "magnetic_energy", "total_stress", "alpha_Sh", "growth_rate"):
        assert key in sc
    assert len(out["time_series"]) >= 2


def test_vector_potential_oracle_is_znf_safe():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(B0=0.0), steps=3)
    sc = out["scalars"]
    # ZNF: no net-flux alpha, but the shear-normalized alpha is present and finite
    assert "transport_alpha" not in sc
    assert "alpha_Sh" in sc
    import math

    assert math.isfinite(sc["alpha_Sh"])


def test_vector_potential_emits_flux_diagnostics():
    """FJ-04: the ZNF curl workhorse must expose mean flux + mean/fluct split."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=3)
    sc = out["scalars"]
    for key in (
        "mean_bx", "mean_by", "mean_bz",
        "mag_energy_mean", "mag_energy_fluct",
        "flux_drift_bx", "flux_drift_by", "flux_drift_bz",
    ):
        assert key in sc


def test_vector_potential_rejects_resume():
    jax.config.update("jax_enable_x64", True)
    from production.oracles import ProductionOracleNotImplementedError

    class _FakeCkpt:
        attrs = {}
        tstep = 0

    with pytest.raises(ProductionOracleNotImplementedError):
        run_supported_spec(_vp_spec(), steps=3, resume_checkpoint=_FakeCkpt())


def test_vector_potential_rejects_checkpoint_flags():
    """F12: checkpoint/snapshot must not be silently ignored on the curl path."""
    jax.config.update("jax_enable_x64", True)
    from production.oracles import ProductionOracleNotImplementedError

    with pytest.raises(ProductionOracleNotImplementedError, match="checkpoint"):
        run_supported_spec(_vp_spec(), steps=3, checkpoint_every=1)
