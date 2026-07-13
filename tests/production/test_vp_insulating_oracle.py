"""Insulating (vacuum-matched) plane-Couette vector-potential contract.

True insulating walls match the fluctuation field to a decaying exterior
potential per Fourier mode -- distinct from both the conducting (b_x = 0) and
pseudo-vacuum (b_tang = 0) conventions.  The linear anchor is
``examples.pcf_linear_jax.PlaneCouetteLinear(magnetic_bc="insulating")``,
itself validated against the analytic resistive slab decay rate
``s = -eta (mu^2 + k^2)`` with ``mu tan(mu) = k``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from production.oracles import run_supported_spec

# Scale-aware roundoff ceiling for the PCF slab family: div B and the
# vacuum-matching residual carry no 1/r projection error there, so the
# measured floors (5e-21 .. 1.7e-16 across the tests below, on O(0.1)-scale
# fields) are pure roundoff and the gate is a few hundred machine epsilons.
# The cylindrical TC family keeps its separate, resolution-dependent
# projected-witness tolerance (see test_tc_vector_potential_oracle.py).
SOLENOIDAL_CEIL = 1.0e3 * np.finfo(float).eps  # ~2.2e-13
DIVERGENCE_GUARD = 1.0e-12


def _pcf_insulating_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vp_insulating_smoke",
        "spec_hash": "vp-insulating-smoke-hash",
        "numerics_contract_version": 2,
        "precision": "float64",
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {
            "type": "gpu_generated_saturated_dns",
            "divergence_b_guard_l2": DIVERGENCE_GUARD,
        },
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "insulating"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 2.0e-2,
            "eta_mag": 2.0e-2,
            "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1.0e-3, "final_time": 0.01},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {
            "artifact_id": "pcf_mri_vp_insulating_smoke",
            "regeneration_command": "test-only spec; no committed golden",
        },
    }
    spec["nondimensional_groups"].update(groups)
    spec["forcing"]["B0"] = spec["nondimensional_groups"]["B0"]
    return spec


def _max_series(out, key):
    return max(row[key] for row in out["time_series"] if key in row)


def test_insulating_oracle_holds_divergence_and_matching_rows():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_pcf_insulating_spec(), steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    assert sc["magnetic_bc"] == "insulating"
    assert sc["divergence_b_guard_l2"] == DIVERGENCE_GUARD
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert _max_series(out, "divergence_b_l2") < SOLENOIDAL_CEIL
    assert sc["insulating_bc_residual"] < SOLENOIDAL_CEIL
    assert _max_series(out, "insulating_bc_residual") < SOLENOIDAL_CEIL


@pytest.mark.integration
def test_insulating_finite_amplitude_horizon_stays_solenoidal():
    """50 nonlinear steps at finite amplitude: div B and the vacuum-matching
    residual must hold their roundoff floors (measured ~1e-16)."""
    jax.config.update("jax_enable_x64", True)
    from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyInsulatingJax

    solver = PlaneCouetteMRIShearpyInsulatingJax(
        N=(17, 8, 16),
        domain=((-1.0, 1.0), (0.0, 4.0), (0.0, 1.0)),
        Re=50.0,
        Rm=50.0,
        omega=2.0 / 3.0,
        shear_rate=1.0,
        background_b=(0.0, 0.0, 0.05),
        dt=1e-3,
        family="L",
        perturbation_amplitude=0.05,
        magnetic_amplitude=1e-3,
    )
    state = solver.initial_state()
    divbs, residuals = [], []
    for _ in range(5):
        state = solver.solve(state, 10)
        diag = solver.diagnostics(state)
        divbs.append(float(diag["divB_L2"]))
        residuals.append(float(diag["insulating_bc_residual"]))
    assert max(divbs) < SOLENOIDAL_CEIL
    assert max(residuals) < SOLENOIDAL_CEIL


def test_linear_insulating_operator_matches_analytic_slab_decay():
    """The linear anchor itself: with U = 0 and B0 = 0 the leading insulating
    magnetic decay rate is s = -eta (mu0^2 + k^2), mu0 tan(mu0) = k (the even
    poloidal family of the resistive slab with vacuum matching)."""
    from scipy.optimize import brentq

    from examples.pcf_linear_jax import PlaneCouetteLinear

    ky, kz = 0.7, 0.9
    k = float(np.hypot(ky, kz))
    eta, nu = 5e-4, 2e-3
    mu0 = brentq(lambda m: m * np.tan(m) - k, 1e-8, np.pi / 2 - 1e-8)
    s_exact = -eta * (mu0**2 + k**2)
    op = PlaneCouetteLinear(
        nx=96,
        nu=nu,
        eta=eta,
        Uprime=0.0,
        omega=0.0,
        by=0.0,
        bz=0.0,
        mhd=True,
        magnetic_bc="insulating",
    )
    w, _ = op.eigs(ky, kz, n_return=1)
    assert w[0].real == pytest.approx(s_exact, rel=1e-8)


def test_pseudo_vacuum_linear_operator_matches_analytic_leading_decay():
    """Regression for the singular-pencil fix: the pseudo-vacuum operator's
    leading magnetic decay is -eta k^2 (constant-tangential-field mode)."""
    from examples.pcf_linear_jax import PlaneCouetteLinear

    ky, kz = 0.7, 0.9
    k2 = ky * ky + kz * kz
    eta, nu = 5e-4, 2e-3
    op = PlaneCouetteLinear(
        nx=96,
        nu=nu,
        eta=eta,
        Uprime=0.0,
        omega=0.0,
        by=0.0,
        bz=0.0,
        mhd=True,
        magnetic_bc="pseudo_vacuum",
    )
    w, _ = op.eigs(ky, kz, n_return=1)
    assert w[0].real == pytest.approx(-eta * k2, rel=1e-8)


@pytest.mark.slow
def test_insulating_growth_matches_linear_eigenvalue():
    """Physics anchor: the eigenmode-seeded nonlinear DNS at tiny amplitude
    must reproduce the insulating linear MRI growth rate."""
    jax.config.update("jax_enable_x64", True)
    from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyInsulatingJax

    solver = PlaneCouetteMRIShearpyInsulatingJax(
        N=(33, 4, 16),
        domain=((-1.0, 1.0), (0.0, 4.0), (0.0, 1.0)),
        Re=200.0,
        Rm=200.0,
        omega=2.0 / 3.0,
        shear_rate=1.0,
        background_b=(0.0, 0.0, 0.1),
        dt=1e-3,
        family="L",
        perturbation_amplitude=0.0,
        magnetic_amplitude=0.0,
    )
    state, ev = solver.seed_linear_eigenmode(ky_mode=0, kz_mode=1, amp=1e-6)
    assert ev.real > 0.1  # MRI-unstable anchor point
    state = solver.solve(state, 100)
    d = solver.diagnostics(state)
    e0 = float(d["Epert"]) + float(d["Emag"])
    divbs, residuals = [], []
    for _ in range(3):
        state = solver.solve(state, 100)
        d = solver.diagnostics(state)
        divbs.append(float(d["divB_L2"]))
        residuals.append(float(d["insulating_bc_residual"]))
    e1 = float(d["Epert"]) + float(d["Emag"])
    gamma = 0.5 * np.log(e1 / e0) / (300 * solver.dt)
    assert gamma == pytest.approx(ev.real, rel=1e-5)
    assert max(divbs) < SOLENOIDAL_CEIL
    assert max(residuals) < SOLENOIDAL_CEIL
