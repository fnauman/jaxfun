import sys
from pathlib import Path

import numpy as np
import pytest

from examples.pcf_linear_jax import PlaneCouetteLinear
from examples.taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax


def _load_reference_pcf_linear():
    couette_dir = Path(__file__).resolve().parents[2] / "couette"
    sys.path.insert(0, str(couette_dir))
    try:
        from _pcf_linear import PlaneCouetteLinear as ReferencePlaneCouetteLinear
    finally:
        sys.path.pop(0)
    return ReferencePlaneCouetteLinear


def test_pcf_linear_nonmodal_matches_reference_helper():
    ReferencePlaneCouetteLinear = _load_reference_pcf_linear()
    solver = PlaneCouetteLinear.couette(nx=10, Re=400.0, mhd=True, by=0.1, bz=0.2)
    reference = ReferencePlaneCouetteLinear.couette(
        nx=10, Re=400.0, mhd=True, by=0.1, bz=0.2
    )

    w, _ = solver.eigs(ky=1.0, kz=0.5, n_return=5)
    rw, _ = reference.eigs(ky=1.0, kz=0.5, n_return=5)
    assert np.allclose(w, rw, rtol=1.0e-12, atol=1.0e-12)

    rows = solver.nonmodal_growth(1.0, 0.5, [0.0, 0.25], n_modes=8, energy="total")
    ref_rows = reference.nonmodal_growth(
        1.0, 0.5, [0.0, 0.25], n_modes=8, energy="total"
    )
    assert rows == pytest.approx(ref_rows, rel=1.0e-12, abs=1.0e-12)


def test_tc_linear_nonmodal_returns_finite_kinetic_gain():
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.002, N=12, family="L")
    rows = solver.nonmodal_growth(m=0, kz=3.0, times=[0.0, 0.5], n_modes=12)

    assert rows[0]["gain"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    assert rows[1]["gain"] > 0.0
    assert np.isfinite(rows[1]["amplification"])


def test_tc_mri_energy_norms_and_nonmodal_are_finite():
    eta = 0.5
    base = CircularCouette(1.0, 2.0, 1.0, eta**1.5)
    solver = TaylorCouetteMRIJax(
        base, B0=0.1, nu=0.001, eta_mag=0.001, N=10, family="L"
    )

    for kind in ("kinetic", "magnetic", "total"):
        Q = solver.energy_matrix(m=0, kz=3.0, kind=kind)
        assert np.allclose(Q, Q.conj().T, rtol=0.0, atol=1.0e-12)
        rows = solver.nonmodal_growth(
            m=0, kz=3.0, times=[0.0, 0.25], n_modes=10, energy=kind
        )
        assert rows[0]["gain"] == pytest.approx(1.0, rel=1.0e-9, abs=1.0e-9)
        assert rows[1]["gain"] > 0.0
