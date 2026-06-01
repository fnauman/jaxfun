import numpy as np
import pytest

from examples.taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
from tests._parity import (
    tc_linear_eigenvalues,
    tc_linear_nonmodal,
    tc_mri_eigenvalues,
    tc_mri_nonmodal,
)

pytestmark = pytest.mark.integration


def _keplerian_base():
    eta = 0.5
    return CircularCouette(1.0, 2.0, 1.0, eta**1.5)


def test_tc_linear_matches_live_shenfun_eigenvalues_and_nonmodal():
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.002, N=12, family="L")

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)
    assert np.allclose(w, tc_linear_eigenvalues(), rtol=1.0e-11, atol=1.0e-11)

    rows = solver.nonmodal_growth(m=0, kz=3.0, times=[0.0, 0.5], n_modes=12)
    ref_rows = tc_linear_nonmodal()
    for row, ref in zip(rows, ref_rows, strict=True):
        assert row["t"] == pytest.approx(ref["t"], abs=0.0)
        assert row["gain"] == pytest.approx(ref["gain"], rel=1.0e-10, abs=1.0e-10)


@pytest.mark.parametrize("magnetic_bc", ["conducting", "insulating"])
def test_tc_mri_matches_live_shenfun_eigenvalues_and_nonmodal(magnetic_bc):
    solver = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=12,
        family="L",
        magnetic_bc=magnetic_bc,
    )

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)
    assert np.allclose(
        w,
        tc_mri_eigenvalues(magnetic_bc=magnetic_bc),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    solver_small = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=10,
        family="L",
        magnetic_bc=magnetic_bc,
    )
    rows = solver_small.nonmodal_growth(
        m=0, kz=3.0, times=[0.0, 0.25], n_modes=10, energy="total"
    )
    ref_rows = tc_mri_nonmodal(magnetic_bc=magnetic_bc, n=10, energy="total")
    for row, ref in zip(rows, ref_rows, strict=True):
        assert row["t"] == pytest.approx(ref["t"], abs=0.0)
        assert row["gain"] == pytest.approx(ref["gain"], rel=1.0e-8, abs=1.0e-8)
