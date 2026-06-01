import jax.numpy as jnp
import numpy as np
import pytest

from examples.taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
from jaxfun import Domain
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from tests._parity import (
    tc_linear_eigenvalues,
    tc_linear_nonmodal,
    tc_mri_eigenvalues,
    tc_mri_nonmodal,
    tc_radial_dealias_product,
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


def test_radial_polynomial_dealiasing_matches_live_shenfun_product():
    n = 8
    F = FunctionSpace(n, Fourier, domain=Domain(0.0, 2.0 * np.pi))
    S = FunctionSpace(n, Legendre, domain=Domain(1.0, 2.0))
    T = TensorProduct(F, S)
    Tp = T.get_dealiased((1.5, 1.5))
    u = jnp.zeros(T.num_dofs, dtype=complex)
    v = jnp.zeros(T.num_dofs, dtype=complex)
    u = u.at[0, 1].set(0.5).at[1, 2].set(0.75 + 0.25j)
    v = v.at[0, 3].set(-0.4).at[1, 1].set(-0.2 + 0.1j)

    h = Tp.forward(Tp.backward(u) * Tp.backward(v))

    assert np.allclose(
        h, tc_radial_dealias_product(n=n), rtol=1.0e-12, atol=1.0e-12
    )
