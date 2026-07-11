import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.la.solvers import Biharmonic, Helmholtz, laplacian


def test_helmholtz_constructor_matches_manual_form():
    D = FunctionSpace(6, Legendre, bc=(0, 0), domain=Domain(-1.0, 1.0))
    F = FunctionSpace(6, Fourier, domain=Domain(0.0, 2.0 * sp.pi))
    T = TensorProduct(D, F)
    u = TrialFunction(T, name="u_h")
    v = TestFunction(T, name="v_h")
    coords = T.system.base_scalars()
    coeff = 0.125
    nu = 0.01

    actual = Helmholtz(v, u, coeff=coeff, diffusivity=nu, coords=coords)
    expected = inner(v * u, sparse=True) - coeff * inner(
        v * (nu * laplacian(u, coords)), sparse=True
    )

    assert actual.shape == expected.shape
    assert (actual.todense() == expected.todense()).all()


def test_biharmonic_constructor_matches_manual_form():
    B = FunctionSpace(8, Legendre, bc=(0, 0, 0, 0), domain=Domain(-1.0, 1.0))
    F = FunctionSpace(6, Fourier, domain=Domain(0.0, 2.0 * sp.pi))
    T = TensorProduct(B, F)
    u = TrialFunction(T, name="u_b")
    v = TestFunction(T, name="v_b")
    coords = T.system.base_scalars()
    coeff = 0.25
    nu = 0.02
    lap_u = laplacian(u, coords)

    actual = Biharmonic(v, u, coeff=coeff, diffusivity=nu, coords=coords)
    expected = inner(v * lap_u, sparse=True) - coeff * inner(
        v * (nu * laplacian(lap_u, coords)), sparse=True
    )

    assert actual.shape == expected.shape
    assert (actual.todense() == expected.todense()).all()
