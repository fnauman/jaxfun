import jax.numpy as jnp

from jaxfun.galerkin import FunctionSpace, InnerKind, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.operators import Dx


def test_dx_matches_sympy_diff_assembly() -> None:
    V = FunctionSpace(10, Legendre)
    u = TrialFunction(V)
    v = TestFunction(V)
    (x,) = V.system.base_scalars()

    dx_matrix = inner(v * Dx(u, 0, 2), kind=InnerKind.BILINEAR).todense()
    diff_matrix = inner(v * u.diff(x, 2), kind=InnerKind.BILINEAR).todense()

    assert jnp.allclose(dx_matrix, diff_matrix)
