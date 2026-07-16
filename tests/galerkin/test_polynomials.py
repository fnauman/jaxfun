import jax.numpy as jnp
import pytest

from jaxfun.galerkin import Chebyshev, Jacobi, Legendre
from jaxfun.utils.common import Domain, ulp


def test_chebyshev_evaluate_variants():
    C = Chebyshev.Chebyshev(8)
    x = jnp.linspace(-1, 1, 17)
    c = jnp.arange(8.0)
    u1 = jnp.array([C.evaluate(xi, c) for xi in x])
    u2 = jnp.array([C._evaluate2(xi, c) for xi in x])
    assert jnp.linalg.norm(u1 - u2) < ulp(100)
    u3 = C.evaluate(x, c)
    assert jnp.linalg.norm(u3 - u2) < ulp(100)


def test_legendre_evaluate_variants_and_domain_mapping():
    L = Legendre.Legendre(6, domain=Domain(-2, 2))
    x = jnp.linspace(-2, 2, 9)
    c = jnp.arange(6.0)
    u1 = jnp.array([L.evaluate(xi, c) for xi in x])
    u2 = jnp.array([L._evaluate2(L.map_reference_domain(xi), c) for xi in x])
    assert jnp.linalg.norm(u1 - u2) < ulp(100)
    u3 = L.evaluate(x, c)
    assert jnp.linalg.norm(u3 - u2) < ulp(100)
    # Mapping check: reference domain is (-1,1)
    Xref = jnp.array([L.map_reference_domain(xi) for xi in x])
    assert Xref.min() >= -1 - ulp(1) and Xref.max() <= 1 + ulp(1)


def test_jacobi_general_parameters():
    J = Jacobi.Jacobi(5, alpha=1, beta=2)
    x = jnp.linspace(-1, 1, 13)
    c = jnp.arange(5.0)
    u = jnp.array([J.evaluate(xi, c) for xi in x])
    assert jnp.isfinite(u).all()
    # Mass matrix diagonal
    M = J.mass_matrix().todense()
    expected = J.norm_squared() / J.domain_factor
    assert jnp.allclose(M.diagonal(), expected, atol=ulp(1.0))


@pytest.mark.parametrize("alpha,beta", ((0.0, 0.0), (0.5, 1.25), (2.0, 1.0)))
def test_jacobi_evaluate_matches_explicit_complex_basis_sum(alpha, beta):
    J = Jacobi.Jacobi(12, alpha=alpha, beta=beta)
    x = jnp.linspace(-0.95, 0.95, 17)
    c = jnp.linspace(-0.5, 1.0, 9) + 1j * jnp.linspace(0.75, -0.25, 9)

    actual = J.evaluate(x, c)
    expected = J.eval_basis_functions(x)[..., : c.size] @ c

    assert jnp.allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)
