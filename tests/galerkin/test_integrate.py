import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    ChebyshevU,
    FunctionSpace,
    Jacobi,
    TensorProduct,
    Ultraspherical,
    VectorTensorProductSpace,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.inner import integrate
from jaxfun.galerkin.Legendre import Legendre


def test_integrate_fourier_domain_volume() -> None:
    V = FunctionSpace(16, FourierSpace, domain=Domain(0.0, 4.0 * float(sp.pi)))
    ones = jnp.ones(V.num_quad_points)

    assert jnp.allclose(integrate(ones, V), 4.0 * float(sp.pi))


def test_integrate_tensor_product_separable_function() -> None:
    X = FunctionSpace(20, Legendre, domain=Domain(-1.0, 1.0))
    Y = FunctionSpace(16, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)))
    T = TensorProduct(X, Y)
    x, y = T.mesh()
    values = x**2 + jnp.cos(y)

    expected = (2.0 / 3.0) * (2.0 * float(sp.pi))
    assert jnp.allclose(integrate(values, T), expected, rtol=2e-6, atol=2e-6)


def test_integrate_vector_space_sums_components() -> None:
    X = FunctionSpace(18, Legendre, domain=Domain(-1.0, 1.0))
    Y = FunctionSpace(12, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)))
    T = TensorProduct(X, Y)
    V = VectorTensorProductSpace((T, T))
    x, y = T.mesh()
    u = jnp.stack((x**2 + 0.0 * y, 2.0 + 0.0 * x + 0.0 * y))

    expected = ((2.0 / 3.0) + 4.0) * (2.0 * float(sp.pi))
    assert jnp.allclose(integrate(u, V), expected, rtol=2e-6, atol=2e-6)


def test_integrate_chebyshev_uses_physical_not_orthogonality_measure() -> None:
    V = FunctionSpace(17, Chebyshev, domain=Domain(-1.0, 1.0))
    x = V.mesh()

    assert jnp.allclose(integrate(jnp.ones_like(x), V), 2.0, atol=1e-14)
    D = FunctionSpace(17, Chebyshev, bc=(0, 0), domain=Domain(-1.0, 1.0))
    assert jnp.allclose(jnp.sum(D.integration_weights()), 2.0, atol=1e-14)
    assert jnp.allclose(jnp.sum(D.quadrature_weights()), jnp.pi, atol=1e-14)
    assert jnp.allclose(integrate(jnp.ones(D.num_quad_points), D), 2.0, atol=1e-14)

    assert jnp.allclose(integrate(x**2, V), 2.0 / 3.0, atol=1e-14)
    assert jnp.allclose(jnp.sum(V.quadrature_weights()), jnp.pi, atol=1e-14)


def test_integrate_chebyshev_derivative_obeys_fundamental_theorem() -> None:
    V = FunctionSpace(18, Chebyshev, domain=Domain(-2.0, 3.0))
    x = V.mesh()
    values = 1.0 + 0.3 * x + 0.2 * x**3
    coefficients = V.forward(values)
    derivative = V.backward_primitive(coefficients, 1)

    bounds = jnp.asarray([-2.0, 3.0])
    traces = V.evaluate(bounds, coefficients)
    assert jnp.allclose(integrate(derivative, V), traces[1] - traces[0], atol=2e-12)


def test_integrate_chebyshev_tensor_product_is_physical_volume() -> None:
    X = FunctionSpace(19, Chebyshev, bc=(0, 0), domain=Domain(-1.0, 1.0))
    Y = FunctionSpace(16, FourierSpace, domain=Domain(0.0, 4.0))
    T = TensorProduct(X, Y)
    x, y = T.mesh()

    assert jnp.allclose(integrate(1.0 + 0.0 * x + 0.0 * y, T), 8.0, atol=2e-13)


@pytest.mark.parametrize(
    ("space_type", "kwargs"),
    [
        (ChebyshevU.ChebyshevU, {}),
        (Ultraspherical.Ultraspherical, {"lambda_": 2.0}),
        (Jacobi.Jacobi, {"alpha": 0.5, "beta": 1.25}),
    ],
)
def test_integrate_other_weighted_polynomials_uses_physical_measure(
    space_type, kwargs
) -> None:
    V = FunctionSpace(17, space_type, domain=Domain(-2.0, 3.0), **kwargs)
    x = V.mesh()

    assert jnp.allclose(integrate(jnp.ones_like(x), V), 5.0, atol=2e-13)
    assert jnp.allclose(integrate(x, V), 2.5, atol=2e-13)
    assert jnp.allclose(integrate(x**2, V), 35.0 / 3.0, atol=2e-12)
    assert not jnp.allclose(
        jnp.sum(V.quadrature_weights()), 5.0, rtol=1e-10, atol=1e-10
    )


def test_integrate_single_point_weighted_jacobi_is_midpoint_rule() -> None:
    V = FunctionSpace(
        1,
        Jacobi.Jacobi,
        domain=Domain(0.0, 5.0),
        alpha=0.5,
        beta=1.25,
    )
    x = V.mesh()

    assert jnp.allclose(integrate(jnp.ones_like(x), V), 5.0, atol=1e-14)
    assert jnp.allclose(integrate(x, V), 5.0 * x[0], atol=1e-14)
