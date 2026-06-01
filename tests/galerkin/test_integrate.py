import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import FunctionSpace, TensorProduct, VectorTensorProductSpace
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
