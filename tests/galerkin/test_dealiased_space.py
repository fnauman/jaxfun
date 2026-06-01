import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import FunctionSpace, TensorProduct, VectorTensorProductSpace
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.Legendre import Legendre


def test_fourier_dealiased_roundtrip_preserves_coefficients() -> None:
    V = FunctionSpace(16, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)))
    Vp = V.get_dealiased(1.5)
    coeffs = jnp.zeros(V.num_dofs, dtype=complex).at[1].set(1.0).at[3].set(0.25j)

    assert Vp.num_dofs == V.num_dofs
    assert Vp.num_quad_points == 24
    assert jnp.allclose(Vp.forward(Vp.backward(coeffs)), coeffs, atol=2e-6)


def test_tensor_product_dealiased_padding_tuple() -> None:
    X = FunctionSpace(12, Legendre, domain=Domain(-1.0, 1.0))
    Y = FunctionSpace(16, FourierSpace, domain=Domain(0.0, 2.0 * float(sp.pi)))
    T = TensorProduct(X, Y)
    Tp = T.get_dealiased((1.0, 1.5))

    assert Tp.num_dofs == T.num_dofs
    assert Tp.num_quad_points == (12, 24)


def test_vector_dealiased_preserves_component_shapes() -> None:
    X = FunctionSpace(12, Legendre, bc=(0, 0, 0, 0))
    D = FunctionSpace(12, Legendre, bc=(0, 0))
    Y = FunctionSpace(16, FourierSpace)
    TB = TensorProduct(X, Y)
    TD = TensorProduct(D, Y)
    V = VectorTensorProductSpace((TB, TD, TD))
    Vp = V.get_dealiased((1.0, 1.5))

    assert Vp.num_dofs == V.num_dofs
    assert tuple(space.num_quad_points for space in Vp) == (
        (12, 24),
        (12, 24),
        (12, 24),
    )
