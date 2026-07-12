import jax.numpy as jnp

from jaxfun.galerkin import FunctionSpace, TensorProduct, VectorTensorProductSpace
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre


def test_vectortensorproductspace_returns_tuple_for_ragged_forward_coefficients():
    wall_normal = FunctionSpace(8, Legendre, bc=(0, 0, 0, 0))
    tangential = FunctionSpace(8, Legendre, bc=(0, 0))
    periodic = FunctionSpace(6, Fourier)
    TB = TensorProduct(wall_normal, periodic)
    TD = TensorProduct(tangential, periodic)
    V = VectorTensorProductSpace((TB, TD, TD))

    physical = jnp.ones((3,) + TB.num_quad_points)
    coeffs = V.forward(physical)

    assert isinstance(coeffs, tuple)
    assert tuple(c.shape for c in coeffs) == (TB.num_dofs, TD.num_dofs, TD.num_dofs)
    assert V.num_dofs == (TB.num_dofs, TD.num_dofs, TD.num_dofs)


def test_vectortensorproductspace_still_stacks_equal_shapes():
    tangential = FunctionSpace(8, Legendre, bc=(0, 0))
    periodic = FunctionSpace(6, Fourier)
    TD = TensorProduct(tangential, periodic)
    V = VectorTensorProductSpace(TD)

    physical = jnp.ones((2,) + TD.num_quad_points)
    coeffs = V.forward(physical)

    assert not isinstance(coeffs, tuple)
    assert coeffs.shape == (2,) + TD.num_dofs
