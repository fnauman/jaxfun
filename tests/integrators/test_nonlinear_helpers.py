import jax.numpy as jnp

from jaxfun.integrators.nonlinear import physical_cross


def test_physical_cross_matches_pointwise_numpy_formula() -> None:
    a = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0]))
    b = (jnp.array([0.5, -1.0]), jnp.array([2.0, 1.5]), jnp.array([-3.0, 0.25]))

    c = physical_cross(a, b)

    assert jnp.allclose(c[0], a[1] * b[2] - a[2] * b[1])
    assert jnp.allclose(c[1], a[2] * b[0] - a[0] * b[2])
    assert jnp.allclose(c[2], a[0] * b[1] - a[1] * b[0])
