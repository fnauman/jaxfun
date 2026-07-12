import jax.numpy as jnp

from jaxfun.galerkin.Chebyshev import Chebyshev


def test_chebyshev_derivative_coeffs_accept_complex_coefficients() -> None:
    V = Chebyshev(8)
    coeffs = jnp.zeros(8, dtype=complex).at[2].set(1.0 + 0.5j)

    deriv = V.derivative_coeffs(coeffs, 1)

    assert deriv.dtype == coeffs.dtype
    assert bool(jnp.isfinite(deriv).all())
