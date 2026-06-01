import jax.numpy as jnp
import pytest

from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre


def test_tensorproduct_wavenumbers_scaled_to_physical_domain():
    D = FunctionSpace(6, Legendre)
    F = FunctionSpace(8, Fourier, domain=(0, 4 * jnp.pi))
    T = TensorProduct(D, F)

    K0, K1 = T.local_wavenumbers(scaled=True)

    assert K0.shape == (6, 1)
    assert K1.shape == (1, 8)
    assert jnp.all(K0 == 0)
    assert K1[0, 1] == pytest.approx(0.5)
    assert K1[0, 4] == pytest.approx(-2.0)


def test_tensorproduct_mask_nyquist_zeroes_fourier_highest_mode():
    D = FunctionSpace(6, Legendre)
    F = FunctionSpace(8, Fourier)
    T = TensorProduct(D, F)
    coeffs = jnp.ones(T.num_dofs)

    masked = T.mask_nyquist(coeffs)

    assert jnp.all(masked[:, 4] == 0)
    assert jnp.all(masked[:, :4] == 1)
    assert jnp.all(masked[:, 5:] == 1)
