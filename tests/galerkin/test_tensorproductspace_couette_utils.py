import jax.numpy as jnp
import pytest

from jaxfun.galerkin import FunctionSpace, K_over_K2, TensorProduct
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


def test_k_over_k2_helper_guards_zero_mode_and_selects_periodic_axes():
    D = FunctionSpace(6, Legendre)
    F1 = FunctionSpace(8, Fourier, domain=(0, 4 * jnp.pi))
    F2 = FunctionSpace(6, Fourier, domain=(0, 2 * jnp.pi))
    T = TensorProduct(D, F1, F2)

    K = T.local_wavenumbers(scaled=True)
    Ky_over_k2, Kz_over_k2 = K_over_K2(K, axes=(1, 2))
    k2 = K[1] * K[1] + K[2] * K[2]
    denom = jnp.where(k2 == 0, 1, k2)

    assert jnp.isfinite(Ky_over_k2).all()
    assert jnp.isfinite(Kz_over_k2).all()
    assert Ky_over_k2[0, 0, 0] == pytest.approx(0.0)
    assert Kz_over_k2[0, 0, 0] == pytest.approx(0.0)
    assert jnp.allclose(Ky_over_k2, K[1] / denom)
    assert jnp.allclose(Kz_over_k2, K[2] / denom)
