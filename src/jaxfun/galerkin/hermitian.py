"""Experimental two-periodic-axis real/Hermitian transform prototype.

The last periodic axis is stored with ``rfft`` while the preceding periodic
axis retains normal complex FFT ordering.  This module is deliberately not
wired into :class:`TensorProductSpace`; it defines and tests the representation
contract needed before a public storage migration is considered.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def periodic_real_forward(values: Array) -> Array:
    """Transform real ``(..., y, z)`` samples to ``(..., ky, kz>=0)``."""

    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError("Hermitian forward transform requires real physical values")
    z_half = jnp.fft.rfft(values, axis=-1, norm="forward")
    return jnp.fft.fft(z_half, axis=-2, norm="forward")


def periodic_real_backward(coefficients: Array, *, shape: tuple[int, int]) -> Array:
    """Invert ``(..., ky, kz>=0)`` coefficients to real physical samples."""

    ny, nz = shape
    expected = (ny, nz // 2 + 1)
    if tuple(coefficients.shape[-2:]) != expected:
        raise ValueError(
            f"Hermitian coefficients must end in {expected}, got "
            f"{coefficients.shape[-2:]}"
        )
    z_half = jnp.fft.ifft(coefficients, axis=-2, norm="forward")
    return jnp.fft.irfft(z_half, n=nz, axis=-1, norm="forward")


def compress_periodic_coefficients(
    coefficients: Array, *, shape: tuple[int, int]
) -> Array:
    """Retain the nonnegative last-axis modes of a full Hermitian spectrum."""

    ny, nz = shape
    if tuple(coefficients.shape[-2:]) != shape:
        raise ValueError(f"full coefficients must end in {shape}")
    return coefficients[..., :ny, : nz // 2 + 1]


def expand_periodic_coefficients(
    coefficients: Array, *, shape: tuple[int, int]
) -> Array:
    """Reconstruct a full 2-D Hermitian spectrum from ``kz >= 0`` storage."""

    ny, nz = shape
    expected = (ny, nz // 2 + 1)
    if tuple(coefficients.shape[-2:]) != expected:
        raise ValueError(f"half coefficients must end in {expected}")
    full = jnp.zeros((*coefficients.shape[:-2], ny, nz), dtype=coefficients.dtype)
    full = full.at[..., : expected[0], : expected[1]].set(coefficients)
    negative_ky = (-jnp.arange(ny)) % ny
    positive_kz = jnp.arange(1, (nz + 1) // 2)
    return full.at[..., :, nz - positive_kz].set(
        jnp.conj(coefficients[..., negative_ky[:, None], positive_kz[None, :]])
    )


def mask_periodic_nyquist(coefficients: Array, *, shape: tuple[int, int]) -> Array:
    """Zero both even-grid Nyquist planes, matching JAXfun derivatives."""

    ny, nz = shape
    out = coefficients
    if ny % 2 == 0:
        out = out.at[..., ny // 2, :].set(0)
    if nz % 2 == 0:
        out = out.at[..., nz // 2].set(0)
    return out


def differentiate_periodic_real(
    coefficients: Array,
    order: tuple[int, int],
    *,
    shape: tuple[int, int],
    lengths: tuple[float, float],
) -> Array:
    """Differentiate stored Hermitian coefficients in ``y`` and ``z``."""

    ny, nz = shape
    dy, dz = order
    ky = jnp.fft.fftfreq(ny) * ny * (2.0 * jnp.pi / lengths[0])
    kz = jnp.fft.rfftfreq(nz) * nz * (2.0 * jnp.pi / lengths[1])
    if dy % 2 and ny % 2 == 0:
        ky = ky.at[ny // 2].set(0)
    if dz % 2 and nz % 2 == 0:
        kz = kz.at[nz // 2].set(0)
    multiplier = (1j * ky[:, None]) ** dy * (1j * kz[None, :]) ** dz
    return coefficients * multiplier


def pad_periodic_real(
    coefficients: Array,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> Array:
    """Zero-pad a masked Hermitian spectrum to a larger even periodic grid."""

    sy, sz = source_shape
    ty, tz = target_shape
    if ty < sy or tz < sz or any(n % 2 for n in (*source_shape, *target_shape)):
        raise ValueError("padding requires even target dimensions >= even sources")
    y_zeros = jnp.zeros(
        (*coefficients.shape[:-2], ty - sy, coefficients.shape[-1]),
        dtype=coefficients.dtype,
    )
    padded_y = jnp.concatenate(
        (coefficients[..., : sy // 2, :], y_zeros, coefficients[..., sy // 2 :, :]),
        axis=-2,
    )
    z_zeros = jnp.zeros(
        (*padded_y.shape[:-1], tz // 2 + 1 - padded_y.shape[-1]),
        dtype=coefficients.dtype,
    )
    return jnp.concatenate((padded_y, z_zeros), axis=-1)


def truncate_periodic_real(
    coefficients: Array,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> Array:
    """Truncate a padded Hermitian spectrum to a smaller even grid."""

    sy, sz = source_shape
    ty, tz = target_shape
    if ty > sy or tz > sz or any(n % 2 for n in (*source_shape, *target_shape)):
        raise ValueError("truncation requires even target dimensions <= even sources")
    return jnp.concatenate(
        (
            coefficients[..., : ty // 2, : tz // 2 + 1],
            coefficients[..., sy - ty // 2 :, : tz // 2 + 1],
        ),
        axis=-2,
    )
