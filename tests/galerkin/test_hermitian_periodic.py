import jax
import jax.numpy as jnp

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from jaxfun.galerkin.hermitian import (
    compress_periodic_coefficients,
    differentiate_periodic_real,
    expand_periodic_coefficients,
    mask_periodic_nyquist,
    pad_periodic_real,
    periodic_real_backward,
    periodic_real_forward,
    truncate_periodic_real,
)


def _full_forward(values):
    return jnp.fft.fft(
        jnp.fft.fft(values, axis=-1, norm="forward"), axis=-2, norm="forward"
    )


def test_periodic_real_storage_roundtrip_and_full_complex_parity() -> None:
    values = jax.random.normal(jax.random.PRNGKey(4), (5, 8, 10))
    half = periodic_real_forward(values)
    full = _full_forward(values)

    assert half.shape == (5, 8, 6)
    assert half.size / full.size == 0.6
    assert jnp.allclose(half, full[..., :6], rtol=2.0e-13, atol=2.0e-13)
    assert jnp.allclose(
        periodic_real_backward(half, shape=(8, 10)),
        values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    reconstructed = expand_periodic_coefficients(half, shape=(8, 10))
    assert jnp.allclose(reconstructed, full, rtol=2.0e-13, atol=2.0e-13)
    assert jnp.array_equal(
        compress_periodic_coefficients(reconstructed, shape=(8, 10)), half
    )


def test_periodic_real_storage_roundtrip_for_odd_spanwise_size() -> None:
    shape = (7, 5)
    values = jax.random.normal(jax.random.PRNGKey(5), (3, *shape))
    half = periodic_real_forward(values)
    full = _full_forward(values)

    reconstructed = expand_periodic_coefficients(half, shape=shape)

    assert jnp.allclose(reconstructed, full, rtol=2.0e-13, atol=2.0e-13)
    assert jnp.allclose(
        periodic_real_backward(half, shape=shape),
        values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_periodic_real_derivatives_match_full_complex_reference() -> None:
    shape = (8, 10)
    lengths = (4.0 * jnp.pi, 2.0 * jnp.pi)
    values = jax.random.normal(jax.random.PRNGKey(7), (3, *shape))
    half = mask_periodic_nyquist(periodic_real_forward(values), shape=shape)
    full = _full_forward(values)
    full = full.at[..., shape[0] // 2, :].set(0)
    full = full.at[..., shape[1] // 2].set(0)
    ky = jnp.fft.fftfreq(shape[0]) * shape[0] * (2 * jnp.pi / lengths[0])
    kz = jnp.fft.fftfreq(shape[1]) * shape[1] * (2 * jnp.pi / lengths[1])
    ky = ky.at[shape[0] // 2].set(0)
    expected_coeff = full * (1j * ky[:, None]) * (1j * kz[None, :]) ** 2
    expected = jnp.fft.ifft(
        jnp.fft.ifft(expected_coeff, axis=-2, norm="forward"),
        axis=-1,
        norm="forward",
    ).real
    actual = periodic_real_backward(
        differentiate_periodic_real(half, (1, 2), shape=shape, lengths=lengths),
        shape=shape,
    )
    assert jnp.allclose(actual, expected, rtol=3.0e-12, atol=3.0e-12)


def test_periodic_real_padding_and_dealiased_product_match_full_complex() -> None:
    source = (8, 8)
    padded = (12, 12)
    a = jax.random.normal(jax.random.PRNGKey(10), (2, *source))
    b = jax.random.normal(jax.random.PRNGKey(11), (2, *source))
    ah = mask_periodic_nyquist(periodic_real_forward(a), shape=source)
    bh = mask_periodic_nyquist(periodic_real_forward(b), shape=source)
    product = periodic_real_backward(
        pad_periodic_real(ah, source_shape=source, target_shape=padded),
        shape=padded,
    ) * periodic_real_backward(
        pad_periodic_real(bh, source_shape=source, target_shape=padded),
        shape=padded,
    )
    result = truncate_periodic_real(
        periodic_real_forward(product), source_shape=padded, target_shape=source
    )

    # Expanding the same half spectrum establishes a full-complex oracle.
    full_product = _full_forward(product)
    expected = jnp.concatenate(
        (
            full_product[..., : source[0] // 2, : source[1] // 2 + 1],
            full_product[..., padded[0] - source[0] // 2 :, : source[1] // 2 + 1],
        ),
        axis=-2,
    )
    assert jnp.allclose(result, expected, rtol=3.0e-12, atol=3.0e-12)


def _hermitian_roundtrip_tree(state, shape):
    def roundtrip(value):
        if getattr(value, "ndim", 0) == 3 and tuple(value.shape[-2:]) == shape:
            return expand_periodic_coefficients(
                compress_periodic_coefficients(value, shape=shape), shape=shape
            )
        return value

    return jax.tree.map(roundtrip, state)


def _assert_primary_close(left, right, *, mhd):
    lhs = (*left.flow.u, left.flow.g, *left.A) if mhd else (*left.u, left.g)
    rhs = (*right.flow.u, right.flow.g, *right.A) if mhd else (*right.u, right.g)
    assert all(
        bool(jnp.allclose(a, b, rtol=3.0e-12, atol=3.0e-13))
        for a, b in zip(lhs, rhs, strict=True)
    )


def test_pcf_hydro_and_mhd_trajectories_survive_hermitian_storage_roundtrips() -> None:
    shape = (8, 8)
    solvers = (
        PlaneCouetteFluctuationJax(
            N=(9, *shape),
            dt=1.0e-3,
            time_integrator="CNAB2",
            padding_factor=(1.0, 1.0, 1.0),
            perturbation_amplitude=0.01,
        ),
        PlaneCouetteMHDJax(
            N=(9, *shape),
            dt=1.0e-3,
            time_integrator="CNAB2",
            padding_factor=(1.0, 1.0, 1.0),
            perturbation_amplitude=0.01,
            magnetic_amplitude=0.005,
        ),
    )
    for solver in solvers:
        initial = solver.initial_state()
        expected = solver.solve(initial, 3)
        actual = initial
        for _ in range(3):
            actual = solver.step(_hermitian_roundtrip_tree(actual, shape))
        actual = _hermitian_roundtrip_tree(actual, shape)
        _assert_primary_close(
            actual, expected, mhd=isinstance(solver, PlaneCouetteMHDJax)
        )
