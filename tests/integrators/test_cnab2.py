import jax.numpy as jnp
import pytest

from jaxfun.integrators.cnab2 import ab2_extrapolate, cnab2_rhs, scan_steps


def test_ab2_extrapolate_bootstraps_with_current_tree():
    current = (jnp.array([2.0, 4.0]), jnp.array([1.0]))
    previous = (jnp.array([10.0, 20.0]), jnp.array([5.0]))

    out = ab2_extrapolate(current, previous, have_previous=False)

    assert jnp.allclose(out[0], current[0])
    assert jnp.allclose(out[1], current[1])


def test_cnab2_rhs_uses_ab2_after_first_step():
    explicit = (jnp.array([10.0, 20.0]),)
    current = (jnp.array([2.0, 4.0]),)
    previous = (jnp.array([1.0, 3.0]),)

    out = cnab2_rhs(explicit, current, previous, have_previous=True)

    expected_nonlinear = 1.5 * current[0] - 0.5 * previous[0]
    assert jnp.allclose(out[0], explicit[0] - expected_nonlinear)


def test_scan_steps_matches_repeated_step_and_rejects_negative_steps():
    def step(x):
        return 2.0 * x + 1.0

    scanned = scan_steps(step, jnp.array(0.0), 4)
    manual = jnp.array(0.0)
    for _ in range(4):
        manual = step(manual)

    assert scanned == pytest.approx(float(manual))
    with pytest.raises(ValueError):
        scan_steps(step, jnp.array(0.0), -1)
