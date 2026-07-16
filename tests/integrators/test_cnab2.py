import jax
import jax.numpy as jnp
import pytest

from jaxfun.integrators.cnab2 import (
    ScanRolloutCache,
    ab2_extrapolate,
    cnab2_rhs,
    scan_steps,
)


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


def test_scan_steps_single_device_matches_eager_tree_schedule():
    if jax.device_count() != 1:
        pytest.skip("single-device lax.scan path is not active")

    def step(state):
        x = state["x"]
        y = state["nested"][0]
        return {
            "x": 0.75 * x + jnp.sin(y),
            "nested": (0.5 * y - 0.1 * x, state["nested"][1] + x * y),
        }

    state0 = {
        "x": jnp.array([0.1, -0.2, 0.3]),
        "nested": (jnp.array([0.4, 0.2, -0.1]), jnp.array([1.0, -1.0, 0.5])),
    }
    scanned = jax.jit(lambda state: scan_steps(step, state, 7))(state0)
    eager = state0
    for _ in range(7):
        eager = step(eager)

    matches = jax.tree.leaves(
        jax.tree.map(lambda a, b: jnp.allclose(a, b), scanned, eager)
    )
    assert all(bool(match) for match in matches)


def test_scan_rollout_cache_reuses_and_bounds_static_lengths():
    cache = ScanRolloutCache(lambda value: value + 1, max_entries=2)
    state = jnp.asarray(0)

    for _ in range(200):
        state = cache(state, 1)
    assert int(state) == 200
    info = cache.info()
    assert info.live_entries == 1
    assert info.step_counts == (1,)
    assert info.hits == 199
    assert info.misses == 1
    assert info.evictions == 0

    state = cache(state, 2)
    state = cache(state, 3)
    assert int(state) == 205
    info = cache.info()
    assert info.live_entries == 2
    assert info.step_counts == (2, 3)
    assert info.evictions == 1


def test_scan_rollout_cache_rebind_drops_obsolete_variants():
    cache = ScanRolloutCache(lambda value: value + 1)
    state = cache(jnp.asarray(0), 4)
    assert int(state) == 4
    assert cache.info().live_entries == 1

    cache.rebind(lambda value: value + 2)
    info = cache.info()
    assert info.generation == 1
    assert info.live_entries == 0
    assert info.step_counts == ()

    state = cache(state, 3)
    assert int(state) == 10
    assert cache.info().live_entries == 1
