"""Crank-Nicolson / Adams-Bashforth-2 helpers for coupled systems.

The formulas mirror the Taylor-Couette CNAB2 update in
``couette/taylor_couette_dns.py:288-313``: the linear operator is split into
an implicit Crank-Nicolson block and an explicit Crank-Nicolson block, while
the nonlinear term uses Adams-Bashforth-2 with an IMEX-Euler first step.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


@dataclass(frozen=True)
class ScanRolloutCacheInfo:
    """Operational counters for a bounded persistent scan cache."""

    generation: int
    max_entries: int
    live_entries: int
    step_counts: tuple[int, ...]
    hits: int
    misses: int
    evictions: int


class ScanRolloutCache[T]:
    """Reuse compiled scan rollouts without retaining unbounded executables.

    Each distinct block length needs a static lax.scan executable for
    reverse-mode compatibility. The small LRU bounds those variants, while
    rebind clears every compiled executable when a solver rebuilds
    timestep-dependent factors. Runtime controller values may be supplied by
    ``dynamic_args`` so adaptive changes do not alter the compiled program.
    """

    def __init__(
        self,
        step: Callable[..., T],
        *,
        max_entries: int = 8,
        dynamic_args: Callable[[], tuple[object, ...]] | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._step = step
        self._scan_step = jax.checkpoint(step)
        self._dynamic_args = dynamic_args or tuple
        self._max_entries = int(max_entries)
        self._rollouts: OrderedDict[int, Callable[..., T]] = OrderedDict()
        self._generation = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _compile_for_steps(self, steps: int) -> Callable[..., T]:
        step = self._scan_step

        @jax.jit
        def rollout(state: T, dynamic_args: tuple[object, ...]) -> T:
            return scan_steps(step, state, steps, *dynamic_args)

        return rollout

    @staticmethod
    def _clear_compiled(rollout: Callable[..., T]) -> None:
        clear_cache = getattr(rollout, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

    def __call__(self, state: T, steps: int) -> T:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return state
        dynamic_args = tuple(self._dynamic_args())
        if jax.device_count() > 1 or has_concrete_multi_device_leaf(state):
            return scan_steps(self._step, state, steps, *dynamic_args)

        rollout = self._rollouts.pop(steps, None)
        if rollout is None:
            self._misses += 1
            rollout = self._compile_for_steps(steps)
            if len(self._rollouts) >= self._max_entries:
                _old_steps, old_rollout = self._rollouts.popitem(last=False)
                self._clear_compiled(old_rollout)
                self._evictions += 1
        else:
            self._hits += 1
        self._rollouts[steps] = rollout
        return rollout(state, dynamic_args)

    def rebind(self, step: Callable[..., T]) -> None:
        """Drop obsolete executables and bind a rebuilt solver step."""
        for rollout in self._rollouts.values():
            self._clear_compiled(rollout)
        self._rollouts.clear()
        self._step = step
        self._scan_step = jax.checkpoint(step)
        self._generation += 1

    def info(self) -> ScanRolloutCacheInfo:
        """Return stable counters without exposing JAX cache internals."""
        return ScanRolloutCacheInfo(
            generation=self._generation,
            max_entries=self._max_entries,
            live_entries=len(self._rollouts),
            step_counts=tuple(self._rollouts),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    def compiled_memory_analysis(self, state: T, steps: int) -> object | None:
        """Return XLA memory analysis for an already-cached rollout.

        Benchmarking production must inspect the same compiled ``solve`` graph
        used by the time integrator. Reconstructing a separate jitted scan can
        hide constant-capture and executable-size regressions, so expose this
        narrow read-only hook while keeping the cached callable private.

        ``None`` is returned for zero-step and concrete multi-device rollouts,
        which do not use the single-device executable cache.
        """

        steps = int(steps)
        if (
            steps == 0
            or jax.device_count() > 1
            or has_concrete_multi_device_leaf(state)
        ):
            return None
        rollout = self._rollouts.get(steps)
        if rollout is None:
            raise ValueError(
                f"rollout for {steps} steps is not cached; execute it before analysis"
            )
        return (
            rollout.lower(state, tuple(self._dynamic_args()))
            .compile()
            .memory_analysis()
        )


def has_concrete_multi_device_leaf(tree: object) -> bool:
    """Return whether a concrete pytree leaf spans multiple JAX devices."""

    for leaf in jax.tree.leaves(tree):
        devices = getattr(leaf, "devices", None)
        if devices is None:
            continue
        try:
            if len(devices()) > 1:
                return True
        except (jax.errors.ConcretizationTypeError, TypeError, AttributeError):
            continue
    return False


def batch_components(
    fn: Callable[[object], object], values: tuple[object, ...]
) -> object:
    """Batch component transforms without collapsing concrete sharding.

    Vmap is the fast single-device path. For concrete multi-device arrays,
    applying the transform component-by-component preserves each transform's
    explicit sharding. Transposing the result pytrees into tuples avoids
    introducing a leading component axis that would change which axis a
    spectral NamedSharding partitions.
    """

    if jax.device_count() > 1 or has_concrete_multi_device_leaf(values):
        transformed = tuple(fn(value) for value in values)
        return jax.tree.map(lambda *leaves: tuple(leaves), *transformed)
    return jax.vmap(fn)(jnp.stack(values))


def ab2_extrapolate(current: T, previous: T, have_previous: bool | jax.Array) -> T:
    """Return ``current`` on the first step, else ``1.5*current - 0.5*previous``.

    ``current`` and ``previous`` may be arbitrary JAX pytrees with matching
    leaves.  ``have_previous`` is intentionally array-compatible so callers can
    use this inside ``jax.lax.scan`` without Python control flow on traced data.
    """
    try:
        concrete_have = bool(have_previous)
    except Exception:
        concrete_have = None
    if concrete_have is False:
        return current
    if concrete_have is True:
        return jax.tree.map(lambda c, p: 1.5 * c - 0.5 * p, current, previous)

    have = jnp.asarray(have_previous) != 0
    return jax.tree.map(
        lambda c, p: jnp.where(have, 1.5 * c - 0.5 * p, c),
        current,
        previous,
    )


def variable_ab2_extrapolate(
    current: T,
    previous: T,
    have_previous: bool | jax.Array,
    dt: float | jax.Array,
    previous_dt: float | jax.Array,
) -> T:
    """Extrapolate a nonlinear term to the midpoint of a variable step.

    For r = dt / previous_dt the second-order coefficients are
    (1 + r/2) * current - (r/2) * previous. The bootstrap returns
    current and uses a safe denominator so an unset zero previous_dt
    cannot contaminate traced-but-masked branches.
    """

    have = jnp.asarray(have_previous) != 0
    dt_array = jnp.asarray(dt)
    previous_dt_array = jnp.asarray(previous_dt, dtype=dt_array.dtype)
    denominator = jnp.where(have, previous_dt_array, dt_array)
    ratio = dt_array / denominator
    return jax.tree.map(
        lambda c, p: jnp.where(
            have,
            (1.0 + 0.5 * ratio) * c - 0.5 * ratio * p,
            c,
        ),
        current,
        previous,
    )


def cnab2_rhs(
    explicit_linear: T,
    nonlinear_current: T,
    nonlinear_previous: T,
    have_previous: bool | jax.Array,
) -> T:
    """Combine explicit linear and nonlinear terms for one CNAB2 step.

    The returned tree is ``explicit_linear - N_ab`` where ``N_ab`` is the
    first-step IMEX-Euler nonlinear term or the AB2 extrapolation.
    """
    nonlinear = ab2_extrapolate(nonlinear_current, nonlinear_previous, have_previous)
    return jax.tree.map(lambda rhs, n: rhs - n, explicit_linear, nonlinear)


def scan_steps(
    step: Callable[..., T], state: T, steps: int, *dynamic_args: object
) -> T:
    """Advance ``state`` with ``step`` using ``jax.lax.scan``.

    This keeps Couette time loops staged as one JAX loop while preserving the
    solver-specific ``step(state) -> state`` API.  When multiple JAX devices
    are visible, concrete states use an eager loop so replicated and sharded
    rollouts follow the same schedule and sharded transform paths avoid tracing
    through ``jax.lax.scan``.
    """
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return state
    if jax.device_count() > 1 or has_concrete_multi_device_leaf(state):
        out = state
        for _ in range(steps):
            out = step(out, *dynamic_args)
        return out

    def body(carry: T, _unused: None) -> tuple[T, None]:
        return step(carry, *dynamic_args), None

    final, _ = jax.lax.scan(body, state, xs=None, length=steps)
    return final
