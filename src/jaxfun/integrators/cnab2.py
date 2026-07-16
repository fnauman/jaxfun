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
    timestep-dependent factors.
    """

    def __init__(self, step: Callable[[T], T], *, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._step = step
        self._max_entries = int(max_entries)
        self._rollouts: OrderedDict[int, Callable[[T], T]] = OrderedDict()
        self._generation = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _compile_for_steps(self, steps: int) -> Callable[[T], T]:
        step = self._step

        @jax.jit
        def rollout(state: T) -> T:
            return scan_steps(step, state, steps)

        return rollout

    @staticmethod
    def _clear_compiled(rollout: Callable[[T], T]) -> None:
        clear_cache = getattr(rollout, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()

    def __call__(self, state: T, steps: int) -> T:
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return state
        if jax.device_count() > 1 or _has_concrete_multi_device_leaf(state):
            return scan_steps(self._step, state, steps)

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
        return rollout(state)

    def rebind(self, step: Callable[[T], T]) -> None:
        """Drop obsolete executables and bind a rebuilt solver step."""
        for rollout in self._rollouts.values():
            self._clear_compiled(rollout)
        self._rollouts.clear()
        self._step = step
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


def _has_concrete_multi_device_leaf(tree: object) -> bool:
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


def scan_steps(step: Callable[[T], T], state: T, steps: int) -> T:
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
    if jax.device_count() > 1 or _has_concrete_multi_device_leaf(state):
        out = state
        for _ in range(steps):
            out = step(out)
        return out

    def body(carry: T, _unused: None) -> tuple[T, None]:
        return step(carry), None

    final, _ = jax.lax.scan(body, state, xs=None, length=steps)
    return final
