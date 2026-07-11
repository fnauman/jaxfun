"""Crank-Nicolson / Adams-Bashforth-2 helpers for coupled systems.

The formulas mirror the Taylor-Couette CNAB2 update in
``couette/taylor_couette_dns.py:288-313``: the linear operator is split into
an implicit Crank-Nicolson block and an explicit Crank-Nicolson block, while
the nonlinear term uses Adams-Bashforth-2 with an IMEX-Euler first step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


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
