"""Reusable helpers for coupled multi-equation IMEX stages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import jax

T = TypeVar("T")


def ars_stage_rhs(
    base_rhs: T,
    nonlinear_history: Sequence[T],
    linear_history: Sequence[T],
    a,
    b,
    dt: float,
    rk: int,
) -> T:
    """Assemble one ARS IMEX-RK explicit RHS for a coupled pytree state.

    This is the coupled-equation equivalent of the stage accumulation in
    ``shenfun/shenfun/utilities/integrators.py:PDEIMEXRK`` and the Couette KMM
    channel-flow update in ``shenfun/demo/ChannelFlow.py:177-194``.  Each leaf
    of ``base_rhs`` is one equation's mass-applied initial state; histories hold
    matching pytrees of nonlinear and linear stage scalar products.
    """
    rk = int(rk)
    rhs = base_rhs
    for j in range(rk + 1):
        rhs = jax.tree.map(
            lambda r, n, coeff=b[rk + 1, j]: r + dt * coeff * n,
            rhs,
            nonlinear_history[j],
        )
    for j in range(rk):
        rhs = jax.tree.map(
            lambda r, ell, coeff=a[rk + 1, j + 1]: r + dt * coeff * ell,
            rhs,
            linear_history[j],
        )
    return rhs
