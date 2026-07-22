"""Constant-step third-order semi-implicit BDF3/EXT3 helpers."""

from __future__ import annotations

from typing import TypeVar

import jax

T = TypeVar("T")

BDF3_LEADING = 11.0 / 6.0
IMPLICIT_SCALE = 1.0 / BDF3_LEADING


def sbdf3_mass_history(current: T, previous: T, older: T) -> T:
    """Return the normalized BDF3 solution-history contribution.

    Inputs are already mass-matrix rows. The result is
    ``18/11 current - 9/11 previous + 2/11 older``.
    """

    return jax.tree.map(
        lambda n, nm1, nm2: (18.0 * n - 9.0 * nm1 + 2.0 * nm2) / 11.0,
        current,
        previous,
        older,
    )


def sbdf3_explicit_history(current: T, previous: T, older: T) -> T:
    """Return normalized third-order explicit extrapolation rows.

    The result is ``(6/11) * (3 current - 3 previous + older)``.
    """

    return jax.tree.map(
        lambda n, nm1, nm2: (18.0 * n - 18.0 * nm1 + 6.0 * nm2) / 11.0,
        current,
        previous,
        older,
    )


def sbdf3_rhs(
    mass_current: T,
    mass_previous: T,
    mass_older: T,
    nonlinear_current: T,
    nonlinear_previous: T,
    nonlinear_older: T,
    dt,
) -> T:
    """Assemble the normalized right-hand side for ``M u' = L u + N``.

    The corresponding implicit operator is ``M - (6/11) dt L``.
    """

    mass = sbdf3_mass_history(mass_current, mass_previous, mass_older)
    nonlinear = sbdf3_explicit_history(
        nonlinear_current, nonlinear_previous, nonlinear_older
    )
    return jax.tree.map(lambda m, n: m + dt * n, mass, nonlinear)
