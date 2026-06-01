"""Named Galerkin operator constructors used by Couette solvers."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from jaxfun.galerkin import TrialFunction, inner
from jaxfun.typing import GalerkinOperator


def _coords_from_trial(trial: TrialFunction) -> tuple[sp.Symbol, ...]:
    return tuple(trial.function_space.system.base_scalars())


def laplacian(expr: sp.Expr, coords: Sequence[sp.Symbol]) -> sp.Expr:
    """Return the Cartesian Laplacian over *coords*."""
    return sum(sp.diff(expr, coord, 2) for coord in coords)


def Helmholtz(
    test: sp.Expr,
    trial: TrialFunction,
    *,
    coeff: float | sp.Expr = 1.0,
    diffusivity: float | sp.Expr = 1.0,
    coords: Sequence[sp.Symbol] | None = None,
    sparse: bool = True,
) -> GalerkinOperator:
    """Assemble ``(v,u) - coeff*(v,diffusivity*lap(u))``.

    This mirrors the KMM Helmholtz blocks in
    ``shenfun/demo/ChannelFlow.py:149-155`` and is reusable for scalar
    Fourier-polynomial implicit solves.
    """
    coords = _coords_from_trial(trial) if coords is None else tuple(coords)
    return inner(test * trial, sparse=sparse) - coeff * inner(
        test * (diffusivity * laplacian(trial, coords)), sparse=sparse
    )


def Biharmonic(
    test: sp.Expr,
    trial: TrialFunction,
    *,
    coeff: float | sp.Expr = 1.0,
    diffusivity: float | sp.Expr = 1.0,
    coords: Sequence[sp.Symbol] | None = None,
    sparse: bool = True,
) -> GalerkinOperator:
    """Assemble ``(v,lap(u)) - coeff*(v,diffusivity*lap(lap(u)))``.

    This is the wall-normal velocity block used by the KMM Couette solvers in
    ``shenfun/demo/ChannelFlow.py:149-155``.
    """
    coords = _coords_from_trial(trial) if coords is None else tuple(coords)
    lap_u = laplacian(trial, coords)
    return inner(test * lap_u, sparse=sparse) - coeff * inner(
        test * (diffusivity * laplacian(lap_u, coords)), sparse=sparse
    )
