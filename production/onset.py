"""PCF-MRI linear onset driver (FJ-10).

A discrete-mode lattice driver over ``k_y = 2*pi*n/Ly`` and ``k_z = 2*pi*m/Lz`` plus
a bracket-and-bisect on the resistive Reynolds number ``Rm`` at fixed
``(Re, B0, BC, aspect)``. The growth rate increases with ``Rm`` (less resistive), so a
sign change of the maximum real eigenvalue brackets the marginal ``Rm``.

Convention traps (encoded here and in the tests):

* the shearing-box uses ``S = +1`` with ``U'(x) = -S`` (signed shear), via
  ``PlaneCouetteLinear.shearpy(shear_rate=1.0)``;
* the wall eigensolver domain is the full width ``[-1, 1]`` so the half-gap is
  ``h = 1`` -- construct the operator directly rather than via any ``a = 0.5`` atlas
  helper;
* the imposed vertical field magnitude is ``B0 = bz``.

The driver is eigensolver-agnostic: pass a factory ``op_factory(Rm)`` returning any
object exposing ``growth_rate(ky, kz) -> float``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LatticeMode:
    n: int
    m: int
    ky: float
    kz: float
    growth: float


@dataclass(frozen=True)
class CriticalResult:
    ky: float
    kz: float
    critical_Rm: float
    growth_at_hi: float
    iterations: int
    bracket: tuple[float, float]


def lattice_wavenumbers(
    *, n_y: int, n_z: int, Ly: float, Lz: float, include_ky_zero: bool = True
) -> list[tuple[int, int, float, float]]:
    """Return ``(n, m, k_y, k_z)`` over the admissible discrete-mode lattice."""

    modes: list[tuple[int, int, float, float]] = []
    n_start = 0 if include_ky_zero else 1
    for n in range(n_start, n_y + 1):
        for m in range(1, n_z + 1):  # k_z = 0 has no MRI; start at m = 1
            ky = 2.0 * math.pi * n / Ly
            kz = 2.0 * math.pi * m / Lz
            if ky == 0.0 and kz == 0.0:
                continue
            modes.append((n, m, ky, kz))
    return modes


def growth_over_lattice(
    op: Any, *, n_y: int, n_z: int, Ly: float, Lz: float, include_ky_zero: bool = True
) -> list[LatticeMode]:
    """Evaluate growth over the lattice, sorted most-unstable first."""

    out = [
        LatticeMode(n, m, ky, kz, float(op.growth_rate(ky, kz)))
        for (n, m, ky, kz) in lattice_wavenumbers(
            n_y=n_y, n_z=n_z, Ly=Ly, Lz=Lz, include_ky_zero=include_ky_zero
        )
    ]
    out.sort(key=lambda mode: mode.growth, reverse=True)
    return out


def critical_Rm(
    op_factory: Callable[[float], Any],
    *,
    ky: float,
    kz: float,
    Rm_lo: float,
    Rm_hi: float,
    tol: float = 1.0e-3,
    max_iter: int = 80,
    expand: bool = True,
) -> CriticalResult:
    """Bracket and bisect the marginal ``Rm`` (growth = 0) for a fixed mode.

    ``op_factory(Rm)`` returns an eigensolver; growth is assumed monotone increasing
    in ``Rm``. If ``expand`` and the initial bracket does not straddle marginality,
    ``Rm_hi`` is doubled up to a cap before giving up.
    """

    def growth(rm: float) -> float:
        return float(op_factory(rm).growth_rate(ky, kz))

    g_lo = growth(Rm_lo)
    g_hi = growth(Rm_hi)
    iterations = 0
    while expand and g_lo > 0.0 and iterations < 40:
        # already unstable at the low end: lower the bracket
        Rm_hi = Rm_lo
        g_hi = g_lo
        Rm_lo *= 0.5
        g_lo = growth(Rm_lo)
        iterations += 1
    while expand and g_hi < 0.0 and iterations < 40:
        Rm_lo = Rm_hi
        g_lo = g_hi
        Rm_hi *= 2.0
        g_hi = growth(Rm_hi)
        iterations += 1
    if not (g_lo <= 0.0 <= g_hi):
        raise ValueError(
            f"could not bracket marginal Rm for (ky={ky:g}, kz={kz:g}): "
            f"growth({Rm_lo:g})={g_lo:g}, growth({Rm_hi:g})={g_hi:g}"
        )

    bracket = (Rm_lo, Rm_hi)
    lo, hi = Rm_lo, Rm_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        g_mid = growth(mid)
        iterations += 1
        if g_mid > 0.0:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= tol * max(1.0, hi):
            break
    rm_c = 0.5 * (lo + hi)
    return CriticalResult(
        ky=ky,
        kz=kz,
        critical_Rm=rm_c,
        growth_at_hi=growth(hi),
        iterations=iterations,
        bracket=bracket,
    )


def critical_Rm_over_lattice(
    op_factory: Callable[[float], Any],
    *,
    n_y: int,
    n_z: int,
    Ly: float,
    Lz: float,
    Rm_lo: float,
    Rm_hi: float,
    tol: float = 1.0e-3,
) -> dict[str, Any]:
    """Return the lowest-critical-Rm mode plus its next-nearest competitors."""

    results: list[tuple[tuple[int, int], CriticalResult]] = []
    for n, m, ky, kz in lattice_wavenumbers(n_y=n_y, n_z=n_z, Ly=Ly, Lz=Lz):
        try:
            res = critical_Rm(
                op_factory, ky=ky, kz=kz, Rm_lo=Rm_lo, Rm_hi=Rm_hi, tol=tol
            )
        except ValueError:
            continue
        results.append(((n, m), res))
    if not results:
        raise ValueError("no mode reached marginality within the search bracket")
    results.sort(key=lambda item: item[1].critical_Rm)
    (winner_nm, winner) = results[0]
    return {
        "winner": {"n": winner_nm[0], "m": winner_nm[1], "result": winner},
        "competitors": [
            {"n": nm[0], "m": nm[1], "critical_Rm": res.critical_Rm}
            for nm, res in results[1:5]
        ],
    }
