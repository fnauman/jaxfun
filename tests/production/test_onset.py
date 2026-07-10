"""FJ-10: onset lattice/bisection driver logic (synthetic, no eigensolves)."""

from __future__ import annotations

import math

import pytest

from production.onset import (
    critical_Rm,
    critical_Rm_over_lattice,
    growth_over_lattice,
    lattice_wavenumbers,
)


class _SyntheticOp:
    """growth(ky,kz) = strength(ky,kz) * (Rm - Rm_c(ky,kz)); monotone in Rm."""

    def __init__(self, Rm, *, rm_c):
        self.Rm = Rm
        self._rm_c = rm_c

    def growth_rate(self, ky, kz):
        rm_c = self._rm_c(ky, kz)
        return 1e-3 * (self.Rm - rm_c)


def test_lattice_wavenumbers_cover_grid():
    modes = lattice_wavenumbers(n_y=1, n_z=2, Ly=2.0, Lz=0.5)
    kz = {round(kz, 6) for (_, _, _, kz) in modes}
    ky = {round(ky, 6) for (_, _, ky, _) in modes}
    assert 0.0 in ky and round(2 * math.pi / 2.0, 6) in ky
    assert round(2 * math.pi / 0.5, 6) in kz  # m=1
    assert all(kz_ != 0.0 for (_, _, _, kz_) in modes)  # kz starts at m=1


def test_critical_Rm_bisects_a_monotone_growth():
    res = critical_Rm(
        lambda Rm: _SyntheticOp(Rm, rm_c=lambda ky, kz: 415.0),
        ky=0.0, kz=12.566, Rm_lo=350.0, Rm_hi=500.0, tol=1e-4,
    )
    assert res.critical_Rm == pytest.approx(415.0, abs=0.1)


def test_critical_Rm_expands_bracket_when_needed():
    res = critical_Rm(
        lambda Rm: _SyntheticOp(Rm, rm_c=lambda ky, kz: 3000.0),
        ky=0.0, kz=25.0, Rm_lo=350.0, Rm_hi=500.0, tol=1e-3,
    )
    assert res.critical_Rm == pytest.approx(3000.0, rel=1e-2)


def test_growth_over_lattice_sorted_most_unstable_first():
    op = _SyntheticOp(1000.0, rm_c=lambda ky, kz: 400.0 + 10.0 * kz)
    modes = growth_over_lattice(op, n_y=1, n_z=3, Ly=2.0, Lz=0.5)
    growths = [m.growth for m in modes]
    assert growths == sorted(growths, reverse=True)


def test_critical_Rm_over_lattice_picks_lowest():
    # lower kz -> lower critical Rm here, so the m=1 mode should win
    factory = lambda Rm: _SyntheticOp(Rm, rm_c=lambda ky, kz: 300.0 + 5.0 * kz)
    out = critical_Rm_over_lattice(
        factory, n_y=1, n_z=3, Ly=2.0, Lz=0.5, Rm_lo=300.0, Rm_hi=5000.0
    )
    winner = out["winner"]
    assert winner["m"] == 1
    assert out["competitors"]  # next-nearest recorded
