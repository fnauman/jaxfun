"""Validation tests for the dense linear eigenvalue / non-modal Couette tools.

These complement ``test_taylor_couette.py`` (which covers the cylindrical
solvers' modal onsets) by exercising the shared transient-growth machinery
(``_linear_analysis``) and the plane-Couette operator (``_pcf_linear``) against
exact and published benchmarks:

  * transient-growth helper vs a direct matrix-exponential propagator
    (pure-math unit test, no physics),
  * plane Couette is linearly stable for all Re (Romanov 1973),
  * the least-damped plane-Couette eigenvalue equals the analytic streamwise-roll
    viscous decay s = -nu (kz^2 + (pi/2)^2) on [-1, 1],
  * optimal transient growth scales as G ~ Re^2 with t_opt ~ Re
    (Gustavsson 1991; Reddy & Henningson 1993; Trefethen et al. 1993),
  * the streamwise-vortex optimal G* = 1165.9 at Re=1000, (alpha,beta)=(0,1.66)
    (Butler & Farrell 1992; Reddy-Henningson NASA report),
  * the MHD operator's *kinetic* energy gain reduces exactly to the hydro value
    as B0 -> 0, while an imposed field monotonically suppresses transient growth.

Run with the shenfun environment:
    conda run -n shenfun pytest -q demo/test_couette_linear.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy.linalg import eig, expm, svdvals

sys.path.insert(0, os.path.dirname(__file__))

from _linear_analysis import (
    FINITE_CAP,
    finite_eigensystem,
    parse_times,
    transient_growth_from_eigs,
)
from _pcf_linear import PlaneCouetteLinear


# ---------------------------------------------------------------------------
# transient_growth_from_eigs: pure-math unit test vs matrix exponential
# ---------------------------------------------------------------------------
def test_modal_transient_growth_matches_matrix_exponential():
    """For a full-rank non-normal A and metric Q, the modal-expansion gain must
    equal sigma_max(F exp(At) F^{-1})^2 with Q = F^H F, computed directly."""
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A - 3.0 * np.eye(n)                       # push spectrum left (decaying)
    # SPD metric Q = L L^H
    Lq = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q = Lq @ Lq.conj().T
    F = np.linalg.cholesky(Q).conj().T            # Q = F^H F

    evals, V = eig(A)
    times = [0.0, 0.3, 1.0, 2.5]
    rows = transient_growth_from_eigs(evals, V, Q, times)

    for r in rows:
        prop = F @ expm(A * r["t"]) @ np.linalg.inv(F)
        g_direct = float(svdvals(prop)[0] ** 2)
        assert math.isclose(r["gain"], g_direct, rel_tol=1e-7, abs_tol=1e-9)
    # t=0 is the identity -> unit gain
    assert math.isclose(rows[0]["gain"], 1.0, rel_tol=1e-10)


def test_parse_times_roundtrip_and_validation():
    assert np.allclose(parse_times("1, 5; 10"), [1.0, 5.0, 10.0])
    assert np.allclose(parse_times([2.0, 4.0]), [2.0, 4.0])
    with pytest.raises(ValueError):
        parse_times("-1, 2")
    with pytest.raises(ValueError):
        parse_times("")


# ---------------------------------------------------------------------------
# Plane Couette modal stability (Romanov 1973) and exact Squire eigenvalues
# ---------------------------------------------------------------------------
def test_plane_couette_linearly_stable_all_modes():
    """Romanov (1973): plane Couette flow has no exponentially growing mode."""
    pcf = PlaneCouetteLinear.couette(nx=64, Re=1000.0, U_wall=1.0, mhd=False)
    for ky, kz in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (0.5, 2.0)]:
        assert pcf.growth_rate(ky, kz) < 0.0


@pytest.mark.parametrize("kz", [1.0, 2.5])
def test_plane_couette_streamwise_roll_eigenvalue_is_analytic(kz):
    """For ky=0 the spanwise/streamwise velocity decouples into a heat equation;
    its slowest mode decays at s = -nu (kz^2 + (pi/2)^2) on the half-gap-1 domain,
    and the full Squire ladder -nu (kz^2 + (j pi/2)^2) appears in the spectrum."""
    Re = 1000.0
    nu = 1.0 / Re
    pcf = PlaneCouetteLinear.couette(nx=96, Re=Re, U_wall=1.0, mhd=False)
    w, _ = pcf.eigs(0.0, kz, n_return=40)
    lead = w[0].real
    assert math.isclose(lead, -nu * (kz**2 + (math.pi / 2) ** 2), rel_tol=1e-6)
    spec = w.real
    for j in (1, 2, 3):
        target = -nu * (kz**2 + (j * math.pi / 2) ** 2)
        assert np.min(np.abs(spec - target)) < 1e-7 * abs(target) + 1e-9


# ---------------------------------------------------------------------------
# Plane Couette optimal transient growth: scaling and the canonical value
# ---------------------------------------------------------------------------
def test_plane_couette_optimal_growth_butler_farrell():
    """Streamwise-vortex optimal at Re=1000, (alpha,beta)=(0,1.66): G*~1165."""
    pcf = PlaneCouetteLinear.couette(nx=80, Re=1000.0, U_wall=1.0, mhd=False)
    g = pcf.nonmodal_growth(0.0, 1.66, [139.0])[0]["gain"]
    assert math.isclose(g, 1165.93, rel_tol=2e-3)


def test_plane_couette_optimal_growth_scales_as_Re_squared():
    """G_max ~ Re^2 and t_opt ~ Re (Gustavsson 1991; Reddy & Henningson 1993)."""
    g_over_re2 = []
    t_over_re = []
    for Re in (500.0, 1000.0, 2000.0):
        pcf = PlaneCouetteLinear.couette(nx=80, Re=Re, U_wall=1.0, mhd=False)
        rows = pcf.nonmodal_growth(0.0, 1.66, np.linspace(0.4 * Re * 0.3,
                                                          0.4 * Re, 40))
        best = max(rows, key=lambda r: r["gain"])
        g_over_re2.append(best["gain"] / Re**2)
        t_over_re.append(best["t"] / Re)
    assert np.std(g_over_re2) / np.mean(g_over_re2) < 0.02
    assert np.std(t_over_re) / np.mean(t_over_re) < 0.05


# ---------------------------------------------------------------------------
# MHD plane Couette: energy-norm behaviour
# ---------------------------------------------------------------------------
def test_mhd_kinetic_norm_reduces_to_hydro_at_zero_field():
    """With B0=0 the velocity decouples; the *kinetic* gain matches hydro to
    machine precision, while the *total* (kin+mag) gain need not (the shear
    drives independent magnetic transient growth)."""
    ky, kz, t = 1.0, 1.0, [50.0]
    hyd = PlaneCouetteLinear.couette(nx=48, Re=500.0, mhd=False)
    mhd = PlaneCouetteLinear.couette(nx=48, Re=500.0, Rm=500.0, mhd=True,
                                     by=0.0, bz=0.0)
    g_hyd = hyd.nonmodal_growth(ky, kz, t)[0]["gain"]
    g_kin = mhd.nonmodal_growth(ky, kz, t, energy="kinetic")[0]["gain"]
    g_tot = mhd.nonmodal_growth(ky, kz, t, energy="total")[0]["gain"]
    assert math.isclose(g_hyd, g_kin, rel_tol=1e-8)
    assert g_tot >= g_kin - 1e-12
    with pytest.raises(ValueError):
        hyd.nonmodal_growth(ky, kz, t, energy="magnetic")


def test_mhd_imposed_field_suppresses_transient_growth():
    """An imposed vertical field reduces total-energy transient growth
    monotonically via magnetic tension (cf. Camobreco et al. 2020/2021)."""
    ky, kz, t = 1.0, 1.0, [50.0]
    gains = []
    for bz in (0.0, 0.05, 0.1, 0.2):
        mhd = PlaneCouetteLinear.couette(nx=48, Re=500.0, Rm=500.0, mhd=True,
                                         by=0.0, bz=bz)
        gains.append(mhd.nonmodal_growth(ky, kz, t, energy="total")[0]["gain"])
    assert all(b < a for a, b in zip(gains, gains[1:]))


def test_finite_cap_constant_is_shared():
    """The non-modal modal-basis cap is a single shared default everywhere."""
    assert FINITE_CAP == 1.0e8
    # exercising the default path must reproduce the explicit-cap result
    pcf = PlaneCouetteLinear.couette(nx=48, Re=1000.0, mhd=False)
    a = pcf.nonmodal_growth(0.0, 1.66, [139.0])[0]["gain"]
    b = pcf.nonmodal_growth(0.0, 1.66, [139.0], finite_cap=FINITE_CAP)[0]["gain"]
    assert math.isclose(a, b, rel_tol=1e-12)
