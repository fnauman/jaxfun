"""Validation tests for the Taylor-Couette hydro and MHD/MRI linear solvers.

These assert against classic published benchmarks:
  * circular-Couette base flow and the Rayleigh criterion (exact/algebraic),
  * hydrodynamic critical Reynolds number Re_c=68.19, a_c~3.16 at eta=0.5
    (Fasel & Booz 1984),
  * principle of exchange of stabilities (stationary m=0 onset),
  * ideal local Keplerian MRI optimum s_max=0.75 Omega, (k vA)^2=0.9375 Omega^2
    (Balbus & Hawley 1991),
  * the Keplerian hydro null (Rayleigh-stable) vs MRI onset with an axial field.

Run with the shenfun environment:  conda run -n shenfun pytest -q demo/test_taylor_couette.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
from taylor_couette_mri import (
    TaylorCouetteMRI,
    mri_keplerian_optimum,
    mri_local_growth,
)


# ---------------------------------------------------------------------------
# Base flow and Rayleigh criterion (algebraic, fast)
# ---------------------------------------------------------------------------
def test_base_flow_matches_wall_rotation():
    base = CircularCouette(1.0, 2.0, 1.3, 0.4)
    assert math.isclose(base.V(base.R1), base.Omega1 * base.R1, rel_tol=1e-13)
    assert math.isclose(base.V(base.R2), base.Omega2 * base.R2, rel_tol=1e-13)
    # 2*Omega + r*Omega' = 2a everywhere
    rr = np.linspace(base.R1, base.R2, 17)
    twoOm = 2 * base.Omega(rr)
    rOmp = -2 * base.b / rr**2
    assert np.allclose(twoOm + rOmp, 2 * base.a, atol=1e-12)


@pytest.mark.parametrize(
    "mu, expect_stable",
    [(0.0, False), (0.20, False), (0.30, True), (0.60, True)],
)
def test_rayleigh_criterion_line(mu, expect_stable):
    # eta=0.5 -> Rayleigh line mu=eta^2=0.25
    base = CircularCouette(1.0, 2.0, 1.0, mu)
    assert base.rayleigh_stable() is expect_stable


def test_keplerian_is_rayleigh_stable():
    eta = 0.5
    base = CircularCouette(1.0, 1.0 / eta, 1.0, eta**1.5)  # mu=eta^1.5 Keplerian
    assert base.rayleigh_stable()
    assert math.isclose(base.q_shear(base.R1), 1.5, rel_tol=0.2)


# ---------------------------------------------------------------------------
# Hydrodynamic linear stability
# ---------------------------------------------------------------------------
def test_exchange_of_stabilities_m0():
    """m=0 axisymmetric Taylor onset is stationary: Im(s_leading) ~ 0."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    s = TaylorCouetteLinear(base, nu=2e-3, N=40)
    w, _ = s.eigs(0, 3.14, n_return=1)
    assert abs(w[0].imag) < 1e-9
    assert w[0].real > 0  # supercritical -> unstable


@pytest.mark.slow
def test_critical_reynolds_eta_half():
    """eta=0.5, mu=0: Re_c=68.19, a_c~3.16 (Fasel & Booz 1984)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    s = TaylorCouetteLinear(base, N=40)
    res = s.critical_reynolds(m=0, kz_list=np.linspace(2.6, 3.8, 9))
    assert abs(res["Re_c"] - 68.19) < 1.0
    assert abs(res["a_c"] - 3.16) < 0.1


def test_critical_reynolds_returns_none_without_unstable_band():
    """Stable hydro scans should not fail with an unpacking TypeError."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteLinear(base, N=24)
    assert s.critical_reynolds(m=0, kz_list=np.array([1.0, 2.0]), refine=False) is None


def test_critical_reynolds_keeps_coarse_result_if_refinement_misses():
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    s = TaylorCouetteLinear(base, N=24)
    calls = []

    def fake_critical_over_kz(m, kz_list, **kw):
        calls.append(np.asarray(kz_list))
        return (3.1, 0.014) if len(calls) == 1 else None

    s.critical_over_kz = fake_critical_over_kz
    res = s.critical_reynolds(m=0, kz_list=np.array([3.0, 3.2]), refine=True)

    assert res["kz_c"] == 3.1
    assert res["nu_c"] == 0.014
    assert math.isclose(res["Re_c"], base.Omega1 * base.R1 * base.gap / 0.014)
    assert len(calls) == 2


def test_keplerian_hydro_is_stable():
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteLinear(base, nu=1e-3, N=40)
    _, g, _ = s.max_growth_over_kz(0, np.linspace(0.5, 8, 12))
    assert g < 1e-7


def test_hydro_nonaxisymmetric_mirror_symmetry():
    """Changing m -> -m conjugates the non-axisymmetric spectrum."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.2)
    s = TaylorCouetteLinear(base, nu=2e-3, N=28)
    wp, _ = s.eigs(1, 2.7, n_return=4)
    wm, _ = s.eigs(-1, 2.7, n_return=4)
    assert np.allclose(wp.real, wm.real, atol=1e-9)
    assert np.allclose(wp.imag, -wm.imag, atol=1e-9)


# ---------------------------------------------------------------------------
# Ideal local MRI dispersion (analytic)
# ---------------------------------------------------------------------------
def test_local_mri_keplerian_optimum():
    opt = mri_keplerian_optimum(Omega=1.0, vA=1.0)
    assert abs(opt["s_max_over_Omega"] - 0.75) < 2e-3
    assert abs(opt["wa2_opt_over_O2"] - 0.9375) < 5e-3


def test_local_mri_cutoff_and_stability():
    # ideal Keplerian: unstable for 0<omega_A^2<3, stable beyond cutoff
    O, k2, dO2 = 1.0, 1.0, -3.0
    assert mri_local_growth(0.5, O, k2, dO2) > 0  # inside band
    assert mri_local_growth(math.sqrt(3.0) * 1.01, O, k2, dO2) == 0.0  # past cutoff


# ---------------------------------------------------------------------------
# Global MHD / MRI
# ---------------------------------------------------------------------------
def test_mri_null_keplerian_no_field_is_stable():
    """B0=0 quasi-Keplerian: no MRI, magnetic modes decay -> stable."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.0, nu=1e-3, eta_mag=1e-3, N=36)
    _, g, _ = s.max_growth_over_kz(0, np.linspace(0.5, 6, 12))
    assert g < 1e-7


def _eval_complex_mode(space, coeffs, points, derivative=False):
    from shenfun import Dx, Function

    real_part = Function(space)
    imag_part = Function(space)
    real_part[:] = 0.0
    imag_part[:] = 0.0
    real_part[space.slice()] = np.asarray(coeffs).real
    imag_part[space.slice()] = np.asarray(coeffs).imag
    if derivative:
        return Dx(real_part, 0, 1).eval(points) + 1j * Dx(imag_part, 0, 1).eval(points)
    return real_part.eval(points) + 1j * imag_part.eval(points)


def test_mri_leading_mode_is_magnetically_solenoidal():
    """Direct-component MHD eigenmodes must still satisfy div(B)=0."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=28)
    m, kz = 0, 3.0
    _, V = s.eigs(m, kz, n_return=1)
    n = s.n
    rr = np.linspace(base.R1, base.R2, 200)

    br = _eval_complex_mode(s.SDv, V[4 * n : 5 * n, 0], rr)
    dbr = _eval_complex_mode(s.SDv, V[4 * n : 5 * n, 0], rr, derivative=True)
    bt = _eval_complex_mode(s.Sbt, V[5 * n : 6 * n, 0], rr)
    bz = _eval_complex_mode(s.Sbz, V[6 * n : 7 * n, 0], rr)

    divb = dbr + br / rr + 1j * m * bt / rr + 1j * kz * bz
    bmag = np.sqrt(np.abs(br) ** 2 + np.abs(bt) ** 2 + np.abs(bz) ** 2)
    rel = np.max(np.abs(divb)) / max(np.max(bmag), 1e-30)
    assert rel < 1e-8


def test_critical_rm_uses_fixed_pm_and_lundquist_controls():
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=20)
    calls = []

    def fake_eta(m, kz, Pm=None, S=None, **kw):
        calls.append((m, kz, Pm, S, kw))
        return 0.02 if kz < 3.0 else 0.01

    s.critical_eta_mag_fixed_controls = fake_eta
    res = s.critical_Rm(m=0, kz_list=np.array([2.0, 4.0]), Pm=0.2, S=4.11, iters=3)

    assert res["kz_c"] == 2.0
    assert res["eta_mag_c"] == 0.02
    assert res["Pm"] == 0.2
    assert res["S_c"] == 4.11
    assert math.isclose(res["nu_c"], 0.2 * res["eta_mag_c"])
    assert math.isclose(res["B0_c"], 4.11 * res["eta_mag_c"] / base.gap)
    assert all(
        call[2] == 0.2 and call[3] == 4.11 and call[4]["iters"] == 3 for call in calls
    )


def test_conducting_btheta_bc_is_satisfied():
    """The b_theta Robin basis must satisfy d(r b_theta)/dr = 0 at both walls."""
    from shenfun import Dx, Function

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=24)
    walls = np.array([base.R1, base.R2])
    f = Function(s.Sbt)
    f[:] = 0.0
    f[3] = 1.0
    vb = f.eval(walls)
    db = Dx(f, 0, 1).eval(walls)
    # b + r*b'_phys = d(r b)/dr = 0
    assert abs(vb[0] + base.R1 * db[0]) < 1e-10
    assert abs(vb[1] + base.R2 * db[1]) < 1e-10


@pytest.mark.slow
def test_mri_unstable_with_axial_field():
    """Quasi-Keplerian + axial field at high Rm: MRI-unstable (B0=0 was stable)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=36)
    _, g, _ = s.max_growth_over_kz(0, np.linspace(0.5, 8, 20))
    assert g > 1e-2


# ---------------------------------------------------------------------------
# Insulating (vacuum) magnetic walls -- poloidal flux-function formulation, m=0
# ---------------------------------------------------------------------------
def test_flux_formulation_reproduces_conducting_growth():
    """The m=0 poloidal flux-function MHD operator (b_r=-(ikz/r)chi,
    b_z=(1/r)chi') must reproduce the primitive (b_r,b_theta,b_z) conducting
    eigenvalues -- a correctness check on the whole reformulated operator
    (Lorentz, induction, the Stokes operator for chi, and the chi=0 conducting
    BC that makes b_z'=0 emerge from the chi equation at the wall)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=40)  # conducting
    kz = 6.0
    wp, _ = s.eigs(0, kz, n_return=2)  # primitive 7-field
    L0, Lnu, Leta, M = s._assemble_flux_parts(kz)  # flux, conducting bases
    wf = s._spectrum(L0 + s.nu * Lnu + s.eta_mag * Leta, M)
    assert abs(wf[0].real - wp[0].real) < 1e-6
    assert abs(wf[1].real - wp[1].real) < 1e-6


def test_insulating_eigenmode_is_solenoidal_by_construction():
    """The flux representation gives div(b)=0 identically: with
    b_r=-(ikz/r)chi and b_z=(1/r)chi', (1/r)(r b_r)' + ikz b_z = 0 for any chi.
    Reconstruct the leading insulating eigenmode and confirm to roundoff."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=40, magnetic_bc="insulating"
    )
    kz = 6.0
    Schi, _ = s._flux_bases(kz)
    _, V = s.eigs(0, kz, n_return=1)
    n = s.n
    rr = np.linspace(base.R1, base.R2, 200)
    chi = _eval_complex_mode(Schi, V[4 * n : 5 * n, 0], rr)
    chip = _eval_complex_mode(Schi, V[4 * n : 5 * n, 0], rr, derivative=True)
    br = -(1j * kz / rr) * chi
    dbr = -1j * kz * (chip / rr - chi / rr**2)  # d/dr of -(ikz/r)chi
    bz = chip / rr
    divb = dbr + br / rr + 1j * kz * bz
    bmag = np.sqrt(np.abs(br) ** 2 + np.abs(bz) ** 2)
    assert np.max(np.abs(divb)) / max(np.max(bmag), 1e-30) < 1e-10


def test_insulating_null_keplerian_no_field_is_stable():
    """B0=0 quasi-Keplerian with insulating walls: no MRI, modes decay."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(
        base, B0=0.0, nu=1e-3, eta_mag=1e-3, N=32, magnetic_bc="insulating"
    )
    _, g, _ = s.max_growth_over_kz(0, np.linspace(0.5, 6, 10))
    assert g < 1e-7


def test_insulating_m_neq_0_not_supported():
    """Non-axisymmetric insulating walls couple the poloidal/toroidal scalars at
    the wall and are not yet implemented -- must raise, not mis-solve."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=24, magnetic_bc="insulating"
    )
    with pytest.raises(NotImplementedError):
        s.eigs(1, 3.0)


def test_insulating_kz_zero_raises():
    """The insulating flux bases are kz-dependent (the exterior modified-Bessel
    field has no kz=0 limit here), so kz=0 must raise, not divide by zero."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    s = TaylorCouetteMRI(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=24, magnetic_bc="insulating"
    )
    with pytest.raises(ValueError):
        s._flux_bases(0.0)


@pytest.mark.slow
def test_insulating_mri_unstable_and_easier_than_conducting():
    """Insulating quasi-Keplerian + axial field is MRI-unstable, and its onset is
    easier (lower critical Rm) than conducting walls at the same Pm -- the
    expected ordering (Ruediger et al. 2023: insulating Rm_min=16.5 < conducting
    24.7 as Pm->0)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    ins = TaylorCouetteMRI(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=36, magnetic_bc="insulating"
    )
    _, g, _ = ins.max_growth_over_kz(0, np.linspace(0.5, 8, 16))
    assert g > 1e-2  # MRI-unstable
    kzs = np.linspace(1.0, 8.0, 10) / base.gap
    cond = TaylorCouetteMRI(base, B0=0.1, nu=1e-3, eta_mag=1e-3, N=36)
    rc = cond.critical_Rm(m=0, kz_list=kzs, Pm=0.1, S=4.11, iters=20)
    ri = ins.critical_Rm(m=0, kz_list=kzs, Pm=0.1, S=5.21, iters=20)
    assert rc is not None and ri is not None
    assert ri["Rm_c"] < rc["Rm_c"]  # insulating destabilises more easily
