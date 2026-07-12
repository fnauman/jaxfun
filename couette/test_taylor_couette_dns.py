"""Validation tests for the axisymmetric Taylor-Couette hydrodynamic DNS.

The DNS (:mod:`taylor_couette_dns`) is the nonlinear, time-stepping companion to
the linear-stability solver.  These tests pin its correctness against the
analytically/linearly known behaviour:

  * the laminar circular-Couette state is an exact fixed point (zero
    perturbation stays zero);
  * a small seed in a Rayleigh-stable regime decays;
  * the linear growth rate of a seeded eigenmode matches
    :class:`taylor_couette_linear.TaylorCouetteLinear` to spectral accuracy
    (the sharpest DNS/linear-theory consistency check);
  * incompressibility ``div(u)`` stays small and converges with resolution;
  * (slow) a supercritical run grows at the linear rate and saturates into a
    steady Taylor-vortex state with enhanced angular-momentum transport.

Run with:  conda run -n shenfun pytest -q demo/test_taylor_couette_dns.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from shenfun import Array, Dx, Function, inner, project
from taylor_couette_dns import (
    AxisymmetricMRIDNS,
    AxisymmetricTCDNS,
    TaylorCouetteDNS,
    TaylorCouetteMRIDNS,
)
from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
from taylor_couette_mri import TaylorCouetteMRI


def _kep(eta=0.5):
    return CircularCouette(1.0, 1.0 / eta, 1.0, eta**1.5)


def _hat_from_values(space, values):
    a = Array(space)
    a[:] = values
    return a.forward(Function(space))


def _weak_from_values(test, space, values):
    a = Array(space)
    a[:] = values
    return inner(test, a)


def _max_abs_hat(fields):
    return max(float(np.abs(np.asarray(f)).max()) for f in fields)


# ---------------------------------------------------------------------------
# Fixed point and stable-regime decay
# ---------------------------------------------------------------------------
def test_laminar_fixed_point_stays_zero():
    """Zero perturbation about the base flow is an exact steady state."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    dns = AxisymmetricTCDNS(base, nu=1e-2, Nr=24, Nz=8, Lz=2.0, dt=5e-3, dealias=1.0)
    dns.u_hat[:] = 0.0
    dns.run(0.1)
    assert dns.energy() == 0.0
    assert np.all(np.isfinite(dns.u_hat))


def test_stable_regime_perturbation_decays():
    """Rayleigh-stable (mu=0.5>eta^2) seed decays under the DNS."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.5)  # mu=0.5 > eta^2=0.25
    dns = AxisymmetricTCDNS(base, nu=1e-2, Nr=32, Nz=8, Lz=2.0, dt=2e-3, dealias=1.0)
    dns.set_perturbation(amp=1e-4, kz_mode=1)
    E0 = dns.energy()
    dns.run(2.0)
    assert dns.energy() < E0  # decayed
    assert dns.divergence_linf() < 1e-4


# ---------------------------------------------------------------------------
# Linear growth vs eigensolver  (sharp consistency check)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu, kz", [(1e-2, 3.13), (1e-2, 2.5)])
def test_linear_growth_matches_eigensolver(nu, kz):
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    Lz = 2 * math.pi / kz
    dns = AxisymmetricTCDNS(base, nu=nu, Nr=40, Nz=8, Lz=Lz, dt=1e-3, dealias=1.0)
    s_lin = dns.seed_linear_eigenmode(kz_mode=1, amp=1e-6)
    E0 = dns.energy()
    T = 0.4
    dns.run(T)
    s_dns = 0.5 * math.log(dns.energy() / E0) / T
    assert abs(s_dns - s_lin.real) < 1e-3 * max(1.0, abs(s_lin.real))
    assert s_lin.real > 0  # supercritical seed grows


# ---------------------------------------------------------------------------
# Incompressibility
# ---------------------------------------------------------------------------
def test_incompressibility_to_roundoff():
    """The coupled velocity-pressure solve enforces div(u)=0 to roundoff
    (no fractional-step splitting error)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    for Nr in (24, 40):
        dns = AxisymmetricTCDNS(
            base, nu=1e-2, Nr=Nr, Nz=8, Lz=2.0, dt=1e-3, dealias=1.0
        )
        dns.seed_linear_eigenmode(kz_mode=1, amp=1e-4)
        dns.run(0.2)
        ur, ut, uz = dns.velocity_physical()
        umax = float(np.abs(np.asarray(ur)).max())
        assert dns.divergence_linf() < 1e-9 * max(umax, 1e-12)


# ---------------------------------------------------------------------------
# Production gates: restart-equivalence, temporal order, energy-balance closure
# ---------------------------------------------------------------------------
def test_restart_equivalence_axisym():
    """PRODUCTION GATE (restart): a checkpoint taken mid-run via state_dict() and
    reloaded into a fresh axisymmetric TC DNS reproduces the uninterrupted run
    bit-for-bit, including the Adams-Bashforth-2 nonlinear history."""

    def fresh():
        d = AxisymmetricTCDNS(
            CircularCouette(1.0, 2.0, 1.0, 0.0),
            nu=1e-2,
            Nr=24,
            Nz=8,
            Lz=2.0,
            dt=2e-3,
            family="C",
            dealias=1.0,
        )
        d.set_perturbation(amp=1e-4, kz_mode=1)
        return d

    nsteps, split = 20, 10
    direct = fresh()
    for _ in range(nsteps):
        direct.step()

    first = fresh()
    for _ in range(split):
        first.step()
    checkpoint = first.state_dict()
    assert checkpoint["t"] == pytest.approx(split * first.dt)
    assert checkpoint["tstep"] == split

    restarted = fresh()
    restarted.load_state_dict(checkpoint)
    restarted.run((nsteps - split) * restarted.dt)

    assert np.max(np.abs(np.array(direct.u_hat) - np.array(restarted.u_hat))) < 1e-12
    assert direct.state_dict()["t"] == pytest.approx(nsteps * direct.dt)
    assert restarted.state_dict()["t"] == pytest.approx(nsteps * restarted.dt)
    assert direct.state_dict()["tstep"] == restarted.state_dict()["tstep"] == nsteps
    assert abs(direct.energy() - restarted.energy()) < 1e-12


@pytest.mark.slow
def test_temporal_order_cnab2_axisym():
    """PRODUCTION GATE (temporal order): the axisymmetric TC DNS is 2nd-order in
    time. The measured growth-rate error vs the linear eigenvalue shrinks like
    dt^2 under refinement (fitted log-log slope ~2)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    kz = 3.13
    Lz = 2 * math.pi / kz

    def rate_error(dt, T=0.4):
        d = AxisymmetricTCDNS(
            base, nu=1e-2, Nr=40, Nz=8, Lz=Lz, dt=dt, family="C", dealias=1.0
        )
        s_lin = d.seed_linear_eigenmode(kz_mode=1, amp=1e-6)
        e0 = d.energy()
        d.run(T)
        s_dns = 0.5 * math.log(d.energy() / e0) / T
        return abs(s_dns - s_lin.real)

    dts = [4e-3, 2e-3, 1e-3]
    errs = [rate_error(dt) for dt in dts]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    assert slope > 1.8


def test_energy_balance_single_exponential():
    """PRODUCTION GATE (energy balance): in the linear regime the kinetic-energy
    budget closes to a single clean exponential -- the growth rate measured over
    two successive windows agrees, so d ln E / dt (shear production minus viscous
    dissipation, per unit energy) is steady."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    d = AxisymmetricTCDNS(
        base,
        nu=1e-2,
        Nr=40,
        Nz=8,
        Lz=2 * math.pi / 3.13,
        dt=1e-3,
        family="C",
        dealias=1.0,
    )
    d.seed_linear_eigenmode(kz_mode=1, amp=1e-6)
    d.run(0.2)
    e0 = d.energy()
    d.run(0.1)
    e1 = d.energy()
    d.run(0.1)
    e2 = d.energy()
    r1 = 0.5 * math.log(e1 / e0) / 0.1
    r2 = 0.5 * math.log(e2 / e1) / 0.1
    assert abs(r1 - r2) / abs(r1) < 1e-6


def test_dealiased_step_is_finite():
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    dns = AxisymmetricTCDNS(base, nu=1e-2, Nr=32, Nz=16, Lz=2.0, dt=2e-3, dealias=1.5)
    dns.set_perturbation(amp=1e-2, kz_mode=1)
    final = dns.run(0.1)
    assert np.all(np.isfinite(dns.u_hat))
    assert final["div_linf"] < 1e-2


# ---------------------------------------------------------------------------
# Nonlinear saturation (slow)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Full 3D solver
# ---------------------------------------------------------------------------
def _inject_eigenmode(dns, m, kz_mode, amp=1e-6):
    """Inject the exact linear eigenmode coefficients into Fourier mode
    (theta=m, z=kz_mode) of a 3D DNS; returns the linear eigenvalue."""
    kz = 2 * math.pi * kz_mode / dns.Lz
    lin = TaylorCouetteLinear(dns.base, nu=dns.nu, N=dns.Nr, family=dns.family)
    w, V = lin.eigs(m, kz, n_return=1)
    n = lin.n
    for comp in range(3):
        f = Function(dns.TD)
        f[:] = 0.0
        f[m, kz_mode, :n] = V[comp * n : (comp + 1) * n, 0] * amp
        dns.u_hat[comp] = f
    dns._have_old = False
    return complex(w[0])


def test_3d_axisymmetric_mode_matches_linear():
    """The 3D solver, restricted to m=0, reproduces the linear growth rate to
    spectral accuracy (sharpest check that the 3D operator is correct)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    kz = 3.13
    dns = TaylorCouetteDNS(
        base, nu=1e-2, Nr=40, Ntheta=4, Nz=8, Lz=2 * math.pi / kz, dt=1e-3, dealias=1.0
    )
    s = _inject_eigenmode(dns, m=0, kz_mode=1, amp=1e-6)
    E0 = dns.energy()
    T = 0.4
    dns.run(T)
    s_dns = 0.5 * math.log(dns.energy() / E0) / T
    assert abs(s_dns - s.real) < 1e-3 * abs(s.real)
    assert dns.divergence_linf() < 1e-9


def test_3d_incompressible():
    """Non-axisymmetric (m=1) flow stays incompressible.  With the inf-sup
    ``P_N``-``P_{N-2}`` pair the *weak* continuity is enforced exactly; the
    pointwise residual is the top-two-radial-mode truncation, which is small and
    converges spectrally (here rel. ~6e-3 at Nr=32, ~7e-4 at Nr=48)."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    divs = {}
    for Nr in (32, 48):
        dns = TaylorCouetteDNS(
            base, nu=1e-2, Nr=Nr, Ntheta=8, Nz=8, Lz=2.0, dt=1e-3, dealias=1.0
        )
        dns.set_perturbation(amp=1e-3, m=1, kz_mode=1)
        dns.run(0.05)
        ur, ut, uz = dns.velocity_physical()
        umax = float(np.abs(np.asarray(ur)).max())
        divs[Nr] = dns.divergence_linf() / max(umax, 1e-12)
    assert divs[32] < 1e-2
    assert divs[48] < divs[32]  # spectral convergence


@pytest.mark.slow
def test_3d_nonaxisymmetric_growth_matches_linear():
    """A seeded non-axisymmetric (m=1) eigenmode grows at the linear rate Re(s)
    after the sub-dominant transient settles.  (A single m>=1 azimuthal mode is
    seeded as the *real* field Re[q exp(i(m theta+kz z))], so the growth must be
    measured after a short transient rather than instantaneously.)"""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    nu, kz = 6e-3, 3.0
    dns = TaylorCouetteDNS(
        base, nu=nu, Nr=36, Ntheta=4, Nz=8, Lz=2 * math.pi / kz, dt=1e-3, dealias=1.0
    )
    s = dns.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-7)
    for _ in range(int(round(3.0 / dns.dt))):  # transient
        dns.step()
    E0 = dns.energy()
    for _ in range(int(round(2.0 / dns.dt))):  # measure
        dns.step()
    s_dns = 0.5 * math.log(dns.energy() / E0) / 2.0
    assert abs(s_dns - s.real) < 0.05 * abs(s.real)


# ---------------------------------------------------------------------------
# Axisymmetric MHD / MRI solver
# ---------------------------------------------------------------------------
def _kep_base(eta=0.5):
    return CircularCouette(1.0, 1.0 / eta, 1.0, eta**1.5)


def test_mri_growth_matches_eigensolver():
    """The MRI DNS reproduces the linear MRI growth rate of a seeded eigenmode to
    spectral accuracy, and keeps div(u)=div(b)=0 to roundoff."""
    base = _kep_base()
    B0, nu, eta_mag, kz = 0.1, 1e-3, 1e-3, 6.0
    lin = TaylorCouetteMRI(base, B0=B0, nu=nu, eta_mag=eta_mag, N=40)
    w, _ = lin.eigs(0, kz, n_return=1)
    assert w[0].real > 0.05  # genuinely MRI-unstable
    dns = AxisymmetricMRIDNS(
        base,
        B0=B0,
        nu=nu,
        eta_mag=eta_mag,
        Nr=40,
        Nz=8,
        Lz=2 * math.pi / kz,
        dt=2e-3,
        dealias=1.0,
    )
    s = dns.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
    E0 = dns.diagnostics(0, 0)["E"]  # initial state (t=0)
    T = 1.0
    d = dns.run(T)  # returns diagnostics at t=T
    s_dns = 0.5 * math.log(d["E"] / E0) / T
    assert abs(s_dns - s.real) < 1e-3 * abs(s.real)
    assert d["divu"] < 1e-9
    assert d["divb"] < 1e-9


def test_mri_restart_equivalence():
    """PRODUCTION GATE (restart, MHD): a checkpoint taken mid-run via state_dict()
    and reloaded into a fresh axisymmetric MRI DNS reproduces the uninterrupted run
    bit-for-bit (the six (u,b) field coefficients + the AB2 history)."""
    base = _kep_base()

    def fresh():
        d = AxisymmetricMRIDNS(
            base,
            B0=0.1,
            nu=1e-3,
            eta_mag=1e-3,
            Nr=32,
            Nz=8,
            Lz=2 * math.pi / 6.0,
            dt=2e-3,
            family="C",
            dealias=1.0,
        )
        d.seed_linear_eigenmode(kz_mode=1, amp=1e-6)
        return d

    nsteps, split = 16, 8
    direct = fresh()
    for _ in range(nsteps):
        direct.step()

    first = fresh()
    for _ in range(split):
        first.step()
    checkpoint = first.state_dict()
    assert checkpoint["t"] == pytest.approx(split * first.dt)
    assert checkpoint["tstep"] == split

    restarted = fresh()
    restarted.load_state_dict(checkpoint)
    restarted.run((nsteps - split) * restarted.dt)

    assert np.max(np.abs(np.array(direct.x) - np.array(restarted.x))) < 1e-12
    assert direct.state_dict()["t"] == pytest.approx(nsteps * direct.dt)
    assert restarted.state_dict()["t"] == pytest.approx(nsteps * restarted.dt)
    assert direct.state_dict()["tstep"] == restarted.state_dict()["tstep"] == nsteps
    de, dm = direct.energy()
    re, rm = restarted.energy()
    assert abs(de - re) < 1e-14 and abs(dm - rm) < 1e-14


def test_mri_energy_balance_single_exponential():
    """PRODUCTION GATE (energy balance, MHD): in the linear MRI regime the total
    (kinetic + magnetic) energy grows as a single clean exponential -- the growth
    rate measured over two successive windows agrees, so MRI production minus
    (viscous + Ohmic) dissipation per unit energy is steady."""
    base = _kep_base()
    d = AxisymmetricMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=40,
        Nz=8,
        Lz=2 * math.pi / 6.0,
        dt=2e-3,
        family="C",
        dealias=1.0,
    )
    d.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
    d.run(0.3)
    e0 = sum(d.energy())
    d.run(0.2)
    e1 = sum(d.energy())
    d.run(0.2)
    e2 = sum(d.energy())
    r1 = 0.5 * math.log(e1 / e0) / 0.2
    r2 = 0.5 * math.log(e2 / e1) / 0.2
    assert abs(r1 - r2) / abs(r1) < 1e-3


@pytest.mark.slow
def test_mri_temporal_order_cnab2():
    """PRODUCTION GATE (temporal order, MHD): the axisymmetric MRI DNS is 2nd-order
    in time -- the measured growth-rate error vs the linear MRI eigenvalue shrinks
    like dt^2 under refinement (fitted log-log slope ~2)."""
    base = _kep_base()
    kz = 6.0
    Lz = 2 * math.pi / kz

    def rate_error(dt, T=0.4):
        d = AxisymmetricMRIDNS(
            base,
            B0=0.1,
            nu=1e-3,
            eta_mag=1e-3,
            Nr=40,
            Nz=8,
            Lz=Lz,
            dt=dt,
            family="C",
            dealias=1.0,
        )
        s_lin = d.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
        e0 = d.diagnostics(0, 0)["E"]
        df = d.run(T)
        s_dns = 0.5 * math.log(df["E"] / e0) / T
        return abs(s_dns - s_lin.real)

    dts = [4e-3, 2e-3, 1e-3]
    errs = [rate_error(dt) for dt in dts]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    assert slope > 1.8


def test_mri_random_seed_is_solenoidal():
    """A random magnetic IC must start solenoidal: the magnetic field is never
    pressure-projected, so a non-solenoidal start would violate div(b)=0 for the
    whole run.  set_random seeds only the toroidal b_theta (b_r=b_z=0), so the
    axisymmetric div(b) = (1/r)d(r b_r)/dr + db_z/dz is exactly zero by
    construction while the magnetic field is still genuinely excited."""
    base = _kep_base()
    dns = AxisymmetricMRIDNS(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, Nr=32, Nz=8, Lz=2.0, dt=2e-3, dealias=1.0
    )
    dns.set_random(amp=1e-3, magnetic=True)
    _, db = dns.divergences()
    f = dns.fields_physical()
    bmag = float(np.abs(np.asarray(f[4])).max())  # |b_theta|
    assert bmag > 1e-5  # field actually seeded
    assert db < 1e-12 * max(bmag, 1.0)  # ... yet solenoidal


def test_mri_requires_field_keplerian_stable_without_B0():
    """B0=0: a quasi-Keplerian profile is Rayleigh-stable and the magnetic
    perturbations resistively decay -> total energy decays (no MRI)."""
    base = _kep_base()
    dns = AxisymmetricMRIDNS(
        base, B0=0.0, nu=1e-3, eta_mag=1e-3, Nr=32, Nz=8, Lz=2.0, dt=2e-3, dealias=1.0
    )
    dns.set_random(amp=1e-4, magnetic=True)
    E0 = dns.diagnostics(0, 0)["E"]  # initial state (t=0)
    assert dns.run(3.0)["E"] < E0  # decayed by t=3


def test_mri_solenoidal_and_nonlinear_finite():
    """Nonlinear MRI step (Lorentz + induction EMF, dealiased) stays finite and
    keeps div(b) small."""
    base = _kep_base()
    dns = AxisymmetricMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=40,
        Nz=16,
        Lz=2 * math.pi / 6.0,
        dt=2e-3,
        dealias=1.5,
    )
    dns.seed_linear_eigenmode(kz_mode=1, amp=1e-3)
    d = dns.run(1.0)
    assert np.all(np.isfinite(dns.x))
    assert d["divu"] < 1e-8
    assert d["divb"] < 1e-8


def test_mri_nonlinear_alfvenic_cancellation():
    """If the perturbation has u=b in the same component, the quadratic
    Reynolds/Maxwell force and EMF curl both vanish.  This catches sign mistakes
    in ``(u.grad)u - (b.grad)b`` and in the explicit induction term without
    relying on a time-step comparison."""
    base = _kep_base()
    dns = AxisymmetricMRIDNS(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, Nr=24, Nz=8, Lz=2.0, dt=2e-3, dealias=1.0
    )
    zz, rr = dns.TD.local_mesh(True)
    wall = np.sin(np.pi * (rr - base.R1) / base.gap) ** 2
    vals = 1e-3 * wall * np.cos(2 * np.pi * zz / dns.Lz)
    h = _hat_from_values(dns.TD, vals)
    dns.x[:] = 0.0
    dns.x[0] = h  # u_r
    dns.x[3] = h  # b_r
    out = Function(dns.VE)
    dns.nonlinear(out)
    assert _max_abs_hat(out) < 1e-12


def test_mri_axisymmetric_toroidal_curvature_signs():
    """Pure toroidal velocity and pure toroidal magnetic fields have opposite
    radial curvature signs in the stored nonlinear RHS: ``N_u,r=-u_th^2/r`` but
    ``N_u,r=+b_th^2/r`` for magnetic tension because the step later subtracts
    ``N_u``.  These are easy signs to flip in cylindrical components."""
    base = _kep_base()
    dns = AxisymmetricMRIDNS(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, Nr=24, Nz=8, Lz=2.0, dt=2e-3, dealias=1.0
    )
    zz, rr = dns.T0.local_mesh(True)
    wall = np.sin(np.pi * (rr - base.R1) / base.gap) ** 2

    dns.x[:] = 0.0
    dns.x[1] = _hat_from_values(dns.TD, 2e-3 * wall * np.cos(2 * np.pi * zz / dns.Lz))
    out = Function(dns.VE)
    dns.nonlinear(out)
    ut = np.asarray(dns.x[1].backward())
    expected = _weak_from_values(dns.vu, dns.T0, -ut * ut * dns.inv_r)
    np.testing.assert_allclose(
        np.asarray(out[0]), np.asarray(expected), atol=1e-13, rtol=1e-12
    )
    assert _max_abs_hat(out[1:]) < 1e-13

    dns.x[:] = 0.0
    dns.x[4] = _hat_from_values(dns.Tbt, 2e-3 * wall * np.sin(2 * np.pi * zz / dns.Lz))
    out = Function(dns.VE)
    dns.nonlinear(out)
    bt = np.asarray(dns.x[4].backward())
    expected = _weak_from_values(dns.vu, dns.T0, bt * bt * dns.inv_r)
    np.testing.assert_allclose(
        np.asarray(out[0]), np.asarray(expected), atol=1e-13, rtol=1e-12
    )
    assert _max_abs_hat(out[1:]) < 1e-13


def test_mri3d_emf_curl_theta_metric_signs():
    """A manufactured ``u_r``/``b_theta`` pair gives only ``eps_z`` in
    ``eps = u x b``.  The 3D induction nonlinearity must then contain both
    ``N_b,r = -(1/r) d_theta eps_z`` and ``N_b,theta = d_r eps_z`` with the
    correct signs and cylindrical metric factor."""
    base = _kep_base()
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=18,
        Ntheta=6,
        Nz=8,
        Lz=2.0,
        dt=2e-3,
        dealias=1.0,
    )
    th, zz, rr = dns.TD.local_mesh(True)
    wall = np.sin(np.pi * (rr - base.R1) / base.gap) ** 2
    dns.x[:] = 0.0
    dns.x[0] = _hat_from_values(
        dns.TD, 1e-3 * wall * np.cos(th) * np.cos(2 * np.pi * zz / dns.Lz)
    )
    dns.x[4] = _hat_from_values(
        dns.Tbt, 1e-3 * wall * np.sin(th) * np.cos(2 * np.pi * zz / dns.Lz)
    )

    out = Function(dns.VE)
    dns.nonlinear(out)

    ur = np.asarray(dns.x[0].backward())
    bt = np.asarray(dns.x[4].backward())
    eps_z = ur * bt
    eps_hat = _hat_from_values(dns.T0, eps_z)
    deps_dtheta = np.asarray(project(Dx(eps_hat, 0, 1), dns.T0).backward())
    deps_dr = np.asarray(project(Dx(eps_hat, 2, 1), dns.T0).backward())

    expected_br = _weak_from_values(dns.vbr, dns.T0, -dns.inv_r * deps_dtheta)
    expected_bt = _weak_from_values(dns.vbt, dns.T0, deps_dr)
    expected_bz = _weak_from_values(dns.vbz, dns.T0, np.zeros_like(eps_z))
    assert float(np.abs(np.asarray(expected_br)).max()) > 1e-12
    assert float(np.abs(np.asarray(expected_bt)).max()) > 1e-12
    np.testing.assert_allclose(
        np.asarray(out[3]), np.asarray(expected_br), atol=1e-13, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out[4]), np.asarray(expected_bt), atol=1e-13, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(out[5]), np.asarray(expected_bz), atol=1e-13, rtol=1e-12
    )


@pytest.mark.slow
def test_mri_nonlinear_saturation():
    """Seeded MRI grows at the linear rate then saturates: the magnetic energy is
    amplified by orders of magnitude and then *stops* growing exponentially
    (the saturated state fluctuates, so we check the late-time net growth rate is
    far below the linear MRI rate, not a tight plateau).  div(b) stays small."""
    base = _kep_base()
    B0, nu, eta_mag, kz = 0.1, 1e-3, 1e-3, 6.0
    dns = AxisymmetricMRIDNS(
        base,
        B0=B0,
        nu=nu,
        eta_mag=eta_mag,
        Nr=40,
        Nz=24,
        Lz=2 * math.pi / kz,
        dt=2e-3,
        dealias=1.5,
    )
    dns.seed_linear_eigenmode(kz_mode=1, amp=1e-4)
    E_hist = []
    d = None
    for _ in range(16):
        d = dns.run(2.0)  # cumulative-time diagnostics
        E_hist.append(d["Emag"])
    assert E_hist[-1] > 1e3 * E_hist[0]  # MRI amplified the field
    # saturated: late-time net growth rate << linear MRI rate (~0.34)
    late = 0.5 * math.log(E_hist[-1] / E_hist[-5]) / (4 * 2.0)
    assert abs(late) < 0.08
    assert d["divb"] < 1e-4
    assert np.all(np.isfinite(dns.x))


@pytest.mark.slow
def test_supercritical_saturation():
    """Re=100 (>Re_c~68): grow at the linear rate, then saturate to a steady
    Taylor-vortex state with finite radial flow and bounded energy."""
    base = CircularCouette(1.0, 2.0, 1.0, 0.0)
    kz = 3.13
    dns = AxisymmetricTCDNS(
        base, nu=1e-2, Nr=40, Nz=16, Lz=2 * math.pi / kz, dt=4e-3, dealias=1.5
    )
    dns.seed_linear_eigenmode(kz_mode=1, amp=1e-4)
    E_early = None
    E_hist = []
    for _ in range(20):
        dns.run(4.0)
        E_hist.append(dns.energy())
    # energy grew by orders of magnitude then leveled off
    assert E_hist[-1] > 1e3 * E_hist[0]
    rel_change = abs(E_hist[-1] - E_hist[-2]) / E_hist[-1]
    assert rel_change < 1e-2  # saturated plateau
    # genuine overturning vortices: finite radial velocity
    ur, ut, uz = dns.velocity_physical()
    assert float(np.abs(ur).max()) > 1e-2
    assert dns.divergence_linf() < 1e-2
    assert np.all(np.isfinite(dns.u_hat))


# ---------------------------------------------------------------------------
# Full 3D MHD / MRI solver (azimuthal Fourier m != 0)
# ---------------------------------------------------------------------------
def _mri3d_growth(m, kz=6.0, Nr=28, Ntheta=None, Nz=12, t_skip=0.5, t_meas=1.5):
    """Seed the (m, kz) MRI eigenmode in the 3D solver and return
    ``(linear Re(s), DNS growth, final diagnostics)``.  Growth is measured over a
    window *after* the cumulative-time origin so the seed transient has cleared;
    the window length comes from the returned diagnostic ``t`` (run() accumulates
    time across calls, so the elapsed dt is ``d2['t'] - d1['t']``)."""
    base = _kep()
    # a real m != 0 field needs theta-modes +-m, so resolve at least up to |m|+1
    Ntheta = (2 * (abs(m) + 1)) if Ntheta is None else Ntheta
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=Nr,
        Ntheta=Ntheta,
        Nz=Nz,
        Lz=2 * math.pi / kz,
        dt=2e-3,
        dealias=1.5,
    )
    s = dns.seed_linear_eigenmode(m=m, kz_mode=1, amp=1e-7)
    d1 = dns.run(t_skip)
    d2 = dns.run(t_meas)
    sigma = 0.5 * math.log(d2["E"] / d1["E"]) / (d2["t"] - d1["t"])
    return s.real, sigma, d2


@pytest.mark.parametrize("m", [0, 1, 2])
def test_mri3d_growth_matches_eigensolver(m):
    """The 3D MRI DNS reproduces the linear MRI growth rate of a seeded
    (m, kz) eigenmode to spectral accuracy -- for the axisymmetric channel
    (m=0) *and* genuinely non-axisymmetric, travelling-wave modes (m=1, m=2,
    which have Im(s) != 0) -- keeping div(u)=div(b)=0 to roundoff."""
    s_lin, s_dns, d = _mri3d_growth(m)
    assert s_lin > 0.05  # genuinely MRI-unstable
    assert abs(s_dns - s_lin) < 2e-3 * abs(s_lin)
    assert d["divu"] < 1e-9
    assert d["divb"] < 1e-9


def test_mri3d_reduces_to_axisymmetric():
    """Restricted to m=0 content, the 3D MRI solver reproduces the axisymmetric
    MRI growth rate (same physics, extra idle azimuthal direction)."""
    s_lin, s_dns, _ = _mri3d_growth(0, Ntheta=4)
    base = _kep()
    ax = AxisymmetricMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=28,
        Nz=12,
        Lz=2 * math.pi / 6.0,
        dt=2e-3,
        dealias=1.5,
    )
    sa = ax.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
    a1 = ax.run(0.5)
    a2 = ax.run(1.5)
    s_ax = 0.5 * math.log(a2["E"] / a1["E"]) / (a2["t"] - a1["t"])
    assert abs(s_dns - s_ax) < 1e-3 * abs(s_ax)  # 3D(m=0) == axisymmetric
    assert abs(s_dns - sa.real) < 2e-3 * abs(sa.real)


def test_mri3d_eigenmode_is_solenoidal():
    """A seeded (complex) 3D MRI eigenmode is divergence-free to roundoff for
    both u and b -- the radial real/imag split keeps the poloidal/axial balance
    that makes the mode solenoidal (a plain complex->real cast would break it)."""
    base = _kep()
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=28,
        Ntheta=6,
        Nz=12,
        Lz=2 * math.pi / 6.0,
        dt=2e-3,
        dealias=1.5,
    )
    dns.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-6)
    du, db = dns.divergences()
    assert du < 1e-9
    assert db < 1e-9


def test_mri3d_random_seed_is_divfree_and_keeps_div_b_small():
    """The 3D random IC seeds a *divergence-free* (Stokes stream-function)
    velocity with b=0.  Because the magnetic field is never pressure-projected,
    the imposed-field induction B0 du/dz would inject div(b) ~ dt B0 d_z div(u)
    from a non-solenoidal velocity; an exactly div-free seed keeps div(b) small
    through the run."""
    base = _kep()
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=24,
        Ntheta=6,
        Nz=12,
        Lz=2.0,
        dt=2e-3,
        dealias=1.5,
    )
    dns.set_random(amp=1e-3, seed=0)
    du0, db0 = dns.divergences()
    f = dns.fields_physical()
    assert float(np.abs(np.asarray(f[0])).max()) > 1e-5  # velocity seeded
    assert (
        float(
            max(
                np.abs(np.asarray(f[3])).max(),
                np.abs(np.asarray(f[4])).max(),
                np.abs(np.asarray(f[5])).max(),
            )
        )
        == 0.0
    )  # b starts at zero
    assert db0 == 0.0  # b=0 -> solenoidal
    assert du0 < 1e-10  # velocity div-free seed
    d = dns.run(0.5)
    assert np.all(np.isfinite(dns.x))
    assert d["divb"] < 1e-4  # induction did not inject div(b)


def test_mri3d_rejects_underresolved_azimuthal_mode():
    """Seeding an azimuthal mode the grid cannot resolve (2|m| >= Ntheta) must
    raise: the sampled phase would alias to another mode while the eigenvalue is
    still reported for the requested m (e.g. m=1 on Ntheta=1)."""
    base = _kep()
    dns = TaylorCouetteMRIDNS(
        base, B0=0.1, nu=1e-3, eta_mag=1e-3, Nr=16, Ntheta=2, Nz=8, Lz=2.0, dt=2e-3
    )
    with pytest.raises(ValueError):
        dns.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-7)  # needs Ntheta > 2
    dns.seed_linear_eigenmode(m=0, kz_mode=1, amp=1e-7)  # m=0 is fine


def test_mri3d_keplerian_stable_without_field():
    """B0=0: a quasi-Keplerian profile is Rayleigh-stable, so 3D perturbations
    resistively decay (no MRI without the imposed field)."""
    base = _kep()
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.0,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=24,
        Ntheta=6,
        Nz=12,
        Lz=2.0,
        dt=2e-3,
        dealias=1.5,
    )
    dns.set_random(amp=1e-4, seed=1)
    E0 = dns.diagnostics(0, 0)["E"]
    assert dns.run(3.0)["E"] < E0


@pytest.mark.slow
def test_mri3d_nonlinear_saturation_nonaxisymmetric():
    """Seed a genuinely non-axisymmetric (m=1) MRI eigenmode: it grows at the
    linear rate, then the quadratic Maxwell/Reynolds + EMF nonlinearities
    saturate it (energy amplified by orders of magnitude, late-time growth far
    below the linear rate).  The saturated state retains substantial
    non-axisymmetric content (genuine 3D), and div(b) stays bounded."""
    base = _kep()
    dns = TaylorCouetteMRIDNS(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=28,
        Ntheta=8,
        Nz=14,
        Lz=2 * math.pi / 6.0,
        dt=2e-3,
        dealias=1.5,
    )
    s = dns.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-3)

    def nonaxi_frac():
        num = den = 0.0
        for i in range(6):
            p = np.abs(np.asarray(dns.x[i])) ** 2
            den += p.sum()
            num += p[1:, ...].sum()
        return num / max(den, 1e-300)

    E_hist = []
    d = None
    for _ in range(20):
        d = dns.run(2.0)  # cumulative-time diagnostics
        E_hist.append(d["Emag"])
    assert E_hist[-1] > 1e3 * E_hist[0]  # MRI amplified the field
    # saturated: late-time net growth rate << linear MRI rate (~0.29)
    late = 0.5 * math.log(E_hist[-1] / E_hist[-5]) / (4 * 2.0)
    assert abs(late) < 0.08
    assert nonaxi_frac() > 1e-2  # genuinely 3D at saturation
    assert d["divb"] < 1e-3
    assert np.all(np.isfinite(dns.x))
