"""Validation tests for the primitive-variable plane-Couette MRI DNS
(pcf_mri_primitive.AxisymmetricPCFMRIDNS).

These mirror the Taylor-Couette MRI DNS gates. Because this solver evolves the
magnetic field DIRECTLY (primitive variables), the linear MRI eigenmode is
injected by a direct block-copy of the eigenvector (no vector-potential gauge
inversion), so the DNS growth rate matches the linear eigenvalue to spectral
accuracy -- the quantitative growth-vs-linear comparison the vector-potential
solver (pcf_mhd_mri_shearpy.py) could not provide cleanly.

Run (cap CPUs):
    OMP_NUM_THREADS=4 python -m pytest test_pcf_mri_primitive.py -v
"""

import math

import numpy as np
import pytest
from pcf_mri_primitive import PCFMRIDNS, AxisymmetricPCFMRIDNS


def _mk(dt=2e-3, Nx=40, Nz=16):
    return AxisymmetricPCFMRIDNS(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nx=Nx,
        Nz=Nz,
        Lz=1.0,
        dt=dt,
        family="C",
        dealias=1.0,
    )


def test_seed_eigenmode_is_solenoidal():
    """Injecting the linear MRI eigenmode (direct block-copy of u and b) yields a
    velocity AND magnetic field that are divergence-free to roundoff."""
    d = _mk()
    s = d.seed_linear_eigenmode(kz_mode=1, amp=1e-6)
    assert s.real > 0.05  # genuinely MRI-unstable
    diag = d.diagnostics(0.0, 0)
    assert diag["divu"] < 1e-12
    assert diag["divb"] < 1e-12


def test_mri_growth_matches_linear():
    """PRODUCTION GATE (growth-vs-linear): seed the linear MRI eigenmode and
    integrate; the measured DNS growth rate matches the linear eigenvalue to
    spectral accuracy, and div(u)=div(b)=0 throughout."""
    d = _mk()
    s_lin = d.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
    e0 = sum(d.energy())
    T = 0.6
    df = d.run(T)
    s_dns = 0.5 * math.log(df["E"] / e0) / T
    assert abs(s_dns - s_lin.real) < 1e-4 * max(1.0, abs(s_lin.real))
    assert df["divu"] < 1e-9
    assert df["divb"] < 1e-9


def test_restart_equivalence():
    """PRODUCTION GATE (restart): a checkpoint taken mid-run via state_dict() and
    reloaded into a fresh solver reproduces the uninterrupted run bit-for-bit
    (the six (u,b) field coefficients + the AB2 history)."""

    def fresh():
        d = _mk()
        d.set_perturbation(amp=1e-3, kz_mode=1)
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


def test_energy_balance_single_exponential():
    """PRODUCTION GATE (energy balance): in the linear MRI regime the total
    (kinetic + magnetic) energy grows as a single clean exponential -- the growth
    rate measured over two successive windows agrees, so MRI production minus
    (viscous + Ohmic) dissipation per unit energy is steady."""
    d = _mk()
    d.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
    d.run(0.3)
    e0 = sum(d.energy())
    d.run(0.2)
    e1 = sum(d.energy())
    d.run(0.2)
    e2 = sum(d.energy())
    r1 = 0.5 * math.log(e1 / e0) / 0.2
    r2 = 0.5 * math.log(e2 / e1) / 0.2
    assert abs(r1 - r2) / abs(r1) < 1e-6


@pytest.mark.slow
def test_temporal_order_cnab2():
    """PRODUCTION GATE (temporal order): the primitive MRI DNS is 2nd-order in
    time -- the measured growth-rate error vs the linear MRI eigenvalue shrinks
    like dt^2 under refinement (fitted log-log slope ~2)."""

    def rate_error(dt, T=0.4):
        d = _mk(dt=dt)
        s_lin = d.seed_linear_eigenmode(kz_mode=1, amp=1e-7)
        e0 = sum(d.energy())
        d.run(T)
        s_dns = 0.5 * math.log(sum(d.energy()) / e0) / T
        return abs(s_dns - s_lin.real)

    dts = [4e-3, 2e-3, 1e-3]
    errs = [rate_error(dt) for dt in dts]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    assert slope > 1.8


def test_divergence_free_over_time():
    """PRODUCTION GATE (divergence over time): a solenoidal (eigenmode) start stays
    divergence-free in BOTH u and b across the whole integration -- the coupled
    saddle point holds div(u)=0 to roundoff and the induction equation preserves
    div(b)=0 (driven only by the roundoff-level div(u))."""
    d = _mk()
    d.seed_linear_eigenmode(kz_mode=1, amp=1e-4)
    df = d.run(0.3)
    assert df["divu"] < 1e-9
    assert df["divb"] < 1e-9
    assert np.all(np.isfinite(np.array(d.x)))


def test_perturbation_divergence_stays_controlled():
    """A NON-solenoidal channel-mode perturbation is projected and stays
    divergence-controlled (the saddle point removes the bulk div(u); the residual
    that feeds div(b) through the imposed field stays small), matching the
    Taylor-Couette stable-regime threshold."""
    d = _mk()
    d.set_perturbation(amp=1e-3, kz_mode=1)
    df = d.run(0.2)
    assert df["divu"] < 1e-4  # same threshold as the TC stable-regime test
    assert df["divb"] < 1e-4
    assert np.all(np.isfinite(np.array(d.x)))


# ---------------------------------------------------------------------------
# Hydro (B0=0) plane-Couette gates -- the same solver with no imposed field
# ---------------------------------------------------------------------------
def _mk_hydro(dt=2e-3, Nx=40):
    return AxisymmetricPCFMRIDNS(
        S=1.0,
        omega=0.0,
        B0=0.0,
        nu=1e-2,
        eta_mag=1e-2,
        Nx=Nx,
        Nz=16,
        Lz=1.0,
        dt=dt,
        family="C",
        dealias=1.0,
    )


def test_hydro_decay_matches_linear():
    """PRODUCTION GATE (hydro decay-vs-linear): non-rotating plane Couette is
    linearly stable; seed the leading hydro eigenmode (velocity only) and the DNS
    energy decays at the linear eigenvalue rate to spectral accuracy, with the
    magnetic field staying identically zero (pure hydro)."""
    d = _mk_hydro()
    s_lin = d.seed_hydro_eigenmode(kz_mode=1, amp=1e-4)
    assert s_lin.real < 0.0  # plane Couette is stable
    e0 = d.diagnostics(0.0, 0)["Ekin"]
    T = 0.4
    df = d.run(T)
    s_dns = 0.5 * math.log(df["Ekin"] / e0) / T
    assert abs(s_dns - s_lin.real) < 1e-4 * max(1.0, abs(s_lin.real))
    assert df["Emag"] == 0.0  # b decouples and stays 0
    assert df["divu"] < 1e-9


def test_hydro_restart_equivalence():
    """PRODUCTION GATE (hydro restart): checkpoint mid-run via state_dict and reload
    reproduces the uninterrupted hydro run bit-for-bit."""

    def fresh():
        d = _mk_hydro()
        d.seed_hydro_eigenmode(kz_mode=1, amp=1e-3)
        return d

    nsteps, split = 16, 8
    direct = fresh()
    for _ in range(nsteps):
        direct.step()
    first = fresh()
    for _ in range(split):
        first.step()
    ckpt = first.state_dict()
    assert ckpt["t"] == pytest.approx(split * first.dt)
    assert ckpt["tstep"] == split
    restarted = fresh()
    restarted.load_state_dict(ckpt)
    restarted.run((nsteps - split) * restarted.dt)
    assert np.max(np.abs(np.array(direct.x) - np.array(restarted.x))) < 1e-12
    assert direct.state_dict()["t"] == pytest.approx(nsteps * direct.dt)
    assert restarted.state_dict()["t"] == pytest.approx(nsteps * restarted.dt)
    assert direct.state_dict()["tstep"] == restarted.state_dict()["tstep"] == nsteps


@pytest.mark.slow
def test_hydro_temporal_order_cnab2():
    """PRODUCTION GATE (hydro temporal order): the decay-rate error vs the linear
    plane-Couette eigenvalue shrinks like dt^2 (CNAB2 ~ slope 2)."""

    def rate_error(dt, T=0.3):
        d = _mk_hydro(dt=dt)
        s_lin = d.seed_hydro_eigenmode(kz_mode=1, amp=1e-4)
        e0 = d.diagnostics(0.0, 0)["Ekin"]
        df = d.run(T)
        s_dns = 0.5 * math.log(df["Ekin"] / e0) / T
        return abs(s_dns - s_lin.real)

    dts = [4e-3, 2e-3, 1e-3]
    errs = [rate_error(dt) for dt in dts]
    assert all(b < a for a, b in zip(errs, errs[1:]))
    slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
    assert slope > 1.8


# ---------------------------------------------------------------------------
# Full 3D solver (PCFMRIDNS): functional tests for streamwise ky != 0 modes
# ---------------------------------------------------------------------------
def _mk3d(dt=2e-3, Nx=32):
    return PCFMRIDNS(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nx=Nx,
        Ny=8,
        Nz=16,
        Ly=2.0 * math.pi,
        Lz=1.0,
        dt=dt,
        family="C",
        dealias=1.0,
    )


def test_3d_ky0_reduces_to_axisymmetric_growth():
    """The 3D solver restricted to the ky=0 channel mode reproduces the linear MRI
    growth rate (and the 2D solver's value) to spectral accuracy -- validates every
    non-advection operator term in the 3D build."""
    d = _mk3d()
    s_lin = d.seed_linear_eigenmode(ky_mode=0, kz_mode=1, amp=1e-6)
    assert s_lin.real > 0.05
    e0 = d.diagnostics(0.0, 0)["E"]
    T = 0.5
    df = d.run(T)
    s_dns = 0.5 * math.log(df["E"] / e0) / T
    assert abs(s_dns - s_lin.real) < 1e-4 * max(1.0, abs(s_lin.real))
    assert df["divu"] < 1e-9 and df["divb"] < 1e-9


def test_3d_nonaxisymmetric_seed_is_solenoidal():
    """A seeded ky!=0 (streamwise) MRI eigenmode is divergence-free in both u and b
    to roundoff -- the direct block-copy injection plus the coupled saddle point."""
    d = _mk3d()
    s_lin = d.seed_linear_eigenmode(ky_mode=1, kz_mode=1, amp=1e-6)
    assert np.isfinite(s_lin.real)
    diag = d.diagnostics(0.0, 0)
    assert diag["divu"] < 1e-12
    assert diag["divb"] < 1e-12


def test_3d_nonaxisymmetric_run_finite_and_divergence_controlled():
    """A short run of a ky!=0 mode stays finite and divergence-controlled. (The
    base shear makes ky!=0 modes only instantaneously modal, so growth is not
    pinned here -- functionality + solenoidality is the gate.)"""
    d = _mk3d()
    d.seed_linear_eigenmode(ky_mode=1, kz_mode=1, amp=1e-6)
    df = d.run(0.2)
    assert np.all(np.isfinite(np.array(d.x)))
    assert df["divu"] < 1e-8
    assert df["divb"] < 1e-8


def test_3d_restart_equivalence():
    """Checkpoint/restart of the 3D solver is bit-for-bit (six (u,b) fields + AB2
    history) for a ky!=0 mode."""

    def fresh():
        d = _mk3d()
        d.seed_linear_eigenmode(ky_mode=1, kz_mode=1, amp=1e-4)
        return d

    nsteps, split = 12, 6
    direct = fresh()
    for _ in range(nsteps):
        direct.step()
    first = fresh()
    for _ in range(split):
        first.step()
    ckpt = first.state_dict()
    assert ckpt["t"] == pytest.approx(split * first.dt)
    assert ckpt["tstep"] == split
    restarted = fresh()
    restarted.load_state_dict(ckpt)
    restarted.run((nsteps - split) * restarted.dt)
    assert np.max(np.abs(np.array(direct.x) - np.array(restarted.x))) < 1e-12
    assert direct.state_dict()["t"] == pytest.approx(nsteps * direct.dt)
    assert restarted.state_dict()["t"] == pytest.approx(nsteps * restarted.dt)
    assert direct.state_dict()["tstep"] == restarted.state_dict()["tstep"] == nsteps
