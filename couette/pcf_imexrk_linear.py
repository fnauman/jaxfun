"""IMEXRK linear plane-Couette time-stepper companion.

This is the plane-Couette counterpart of ``taylor_couette_imexrk.py``.  Both
steppers share the IMEXRK Butcher tableaux (:func:`_linear_analysis.imex_tableau`)
and the descriptor-system step core (:func:`_linear_analysis.imexrk_step`), so a
DNS-style time-stepped comparison advances plane Couette and Taylor Couette with
*identical* integrator logic -- the apples-to-apples time-stepping path that the
eigenvalue and non-modal comparisons cannot exercise on their own.

Scope: the dense primitive-variable linear operator assembled by
``PlaneCouetteGalerkinLinear`` (hydro or MHD).  The default ``diffusion`` split
treats viscous/resistive diffusion plus the pressure/continuity (and, in MHD,
the magnetic-pressure/solenoidal) saddle-point rows implicitly and all remaining
linear couplings -- advection, Coriolis, shear, the imposed-field induction
couplings -- explicitly, matching the IMEX split used by the PCF DNS solvers.
``--split full`` is a fully-implicit reference.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.linalg import solve

sys.path.insert(0, os.path.dirname(__file__))

from _linear_analysis import imex_tableau, imexrk_step  # noqa: E402
from pcf_galerkin_linear import PlaneCouetteGalerkinLinear  # noqa: E402


class PlaneCouetteIMEXRKLinearStepper:
    """Dense descriptor-system IMEXRK stepper for the PCF linear operator."""

    def __init__(self, operator: PlaneCouetteGalerkinLinear, ky=0.0, kz=1.0,
                 dt=1.0e-3, scheme="IMEXRK222", split="diffusion",
                 energy="total"):
        self.operator = operator
        self.ky = float(ky)
        self.kz = float(kz)
        self.dt = float(dt)
        self.scheme = scheme.upper()
        self.split = split
        self.energy_kind = energy
        self.n = operator.n
        # Zero-mass (saddle-point) constraint blocks: continuity pressure ``p``
        # and, for MHD, the magnetic-pressure / solenoidal multiplier ``phi``.
        self.pblocks = [3, 7] if operator.mhd else [3]
        self.a, self.b = imex_tableau(self.scheme)
        self.Aimp, self.Aexp, self.M = self._assemble_split()
        self._lhs_cache = {}
        self.Q = operator.energy_matrix(self.energy_kind)

    @classmethod
    def couette(cls, N=48, Re=1000.0, Rm=None, family="C", mhd=False,
                half_gap=1.0, U_wall=1.0, by=0.0, bz=0.0,
                magnetic_bc="conducting", **kw):
        op = PlaneCouetteGalerkinLinear.couette(
            N=N, Re=Re, Rm=Rm, family=family, half_gap=half_gap, U_wall=U_wall,
            mhd=mhd, by=by, bz=bz, magnetic_bc=magnetic_bc)
        return cls(op, **kw)

    @classmethod
    def shearbox(cls, N=48, Re=1000.0, Rm=None, family="C", shear_rate=1.0,
                 omega=2.0 / 3.0, by=0.0, bz=0.025, magnetic_bc="conducting",
                 **kw):
        op = PlaneCouetteGalerkinLinear.shearbox(
            N=N, Re=Re, Rm=Rm, family=family, shear_rate=shear_rate,
            omega=omega, by=by, bz=bz, magnetic_bc=magnetic_bc)
        return cls(op, **kw)

    def _assemble_split(self):
        L0, Lnu, Leta, M = self.operator.assemble_parts(self.ky, self.kz)
        diffusion = self.operator.nu * Lnu + self.operator.eta * Leta
        if self.split == "full":
            return L0 + diffusion, np.zeros_like(L0), M
        if self.split != "diffusion":
            raise ValueError("split must be 'diffusion' or 'full'")
        n = self.n
        pressure = np.zeros_like(L0)
        for c in self.pblocks:
            col = slice(c * n, (c + 1) * n)
            pressure[:, col] = L0[:, col]                 # gradient columns
            pressure[c * n:(c + 1) * n, :] = L0[c * n:(c + 1) * n, :]  # constraint rows
        Aimp = diffusion + pressure
        Aexp = L0 - pressure
        return Aimp, Aexp, M

    def _solve_stage(self, gamma, rhs):
        key = float(gamma)
        lhs = self._lhs_cache.get(key)
        if lhs is None:
            lhs = self.M - self.dt * gamma * self.Aimp
            lhs = np.array(lhs, dtype=complex, copy=True)
            for c in self.pblocks:
                row = c * self.n
                lhs[row, :] = 0.0
                lhs[row, row] = 1.0
            self._lhs_cache[key] = lhs
        rr = np.array(rhs, dtype=complex, copy=True)
        for c in self.pblocks:
            rr[c * self.n] = 0.0
        return solve(lhs, rr, assume_a="gen")

    def step(self, q):
        return imexrk_step(q, self.Aimp, self.Aexp, self.M, self.a, self.b,
                           self.dt, self._solve_stage)

    def nsteps_for(self, end_time):
        nsteps_float = float(end_time) / self.dt
        nsteps = int(round(nsteps_float))
        if not np.isclose(nsteps * self.dt, float(end_time),
                          rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("end_time must be an integer multiple of dt")
        return nsteps

    def integrate(self, q0, end_time, return_history=False):
        q = np.asarray(q0, dtype=complex)
        t = 0.0
        rows = [(t, q.copy())] if return_history else None
        for _ in range(self.nsteps_for(end_time)):
            q = self.step(q)
            t += self.dt
            if rows is not None:
                rows.append((t, q.copy()))
        return rows if rows is not None else q

    def leading_eigenmode(self):
        w, V = self.operator.eigs(self.ky, self.kz, n_return=1)
        return w[0], V[:, 0]

    def energy(self, q):
        q = np.asarray(q, dtype=complex)
        return float(np.real(q.conj().T @ self.Q @ q))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mhd", action="store_true")
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--family", choices=("C", "L"), default="C")
    p.add_argument("--Re", type=float, default=1000.0)
    p.add_argument("--Rm", type=float, default=None)
    p.add_argument("--Uprime", type=float, default=1.0)
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--by", type=float, default=0.0)
    p.add_argument("--bz", type=float, default=0.0)
    p.add_argument("--ky", type=float, default=0.0)
    p.add_argument("--kz", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1.0e-3)
    p.add_argument("--end-time", type=float, default=0.05)
    p.add_argument("--scheme", choices=("IMEXRK111", "IMEXRK222", "IMEXRK443"),
                   default="IMEXRK222")
    p.add_argument("--split", choices=("diffusion", "full"), default="diffusion")
    p.add_argument("--energy", choices=("total", "kinetic", "magnetic"),
                   default="total")
    return p.parse_args()


def main():
    args = _parse_args()
    rm = args.Re if args.Rm is None else args.Rm
    op = PlaneCouetteGalerkinLinear(
        N=args.N, family=args.family, nu=abs(args.Uprime) / args.Re,
        eta=abs(args.Uprime) / rm, Uprime=args.Uprime, omega=args.omega,
        mhd=args.mhd, by=args.by, bz=args.bz)
    stepper = PlaneCouetteIMEXRKLinearStepper(
        op, ky=args.ky, kz=args.kz, dt=args.dt, scheme=args.scheme,
        split=args.split, energy=args.energy)
    s, q0 = stepper.leading_eigenmode()
    nsteps = stepper.nsteps_for(args.end_time)
    e0 = stepper.energy(q0)
    q1 = stepper.integrate(q0, args.end_time)
    e1 = stepper.energy(q1)
    measured = 0.5 * np.log(e1 / e0) / args.end_time

    print("Plane-Couette linear IMEXRK check")
    print(f"  scheme={args.scheme} split={args.split} mhd={args.mhd}")
    print(f"  leading eigenvalue s={s.real:+.8e} {s.imag:+.8e}i")
    print(f"  end_time={args.end_time:g} dt={args.dt:g} steps={nsteps}")
    print(f"  exact energy-amplitude factor={np.exp(s.real * args.end_time):.8e}")
    print(f"  measured energy-amplitude factor={np.sqrt(e1 / e0):.8e}")
    print(f"  measured growth rate={measured:+.8e}")


if __name__ == "__main__":
    main()
