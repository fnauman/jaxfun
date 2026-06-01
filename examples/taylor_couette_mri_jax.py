"""Taylor-Couette MHD/MRI linear stability using jaxfun.

This ports the reference eigenproblem in couette/taylor_couette_mri.py.
The conducting-wall primitive system follows reference lines 137-290; the
axisymmetric insulating-wall flux formulation follows lines 292-411.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import sympy as sp
from scipy.special import iv, kv

from jaxfun.galerkin import FunctionSpace, InnerKind, TestFunction, TrialFunction, inner
from jaxfun.la import generalized_eig

try:
    from examples.taylor_couette_linear_jax import (
        CircularCouette,
        TaylorCouetteLinearJax,
    )
except ModuleNotFoundError:  # direct script execution from examples/
    from taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax


def mri_local_growth(omega_A, Omega, kappa2, dOmega2_dlnr):
    """Ideal local axisymmetric MRI growth from the shenfun reference."""
    A = omega_A**2 + 0.5 * kappa2
    C = omega_A**2 * (omega_A**2 + dOmega2_dlnr)
    disc = A**2 - C
    if disc < 0:
        return 0.0
    s2 = -A + math.sqrt(disc)
    return math.sqrt(s2) if s2 > 0 else 0.0


def mri_keplerian_optimum(Omega=1.0, vA=1.0):
    """Return the ideal local Keplerian MRI optimum."""
    q = 1.5
    kappa2 = (4 - 2 * q) * Omega**2
    dOmega2_dlnr = -2 * q * Omega**2
    wa = np.linspace(1e-3, math.sqrt(3.0) * Omega * 0.999, 4000)
    growth = np.array([mri_local_growth(w, Omega, kappa2, dOmega2_dlnr) for w in wa])
    i = int(np.argmax(growth))
    return {
        "s_max": float(growth[i]),
        "s_max_over_Omega": float(growth[i] / Omega),
        "wa2_opt_over_O2": float((wa[i] / Omega) ** 2),
        "theory_s_max_over_Omega": 0.75,
        "theory_wa2_opt": 15.0 / 16.0,
        "theory_cutoff_wa2": 3.0,
    }


class TaylorCouetteMRIJax:
    """Dense Taylor-Couette MRI eigenproblem assembled with jaxfun."""

    def __init__(
        self,
        base: CircularCouette,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        N=48,
        family="L",
        magnetic_bc="conducting",
    ):
        if magnetic_bc not in ("conducting", "insulating"):
            raise NotImplementedError(f"magnetic_bc={magnetic_bc!r} not implemented")
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.N = int(N)
        self.family = family.upper()
        self.family_cls = TaylorCouetteLinearJax._family_class(self.family)
        self.magnetic_bc = magnetic_bc
        self.Jm = 0.5 * (base.R2 - base.R1)

        self._linear = TaylorCouetteLinearJax(base, nu=nu, N=N, family=family)
        self.system = self._linear.system
        self.r = self._linear.r
        self.SDv = self._linear.SD
        self.SP = self._linear.SP
        self.n = self._linear.n
        dom = (base.R1, base.R2)

        spaces = {"ur": self.SDv, "ut": self.SDv, "uz": self.SDv, "p": self.SP}
        if magnetic_bc == "conducting":
            self.Sbt = FunctionSpace(
                N,
                self.family_cls,
                domain=dom,
                system=self.system,
                bc={
                    "left": {"R": (base.R1 / self.Jm, 0)},
                    "right": {"R": (base.R2 / self.Jm, 0)},
                },
            )
            self.Sbz = FunctionSpace(
                N,
                self.family_cls,
                bc={"left": {"N": 0}, "right": {"N": 0}},
                domain=dom,
                system=self.system,
            )
            if int(self.Sbt.dim) != self.n or int(self.Sbz.dim) != self.n:
                raise ValueError((self.Sbt.dim, self.Sbz.dim, self.n))
            spaces.update(br=self.SDv, bt=self.Sbt, bz=self.Sbz)
        self.tv = {name: TestFunction(space) for name, space in spaces.items()}
        self.tr = {name: TrialFunction(space) for name, space in spaces.items()}

        self.Pm = self.nu / self.eta_mag if self.eta_mag > 0 else math.inf
        d, R1, O1 = base.gap, base.R1, base.Omega1
        self.Re = O1 * R1 * d / self.nu if self.nu > 0 else math.inf
        self.Rm = O1 * R1 * d / self.eta_mag if self.eta_mag > 0 else math.inf
        self.S = self.B0 * d / self.eta_mag if self.eta_mag > 0 else math.inf
        self.Ha = (
            self.B0 * d / math.sqrt(self.nu * self.eta_mag)
            if self.nu * self.eta_mag > 0
            else math.inf
        )

    @staticmethod
    def _zero_coeff(coeff) -> bool:
        if coeff is None:
            return False
        if coeff == 0:
            return True
        is_zero = getattr(coeff, "is_zero", None)
        if is_zero is not None:
            return bool(is_zero)
        try:
            return complex(coeff) == 0
        except (TypeError, ValueError):
            return False

    def _trial_expr(self, trial, order: int):
        return trial if order == 0 else sp.diff(trial, self.r, order)

    def _blk(self, test, trial, terms):
        out = np.zeros((self.n, self.n), dtype=complex)
        for coeff, order in terms:
            if self._zero_coeff(coeff):
                continue
            t = self._trial_expr(trial, order)
            expr = t if coeff is None else coeff * t
            res = inner(test * expr, kind=InnerKind.BILINEAR, num_quad_points=self.N)
            out = out + np.asarray(res.todense(), dtype=complex)
        return out

    def _lap_terms(self, m, kz):
        r = self.r
        return [
            (None, 2),
            (1 / r, 1),
            (-(sp.Integer(int(m) ** 2) / r**2 + sp.Float(float(kz) ** 2)), 0),
        ]

    def assemble_parts(self, m, kz, B0=None):
        """Return (L0, Lnu, Leta, M) with L=L0+nu*Lnu+eta*Leta."""
        if self.magnetic_bc == "insulating":
            if int(m) != 0:
                raise NotImplementedError(
                    "insulating walls are implemented for m=0 only"
                )
            return self._assemble_flux_parts(kz, B0=B0)

        n = self.n
        r = self.r
        base = self.base
        m_s = sp.Integer(int(m))
        kz_s = sp.Float(float(kz))
        B0 = self.B0 if B0 is None else float(B0)
        ikzB0 = sp.I * kz_s * sp.Float(B0)
        imOm = sp.I * m_s * base.omega_expr(r)

        lap = self._lap_terms(m, kz)
        lv = lap + [(-1 / r**2, 0)]
        couple = [(2 * m_s * sp.I / r**2, 0)]

        L0 = np.zeros((7 * n, 7 * n), dtype=complex)
        Lnu = np.zeros_like(L0)
        Leta = np.zeros_like(L0)
        M = np.zeros_like(L0)
        idx = {"ur": 0, "ut": 1, "uz": 2, "p": 3, "br": 4, "bt": 5, "bz": 6}

        def put(block, row, col, value):
            i, j = idx[row], idx[col]
            block[i * n : (i + 1) * n, j * n : (j + 1) * n] += value

        tv, tr = self.tv, self.tr
        put(L0, "ur", "ur", self._blk(tv["ur"], tr["ur"], [(-imOm, 0)]))
        put(Lnu, "ur", "ur", self._blk(tv["ur"], tr["ur"], lv))
        put(
            L0, "ur", "ut", self._blk(tv["ur"], tr["ut"], [(base.two_omega_expr(r), 0)])
        )
        put(Lnu, "ur", "ut", -self._blk(tv["ur"], tr["ut"], couple))
        put(L0, "ur", "p", -self._blk(tv["ur"], tr["p"], [(None, 1)]))
        put(L0, "ur", "br", self._blk(tv["ur"], tr["br"], [(ikzB0, 0)]))
        put(M, "ur", "ur", self._blk(tv["ur"], tr["ur"], [(None, 0)]))

        put(
            L0,
            "ut",
            "ur",
            self._blk(tv["ut"], tr["ur"], [(-sp.Float(base.shear2a), 0)]),
        )
        put(Lnu, "ut", "ur", self._blk(tv["ut"], tr["ur"], couple))
        put(L0, "ut", "ut", self._blk(tv["ut"], tr["ut"], [(-imOm, 0)]))
        put(Lnu, "ut", "ut", self._blk(tv["ut"], tr["ut"], lv))
        put(L0, "ut", "p", -self._blk(tv["ut"], tr["p"], [(sp.I * m_s / r, 0)]))
        put(L0, "ut", "bt", self._blk(tv["ut"], tr["bt"], [(ikzB0, 0)]))
        put(M, "ut", "ut", self._blk(tv["ut"], tr["ut"], [(None, 0)]))

        put(L0, "uz", "uz", self._blk(tv["uz"], tr["uz"], [(-imOm, 0)]))
        put(Lnu, "uz", "uz", self._blk(tv["uz"], tr["uz"], lap))
        put(L0, "uz", "p", -self._blk(tv["uz"], tr["p"], [(sp.I * kz_s, 0)]))
        put(L0, "uz", "bz", self._blk(tv["uz"], tr["bz"], [(ikzB0, 0)]))
        put(M, "uz", "uz", self._blk(tv["uz"], tr["uz"], [(None, 0)]))

        put(L0, "p", "ur", self._blk(tv["p"], tr["ur"], [(None, 1), (1 / r, 0)]))
        put(L0, "p", "ut", self._blk(tv["p"], tr["ut"], [(sp.I * m_s / r, 0)]))
        put(L0, "p", "uz", self._blk(tv["p"], tr["uz"], [(sp.I * kz_s, 0)]))

        put(L0, "br", "ur", self._blk(tv["br"], tr["ur"], [(ikzB0, 0)]))
        put(L0, "br", "br", self._blk(tv["br"], tr["br"], [(-imOm, 0)]))
        put(Leta, "br", "br", self._blk(tv["br"], tr["br"], lv))
        put(Leta, "br", "bt", -self._blk(tv["br"], tr["bt"], couple))
        put(M, "br", "br", self._blk(tv["br"], tr["br"], [(None, 0)]))

        put(L0, "bt", "ut", self._blk(tv["bt"], tr["ut"], [(ikzB0, 0)]))
        put(
            L0,
            "bt",
            "br",
            self._blk(tv["bt"], tr["br"], [(base.r_omega_prime_expr(r), 0)]),
        )
        put(Leta, "bt", "br", self._blk(tv["bt"], tr["br"], couple))
        put(L0, "bt", "bt", self._blk(tv["bt"], tr["bt"], [(-imOm, 0)]))
        put(Leta, "bt", "bt", self._blk(tv["bt"], tr["bt"], lv))
        put(M, "bt", "bt", self._blk(tv["bt"], tr["bt"], [(None, 0)]))

        put(L0, "bz", "uz", self._blk(tv["bz"], tr["uz"], [(ikzB0, 0)]))
        put(L0, "bz", "bz", self._blk(tv["bz"], tr["bz"], [(-imOm, 0)]))
        put(Leta, "bz", "bz", self._blk(tv["bz"], tr["bz"], lap))
        put(M, "bz", "bz", self._blk(tv["bz"], tr["bz"], [(None, 0)]))
        return L0, Lnu, Leta, M

    def _flux_bases(self, kz):
        key = round(float(kz), 12)
        cache = self.__dict__.setdefault("_flux_basis_cache", {})
        if key in cache:
            return cache[key]
        N, b, dom = self.N, self.base, (self.base.R1, self.base.R2)
        if self.magnetic_bc == "conducting":
            Schi = FunctionSpace(
                N, self.family_cls, bc=(0, 0), domain=dom, system=self.system
            )
            Sbth = FunctionSpace(
                N,
                self.family_cls,
                domain=dom,
                system=self.system,
                bc={
                    "left": {"R": (b.R1 / self.Jm, 0)},
                    "right": {"R": (b.R2 / self.Jm, 0)},
                },
            )
        else:
            k = abs(float(kz))
            if k < 1e-12:
                raise ValueError("insulating BCs require kz != 0")
            kap_in = k * iv(1, k * b.R1) / iv(0, k * b.R1)
            kap_out = -k * kv(1, k * b.R2) / kv(0, k * b.R2)
            Schi = FunctionSpace(
                N,
                self.family_cls,
                domain=dom,
                system=self.system,
                bc={
                    "left": {"R": (-kap_in / (k * k * self.Jm), 0)},
                    "right": {"R": (-kap_out / (k * k * self.Jm), 0)},
                },
            )
            Sbth = FunctionSpace(
                N, self.family_cls, bc=(0, 0), domain=dom, system=self.system
            )
        if int(Schi.dim) != self.n or int(Sbth.dim) != self.n:
            raise ValueError((Schi.dim, Sbth.dim, self.n))
        cache[key] = (Schi, Sbth)
        return cache[key]

    def _assemble_flux_parts(self, kz, B0=None):
        n = self.n
        r = self.r
        base = self.base
        B0 = self.B0 if B0 is None else float(B0)
        kz_s = sp.Float(float(kz))
        ikzB0 = sp.I * kz_s * sp.Float(B0)
        Schi, Sbth = self._flux_bases(kz)
        spaces = {
            "ur": self.SDv,
            "ut": self.SDv,
            "uz": self.SDv,
            "p": self.SP,
            "chi": Schi,
            "bt": Sbth,
        }
        tv = {name: TestFunction(space) for name, space in spaces.items()}
        tr = {name: TrialFunction(space) for name, space in spaces.items()}
        idx = {"ur": 0, "ut": 1, "uz": 2, "p": 3, "chi": 4, "bt": 5}

        L0 = np.zeros((6 * n, 6 * n), dtype=complex)
        Lnu = np.zeros_like(L0)
        Leta = np.zeros_like(L0)
        M = np.zeros_like(L0)

        def put(block, row, col, value):
            i, j = idx[row], idx[col]
            block[i * n : (i + 1) * n, j * n : (j + 1) * n] += value

        Lp = [(None, 2), (1 / r, 1), (-(kz_s**2), 0)]
        Lv = Lp + [(-1 / r**2, 0)]
        Lchi = [(None, 2), (-1 / r, 1), (-(kz_s**2), 0)]

        put(
            L0, "ur", "ut", self._blk(tv["ur"], tr["ut"], [(base.two_omega_expr(r), 0)])
        )
        put(L0, "ur", "p", -self._blk(tv["ur"], tr["p"], [(None, 1)]))
        put(Lnu, "ur", "ur", self._blk(tv["ur"], tr["ur"], Lv))
        put(
            L0,
            "ur",
            "chi",
            self._blk(tv["ur"], tr["chi"], [(kz_s**2 * sp.Float(B0) / r, 0)]),
        )
        put(M, "ur", "ur", self._blk(tv["ur"], tr["ur"], [(None, 0)]))

        put(
            L0,
            "ut",
            "ur",
            self._blk(tv["ut"], tr["ur"], [(-sp.Float(base.shear2a), 0)]),
        )
        put(Lnu, "ut", "ut", self._blk(tv["ut"], tr["ut"], Lv))
        put(L0, "ut", "bt", self._blk(tv["ut"], tr["bt"], [(ikzB0, 0)]))
        put(M, "ut", "ut", self._blk(tv["ut"], tr["ut"], [(None, 0)]))

        put(L0, "uz", "p", -self._blk(tv["uz"], tr["p"], [(sp.I * kz_s, 0)]))
        put(Lnu, "uz", "uz", self._blk(tv["uz"], tr["uz"], Lp))
        put(L0, "uz", "chi", self._blk(tv["uz"], tr["chi"], [(ikzB0 / r, 1)]))
        put(M, "uz", "uz", self._blk(tv["uz"], tr["uz"], [(None, 0)]))

        put(L0, "p", "ur", self._blk(tv["p"], tr["ur"], [(None, 1), (1 / r, 0)]))
        put(L0, "p", "uz", self._blk(tv["p"], tr["uz"], [(sp.I * kz_s, 0)]))

        put(L0, "chi", "ur", self._blk(tv["chi"], tr["ur"], [(-sp.Float(B0) * r, 0)]))
        put(Leta, "chi", "chi", self._blk(tv["chi"], tr["chi"], Lchi))
        put(M, "chi", "chi", self._blk(tv["chi"], tr["chi"], [(None, 0)]))

        put(L0, "bt", "ut", self._blk(tv["bt"], tr["ut"], [(ikzB0, 0)]))
        put(
            L0,
            "bt",
            "chi",
            self._blk(
                tv["bt"],
                tr["chi"],
                [(-sp.I * kz_s * base.r_omega_prime_expr(r) / r, 0)],
            ),
        )
        put(Leta, "bt", "bt", self._blk(tv["bt"], tr["bt"], Lv))
        put(M, "bt", "bt", self._blk(tv["bt"], tr["bt"], [(None, 0)]))
        return L0, Lnu, Leta, M

    def assemble(self, m, kz):
        L0, Lnu, Leta, M = self.assemble_parts(m, kz)
        return L0 + self.nu * Lnu + self.eta_mag * Leta, M

    def eigs(self, m, kz, n_return=8, finite_cap=1.0e6):
        w, V = generalized_eig(
            *self.assemble(m, kz), vectors=True, finite_cap=finite_cap
        )
        return w[:n_return], V[:, :n_return]

    def growth_rate(self, m, kz):
        w = generalized_eig(*self.assemble(m, kz))
        return float(w[0].real) if len(w) else float("nan")

    def max_growth_over_kz(self, m, kz_list):
        growth = np.array([self.growth_rate(m, kz) for kz in kz_list])
        i = int(np.argmax(growth))
        return float(kz_list[i]), float(growth[i]), growth


def _default_omega2(R1, R2, Omega1):
    eta = R1 / R2
    return Omega1 * eta**1.5


def main(argv=None):
    parser = argparse.ArgumentParser(description="Taylor-Couette MRI with jaxfun")
    parser.add_argument("--R1", type=float, default=1.0)
    parser.add_argument("--R2", type=float, default=2.0)
    parser.add_argument("--Omega1", type=float, default=1.0)
    parser.add_argument("--Omega2", type=float, default=None)
    parser.add_argument("--B0", type=float, default=0.1)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--eta-mag", type=float, default=1e-3)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--family", choices=["L", "C"], default="L")
    parser.add_argument(
        "--magnetic-bc", choices=["conducting", "insulating"], default="conducting"
    )
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--kz", type=float, default=3.0)
    parser.add_argument("--local-check", action="store_true")
    args = parser.parse_args(argv)

    if args.local_check:
        print(mri_keplerian_optimum())
        return 0

    Omega2 = (
        args.Omega2
        if args.Omega2 is not None
        else _default_omega2(args.R1, args.R2, args.Omega1)
    )
    base = CircularCouette(args.R1, args.R2, args.Omega1, Omega2)
    solver = TaylorCouetteMRIJax(
        base,
        B0=args.B0,
        nu=args.nu,
        eta_mag=args.eta_mag,
        N=args.N,
        family=args.family,
        magnetic_bc=args.magnetic_bc,
    )
    w, _ = solver.eigs(args.m, args.kz, n_return=6)
    print(base.describe())
    print(
        f"B0={args.B0:g} nu={args.nu:g} eta_mag={args.eta_mag:g} "
        f"Pm={solver.Pm:g} walls={args.magnetic_bc}"
    )
    print(f"kz={args.kz:g}: leading eigenvalues")
    for value in w:
        print(f"   s = {value.real:+.6e}  {value.imag:+.6e} i")
    print(f"growth rate = {w[0].real:+.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
