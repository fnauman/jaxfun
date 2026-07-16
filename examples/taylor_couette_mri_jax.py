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
from jaxfun.la import (
    NONMODAL_FINITE_CAP,
    finite_eigensystem,
    generalized_eig,
    parse_times,
    print_transient_growth,
    transient_growth_from_eigs,
)

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
        family="C",
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

    def _dense_inner(self, test_expr, trial_expr):
        res = inner(
            test_expr * trial_expr, kind=InnerKind.BILINEAR, num_quad_points=self.N
        )
        return np.asarray(res.todense(), dtype=complex)

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

    def energy_matrix(self, m, kz, kind="total"):
        r"""Cylindrical perturbation-energy metric.

        ``kind`` selects ``'kinetic'``, ``'magnetic'``, or ``'total'``.
        Reference: ``couette/taylor_couette_mri.py:482-531``.
        """
        if kind not in ("total", "kinetic", "magnetic"):
            raise ValueError("kind must be 'total', 'kinetic', or 'magnetic'")
        want_kin = kind in ("total", "kinetic")
        want_mag = kind in ("total", "magnetic")
        n = self.n
        r = self.r

        if self.magnetic_bc == "conducting":
            Q = np.zeros((7 * n, 7 * n), dtype=complex)
            idx = {"ur": 0, "ut": 1, "uz": 2, "p": 3, "br": 4, "bt": 5, "bz": 6}
            names = [] if not want_kin else ["ur", "ut", "uz"]
            names += [] if not want_mag else ["br", "bt", "bz"]
            for name in names:
                W = self._blk(self.tv[name], self.tr[name], [(r, 0)])
                W = 0.5 * (W + W.conj().T)
                sl = slice(idx[name] * n, (idx[name] + 1) * n)
                Q[sl, sl] = W
            return Q

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
        Q = np.zeros((6 * n, 6 * n), dtype=complex)
        names = [] if not want_kin else ["ur", "ut", "uz"]
        names += [] if not want_mag else ["bt"]
        for name in names:
            W = self._blk(tv[name], tr[name], [(r, 0)])
            W = 0.5 * (W + W.conj().T)
            sl = slice(idx[name] * n, (idx[name] + 1) * n)
            Q[sl, sl] = W
        if want_mag:
            kz_s = sp.Float(float(kz))
            chi_metric = self._dense_inner(tv["chi"], (kz_s**2 / r) * tr["chi"])
            chi_metric += self._dense_inner(
                sp.diff(tv["chi"], r, 1), (1 / r) * sp.diff(tr["chi"], r, 1)
            )
            chi_metric = 0.5 * (chi_metric + chi_metric.conj().T)
            sl = slice(idx["chi"] * n, (idx["chi"] + 1) * n)
            Q[sl, sl] = chi_metric
        return Q

    def nonmodal_growth(
        self,
        m,
        kz,
        times,
        n_modes=None,
        finite_cap=NONMODAL_FINITE_CAP,
        energy="total",
    ):
        """Optimal linear transient growth in the selected energy norm.

        Reference: ``couette/taylor_couette_mri.py:533-545``.
        """
        w, V = finite_eigensystem(
            *self.assemble(m, kz), finite_cap=finite_cap, n_return=n_modes
        )
        return transient_growth_from_eigs(
            w, V, self.energy_matrix(m, kz, energy), times
        )

    def growth_rate(self, m, kz):
        w = generalized_eig(*self.assemble(m, kz))
        return float(w[0].real) if len(w) else float("nan")

    def max_growth_over_kz(self, m, kz_list):
        growth = np.array([self.growth_rate(m, kz) for kz in kz_list])
        i = int(np.argmax(growth))
        return float(kz_list[i]), float(growth[i]), growth

    def critical_eta_mag(self, m, kz, lo=1e-5, hi=1.0, iters=34):
        """Largest eta_mag still unstable at fixed dimensional B0 and nu."""
        L0, Lnu, Leta, M = self.assemble_parts(m, kz)
        L0nu = L0 + self.nu * Lnu

        def growth(eta):
            w = generalized_eig(L0nu + eta * Leta, M)
            return w[0].real if len(w) else -np.inf

        if growth(lo) < 0:
            return None
        if growth(hi) > 0:
            return hi
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            if growth(mid) > 0:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def critical_Rm_fixed_B0_nu(self, m=0, kz_list=None, **kw):
        """Critical Rm along the fixed dimensional B0/nu scan."""
        b = self.base
        if kz_list is None:
            kz_list = np.linspace(1.0, 8.0, 18) / b.gap
        best = None
        for kz in kz_list:
            eta_c = self.critical_eta_mag(m, kz, **kw)
            if eta_c is None:
                continue
            if best is None or eta_c > best[1]:
                best = (float(kz), float(eta_c))
        if best is None:
            return None
        kz_c, eta_c = best
        Rm_c = b.Omega1 * b.R1 * b.gap / eta_c
        return {
            "kz_c": kz_c,
            "eta_mag_c": eta_c,
            "Rm_c": Rm_c,
            "S_c": self.B0 * b.gap / eta_c,
        }

    def critical_eta_mag_fixed_controls(
        self, m, kz, Pm=None, S=None, lo=1e-5, hi=1.0, iters=34
    ):
        """Largest eta_mag still unstable at fixed Pm and Lundquist S."""
        b = self.base
        Pm = self.Pm if Pm is None else float(Pm)
        S = self.S if S is None else float(S)
        if not (Pm > 0 and np.isfinite(Pm)):
            raise ValueError("Pm must be a positive finite number")
        if not (S >= 0 and np.isfinite(S)):
            raise ValueError("S must be a non-negative finite number")

        def growth(eta):
            nu = Pm * eta
            B0 = S * eta / b.gap
            L0, Lnu, Leta, M = self.assemble_parts(m, kz, B0=B0)
            w = generalized_eig(L0 + nu * Lnu + eta * Leta, M)
            return w[0].real if len(w) else -np.inf

        if growth(lo) < 0:
            return None
        if growth(hi) > 0:
            return hi
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            if growth(mid) > 0:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def critical_Rm(self, m=0, kz_list=None, Pm=None, S=None, **kw):
        """Critical magnetic Reynolds number at fixed Pm and Lundquist S."""
        b = self.base
        if kz_list is None:
            kz_list = np.linspace(1.0, 8.0, 18) / b.gap
        Pm = self.Pm if Pm is None else float(Pm)
        S = self.S if S is None else float(S)
        best = None
        for kz in kz_list:
            eta_c = self.critical_eta_mag_fixed_controls(m, kz, Pm=Pm, S=S, **kw)
            if eta_c is None:
                continue
            if best is None or eta_c > best[1]:
                best = (float(kz), float(eta_c))
        if best is None:
            return None
        kz_c, eta_c = best
        return {
            "kz_c": kz_c,
            "eta_mag_c": eta_c,
            "nu_c": Pm * eta_c,
            "B0_c": S * eta_c / b.gap,
            "Rm_c": b.Omega1 * b.R1 * b.gap / eta_c,
            "Pm": Pm,
            "S_c": S,
        }


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
    parser.add_argument("--family", choices=["L", "C"], default="C")
    parser.add_argument(
        "--magnetic-bc", choices=["conducting", "insulating"], default="conducting"
    )
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--kz", type=float, default=3.0)
    parser.add_argument("--kz-min", type=float, default=0.5)
    parser.add_argument("--kz-max", type=float, default=8.0)
    parser.add_argument("--kz-num", type=int, default=30)
    parser.add_argument("--nonmodal", action="store_true")
    parser.add_argument("--times", type=str, default="1,5,10,20")
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument(
        "--energy", choices=["total", "kinetic", "magnetic"], default="total"
    )
    parser.add_argument("--critical-rm", action="store_true")
    parser.add_argument("--critical-fixed-B0-nu", action="store_true")
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
    print(base.describe())
    print(
        f"B0={args.B0:g} nu={args.nu:g} eta_mag={args.eta_mag:g} "
        f"Pm={solver.Pm:g} walls={args.magnetic_bc}"
    )
    print(
        f"Re={solver.Re:.3g} Rm={solver.Rm:.3g} "
        f"S(Lundquist)={solver.S:.3g} Ha={solver.Ha:.3g}"
    )

    if args.nonmodal:
        rows = solver.nonmodal_growth(
            args.m,
            args.kz,
            parse_times(args.times),
            n_modes=args.n_modes,
            energy=args.energy,
        )
        print(
            f"kz={args.kz:g}: MHD/MRI non-modal transient growth "
            f"({args.energy} energy):"
        )
        print_transient_growth(rows)
        return 0

    if args.critical_rm or args.critical_fixed_B0_nu:
        kzs = np.linspace(args.kz_min, args.kz_max, args.kz_num)
        result = (
            solver.critical_Rm_fixed_B0_nu(args.m, kzs)
            if args.critical_fixed_B0_nu
            else solver.critical_Rm(args.m, kzs)
        )
        print(result)
        return 0

    w, _ = solver.eigs(args.m, args.kz, n_return=6)
    print(f"kz={args.kz:g}: leading eigenvalues")
    for value in w:
        print(f"   s = {value.real:+.6e}  {value.imag:+.6e} i")
    print(f"growth rate = {w[0].real:+.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
