"""Taylor-Couette hydrodynamic linear stability using jaxfun.

This ports the dense generalized eigenproblem from
couette/taylor_couette_linear.py to jaxfun's Galerkin assembly.  The
reference implementation is read-only ground truth; the block structure below
follows couette/taylor_couette_linear.py:145-252.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import sympy as sp

from jaxfun.coordinates import CartCoordSys, x as coord_x
from jaxfun.galerkin import FunctionSpace, InnerKind, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.la import (
    NONMODAL_FINITE_CAP,
    finite_eigensystem,
    generalized_eig,
    parse_times,
    print_transient_growth,
    transient_growth_from_eigs,
)


class CircularCouette:
    """Circular-Couette base flow from couette/taylor_couette_linear.py.

    Reference: couette/taylor_couette_linear.py:71-127.
    """

    def __init__(self, R1=1.0, R2=2.0, Omega1=1.0, Omega2=0.0):
        if not (R2 > R1 > 0):
            raise ValueError("require 0 < R1 < R2")
        self.R1, self.R2 = float(R1), float(R2)
        self.Omega1, self.Omega2 = float(Omega1), float(Omega2)
        d2 = self.R2**2 - self.R1**2
        self.a = (self.Omega2 * self.R2**2 - self.Omega1 * self.R1**2) / d2
        self.b = (self.Omega1 - self.Omega2) * self.R1**2 * self.R2**2 / d2
        self.gap = self.R2 - self.R1
        self.eta = self.R1 / self.R2
        self.mu = self.Omega2 / self.Omega1 if self.Omega1 != 0 else math.inf
        self.shear2a = 2 * self.a

    def Omega(self, r):
        return self.a + self.b / np.asarray(r) ** 2

    def V(self, r):
        r = np.asarray(r)
        return self.a * r + self.b / r

    def kappa2(self, r):
        return 4 * self.a * self.Omega(r)

    def rayleigh_stable(self):
        rr = np.linspace(self.R1, self.R2, 200)
        return bool(np.all(self.kappa2(rr) > 0))

    def q_shear(self, r):
        r = np.asarray(r)
        return 2 * self.b / (self.a * r**2 + self.b)

    def omega_expr(self, r):
        return self.a + self.b / r**2

    def two_omega_expr(self, r):
        return 2 * self.a + 2 * self.b / r**2

    def r_omega_prime_expr(self, r):
        return -2 * self.b / r**2

    def kappa2_expr(self, r):
        return 4 * self.a * self.omega_expr(r)

    def describe(self):
        return (
            f"CircularCouette: R1={self.R1:g} R2={self.R2:g} "
            f"eta={self.eta:.4f} Omega1={self.Omega1:g} "
            f"Omega2={self.Omega2:g} mu={self.mu:.4f}\n"
            f"  a={self.a:.6g} b={self.b:.6g}  "
            f"Rayleigh-{'stable' if self.rayleigh_stable() else 'UNSTABLE'} "
            f"(kappa^2 at R1={self.kappa2(self.R1):.4g}, "
            f"R2={self.kappa2(self.R2):.4g})"
        )


class TaylorCouetteLinearJax:
    """Dense hydrodynamic Taylor-Couette eigenproblem assembled by jaxfun.

    Reference: couette/taylor_couette_linear.py:145-252.
    """

    def __init__(self, base: CircularCouette, nu=1.0e-3, N=48, family="C"):
        self.base = base
        self.nu = float(nu)
        self.N = int(N)
        self.family = family.upper()
        dom = (base.R1, base.R2)
        family_cls = self._family_class(self.family)
        self.system = CartCoordSys("TC", (coord_x,))
        self.r = self.system.base_scalars()[0]

        # Velocity uses no-slip Dirichlet modes (N-2 active dofs).  Pressure
        # uses the first N-2 orthogonal modes, matching shenfun's sliced P_N/P_{N-2}
        # pair while preserving N-point quadrature in all blocks.
        self.SD = FunctionSpace(
            N, family_cls, bc=(0, 0), domain=dom, system=self.system, name="SD"
        )
        self.SP = FunctionSpace(
            N - 2, family_cls, domain=dom, system=self.system, name="SP"
        )
        self.n = int(self.SD.dim)
        if int(self.SP.dim) != self.n:
            raise ValueError((self.SP.dim, self.n))

        self.vD = TestFunction(self.SD)
        self.uD = TrialFunction(self.SD)
        self.qP = TestFunction(self.SP)
        self.pP = TrialFunction(self.SP)

    @staticmethod
    def _family_class(family: str):
        if family.startswith("L"):
            return Legendre
        if family.startswith("C"):
            return Chebyshev
        raise ValueError("family must be 'L' or 'C'")

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

    def _dense(self, test, expr, shape):
        if expr == 0:
            return np.zeros(shape, dtype=complex)
        res = inner(test * expr, kind=InnerKind.BILINEAR, num_quad_points=self.N)
        return np.asarray(res.todense(), dtype=complex)

    def _trial_expr(self, trial, order: int):
        return trial if order == 0 else sp.diff(trial, self.r, order)

    def _A(self, coeff, order):
        """Return (v_D, coeff * d^order u_D/dr^order).

        Reference: couette/taylor_couette_linear.py:180-191.
        """
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        u = self._trial_expr(self.uD, order)
        expr = u if coeff is None else coeff * u
        return self._dense(self.vD, expr, (self.n, self.n))

    def _Avp(self, coeff, order):
        """Return velocity-test / pressure-trial block.

        Reference: couette/taylor_couette_linear.py:193-199.
        """
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        p = self._trial_expr(self.pP, order)
        expr = p if coeff is None else coeff * p
        return self._dense(self.vD, expr, (self.n, self.n))

    def _Aqu(self, coeff, order):
        """Return pressure-test / velocity-trial block.

        Reference: couette/taylor_couette_linear.py:201-207.
        """
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        u = self._trial_expr(self.uD, order)
        expr = u if coeff is None else coeff * u
        return self._dense(self.qP, expr, (self.n, self.n))

    def _laplacian_scalar(self, m, kz):
        r = self.r
        return (
            self._A(None, 2)
            + self._A(1 / r, 1)
            - self._A(sp.Integer(int(m) ** 2) / r**2 + sp.Float(float(kz) ** 2), 0)
        )

    def assemble_parts(self, m, kz):
        """Build (L0, Lv, M) for mode (m, kz).

        Reference: couette/taylor_couette_linear.py:209-234.
        """
        n = self.n
        r = self.r
        base = self.base
        m_i = int(m)
        m_s = sp.Integer(m_i)
        kz_f = float(kz)
        kz_s = sp.Float(kz_f)

        Lap = self._laplacian_scalar(m_i, kz_f)
        Mvv = self._A(None, 0)

        lv_diag = Lap - self._A(1 / r**2, 0)
        lv_z = Lap
        adv = -1j * self._A(m_s * base.omega_expr(r), 0)

        twoOmega = self._A(base.two_omega_expr(r), 0)
        shear = self._A(sp.Float(base.shear2a), 0)
        couple_rt = 1j * self._A(2 * m_s / r**2, 0)

        Gr = self._Avp(None, 1)
        Gt = 1j * self._Avp(m_s / r, 0)
        Gz = 1j * self._Avp(kz_s, 0)

        Dr = self._Aqu(None, 1) + self._Aqu(1 / r, 0)
        Dt = 1j * self._Aqu(m_s / r, 0)
        Dz = 1j * self._Aqu(kz_s, 0)

        L0 = np.zeros((4 * n, 4 * n), dtype=complex)
        Lv = np.zeros((4 * n, 4 * n), dtype=complex)
        M = np.zeros((4 * n, 4 * n), dtype=complex)

        def put(block, i, j, value):
            block[i * n : (i + 1) * n, j * n : (j + 1) * n] = value

        put(L0, 0, 0, adv)
        put(Lv, 0, 0, lv_diag)
        put(L0, 0, 1, twoOmega)
        put(Lv, 0, 1, -couple_rt)
        put(L0, 0, 3, -Gr)
        put(M, 0, 0, Mvv)

        put(L0, 1, 0, -shear)
        put(Lv, 1, 0, couple_rt)
        put(L0, 1, 1, adv)
        put(Lv, 1, 1, lv_diag)
        put(L0, 1, 3, -Gt)
        put(M, 1, 1, Mvv)

        put(L0, 2, 2, adv)
        put(Lv, 2, 2, lv_z)
        put(L0, 2, 3, -Gz)
        put(M, 2, 2, Mvv)

        put(L0, 3, 0, Dr)
        put(L0, 3, 1, Dt)
        put(L0, 3, 2, Dz)
        return L0, Lv, M

    def assemble(self, m, kz):
        L0, Lv, M = self.assemble_parts(m, kz)
        return L0 + self.nu * Lv, M

    def eigs(self, m, kz, n_return=8, finite_cap=1.0e6):
        w, V = generalized_eig(
            *self.assemble(m, kz), vectors=True, finite_cap=finite_cap
        )
        return w[:n_return], V[:, :n_return]

    def energy_matrix(self):
        """Cylindrical kinetic-energy metric for ``(u_r,u_theta,u_z,p)``.

        Reference: ``couette/taylor_couette_linear.py:328-335``.
        """
        Q = np.zeros((4 * self.n, 4 * self.n), dtype=complex)
        W = self._A(self.r, 0)
        W = 0.5 * (W + W.conj().T)
        for comp in range(3):
            sl = slice(comp * self.n, (comp + 1) * self.n)
            Q[sl, sl] = W
        return Q

    def nonmodal_growth(
        self, m, kz, times, n_modes=None, finite_cap=NONMODAL_FINITE_CAP
    ):
        """Optimal linear transient growth in kinetic-energy norm.

        Reference: ``couette/taylor_couette_linear.py:337-341``.
        """
        w, V = finite_eigensystem(
            *self.assemble(m, kz), finite_cap=finite_cap, n_return=n_modes
        )
        return transient_growth_from_eigs(w, V, self.energy_matrix(), times)

    def growth_rate(self, m, kz):
        w = generalized_eig(*self.assemble(m, kz))
        return float(w[0].real) if len(w) else float("nan")

    def max_growth_over_kz(self, m, kz_list):
        growth = np.array([self.growth_rate(m, kz) for kz in kz_list])
        i = int(np.argmax(growth))
        return float(kz_list[i]), float(growth[i]), growth

    def critical_nu(self, m, kz, nu_lo=1e-5, nu_hi=2.0, iters=34):
        L0, Lv, M = self.assemble_parts(m, kz)

        def growth(nu):
            w = generalized_eig(L0 + nu * Lv, M)
            return w[0].real if len(w) else -np.inf

        if growth(nu_lo) < 0:
            return None
        if growth(nu_hi) > 0:
            return nu_hi
        for _ in range(iters):
            nu_mid = 0.5 * (nu_lo + nu_hi)
            if growth(nu_mid) > 0:
                nu_lo = nu_mid
            else:
                nu_hi = nu_mid
        return 0.5 * (nu_lo + nu_hi)

    def critical_over_kz(self, m, kz_list, **kw):
        best = None
        for kz in kz_list:
            nu_c = self.critical_nu(m, kz, **kw)
            if nu_c is None:
                continue
            if best is None or nu_c > best[1]:
                best = (float(kz), float(nu_c))
        return best

    def critical_reynolds(self, m=0, kz_list=None, refine=True):
        base = self.base
        if kz_list is None:
            kz_list = np.linspace(2.0, 4.5, 16) / base.gap
        best = self.critical_over_kz(m, kz_list)
        if best is None:
            return None
        kz_c, nu_c = best
        if refine:
            dk = (kz_list[1] - kz_list[0]) if len(kz_list) > 1 else 0.2
            fine = np.linspace(kz_c - dk, kz_c + dk, 11)
            fine_best = self.critical_over_kz(m, fine)
            if fine_best is not None:
                kz_c, nu_c = fine_best
        Re_c = base.Omega1 * base.R1 * base.gap / nu_c
        return {"kz_c": kz_c, "nu_c": nu_c, "Re_c": Re_c, "a_c": kz_c * base.gap}


def _build(args):
    base = CircularCouette(args.R1, args.R2, args.Omega1, args.Omega2)
    solver = TaylorCouetteLinearJax(base, nu=args.nu, N=args.N, family=args.family)
    return base, solver


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Taylor-Couette hydrodynamic linear stability with jaxfun"
    )
    parser.add_argument("--R1", type=float, default=1.0)
    parser.add_argument("--R2", type=float, default=2.0)
    parser.add_argument("--Omega1", type=float, default=1.0)
    parser.add_argument("--Omega2", type=float, default=0.0)
    parser.add_argument("--nu", type=float, default=2.0e-3)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--family", choices=["L", "C"], default="C")
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--kz", type=float, default=None)
    parser.add_argument("--kz-min", type=float, default=0.5)
    parser.add_argument("--kz-max", type=float, default=8.0)
    parser.add_argument("--kz-num", type=int, default=40)
    parser.add_argument("--nonmodal", action="store_true")
    parser.add_argument("--times", type=str, default="1,5,10,20")
    parser.add_argument("--n-modes", type=int, default=None)
    args = parser.parse_args(argv)

    base, solver = _build(args)
    print(base.describe())
    print(f"  nu={args.nu:g}  N={args.N}  family={args.family}  m={args.m}")

    if args.nonmodal:
        kz = args.kz if args.kz is not None else 3.0 / base.gap
        rows = solver.nonmodal_growth(
            args.m, kz, parse_times(args.times), n_modes=args.n_modes
        )
        print(f"\nkz={kz:g}: hydrodynamic non-modal transient growth:")
        print_transient_growth(rows)
    elif args.kz is not None:
        w, _ = solver.eigs(args.m, args.kz, n_return=6)
        print(f"\nkz={args.kz:g}: leading eigenvalues (growth, freq):")
        for value in w:
            print(f"   s = {value.real:+.6e}  {value.imag:+.6e} i")
        print(f"  growth rate = {w[0].real:+.6e}")
    else:
        kzs = np.linspace(args.kz_min, args.kz_max, args.kz_num)
        kz_best, growth_best, _ = solver.max_growth_over_kz(args.m, kzs)
        print(f"\nkz scan [{args.kz_min},{args.kz_max}] (m={args.m}):")
        print(f"  most unstable kz = {kz_best:.4f}  growth = {growth_best:+.6e}")
        state = "UNSTABLE" if growth_best > 1e-9 else "stable"
        print(f"  -> base flow is {state} at nu={args.nu:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
