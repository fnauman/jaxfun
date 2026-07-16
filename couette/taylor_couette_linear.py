r"""
Linear stability of Taylor-Couette flow (hydrodynamic and MHD) with shenfun.

This is a global spectral generalized-eigenvalue solver for the viscous,
incompressible flow between two concentric, independently rotating cylinders
(radii ``R1 < R2``, angular velocities ``Omega1, Omega2``).  It is the
cylindrical-geometry companion to the Cartesian plane-Couette MHD/MRI demos in
this directory (``pcf_mhd_*``): instead of a local shearing box it uses the true
annulus and the exact circular-Couette base flow.

Base flow (circular Couette / "ideal Couette" profile)::

    Omega(r) = a + b / r**2,   V(r) = Omega(r) * r
    a = (Omega2 R2**2 - Omega1 R1**2) / (R2**2 - R1**2)
    b = (Omega1 - Omega2) R1**2 R2**2 / (R2**2 - R1**2)

so that ``V(R1)=Omega1 R1`` and ``V(R2)=Omega2 R2``.  Useful identities used in
the linearised equations::

    2*Omega + r*dOmega/dr = (1/r) d(r**2 Omega)/dr = 2 a        (constant)
    r dOmega/dr           = -2 b / r**2
    kappa**2 (epicyclic)  = (1/r**3) d(r**2 Omega)**2/dr = 4 a Omega(r)

The Rayleigh (inviscid centrifugal) criterion for axisymmetric stability is
``kappa**2 = 4 a Omega(r) > 0`` everywhere; a Keplerian profile
``Omega ~ r**(-3/2)`` has ``kappa**2 = Omega**2 > 0`` and is therefore
Rayleigh-stable yet (with a magnetic field) MRI-unstable.

Discretisation
--------------
Perturbations are taken as ``q(r) exp(s t + i m theta + i kz z)`` with azimuthal
mode number ``m`` and axial wavenumber ``kz`` fixed parameters.  Only the radial
direction is discretised, with a 1D Chebyshev/Legendre Galerkin basis on
``[R1, R2]`` (the OrrSommerfeld-eigs strong-form / plain-measure pattern, with
the cylindrical 1/r factors carried as explicit sympy coefficients of the
radial coordinate symbol ``x``).  Velocity uses a Dirichlet (no-slip) basis,
pressure an orthogonal basis sliced to N-2 modes (the inf-sup-stable
``P_N`` - ``P_{N-2}`` pair).  The resulting generalised eigenvalue problem

    L q = s M q

is solved with ``scipy.linalg.eig``; ``M`` is singular (no time derivative in
the continuity row / pressure column) so non-physical eigenvalues appear at
infinity and are filtered out.

This module implements the HYDRODYNAMIC problem.  The MHD / MRI extension lives
in :mod:`taylor_couette_mri`.
"""

from __future__ import annotations

from _demo_utils import default_thread_cap

default_thread_cap()

import argparse
import math

import numpy as np
import sympy as sp
from _linear_analysis import (
    FINITE_CAP,
    finite_eigensystem,
    parse_times,
    print_transient_growth,
    transient_growth_from_eigs,
)
from scipy.linalg import eig
from shenfun import Dx, FunctionSpace, TestFunction, TrialFunction, inner

# Radial coordinate symbol.  A shenfun FunctionSpace built on ``domain=(R1, R2)``
# uses the sympy symbol ``x`` for its (physical) coordinate, so cylindrical
# coefficients such as 1/r, 1/r**2 are written with ``r = x`` below.
x = sp.Symbol("x", real=True)


# ---------------------------------------------------------------------------
# Base flow
# ---------------------------------------------------------------------------
class CircularCouette:
    """Circular-Couette base flow ``Omega(r) = a + b/r**2`` and derived fields."""

    def __init__(self, R1=1.0, R2=2.0, Omega1=1.0, Omega2=0.0):
        if not (R2 > R1 > 0):
            raise ValueError("require 0 < R1 < R2")
        self.R1, self.R2 = float(R1), float(R2)
        self.Omega1, self.Omega2 = float(Omega1), float(Omega2)
        d2 = self.R2**2 - self.R1**2
        self.a = (self.Omega2 * self.R2**2 - self.Omega1 * self.R1**2) / d2
        self.b = (self.Omega1 - self.Omega2) * self.R1**2 * self.R2**2 / d2
        self.gap = self.R2 - self.R1
        self.eta = self.R1 / self.R2  # radius ratio
        self.mu = self.Omega2 / self.Omega1 if self.Omega1 != 0 else math.inf

        # sympy expressions in the radial symbol x
        self.Omega_sym = self.a + self.b / x**2  # Omega(r)
        self.twoOmega_sym = 2 * self.a + 2 * self.b / x**2  # 2 Omega
        self.rOmega_p_sym = -2 * self.b / x**2  # r dOmega/dr
        # 2 Omega + r dOmega/dr = 2 a   (constant)
        self.shear2a = 2 * self.a
        self.kappa2_sym = 4 * self.a * self.Omega_sym  # epicyclic kappa^2

    # numeric helpers -------------------------------------------------------
    def Omega(self, r):
        return self.a + self.b / np.asarray(r) ** 2

    def V(self, r):
        r = np.asarray(r)
        return self.a * r + self.b / r

    def kappa2(self, r):
        return 4 * self.a * self.Omega(r)

    def rayleigh_stable(self):
        """True if kappa^2 > 0 over the whole gap (inviscid centrifugal stab.)."""
        rr = np.linspace(self.R1, self.R2, 200)
        return bool(np.all(self.kappa2(rr) > 0))

    def q_shear(self, r):
        """Local shear exponent q = -dln Omega/dln r."""
        r = np.asarray(r)
        return 2 * self.b / (self.a * r**2 + self.b)

    def describe(self):
        return (
            f"CircularCouette: R1={self.R1:g} R2={self.R2:g} eta={self.eta:.4f} "
            f"Omega1={self.Omega1:g} Omega2={self.Omega2:g} mu={self.mu:.4f}\n"
            f"  a={self.a:.6g} b={self.b:.6g}  "
            f"Rayleigh-{'stable' if self.rayleigh_stable() else 'UNSTABLE'} "
            f"(kappa^2 at R1={self.kappa2(self.R1):.4g}, R2={self.kappa2(self.R2):.4g})"
        )


# ---------------------------------------------------------------------------
# Linear stability operator
# ---------------------------------------------------------------------------
class TaylorCouetteLinear:
    r"""Assemble and solve the hydrodynamic TC linear-stability eigenproblem.

    Parameters
    ----------
    base : CircularCouette
    nu : float
        Kinematic viscosity.
    N : int
        Number of radial quadrature points / modes.
    family : {'L', 'C'}
        Legendre or Chebyshev radial basis.
    """

    def __init__(self, base: CircularCouette, nu=1.0e-3, N=48, family="C"):
        self.base = base
        self.nu = float(nu)
        self.N = int(N)
        self.family = family
        dom = (base.R1, base.R2)
        # velocity: no-slip Dirichlet; pressure: orthogonal, sliced to N-2
        self.SD = FunctionSpace(N, family, bc=(0, 0), domain=dom)
        self.SP = FunctionSpace(N, family, domain=dom)
        self.SP.slice = lambda: slice(0, N - 2)
        self.n = self.SD.dim()
        assert self.SP.dim() == self.n, (self.SP.dim(), self.n)

        self.vD = TestFunction(self.SD)
        self.uD = TrialFunction(self.SD)
        self.qP = TestFunction(self.SP)
        self.pP = TrialFunction(self.SP)

    # helpers to extract dense blocks --------------------------------------
    def _dense(self, test, expr, shape):
        """Scalar-product (test, expr) -> dense complex block.

        ``inner`` returns a single SpectralMatrix for a one-term integrand but a
        *list* of them when the coefficient has several additive terms (e.g.
        ``Omega = a + b/r**2``); both cases are summed here.  A symbolically
        zero coefficient yields an empty result -> return zeros.
        """
        res = inner(test, expr)
        if isinstance(res, list):
            if len(res) == 0:
                return np.zeros(shape, dtype=complex)
            out = res[0].diags().toarray().astype(complex)
            for r in res[1:]:
                out = out + r.diags().toarray()
            return out
        return res.diags().toarray().astype(complex)

    @staticmethod
    def _zero_coeff(coeff):
        if coeff is None:
            return False
        try:
            return bool(sp.simplify(sp.sympify(coeff)) == 0)
        except (TypeError, ValueError, AttributeError):
            return complex(coeff) == 0

    def _A(self, coeff, order):
        """(v_D, coeff(r) * d^order u_D / dr^order) as a dense (n,n) block."""
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        u = self.uD if order == 0 else Dx(self.uD, 0, order)
        expr = u if coeff is None else coeff * u
        return self._dense(self.vD, expr, (self.n, self.n))

    def _Avp(self, coeff, order):
        """(v_D, coeff * d^order p / dr^order):  velocity-test x pressure-trial."""
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        p = self.pP if order == 0 else Dx(self.pP, 0, order)
        expr = p if coeff is None else coeff * p
        return self._dense(self.vD, expr, (self.n, self.n))

    def _Aqu(self, coeff, order):
        """(q_P, coeff * d^order u / dr^order):  pressure-test x velocity-trial."""
        if self._zero_coeff(coeff):
            return np.zeros((self.n, self.n), dtype=complex)
        u = self.uD if order == 0 else Dx(self.uD, 0, order)
        expr = u if coeff is None else coeff * u
        return self._dense(self.qP, expr, (self.n, self.n))

    def _laplacian_scalar(self, m, kz):
        """Scalar cylindrical Laplacian block: u'' + (1/r)u' - (m^2/r^2 + kz^2) u."""
        return (
            self._A(None, 2)
            + self._A(1 / x, 1)
            - self._A(sp.Integer(m**2) / x**2 + sp.Float(kz**2), 0)
        )

    def assemble_parts(self, m, kz):
        """Build (L0, Lv, M) so that L = L0 + nu*Lv for mode (m, kz).

        Separating the viscous part lets a critical-viscosity bisection reuse a
        single assembly (only the cheap ``eig`` is repeated).
        """
        n = self.n
        b = self.base
        Om = b.Omega_sym
        m_s = sp.Integer(int(m))
        kz_s = sp.Float(float(kz))
        imOm = sp.I * m_s * Om

        Lap = self._laplacian_scalar(m, kz)
        Mvv = self._A(None, 0)  # velocity mass (v, u)

        # vector-Laplacian pieces (coefficients of nu)
        lv_diag = Lap - self._A(1 / x**2, 0)  # acts on u_r and u_theta
        lv_z = Lap  # acts on u_z
        adv = -self._A(imOm, 0)  # -i m Omega (each comp)

        # cross coupling coefficient blocks
        twoOmega = self._A(b.twoOmega_sym, 0)  # 2 Omega
        shear = self._A(sp.Float(b.shear2a), 0)  # 2 a  (=2Omega + r Omega')
        couple_rt = self._A(2 * m_s * sp.I / x**2, 0)  # 2 i m / r^2

        # pressure-gradient blocks (velocity-test x pressure)
        Gr = self._Avp(None, 1)  # dp/dr
        Gt = self._Avp(sp.I * m_s / x, 0)  # i m p / r
        Gz = self._Avp(sp.I * kz_s, 0)  # i kz p

        # continuity blocks (pressure-test x velocity)
        Dr = self._Aqu(None, 1) + self._Aqu(1 / x, 0)  # d/dr + 1/r
        Dt = self._Aqu(sp.I * m_s / x, 0)  # i m / r
        Dz = self._Aqu(sp.I * kz_s, 0)  # i kz

        # block index: 0 u_r, 1 u_theta, 2 u_z, 3 p
        L0 = np.zeros((4 * n, 4 * n), dtype=complex)
        Lv = np.zeros((4 * n, 4 * n), dtype=complex)
        M = np.zeros((4 * n, 4 * n), dtype=complex)

        def put(blk, i, j, val):
            blk[i * n : (i + 1) * n, j * n : (j + 1) * n] = val

        # r-momentum:  s u_r = -imOm u_r + 2Om u_theta - dp/dr + nu(Lap-1/r^2)u_r
        put(L0, 0, 0, adv)
        put(Lv, 0, 0, lv_diag)
        put(L0, 0, 1, twoOmega)
        put(Lv, 0, 1, -couple_rt)
        put(L0, 0, 3, -Gr)
        put(M, 0, 0, Mvv)

        # theta-momentum: s u_theta = -imOm u_theta - 2a u_r - (im/r)p
        #                              + nu(Lap-1/r^2)u_theta + 2 i m/r^2 coupling
        put(L0, 1, 0, -shear)
        put(Lv, 1, 0, couple_rt)
        put(L0, 1, 1, adv)
        put(Lv, 1, 1, lv_diag)
        put(L0, 1, 3, -Gt)
        put(M, 1, 1, Mvv)

        # z-momentum: s u_z = -imOm u_z - i kz p + nu Lap u_z
        put(L0, 2, 2, adv)
        put(Lv, 2, 2, lv_z)
        put(L0, 2, 3, -Gz)
        put(M, 2, 2, Mvv)

        # continuity: 0 = Dr u_r + Dt u_theta + Dz u_z
        put(L0, 3, 0, Dr)
        put(L0, 3, 1, Dt)
        put(L0, 3, 2, Dz)
        # M row 3 stays zero (no time derivative -> algebraic constraint)

        return L0, Lv, M

    def assemble(self, m, kz):
        """Build (L, M) for azimuthal mode m and axial wavenumber kz."""
        L0, Lv, M = self.assemble_parts(m, kz)
        return L0 + self.nu * Lv, M

    # solving ---------------------------------------------------------------
    @staticmethod
    def _leading(L, M, finite_cap=1e6):
        w = eig(L, M, right=False)
        w = w[np.isfinite(w) & (np.abs(w) < finite_cap)]
        return w[np.argsort(-w.real)] if len(w) else w

    def eigs(self, m, kz, n_return=8, finite_cap=1e6):
        """Return (eigenvalues, eigenvectors) sorted by descending Re(s)."""
        L, M = self.assemble(m, kz)
        w, V = eig(L, M)
        good = np.isfinite(w) & (np.abs(w) < finite_cap)
        w, V = w[good], V[:, good]
        order = np.argsort(-w.real)
        w, V = w[order], V[:, order]
        return w[:n_return], V[:, :n_return]

    def growth_rate(self, m, kz):
        """Largest real part of the spectrum (the linear growth rate)."""
        w = self._leading(*self.assemble(m, kz))
        return float(w[0].real) if len(w) else float("nan")

    def energy_matrix(self):
        """Cylindrical kinetic-energy metric for ``(u_r,u_theta,u_z,p)``."""
        n = self.n
        Q = np.zeros((4 * n, 4 * n), dtype=complex)
        W = self._A(x, 0)
        W = 0.5 * (W + W.conj().T)
        for comp in range(3):
            Q[comp * n : (comp + 1) * n, comp * n : (comp + 1) * n] = W
        return Q

    def nonmodal_growth(self, m, kz, times, n_modes=None, finite_cap=FINITE_CAP):
        """Optimal linear transient growth in kinetic-energy norm."""
        w, V = finite_eigensystem(
            *self.assemble(m, kz), finite_cap=finite_cap, n_return=n_modes
        )
        return transient_growth_from_eigs(w, V, self.energy_matrix(), times)

    def max_growth_over_kz(self, m, kz_list):
        """Scan kz; return (kz_best, growth_best, all_growths)."""
        g = np.array([self.growth_rate(m, kz) for kz in kz_list])
        i = int(np.argmax(g))
        return float(kz_list[i]), float(g[i]), g

    # critical-viscosity / Reynolds tools (reuse one assembly) --------------
    def critical_nu(self, m, kz, nu_lo=1e-5, nu_hi=2.0, iters=34):
        """Largest nu still (marginally) unstable at (m, kz), else None.

        Growth decreases with nu, so bisect the sign of the leading growth
        rate.  One assembly is shared across the whole bisection.
        """
        L0, Lv, M = self.assemble_parts(m, kz)

        def growth(nu):
            w = self._leading(L0 + nu * Lv, M)
            return w[0].real if len(w) else -np.inf

        if growth(nu_lo) < 0:
            return None  # stable even at the smallest nu
        if growth(nu_hi) > 0:
            return nu_hi  # unstable even at the largest nu
        for _ in range(iters):
            nm = 0.5 * (nu_lo + nu_hi)
            if growth(nm) > 0:
                nu_lo = nm
            else:
                nu_hi = nm
        return 0.5 * (nu_lo + nu_hi)

    def critical_over_kz(self, m, kz_list, **kw):
        """Return (kz_c, nu_c) at the global critical point (max nu over kz)."""
        best = None
        for kz in kz_list:
            nu_c = self.critical_nu(m, kz, **kw)
            if nu_c is None:
                continue
            if best is None or nu_c > best[1]:
                best = (float(kz), float(nu_c))
        return best

    def critical_reynolds(self, m=0, kz_list=None, refine=True):
        """Critical inner Reynolds number Re_c = Omega1 R1 d / nu_c and a_c=kz_c*d.

        Returns dict with kz_c, nu_c, Re_c, a_c, or None if no critical point
        is found in the supplied viscosity and kz search ranges.
        """
        b = self.base
        if kz_list is None:
            kz_list = np.linspace(2.0, 4.5, 16) / b.gap
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
        Re_c = b.Omega1 * b.R1 * b.gap / nu_c
        return {"kz_c": kz_c, "nu_c": nu_c, "Re_c": Re_c, "a_c": kz_c * b.gap}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build(args):
    base = CircularCouette(args.R1, args.R2, args.Omega1, args.Omega2)
    solver = TaylorCouetteLinear(base, nu=args.nu, N=args.N, family=args.family)
    return base, solver


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Taylor-Couette hydrodynamic linear stability"
    )
    p.add_argument("--R1", type=float, default=1.0)
    p.add_argument("--R2", type=float, default=2.0)
    p.add_argument("--Omega1", type=float, default=1.0)
    p.add_argument("--Omega2", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=2.0e-3)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--family", choices=["L", "C"], default="C")
    p.add_argument("--m", type=int, default=0)
    p.add_argument(
        "--kz", type=float, default=None, help="axial wavenumber; if unset, scan"
    )
    p.add_argument("--kz-min", type=float, default=0.5)
    p.add_argument("--kz-max", type=float, default=8.0)
    p.add_argument("--kz-num", type=int, default=40)
    p.add_argument(
        "--nonmodal",
        action="store_true",
        help="compute optimal transient growth instead of an eigenvalue scan",
    )
    p.add_argument(
        "--times",
        type=str,
        default="1,5,10,20",
        help="comma-separated times for --nonmodal",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=None,
        help="number of finite eigenmodes retained for --nonmodal",
    )
    args = p.parse_args(argv)

    base, solver = _build(args)
    print(base.describe())
    print(f"  nu={args.nu:g}  N={args.N}  family={args.family}  m={args.m}")

    if args.nonmodal:
        kz = args.kz if args.kz is not None else 3.14 / base.gap
        rows = solver.nonmodal_growth(
            args.m, kz, parse_times(args.times), n_modes=args.n_modes
        )
        print(f"\nkz={kz:g}: hydrodynamic non-modal transient growth (energy norm):")
        print_transient_growth(rows)
        return 0

    if args.kz is not None:
        w, _ = solver.eigs(args.m, args.kz, n_return=6)
        print(f"\nkz={args.kz:g}: leading eigenvalues (growth, freq):")
        for s in w:
            print(f"   s = {s.real:+.6e}  {s.imag:+.6e} i")
        print(f"  growth rate = {w[0].real:+.6e}")
    else:
        kzs = np.linspace(args.kz_min, args.kz_max, args.kz_num)
        kb, gb, g = solver.max_growth_over_kz(args.m, kzs)
        print(f"\nkz scan [{args.kz_min},{args.kz_max}] (m={args.m}):")
        print(f"  most unstable kz = {kb:.4f}  growth = {gb:+.6e}")
        print(
            f"  -> base flow is {'UNSTABLE' if gb > 1e-9 else 'stable'} at nu={args.nu:g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
