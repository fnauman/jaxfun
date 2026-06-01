r"""
MHD linear stability of Taylor-Couette flow: the standard magnetorotational
instability (MRI) with an imposed uniform axial field, in the annulus.

This extends :mod:`taylor_couette_linear` (hydrodynamic circular-Couette
stability) to incompressible, resistive, viscous MHD with a uniform background
field ``B0 = B0_z e_z``.  Magnetic field is measured in Alfven-speed units
(``b -> b/sqrt(mu0 rho)``), so ``v_A = B0_z``.  Perturbations
``~ exp(s t + i m theta + i kz z)`` are expanded with a 1D Chebyshev/Legendre
radial basis on ``[R1, R2]`` (the same OrrSommerfeld strong-form pattern as the
hydrodynamic solver), and the coupled generalised eigenvalue problem

    L q = s M q ,      q = (u_r, u_theta, u_z, Pi, b_r, b_theta, b_z)

is solved with ``scipy.linalg.eig``.  ``Pi`` is the total (gas + magnetic)
pressure; using it absorbs ``grad(B0 b_z)`` so the imposed-field Lorentz force is
simply ``i kz B0 b`` per component.

Linearised equations (Alfven units, base flow ``U = r Omega(r) e_theta``)::

    s u_r     = -i m Om u_r + 2 Om u_theta            - dPi/dr        + nu Lv[u_r ] + i kz B0 b_r
    s u_theta = -i m Om u_theta - 2a u_r              - (i m/r) Pi    + nu Lv[u_th] + i kz B0 b_theta
    s u_z     = -i m Om u_z                           - i kz Pi       + nu Lp[u_z ] + i kz B0 b_z
        0     = (d/dr + 1/r) u_r + (i m/r) u_theta + i kz u_z                       (continuity)
    s b_r     = i kz B0 u_r       - i m Om b_r                         + eta Lv[b_r]
    s b_theta = i kz B0 u_theta   + r Om' b_r - i m Om b_theta         + eta Lv[b_th]
    s b_z     = i kz B0 u_z       - i m Om b_z                         + eta Lp[b_z]

with ``2a = 2 Om + r Om'`` (constant), ``r Om' = -2 b/r^2``, scalar Laplacian
``Lp = d^2/dr^2 + (1/r) d/dr - (m^2/r^2 + kz^2)`` and the vector-Laplacian
diagonal piece ``Lv = Lp - 1/r^2`` (the ``+- 2 i m/r^2`` r/theta cross terms are
included).  The radial-field induction has no shear source (it is only advected);
the azimuthal field is generated from ``b_r`` at rate ``r Om' = r dOmega/dr`` -
the MRI field-stretching term.

Perfectly-conducting walls (axisymmetric form, Rudiger/Hollerbach)::

    b_r = 0 ,   d(r b_theta)/dr = 0  (=> b_theta + r b_theta' = 0) ,   b_z' = 0

implemented as Dirichlet / Robin / Neumann radial bases respectively.

Insulating (vacuum) walls match the interior field to a current-free exterior
(modified-Bessel ``I_m`` inside, ``K_m`` outside).  For ``m = 0`` this is handled
with a **poloidal flux function** ``chi`` (``b_r = -(i kz/r) chi``,
``b_z = (1/r) chi'``) so ``div(b) = 0`` is built in and the vacuum match becomes a
single-field Robin ``chi' = (kz**2/kappa) chi`` (``kappa`` the exterior potential's
log-derivative); see :meth:`TaylorCouetteMRI._assemble_flux_parts`.

Validation targets
------------------
* Ideal local WKB MRI (Keplerian q=3/2): ``s_max = (3/4) Omega`` at
  ``(kz v_A)^2 = (15/16) Omega^2``; cutoff ``(kz v_A)^2 = 3 Omega^2``
  (Balbus & Hawley 1991).  See :func:`mri_local_growth`.
* The Keplerian profile is Rayleigh-stable (hydro null) yet MRI-unstable.
* Global resistive onset, eta=0.5 quasi-Keplerian (Rudiger et al. 2023):
  conducting walls ``Rm_min ~ 24.7`` (``S=4.11``), insulating ``Rm_min ~ 16.5``
  (``S=5.21``) as ``Pm -> 0`` -- insulating destabilises more easily.
"""
from __future__ import annotations

from _demo_utils import default_thread_cap


default_thread_cap()

import argparse
import math

import numpy as np
import sympy as sp
from scipy.linalg import eig
from scipy.special import iv, kv

from shenfun import FunctionSpace, TestFunction, TrialFunction, inner, Dx

from taylor_couette_linear import CircularCouette, x


# ---------------------------------------------------------------------------
# Local (WKB) MRI dispersion relation -- analytic reference
# ---------------------------------------------------------------------------
def mri_local_growth(omega_A, Omega, kappa2, dOmega2_dlnr):
    r"""Max growth rate of the ideal axisymmetric MRI dispersion relation

        s^4 + 2 s^2 (omega_A^2 + kappa^2/2) + omega_A^2 (omega_A^2 + dOmega2_dlnr) = 0

    Returns the largest real part over the (bi-quadratic) roots (0 if stable).
    """
    A = omega_A**2 + 0.5 * kappa2
    C = omega_A**2 * (omega_A**2 + dOmega2_dlnr)
    disc = A**2 - C
    if disc < 0:
        return 0.0                      # complex s^2 -> oscillatory, |Re s| from sqrt
    s2 = -A + math.sqrt(disc)           # the larger root of s^2
    return math.sqrt(s2) if s2 > 0 else 0.0


def mri_keplerian_optimum(Omega=1.0, vA=1.0):
    """Scan the ideal dispersion for a Keplerian profile; return dict with
    s_max, the optimal omega_A^2/Omega^2, and the cutoff, vs theory
    (0.75 Omega, 0.9375, 3.0)."""
    q = 1.5
    kappa2 = (4 - 2 * q) * Omega**2          # = Omega^2 for q=3/2
    dOmega2_dlnr = -2 * q * Omega**2         # = -3 Omega^2
    wa = np.linspace(1e-3, math.sqrt(3.0) * Omega * 0.999, 4000)
    s = np.array([mri_local_growth(w, Omega, kappa2, dOmega2_dlnr) for w in wa])
    i = int(np.argmax(s))
    return {
        "s_max": float(s[i]),
        "s_max_over_Omega": float(s[i] / Omega),
        "wa2_opt_over_O2": float((wa[i] / Omega) ** 2),
        "theory_s_max_over_Omega": 0.75,
        "theory_wa2_opt": 15.0 / 16.0,
        "theory_cutoff_wa2": 3.0,
    }


# ---------------------------------------------------------------------------
# Global MHD eigenvalue solver in the annulus
# ---------------------------------------------------------------------------
class TaylorCouetteMRI:
    r"""Resistive, viscous MHD Taylor-Couette linear-stability eigenproblem.

    Parameters
    ----------
    base : CircularCouette
    B0 : float
        Imposed uniform axial field in Alfven-speed units (v_A = B0).
    nu : float
        Kinematic viscosity.
    eta_mag : float
        Magnetic diffusivity (resistivity).  Magnetic Prandtl Pm = nu/eta_mag.
    N : int
    family : {'L','C'}
    magnetic_bc : {'conducting', 'insulating'}
        Perfectly-conducting walls (any ``m``, primitive ``b_r,b_theta,b_z``
        formulation) or insulating / vacuum-matching walls (axisymmetric ``m = 0``
        only).  Insulating walls use a **poloidal flux-function** formulation
        ``b_r = -(i kz/r) chi``, ``b_z = (1/r) chi'`` so ``div(b) = 0`` holds by
        construction and the vacuum match is a single-field Robin on ``chi``
        (the primitive form's per-component vacuum BCs are coupled and would
        leave ``div(b) ~ O(1)``); see :meth:`_assemble_flux_parts`.
    """

    def __init__(self, base: CircularCouette, B0=0.1, nu=1e-3, eta_mag=1e-3,
                 N=48, family="L", magnetic_bc="conducting"):
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.N = int(N)
        self.family = family
        self.magnetic_bc = magnetic_bc
        dom = (base.R1, base.R2)
        if magnetic_bc not in ("conducting", "insulating"):
            raise NotImplementedError(f"magnetic_bc={magnetic_bc!r} not implemented")

        # velocity (no-slip) and radial magnetic field (b_r=0): Dirichlet
        self.SDv = FunctionSpace(N, family, bc=(0, 0), domain=dom)
        # pressure: orthogonal, sliced to N-2 (inf-sup-stable P_N - P_{N-2})
        self.SP = FunctionSpace(N, family, domain=dom)
        self.SP.slice = lambda: slice(0, N - 2)
        self.Jm = 0.5 * (base.R2 - base.R1)            # dr_phys/dr_ref (linear map)
        self.n = self.SDv.dim()
        assert self.SP.dim() == self.n, (self.SP.dim(), self.n)

        # velocity/pressure test+trial (always).  The CONDUCTING (primitive
        # 7-field) path also caches the magnetic b_r/b_theta/b_z bases here; the
        # INSULATING (flux-function) path builds its kz-dependent chi/b_theta bases
        # per assembly in ``_assemble_flux_parts``.
        spaces = dict(ur=self.SDv, ut=self.SDv, uz=self.SDv, p=self.SP)
        if magnetic_bc == "conducting":
            # b_theta: Robin d(r b_theta)/dr=0 <=> b_theta + r*b_theta'_phys=0.
            # shenfun Robin {'R':(c,d)} is u + c*u'_REF = d (reference-coord
            # derivative), so c = r_wall / J, J = (R2-R1)/2.  b_z: Neumann b_z'=0.
            self.Sbt = FunctionSpace(
                N, family, domain=dom,
                bc={"left": {"R": (base.R1 / self.Jm, 0)},
                    "right": {"R": (base.R2 / self.Jm, 0)}})
            self.Sbz = FunctionSpace(
                N, family, bc={"left": {"N": 0}, "right": {"N": 0}}, domain=dom)
            assert self.Sbt.dim() == self.n and self.Sbz.dim() == self.n
            spaces.update(br=self.SDv, bt=self.Sbt, bz=self.Sbz)
        self.tv = {s: TestFunction(sp_) for s, sp_ in spaces.items()}
        self.tr = {s: TrialFunction(sp_) for s, sp_ in spaces.items()}

        # Pm and dimensionless control numbers (Liu/Goodman/Ji convention)
        self.Pm = self.nu / self.eta_mag if self.eta_mag > 0 else math.inf
        d, R1, O1 = base.gap, base.R1, base.Omega1
        self.Re = O1 * R1 * d / self.nu if self.nu > 0 else math.inf
        self.Rm = O1 * R1 * d / self.eta_mag if self.eta_mag > 0 else math.inf
        self.S = self.B0 * d / self.eta_mag if self.eta_mag > 0 else math.inf   # Lundquist
        self.Ha = self.B0 * d / math.sqrt(self.nu * self.eta_mag) if self.nu * self.eta_mag > 0 else math.inf

    # ---- generic dense block (test, trial, [(coeff, order), ...]) ----------
    def _blk(self, test, trial, terms):
        n = self.n
        out = np.zeros((n, n), dtype=complex)
        for coeff, order in terms:
            if coeff is not None:
                try:
                    if bool(sp.simplify(sp.sympify(coeff)) == 0):
                        continue
                except (TypeError, ValueError, AttributeError):
                    if complex(coeff) == 0:
                        continue
            t = trial if order == 0 else Dx(trial, 0, order)
            expr = t if coeff is None else coeff * t
            res = inner(test, expr)
            if isinstance(res, list):
                for r in res:
                    out = out + r.diags().toarray()
            else:
                out = out + res.diags().toarray()
        return out

    def _lap_terms(self, m, kz):
        """Scalar cylindrical Laplacian terms."""
        return [(None, 2), (1 / x, 1),
                (-(sp.Integer(m**2) / x**2 + sp.Float(kz**2)), 0)]

    def assemble_parts(self, m, kz, B0=None):
        """Return (L0, Lnu, Leta, M) with L = L0 + nu*Lnu + eta_mag*Leta.

        Conducting walls use the primitive 7-field (u_r,u_theta,u_z,Pi,b_r,
        b_theta,b_z) system (any ``m``); insulating walls use the 6-field
        flux-function (u_r,u_theta,u_z,Pi,chi,b_theta) system (``m = 0`` only,
        :meth:`_assemble_flux_parts`).
        """
        if self.magnetic_bc == "insulating":
            if int(m) != 0:
                raise NotImplementedError(
                    "insulating walls are implemented for the axisymmetric m=0 "
                    "MRI only (m!=0 couples poloidal and toroidal scalars at the "
                    "wall via the vacuum field)")
            return self._assemble_flux_parts(kz, B0)
        n = self.n
        b = self.base
        Om = b.Omega_sym
        m_s = sp.Integer(int(m))
        kz_s = sp.Float(float(kz))
        imOm = sp.I * m_s * Om
        B0 = self.B0 if B0 is None else float(B0)
        ikzB0 = sp.I * kz_s * sp.Float(B0)

        lap = self._lap_terms(m, kz)
        lv = lap + [(-1 / x**2, 0)]                  # vector-Laplacian diag piece
        couple = [(2 * m_s * sp.I / x**2, 0)]        # +- 2 i m / r^2

        L0 = np.zeros((7 * n, 7 * n), dtype=complex)
        Lnu = np.zeros((7 * n, 7 * n), dtype=complex)
        Leta = np.zeros((7 * n, 7 * n), dtype=complex)
        M = np.zeros((7 * n, 7 * n), dtype=complex)
        idx = dict(ur=0, ut=1, uz=2, p=3, br=4, bt=5, bz=6)

        def put(blk, ri, ci, val):
            blk[ri * n:(ri + 1) * n, ci * n:(ci + 1) * n] += val

        tv, tr = self.tv, self.tr

        # ----- r-momentum (test ur) -----
        put(L0, idx["ur"], idx["ur"], self._blk(tv["ur"], tr["ur"], [(-imOm, 0)]))
        put(Lnu, idx["ur"], idx["ur"], self._blk(tv["ur"], tr["ur"], lv))
        put(L0, idx["ur"], idx["ut"], self._blk(tv["ur"], tr["ut"], [(b.twoOmega_sym, 0)]))
        put(Lnu, idx["ur"], idx["ut"], -self._blk(tv["ur"], tr["ut"], couple))
        put(L0, idx["ur"], idx["p"], -self._blk(tv["ur"], tr["p"], [(None, 1)]))
        put(L0, idx["ur"], idx["br"], self._blk(tv["ur"], tr["br"], [(ikzB0, 0)]))
        put(M, idx["ur"], idx["ur"], self._blk(tv["ur"], tr["ur"], [(None, 0)]))

        # ----- theta-momentum (test ut) -----
        put(L0, idx["ut"], idx["ur"], self._blk(tv["ut"], tr["ur"], [(-sp.Float(b.shear2a), 0)]))
        put(Lnu, idx["ut"], idx["ur"], self._blk(tv["ut"], tr["ur"], couple))
        put(L0, idx["ut"], idx["ut"], self._blk(tv["ut"], tr["ut"], [(-imOm, 0)]))
        put(Lnu, idx["ut"], idx["ut"], self._blk(tv["ut"], tr["ut"], lv))
        put(L0, idx["ut"], idx["p"], -self._blk(tv["ut"], tr["p"], [(sp.I * m_s / x, 0)]))
        put(L0, idx["ut"], idx["bt"], self._blk(tv["ut"], tr["bt"], [(ikzB0, 0)]))
        put(M, idx["ut"], idx["ut"], self._blk(tv["ut"], tr["ut"], [(None, 0)]))

        # ----- z-momentum (test uz) -----
        put(L0, idx["uz"], idx["uz"], self._blk(tv["uz"], tr["uz"], [(-imOm, 0)]))
        put(Lnu, idx["uz"], idx["uz"], self._blk(tv["uz"], tr["uz"], lap))
        put(L0, idx["uz"], idx["p"], -self._blk(tv["uz"], tr["p"], [(sp.I * kz_s, 0)]))
        put(L0, idx["uz"], idx["bz"], self._blk(tv["uz"], tr["bz"], [(ikzB0, 0)]))
        put(M, idx["uz"], idx["uz"], self._blk(tv["uz"], tr["uz"], [(None, 0)]))

        # ----- continuity (test p) -----
        put(L0, idx["p"], idx["ur"], self._blk(tv["p"], tr["ur"], [(None, 1), (1 / x, 0)]))
        put(L0, idx["p"], idx["ut"], self._blk(tv["p"], tr["ut"], [(sp.I * m_s / x, 0)]))
        put(L0, idx["p"], idx["uz"], self._blk(tv["p"], tr["uz"], [(sp.I * kz_s, 0)]))
        # M row p stays zero (constraint)

        # ----- b_r induction (test br) -----
        put(L0, idx["br"], idx["ur"], self._blk(tv["br"], tr["ur"], [(ikzB0, 0)]))
        put(L0, idx["br"], idx["br"], self._blk(tv["br"], tr["br"], [(-imOm, 0)]))
        put(Leta, idx["br"], idx["br"], self._blk(tv["br"], tr["br"], lv))
        put(Leta, idx["br"], idx["bt"], -self._blk(tv["br"], tr["bt"], couple))
        put(M, idx["br"], idx["br"], self._blk(tv["br"], tr["br"], [(None, 0)]))

        # ----- b_theta induction (test bt) -----
        put(L0, idx["bt"], idx["ut"], self._blk(tv["bt"], tr["ut"], [(ikzB0, 0)]))
        put(L0, idx["bt"], idx["br"], self._blk(tv["bt"], tr["br"], [(b.rOmega_p_sym, 0)]))
        put(Leta, idx["bt"], idx["br"], self._blk(tv["bt"], tr["br"], couple))
        put(L0, idx["bt"], idx["bt"], self._blk(tv["bt"], tr["bt"], [(-imOm, 0)]))
        put(Leta, idx["bt"], idx["bt"], self._blk(tv["bt"], tr["bt"], lv))
        put(M, idx["bt"], idx["bt"], self._blk(tv["bt"], tr["bt"], [(None, 0)]))

        # ----- b_z induction (test bz) -----
        put(L0, idx["bz"], idx["uz"], self._blk(tv["bz"], tr["uz"], [(ikzB0, 0)]))
        put(L0, idx["bz"], idx["bz"], self._blk(tv["bz"], tr["bz"], [(-imOm, 0)]))
        put(Leta, idx["bz"], idx["bz"], self._blk(tv["bz"], tr["bz"], lap))
        put(M, idx["bz"], idx["bz"], self._blk(tv["bz"], tr["bz"], [(None, 0)]))

        return L0, Lnu, Leta, M

    # ---- insulating walls: poloidal flux-function formulation (m=0) ---------
    def _flux_bases(self, kz):
        r"""``(Schi, Sbth)`` radial bases for the m=0 flux-function system.

        Poloidal flux ``chi`` (``b_r = -(i kz/r) chi``, ``b_z = (1/r) chi'``):
          * conducting wall ``b_r = 0`` -> ``chi = 0`` (Dirichlet); the chi
            equation at the wall then forces ``chi'' = chi'/r`` i.e. ``b_z' = 0``
            automatically, so a single Dirichlet condition captures the perfect
            conductor;
          * insulating wall -> ``chi'/chi = kz**2/kappa`` (Robin), where
            ``kappa = psi'/psi`` is the modified-Bessel log-derivative of the
            exterior potential (``I_0`` inside, ``K_0`` outside).
        Toroidal ``b_theta``: conducting Robin ``d(r b_theta)/dr = 0`` /
        insulating Dirichlet ``b_theta = 0`` (vacuum toroidal field vanishes).

        The Robin coefficients depend only on ``kz`` (not on ``B0``/``eta_mag``),
        so the bases are cached by ``kz`` -- ``critical_Rm`` re-evaluates the same
        ``kz`` many times during the ``B0``/``eta_mag`` bisection.
        """
        key = round(float(kz), 12)
        cache = self.__dict__.setdefault("_flux_basis_cache", {})
        if key in cache:
            return cache[key]
        N, fam, J, b = self.N, self.family, self.Jm, self.base
        dom = (b.R1, b.R2)
        if self.magnetic_bc == "conducting":
            Schi = FunctionSpace(N, fam, bc=(0, 0), domain=dom)          # chi=0
            Sbth = FunctionSpace(N, fam, domain=dom,
                                 bc={"left": {"R": (b.R1 / J, 0)},
                                     "right": {"R": (b.R2 / J, 0)}})
        else:                                                            # insulating
            k = abs(float(kz))
            if k < 1e-12:
                raise ValueError("insulating BCs require kz != 0")
            kap_in = k * iv(1, k * b.R1) / iv(0, k * b.R1)               # I_0'=I_1
            kap_out = -k * kv(1, k * b.R2) / kv(0, k * b.R2)             # K_0'=-K_1
            # chi'_phys = (kz^2/kappa) chi -> shenfun Robin c = -kappa/(kz^2 J)
            Schi = FunctionSpace(N, fam, domain=dom,
                                 bc={"left": {"R": (-kap_in / (k * k * J), 0)},
                                     "right": {"R": (-kap_out / (k * k * J), 0)}})
            Sbth = FunctionSpace(N, fam, bc=(0, 0), domain=dom)          # b_theta=0
        assert Schi.dim() == self.n and Sbth.dim() == self.n
        cache[key] = (Schi, Sbth)
        return cache[key]

    def _assemble_flux_parts(self, kz, B0=None):
        r"""(L0, Lnu, Leta, M) for the m=0 flux-function MHD system.

        Variables ``(u_r, u_theta, u_z, Pi, chi, b_theta)``.  div(b)=0 is built
        into the ``chi`` poloidal representation, so the insulating vacuum match
        is a single-field Robin on ``chi`` (see :meth:`_flux_bases`).  Equations
        (Alfven units, total pressure ``Pi``)::

            s u_r = 2 Om u_t - dPi/dr + nu Lv u_r + (kz^2 B0/r) chi
            s u_t = -2a u_r          + nu Lv u_t + i kz B0 b_t
            s u_z = -i kz Pi         + nu Lp u_z + (i kz B0/r) chi'
              0   = u_r' + u_r/r + i kz u_z
            s chi = -B0 r u_r + eta Lchi chi                 (Lchi = d2 - (1/r)d - kz^2)
            s b_t = i kz B0 u_t - i kz Om' chi + eta Lv b_t
        """
        n = self.n
        b = self.base
        B0 = self.B0 if B0 is None else float(B0)
        kz_s = sp.Float(float(kz))
        ikzB0 = sp.I * kz_s * sp.Float(B0)
        Schi, Sbth = self._flux_bases(kz)
        sp_map = dict(ur=self.SDv, ut=self.SDv, uz=self.SDv,
                      p=self.SP, chi=Schi, bt=Sbth)
        tv = {s: TestFunction(v) for s, v in sp_map.items()}
        tr = {s: TrialFunction(v) for s, v in sp_map.items()}
        idx = dict(ur=0, ut=1, uz=2, p=3, chi=4, bt=5)

        L0 = np.zeros((6 * n, 6 * n), dtype=complex)
        Lnu = np.zeros_like(L0)
        Leta = np.zeros_like(L0)
        M = np.zeros_like(L0)

        def put(blk, ri, ci, val):
            blk[idx[ri]*n:(idx[ri]+1)*n, idx[ci]*n:(idx[ci]+1)*n] += val

        Lp = [(None, 2), (1 / x, 1), (-(kz_s**2), 0)]      # scalar Laplacian (m=0)
        Lv = Lp + [(-1 / x**2, 0)]                          # vector-Laplacian diag
        Lchi = [(None, 2), (-1 / x, 1), (-(kz_s**2), 0)]    # Stokes operator for chi

        # r-momentum
        put(L0, "ur", "ut", self._blk(tv["ur"], tr["ut"], [(b.twoOmega_sym, 0)]))
        put(L0, "ur", "p", -self._blk(tv["ur"], tr["p"], [(None, 1)]))
        put(Lnu, "ur", "ur", self._blk(tv["ur"], tr["ur"], Lv))
        put(L0, "ur", "chi", self._blk(tv["ur"], tr["chi"], [(kz_s**2 * sp.Float(B0) / x, 0)]))
        put(M, "ur", "ur", self._blk(tv["ur"], tr["ur"], [(None, 0)]))
        # theta-momentum
        put(L0, "ut", "ur", self._blk(tv["ut"], tr["ur"], [(-sp.Float(b.shear2a), 0)]))
        put(Lnu, "ut", "ut", self._blk(tv["ut"], tr["ut"], Lv))
        put(L0, "ut", "bt", self._blk(tv["ut"], tr["bt"], [(ikzB0, 0)]))
        put(M, "ut", "ut", self._blk(tv["ut"], tr["ut"], [(None, 0)]))
        # z-momentum
        put(L0, "uz", "p", -self._blk(tv["uz"], tr["p"], [(sp.I * kz_s, 0)]))
        put(Lnu, "uz", "uz", self._blk(tv["uz"], tr["uz"], Lp))
        put(L0, "uz", "chi", self._blk(tv["uz"], tr["chi"], [(ikzB0 / x, 1)]))
        put(M, "uz", "uz", self._blk(tv["uz"], tr["uz"], [(None, 0)]))
        # continuity (test Pi)
        put(L0, "p", "ur", self._blk(tv["p"], tr["ur"], [(None, 1), (1 / x, 0)]))
        put(L0, "p", "uz", self._blk(tv["p"], tr["uz"], [(sp.I * kz_s, 0)]))
        # poloidal flux chi:  s chi = -B0 r u_r + eta Lchi chi
        put(L0, "chi", "ur", self._blk(tv["chi"], tr["ur"], [(-sp.Float(B0) * x, 0)]))
        put(Leta, "chi", "chi", self._blk(tv["chi"], tr["chi"], Lchi))
        put(M, "chi", "chi", self._blk(tv["chi"], tr["chi"], [(None, 0)]))
        # toroidal b_theta:  s b_t = i kz B0 u_t - i kz Om' chi + eta Lv b_t
        put(L0, "bt", "ut", self._blk(tv["bt"], tr["ut"], [(ikzB0, 0)]))
        put(L0, "bt", "chi", self._blk(tv["bt"], tr["chi"], [(-sp.I * kz_s * b.rOmega_p_sym / x, 0)]))
        put(Leta, "bt", "bt", self._blk(tv["bt"], tr["bt"], Lv))
        put(M, "bt", "bt", self._blk(tv["bt"], tr["bt"], [(None, 0)]))
        return L0, Lnu, Leta, M

    def assemble(self, m, kz):
        L0, Lnu, Leta, M = self.assemble_parts(m, kz)
        return L0 + self.nu * Lnu + self.eta_mag * Leta, M

    # ---- solving ----------------------------------------------------------
    @staticmethod
    def _spectrum(L, M, finite_cap=1e6):
        w = eig(L, M, right=False)
        w = w[np.isfinite(w) & (np.abs(w) < finite_cap)]
        return w[np.argsort(-w.real)] if len(w) else w

    def eigs(self, m, kz, n_return=8):
        r"""Leading ``n_return`` eigenpairs ``(w, V)``, sorted by decreasing growth.

        The eigenvector layout depends on ``magnetic_bc`` -- callers that slice
        ``V`` by physical-field block must branch on it:

        * ``conducting``: ``V`` has ``7*n`` rows, blocks
          ``(u_r, u_theta, u_z, Pi, b_r, b_theta, b_z)`` at offsets ``0..6 * n``;
        * ``insulating``: ``V`` has ``6*n`` rows, the flux-function blocks
          ``(u_r, u_theta, u_z, Pi, chi, b_theta)`` (``b_r = -(i kz/r) chi``,
          ``b_z = (1/r) chi'``), so e.g. ``V[4*n:5*n]`` is ``chi``, not ``b_r``.

        ``n = self.n`` (``= N - 2``).
        """
        L, M = self.assemble(m, kz)
        w, V = eig(L, M)
        good = np.isfinite(w) & (np.abs(w) < 1e6)
        w, V = w[good], V[:, good]
        order = np.argsort(-w.real)
        return w[order][:n_return], V[:, order][:, :n_return]

    def growth_rate(self, m, kz):
        w = self._spectrum(*self.assemble(m, kz))
        return float(w[0].real) if len(w) else float("nan")

    def max_growth_over_kz(self, m, kz_list):
        g = np.array([self.growth_rate(m, kz) for kz in kz_list])
        i = int(np.argmax(g))
        return float(kz_list[i]), float(g[i]), g

    def critical_eta_mag(self, m, kz, lo=1e-5, hi=1.0, iters=34):
        """Largest eta_mag still unstable at fixed dimensional ``B0`` and ``nu``.

        This is useful for ad hoc scans, but it is not a fixed-``Pm`` /
        fixed-Lundquist-number MRI threshold because ``Pm=nu/eta_mag`` and
        ``S=B0*d/eta_mag`` both change during the bisection.
        """
        L0, Lnu, Leta, M = self.assemble_parts(m, kz)
        L0nu = L0 + self.nu * Lnu

        def growth(eta):
            w = self._spectrum(L0nu + eta * Leta, M)
            return w[0].real if len(w) else -np.inf

        if growth(lo) < 0:
            return None
        if growth(hi) > 0:
            return hi
        for _ in range(iters):
            mid = math.sqrt(lo * hi)          # geometric bisection (Rm spans decades)
            if growth(mid) > 0:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def critical_Rm_fixed_B0_nu(self, m=0, kz_list=None, **kw):
        """Critical Rm along the fixed dimensional ``B0``/``nu`` scan."""
        b = self.base
        if kz_list is None:
            kz_list = np.linspace(1.0, 8.0, 18) / b.gap
        best = None   # maximize eta_mag (minimize Rm)
        for kz in kz_list:
            ec = self.critical_eta_mag(m, kz, **kw)
            if ec is None:
                continue
            if best is None or ec > best[1]:
                best = (float(kz), float(ec))
        if best is None:
            return None
        kz_c, eta_c = best
        Rm_c = b.Omega1 * b.R1 * b.gap / eta_c
        return {"kz_c": kz_c, "eta_mag_c": eta_c, "Rm_c": Rm_c,
                "S_c": self.B0 * b.gap / eta_c}

    def critical_eta_mag_fixed_controls(self, m, kz, Pm=None, S=None,
                                        lo=1e-5, hi=1.0, iters=34):
        """Largest eta_mag still unstable at fixed ``Pm`` and Lundquist ``S``."""
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
            w = self._spectrum(L0 + nu * Lnu + eta * Leta, M)
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
        """Critical magnetic Reynolds number at fixed ``Pm`` and Lundquist ``S``.

        This is the parameter path used for standard MRI threshold comparisons:
        ``eta_mag`` is varied while ``nu = Pm*eta_mag`` and
        ``B0 = S*eta_mag/d`` are updated consistently.
        """
        b = self.base
        if kz_list is None:
            kz_list = np.linspace(1.0, 8.0, 18) / b.gap
        Pm = self.Pm if Pm is None else float(Pm)
        S = self.S if S is None else float(S)
        best = None
        for kz in kz_list:
            ec = self.critical_eta_mag_fixed_controls(m, kz, Pm=Pm, S=S, **kw)
            if ec is None:
                continue
            if best is None or ec > best[1]:
                best = (float(kz), float(ec))
        if best is None:
            return None
        kz_c, eta_c = best
        Rm_c = b.Omega1 * b.R1 * b.gap / eta_c
        nu_c = Pm * eta_c
        B0_c = S * eta_c / b.gap
        return {"kz_c": kz_c, "eta_mag_c": eta_c, "nu_c": nu_c,
                "B0_c": B0_c, "Rm_c": Rm_c, "Pm": Pm, "S_c": S}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Taylor-Couette MHD / MRI linear stability")
    p.add_argument("--R1", type=float, default=1.0)
    p.add_argument("--R2", type=float, default=2.0)
    p.add_argument("--Omega1", type=float, default=1.0)
    p.add_argument("--Omega2", type=float, default=None,
                   help="outer rotation; default = Keplerian mu=eta^1.5")
    p.add_argument("--B0", type=float, default=0.1)
    p.add_argument("--nu", type=float, default=1e-3)
    p.add_argument("--eta-mag", type=float, default=1e-3)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--family", choices=["L", "C"], default="L")
    p.add_argument("--magnetic-bc", choices=["conducting", "insulating"],
                   default="conducting",
                   help="magnetic wall BC; insulating uses the flux-function "
                        "formulation and is axisymmetric (m=0) only")
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--kz", type=float, default=None)
    p.add_argument("--kz-min", type=float, default=0.5)
    p.add_argument("--kz-max", type=float, default=8.0)
    p.add_argument("--kz-num", type=int, default=30)
    p.add_argument("--local-check", action="store_true",
                   help="print the ideal local Keplerian MRI optimum and exit")
    args = p.parse_args(argv)

    if args.local_check:
        opt = mri_keplerian_optimum()
        print("Ideal local Keplerian MRI optimum (q=3/2):")
        print(f"  s_max/Omega = {opt['s_max_over_Omega']:.4f}  (theory 0.75)")
        print(f"  (kz vA)^2/Omega^2 = {opt['wa2_opt_over_O2']:.4f}  (theory 0.9375)")
        return 0

    eta = args.R1 / args.R2
    Omega2 = args.Omega2 if args.Omega2 is not None else args.Omega1 * eta**1.5
    base = CircularCouette(args.R1, args.R2, args.Omega1, Omega2)
    solver = TaylorCouetteMRI(base, B0=args.B0, nu=args.nu, eta_mag=args.eta_mag,
                              N=args.N, family=args.family,
                              magnetic_bc=args.magnetic_bc)
    print(base.describe())
    print(f"  B0={args.B0:g} nu={args.nu:g} eta_mag={args.eta_mag:g} "
          f"Pm={solver.Pm:g} walls={args.magnetic_bc}")
    print(f"  Re={solver.Re:.3g} Rm={solver.Rm:.3g} S(Lundquist)={solver.S:.3g} Ha={solver.Ha:.3g}")

    if args.kz is not None:
        w, _ = solver.eigs(args.m, args.kz, n_return=6)
        print(f"\nkz={args.kz:g}: leading eigenvalues:")
        for s in w:
            print(f"   s = {s.real:+.6e}  {s.imag:+.6e} i")
    else:
        kzs = np.linspace(args.kz_min, args.kz_max, args.kz_num)
        kb, gb, _ = solver.max_growth_over_kz(args.m, kzs)
        print(f"\nkz scan (m={args.m}): most unstable kz={kb:.4f} growth={gb:+.6e}")
        print(f"  -> {'MRI-UNSTABLE' if gb > 1e-9 else 'stable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
