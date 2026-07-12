r"""
Axisymmetric hydrodynamic Taylor-Couette DNS (shenfun).

Nonlinear, time-stepping companion to the linear-stability solver
:mod:`taylor_couette_linear`.  It integrates the incompressible Navier-Stokes
equations for *axisymmetric* (``d/dtheta = 0``) flow in the annulus
``r in [R1, R2]``, ``z in [0, Lz]`` (axially periodic), keeping all three
velocity components ``(u_r, u_theta, u_z)`` (swirl retained).

Formulation
-----------
We integrate the perturbation ``u`` about the exact circular-Couette base flow

    U(r) = V(r) e_theta,   V(r) = a r + b/r,   Omega(r) = V/r = a + b/r**2

so the walls are homogeneous (``u = 0`` at ``r = R1, R2``) and the laminar state
is the exact fixed point ``u = 0``.  Substituting ``W = U + u`` into the
axisymmetric Navier-Stokes equations and subtracting the base balance
(``dP_base/dr = V**2/r``) gives, for the perturbation,

    du_r/dt   = -dp/dr + nu(L - 1/r**2) u_r + 2 Omega u_theta - N_r
    du_th/dt  =          nu(L - 1/r**2) u_th - 2a u_r          - N_th
    du_z/dt   = -dp/dz + nu  L          u_z                    - N_z
    0         = du_r/dr + u_r/r + du_z/dz                              (div u = 0)

with the *axisymmetric scalar Laplacian* ``L f = f_rr + f_r/r + f_zz`` and the
constant ``2a = 2 Omega + r Omega' = (1/r) d(r**2 Omega)/dr``.  The only linear
base-flow coupling for axisymmetric modes is the algebraic centrifugal/Coriolis
pair ``+2 Omega u_theta`` (r) and ``-2a u_r`` (theta) -- the same blocks that
drive the centrifugal (Taylor) instability in :mod:`taylor_couette_linear`.  The
quadratic perturbation self-advection (with the cylindrical metric terms) is

    N_r  = u_r u_r,r + u_z u_r,z - u_theta**2 / r
    N_th = u_r u_th,r + u_z u_th,z + u_r u_theta / r
    N_z  = u_r u_z,r + u_z u_z,z

Discretisation follows the linear solver: only the *radial* operators carry the
cylindrical ``1/r`` factors explicitly (plain Cartesian measure ``dr dz``; the
OrrSommerfeld strong-form pattern), the axial direction is Fourier, velocity uses
a no-slip Dirichlet basis and pressure the inf-sup-stable ``P_N``-``P_{N-2}``
pair.  Time stepping is IMEX: Crank-Nicolson for the linear (viscous + coupling +
pressure) operator and 2nd-order Adams-Bashforth for the nonlinear advection
(CNAB2, with an IMEX-Euler bootstrap).  Each step is a single coupled
velocity-pressure block solve per axial Fourier mode (shenfun ``BlockMatrix``),
so incompressibility is enforced exactly (no fractional-step splitting error).

The full 3D solver (azimuthal Fourier modes ``m != 0``) is a separate, later
layer; this module is the axisymmetric first step.
"""

from __future__ import annotations

from _demo_utils import default_thread_cap

default_thread_cap()

import argparse
import math

import numpy as np
from shenfun import (
    Array,
    BlockMatrix,
    CompositeSpace,
    Dx,
    Function,
    FunctionSpace,
    Project,
    TensorProductSpace,
    TestFunction,
    TrialFunction,
    comm,
    inner,
    la,
    project,
)
from taylor_couette_linear import CircularCouette


def _as_list(res):
    """``inner`` returns a single TPMatrix or a list; normalise to a list."""
    return res if isinstance(res, list) else [res]


def _require_resolved_m(m, Ntheta):
    """Reject an azimuthal seed mode the grid cannot represent.

    A real field ``Re[q e^{i m theta}]`` needs the ``+m`` and ``-m`` Fourier modes
    to be distinct and below the Nyquist mode, i.e. ``2|m| < Ntheta``.  Otherwise
    the sampled phase aliases to a different mode while the linear eigenvalue is
    still reported for the requested ``m`` -- e.g. ``--m 1 --Ntheta 1`` samples a
    ``theta``-constant field but evolves/labels it as ``m = 1``.
    """
    m = int(m)
    Ntheta = int(Ntheta)
    if 2 * abs(m) >= Ntheta:
        raise ValueError(
            f"azimuthal mode |m|={abs(m)} is unresolved by Ntheta={Ntheta}: a real "
            f"m-mode needs the +/-m Fourier modes distinct and below Nyquist "
            f"(2|m| < Ntheta).  Use Ntheta >= {2 * abs(m) + 1} or a smaller |m|."
        )


class AxisymmetricTCDNS:
    r"""Axisymmetric incompressible Taylor-Couette DNS (perturbation form).

    Parameters
    ----------
    base : CircularCouette
        Circular-Couette base flow.
    nu : float
        Kinematic viscosity.  ``Re = Omega1 R1 d / nu`` with ``d = R2 - R1``.
    Nr, Nz : int
        Radial (Legendre/Chebyshev) and axial (Fourier) resolution.
    Lz : float
        Axial period.  If ``None`` a default of ``2 pi / kz_default`` is chosen.
    dt : float
        Time step.
    family : {'L', 'C'}
        Radial basis family (Legendre or Chebyshev).
    dealias : float
        Padding factor for 3/2-rule dealiasing of the quadratic nonlinearity
        (``1.0`` disables padding).
    """

    def __init__(
        self,
        base: CircularCouette,
        nu=1.0e-2,
        Nr=48,
        Nz=32,
        Lz=None,
        dt=2.0e-3,
        family="L",
        dealias=1.5,
    ):
        self.base = base
        self.nu = float(nu)
        self.Nr = int(Nr)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.13 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        dom = (base.R1, base.R2)

        # ---- spaces -------------------------------------------------------
        self.F = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nr, family, bc=(0, 0), domain=dom)  # velocity
        self.S0 = FunctionSpace(self.Nr, family, domain=dom)  # orthogonal
        self.SP = FunctionSpace(self.Nr, family, domain=dom)  # pressure
        self.SP.slice = lambda: slice(0, self.Nr - 2)

        self.TD = TensorProductSpace(comm, (self.F, self.SD), axes=(1, 0))
        self.T0 = TensorProductSpace(comm, (self.F, self.S0), axes=(1, 0))
        self.TP = TensorProductSpace(
            comm, (self.F, self.SP), axes=(1, 0), modify_spaces_inplace=True
        )

        self.VV = CompositeSpace([self.TD, self.TD, self.TD])  # velocity
        self.VQ = CompositeSpace([self.TD, self.TD, self.TD, self.TP])  # +pressure

        self.r = self.TD.coors.psi[1]  # radial sympy symbol
        X = self.T0.local_mesh(True)
        self.rphys = X[1]
        self.zphys = X[0]
        self.inv_r = 1.0 / self.rphys

        # padded (dealiased) grid: pad BOTH the axial Fourier (axis 0) and the
        # radial (axis 1) directions, else high axial modes alias back in.
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
            Xp = self.T0p.local_mesh(True)
            self.inv_r_p = 1.0 / Xp[1]
        else:
            self.T0p = None
            self.inv_r_p = self.inv_r

        # ---- operators ----------------------------------------------------
        self._build_operators()

        # ---- fields -------------------------------------------------------
        self.u_hat = Function(self.VV)  # (u_r, u_th, u_z) coefficients
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VV)  # inner(v, N^n)
        self.N_old = Function(self.VV)  # inner(v, N^{n-1})
        self._have_old = False
        self.vts = TestFunction(self.TD)

        if comm.Get_rank() == 0:
            print(f"AxisymmetricTCDNS: {base.describe()}")
            print(
                f"  nu={self.nu:g} Re={self.Re:.3f}  Nr={self.Nr} Nz={self.Nz} "
                f"Lz={self.Lz:.4f} dt={self.dt:g} family={family} dealias={self.dealias:g}"
            )

    # ------------------------------------------------------------------
    # operator assembly
    # ------------------------------------------------------------------
    def _lap(self, u):
        r = self.r
        return Dx(u, 1, 2) + (1 / r) * Dx(u, 1, 1) + Dx(u, 0, 2)

    def _build_operators(self):
        r = self.r
        nu = self.nu
        dt = self.dt
        a = self.base.a
        # IMPORTANT: build Omega(r) in this space's radial symbol ``self.r`` (axis
        # 1 = ``y``).  base.Omega_sym is written in the linear solver's symbol
        # ``x``, which in this 2D space is the *axial* coordinate -> wrong axis.
        Om = self.base.a + self.base.b / r**2  # a + b/r**2
        twoOm = 2 * Om

        # implicit coupled operator over VQ:  M/dt - 1/2 A + grad p ; div = 0
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ur, ut, uz, p = up
        vr, vt, vz, q = vq
        imp = []
        # time derivative (mass / dt)
        imp += _as_list(inner(vr, ur * (1.0 / dt)))
        imp += _as_list(inner(vt, ut * (1.0 / dt)))
        imp += _as_list(inner(vz, uz * (1.0 / dt)))
        # -1/2 viscous (with -1/r^2 for r, theta) and -1/2 coupling
        imp += _as_list(inner(vr, -0.5 * nu * self._lap(ur)))
        imp += _as_list(inner(vr, 0.5 * nu * (1 / r**2) * ur))
        imp += _as_list(inner(vr, -0.5 * twoOm * ut))
        imp += _as_list(inner(vt, -0.5 * nu * self._lap(ut)))
        imp += _as_list(inner(vt, 0.5 * nu * (1 / r**2) * ut))
        imp += _as_list(inner(vt, -0.5 * (-2 * a) * ur))
        imp += _as_list(inner(vz, -0.5 * nu * self._lap(uz)))
        # pressure gradient  +grad p
        imp += _as_list(inner(vr, Dx(p, 1, 1)))
        imp += _as_list(inner(vz, Dx(p, 0, 1)))
        # continuity  div u = 0
        imp += _as_list(inner(q, Dx(ur, 1, 1)))
        imp += _as_list(inner(q, (1 / r) * ur))
        imp += _as_list(inner(q, Dx(uz, 0, 1)))
        self.Limp = la.BlockMatrixSolver(imp)

        # explicit velocity-velocity operator over VV: M/dt + 1/2 A
        uu = TrialFunction(self.VV)
        vv = TestFunction(self.VV)
        er, et, ez = uu
        tr, tt, tz = vv
        exp = []
        exp += _as_list(inner(tr, er * (1.0 / dt)))
        exp += _as_list(inner(tt, et * (1.0 / dt)))
        exp += _as_list(inner(tz, ez * (1.0 / dt)))
        exp += _as_list(inner(tr, 0.5 * nu * self._lap(er)))
        exp += _as_list(inner(tr, -0.5 * nu * (1 / r**2) * er))
        exp += _as_list(inner(tr, 0.5 * twoOm * et))
        exp += _as_list(inner(tt, 0.5 * nu * self._lap(et)))
        exp += _as_list(inner(tt, -0.5 * nu * (1 / r**2) * et))
        exp += _as_list(inner(tt, 0.5 * (-2 * a) * er))
        exp += _as_list(inner(tz, 0.5 * nu * self._lap(ez)))
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    # nonlinear term:  returns inner(v, N) as a Function over VV
    # ------------------------------------------------------------------
    def _phys(self, comp_hat):
        """field, d/dr, d/dz of one velocity component on the working grid."""
        pf = (self.dealias, self.dealias) if self.dealias > 1.0 else None
        f = comp_hat.backward(padding_factor=pf) if pf else comp_hat.backward()
        fr = (
            project(Dx(comp_hat, 1, 1), self.T0).backward(padding_factor=pf)
            if pf
            else project(Dx(comp_hat, 1, 1), self.T0).backward()
        )
        fz = (
            project(Dx(comp_hat, 0, 1), self.T0).backward(padding_factor=pf)
            if pf
            else project(Dx(comp_hat, 0, 1), self.T0).backward()
        )
        return np.asarray(f), np.asarray(fr), np.asarray(fz)

    def nonlinear(self, out):
        """Compute ``out[i] = inner(v_i, N_i)`` for the advection term."""
        ur, urr, urz = self._phys(self.u_hat[0])
        ut, utr, utz = self._phys(self.u_hat[1])
        uz, uzr, uzz = self._phys(self.u_hat[2])
        invr = self.inv_r_p

        n_r = ur * urr + uz * urz - ut * ut * invr
        n_t = ur * utr + uz * utz + ur * ut * invr
        n_z = ur * uzr + uz * uzz

        if self.dealias > 1.0:
            # forward through the padded space -> dealiased coeffs -> standard grid
            ar = self._dealias(n_r)
            at = self._dealias(n_t)
            az = self._dealias(n_z)
        else:
            ar = Array(self.T0)
            ar[:] = n_r
            at = Array(self.T0)
            at[:] = n_t
            az = Array(self.T0)
            az[:] = n_z
        out[0] = inner(self.vts, ar)
        out[1] = inner(self.vts, at)
        out[2] = inner(self.vts, az)
        return out

    def _dealias(self, padded_values):
        """Truncate a product computed on the padded grid back to the std grid.

        Forward through the padded space (drops the aliased high modes), copy the
        truncated coefficients into a clean base-space ``Function`` and transform
        back to the standard quadrature grid.
        """
        ap = Array(self.T0p)
        ap[:] = padded_values
        g = Function(self.T0)
        g[:] = ap.forward()
        return g.backward()

    # ------------------------------------------------------------------
    # one CNAB2 time step
    # ------------------------------------------------------------------
    def step(self):
        # nonlinear at current state
        self.nonlinear(self.N_hat)

        # explicit part:  rhs_v = (M/dt + 1/2 A) u^n
        rhs_v = Function(self.VV)
        rhs_v = self.Lexp.matvec(self.u_hat, rhs_v)

        # Adams-Bashforth combination of the nonlinear term
        for i in range(3):
            if self._have_old:
                self.rhs[i] = rhs_v[i] - (1.5 * self.N_hat[i] - 0.5 * self.N_old[i])
            else:
                self.rhs[i] = rhs_v[i] - self.N_hat[i]  # IMEX-Euler bootstrap
        self.rhs[3] = 0.0

        # coupled solve (fix the k=0 pressure null space)
        self.sol = self.Limp(self.rhs, u=self.sol, constraints=((3, 0, 0),))

        for i in range(3):
            self.u_hat[i] = self.sol[i]
        self.p_hat[:] = self.sol[3]

        # shift nonlinear history
        self.N_old[:] = self.N_hat
        self._have_old = True
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        self._t += self.dt
        self._tstep += 1

    # ------------------------------------------------------------------
    # initial conditions
    # ------------------------------------------------------------------
    def set_perturbation(self, amp=1e-3, kz_mode=1, seed=0):
        """Seed a small divergence-free, no-slip perturbation at one axial mode.

        The meridional ``(u_r, u_z)`` flow is built from a Stokes stream function
        ``psi = g(r) cos(kz z)`` via ``u_r = -(1/r) dpsi/dz``, ``u_z = (1/r) dpsi/dr``
        (so ``div u = 0`` identically).  Choosing ``g = sin^2(pi (r-R1)/d)`` makes
        ``g`` and ``g'`` vanish at both walls, so ``u_r`` and ``u_z`` satisfy no-slip
        exactly and the discrete divergence starts at roundoff.  ``u_theta`` is an
        independent wall-vanishing swirl seed.
        """
        R1 = self.base.R1
        d = self.base.gap
        kz = 2 * np.pi * kz_mode / self.Lz
        rr = self.rphys
        zz = self.zphys
        arg = np.pi * (rr - R1) / d
        g = np.sin(arg) ** 2  # g = g' = 0 at both walls
        gp = (2 * np.pi / d) * np.sin(arg) * np.cos(arg)
        ur = amp * (kz / rr) * g * np.sin(kz * zz)  # = -(1/r) dpsi/dz
        uz = amp * (1.0 / rr) * gp * np.cos(kz * zz)  # =  (1/r) dpsi/dr
        ut = amp * np.sin(arg) * np.cos(kz * zz)
        a_r = Array(self.TD)
        a_r[:] = ur
        a_t = Array(self.TD)
        a_t[:] = ut
        a_z = Array(self.TD)
        a_z[:] = uz
        self.u_hat[0] = a_r.forward(Function(self.TD))
        self.u_hat[1] = a_t.forward(Function(self.TD))
        self.u_hat[2] = a_z.forward(Function(self.TD))
        self._have_old = False

    def seed_linear_eigenmode(self, kz_mode=1, amp=1e-6, which=0):
        """Seed the exact discrete leading eigenmode at one axial wavenumber.

        Uses :class:`taylor_couette_linear.TaylorCouetteLinear` at the matching
        ``(Nr, family, nu)`` so the radial Dirichlet bases coincide; the velocity
        eigenvector blocks are injected directly into Fourier mode ``kz_mode``.
        With this seed the measured growth rate matches linear theory immediately
        (no transient), which is the sharpest DNS/linear consistency check.
        Returns the linear eigenvalue ``s``.
        """
        from taylor_couette_linear import TaylorCouetteLinear

        kz = 2 * np.pi * kz_mode / self.Lz
        lin = TaylorCouetteLinear(self.base, nu=self.nu, N=self.Nr, family=self.family)
        w, V = lin.eigs(0, kz, n_return=which + 1)
        n = lin.n
        vec = V[:, which]
        for comp in range(3):
            f = Function(self.TD)
            f[:] = 0.0
            f[kz_mode, :n] = vec[comp * n : (comp + 1) * n] * amp
            self.u_hat[comp] = f
        self._have_old = False
        return complex(w[which])

    def set_random(self, amp=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        R1, R2 = self.base.R1, self.base.R2
        d = self.base.gap
        wall = np.sin(np.pi * (self.rphys - R1) / d)
        for i in range(3):
            a = Array(self.TD)
            a[:] = amp * wall * rng.standard_normal(a.shape)
            self.u_hat[i] = a.forward(Function(self.TD))
        # project to divergence-free by one pressure solve (optional; small)
        self._have_old = False

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def velocity_physical(self):
        return (
            self.u_hat[0].backward(),
            self.u_hat[1].backward(),
            self.u_hat[2].backward(),
        )

    def energy(self):
        ur, ut, uz = self.velocity_physical()
        e = 0.5 * inner(1, (ur * ur + ut * ut + uz * uz) * self.rphys)
        return float(e)

    def divergence_linf(self):
        """Pointwise max |div u| = |d u_r/dr + u_r/r + d u_z/dz|.

        The radial and axial derivatives are projected separately and summed in
        physical space (with the explicit ``1/r``).  NB: combining the three
        terms into one ``inner``/``project`` expression -- i.e. mixing
        ``Dx(f_hat, ...)`` with a sympy ``(1/r)*f_hat`` coefficient -- mis-evaluates
        in shenfun and reports a spurious O(amplitude) "divergence"; keep them
        separate.  The coupled solve enforces continuity to roundoff, so this is
        ~1e-13.
        """
        dur_dr = np.asarray(project(Dx(self.u_hat[0], 1, 1), self.T0).backward())
        duz_dz = np.asarray(project(Dx(self.u_hat[2], 0, 1), self.T0).backward())
        ur = np.asarray(self.u_hat[0].backward())
        div = dur_dr + ur * self.inv_r + duz_dz
        return float(np.abs(div).max())

    def wall_residual(self):
        ur, ut, uz = self.velocity_physical()
        # values at the radial-extreme quadrature points (closest to walls)
        return float(
            max(
                np.abs(ur[:, 0]).max(),
                np.abs(ur[:, -1]).max(),
                np.abs(ut[:, 0]).max(),
                np.abs(ut[:, -1]).max(),
                np.abs(uz[:, 0]).max(),
                np.abs(uz[:, -1]).max(),
            )
        )

    def torque_diagnostic(self):
        """Mean axial-z, azimuthally-averaged angular-momentum flux proxy.

        Returns the volume-averaged perturbation azimuthal kinetic energy as a
        cheap scalar to watch saturation; a full wall-torque diagnostic is added
        with the validation suite.
        """
        ur, ut, uz = self.velocity_physical()
        return float(inner(1, (ut * ut) * self.rphys))

    def diagnostics(self, t, tstep):
        return {
            "t": float(t),
            "tstep": int(tstep),
            "E": self.energy(),
            "div_linf": self.divergence_linf(),
            "wall": self.wall_residual(),
            "Eth": self.torque_diagnostic(),
        }

    # ------------------------------------------------------------------
    # run loop
    # ------------------------------------------------------------------
    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
        # time accumulates across successive run() calls (lazy-init)
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        nsteps = int(round(end_time / self.dt))
        for k in range(1, nsteps + 1):
            self.step()
            if assert_finite and not np.all(np.isfinite(self.u_hat)):
                raise RuntimeError(f"non-finite velocity at t={self._t:g}")
            if (moderror and self._tstep % moderror == 0) or k == nsteps:
                d = self.diagnostics(self._t, self._tstep)
                if on_diag is not None:
                    on_diag(d)
                elif moderror and comm.Get_rank() == 0:
                    print(
                        f"t={d['t']:8.4f} E={d['E']:.6e} "
                        f"div={d['div_linf']:.2e} wall={d['wall']:.2e} "
                        f"Eth={d['Eth']:.6e}"
                    )
        return self.diagnostics(self._t, self._tstep)

    def growth_rate(self, t0, t1, restart=True, amp=1e-6, kz_mode=1, seed=0):
        """Measure the linear growth rate by evolving a tiny seed.

        Returns ``sigma = 0.5 d ln E / dt`` averaged over ``[t0, t1]`` (after a
        ``t0`` transient lets the leading mode emerge).
        """
        if restart:
            self.set_perturbation(amp=amp, kz_mode=kz_mode, seed=seed)
        # transient
        n0 = int(round(t0 / self.dt))
        for _ in range(n0):
            self.step()
        E0 = self.energy()
        n1 = int(round((t1 - t0) / self.dt))
        for _ in range(n1):
            self.step()
        E1 = self.energy()
        dt_meas = n1 * self.dt
        return 0.5 * math.log(E1 / E0) / dt_meas

    # ------------------------------------------------------------------
    # checkpoint / restart (exact CNAB2 continuation)
    # ------------------------------------------------------------------
    def state_dict(self):
        """Serializable state for an exact restart: perturbation velocity
        coefficients, the Adams-Bashforth-2 history, and the clock."""
        return {
            "u_hat": np.array(self.u_hat, copy=True),
            "N_old": np.array(self.N_old, copy=True),
            "have_old": bool(self._have_old),
            "t": float(getattr(self, "_t", 0.0)),
            "tstep": int(getattr(self, "_tstep", 0)),
        }

    def load_state_dict(self, state):
        """Restore a checkpoint produced by :meth:`state_dict`."""
        self.u_hat[:] = state["u_hat"]
        self.N_old[:] = state["N_old"]
        self._have_old = bool(state["have_old"])
        self._t = float(state["t"])
        self._tstep = int(state["tstep"])
        return self


# ===========================================================================
# Full 3D solver (azimuthal Fourier modes m != 0)
# ===========================================================================
class TaylorCouetteDNS:
    r"""Full 3D incompressible Taylor-Couette DNS (perturbation form).

    Extends :class:`AxisymmetricTCDNS` to non-axisymmetric flow by adding an
    azimuthal Fourier direction ``theta in [0, 2 pi)``.  The fields depend on
    ``(theta, z, r)``; ``theta`` and ``z`` are Fourier, ``r`` is a no-slip
    Dirichlet basis.  The perturbation ``u`` about ``U = V(r) e_theta`` now feels
    the full set of linear couplings (cf. ``TaylorCouetteLinear.assemble_parts``):

      * base-shear advection ``-Omega d/dtheta`` (``-i m Omega`` per mode);
      * centrifugal/Coriolis ``+2 Omega u_theta`` (r) and ``-2a u_r`` (theta);
      * viscous cross-coupling ``-/+ (2/r**2) d u_{theta/r}/dtheta``
        (``-/+ 2 i m / r**2``);
      * full scalar Laplacian ``L f = f_rr + f_r/r + f_{theta theta}/r**2 + f_zz``;
      * continuity ``u_r,r + u_r/r + u_theta,theta/r + u_z,z = 0``.

    The quadratic self-advection carries every cylindrical metric term.  Time
    stepping, the coupled per-``(m, kz)`` velocity-pressure block solve, and 3/2
    dealiasing are exactly as in the axisymmetric solver, now in 3D.  Setting the
    azimuthal resolution to capture only ``m = 0`` reproduces the axisymmetric
    results bit-for-bit.
    """

    def __init__(
        self,
        base: CircularCouette,
        nu=1.0e-2,
        Nr=40,
        Ntheta=16,
        Nz=32,
        Lz=None,
        dt=2.0e-3,
        family="L",
        dealias=1.5,
    ):
        self.base = base
        self.nu = float(nu)
        self.Nr = int(Nr)
        self.Ntheta = int(Ntheta)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.13 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        dom = (base.R1, base.R2)

        # theta: complex Fourier;  z: real Fourier;  r: Dirichlet / orthogonal
        self.Ft = FunctionSpace(
            self.Ntheta, "Fourier", dtype="D", domain=(0, 2 * math.pi)
        )
        self.Fz = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nr, family, bc=(0, 0), domain=dom)
        self.S0 = FunctionSpace(self.Nr, family, domain=dom)
        self.SP = FunctionSpace(self.Nr, family, domain=dom)
        self.SP.slice = lambda: slice(0, self.Nr - 2)

        ax = (2, 0, 1)  # radial (axis 2) is the solve axis
        self.TD = TensorProductSpace(comm, (self.Ft, self.Fz, self.SD), axes=ax)
        self.T0 = TensorProductSpace(comm, (self.Ft, self.Fz, self.S0), axes=ax)
        self.TP = TensorProductSpace(
            comm, (self.Ft, self.Fz, self.SP), axes=ax, modify_spaces_inplace=True
        )
        self.VV = CompositeSpace([self.TD, self.TD, self.TD])
        self.VQ = CompositeSpace([self.TD, self.TD, self.TD, self.TP])

        self.r = self.TD.coors.psi[2]  # radial symbol (axis 2)
        X = self.T0.local_mesh(True)
        self.rphys = X[2]
        self.inv_r = 1.0 / self.rphys
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias, self.dealias))
            self.inv_r_p = 1.0 / self.T0p.local_mesh(True)[2]
        else:
            self.T0p = None
            self.inv_r_p = self.inv_r

        self._build_operators()

        self.u_hat = Function(self.VV)
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VV)
        self.N_old = Function(self.VV)
        self._have_old = False
        self.vts = TestFunction(self.TD)

        if comm.Get_rank() == 0:
            print(f"TaylorCouetteDNS(3D): {base.describe()}")
            print(
                f"  nu={self.nu:g} Re={self.Re:.3f}  Nr={self.Nr} Ntheta={self.Ntheta} "
                f"Nz={self.Nz} Lz={self.Lz:.4f} dt={self.dt:g} dealias={self.dealias:g}"
            )

    # ------------------------------------------------------------------
    def _lap(self, u):
        r = self.r
        return (
            Dx(u, 2, 2) + (1 / r) * Dx(u, 2, 1) + (1 / r**2) * Dx(u, 0, 2) + Dx(u, 1, 2)
        )

    def _avv_terms(self, ur, ut, uz, vr, vt, vz, sign):
        """Velocity-velocity linear operator A (viscous + all couplings),
        scaled by ``sign`` (Crank-Nicolson: -0.5 implicit, +0.5 explicit)."""
        r = self.r
        nu = self.nu
        a = self.base.a
        Om = self.base.a + self.base.b / r**2
        out = []
        # r-momentum
        out += _as_list(inner(vr, sign * nu * self._lap(ur)))
        out += _as_list(inner(vr, sign * (-nu) * (1 / r**2) * ur))
        out += _as_list(inner(vr, sign * (-nu) * (2 / r**2) * Dx(ut, 0, 1)))
        out += _as_list(inner(vr, sign * (-Om) * Dx(ur, 0, 1)))  # -Omega d/dtheta
        out += _as_list(inner(vr, sign * (2 * Om) * ut))  # +2 Omega u_theta
        # theta-momentum
        out += _as_list(inner(vt, sign * nu * self._lap(ut)))
        out += _as_list(inner(vt, sign * (-nu) * (1 / r**2) * ut))
        out += _as_list(inner(vt, sign * nu * (2 / r**2) * Dx(ur, 0, 1)))
        out += _as_list(inner(vt, sign * (-Om) * Dx(ut, 0, 1)))
        out += _as_list(inner(vt, sign * (-2 * a) * ur))  # -2a u_r
        # z-momentum
        out += _as_list(inner(vz, sign * nu * self._lap(uz)))
        out += _as_list(inner(vz, sign * (-Om) * Dx(uz, 0, 1)))
        return out

    def _build_operators(self):
        r = self.r
        dt = self.dt
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ur, ut, uz, p = up
        vr, vt, vz, q = vq
        imp = []
        imp += _as_list(inner(vr, ur * (1.0 / dt)))
        imp += _as_list(inner(vt, ut * (1.0 / dt)))
        imp += _as_list(inner(vz, uz * (1.0 / dt)))
        imp += self._avv_terms(ur, ut, uz, vr, vt, vz, sign=-0.5)
        # pressure gradient  +grad p   (r, theta, z)
        imp += _as_list(inner(vr, Dx(p, 2, 1)))
        imp += _as_list(inner(vt, (1 / r) * Dx(p, 0, 1)))
        imp += _as_list(inner(vz, Dx(p, 1, 1)))
        # continuity  u_r,r + u_r/r + u_theta,theta/r + u_z,z
        imp += _as_list(inner(q, Dx(ur, 2, 1)))
        imp += _as_list(inner(q, (1 / r) * ur))
        imp += _as_list(inner(q, (1 / r) * Dx(ut, 0, 1)))
        imp += _as_list(inner(q, Dx(uz, 1, 1)))
        self.Limp = la.BlockMatrixSolver(imp)

        uu = TrialFunction(self.VV)
        vv = TestFunction(self.VV)
        er, et, ez = uu
        tr, tt, tz = vv
        exp = []
        exp += _as_list(inner(tr, er * (1.0 / dt)))
        exp += _as_list(inner(tt, et * (1.0 / dt)))
        exp += _as_list(inner(tz, ez * (1.0 / dt)))
        exp += self._avv_terms(er, et, ez, tr, tt, tz, sign=0.5)
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    def _phys(self, comp_hat):
        """field and d/dr, d/dtheta, d/dz on the (padded) working grid."""
        pf = (self.dealias,) * 3 if self.dealias > 1.0 else None

        def bw(expr_hat):
            f = project(expr_hat, self.T0)
            return np.asarray(f.backward(padding_factor=pf) if pf else f.backward())

        field = np.asarray(
            comp_hat.backward(padding_factor=pf) if pf else comp_hat.backward()
        )
        return (
            field,
            bw(Dx(comp_hat, 2, 1)),
            bw(Dx(comp_hat, 0, 1)),
            bw(Dx(comp_hat, 1, 1)),
        )

    def _dealias(self, padded_values):
        ap = Array(self.T0p)
        ap[:] = padded_values
        g = Function(self.T0)
        g[:] = ap.forward()
        return g.backward()

    def nonlinear(self, out):
        ur, urr, urt, urz = self._phys(self.u_hat[0])
        ut, utr, utt, utz = self._phys(self.u_hat[1])
        uz, uzr, uzt, uzz = self._phys(self.u_hat[2])
        invr = self.inv_r_p
        n_r = ur * urr + (ut * invr) * urt + uz * urz - ut * ut * invr
        n_t = ur * utr + (ut * invr) * utt + uz * utz + ur * ut * invr
        n_z = ur * uzr + (ut * invr) * uzt + uz * uzz
        if self.dealias > 1.0:
            ar = self._dealias(n_r)
            at = self._dealias(n_t)
            az = self._dealias(n_z)
        else:
            ar = Array(self.T0)
            ar[:] = n_r
            at = Array(self.T0)
            at[:] = n_t
            az = Array(self.T0)
            az[:] = n_z
        out[0] = inner(self.vts, ar)
        out[1] = inner(self.vts, at)
        out[2] = inner(self.vts, az)
        return out

    # ------------------------------------------------------------------
    def step(self):
        self.nonlinear(self.N_hat)
        rhs_v = Function(self.VV)
        rhs_v = self.Lexp.matvec(self.u_hat, rhs_v)
        for i in range(3):
            if self._have_old:
                self.rhs[i] = rhs_v[i] - (1.5 * self.N_hat[i] - 0.5 * self.N_old[i])
            else:
                self.rhs[i] = rhs_v[i] - self.N_hat[i]
        self.rhs[3] = 0.0
        self.sol = self.Limp(self.rhs, u=self.sol, constraints=((3, 0, 0),))
        for i in range(3):
            self.u_hat[i] = self.sol[i]
        self.p_hat[:] = self.sol[3]
        self.N_old[:] = self.N_hat
        self._have_old = True
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        self._t += self.dt
        self._tstep += 1

    # ------------------------------------------------------------------
    def seed_linear_eigenmode(self, m=1, kz_mode=1, amp=1e-6, which=0):
        """Seed the real part of the linear eigenmode at ``(m, kz)``.

        Builds ``u = Re[q(r) exp(i(m theta + kz z))]`` from the
        :class:`taylor_couette_linear.TaylorCouetteLinear` eigenvector at the
        matching ``(Nr, family, nu)``.  Returns the linear eigenvalue ``s``;
        ``Re(s)`` is what the DNS energy growth must reproduce.
        """
        from taylor_couette_linear import TaylorCouetteLinear

        _require_resolved_m(m, self.Ntheta)
        kz = 2 * math.pi * kz_mode / self.Lz
        lin = TaylorCouetteLinear(self.base, nu=self.nu, N=self.Nr, family=self.family)
        w, V = lin.eigs(m, kz, n_return=which + 1)
        n = lin.n
        X = self.TD.local_mesh(True)
        th, zz, rr = X[0], X[1], X[2]
        rpts = np.asarray(rr[0, 0, :])
        phase = np.exp(1j * (m * th + kz * zz))
        for comp in range(3):
            fr = Function(lin.SD)
            fr[:] = 0.0
            fr[lin.SD.slice()] = V[comp * n : (comp + 1) * n, which]
            prof = np.asarray(fr.eval(rpts))
            field = (amp * prof[None, None, :] * phase).real
            a = Array(self.TD)
            a[:] = field
            self.u_hat[comp] = a.forward(Function(self.TD))
        self._have_old = False
        return complex(w[which])

    def set_perturbation(self, amp=1e-3, m=1, kz_mode=1, seed=0):
        """Real ``cos(m theta) cos(kz z)`` wall-vanishing seed in all components."""
        _require_resolved_m(m, self.Ntheta)
        R1 = self.base.R1
        d = self.base.gap
        kz = 2 * math.pi * kz_mode / self.Lz
        X = self.TD.local_mesh(True)
        th, zz, rr = X[0], X[1], X[2]
        shape = np.sin(np.pi * (rr - R1) / d)
        base_field = amp * shape * np.cos(m * th) * np.cos(kz * zz)
        for comp in range(3):
            a = Array(self.TD)
            a[:] = base_field
            self.u_hat[comp] = a.forward(Function(self.TD))
        self._have_old = False

    def set_random(self, amp=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        R1 = self.base.R1
        d = self.base.gap
        wall = np.sin(np.pi * (self.rphys - R1) / d)
        for comp in range(3):
            a = Array(self.TD)
            a[:] = amp * wall * rng.standard_normal(a.shape)
            self.u_hat[comp] = a.forward(Function(self.TD))
        self._have_old = False

    # ------------------------------------------------------------------
    def velocity_physical(self):
        return (
            self.u_hat[0].backward(),
            self.u_hat[1].backward(),
            self.u_hat[2].backward(),
        )

    def energy(self):
        ur, ut, uz = self.velocity_physical()
        return float(0.5 * inner(1, (ur * ur + ut * ut + uz * uz) * self.rphys))

    def divergence_linf(self):
        """max |u_r,r + u_r/r + u_theta,theta/r + u_z,z| (separate projections)."""
        dur_dr = np.asarray(project(Dx(self.u_hat[0], 2, 1), self.T0).backward())
        dut_dt = np.asarray(project(Dx(self.u_hat[1], 0, 1), self.T0).backward())
        duz_dz = np.asarray(project(Dx(self.u_hat[2], 1, 1), self.T0).backward())
        ur = np.asarray(self.u_hat[0].backward())
        div = dur_dr + ur * self.inv_r + dut_dt * self.inv_r + duz_dz
        return float(np.abs(div).max())

    def diagnostics(self, t, tstep):
        return {
            "t": float(t),
            "tstep": int(tstep),
            "E": self.energy(),
            "div_linf": self.divergence_linf(),
        }

    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
        # time accumulates across successive run() calls (lazy-init)
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        nsteps = int(round(end_time / self.dt))
        for k in range(1, nsteps + 1):
            self.step()
            if assert_finite and not np.all(np.isfinite(self.u_hat)):
                raise RuntimeError(f"non-finite velocity at t={self._t:g}")
            if (moderror and self._tstep % moderror == 0) or k == nsteps:
                d = self.diagnostics(self._t, self._tstep)
                if on_diag is not None:
                    on_diag(d)
                elif moderror and comm.Get_rank() == 0:
                    print(f"t={d['t']:8.4f} E={d['E']:.6e} div={d['div_linf']:.2e}")
        return self.diagnostics(self._t, self._tstep)


# ===========================================================================
# Axisymmetric MHD / MRI solver (imposed uniform axial field B0)
# ===========================================================================
class AxisymmetricMRIDNS:
    r"""Axisymmetric resistive-MHD Taylor-Couette DNS with an imposed axial field.

    Time-steps the incompressible, viscous, resistive MHD equations for
    *axisymmetric* (`d/dtheta = 0`) perturbations ``(u, b)`` about the base state
    ``W = V(r) e_theta``, ``B = B0 e_z`` (magnetic field in Alfven units,
    ``v_A = B0``).  This is the nonlinear, time-stepping companion to the linear
    MRI eigensolver :class:`taylor_couette_mri.TaylorCouetteMRI`, and the standard
    **magnetorotational instability** (MRI) lives here: a Rayleigh-stable
    (quasi-Keplerian) profile is destabilised by the axial field.

    Total-pressure formulation (``Pi = p + B0 b_z`` absorbs the imposed-field
    magnetic pressure), so the linear Lorentz / induction couplings are just
    ``+ B0 db/dz`` and ``+ B0 du/dz`` with the field-stretching source
    ``r Omega' b_r`` feeding ``b_theta`` (see ``taylor_couette_mri``).  The linear
    operator is identical to ``TaylorCouetteMRI.assemble_parts(0, kz)`` and is
    advanced implicitly (Crank-Nicolson); the quadratic nonlinearities -- the
    Maxwell/Reynolds advection ``(u.grad)u - (b.grad)b`` and the induction EMF curl
    ``curl(u x b)`` -- are advanced explicitly (AB2), pseudo-spectral with 3/2
    dealiasing.  Each step is one coupled 7-field block solve per axial Fourier
    mode (``u_r, u_theta, u_z, Pi, b_r, b_theta, b_z``) so ``div(u) = 0`` exactly;
    ``div(b) = 0`` is preserved by the induction dynamics and monitored.

    Perfectly-conducting walls: ``b_r = 0`` (Dirichlet), ``d(r b_theta)/dr = 0``
    (Robin), ``b_z' = 0`` (Neumann) -- the same radial bases as the eigensolver.
    """

    def __init__(
        self,
        base: CircularCouette,
        B0=0.2,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=48,
        Nz=32,
        Lz=None,
        dt=2.0e-3,
        family="L",
        dealias=1.5,
    ):
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nr = int(Nr)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.0 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        self.Rm = base.Omega1 * base.R1 * base.gap / self.eta_mag
        self.Pm = self.nu / self.eta_mag
        self.S = self.B0 * base.gap / self.eta_mag  # Lundquist number
        dom = (base.R1, base.R2)
        Jm = 0.5 * (base.R2 - base.R1)

        self.F = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nr, family, bc=(0, 0), domain=dom)  # u, b_r
        self.S0 = FunctionSpace(self.Nr, family, domain=dom)  # orthogonal
        self.SP = FunctionSpace(self.Nr, family, domain=dom)
        self.SP.slice = lambda: slice(0, self.Nr - 2)
        self.Sbt = FunctionSpace(
            self.Nr,
            family,
            domain=dom,  # b_theta Robin
            bc={"left": {"R": (base.R1 / Jm, 0)}, "right": {"R": (base.R2 / Jm, 0)}},
        )
        self.Sbz = FunctionSpace(
            self.Nr,
            family,
            domain=dom,  # b_z Neumann
            bc={"left": {"N": 0}, "right": {"N": 0}},
        )

        ax = (1, 0)
        self.TD = TensorProductSpace(comm, (self.F, self.SD), axes=ax)
        self.T0 = TensorProductSpace(comm, (self.F, self.S0), axes=ax)
        self.TP = TensorProductSpace(
            comm, (self.F, self.SP), axes=ax, modify_spaces_inplace=True
        )
        self.Tbt = TensorProductSpace(comm, (self.F, self.Sbt), axes=ax)
        self.Tbz = TensorProductSpace(comm, (self.F, self.Sbz), axes=ax)

        # u_r,u_th,u_z, Pi, b_r,b_th,b_z   and the 6 evolving fields (no pressure)
        self.VQ = CompositeSpace(
            [self.TD, self.TD, self.TD, self.TP, self.TD, self.Tbt, self.Tbz]
        )
        self.VE = CompositeSpace(
            [self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz]
        )

        self.r = self.TD.coors.psi[1]
        X = self.T0.local_mesh(True)
        self.rphys = X[1]
        self.inv_r = 1.0 / self.rphys
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
            self.inv_r_p = 1.0 / self.T0p.local_mesh(True)[1]
        else:
            self.T0p = None
            self.inv_r_p = self.inv_r

        self._build_operators()

        self.x = Function(self.VE)  # (u_r,u_th,u_z, b_r,b_th,b_z)
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VE)
        self.N_old = Function(self.VE)
        self._have_old = False
        self.vu = TestFunction(self.TD)
        self.vbr = TestFunction(self.TD)
        self.vbt = TestFunction(self.Tbt)
        self.vbz = TestFunction(self.Tbz)

        # Cached projections onto the orthogonal space (assembled once; recomputed
        # against the in-place-updated fields each step -- ``project()`` would
        # rebuild the per-mode solver every call and dominates the runtime, ~108
        # vs ~14 ms/step).
        #
        # INVARIANT: each ``Project`` captures ``self.x[i]`` symbolically, so it must
        # keep seeing the *same memory*.  ``step()`` updates the state via
        # ``self.x[i] = ...``, which is an in-place numpy item-assignment into the
        # composite ``Function``'s row i (CompositeSpace stores all fields in one
        # array) -- NOT a rebind of the sub-function object.  Do not replace those
        # assignments with anything that allocates a new array for ``self.x[i]``
        # (e.g. ``self.x[i] = self.x[i] + d``), or the cached Projects would silently
        # evaluate stale fields.  Likewise ``self._eps[k][:] = ...`` below.
        self._Pdr = [Project(Dx(self.x[i], 1, 1), self.T0) for i in range(6)]
        self._Pdz = [Project(Dx(self.x[i], 0, 1), self.T0) for i in range(6)]
        self._eps = [Function(self.T0) for _ in range(3)]  # EMF eps_r, eps_t, eps_z
        self._Petz = Project(Dx(self._eps[1], 0, 1), self.T0)  # d eps_t / dz
        self._Perz = Project(Dx(self._eps[0], 0, 1), self.T0)  # d eps_r / dz
        self._Pezr = Project(Dx(self._eps[2], 1, 1), self.T0)  # d eps_z / dr
        self._Petr = Project(Dx(self._eps[1], 1, 1), self.T0)  # d eps_t / dr

        if comm.Get_rank() == 0:
            print(f"AxisymmetricMRIDNS: {base.describe()}")
            print(
                f"  B0={self.B0:g} nu={self.nu:g} eta={self.eta_mag:g} "
                f"Re={self.Re:.2f} Rm={self.Rm:.2f} Pm={self.Pm:g} S={self.S:.3f}"
            )
            print(
                f"  Nr={self.Nr} Nz={self.Nz} Lz={self.Lz:.4f} dt={self.dt:g} "
                f"dealias={self.dealias:g}"
            )

    # ------------------------------------------------------------------
    def _lap(self, u):
        r = self.r
        return Dx(u, 1, 2) + (1 / r) * Dx(u, 1, 1) + Dx(u, 0, 2)

    def _Lxx(self, ur, ut, uz, br, bt, bz, vr, vt, vz, cr, ct, cz, sign):
        """Evolving-field linear MHD operator (no pressure/continuity), x sign."""
        r = self.r
        nu, eta, B0 = self.nu, self.eta_mag, self.B0
        a = self.base.a
        Om = self.base.a + self.base.b / r**2
        rOmp = -2 * self.base.b / r**2  # r dOmega/dr
        dz = lambda f: Dx(f, 0, 1)
        out = []
        # r-momentum:  nu(L-1/r^2)u_r + 2 Om u_theta + B0 db_r/dz
        out += _as_list(inner(vr, sign * nu * self._lap(ur)))
        out += _as_list(inner(vr, sign * (-nu) * (1 / r**2) * ur))
        out += _as_list(inner(vr, sign * (2 * Om) * ut))
        out += _as_list(inner(vr, sign * B0 * dz(br)))
        # theta-momentum:  nu(L-1/r^2)u_th - 2a u_r + B0 db_th/dz
        out += _as_list(inner(vt, sign * nu * self._lap(ut)))
        out += _as_list(inner(vt, sign * (-nu) * (1 / r**2) * ut))
        out += _as_list(inner(vt, sign * (-2 * a) * ur))
        out += _as_list(inner(vt, sign * B0 * dz(bt)))
        # z-momentum:  nu L u_z + B0 db_z/dz
        out += _as_list(inner(vz, sign * nu * self._lap(uz)))
        out += _as_list(inner(vz, sign * B0 * dz(bz)))
        # b_r induction:  eta(L-1/r^2)b_r + B0 du_r/dz
        out += _as_list(inner(cr, sign * eta * self._lap(br)))
        out += _as_list(inner(cr, sign * (-eta) * (1 / r**2) * br))
        out += _as_list(inner(cr, sign * B0 * dz(ur)))
        # b_theta induction:  eta(L-1/r^2)b_th + B0 du_th/dz + rOm' b_r
        out += _as_list(inner(ct, sign * eta * self._lap(bt)))
        out += _as_list(inner(ct, sign * (-eta) * (1 / r**2) * bt))
        out += _as_list(inner(ct, sign * B0 * dz(ut)))
        out += _as_list(inner(ct, sign * rOmp * br))
        # b_z induction:  eta L b_z + B0 du_z/dz
        out += _as_list(inner(cz, sign * eta * self._lap(bz)))
        out += _as_list(inner(cz, sign * B0 * dz(uz)))
        return out

    def _build_operators(self):
        r = self.r
        dt = self.dt
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ur, ut, uz, p, br, bt, bz = up
        vr, vt, vz, q, cr, ct, cz = vq
        imp = []
        for vv, uu in ((vr, ur), (vt, ut), (vz, uz), (cr, br), (ct, bt), (cz, bz)):
            imp += _as_list(inner(vv, uu * (1.0 / dt)))
        imp += self._Lxx(ur, ut, uz, br, bt, bz, vr, vt, vz, cr, ct, cz, sign=-0.5)
        imp += _as_list(inner(vr, Dx(p, 1, 1)))  # +dPi/dr
        imp += _as_list(inner(vz, Dx(p, 0, 1)))  # +dPi/dz
        imp += _as_list(inner(q, Dx(ur, 1, 1)))  # continuity
        imp += _as_list(inner(q, (1 / r) * ur))
        imp += _as_list(inner(q, Dx(uz, 0, 1)))
        self.Limp = la.BlockMatrixSolver(imp)

        ue = TrialFunction(self.VE)
        ve = TestFunction(self.VE)
        eur, eut, euz, ebr, ebt, ebz = ue
        tur, tut, tuz, tbr, tbt, tbz = ve
        exp = []
        for vv, uu in (
            (tur, eur),
            (tut, eut),
            (tuz, euz),
            (tbr, ebr),
            (tbt, ebt),
            (tbz, ebz),
        ):
            exp += _as_list(inner(vv, uu * (1.0 / dt)))
        exp += self._Lxx(
            eur, eut, euz, ebr, ebt, ebz, tur, tut, tuz, tbr, tbt, tbz, sign=0.5
        )
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    def _phys(self, i):
        """field, d/dr, d/dz of evolving-field component ``i`` (cached projects)."""
        pf = (self.dealias, self.dealias) if self.dealias > 1.0 else None

        def bw(f):
            return np.asarray(f.backward(padding_factor=pf) if pf else f.backward())

        field = bw(self.x[i])
        return field, bw(self._Pdr[i]()), bw(self._Pdz[i]())

    def _set_hat(self, k, padded_values):
        """Dealias a working-grid product into the spectral buffer ``_eps[k]``."""
        if self.dealias > 1.0:
            ap = Array(self.T0p)
            ap[:] = padded_values
            self._eps[k][:] = ap.forward()
        else:
            ar = Array(self.T0)
            ar[:] = padded_values
            self._eps[k][:] = ar.forward(Function(self.T0))

    def nonlinear(self, out):
        ur, urr, urz = self._phys(0)
        ut, utr, utz = self._phys(1)
        uz, uzr, uzz = self._phys(2)
        br, brr, brz = self._phys(3)
        bt, btr, btz = self._phys(4)
        bz, bzr, bzz = self._phys(5)
        ir = self.inv_r_p
        # momentum:  N_u = (u.grad)u - (b.grad)b   (subtracted in step)
        au_r = ur * urr + uz * urz - ut * ut * ir
        au_t = ur * utr + uz * utz + ur * ut * ir
        au_z = ur * uzr + uz * uzz
        lb_r = br * brr + bz * brz - bt * bt * ir
        lb_t = br * btr + bz * btz + br * bt * ir
        lb_z = br * bzr + bz * bzz
        nu_r, nu_t, nu_z = au_r - lb_r, au_t - lb_t, au_z - lb_z
        # induction EMF eps = u x b  (axisymmetric) -> buffers _eps[0,1,2].
        # ORDERING INVARIANT: _Petz/_Perz/_Pezr/_Petr hold *symbolic* references to
        # _eps[0,1,2], so the EMF-curl terms (nb_*) must be fully materialised into
        # numpy arrays *before* _eps is reused for the momentum dealiasing below.
        # Do not move the momentum-dealiasing block above the nb_* lines.
        self._set_hat(0, ut * bz - uz * bt)  # eps_r
        self._set_hat(1, uz * br - ur * bz)  # eps_t
        self._set_hat(2, ur * bt - ut * br)  # eps_z
        et_phys = np.asarray(self._eps[1].backward())
        # N_b = -curl(eps):  (curl)_r=-d_z e_t, _t=d_z e_r-d_r e_z, _z=d_r e_t+e_t/r
        nb_r = np.asarray(self._Petz().backward())  # +d_z e_t
        nb_t = -np.asarray(self._Perz().backward()) + np.asarray(
            self._Pezr().backward()
        )
        nb_z = -np.asarray(self._Petr().backward()) - et_phys * self.inv_r
        # dealias the momentum products onto the standard grid (REUSES _eps -- the
        # EMF-curl terms above are already materialised, so this is safe)
        if self.dealias > 1.0:
            for vals, k in ((nu_r, 0), (nu_t, 1), (nu_z, 2)):
                self._set_hat(k, vals)
            nu_r = np.asarray(self._eps[0].backward())
            nu_t = np.asarray(self._eps[1].backward())
            nu_z = np.asarray(self._eps[2].backward())
        ar = Array(self.T0)

        def proj(test, vals):
            ar[:] = vals
            return inner(test, ar)

        out[0] = proj(self.vu, nu_r)
        out[1] = proj(self.vu, nu_t)
        out[2] = proj(self.vu, nu_z)
        out[3] = proj(self.vbr, nb_r)
        out[4] = proj(self.vbt, nb_t)
        out[5] = proj(self.vbz, nb_z)
        return out

    # ------------------------------------------------------------------
    def step(self):
        self.nonlinear(self.N_hat)
        rhs_e = Function(self.VE)
        rhs_e = self.Lexp.matvec(self.x, rhs_e)
        for i in range(6):
            if self._have_old:
                e = rhs_e[i] - (1.5 * self.N_hat[i] - 0.5 * self.N_old[i])
            else:
                e = rhs_e[i] - self.N_hat[i]
            # VE order (u,u,u,b,b,b) -> VQ order (u,u,u,Pi,b,b,b)
            self.rhs[i if i < 3 else i + 1] = e
        self.rhs[3] = 0.0
        self.sol = self.Limp(self.rhs, u=self.sol, constraints=((3, 0, 0),))
        for i in range(6):
            self.x[i] = self.sol[i if i < 3 else i + 1]
        self.p_hat[:] = self.sol[3]
        self.N_old[:] = self.N_hat
        self._have_old = True
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        self._t += self.dt
        self._tstep += 1

    # ------------------------------------------------------------------
    def seed_linear_eigenmode(self, kz_mode=1, amp=1e-6, which=0):
        """Inject the leading MRI eigenmode (m=0) at axial mode ``kz_mode``.

        Uses :class:`taylor_couette_mri.TaylorCouetteMRI` at the matching
        ``(Nr, family, nu, eta_mag, B0)``; the seven eigenvector blocks
        (``u_r,u_th,u_z,Pi,b_r,b_th,b_z``) are injected into Fourier mode
        ``kz_mode`` (radial bases coincide).  Returns the linear eigenvalue.
        """
        from taylor_couette_mri import TaylorCouetteMRI

        kz = 2 * np.pi * kz_mode / self.Lz
        lin = TaylorCouetteMRI(
            self.base,
            B0=self.B0,
            nu=self.nu,
            eta_mag=self.eta_mag,
            N=self.Nr,
            family=self.family,
        )
        w, V = lin.eigs(0, kz, n_return=which + 1)
        n = lin.n
        vec = V[:, which]
        # VE field comp -> eigenvector block and target space
        blocks = [
            (0, 0, self.TD),
            (1, 1, self.TD),
            (2, 2, self.TD),
            (3, 4, self.TD),
            (4, 5, self.Tbt),
            (5, 6, self.Tbz),
        ]
        for ve_i, blk, space in blocks:
            f = Function(space)
            f[:] = 0.0
            f[kz_mode, :n] = vec[blk * n : (blk + 1) * n] * amp
            self.x[ve_i] = f
        self._have_old = False
        return complex(w[which])

    def set_random(self, amp=1e-3, seed=0, magnetic=True):
        """Random **divergence-free** velocity IC + optional solenoidal toroidal b.

        The meridional ``(u_r, u_z)`` flow is built from a Stokes stream function
        ``psi = g(r) sum_k (a_k cos kz z + b_k sin kz z)`` (``u_r = -(1/r) dpsi/dz``,
        ``u_z = (1/r) dpsi/dr``, ``g = sin^2(pi (r-R1)/d)``) so ``div(u) = 0`` and
        no-slip hold exactly; ``u_theta`` is an independent wall-vanishing swirl.
        Seeding ``u`` *exactly* divergence-free matters because ``b`` is never
        pressure-projected: a non-solenoidal ``u`` would inject
        ``div(b) ~ dt B0 d_z div(u)`` through the imposed-field induction.

        The magnetic seed is **purely toroidal** (``b_theta`` only, ``b_r=b_z=0``):
        ``div(b) = (1/r) d(r b_r)/dr + db_z/dz`` does not involve ``b_theta`` for
        axisymmetric flow, so this is solenoidal by construction; the poloidal
        field develops dynamically.  For a solenoidal poloidal seed use
        :meth:`seed_linear_eigenmode`.
        """
        rng = np.random.default_rng(seed)
        R1 = self.base.R1
        d = self.base.gap
        X = self.T0.local_mesh(True)
        zz, rr = X[0], X[1]
        arg = np.pi * (rr - R1) / d
        g = np.sin(arg) ** 2  # g = g' = 0 at walls
        gp = (2 * np.pi / d) * np.sin(arg) * np.cos(arg)
        ur = np.zeros_like(rr)
        uz = np.zeros_like(rr)
        ut = np.zeros_like(rr)
        for k in range(1, max(1, self.Nz // 3) + 1):
            kz = 2 * np.pi * k / self.Lz
            ak, bk = rng.standard_normal(), rng.standard_normal()
            phase = ak * np.cos(kz * zz) + bk * np.sin(kz * zz)
            dphase = kz * (-ak * np.sin(kz * zz) + bk * np.cos(kz * zz))
            ur += -(1.0 / rr) * g * dphase  # = -(1/r) dpsi/dz
            uz += (1.0 / rr) * gp * phase  # =  (1/r) dpsi/dr
            ut += np.sin(arg) * phase  # wall-vanishing swirl
        scale = amp / max(np.abs(ur).max(), np.abs(uz).max(), np.abs(ut).max(), 1e-30)
        for i, fld in ((0, ur), (1, ut), (2, uz)):
            a = Array(self.TD)
            a[:] = scale * fld
            self.x[i] = a.forward(Function(self.TD))
        # magnetic: b_r = b_z = 0; toroidal b_theta random (solenoidal for axisym)
        z3 = Array(self.TD)
        z3[:] = 0.0
        self.x[3] = z3.forward(Function(self.TD))
        bt = Array(self.Tbt)
        bt[:] = amp * np.sin(arg) * rng.standard_normal(bt.shape) if magnetic else 0.0
        self.x[4] = bt.forward(Function(self.Tbt))
        z5 = Array(self.Tbz)
        z5[:] = 0.0
        self.x[5] = z5.forward(Function(self.Tbz))
        self._have_old = False

    # ------------------------------------------------------------------
    def fields_physical(self):
        return [self.x[i].backward() for i in range(6)]

    def energy(self):
        f = self.fields_physical()
        ek = 0.5 * inner(1, (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) * self.rphys)
        em = 0.5 * inner(1, (f[3] ** 2 + f[4] ** 2 + f[5] ** 2) * self.rphys)
        return float(ek), float(em)

    def _div(self, fr_hat, fz_hat):
        dfr = np.asarray(project(Dx(fr_hat, 1, 1), self.T0).backward())
        dfz = np.asarray(project(Dx(fz_hat, 0, 1), self.T0).backward())
        fr = np.asarray(fr_hat.backward())
        return float(np.abs(dfr + fr * self.inv_r + dfz).max())

    def divergences(self):
        return self._div(self.x[0], self.x[2]), self._div(self.x[3], self.x[5])

    def diagnostics(self, t, tstep):
        ek, em = self.energy()
        du, db = self.divergences()
        return {
            "t": float(t),
            "tstep": int(tstep),
            "Ekin": ek,
            "Emag": em,
            "E": ek + em,
            "divu": du,
            "divb": db,
        }

    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
        # time accumulates across successive run() calls (lazy-init)
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        nsteps = int(round(end_time / self.dt))
        for k in range(1, nsteps + 1):
            self.step()
            if assert_finite and not np.all(np.isfinite(self.x)):
                raise RuntimeError(f"non-finite field at t={self._t:g}")
            if (moderror and self._tstep % moderror == 0) or k == nsteps:
                d = self.diagnostics(self._t, self._tstep)
                if on_diag is not None:
                    on_diag(d)
                elif moderror and comm.Get_rank() == 0:
                    print(
                        f"t={d['t']:8.4f} Ekin={d['Ekin']:.4e} Emag={d['Emag']:.4e} "
                        f"divu={d['divu']:.1e} divb={d['divb']:.1e}"
                    )
        return self.diagnostics(self._t, self._tstep)

    # ------------------------------------------------------------------
    # checkpoint / restart (exact CNAB2 continuation)
    # ------------------------------------------------------------------
    def state_dict(self):
        """Serializable state for an exact restart: the six (u, b) field
        coefficients, the Adams-Bashforth-2 history, and the clock."""
        return {
            "x": np.array(self.x, copy=True),
            "N_old": np.array(self.N_old, copy=True),
            "have_old": bool(self._have_old),
            "t": float(getattr(self, "_t", 0.0)),
            "tstep": int(getattr(self, "_tstep", 0)),
        }

    def load_state_dict(self, state):
        """Restore a checkpoint produced by :meth:`state_dict`."""
        self.x[:] = state["x"]
        self.N_old[:] = state["N_old"]
        self._have_old = bool(state["have_old"])
        self._t = float(state["t"])
        self._tstep = int(state["tstep"])
        return self


# ===========================================================================
# Full 3D MHD / MRI solver (azimuthal Fourier modes m != 0)
# ===========================================================================
class TaylorCouetteMRIDNS:
    r"""Full 3D resistive-MHD Taylor-Couette DNS with an imposed axial field.

    Combines the azimuthal-Fourier 3D machinery of :class:`TaylorCouetteDNS`
    with the imposed-axial-field MHD physics of :class:`AxisymmetricMRIDNS`,
    giving the nonlinear, time-stepping companion to the linear MRI eigensolver
    :class:`taylor_couette_mri.TaylorCouetteMRI` for *non-axisymmetric*
    (``m != 0``) perturbations.  Fields ``(u, b)`` depend on ``(theta, z, r)``;
    ``theta`` and ``z`` are Fourier, ``r`` is a no-slip / conducting-wall radial
    basis.  The base state is ``W = V(r) e_theta``, ``B = B0 e_z`` (Alfven units,
    ``v_A = B0``); the perturbation feels the full set of linear couplings of
    ``TaylorCouetteMRI.assemble_parts(m, kz)``:

      * base-shear advection ``-Omega d/dtheta`` (``-i m Omega`` per mode) on
        every velocity *and* field component;
      * centrifugal/Coriolis ``+2 Omega u_theta`` (r) and ``-2a u_r`` (theta);
      * the MRI field-stretching source ``r Omega' b_r`` feeding ``b_theta``;
      * imposed-field Lorentz ``+B0 db/dz`` and induction ``+B0 du/dz``;
      * viscous/resistive cross-coupling ``-/+ (2/r**2) d(.)_{theta/r}/dtheta``
        between the ``r`` and ``theta`` components of ``u`` and of ``b``;
      * full scalar Laplacian ``L f = f_rr + f_r/r + f_{theta theta}/r**2 + f_zz``;
      * continuity ``u_r,r + u_r/r + u_theta,theta/r + u_z,z = 0``.

    The quadratic nonlinearities carry every cylindrical metric term: the
    Reynolds/Maxwell advection ``(u.grad)u - (b.grad)b`` and the induction EMF
    curl ``-curl(u x b)`` (now with the azimuthal ``(1/r) d/dtheta`` curl pieces).
    Time stepping, the coupled per-``(m, kz)`` 7-field velocity-pressure block
    solve (``u_r, u_theta, u_z, Pi, b_r, b_theta, b_z``) so ``div(u) = 0``
    exactly, the total-pressure formulation ``Pi = p + B0 b_z``, and 3/2-rule
    dealiasing are exactly as in :class:`AxisymmetricMRIDNS`, now in 3D.

    Perfectly-conducting walls are *m-independent* here: ``b_r = 0`` forces
    ``d b_r/dtheta = 0`` on the wall, so the tangential-E conditions reduce to
    ``d(r b_theta)/dr = 0`` (Robin) and ``b_z' = 0`` (Neumann) for every ``m`` --
    the same radial bases as the axisymmetric solver and the eigensolver.
    Restricting the azimuthal content to ``m = 0`` reproduces the axisymmetric
    MRI results.
    """

    def __init__(
        self,
        base: CircularCouette,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=40,
        Ntheta=8,
        Nz=32,
        Lz=None,
        dt=2.0e-3,
        family="L",
        dealias=1.5,
    ):
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nr = int(Nr)
        self.Ntheta = int(Ntheta)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.0 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        self.Rm = base.Omega1 * base.R1 * base.gap / self.eta_mag
        self.Pm = self.nu / self.eta_mag
        self.S = self.B0 * base.gap / self.eta_mag  # Lundquist number
        dom = (base.R1, base.R2)
        Jm = 0.5 * (base.R2 - base.R1)

        # theta: complex Fourier;  z: real Fourier;  r: Dirichlet / conducting
        self.Ft = FunctionSpace(
            self.Ntheta, "Fourier", dtype="D", domain=(0, 2 * math.pi)
        )
        self.Fz = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nr, family, bc=(0, 0), domain=dom)  # u, b_r
        self.S0 = FunctionSpace(self.Nr, family, domain=dom)  # orthogonal
        self.SP = FunctionSpace(self.Nr, family, domain=dom)
        self.SP.slice = lambda: slice(0, self.Nr - 2)
        self.Sbt = FunctionSpace(
            self.Nr,
            family,
            domain=dom,  # b_theta Robin
            bc={"left": {"R": (base.R1 / Jm, 0)}, "right": {"R": (base.R2 / Jm, 0)}},
        )
        self.Sbz = FunctionSpace(
            self.Nr,
            family,
            domain=dom,  # b_z Neumann
            bc={"left": {"N": 0}, "right": {"N": 0}},
        )

        ax = (2, 0, 1)  # radial (axis 2) is the solve axis
        self.TD = TensorProductSpace(comm, (self.Ft, self.Fz, self.SD), axes=ax)
        self.T0 = TensorProductSpace(comm, (self.Ft, self.Fz, self.S0), axes=ax)
        self.TP = TensorProductSpace(
            comm, (self.Ft, self.Fz, self.SP), axes=ax, modify_spaces_inplace=True
        )
        self.Tbt = TensorProductSpace(comm, (self.Ft, self.Fz, self.Sbt), axes=ax)
        self.Tbz = TensorProductSpace(comm, (self.Ft, self.Fz, self.Sbz), axes=ax)

        # u_r,u_th,u_z, Pi, b_r,b_th,b_z   and the 6 evolving fields (no pressure)
        self.VQ = CompositeSpace(
            [self.TD, self.TD, self.TD, self.TP, self.TD, self.Tbt, self.Tbz]
        )
        self.VE = CompositeSpace(
            [self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz]
        )

        self.r = self.TD.coors.psi[2]  # radial symbol (axis 2)
        X = self.T0.local_mesh(True)
        self.rphys = X[2]
        self.inv_r = 1.0 / self.rphys
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias, self.dealias))
            self.inv_r_p = 1.0 / self.T0p.local_mesh(True)[2]
        else:
            self.T0p = None
            self.inv_r_p = self.inv_r

        self._build_operators()

        self.x = Function(self.VE)  # (u_r,u_th,u_z, b_r,b_th,b_z)
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VE)
        self.N_old = Function(self.VE)
        self._have_old = False
        self.vu = TestFunction(self.TD)
        self.vbr = TestFunction(self.TD)
        self.vbt = TestFunction(self.Tbt)
        self.vbz = TestFunction(self.Tbz)

        # Cached projections onto the orthogonal space (assembled once; recomputed
        # against the in-place-updated fields each step).  See the INVARIANT note
        # in AxisymmetricMRIDNS: each ``Project`` captures ``self.x[i]`` / ``_eps[k]``
        # symbolically, so ``step()`` must update them with in-place numpy
        # item-assignment (``self.x[i] = ...`` into the composite row, ``_eps[k][:] =``)
        # -- never a rebind that allocates a fresh array -- or the cached Projects
        # would evaluate stale data.
        self._Pdr = [Project(Dx(self.x[i], 2, 1), self.T0) for i in range(6)]  # d/dr
        self._Pdt = [
            Project(Dx(self.x[i], 0, 1), self.T0) for i in range(6)
        ]  # d/dtheta
        self._Pdz = [Project(Dx(self.x[i], 1, 1), self.T0) for i in range(6)]  # d/dz
        self._eps = [Function(self.T0) for _ in range(3)]  # EMF eps_r, eps_t, eps_z
        # EMF-curl pieces (3D adds the azimuthal d/dtheta terms _Pezt, _Pert)
        self._Petz = Project(Dx(self._eps[1], 1, 1), self.T0)  # d eps_t / dz
        self._Petr = Project(Dx(self._eps[1], 2, 1), self.T0)  # d eps_t / dr
        self._Perz = Project(Dx(self._eps[0], 1, 1), self.T0)  # d eps_r / dz
        self._Pezr = Project(Dx(self._eps[2], 2, 1), self.T0)  # d eps_z / dr
        self._Pezt = Project(Dx(self._eps[2], 0, 1), self.T0)  # d eps_z / dtheta
        self._Pert = Project(Dx(self._eps[0], 0, 1), self.T0)  # d eps_r / dtheta

        if comm.Get_rank() == 0:
            print(f"TaylorCouetteMRIDNS(3D): {base.describe()}")
            print(
                f"  B0={self.B0:g} nu={self.nu:g} eta={self.eta_mag:g} "
                f"Re={self.Re:.2f} Rm={self.Rm:.2f} Pm={self.Pm:g} S={self.S:.3f}"
            )
            print(
                f"  Nr={self.Nr} Ntheta={self.Ntheta} Nz={self.Nz} Lz={self.Lz:.4f} "
                f"dt={self.dt:g} dealias={self.dealias:g}"
            )

    # ------------------------------------------------------------------
    def _lap(self, u):
        r = self.r
        return (
            Dx(u, 2, 2) + (1 / r) * Dx(u, 2, 1) + (1 / r**2) * Dx(u, 0, 2) + Dx(u, 1, 2)
        )

    def _Lxx(self, ur, ut, uz, br, bt, bz, vr, vt, vz, cr, ct, cz, sign):
        """3D evolving-field linear MHD operator (no pressure/continuity), x sign."""
        r = self.r
        nu, eta, B0 = self.nu, self.eta_mag, self.B0
        a = self.base.a
        Om = self.base.a + self.base.b / r**2
        rOmp = -2 * self.base.b / r**2  # r dOmega/dr
        dz = lambda f: Dx(f, 1, 1)  # axis 1 = z
        dth = lambda f: Dx(f, 0, 1)  # axis 0 = theta
        out = []
        # r-momentum
        out += _as_list(inner(vr, sign * nu * self._lap(ur)))
        out += _as_list(inner(vr, sign * (-nu) * (1 / r**2) * ur))
        out += _as_list(inner(vr, sign * (-nu) * (2 / r**2) * dth(ut)))
        out += _as_list(inner(vr, sign * (-Om) * dth(ur)))  # -Omega d/dtheta
        out += _as_list(inner(vr, sign * (2 * Om) * ut))  # +2 Omega u_theta
        out += _as_list(inner(vr, sign * B0 * dz(br)))  # +B0 db_r/dz
        # theta-momentum
        out += _as_list(inner(vt, sign * nu * self._lap(ut)))
        out += _as_list(inner(vt, sign * (-nu) * (1 / r**2) * ut))
        out += _as_list(inner(vt, sign * nu * (2 / r**2) * dth(ur)))
        out += _as_list(inner(vt, sign * (-Om) * dth(ut)))
        out += _as_list(inner(vt, sign * (-2 * a) * ur))  # -2a u_r
        out += _as_list(inner(vt, sign * B0 * dz(bt)))
        # z-momentum
        out += _as_list(inner(vz, sign * nu * self._lap(uz)))
        out += _as_list(inner(vz, sign * (-Om) * dth(uz)))
        out += _as_list(inner(vz, sign * B0 * dz(bz)))
        # b_r induction
        out += _as_list(inner(cr, sign * eta * self._lap(br)))
        out += _as_list(inner(cr, sign * (-eta) * (1 / r**2) * br))
        out += _as_list(inner(cr, sign * (-eta) * (2 / r**2) * dth(bt)))
        out += _as_list(inner(cr, sign * (-Om) * dth(br)))  # -Omega d/dtheta
        out += _as_list(inner(cr, sign * B0 * dz(ur)))  # +B0 du_r/dz
        # b_theta induction
        out += _as_list(inner(ct, sign * eta * self._lap(bt)))
        out += _as_list(inner(ct, sign * (-eta) * (1 / r**2) * bt))
        out += _as_list(inner(ct, sign * eta * (2 / r**2) * dth(br)))
        out += _as_list(inner(ct, sign * (-Om) * dth(bt)))
        out += _as_list(inner(ct, sign * rOmp * br))  # r Omega' b_r
        out += _as_list(inner(ct, sign * B0 * dz(ut)))
        # b_z induction
        out += _as_list(inner(cz, sign * eta * self._lap(bz)))
        out += _as_list(inner(cz, sign * (-Om) * dth(bz)))
        out += _as_list(inner(cz, sign * B0 * dz(uz)))
        return out

    def _build_operators(self):
        r = self.r
        dt = self.dt
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ur, ut, uz, p, br, bt, bz = up
        vr, vt, vz, q, cr, ct, cz = vq
        imp = []
        for vv, uu in ((vr, ur), (vt, ut), (vz, uz), (cr, br), (ct, bt), (cz, bz)):
            imp += _as_list(inner(vv, uu * (1.0 / dt)))
        imp += self._Lxx(ur, ut, uz, br, bt, bz, vr, vt, vz, cr, ct, cz, sign=-0.5)
        imp += _as_list(inner(vr, Dx(p, 2, 1)))  # +dPi/dr
        imp += _as_list(inner(vt, (1 / r) * Dx(p, 0, 1)))  # +(1/r) dPi/dtheta
        imp += _as_list(inner(vz, Dx(p, 1, 1)))  # +dPi/dz
        imp += _as_list(inner(q, Dx(ur, 2, 1)))  # continuity
        imp += _as_list(inner(q, (1 / r) * ur))
        imp += _as_list(inner(q, (1 / r) * Dx(ut, 0, 1)))
        imp += _as_list(inner(q, Dx(uz, 1, 1)))
        self.Limp = la.BlockMatrixSolver(imp)

        ue = TrialFunction(self.VE)
        ve = TestFunction(self.VE)
        eur, eut, euz, ebr, ebt, ebz = ue
        tur, tut, tuz, tbr, tbt, tbz = ve
        exp = []
        for vv, uu in (
            (tur, eur),
            (tut, eut),
            (tuz, euz),
            (tbr, ebr),
            (tbt, ebt),
            (tbz, ebz),
        ):
            exp += _as_list(inner(vv, uu * (1.0 / dt)))
        exp += self._Lxx(
            eur, eut, euz, ebr, ebt, ebz, tur, tut, tuz, tbr, tbt, tbz, sign=0.5
        )
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    def _phys(self, i):
        """field, d/dr, d/dtheta, d/dz of evolving component ``i`` (padded grid)."""
        pf = (self.dealias,) * 3 if self.dealias > 1.0 else None

        def bw(f):
            return np.asarray(f.backward(padding_factor=pf) if pf else f.backward())

        field = bw(self.x[i])
        return field, bw(self._Pdr[i]()), bw(self._Pdt[i]()), bw(self._Pdz[i]())

    def _set_hat(self, k, padded_values):
        """Dealias a working-grid product into the spectral buffer ``_eps[k]``."""
        if self.dealias > 1.0:
            ap = Array(self.T0p)
            ap[:] = padded_values
            self._eps[k][:] = ap.forward()
        else:
            ar = Array(self.T0)
            ar[:] = padded_values
            self._eps[k][:] = ar.forward(Function(self.T0))

    def nonlinear(self, out):
        ur, urr, urt, urz = self._phys(0)
        ut, utr, utt, utz = self._phys(1)
        uz, uzr, uzt, uzz = self._phys(2)
        br, brr, brt, brz = self._phys(3)
        bt, btr, btt, btz = self._phys(4)
        bz, bzr, bzt, bzz = self._phys(5)
        ir = self.inv_r_p
        # momentum:  N_u = (u.grad)u - (b.grad)b   (subtracted in step)
        au_r = ur * urr + (ut * ir) * urt + uz * urz - ut * ut * ir
        au_t = ur * utr + (ut * ir) * utt + uz * utz + ur * ut * ir
        au_z = ur * uzr + (ut * ir) * uzt + uz * uzz
        lb_r = br * brr + (bt * ir) * brt + bz * brz - bt * bt * ir
        lb_t = br * btr + (bt * ir) * btt + bz * btz + br * bt * ir
        lb_z = br * bzr + (bt * ir) * bzt + bz * bzz
        nu_r, nu_t, nu_z = au_r - lb_r, au_t - lb_t, au_z - lb_z
        # induction EMF eps = u x b  -> dealiased buffers _eps[0,1,2].
        # ORDERING INVARIANT (see AxisymmetricMRIDNS): the curl projects hold
        # symbolic references to _eps, so the EMF-curl terms (nb_*) must be fully
        # materialised into numpy arrays *before* _eps is reused for the momentum
        # dealiasing below.  Do not move the momentum block above the nb_* lines.
        self._set_hat(0, ut * bz - uz * bt)  # eps_r
        self._set_hat(1, uz * br - ur * bz)  # eps_t
        self._set_hat(2, ur * bt - ut * br)  # eps_z
        et_phys = np.asarray(self._eps[1].backward())  # eps_t on the std grid
        ir0 = self.inv_r  # std-grid 1/r (eps live on T0)
        # N_b = -curl(eps), 3D cylindrical:
        #   (curl)_r = (1/r) d_th e_z - d_z e_t
        #   (curl)_t = d_z e_r - d_r e_z
        #   (curl)_z = d_r e_t + e_t/r - (1/r) d_th e_r
        nb_r = -ir0 * np.asarray(self._Pezt().backward()) + np.asarray(
            self._Petz().backward()
        )
        nb_t = -np.asarray(self._Perz().backward()) + np.asarray(
            self._Pezr().backward()
        )
        nb_z = (
            -np.asarray(self._Petr().backward())
            - et_phys * ir0
            + ir0 * np.asarray(self._Pert().backward())
        )
        # dealias the momentum products onto the standard grid (REUSES _eps -- the
        # EMF-curl terms above are already materialised, so this is safe)
        if self.dealias > 1.0:
            for vals, k in ((nu_r, 0), (nu_t, 1), (nu_z, 2)):
                self._set_hat(k, vals)
            nu_r = np.asarray(self._eps[0].backward())
            nu_t = np.asarray(self._eps[1].backward())
            nu_z = np.asarray(self._eps[2].backward())
        ar = Array(self.T0)

        def proj(test, vals):
            ar[:] = vals
            return inner(test, ar)

        out[0] = proj(self.vu, nu_r)
        out[1] = proj(self.vu, nu_t)
        out[2] = proj(self.vu, nu_z)
        out[3] = proj(self.vbr, nb_r)
        out[4] = proj(self.vbt, nb_t)
        out[5] = proj(self.vbz, nb_z)
        return out

    # ------------------------------------------------------------------
    def step(self):
        self.nonlinear(self.N_hat)
        rhs_e = Function(self.VE)
        rhs_e = self.Lexp.matvec(self.x, rhs_e)
        for i in range(6):
            if self._have_old:
                e = rhs_e[i] - (1.5 * self.N_hat[i] - 0.5 * self.N_old[i])
            else:
                e = rhs_e[i] - self.N_hat[i]
            # VE order (u,u,u,b,b,b) -> VQ order (u,u,u,Pi,b,b,b)
            self.rhs[i if i < 3 else i + 1] = e
        self.rhs[3] = 0.0
        self.sol = self.Limp(self.rhs, u=self.sol, constraints=((3, 0, 0),))
        for i in range(6):
            self.x[i] = self.sol[i if i < 3 else i + 1]
        self.p_hat[:] = self.sol[3]
        self.N_old[:] = self.N_hat
        self._have_old = True
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        self._t += self.dt
        self._tstep += 1

    # ------------------------------------------------------------------
    def seed_linear_eigenmode(self, m=0, kz_mode=1, amp=1e-6, which=0):
        """Seed the real part of the linear MRI eigenmode at ``(m, kz)``.

        Builds ``q = Re[hat q(r) exp(i(m theta + kz z))]`` for all six evolving
        fields from the :class:`taylor_couette_mri.TaylorCouetteMRI` eigenvector at
        the matching ``(Nr, family, nu, eta_mag, B0)`` (radial bases coincide).
        Returns the linear eigenvalue ``s``; ``Re(s)`` is what the DNS energy
        growth must reproduce.  ``m = 0`` recovers the axisymmetric MRI mode.
        """
        from taylor_couette_mri import TaylorCouetteMRI

        _require_resolved_m(m, self.Ntheta)
        kz = 2 * math.pi * kz_mode / self.Lz
        lin = TaylorCouetteMRI(
            self.base,
            B0=self.B0,
            nu=self.nu,
            eta_mag=self.eta_mag,
            N=self.Nr,
            family=self.family,
        )
        w, V = lin.eigs(m, kz, n_return=which + 1)
        n = lin.n
        X = self.TD.local_mesh(True)
        th, zz, rr = X[0], X[1], X[2]
        rpts = np.asarray(rr[0, 0, :])
        phase = np.exp(1j * (m * th + kz * zz))
        # (VE component, eigenvector block, radial eval space, target TPS)
        blocks = [
            (0, 0, lin.SDv, self.TD),
            (1, 1, lin.SDv, self.TD),
            (2, 2, lin.SDv, self.TD),
            (3, 4, lin.SDv, self.TD),
            (4, 5, lin.Sbt, self.Tbt),
            (5, 6, lin.Sbz, self.Tbz),
        ]
        for ve_i, blk, rspace, space in blocks:
            # The radial-only ``rspace`` is real-dtype, so the complex eigenvector
            # block must be evaluated real/imag separately and recombined --
            # assigning a complex array into a real ``Function`` silently drops the
            # imaginary part.  The MRI eigenvector is genuinely complex even for
            # m=0 (u and b are out of phase), and dropping Im(q) destroys the
            # radial/axial balance that makes the mode divergence-free.
            block = V[blk * n : (blk + 1) * n, which]
            fr_re = Function(rspace)
            fr_re[:] = 0.0
            fr_im = Function(rspace)
            fr_im[:] = 0.0
            fr_re[rspace.slice()] = block.real
            fr_im[rspace.slice()] = block.imag
            prof = np.asarray(fr_re.eval(rpts)) + 1j * np.asarray(fr_im.eval(rpts))
            field = (amp * prof[None, None, :] * phase).real
            a = Array(space)
            a[:] = field
            self.x[ve_i] = a.forward(Function(space))
        self._have_old = False
        return complex(w[which])

    def set_random(self, amp=1e-3, seed=0):
        """Random **divergence-free** axisymmetric velocity IC with ``b = 0``.

        The meridional ``(u_r, u_z)`` flow is built from a Stokes stream function
        ``psi = g(r) sum_k (a_k cos kz z + b_k sin kz z)`` (random ``a_k, b_k``)
        via ``u_r = -(1/r) dpsi/dz``, ``u_z = (1/r) dpsi/dr``, so ``div(u) = 0``
        identically; ``g = sin^2(pi (r-R1)/d)`` makes ``g = g' = 0`` at both walls
        so ``u_r, u_z`` (and the independent wall-vanishing swirl ``u_theta``)
        satisfy no-slip and the discrete divergence is at roundoff.

        Seeding **exactly** divergence-free matters for MHD: ``b`` starts at 0
        (solenoidal) and is grown by the imposed-field induction
        ``B0 du/dz = curl(u x B0 e_z)``, which stays a curl -- hence keeps
        ``div(b) = 0`` -- only while ``u`` is divergence-free.  A raw
        (non-solenoidal) random velocity, even after the first coupled solve
        projects it, still feeds the *un*-projected ``u^n`` to the explicit
        induction term and injects ``div(b) ~ dt B0 d_z div(u^n)`` (``b`` is never
        pressure-projected).  This seed excites the (fastest) axisymmetric MRI
        channel; for a non-axisymmetric or exact-eigenmode IC use
        :meth:`seed_linear_eigenmode`.
        """
        rng = np.random.default_rng(seed)
        R1 = self.base.R1
        d = self.base.gap
        X = self.TD.local_mesh(True)
        zz, rr = X[1], X[2]
        arg = np.pi * (rr - R1) / d
        g = np.sin(arg) ** 2  # g = g' = 0 at walls
        gp = (2 * np.pi / d) * np.sin(arg) * np.cos(arg)
        ur = np.zeros_like(rr)
        uz = np.zeros_like(rr)
        ut = np.zeros_like(rr)
        for k in range(1, max(1, self.Nz // 3) + 1):  # a few resolved kz modes
            kz = 2 * np.pi * k / self.Lz
            ak, bk = rng.standard_normal(), rng.standard_normal()
            phase = ak * np.cos(kz * zz) + bk * np.sin(kz * zz)
            dphase = kz * (-ak * np.sin(kz * zz) + bk * np.cos(kz * zz))
            ur += -(1.0 / rr) * g * dphase  # = -(1/r) dpsi/dz
            uz += (1.0 / rr) * gp * phase  # =  (1/r) dpsi/dr
            ut += np.sin(arg) * phase  # wall-vanishing swirl
        scale = amp / max(np.abs(ur).max(), np.abs(uz).max(), np.abs(ut).max(), 1e-30)
        for i, fld in ((0, ur), (1, ut), (2, uz)):
            a = Array(self.TD)
            a[:] = scale * fld
            self.x[i] = a.forward(Function(self.TD))
        for i, space in ((3, self.TD), (4, self.Tbt), (5, self.Tbz)):
            z = Array(space)
            z[:] = 0.0  # b = 0
            self.x[i] = z.forward(Function(space))
        self._have_old = False

    # ------------------------------------------------------------------
    def fields_physical(self):
        return [self.x[i].backward() for i in range(6)]

    def energy(self):
        f = self.fields_physical()
        ek = 0.5 * inner(1, (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) * self.rphys)
        em = 0.5 * inner(1, (f[3] ** 2 + f[4] ** 2 + f[5] ** 2) * self.rphys)
        return float(ek), float(em)

    def _div(self, fr_hat, ft_hat, fz_hat):
        dfr = np.asarray(project(Dx(fr_hat, 2, 1), self.T0).backward())
        dft = np.asarray(project(Dx(ft_hat, 0, 1), self.T0).backward())
        dfz = np.asarray(project(Dx(fz_hat, 1, 1), self.T0).backward())
        fr = np.asarray(fr_hat.backward())
        return float(np.abs(dfr + fr * self.inv_r + dft * self.inv_r + dfz).max())

    def divergences(self):
        return (
            self._div(self.x[0], self.x[1], self.x[2]),
            self._div(self.x[3], self.x[4], self.x[5]),
        )

    def diagnostics(self, t, tstep):
        ek, em = self.energy()
        du, db = self.divergences()
        return {
            "t": float(t),
            "tstep": int(tstep),
            "Ekin": ek,
            "Emag": em,
            "E": ek + em,
            "divu": du,
            "divb": db,
        }

    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
        # time accumulates across successive run() calls (lazy-init)
        if not hasattr(self, "_t"):
            self._t, self._tstep = 0.0, 0
        nsteps = int(round(end_time / self.dt))
        for k in range(1, nsteps + 1):
            self.step()
            if assert_finite and not np.all(np.isfinite(self.x)):
                raise RuntimeError(f"non-finite field at t={self._t:g}")
            if (moderror and self._tstep % moderror == 0) or k == nsteps:
                d = self.diagnostics(self._t, self._tstep)
                if on_diag is not None:
                    on_diag(d)
                elif moderror and comm.Get_rank() == 0:
                    print(
                        f"t={d['t']:8.4f} Ekin={d['Ekin']:.4e} Emag={d['Emag']:.4e} "
                        f"divu={d['divu']:.1e} divb={d['divb']:.1e}"
                    )
        return self.diagnostics(self._t, self._tstep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _linear_analysis_kz(args, base, mhd=False):
    if args.kz is not None:
        return float(args.kz)
    if args.Lz is not None:
        return 2.0 * math.pi * args.kz_mode / float(args.Lz)
    return (3.0 if mhd else 3.13) / base.gap


def _run_linear_analysis(args, base):
    from _linear_analysis import parse_times, print_eigenvalues, print_transient_growth

    # Honour an explicit --m for the linear analysis (the 1D radial eigensolvers
    # support any azimuthal wavenumber regardless of --Ntheta); default to the
    # axisymmetric m=0 onset when --m is not given.
    m = args.m if args.m is not None else 0
    kz = _linear_analysis_kz(args, base, mhd=args.mhd)
    if args.mhd:
        from taylor_couette_mri import TaylorCouetteMRI

        if args.magnetic_bc == "insulating" and m != 0:
            raise SystemExit(
                "insulating-wall MRI linear analysis is m=0 only; "
                "use --magnetic-bc conducting for m!=0"
            )
        solver = TaylorCouetteMRI(
            base,
            B0=args.B0,
            nu=args.nu,
            eta_mag=args.eta_mag,
            N=args.Nr,
            family=args.family,
            magnetic_bc=args.magnetic_bc,
        )
        label = f"Taylor-Couette MHD/MRI ({args.magnetic_bc} walls)"
        nonmodal_kw = dict(energy=args.energy)
    else:
        from taylor_couette_linear import TaylorCouetteLinear

        solver = TaylorCouetteLinear(base, nu=args.nu, N=args.Nr, family=args.family)
        label = "Taylor-Couette hydro"
        nonmodal_kw = {}

    if args.linear_analysis == "eigs":
        w, _ = solver.eigs(m, kz, n_return=6)
        if comm.Get_rank() == 0:
            print(f"{label} linear eigenvalues: m={m}, kz={kz:g}, N={args.Nr}")
            print_eigenvalues(w)
        return 0

    rows = solver.nonmodal_growth(
        m, kz, parse_times(args.times), n_modes=args.n_modes, **nonmodal_kw
    )
    if comm.Get_rank() == 0:
        norm = f", {args.energy} energy" if args.mhd else ""
        print(
            f"{label} non-modal transient growth: m={m}, kz={kz:g}, N={args.Nr}{norm}"
        )
        print_transient_growth(rows)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Taylor-Couette hydro DNS "
        "(axisymmetric by default; --Ntheta>0 for full 3D)"
    )
    p.add_argument("--R1", type=float, default=1.0)
    p.add_argument("--R2", type=float, default=2.0)
    p.add_argument("--Omega1", type=float, default=1.0)
    p.add_argument("--Omega2", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=1.0e-2)
    p.add_argument("--Nr", type=int, default=48)
    p.add_argument(
        "--Ntheta",
        type=int,
        default=0,
        help="azimuthal Fourier resolution; 0 -> axisymmetric solver",
    )
    p.add_argument("--Nz", type=int, default=32)
    p.add_argument("--Lz", type=float, default=None)
    p.add_argument("--dt", type=float, default=2.0e-3)
    p.add_argument("--end-time", type=float, default=2.0)
    p.add_argument("--family", choices=["L", "C"], default="L")
    p.add_argument("--dealias", type=float, default=1.5)
    p.add_argument("--amp", type=float, default=1.0e-3)
    p.add_argument(
        "--m",
        type=int,
        default=None,
        help="azimuthal wavenumber; DNS 3D seed defaults to 1, "
        "--linear-analysis defaults to 0 (both honour an explicit value)",
    )
    p.add_argument("--kz-mode", type=int, default=1)
    p.add_argument("--moderror", type=int, default=50)
    p.add_argument("--seed-random", action="store_true")
    p.add_argument(
        "--mhd",
        action="store_true",
        help="MHD/MRI solver (imposed axial field B0); axisymmetric "
        "by default, --Ntheta>0 selects the full 3D MRI solver",
    )
    p.add_argument("--B0", type=float, default=0.1, help="imposed axial field (MHD)")
    p.add_argument("--eta-mag", type=float, default=1.0e-3, help="resistivity (MHD)")
    p.add_argument(
        "--linear-analysis",
        choices=["none", "eigs", "nonmodal"],
        default="none",
        help="run a linear eigenvalue or non-modal analysis and exit",
    )
    p.add_argument(
        "--magnetic-bc",
        choices=["conducting", "insulating"],
        default="conducting",
        help="magnetic wall BC for --mhd --linear-analysis (insulating is m=0 only)",
    )
    p.add_argument(
        "--energy",
        choices=["total", "kinetic", "magnetic"],
        default="total",
        help="energy norm for --mhd --linear-analysis nonmodal",
    )
    p.add_argument(
        "--kz",
        type=float,
        default=None,
        help="axial wavenumber for --linear-analysis; overrides --Lz/--kz-mode",
    )
    p.add_argument(
        "--times",
        type=str,
        default="1,5,10,20",
        help="comma-separated times for --linear-analysis nonmodal",
    )
    p.add_argument(
        "--n-modes",
        type=int,
        default=None,
        help="number of finite eigenmodes retained for non-modal analysis",
    )
    args = p.parse_args(argv)

    base = CircularCouette(args.R1, args.R2, args.Omega1, args.Omega2)
    if args.linear_analysis != "none":
        return _run_linear_analysis(args, base)
    # DNS 3D azimuthal seed wavenumber: default 1 when --m is not given.
    m_seed = args.m if args.m is not None else 1
    if args.mhd:
        if args.Ntheta > 0:
            dns = TaylorCouetteMRIDNS(
                base,
                B0=args.B0,
                nu=args.nu,
                eta_mag=args.eta_mag,
                Nr=args.Nr,
                Ntheta=args.Ntheta,
                Nz=args.Nz,
                Lz=args.Lz,
                dt=args.dt,
                family=args.family,
                dealias=args.dealias,
            )
            seed = lambda: dns.seed_linear_eigenmode(
                m=m_seed, kz_mode=args.kz_mode, amp=args.amp
            )
        else:
            dns = AxisymmetricMRIDNS(
                base,
                B0=args.B0,
                nu=args.nu,
                eta_mag=args.eta_mag,
                Nr=args.Nr,
                Nz=args.Nz,
                Lz=args.Lz,
                dt=args.dt,
                family=args.family,
                dealias=args.dealias,
            )
            seed = lambda: dns.seed_linear_eigenmode(kz_mode=args.kz_mode, amp=args.amp)
        if args.seed_random:
            dns.set_random(amp=args.amp)
        else:
            seed()
        d0 = dns.diagnostics(0.0, 0)
        if comm.Get_rank() == 0:
            print(
                f"initial: Ekin={d0['Ekin']:.4e} Emag={d0['Emag']:.4e} "
                f"divu={d0['divu']:.1e} divb={d0['divb']:.1e}"
            )
        final = dns.run(args.end_time, moderror=args.moderror)
        if comm.Get_rank() == 0:
            print(
                f"final:   Ekin={final['Ekin']:.4e} Emag={final['Emag']:.4e} "
                f"divu={final['divu']:.1e} divb={final['divb']:.1e}"
            )
        return 0
    if args.Ntheta > 0:
        dns = TaylorCouetteDNS(
            base,
            nu=args.nu,
            Nr=args.Nr,
            Ntheta=args.Ntheta,
            Nz=args.Nz,
            Lz=args.Lz,
            dt=args.dt,
            family=args.family,
            dealias=args.dealias,
        )
        if args.seed_random:
            dns.set_random(amp=args.amp)
        else:
            dns.set_perturbation(amp=args.amp, m=m_seed, kz_mode=args.kz_mode)
    else:
        dns = AxisymmetricTCDNS(
            base,
            nu=args.nu,
            Nr=args.Nr,
            Nz=args.Nz,
            Lz=args.Lz,
            dt=args.dt,
            family=args.family,
            dealias=args.dealias,
        )
        if args.seed_random:
            dns.set_random(amp=args.amp)
        else:
            dns.set_perturbation(amp=args.amp, kz_mode=args.kz_mode)
    d0 = dns.diagnostics(0.0, 0)
    if comm.Get_rank() == 0:
        print(f"initial: E={d0['E']:.6e} div={d0['div_linf']:.2e}")
    final = dns.run(args.end_time, moderror=args.moderror)
    if comm.Get_rank() == 0:
        print(f"final:   E={final['E']:.6e} div={final['div_linf']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
