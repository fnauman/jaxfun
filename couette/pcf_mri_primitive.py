r"""
Primitive-variable plane-Couette / shearing-box MRI DNS (ky=0 axisymmetric + 3D)
================================================================================

Two solver classes:
  * ``AxisymmetricPCFMRIDNS`` -- 2D (ky=0) channel-mode MRI / hydro DNS; clean
    modal growth/decay matching the linear eigenvalue to spectral accuracy.
  * ``PCFMRIDNS`` -- the full 3D (streamwise Fourier ky!=0) extension, the
    Cartesian analogue of ``taylor_couette_dns.TaylorCouetteMRIDNS``.


A nonlinear, primitive-variable resistive-MHD DNS of the magnetorotational
instability in a plane Couette / shearing-box geometry, written as the Cartesian
analogue of the (validated) cylindrical ``AxisymmetricMRIDNS`` in
``taylor_couette_dns.py``.

Motivation.  The existing ``pcf_mhd_mri_shearpy.py`` evolves a magnetic **vector
potential** ``A`` (``B = curl A``).  That makes injecting a linear eigenmode hard:
seeding a desired ``B`` requires inverting ``curl(A)=B`` with ``A`` in a
Dirichlet space, a delicate gauge problem.  This solver instead evolves the
magnetic field ``b`` **directly** (primitive variables), exactly like the
Taylor-Couette MHD DNS, so seeding the linear eigenmode is a direct injection of
the eigenvector blocks -- and the DNS growth rate then matches the linear
eigenvalue to spectral accuracy.

Geometry / basis (axisymmetric channel mode, ``k_y = 0``)
---------------------------------------------------------
Cartesian ``(x, y, z)``: ``x`` wall-normal (Chebyshev, no-slip), ``y`` streamwise
(invariant, ``k_y=0``), ``z`` vertical (real Fourier, period ``Lz``).  The
reference wall-normal domain is ``x in (-1, 1)`` (half-gap ``h=1``) so the modes
coincide with ``_pcf_linear`` without any rescaling.

Base state.  Shear ``U = -S x e_y`` (``dU/dx = -S``), rotation ``Omega`` about
``z`` (Coriolis ``2 Omega``), uniform imposed vertical field ``B = B0 e_z`` in
Alfven units.

Formulation.  Primitive ``(u_x, u_y, u_z, p, b_x, b_y, b_z)`` with a coupled
velocity-pressure + magnetic-field saddle point solved per ``k_z`` Fourier mode
(``BlockMatrixSolver``); CN for the linear (viscous/resistive + Coriolis + shear +
imposed-field) operator and AB2 for the quadratic ``(u.grad)u - (b.grad)b`` and
EMF ``-curl(u x b)`` terms (the classic CNAB2 / IMEX of the TC DNS in this repo).

Linearised operator and all signs follow ``_pcf_linear.PlaneCouetteLinear``
(``shearpy``), the validated linear collocation reference for this geometry:

  x-mom:  nu (D2 - kz^2) u_x + 2 Omega u_y      + B0 d_z b_x  - d_x p
  y-mom:  nu (D2 - kz^2) u_y + (S - 2 Omega) u_x + B0 d_z b_y
  z-mom:  nu (D2 - kz^2) u_z                     + B0 d_z b_z  - d_z p
  div u:  d_x u_x + d_z u_z = 0
  b_x:    eta(D2 - kz^2) b_x                     + B0 d_z u_x  - d_x phi
  b_y:    eta(D2 - kz^2) b_y - S b_x             + B0 d_z u_y
  b_z:    eta(D2 - kz^2) b_z                     + B0 d_z u_z  - d_z phi
  div b:  d_x b_x + d_z b_z = 0

Conducting walls: ``u = 0`` (no-slip), ``b_x = 0`` (Dirichlet, zero normal field)
and ``d_x b_y = d_x b_z = 0`` (Neumann tangential).

Author: built for the fn_shenfun project (companion to taylor_couette_dns.py).
"""
import math

import numpy as np

from shenfun import (FunctionSpace, TensorProductSpace, CompositeSpace,
                     TrialFunction, TestFunction, inner, Dx, Function, Array,
                     Project, comm, la, BlockMatrix)


def _as_list(res):
    return res if isinstance(res, list) else [res]


def _bary_interp_matrix(x_src, x_dst):
    """Barycentric-Lagrange interpolation matrix from nodes ``x_src`` to ``x_dst``.

    Exact for the degree ``len(x_src)-1`` polynomial through the source nodes, so
    moving a spectral eigenvector between two well-resolved grids is spectrally
    accurate."""
    x_src = np.asarray(x_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float)
    n = x_src.size
    # barycentric weights
    w = np.ones(n)
    for j in range(n):
        d = x_src[j] - x_src
        d[j] = 1.0
        w[j] = 1.0 / np.prod(d)
    M = np.zeros((x_dst.size, n))
    for i, xx in enumerate(x_dst):
        diff = xx - x_src
        exact = np.where(np.abs(diff) < 1e-14)[0]
        if exact.size:
            M[i, exact[0]] = 1.0
            continue
        t = w / diff
        M[i, :] = t / t.sum()
    return M


class AxisymmetricPCFMRIDNS:
    r"""Primitive-variable plane-Couette MRI DNS (``k_y = 0`` channel mode).

    Parameters
    ----------
    S : float
        Shear rate (base flow ``U = -S x e_y``).
    omega : float
        Rotation rate ``Omega`` (Coriolis ``2 Omega``).
    B0 : float
        Imposed uniform vertical field ``B0`` (Alfven units, ``v_A = B0``).
    nu, eta_mag : float
        Viscosity and magnetic diffusivity.
    Nx, Nz : int
        Wall-normal (Chebyshev) and vertical (Fourier) resolution.
    Lz : float
        Vertical period.
    dt : float
        Time step.
    family : str
        Wall-normal family ('C' Chebyshev recommended).
    dealias : float
        3/2-rule padding factor (1.0 disables).
    """

    def __init__(self, S=1.0, omega=2.0 / 3.0, B0=0.1, nu=1.0e-3, eta_mag=1.0e-3,
                 Nx=48, Nz=16, Lz=1.0, dt=2.0e-3, family="C", dealias=1.0):
        self.S = float(S)
        self.omega = float(omega)
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.Lz = float(Lz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        dom = (-1.0, 1.0)   # half-gap h=1: modes coincide with _pcf_linear

        # ---- spaces: axis 0 = z (Fourier), axis 1 = x (Chebyshev) -----------
        self.F = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nx, family, bc=(0, 0), domain=dom)   # u, b_x
        self.S0 = FunctionSpace(self.Nx, family, domain=dom)             # orthogonal
        self.SP = FunctionSpace(self.Nx, family, domain=dom)             # pressures
        self.SP.slice = lambda: slice(0, self.Nx - 2)
        self.SN = FunctionSpace(self.Nx, family, domain=dom,             # b_y, b_z Neumann
                                bc={"left": {"N": 0}, "right": {"N": 0}})

        ax = (1, 0)
        self.TD = TensorProductSpace(comm, (self.F, self.SD), axes=ax)
        self.T0 = TensorProductSpace(comm, (self.F, self.S0), axes=ax)
        self.TP = TensorProductSpace(comm, (self.F, self.SP), axes=ax,
                                     modify_spaces_inplace=True)
        self.TN = TensorProductSpace(comm, (self.F, self.SN), axes=ax)

        # u_x,u_y,u_z, Pi, b_x,b_y,b_z   and the 6 evolving fields (no pressure)
        self.VQ = CompositeSpace([self.TD, self.TD, self.TD, self.TP,
                                  self.TD, self.TN, self.TN])
        self.VE = CompositeSpace([self.TD, self.TD, self.TD,
                                  self.TD, self.TN, self.TN])

        X = self.T0.local_mesh(True)
        self.zphys = X[0]
        self.xphys = X[1]
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
        else:
            self.T0p = None

        self._build_operators()

        self.x = Function(self.VE)            # (u_x,u_y,u_z, b_x,b_y,b_z)
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VE)
        self.N_old = Function(self.VE)
        self._have_old = False
        self.vu = TestFunction(self.TD)
        self.vbx = TestFunction(self.TD)
        self.vby = TestFunction(self.TN)
        self.vbz = TestFunction(self.TN)

        # cached projections onto the orthogonal space (assembled once; see the
        # in-place-update invariant documented in AxisymmetricMRIDNS).
        self._Pdx = [Project(Dx(self.x[i], 1, 1), self.T0) for i in range(6)]
        self._Pdz = [Project(Dx(self.x[i], 0, 1), self.T0) for i in range(6)]
        self._eps = [Function(self.T0) for _ in range(3)]   # EMF eps_x, eps_y, eps_z
        self._Pezx = Project(Dx(self._eps[2], 1, 1), self.T0)   # d eps_z / dx
        self._Pexz = Project(Dx(self._eps[0], 0, 1), self.T0)   # d eps_x / dz
        self._Peyz = Project(Dx(self._eps[1], 0, 1), self.T0)   # d eps_y / dz
        self._Peyx = Project(Dx(self._eps[1], 1, 1), self.T0)   # d eps_y / dx

        if comm.Get_rank() == 0:
            self.Re = abs(self.S) / self.nu if self.nu else float("inf")
            self.Rm = abs(self.S) / self.eta_mag if self.eta_mag else float("inf")
            print(f"AxisymmetricPCFMRIDNS: S={self.S:g} Omega={self.omega:g} "
                  f"B0={self.B0:g} nu={self.nu:g} eta={self.eta_mag:g} "
                  f"Nx={self.Nx} Nz={self.Nz} Lz={self.Lz:.4f} dt={self.dt:g} "
                  f"dealias={self.dealias:g}")

    # ------------------------------------------------------------------
    def _lap(self, u):
        """Cartesian Laplacian d_xx + d_zz (no curvature)."""
        return Dx(u, 1, 2) + Dx(u, 0, 2)

    def _Lxx(self, ux, uy, uz, bx, by, bz, vx, vy, vz, cx, cy, cz, sign):
        """Linear MHD operator (no pressure/continuity), CN ``sign``.

        Constant-coefficient plane-Couette shearing-box terms; signs follow
        _pcf_linear.assemble (ky=0): Coriolis 2 Omega, shear (S - 2 Omega),
        omega-effect -S, imposed field B0 d_z."""
        nu, eta, B0, S, Om = self.nu, self.eta_mag, self.B0, self.S, self.omega
        dz = lambda f: Dx(f, 0, 1)
        out = []
        # x-momentum:  nu lap u_x + 2 Om u_y + B0 d_z b_x
        out += _as_list(inner(vx, sign * nu * self._lap(ux)))
        out += _as_list(inner(vx, sign * (2.0 * Om) * uy))
        out += _as_list(inner(vx, sign * B0 * dz(bx)))
        # y-momentum:  nu lap u_y + (S - 2 Om) u_x + B0 d_z b_y
        out += _as_list(inner(vy, sign * nu * self._lap(uy)))
        out += _as_list(inner(vy, sign * (S - 2.0 * Om) * ux))
        out += _as_list(inner(vy, sign * B0 * dz(by)))
        # z-momentum:  nu lap u_z + B0 d_z b_z
        out += _as_list(inner(vz, sign * nu * self._lap(uz)))
        out += _as_list(inner(vz, sign * B0 * dz(bz)))
        # b_x induction:  eta lap b_x + B0 d_z u_x
        out += _as_list(inner(cx, sign * eta * self._lap(bx)))
        out += _as_list(inner(cx, sign * B0 * dz(ux)))
        # b_y induction:  eta lap b_y - S b_x + B0 d_z u_y   (omega effect -S b_x)
        out += _as_list(inner(cy, sign * eta * self._lap(by)))
        out += _as_list(inner(cy, sign * (-S) * bx))
        out += _as_list(inner(cy, sign * B0 * dz(uy)))
        # b_z induction:  eta lap b_z + B0 d_z u_z
        out += _as_list(inner(cz, sign * eta * self._lap(bz)))
        out += _as_list(inner(cz, sign * B0 * dz(uz)))
        return out

    def _build_operators(self):
        dt = self.dt
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ux, uy, uz, p, bx, by, bz = up
        vx, vy, vz, q, cx, cy, cz = vq
        imp = []
        for vv, uu in ((vx, ux), (vy, uy), (vz, uz), (cx, bx), (cy, by), (cz, bz)):
            imp += _as_list(inner(vv, uu * (1.0 / dt)))
        imp += self._Lxx(ux, uy, uz, bx, by, bz, vx, vy, vz, cx, cy, cz, sign=-0.5)
        imp += _as_list(inner(vx, Dx(p, 1, 1)))         # +dPi/dx
        imp += _as_list(inner(vz, Dx(p, 0, 1)))         # +dPi/dz
        imp += _as_list(inner(q, Dx(ux, 1, 1)))         # continuity div u = 0
        imp += _as_list(inner(q, Dx(uz, 0, 1)))
        self.Limp = la.BlockMatrixSolver(imp)

        ue = TrialFunction(self.VE)
        ve = TestFunction(self.VE)
        eux, euy, euz, ebx, eby, ebz = ue
        tux, tuy, tuz, tbx, tby, tbz = ve
        exp = []
        for vv, uu in ((tux, eux), (tuy, euy), (tuz, euz),
                       (tbx, ebx), (tby, eby), (tbz, ebz)):
            exp += _as_list(inner(vv, uu * (1.0 / dt)))
        exp += self._Lxx(eux, euy, euz, ebx, eby, ebz,
                         tux, tuy, tuz, tbx, tby, tbz, sign=0.5)
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    def _phys(self, i):
        pf = (self.dealias, self.dealias) if self.dealias > 1.0 else None

        def bw(f):
            return np.asarray(f.backward(padding_factor=pf) if pf else f.backward())
        return bw(self.x[i]), bw(self._Pdx[i]()), bw(self._Pdz[i]())

    def _set_hat(self, k, padded_values):
        if self.dealias > 1.0:
            ap = Array(self.T0p); ap[:] = padded_values
            self._eps[k][:] = ap.forward()
        else:
            ar = Array(self.T0); ar[:] = padded_values
            self._eps[k][:] = ar.forward(Function(self.T0))

    def nonlinear(self, out):
        ux, uxx, uxz = self._phys(0)
        uy, uyx, uyz = self._phys(1)
        uz, uzx, uzz = self._phys(2)
        bx, bxx, bxz = self._phys(3)
        by, byx, byz = self._phys(4)
        bz, bzx, bzz = self._phys(5)
        # momentum: N_u = (u.grad)u - (b.grad)b   (Cartesian, ky=0 -> only x,z)
        au_x = ux * uxx + uz * uxz
        au_y = ux * uyx + uz * uyz
        au_z = ux * uzx + uz * uzz
        lb_x = bx * bxx + bz * bxz
        lb_y = bx * byx + bz * byz
        lb_z = bx * bzx + bz * bzz
        nu_x, nu_y, nu_z = au_x - lb_x, au_y - lb_y, au_z - lb_z
        # induction EMF eps = u x b -> buffers _eps[0,1,2]; materialise the curl
        # terms BEFORE reusing _eps for momentum dealiasing.
        self._set_hat(0, uy * bz - uz * by)       # eps_x
        self._set_hat(1, uz * bx - ux * bz)       # eps_y
        self._set_hat(2, ux * by - uy * bx)       # eps_z
        # N_b = -curl(eps):  (curl)_x = -d_z eps_y, _y = d_z eps_x - d_x eps_z,
        #                    _z = d_x eps_y    (ky=0)
        nb_x = np.asarray(self._Peyz().backward())                            # +d_z eps_y
        nb_y = -np.asarray(self._Pexz().backward()) + np.asarray(self._Pezx().backward())
        nb_z = -np.asarray(self._Peyx().backward())
        if self.dealias > 1.0:
            for vals, k in ((nu_x, 0), (nu_y, 1), (nu_z, 2)):
                self._set_hat(k, vals)
            nu_x = np.asarray(self._eps[0].backward())
            nu_y = np.asarray(self._eps[1].backward())
            nu_z = np.asarray(self._eps[2].backward())
        ar = Array(self.T0)

        def proj(test, vals):
            ar[:] = vals
            return inner(test, ar)
        out[0] = proj(self.vu, nu_x)
        out[1] = proj(self.vu, nu_y)
        out[2] = proj(self.vu, nu_z)
        out[3] = proj(self.vbx, nb_x)
        out[4] = proj(self.vby, nb_y)
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
            self.rhs[i if i < 3 else i + 1] = e    # VE (u,u,u,b,b,b) -> VQ (u,u,u,Pi,b,b,b)
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
    # diagnostics
    # ------------------------------------------------------------------
    def fields_physical(self):
        # shenfun Arrays (NOT numpy) so inner(1, .) integrates with the measure
        return [self.x[i].backward() for i in range(6)]

    def energy(self):
        f = self.fields_physical()
        ek = 0.5 * inner(1, f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
        em = 0.5 * inner(1, f[3] ** 2 + f[4] ** 2 + f[5] ** 2)
        return float(ek), float(em)

    def _div(self, fx_hat, fz_hat):
        dfx = np.asarray(Project(Dx(fx_hat, 1, 1), self.T0)().backward())
        dfz = np.asarray(Project(Dx(fz_hat, 0, 1), self.T0)().backward())
        dd = Array(self.T0)
        dd[:] = dfx + dfz
        return float(np.sqrt(inner(1, dd * dd)))

    def divergences(self):
        return self._div(self.x[0], self.x[2]), self._div(self.x[3], self.x[5])

    def diagnostics(self, t, tstep):
        ek, em = self.energy()
        du, db = self.divergences()
        return {"t": float(t), "tstep": int(tstep), "Ekin": ek, "Emag": em,
                "E": ek + em, "divu": du, "divb": db}

    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
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
                    print(f"t={d['t']:8.4f} Ekin={d['Ekin']:.4e} Emag={d['Emag']:.4e} "
                          f"divu={d['divu']:.1e} divb={d['divb']:.1e}")
        return self.diagnostics(self._t, self._tstep)

    # ------------------------------------------------------------------
    # initial conditions / eigenmode injection (direct, no gauge inversion)
    # ------------------------------------------------------------------
    def _linear_operator(self):
        from _pcf_linear import PlaneCouetteLinear
        return PlaneCouetteLinear.shearpy(
            nx=self.Nx, Re=abs(self.S) / self.nu, Rm=abs(self.S) / self.eta_mag,
            shear_rate=self.S, omega=self.omega, by=0.0, bz=self.B0,
            velocity_scale=abs(self.S), magnetic_bc="conducting")

    def seed_linear_eigenmode(self, kz_mode=1, amp=1e-6, which=0):
        """Inject the leading linear MRI eigenmode at axial mode ``kz_mode``.

        Uses ``_pcf_linear.PlaneCouetteLinear`` (same parameters, half-gap h=1) to
        get the (u_x,u_y,u_z,b_x,b_y,b_z) eigenvector on the Chebyshev-Lobatto
        grid, interpolates each block to this DNS's wall-normal grid, and injects
        the real field ``Re[ q(x) exp(i kz z) ]`` into Fourier mode ``kz_mode``.
        Returns the linear eigenvalue (physical units)."""
        lin = self._linear_operator()
        kz = 2.0 * math.pi * kz_mode / self.Lz
        w, V = lin.eigs(0.0, kz, n_return=which + 1)
        n = lin.nx
        vec = V[:, which]
        blk = lin._blocks()
        # interpolation Lobatto-grid -> DNS wall-normal quadrature points
        x_dns = np.asarray(self.SD.mesh()).ravel()
        Interp = _bary_interp_matrix(lin.x, x_dns)

        zc = np.cos(kz * self.zphys)
        zs = np.sin(kz * self.zphys)
        names = ("ux", "uy", "uz", "bx", "by", "bz")
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        self.x[:] = 0.0
        for i, (name, space) in enumerate(zip(names, spaces)):
            prof = vec[blk[name] * n:(blk[name] + 1) * n]
            prof_dns = Interp @ prof                      # complex profile on x_dns
            re = prof_dns.real
            im = prof_dns.imag
            # physical real field Re[prof e^{i kz z}] = prof_re cos - prof_im sin
            phys = amp * (re[None, :] * zc - im[None, :] * zs)
            a = Array(space)
            a[:] = phys
            self.x[i] = a.forward(Function(space))
        self._have_old = False
        return complex(w[which])

    def set_perturbation(self, amp=1e-3, kz_mode=1):
        """Deterministic small channel-mode perturbation (no eigenmode), for
        restart/robustness tests that need a non-trivial nonlinear history."""
        kz = 2.0 * math.pi * kz_mode / self.Lz
        wall = (1.0 - self.xphys ** 2)
        self.x[:] = 0.0
        ax = Array(self.TD); ax[:] = amp * wall * np.cos(kz * self.zphys)
        self.x[0] = ax.forward(Function(self.TD))
        ay = Array(self.TD); ay[:] = amp * wall * np.sin(kz * self.zphys)
        self.x[1] = ay.forward(Function(self.TD))
        bt = Array(self.TN); bt[:] = amp * np.cos(kz * self.zphys)
        self.x[4] = bt.forward(Function(self.TN))
        self._have_old = False
        return self

    # ------------------------------------------------------------------
    # checkpoint / restart
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "x": np.array(self.x, copy=True),
            "N_old": np.array(self.N_old, copy=True),
            "have_old": bool(self._have_old),
            "t": float(getattr(self, "_t", 0.0)),
            "tstep": int(getattr(self, "_tstep", 0)),
        }

    def load_state_dict(self, state):
        self.x[:] = state["x"]
        self.N_old[:] = state["N_old"]
        self._have_old = bool(state["have_old"])
        self._t = float(state["t"])
        self._tstep = int(state["tstep"])
        return self

    def linear_growth_rate(self, kz_mode=1, which=0):
        """Linear eigenvalue (physical growth rate) for the seeded mode."""
        lin = self._linear_operator()
        kz = 2.0 * math.pi * kz_mode / self.Lz
        w, _ = lin.eigs(0.0, kz, n_return=which + 1)
        return complex(w[which])

    # ------------------------------------------------------------------
    # hydro (B0=0) eigenmode injection -- plane-Couette stability/decay
    # ------------------------------------------------------------------
    def _hydro_linear_operator(self):
        from _pcf_linear import PlaneCouetteLinear
        return PlaneCouetteLinear(nx=self.Nx, nu=self.nu, eta=self.eta_mag,
                                  Uprime=-self.S, omega=self.omega, mhd=False)

    def seed_hydro_eigenmode(self, kz_mode=1, amp=1e-6, which=0):
        """Seed the leading HYDRO (B0=0) eigenmode (velocity only; b stays 0).

        For non-rotating plane Couette this is a decaying viscous mode (Couette
        flow is linearly stable), so the DNS energy decays at the linear rate --
        the velocity-only analogue of the MRI growth-vs-linear gate. Requires the
        solver to be run with ``B0=0`` so the magnetic field stays identically
        zero. Returns the linear eigenvalue (physical units)."""
        lin = self._hydro_linear_operator()
        kz = 2.0 * math.pi * kz_mode / self.Lz
        w, V = lin.eigs(0.0, kz, n_return=which + 1)
        n = lin.nx
        vec = V[:, which]
        blk = lin._blocks()
        x_dns = np.asarray(self.SD.mesh()).ravel()
        Interp = _bary_interp_matrix(lin.x, x_dns)
        zc = np.cos(kz * self.zphys)
        zs = np.sin(kz * self.zphys)
        self.x[:] = 0.0
        for i, name in enumerate(("ux", "uy", "uz")):
            prof = vec[blk[name] * n:(blk[name] + 1) * n]
            re = (Interp @ prof).real
            im = (Interp @ prof).imag
            a = Array(self.TD)
            a[:] = amp * (re[None, :] * zc - im[None, :] * zs)
            self.x[i] = a.forward(Function(self.TD))
        self._have_old = False
        return complex(w[which])

    def linear_hydro_growth_rate(self, kz_mode=1, which=0):
        lin = self._hydro_linear_operator()
        kz = 2.0 * math.pi * kz_mode / self.Lz
        w, _ = lin.eigs(0.0, kz, n_return=which + 1)
        return complex(w[which])


# ===========================================================================
# Full 3D primitive-variable plane-Couette MRI DNS (streamwise modes ky != 0)
# ===========================================================================
class PCFMRIDNS:
    r"""Full 3D primitive-variable plane-Couette / shearing-box MRI DNS.

    The streamwise-Fourier (``y``) extension of :class:`AxisymmetricPCFMRIDNS`,
    the Cartesian analogue of :class:`taylor_couette_dns.TaylorCouetteMRIDNS`.
    Fields ``(u, b)`` depend on ``(y, z, x)``: ``y`` streamwise (complex Fourier),
    ``z`` vertical (real Fourier), ``x`` wall-normal (Chebyshev, no-slip /
    conducting). Base state ``U = -S x e_y``, rotation ``Omega`` about ``z``,
    imposed vertical field ``B0 e_z`` (Alfven units).

    The ``ky != 0`` modes feel the base-shear advection ``-U d/dy = +S x d/dy`` on
    every component (the term that turns them into sheared waves), plus the
    streamwise pressure-gradient and continuity ``d/dy`` terms; ``ky = 0`` recovers
    the axisymmetric channel mode and its exact modal growth. Signs follow
    ``_pcf_linear.assemble`` (general ``ky, kz``). div(u) is enforced by the coupled
    saddle point; div(b) is preserved by the induction (no magnetic pressure), as
    in the axisymmetric solver and TaylorCouetteMRIDNS.

    NB: the explicit base-shear advection makes ``ky != 0`` modes only *instantane-
    ously* modal (the radial wavenumber shears in time); use it for short
    functional runs (growth ~ the linear rate over a few e-foldings), not long
    saturation runs.
    """

    def __init__(self, S=1.0, omega=2.0 / 3.0, B0=0.1, nu=1.0e-3, eta_mag=1.0e-3,
                 Nx=40, Ny=8, Nz=16, Ly=2.0 * math.pi, Lz=1.0, dt=2.0e-3,
                 family="C", dealias=1.0):
        self.S = float(S)
        self.omega = float(omega)
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.dt = float(dt)
        self.family = family
        self.dealias = float(dealias)
        dom = (-1.0, 1.0)

        # y streamwise (complex Fourier), z vertical (real Fourier), x Chebyshev.
        self.Fy = FunctionSpace(self.Ny, "Fourier", dtype="D", domain=(0, self.Ly))
        self.Fz = FunctionSpace(self.Nz, "Fourier", dtype="d", domain=(0, self.Lz))
        self.SD = FunctionSpace(self.Nx, family, bc=(0, 0), domain=dom)   # u, b_x
        self.S0 = FunctionSpace(self.Nx, family, domain=dom)
        self.SP = FunctionSpace(self.Nx, family, domain=dom)
        self.SP.slice = lambda: slice(0, self.Nx - 2)
        self.SN = FunctionSpace(self.Nx, family, domain=dom,
                                bc={"left": {"N": 0}, "right": {"N": 0}})

        ax = (2, 0, 1)                          # wall-normal x (axis 2) is the solve axis
        self.TD = TensorProductSpace(comm, (self.Fy, self.Fz, self.SD), axes=ax)
        self.T0 = TensorProductSpace(comm, (self.Fy, self.Fz, self.S0), axes=ax)
        self.TP = TensorProductSpace(comm, (self.Fy, self.Fz, self.SP), axes=ax,
                                     modify_spaces_inplace=True)
        self.TN = TensorProductSpace(comm, (self.Fy, self.Fz, self.SN), axes=ax)

        self.VQ = CompositeSpace([self.TD, self.TD, self.TD, self.TP,
                                  self.TD, self.TN, self.TN])
        self.VE = CompositeSpace([self.TD, self.TD, self.TD,
                                  self.TD, self.TN, self.TN])

        self.x_sym = self.TD.coors.psi[2]       # wall-normal coordinate symbol
        X = self.T0.local_mesh(True)
        self.yphys, self.zphys, self.xphys = X[0], X[1], X[2]
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias,) * 3)
        else:
            self.T0p = None

        self._build_operators()

        self.x = Function(self.VE)
        self.p_hat = Function(self.TP)
        self.rhs = Function(self.VQ)
        self.sol = Function(self.VQ)
        self.N_hat = Function(self.VE)
        self.N_old = Function(self.VE)
        self._have_old = False
        self.vu = TestFunction(self.TD)
        self.vbx = TestFunction(self.TD)
        self.vby = TestFunction(self.TN)
        self.vbz = TestFunction(self.TN)

        # cached derivative projects (d/dx axis2, d/dy axis0, d/dz axis1)
        self._Pdx = [Project(Dx(self.x[i], 2, 1), self.T0) for i in range(6)]
        self._Pdy = [Project(Dx(self.x[i], 0, 1), self.T0) for i in range(6)]
        self._Pdz = [Project(Dx(self.x[i], 1, 1), self.T0) for i in range(6)]
        self._eps = [Function(self.T0) for _ in range(3)]   # eps_x, eps_y, eps_z
        self._Peyz = Project(Dx(self._eps[1], 1, 1), self.T0)   # d eps_y / dz
        self._Pezy = Project(Dx(self._eps[2], 0, 1), self.T0)   # d eps_z / dy
        self._Pezx = Project(Dx(self._eps[2], 2, 1), self.T0)   # d eps_z / dx
        self._Pexz = Project(Dx(self._eps[0], 1, 1), self.T0)   # d eps_x / dz
        self._Pexy = Project(Dx(self._eps[0], 0, 1), self.T0)   # d eps_x / dy
        self._Peyx = Project(Dx(self._eps[1], 2, 1), self.T0)   # d eps_y / dx

        if comm.Get_rank() == 0:
            print(f"PCFMRIDNS(3D): S={self.S:g} Omega={self.omega:g} B0={self.B0:g} "
                  f"nu={self.nu:g} eta={self.eta_mag:g} Nx={self.Nx} Ny={self.Ny} "
                  f"Nz={self.Nz} Ly={self.Ly:.4f} Lz={self.Lz:.4f} dt={self.dt:g} "
                  f"dealias={self.dealias:g}")

    # ------------------------------------------------------------------
    def _lap(self, u):
        return Dx(u, 2, 2) + Dx(u, 0, 2) + Dx(u, 1, 2)      # d_xx + d_yy + d_zz

    def _Lxx(self, ux, uy, uz, bx, by, bz, vx, vy, vz, cx, cy, cz, sign):
        nu, eta, B0, S, Om = self.nu, self.eta_mag, self.B0, self.S, self.omega
        dy = lambda f: Dx(f, 0, 1)
        dz = lambda f: Dx(f, 1, 1)
        adv = lambda f: S * self.x_sym * Dx(f, 0, 1)        # -U d/dy = +S x d/dy
        out = []
        # x-momentum
        out += _as_list(inner(vx, sign * nu * self._lap(ux)))
        out += _as_list(inner(vx, sign * (2.0 * Om) * uy))
        out += _as_list(inner(vx, sign * B0 * dz(bx)))
        out += _as_list(inner(vx, sign * adv(ux)))
        # y-momentum
        out += _as_list(inner(vy, sign * nu * self._lap(uy)))
        out += _as_list(inner(vy, sign * (S - 2.0 * Om) * ux))
        out += _as_list(inner(vy, sign * B0 * dz(by)))
        out += _as_list(inner(vy, sign * adv(uy)))
        # z-momentum
        out += _as_list(inner(vz, sign * nu * self._lap(uz)))
        out += _as_list(inner(vz, sign * B0 * dz(bz)))
        out += _as_list(inner(vz, sign * adv(uz)))
        # b_x induction
        out += _as_list(inner(cx, sign * eta * self._lap(bx)))
        out += _as_list(inner(cx, sign * B0 * dz(ux)))
        out += _as_list(inner(cx, sign * adv(bx)))
        # b_y induction (omega effect -S b_x)
        out += _as_list(inner(cy, sign * eta * self._lap(by)))
        out += _as_list(inner(cy, sign * (-S) * bx))
        out += _as_list(inner(cy, sign * B0 * dz(uy)))
        out += _as_list(inner(cy, sign * adv(by)))
        # b_z induction
        out += _as_list(inner(cz, sign * eta * self._lap(bz)))
        out += _as_list(inner(cz, sign * B0 * dz(uz)))
        out += _as_list(inner(cz, sign * adv(bz)))
        return out

    def _build_operators(self):
        dt = self.dt
        up = TrialFunction(self.VQ)
        vq = TestFunction(self.VQ)
        ux, uy, uz, p, bx, by, bz = up
        vx, vy, vz, q, cx, cy, cz = vq
        imp = []
        for vv, uu in ((vx, ux), (vy, uy), (vz, uz), (cx, bx), (cy, by), (cz, bz)):
            imp += _as_list(inner(vv, uu * (1.0 / dt)))
        imp += self._Lxx(ux, uy, uz, bx, by, bz, vx, vy, vz, cx, cy, cz, sign=-0.5)
        imp += _as_list(inner(vx, Dx(p, 2, 1)))         # +dPi/dx
        imp += _as_list(inner(vy, Dx(p, 0, 1)))         # +dPi/dy
        imp += _as_list(inner(vz, Dx(p, 1, 1)))         # +dPi/dz
        imp += _as_list(inner(q, Dx(ux, 2, 1)))         # continuity d_x u_x
        imp += _as_list(inner(q, Dx(uy, 0, 1)))         #          + d_y u_y
        imp += _as_list(inner(q, Dx(uz, 1, 1)))         #          + d_z u_z
        self.Limp = la.BlockMatrixSolver(imp)

        ue = TrialFunction(self.VE)
        ve = TestFunction(self.VE)
        eux, euy, euz, ebx, eby, ebz = ue
        tux, tuy, tuz, tbx, tby, tbz = ve
        exp = []
        for vv, uu in ((tux, eux), (tuy, euy), (tuz, euz),
                       (tbx, ebx), (tby, eby), (tbz, ebz)):
            exp += _as_list(inner(vv, uu * (1.0 / dt)))
        exp += self._Lxx(eux, euy, euz, ebx, eby, ebz,
                         tux, tuy, tuz, tbx, tby, tbz, sign=0.5)
        self.Lexp = BlockMatrix(exp)

    # ------------------------------------------------------------------
    def _phys(self, i):
        pf = (self.dealias,) * 3 if self.dealias > 1.0 else None

        def bw(f):
            return np.asarray(f.backward(padding_factor=pf) if pf else f.backward())
        return (bw(self.x[i]), bw(self._Pdx[i]()), bw(self._Pdy[i]()), bw(self._Pdz[i]()))

    def _set_hat(self, k, vals):
        if self.dealias > 1.0:
            ap = Array(self.T0p); ap[:] = vals
            self._eps[k][:] = ap.forward()
        else:
            ar = Array(self.T0); ar[:] = vals
            self._eps[k][:] = ar.forward(Function(self.T0))

    def nonlinear(self, out):
        ux, uxx, uxy, uxz = self._phys(0)
        uy, uyx, uyy, uyz = self._phys(1)
        uz, uzx, uzy, uzz = self._phys(2)
        bx, bxx, bxy, bxz = self._phys(3)
        by, byx, byy, byz = self._phys(4)
        bz, bzx, bzy, bzz = self._phys(5)
        # momentum N_u = (u.grad)u - (b.grad)b
        au_x = ux * uxx + uy * uxy + uz * uxz
        au_y = ux * uyx + uy * uyy + uz * uyz
        au_z = ux * uzx + uy * uzy + uz * uzz
        lb_x = bx * bxx + by * bxy + bz * bxz
        lb_y = bx * byx + by * byy + bz * byz
        lb_z = bx * bzx + by * bzy + bz * bzz
        nu_x, nu_y, nu_z = au_x - lb_x, au_y - lb_y, au_z - lb_z
        # EMF eps = u x b ; materialise curl terms before reusing _eps for momentum
        self._set_hat(0, uy * bz - uz * by)       # eps_x
        self._set_hat(1, uz * bx - ux * bz)       # eps_y
        self._set_hat(2, ux * by - uy * bx)       # eps_z
        # N_b = -curl(eps):
        #   _x = d_z eps_y - d_y eps_z ; _y = d_x eps_z - d_z eps_x ; _z = d_y eps_x - d_x eps_y
        nb_x = np.asarray(self._Peyz().backward()) - np.asarray(self._Pezy().backward())
        nb_y = np.asarray(self._Pezx().backward()) - np.asarray(self._Pexz().backward())
        nb_z = np.asarray(self._Pexy().backward()) - np.asarray(self._Peyx().backward())
        if self.dealias > 1.0:
            for vals, k in ((nu_x, 0), (nu_y, 1), (nu_z, 2)):
                self._set_hat(k, vals)
            nu_x = np.asarray(self._eps[0].backward())
            nu_y = np.asarray(self._eps[1].backward())
            nu_z = np.asarray(self._eps[2].backward())
        ar = Array(self.T0)

        def proj(test, vals):
            ar[:] = vals
            return inner(test, ar)
        out[0] = proj(self.vu, nu_x)
        out[1] = proj(self.vu, nu_y)
        out[2] = proj(self.vu, nu_z)
        out[3] = proj(self.vbx, nb_x)
        out[4] = proj(self.vby, nb_y)
        out[5] = proj(self.vbz, nb_z)
        return out

    def step(self):
        self.nonlinear(self.N_hat)
        rhs_e = Function(self.VE)
        rhs_e = self.Lexp.matvec(self.x, rhs_e)
        for i in range(6):
            if self._have_old:
                e = rhs_e[i] - (1.5 * self.N_hat[i] - 0.5 * self.N_old[i])
            else:
                e = rhs_e[i] - self.N_hat[i]
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
    def fields_physical(self):
        return [self.x[i].backward() for i in range(6)]

    def energy(self):
        f = self.fields_physical()
        ek = 0.5 * inner(1, f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
        em = 0.5 * inner(1, f[3] ** 2 + f[4] ** 2 + f[5] ** 2)
        return float(ek), float(em)

    def _div(self, fx, fy, fz):
        dfx = np.asarray(Project(Dx(fx, 2, 1), self.T0)().backward())
        dfy = np.asarray(Project(Dx(fy, 0, 1), self.T0)().backward())
        dfz = np.asarray(Project(Dx(fz, 1, 1), self.T0)().backward())
        dd = Array(self.T0); dd[:] = dfx + dfy + dfz
        return float(np.sqrt(inner(1, dd * dd)))

    def divergences(self):
        return (self._div(self.x[0], self.x[1], self.x[2]),
                self._div(self.x[3], self.x[4], self.x[5]))

    def diagnostics(self, t, tstep):
        ek, em = self.energy()
        du, db = self.divergences()
        return {"t": float(t), "tstep": int(tstep), "Ekin": ek, "Emag": em,
                "E": ek + em, "divu": du, "divb": db}

    def run(self, end_time, moderror=0, on_diag=None, assert_finite=True):
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
                    print(f"t={d['t']:8.4f} Ekin={d['Ekin']:.4e} Emag={d['Emag']:.4e} "
                          f"divu={d['divu']:.1e} divb={d['divb']:.1e}")
        return self.diagnostics(self._t, self._tstep)

    # ------------------------------------------------------------------
    def _linear_operator(self):
        from _pcf_linear import PlaneCouetteLinear
        return PlaneCouetteLinear.shearpy(
            nx=self.Nx, Re=abs(self.S) / self.nu, Rm=abs(self.S) / self.eta_mag,
            shear_rate=self.S, omega=self.omega, by=0.0, bz=self.B0,
            velocity_scale=abs(self.S), magnetic_bc="conducting")

    def seed_linear_eigenmode(self, ky_mode=1, kz_mode=1, amp=1e-6, which=0):
        """Inject the leading linear MRI eigenmode at ``(ky, kz)`` as the real field
        ``Re[hat q(x) exp(i(ky y + kz z))]`` (direct block-copy of the _pcf_linear
        eigenvector). ``ky_mode=0`` recovers the axisymmetric channel mode and its
        exact modal growth. Returns the linear eigenvalue (physical units)."""
        lin = self._linear_operator()
        ky = 2.0 * math.pi * ky_mode / self.Ly
        kz = 2.0 * math.pi * kz_mode / self.Lz
        w, V = lin.eigs(ky, kz, n_return=which + 1)
        n = lin.nx
        vec = V[:, which]
        blk = lin._blocks()
        x1d = np.asarray(self.xphys[0, 0, :])
        Interp = _bary_interp_matrix(lin.x, x1d)
        carg = ky * self.yphys + kz * self.zphys
        cc, ss = np.cos(carg), np.sin(carg)
        names = ("ux", "uy", "uz", "bx", "by", "bz")
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        self.x[:] = 0.0
        for i, (name, space) in enumerate(zip(names, spaces)):
            prof = vec[blk[name] * n:(blk[name] + 1) * n]
            re = (Interp @ prof).real
            im = (Interp @ prof).imag
            # Re[prof e^{i(ky y + kz z)}] = re cos - im sin, broadcast over (y,z); re/im over x
            field = amp * (re[None, None, :] * cc - im[None, None, :] * ss)
            a = Array(space); a[:] = field
            self.x[i] = a.forward(Function(space))
        self._have_old = False
        return complex(w[which])

    def state_dict(self):
        return {
            "x": np.array(self.x, copy=True),
            "N_old": np.array(self.N_old, copy=True),
            "have_old": bool(self._have_old),
            "t": float(getattr(self, "_t", 0.0)),
            "tstep": int(getattr(self, "_tstep", 0)),
        }

    def load_state_dict(self, state):
        self.x[:] = state["x"]
        self.N_old[:] = state["N_old"]
        self._have_old = bool(state["have_old"])
        self._t = float(state["t"])
        self._tstep = int(state["tstep"])
        return self
