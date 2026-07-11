"""Dense Plane-Couette linear eigenvalue and non-modal operators.

The operator is a primitive-variable Chebyshev collocation discretisation in the
wall-normal coordinate ``x`` with Fourier perturbations
``exp(s t + i ky y + i kz z)``.  It is intentionally small and dense because the
demo use case is analysis and smoke testing, not production-scale DNS.

Despite the ``*_jax.py`` filename used by the Couette example suite, this module
mirrors ``couette/_pcf_linear.py`` as a NumPy/SciPy dense reference workflow. It
is not a differentiable JAX Galerkin port; the live shenfun parity tests cover
its modal and non-modal outputs separately from the JAX DNS implementations.
"""

from __future__ import annotations

import numpy as np

from jaxfun.la import (
    NONMODAL_FINITE_CAP as FINITE_CAP,
    finite_eigensystem,
    parse_times,
    print_eigenvalues,
    print_transient_growth,
    transient_growth_from_eigs,
)


def cheb_lobatto(n):
    """Chebyshev-Lobatto points, first/second derivative matrices, weights."""
    n = int(n)
    if n < 4:
        raise ValueError("nx must be at least 4")
    N = n - 1
    j = np.arange(n)
    x = np.cos(np.pi * j / N)
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** j
    X = np.tile(x, (n, 1))
    dX = X.T - X
    D = (c[:, None] / c[None, :]) / (dX + np.eye(n))
    D -= np.diag(np.sum(D, axis=1))
    D2 = D @ D

    theta = np.pi * j / N
    w = np.zeros(n)
    if N == 1:
        w[:] = 1.0
    else:
        ii = np.arange(1, N)
        v = np.ones(N - 1)
        if N % 2 == 0:
            w[0] = 1.0 / (N * N - 1.0)
            w[-1] = w[0]
            for k in range(1, N // 2):
                v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1.0)
            v -= np.cos(N * theta[ii]) / (N * N - 1.0)
        else:
            w[0] = 1.0 / (N * N)
            w[-1] = w[0]
            for k in range(1, (N + 1) // 2):
                v -= 2.0 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1.0)
        w[ii] = 2.0 * v / N
    return x, D, D2, w


class PlaneCouetteLinear:
    """Primitive-variable PCF hydro/MHD linear operator.

    Coordinates follow the PCF demos: ``x`` is wall-normal, ``y`` streamwise, and
    ``z`` spanwise/vertical.  The base flow is ``U(x) e_y`` with constant
    ``Uprime``.  When ``mhd=True`` a uniform imposed field
    ``B0=(0, by, bz)`` is included and magnetic perturbations are projected with
    a magnetic-pressure Lagrange multiplier.

    Magnetic wall BCs (``mhd=True``):

    * ``conducting`` -- normal field ``b_x = 0`` with Neumann tangential field
      (``b_y' = b_z' = 0``); the pseudo-vacuum / perfect-conductor condition.
    * ``dirichlet`` -- all components pinned (``b = 0``); a non-physical
      diagnostic BC, useful only for operator/convergence checks.

    Energy norm.  :meth:`nonmodal_growth` measures gain in the *total*
    perturbation energy by default (kinetic + magnetic, equal weights in Alfven
    units).  Because the shear stretches ``b_x`` into ``b_y`` (the Omega-effect),
    the magnetic field has its own transient growth, so the MHD total-energy gain
    does **not** reduce to the hydrodynamic value as ``B0 -> 0`` -- it returns the
    larger of the (then decoupled) kinetic and magnetic gains.  Pass
    ``energy='kinetic'`` to recover the velocity-only gain (which *does* match the
    hydro result at ``B0=0``).
    """

    def __init__(
        self,
        nx=64,
        nu=1.0e-3,
        eta=1.0e-3,
        Uprime=1.0,
        Uoffset=0.0,
        omega=0.0,
        by=0.0,
        bz=0.0,
        mhd=False,
        magnetic_bc="conducting",
    ):
        self.nx = int(nx)
        self.nu = float(nu)
        self.eta = float(eta)
        self.Uprime = float(Uprime)
        self.Uoffset = float(Uoffset)
        self.omega = float(omega)
        self.by = float(by)
        self.bz = float(bz)
        self.mhd = bool(mhd)
        self.magnetic_bc = magnetic_bc
        if magnetic_bc not in ("conducting", "dirichlet", "pseudo_vacuum"):
            raise ValueError(
                "magnetic_bc must be 'conducting', 'dirichlet' or 'pseudo_vacuum'"
            )
        self.x, self.D, self.D2, self.weights = cheb_lobatto(self.nx)

    @classmethod
    def couette(cls, nx=64, Re=400.0, Rm=None, U_wall=1.0, mhd=False, **kw):
        Rm = Re if Rm is None else Rm
        return cls(
            nx=nx,
            nu=float(U_wall) / float(Re),
            eta=float(U_wall) / float(Rm),
            Uprime=float(U_wall),
            Uoffset=0.0,
            mhd=mhd,
            **kw,
        )

    @classmethod
    def shearpy(
        cls,
        nx=64,
        Re=1000.0,
        Rm=1000.0,
        shear_rate=1.0,
        omega=2.0 / 3.0,
        by=0.0,
        bz=0.025,
        velocity_scale=1.0,
        **kw,
    ):
        return cls(
            nx=nx,
            nu=float(velocity_scale) / float(Re),
            eta=float(velocity_scale) / float(Rm),
            Uprime=-float(shear_rate),
            Uoffset=0.0,
            omega=float(omega),
            by=by,
            bz=bz,
            mhd=True,
            **kw,
        )

    def _blocks(self):
        if self.mhd:
            return {
                "ux": 0,
                "uy": 1,
                "uz": 2,
                "p": 3,
                "bx": 4,
                "by": 5,
                "bz": 6,
                "phi": 7,
            }
        return {"ux": 0, "uy": 1, "uz": 2, "p": 3}

    def _put(self, mat, rb, cb, block):
        n = self.nx
        mat[rb * n : (rb + 1) * n, cb * n : (cb + 1) * n] += block

    def _set_row(self, L, M, row, entries):
        L[row, :] = 0.0
        M[row, :] = 0.0
        for col, values in entries:
            L[row, col] = values

    def _apply_velocity_bcs(self, L, M, b):
        n = self.nx
        for name in ("ux", "uy", "uz"):
            blk = b[name]
            for wall in (0, n - 1):
                row = blk * n + wall
                self._set_row(L, M, row, [(blk * n + wall, 1.0)])

    def _apply_magnetic_bcs(self, L, M, b):
        if not self.mhd:
            return
        n = self.nx
        if self.magnetic_bc == "dirichlet":
            for name in ("bx", "by", "bz"):
                blk = b[name]
                for wall in (0, n - 1):
                    row = blk * n + wall
                    self._set_row(L, M, row, [(blk * n + wall, 1.0)])
            return

        if self.magnetic_bc == "pseudo_vacuum":
            # FJ-09: pseudo-vacuum walls. Tangential field vanishes (Dirichlet on
            # b_y, b_z); the normal gradient of the normal component vanishes
            # (Neumann on b_x, the solenoidal complement of the tangential rows).
            bx, by, bz = b["bx"], b["by"], b["bz"]
            for wall in (0, n - 1):
                self._set_row(
                    L, M, bx * n + wall,
                    [(slice(bx * n, (bx + 1) * n), self.D[wall, :])]
                )
                self._set_row(L, M, by * n + wall, [(by * n + wall, 1.0)])
                self._set_row(L, M, bz * n + wall, [(bz * n + wall, 1.0)])
            return

        bx, by, bz = b["bx"], b["by"], b["bz"]
        for wall in (0, n - 1):
            self._set_row(L, M, bx * n + wall, [(bx * n + wall, 1.0)])
            self._set_row(
                L, M, by * n + wall, [(slice(by * n, (by + 1) * n), self.D[wall, :])]
            )
            self._set_row(
                L, M, bz * n + wall, [(slice(bz * n, (bz + 1) * n), self.D[wall, :])]
            )

    def assemble(self, ky, kz):
        ky = float(ky)
        kz = float(kz)
        k2 = ky * ky + kz * kz
        if k2 <= 0.0:
            raise ValueError("at least one of ky or kz must be non-zero")

        b = self._blocks()
        n = self.nx
        nb = len(b)
        I = np.eye(n, dtype=complex)
        Z = np.zeros((n, n), dtype=complex)
        L = np.zeros((nb * n, nb * n), dtype=complex)
        M = np.zeros_like(L)
        lap = self.D2.astype(complex) - k2 * I
        U = self.Uoffset + self.Uprime * self.x
        adv = -1j * ky * np.diag(U)
        grad_y = 1j * ky * I
        grad_z = 1j * kz * I

        ux, uy, uz, p = b["ux"], b["uy"], b["uz"], b["p"]
        for comp in (ux, uy, uz):
            self._put(L, comp, comp, adv + self.nu * lap)
            self._put(M, comp, comp, I)
        self._put(L, ux, p, -self.D)
        self._put(L, uy, p, -grad_y)
        self._put(L, uz, p, -grad_z)
        self._put(L, uy, ux, (-self.Uprime - 2.0 * self.omega) * I)
        self._put(L, ux, uy, (2.0 * self.omega) * I)

        self._put(L, p, ux, self.D)
        self._put(L, p, uy, grad_y)
        self._put(L, p, uz, grad_z)

        if self.mhd:
            bx, by, bz, phi = b["bx"], b["by"], b["bz"], b["phi"]
            kB = ky * self.by + kz * self.bz
            ikB = 1j * kB
            for ublk, bblk in ((ux, bx), (uy, by), (uz, bz)):
                self._put(L, ublk, bblk, ikB * I)
                self._put(L, bblk, ublk, ikB * I)
                self._put(L, bblk, bblk, adv + self.eta * lap)
                self._put(M, bblk, bblk, I)
            self._put(L, by, bx, self.Uprime * I)
            self._put(L, bx, phi, -self.D)
            self._put(L, by, phi, -grad_y)
            self._put(L, bz, phi, -grad_z)
            self._put(L, phi, bx, self.D)
            self._put(L, phi, by, grad_y)
            self._put(L, phi, bz, grad_z)
        else:
            # Keep shape references explicit for linters; pressure has no mass.
            self._put(L, p, p, Z)

        self._apply_velocity_bcs(L, M, b)
        self._apply_magnetic_bcs(L, M, b)
        return L, M

    def energy_matrix(self, kind="total"):
        """Collocation energy metric.

        ``kind`` selects the norm: ``'kinetic'`` (velocity only), ``'magnetic'``
        (imposed-field perturbation only, requires ``mhd=True``), or ``'total'``
        (both).  For a hydro operator ``'total'`` and ``'kinetic'`` coincide.
        """
        if kind not in ("total", "kinetic", "magnetic"):
            raise ValueError("kind must be 'total', 'kinetic', or 'magnetic'")
        if kind == "magnetic" and not self.mhd:
            raise ValueError("magnetic energy norm requires mhd=True")
        b = self._blocks()
        n = self.nx
        Q = np.zeros((len(b) * n, len(b) * n), dtype=complex)
        W = np.diag(self.weights.astype(float))
        names = []
        if kind in ("total", "kinetic"):
            names += ["ux", "uy", "uz"]
        if self.mhd and kind in ("total", "magnetic"):
            names += ["bx", "by", "bz"]
        for name in names:
            blk = b[name]
            Q[blk * n : (blk + 1) * n, blk * n : (blk + 1) * n] = W
        return Q

    def eigs(self, ky, kz, n_return=8, finite_cap=FINITE_CAP):
        return finite_eigensystem(
            *self.assemble(ky, kz), finite_cap=finite_cap, n_return=n_return
        )

    def growth_rate(self, ky, kz):
        w, _ = self.eigs(ky, kz, n_return=1)
        return float(w[0].real) if len(w) else float("nan")

    def nonmodal_growth(
        self, ky, kz, times, n_modes=None, finite_cap=FINITE_CAP, energy="total"
    ):
        """Optimal transient growth in the selected energy norm.

        ``energy`` is passed to :meth:`energy_matrix` (``'total'`` /
        ``'kinetic'`` / ``'magnetic'``).
        """
        w, V = finite_eigensystem(
            *self.assemble(ky, kz), finite_cap=finite_cap, n_return=n_modes
        )
        return transient_growth_from_eigs(w, V, self.energy_matrix(energy), times)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Plane-Couette linear/non-modal analysis"
    )
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--Re", type=float, default=400.0)
    parser.add_argument("--Rm", type=float, default=None)
    parser.add_argument("--ky", type=float, default=1.0)
    parser.add_argument("--kz", type=float, default=1.0)
    parser.add_argument("--mhd", action="store_true")
    parser.add_argument("--shearpy", action="store_true")
    parser.add_argument("--by", type=float, default=0.0)
    parser.add_argument("--bz", type=float, default=0.025)
    parser.add_argument(
        "--magnetic-bc", choices=["conducting", "dirichlet"], default="conducting"
    )
    parser.add_argument("--nonmodal", action="store_true")
    parser.add_argument("--times", type=str, default="1,5,10,20")
    parser.add_argument("--n-modes", type=int, default=None)
    parser.add_argument(
        "--energy", choices=["total", "kinetic", "magnetic"], default="total"
    )
    args = parser.parse_args(argv)

    if args.shearpy:
        solver = PlaneCouetteLinear.shearpy(
            nx=args.nx,
            Re=args.Re,
            Rm=args.Rm or args.Re,
            by=args.by,
            bz=args.bz,
            magnetic_bc=args.magnetic_bc,
        )
    else:
        solver = PlaneCouetteLinear.couette(
            nx=args.nx,
            Re=args.Re,
            Rm=args.Rm,
            mhd=args.mhd,
            by=args.by,
            bz=args.bz,
            magnetic_bc=args.magnetic_bc,
        )

    if args.nonmodal:
        rows = solver.nonmodal_growth(
            args.ky,
            args.kz,
            parse_times(args.times),
            n_modes=args.n_modes,
            energy=args.energy,
        )
        print_transient_growth(rows)
    else:
        w, _ = solver.eigs(args.ky, args.kz, n_return=8)
        print_eigenvalues(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
