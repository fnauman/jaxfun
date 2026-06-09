"""Shenfun-Galerkin primitive-variable plane-Couette linear operators.

This is an additive comparison tool.  It intentionally does not replace the
existing dense collocation operator in ``_pcf_linear.py``; instead it assembles a
radial Galerkin generalized eigenproblem with the same primitive variables used
by the Taylor-Couette linear solvers.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import sympy as sp

from shenfun import Dx, FunctionSpace, TestFunction, TrialFunction, inner

sys.path.insert(0, os.path.dirname(__file__))

from _linear_analysis import (  # noqa: E402
    FINITE_CAP,
    finite_eigensystem,
    parse_times,
    print_eigenvalues,
    print_transient_growth,
    transient_growth_from_eigs,
)

x = sp.Symbol("x", real=True)


def _is_zero(coeff) -> bool:
    try:
        return bool(sp.simplify(sp.sympify(coeff)) == 0)
    except (TypeError, ValueError, AttributeError):
        return complex(coeff) == 0


class PlaneCouetteGalerkinLinear:
    """Primitive-variable Galerkin PCF hydro/MHD operator.

    Coordinates match the PCF demos: ``x`` wall-normal, ``y`` streamwise, ``z``
    spanwise/vertical, perturbations proportional to
    ``exp(s t + i ky y + i kz z)``.
    """

    def __init__(
        self,
        N=64,
        family="C",
        domain=(-1.0, 1.0),
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
        self.N = int(N)
        self.family = family
        self.domain = tuple(float(v) for v in domain)
        self.nu = float(nu)
        self.eta = float(eta)
        self.Uprime = float(Uprime)
        self.Uoffset = float(Uoffset)
        self.omega = float(omega)
        self.by = float(by)
        self.bz = float(bz)
        self.mhd = bool(mhd)
        self.magnetic_bc = magnetic_bc
        if magnetic_bc not in ("conducting", "dirichlet"):
            raise ValueError("magnetic_bc must be 'conducting' or 'dirichlet'")

        self.SD = FunctionSpace(self.N, family, bc=(0, 0), domain=self.domain)
        self.SN = FunctionSpace(
            self.N,
            family,
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=self.domain,
        )
        # Match the pressure-space truncation used by the existing shenfun TC and
        # Stokes demos: keep the full quadrature space but solve only N-2 modes.
        self.SP = FunctionSpace(self.N, family, domain=self.domain)
        self.SP.slice = lambda: slice(0, self.N - 2)
        self.n = self.SD.dim()
        if self.SP.dim() != self.n or self.SN.dim() != self.n:
            raise ValueError("velocity, pressure, and Neumann spaces must align")

        spaces = {"ux": self.SD, "uy": self.SD, "uz": self.SD, "p": self.SP}
        if self.mhd:
            if magnetic_bc == "conducting":
                spaces.update(bx=self.SD, by=self.SN, bz=self.SN, phi=self.SP)
            else:
                spaces.update(bx=self.SD, by=self.SD, bz=self.SD, phi=self.SP)
        self.spaces = spaces
        self.tv = {name: TestFunction(space) for name, space in spaces.items()}
        self.tr = {name: TrialFunction(space) for name, space in spaces.items()}

    @classmethod
    def couette(cls, N=64, Re=1000.0, Rm=None, half_gap=1.0, U_wall=1.0,
                mhd=False, **kw):
        Rm = Re if Rm is None else Rm
        domain = (-float(half_gap), float(half_gap))
        return cls(
            N=N,
            domain=domain,
            nu=float(U_wall) * float(half_gap) / float(Re),
            eta=float(U_wall) * float(half_gap) / float(Rm),
            Uprime=float(U_wall) / float(half_gap),
            Uoffset=0.0,
            mhd=mhd,
            **kw,
        )

    @classmethod
    def shearbox(cls, N=64, Re=1000.0, Rm=None, half_gap=1.0,
                 shear_rate=1.0, omega=2.0 / 3.0, by=0.0, bz=0.0,
                 **kw):
        Rm = Re if Rm is None else Rm
        velocity_scale = abs(float(shear_rate)) * float(half_gap)
        return cls(
            N=N,
            domain=(-float(half_gap), float(half_gap)),
            nu=velocity_scale * float(half_gap) / float(Re),
            eta=velocity_scale * float(half_gap) / float(Rm),
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
            return {"ux": 0, "uy": 1, "uz": 2, "p": 3,
                    "bx": 4, "by": 5, "bz": 6, "phi": 7}
        return {"ux": 0, "uy": 1, "uz": 2, "p": 3}

    def _blk(self, test_name, trial_name, terms):
        out = np.zeros((self.n, self.n), dtype=complex)
        test = self.tv[test_name]
        trial = self.tr[trial_name]
        for coeff, order in terms:
            if coeff is not None and _is_zero(coeff):
                continue
            expr = trial if order == 0 else Dx(trial, 0, order)
            if coeff is not None:
                expr = coeff * expr
            res = inner(test, expr)
            if isinstance(res, list):
                for item in res:
                    out += item.diags().toarray()
            else:
                out += res.diags().toarray()
        return out

    def _put(self, mat, rb, cb, block):
        n = self.n
        mat[rb * n:(rb + 1) * n, cb * n:(cb + 1) * n] += block

    def assemble_parts(self, ky, kz):
        """Return ``(L0, Lnu, Leta, M)`` with ``L = L0 + nu*Lnu + eta*Leta``.

        ``L0`` collects advection, Coriolis/shear couplings, the magnetic
        induction couplings and the pressure / magnetic-pressure saddle-point
        rows; ``Lnu`` is the velocity Laplacian (coefficient ``nu``) and
        ``Leta`` the magnetic Laplacian (coefficient ``eta``).  Hydro callers
        may ignore ``Leta`` (all zeros).  This mirrors the Taylor-Couette
        ``assemble_parts`` split so the shared IMEXRK linear stepper can drive
        both geometries with the same diffusion-implicit / advection-explicit
        operator decomposition.
        """
        ky = float(ky)
        kz = float(kz)
        k2 = ky * ky + kz * kz
        if k2 <= 0.0:
            raise ValueError("at least one of ky or kz must be non-zero")

        b = self._blocks()
        n = self.n
        L0 = np.zeros((len(b) * n, len(b) * n), dtype=complex)
        Lnu = np.zeros_like(L0)
        Leta = np.zeros_like(L0)
        M = np.zeros_like(L0)
        U = sp.Float(self.Uoffset) + sp.Float(self.Uprime) * x
        adv = -sp.I * sp.Float(ky) * U
        lap = [(None, 2), (-sp.Float(k2), 0)]
        grad_y = sp.I * sp.Float(ky)
        grad_z = sp.I * sp.Float(kz)

        for comp in ("ux", "uy", "uz"):
            self._put(L0, b[comp], b[comp], self._blk(comp, comp, [(adv, 0)]))
            self._put(Lnu, b[comp], b[comp], self._blk(comp, comp, lap))
            self._put(M, b[comp], b[comp], self._blk(comp, comp, [(None, 0)]))

        self._put(L0, b["ux"], b["p"], -self._blk("ux", "p", [(None, 1)]))
        self._put(L0, b["uy"], b["p"], -self._blk("uy", "p", [(grad_y, 0)]))
        self._put(L0, b["uz"], b["p"], -self._blk("uz", "p", [(grad_z, 0)]))
        self._put(L0, b["ux"], b["uy"],
                  self._blk("ux", "uy", [(sp.Float(2.0 * self.omega), 0)]))
        self._put(L0, b["uy"], b["ux"],
                  self._blk("uy", "ux",
                            [(sp.Float(-self.Uprime - 2.0 * self.omega), 0)]))

        self._put(L0, b["p"], b["ux"], self._blk("p", "ux", [(None, 1)]))
        self._put(L0, b["p"], b["uy"], self._blk("p", "uy", [(grad_y, 0)]))
        self._put(L0, b["p"], b["uz"], self._blk("p", "uz", [(grad_z, 0)]))

        if self.mhd:
            kB = ky * self.by + kz * self.bz
            ikB = sp.I * sp.Float(kB)
            for u_name, b_name in (("ux", "bx"), ("uy", "by"), ("uz", "bz")):
                self._put(L0, b[u_name], b[b_name],
                          self._blk(u_name, b_name, [(ikB, 0)]))
                self._put(L0, b[b_name], b[u_name],
                          self._blk(b_name, u_name, [(ikB, 0)]))
                self._put(L0, b[b_name], b[b_name],
                          self._blk(b_name, b_name, [(adv, 0)]))
                self._put(Leta, b[b_name], b[b_name],
                          self._blk(b_name, b_name, lap))
                self._put(M, b[b_name], b[b_name],
                          self._blk(b_name, b_name, [(None, 0)]))
            self._put(L0, b["by"], b["bx"],
                      self._blk("by", "bx", [(sp.Float(self.Uprime), 0)]))
            self._put(L0, b["bx"], b["phi"],
                      -self._blk("bx", "phi", [(None, 1)]))
            self._put(L0, b["by"], b["phi"],
                      -self._blk("by", "phi", [(grad_y, 0)]))
            self._put(L0, b["bz"], b["phi"],
                      -self._blk("bz", "phi", [(grad_z, 0)]))
            self._put(L0, b["phi"], b["bx"], self._blk("phi", "bx", [(None, 1)]))
            self._put(L0, b["phi"], b["by"], self._blk("phi", "by", [(grad_y, 0)]))
            self._put(L0, b["phi"], b["bz"], self._blk("phi", "bz", [(grad_z, 0)]))

        return L0, Lnu, Leta, M

    def assemble(self, ky, kz):
        L0, Lnu, Leta, M = self.assemble_parts(ky, kz)
        return L0 + self.nu * Lnu + self.eta * Leta, M

    def energy_matrix(self, kind="total"):
        if kind not in ("total", "kinetic", "magnetic"):
            raise ValueError("kind must be 'total', 'kinetic', or 'magnetic'")
        if kind == "magnetic" and not self.mhd:
            raise ValueError("magnetic energy norm requires mhd=True")
        b = self._blocks()
        Q = np.zeros((len(b) * self.n, len(b) * self.n), dtype=complex)
        names = []
        if kind in ("total", "kinetic"):
            names += ["ux", "uy", "uz"]
        if self.mhd and kind in ("total", "magnetic"):
            names += ["bx", "by", "bz"]
        for name in names:
            W = self._blk(name, name, [(None, 0)])
            W = 0.5 * (W + W.conj().T)
            i = b[name]
            Q[i * self.n:(i + 1) * self.n, i * self.n:(i + 1) * self.n] = W
        return Q

    def eigs(self, ky, kz, n_return=8, finite_cap=FINITE_CAP):
        return finite_eigensystem(*self.assemble(ky, kz), finite_cap=finite_cap,
                                  n_return=n_return)

    def growth_rate(self, ky, kz):
        w, _ = self.eigs(ky, kz, n_return=1)
        return float(w[0].real) if len(w) else float("nan")

    def nonmodal_growth(self, ky, kz, times, n_modes=None, finite_cap=FINITE_CAP,
                        energy="total"):
        w, V = finite_eigensystem(*self.assemble(ky, kz), finite_cap=finite_cap,
                                  n_return=n_modes)
        return transient_growth_from_eigs(w, V, self.energy_matrix(energy), times)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--family", choices=("C", "L"), default="C")
    p.add_argument("--Re", type=float, default=1000.0)
    p.add_argument("--Rm", type=float, default=None)
    p.add_argument("--ky", type=float, default=0.0)
    p.add_argument("--kz", type=float, default=1.0)
    p.add_argument("--mhd", action="store_true")
    p.add_argument("--by", type=float, default=0.0)
    p.add_argument("--bz", type=float, default=0.0)
    p.add_argument("--omega", type=float, default=0.0)
    p.add_argument("--Uprime", type=float, default=1.0)
    p.add_argument("--magnetic-bc", choices=("conducting", "dirichlet"),
                   default="conducting")
    p.add_argument("--nonmodal", action="store_true")
    p.add_argument("--times", default="1")
    p.add_argument("--energy", choices=("total", "kinetic", "magnetic"),
                   default="total")
    return p.parse_args()


def main():
    args = _parse_args()
    rm = args.Re if args.Rm is None else args.Rm
    solver = PlaneCouetteGalerkinLinear(
        N=args.N,
        family=args.family,
        nu=abs(args.Uprime) / args.Re,
        eta=abs(args.Uprime) / rm,
        Uprime=args.Uprime,
        omega=args.omega,
        mhd=args.mhd,
        by=args.by,
        bz=args.bz,
        magnetic_bc=args.magnetic_bc,
    )
    if args.nonmodal:
        rows = solver.nonmodal_growth(args.ky, args.kz, parse_times(args.times),
                                      energy=args.energy)
        print_transient_growth(rows, "PCF Galerkin non-modal growth")
    else:
        w, _ = solver.eigs(args.ky, args.kz, n_return=8)
        print_eigenvalues(w, "PCF Galerkin leading eigenvalues")


if __name__ == "__main__":
    main()

