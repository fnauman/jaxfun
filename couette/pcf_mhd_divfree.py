"""
Plane Couette MHD with magnetic divergence enforced by construction.

This solver advances velocity fluctuations with the KMM Plane Couette/channel
flow formulation from ``ChannelFlow.KMM`` and advances a magnetic vector
potential ``A`` instead of the magnetic field directly:

    dA/dt = U x B + eta*lap(A)
    B = curl(A)
    J = curl(B)

Here ``U = U_wall*x*e_y + u'``.  The magnetic field is recomputed from
``curl(A)`` whenever it is used, so the discrete compatible-space identity
``div(curl(A)) = 0`` is the invariant that protects ``div(B)``.

Component order follows ``ChannelFlow.KMM``:

    0: wall-normal x
    1: streamwise y
    2: spanwise z

The default command-line resolution is meant for a real PCF run.  The
``--family L`` option is useful for small smoke tests because Chebyshev
biharmonic assembly in shenfun needs enough wall-normal modes.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from ChannelFlow import KMM
from shenfun import *  # noqa: F401,F403
from shenfun import config

# NumPy 2 compatibility for older shenfun kernels.
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item() if hasattr(a, "item") else np.asarray(a).item()

try:
    from shenfun.utilities import cleanup as shenfun_cleanup
except Exception:  # pragma: no cover - depends on installed shenfun version
    shenfun_cleanup = None


def _configure_backend(prefer_numba: bool = False):
    """Keep the installed backend unless the caller explicitly asks otherwise."""
    original_mode = config["optimization"]["mode"].lower()
    if prefer_numba:
        try:
            import shenfun.optimization.numba as _numba_backend  # noqa: F401
        except Exception:
            pass
        else:
            config["optimization"]["mode"] = "numba"
    return original_mode, config["optimization"]["mode"].lower()


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class PlaneCouetteMHDDivFree(KMM):
    """Divergence-free magnetic PCF/MHD solver based on a vector potential."""

    def __init__(
        self,
        N=(32, 64, 32),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=400.0,
        Rm=None,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_MHD_divfree",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=0.1,
        magnetic_amplitude=0.05,
        timestepper="IMEXRK222",
        prefer_numba=False,
        store_history=False,
    ):
        self.Re = float(Re)
        self.U_wall = float(U_wall)
        self.Rm = float(Rm) if Rm is not None else float(Re)
        self.eta = self.U_wall / self.Rm
        self.perturbation_amplitude = float(perturbation_amplitude)
        self.magnetic_amplitude = float(magnetic_amplitude)
        self.store_history = bool(store_history)
        self.history = []
        self.last_diagnostics = None

        original_mode, mode = _configure_backend(prefer_numba=prefer_numba)
        nu = self.U_wall / self.Re

        KMM.__init__(
            self,
            N=N,
            domain=domain,
            nu=nu,
            dt=dt,
            conv=conv,
            dpdy=0.0,
            filename=filename,
            family=family,
            padding_factor=padding_factor,
            modplot=-1,
            modsave=modsave,
            moderror=moderror,
            checkpoint=checkpoint,
            timestepper=timestepper,
        )

        self.domain_lengths = tuple(float(bounds[1] - bounds[0]) for bounds in domain)
        self.volume = float(np.prod(self.domain_lengths))

        # Couette base flow Ub(x)*e_y and constant shear.
        self.Ub = self.U_wall * self.X[0]
        self.x_1d = self.X[0][:, 0, 0]
        self.Ub_1d = self.U_wall * self.x_1d
        self.dUb_dx = self.U_wall
        self.ub_total = Array(self.BD)

        # Vector potential A is integrated in TD^3.  With A_y=A_z=0 at walls,
        # B_x = d_y A_z - d_z A_y also has zero normal flux at the walls.
        self.a_ = Function(self.CD)
        self.HA_ = Function(self.CD)
        self.ap = Array(self.CD)

        # B = curl(A) lives in the mixed curl space [TD, TC, TC].
        self.b_ = Function(self.CC)
        self.bb = Array(self.CC)
        self.projBx = Project(curl(self.a_)[0], self.TD, output_array=self.b_[0])
        self.projBy = Project(curl(self.a_)[1], self.TC, output_array=self.b_[1])
        self.projBz = Project(curl(self.a_)[2], self.TC, output_array=self.b_[2])
        self.divb = Project(div(self.b_), self.TC)

        # J = curl(B) has the complementary mixed space [TC, TD, TD].
        self.JS = VectorSpace([self.TC, self.TD, self.TD])
        self.j_ = Function(self.JS)
        self.curlb0 = Project(curl(self.b_)[0], self.TC, output_array=self.j_[0])
        self.curlb1 = Project(curl(self.b_)[1], self.TD, output_array=self.j_[1])
        self.curlb2 = Project(curl(self.b_)[2], self.TD, output_array=self.j_[2])

        # Padded buffers for nonlinear products.
        self.BDp = self.BD.get_dealiased(self.padding_factor)
        self.ub_total_pad = Array(self.BDp)
        self.Xp = self.TDp.local_mesh(bcast=True)
        self.Ub_pad = self.U_wall * self.Xp[0]

        h = TestFunction(self.TD)
        solA = (
            chebyshev.la.Helmholtz
            if self.B0.family() == "chebyshev"
            else la.SolverGeneric1ND
        )
        self.pdesA = (
            self.PDE(
                h,
                self.a_[0],
                lambda f: self.eta * div(grad(f)),
                self.HA_[0],
                dt=self.dt,
                solver=solA,
                name="Ax",
                latex=r"\partial_t A_x = \eta \nabla^2 A_x + (U\times B)_x",
            ),
            self.PDE(
                h,
                self.a_[1],
                lambda f: self.eta * div(grad(f)),
                self.HA_[1],
                dt=self.dt,
                solver=solA,
                name="Ay",
                latex=r"\partial_t A_y = \eta \nabla^2 A_y + (U\times B)_y",
            ),
            self.PDE(
                h,
                self.a_[2],
                lambda f: self.eta * div(grad(f)),
                self.HA_[2],
                dt=self.dt,
                solver=solA,
                name="Az",
                latex=r"\partial_t A_z = \eta \nabla^2 A_z + (U\times B)_z",
            ),
        )

        self.checkpoint.data["0"]["A"] = [self.a_]
        self.file_a = ShenfunFile(
            "_".join((filename, "A")), self.CD, backend="hdf5", mode="w", mesh="uniform"
        )
        self.file_b = ShenfunFile(
            "_".join((filename, "B")), self.CC, backend="hdf5", mode="w", mesh="uniform"
        )

        if comm.Get_rank() == 0:
            print("Plane Couette MHD initialized (divergence-free vector potential)")
            print(f"  Re={self.Re:g}, Rm={self.Rm:g}, nu={nu:.6g}, eta={self.eta:.6g}")
            print(
                f"  N={N}, domain={domain}, dt={dt}, stepper={timestepper}, family={family}"
            )
            if mode != original_mode:
                print(f"  optimization: {original_mode} -> {mode}")
            else:
                print(f"  optimization: {mode}")
            print("  magnetic invariant: B=curl(A), div(B)=0 in compatible spaces")
            print()

    # ------------------------------------------------------------------
    # Field construction
    # ------------------------------------------------------------------
    def update_B_from_A(self):
        """Recompute B coefficients from the current vector potential."""
        self.projBx()
        self.projBy()
        self.projBz()
        self.b_.mask_nyquist(self.mask)
        return self.b_

    def update_J_from_B(self):
        """Recompute current J = curl(B)."""
        self.update_B_from_A()
        self.curlb0()
        self.curlb1()
        self.curlb2()
        self.j_.mask_nyquist(self.mask)
        return self.j_

    def total_velocity_physical_from(self, ubp):
        ubt = self.ub_total
        ubt[...] = ubp
        ubt.v[1] += self.Ub
        return ubt

    # ------------------------------------------------------------------
    # Initialization and restart
    # ------------------------------------------------------------------
    def init_from_checkpoint(self):
        self.checkpoint.read(self.u_, "U", step=0)
        self.checkpoint.read(self.a_, "A", step=0)
        self.g_[:] = 1j * self.K[1] * self.u_[2] - 1j * self.K[2] * self.u_[1]
        self.update_B_from_A()
        self.checkpoint.open()
        tstep = self.checkpoint.f.attrs["tstep"]
        t = self.checkpoint.f.attrs["t"]
        self.checkpoint.close()
        return t, tstep

    def initialize(self, from_checkpoint: bool = False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        X = self.X
        U = Array(self.BD)
        U[...] = 0.0
        if self.perturbation_amplitude > 0:
            if comm.Get_rank() == 0:
                print(
                    f"Adding velocity perturbations amp={self.perturbation_amplitude}"
                )
            wall = 1 - X[0] ** 2
            amp = self.perturbation_amplitude
            ly = self.F1.domain[1]
            lz = self.F2.domain[1]
            U[0] += (
                amp
                * wall
                * np.sin(2 * np.pi * X[1] / ly)
                * np.cos(2 * np.pi * X[2] / lz)
            )
            U[1] += (
                amp
                * wall
                * np.cos(2 * np.pi * X[1] / ly)
                * np.sin(2 * np.pi * X[2] / lz)
            )
            U[2] += (
                amp
                * wall
                * np.sin(4 * np.pi * X[1] / ly)
                * np.cos(4 * np.pi * X[2] / lz)
            )
        U.forward(self.u_)
        self.u_.mask_nyquist(self.mask)
        self.g_[:] = 1j * self.K[1] * self.u_[2] - 1j * self.K[2] * self.u_[1]

        A = Array(self.CD)
        A[...] = 0.0
        if self.magnetic_amplitude > 0:
            if comm.Get_rank() == 0:
                print(
                    f"Adding magnetic perturbations through A amp={self.magnetic_amplitude}"
                )
            wall = 1 - X[0] ** 2
            ky = 2 * np.pi / self.F1.domain[1]
            kz = 2 * np.pi / self.F2.domain[1]
            amp = self.magnetic_amplitude
            # Ax-only seed gives B_y=d_z A_x, B_z=-d_y A_x, B_x=0.
            A[0] += amp * wall * (1.0 / kz) * np.sin(ky * X[1]) * np.sin(kz * X[2])
        A.forward(self.a_)
        self.a_.mask_nyquist(self.mask)
        self.update_B_from_A()

        if comm.Get_rank() == 0:
            diag = self.compute_diagnostics(0.0, 0)
            print("Initial fields ready")
            print(
                f"Initial divB: L2={diag['divb_l2']:.3e} Linf={diag['divb_linf']:.3e}"
            )
            print()
        return 0.0, 0

    # ------------------------------------------------------------------
    # Nonlinear terms
    # ------------------------------------------------------------------
    def convection(self):
        if self.conv != 0:
            raise NotImplementedError("Only conv=0 is implemented for PCF MHD")

        self.update_B_from_A()

        H = self.H_.v
        up = self.u_.backward(padding_factor=self.padding_factor)
        upv = up.v
        bpv = self.b_.backward(padding_factor=self.padding_factor).v

        dudxp = self.dudx().backward(padding_factor=self.padding_factor).v
        dudyp = self.dudy().backward(padding_factor=self.padding_factor).v
        dudzp = self.dudz().backward(padding_factor=self.padding_factor).v
        dvdxp = self.dvdx().backward(padding_factor=self.padding_factor).v
        dvdyp = self.dvdy().backward(padding_factor=self.padding_factor).v
        dvdzp = self.dvdz().backward(padding_factor=self.padding_factor).v
        dwdxp = self.dwdx().backward(padding_factor=self.padding_factor).v
        dwdyp = self.dwdy().backward(padding_factor=self.padding_factor).v
        dwdzp = self.dwdz().backward(padding_factor=self.padding_factor).v

        n0 = upv[0] * dudxp + upv[1] * dudyp + upv[2] * dudzp
        n1 = upv[0] * dvdxp + upv[1] * dvdyp + upv[2] * dvdzp
        n2 = upv[0] * dwdxp + upv[1] * dwdyp + upv[2] * dwdzp

        Ub = self.Ub_1d[:, None, None]
        n0 += Ub * dudyp
        n1 += Ub * dvdyp + upv[0] * self.dUb_dx
        n2 += Ub * dwdyp

        # Lorentz force J x B enters the KMM nonlinear storage as N - J x B.
        self.update_J_from_B()
        jbp = self.j_.backward(padding_factor=self.padding_factor).v
        l0 = jbp[1] * bpv[2] - jbp[2] * bpv[1]
        l1 = jbp[2] * bpv[0] - jbp[0] * bpv[2]
        l2 = jbp[0] * bpv[1] - jbp[1] * bpv[0]
        n0 -= l0
        n1 -= l1
        n2 -= l2

        H[0] = self.TDp.forward(n0, H[0])
        H[1] = self.TDp.forward(n1, H[1])
        H[2] = self.TDp.forward(n2, H[2])
        self.H_.mask_nyquist(self.mask)

        # Vector-potential forcing U x B, using total Couette velocity.
        ubt = self.ub_total_pad
        ubt[...] = up
        ubt.v[1] += self.Ub_pad

        HA = self.HA_.v
        HA[0] = self.TDp.forward(ubt.v[1] * bpv[2] - ubt.v[2] * bpv[1], HA[0])
        HA[1] = self.TDp.forward(ubt.v[2] * bpv[0] - ubt.v[0] * bpv[2], HA[1])
        HA[2] = self.TDp.forward(ubt.v[0] * bpv[1] - ubt.v[1] * bpv[0], HA[2])
        self.HA_.mask_nyquist(self.mask)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _field_l2(field):
        return float(np.sqrt(inner(1, field * field)))

    @staticmethod
    def _field_linf(field):
        return float(np.max(np.abs(field)))

    def compute_diagnostics(self, t, tstep):
        self.update_B_from_A()
        ubp = self.u_.backward(self.ub)
        bp = self.b_.backward(self.bb)
        ubt = self.total_velocity_physical_from(ubp)

        Epert = float(
            inner(1, ubp[0] * ubp[0])
            + inner(1, ubp[1] * ubp[1])
            + inner(1, ubp[2] * ubp[2])
        )
        Etot = float(
            inner(1, ubt[0] * ubt[0])
            + inner(1, ubt[1] * ubt[1])
            + inner(1, ubt[2] * ubt[2])
        )
        Emag = float(
            inner(1, bp[0] * bp[0]) + inner(1, bp[1] * bp[1]) + inner(1, bp[2] * bp[2])
        )

        divu = self.divu().backward()
        divb = self.divb().backward()
        divu_l2 = self._field_l2(divu)
        divb_l2 = self._field_l2(divb)
        divu_linf = self._field_linf(divu)
        divb_linf = self._field_linf(divb)

        uprms = math.sqrt(Epert / self.volume) if Epert > 0 else 0.0
        brms = math.sqrt(Emag / self.volume) if Emag > 0 else 0.0
        divu_rel = divu_l2 / math.sqrt(self.volume) / max(uprms, 1e-16)
        divb_rel = divb_l2 / math.sqrt(self.volume) / max(brms, 1e-16)

        v_tot = ubt[1]
        top_i = int(np.argmax(self.x_1d))
        bot_i = int(np.argmin(self.x_1d))
        top = float(np.mean(v_tot[top_i, :, :]))
        bot = float(np.mean(v_tot[bot_i, :, :]))
        dv_dx_fluct = self.dvdx().backward()
        mean_shear = float(np.mean(dv_dx_fluct + self.dUb_dx))
        bmax = float(np.max(np.abs(bp)))

        diag = {
            "t": float(t),
            "tstep": int(tstep),
            "Epert": Epert,
            "Etot": Etot,
            "Emag": Emag,
            "divu_l2": divu_l2,
            "divu_linf": divu_linf,
            "divu_rel": divu_rel,
            "divb_l2": divb_l2,
            "divb_linf": divb_linf,
            "divb_rel": divb_rel,
            "top_wall_streamwise": top,
            "bottom_wall_streamwise": bot,
            "mean_shear": mean_shear,
            "bmax": bmax,
        }
        self.last_diagnostics = diag
        if self.store_history:
            self.history.append(diag.copy())
        return diag

    def print_diagnostics(self, t, tstep):
        if self.moderror <= 0 or tstep % self.moderror != 0:
            return
        diag = self.compute_diagnostics(t, tstep)
        if comm.Get_rank() == 0:
            print(
                f"t={diag['t']:7.3f} E'={diag['Epert']:10.3e} "
                f"Etot={diag['Etot']:10.3e} Emag={diag['Emag']:10.3e} "
                f"divuL2={diag['divu_l2']:8.2e} divuLinf={diag['divu_linf']:8.2e} "
                f"relU={diag['divu_rel']:6.2e} "
                f"divbL2={diag['divb_l2']:8.2e} divbLinf={diag['divb_linf']:8.2e} "
                f"relB={diag['divb_rel']:6.2e} "
                f"u(1)~{diag['top_wall_streamwise']:+.4f} "
                f"u(-1)~{diag['bottom_wall_streamwise']:+.4f} "
                f"d<u>/dx~{diag['mean_shear']:.4f} |B|max={diag['bmax']:.2e}"
            )

    def update(self, t, tstep):
        self.print_diagnostics(t, tstep)

    def assert_diagnostics(self, diag, max_divb_l2=None, max_divu_l2=None):
        values = [
            diag["Epert"],
            diag["Etot"],
            diag["Emag"],
            diag["divu_l2"],
            diag["divb_l2"],
            diag["bmax"],
        ]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite diagnostic values: {diag}")
        if max_divb_l2 is not None and diag["divb_l2"] > max_divb_l2:
            raise RuntimeError(
                f"divB L2 {diag['divb_l2']:.6e} exceeds {max_divb_l2:.6e}"
            )
        if max_divu_l2 is not None and diag["divu_l2"] > max_divu_l2:
            raise RuntimeError(
                f"divU L2 {diag['divu_l2']:.6e} exceeds {max_divu_l2:.6e}"
            )

    # ------------------------------------------------------------------
    # Time integration and output
    # ------------------------------------------------------------------
    def assemble(self):
        super().assemble()
        for eq in self.pdesA:
            eq.assemble()

    def tofile(self, tstep):
        self.file_u.write(
            tstep, {"u": [self.u_.backward(mesh="uniform")]}, as_scalar=True
        )
        self.file_a.write(
            tstep, {"a": [self.a_.backward(mesh="uniform")]}, as_scalar=True
        )
        self.update_B_from_A()
        self.file_b.write(
            tstep, {"b": [self.b_.backward(mesh="uniform")]}, as_scalar=True
        )

    def solve(
        self,
        t=0.0,
        tstep=0,
        end_time=10.0,
        max_divb_l2=None,
        max_divu_l2=None,
        assert_every_step=False,
    ):
        self.assemble()
        while t < end_time - 1e-12:
            for rk in range(self.PDE.steps()):
                self.prepare_step(rk)

                for eq in self.pdes.values():
                    eq.compute_rhs(rk)
                for eq in self.pdesA:
                    eq.compute_rhs(rk)

                for eq in self.pdes.values():
                    eq.solve_step(rk)
                self.compute_vw(rk)

                for eq in self.pdesA:
                    eq.solve_step(rk)

            t += self.dt
            tstep += 1
            self.update(t, tstep)
            if assert_every_step:
                self.assert_diagnostics(
                    self.compute_diagnostics(t, tstep), max_divb_l2, max_divu_l2
                )
            self.checkpoint.update(t, tstep)
            if self.modsave > 0 and tstep % self.modsave == 0:
                self.tofile(tstep)

        final = self.compute_diagnostics(t, tstep)
        self.assert_diagnostics(final, max_divb_l2, max_divu_l2)
        return final


def _parse_args(defaults):
    p = argparse.ArgumentParser(description="Divergence-free Plane Couette MHD solver")
    p.add_argument("--nx", type=int, default=defaults["N"][0])
    p.add_argument("--ny", type=int, default=defaults["N"][1])
    p.add_argument("--nz", type=int, default=defaults["N"][2])
    p.add_argument("--Re", type=float, default=defaults["Re"])
    p.add_argument("--Rm", type=float, default=defaults["Rm"])
    p.add_argument("--U-wall", type=float, default=defaults["U_wall"])
    p.add_argument("--dt", type=float, default=defaults["dt"])
    p.add_argument("--end-time", type=float, default=defaults["end_time"])
    p.add_argument("--moderror", type=int, default=defaults["moderror"])
    p.add_argument("--modsave", type=int, default=defaults["modsave"])
    p.add_argument("--checkpoint", type=int, default=defaults["checkpoint"])
    p.add_argument("--family", choices=["C", "L"], default=defaults["family"])
    p.add_argument("--timestepper", type=str, default=defaults["timestepper"])
    p.add_argument(
        "--perturbation-amplitude",
        type=float,
        default=defaults["perturbation_amplitude"],
    )
    p.add_argument(
        "--magnetic-amplitude", type=float, default=defaults["magnetic_amplitude"]
    )
    p.add_argument("--filename", type=str, default=defaults["filename"])
    p.add_argument(
        "--prefer-numba", action="store_true", default=defaults["prefer_numba"]
    )
    p.add_argument(
        "--store-history", action="store_true", default=defaults["store_history"]
    )
    p.add_argument("--max-divb-l2", type=float, default=None)
    p.add_argument("--max-divu-l2", type=float, default=None)
    p.add_argument("--assert-every-step", action="store_true")
    p.add_argument(
        "--linear",
        choices=["dns", "eigs", "nonmodal"],
        default="dns",
        help="run a linear eigenvalue or non-modal analysis instead of DNS",
    )
    p.add_argument(
        "--linear-nx",
        type=int,
        default=defaults["N"][0],
        help="wall-normal collocation points for linear analysis",
    )
    p.add_argument(
        "--ky", type=float, default=1.0, help="streamwise linear-analysis wavenumber"
    )
    p.add_argument(
        "--kz", type=float, default=1.0, help="spanwise linear-analysis wavenumber"
    )
    p.add_argument(
        "--linear-times",
        type=str,
        default="1,5,10,20",
        help="comma-separated times for non-modal transient growth",
    )
    p.add_argument(
        "--linear-n-return",
        type=int,
        default=8,
        help="number of leading eigenvalues to print",
    )
    p.add_argument(
        "--linear-n-modes",
        type=int,
        default=None,
        help="number of finite modes retained for non-modal analysis",
    )
    p.add_argument(
        "--linear-finite-cap",
        type=float,
        default=1.0e8,
        help="discard generalized eigenvalues above this magnitude",
    )
    p.add_argument(
        "--linear-by",
        type=float,
        default=0.0,
        help="uniform streamwise imposed field for linear analysis",
    )
    p.add_argument(
        "--linear-bz",
        type=float,
        default=0.0,
        help="uniform spanwise imposed field for linear analysis",
    )
    p.add_argument(
        "--linear-magnetic-bc",
        choices=["conducting", "dirichlet"],
        default="conducting",
    )
    p.add_argument(
        "--linear-energy",
        choices=["total", "kinetic", "magnetic"],
        default="total",
        help="energy norm for --linear nonmodal (kinetic+magnetic by "
        "default; 'kinetic' reduces to the hydro result at B0=0)",
    )
    return p.parse_args()


def _run_linear_variant(args, params):
    from _linear_analysis import parse_times, print_eigenvalues, print_transient_growth
    from _pcf_linear import PlaneCouetteLinear

    lin = PlaneCouetteLinear.couette(
        nx=args.linear_nx,
        Re=params["Re"],
        Rm=params["Rm"],
        U_wall=params["U_wall"],
        mhd=True,
        by=args.linear_by,
        bz=args.linear_bz,
        magnetic_bc=args.linear_magnetic_bc,
    )
    if args.linear == "eigs":
        w, _ = lin.eigs(
            args.ky,
            args.kz,
            n_return=args.linear_n_return,
            finite_cap=args.linear_finite_cap,
        )
        if comm.Get_rank() == 0:
            print(
                f"Plane Couette MHD linear eigenvalues: Re={params['Re']:g}, Rm={params['Rm']:g}, "
                f"ky={args.ky:g}, kz={args.kz:g}, B0=(0,{args.linear_by:g},{args.linear_bz:g})"
            )
            print_eigenvalues(w)
        return {"eigenvalues": [(float(s.real), float(s.imag)) for s in w]}

    rows = lin.nonmodal_growth(
        args.ky,
        args.kz,
        parse_times(args.linear_times),
        n_modes=args.linear_n_modes,
        finite_cap=args.linear_finite_cap,
        energy=args.linear_energy,
    )
    if comm.Get_rank() == 0:
        print(
            f"Plane Couette MHD non-modal growth ({args.linear_energy} energy): "
            f"Re={params['Re']:g}, Rm={params['Rm']:g}, "
            f"ky={args.ky:g}, kz={args.kz:g}, B0=(0,{args.linear_by:g},{args.linear_bz:g})"
        )
        print_transient_growth(rows)
    return {"transient_growth": rows}


def run(argv=None):
    defaults = dict(
        N=(32, 64, 32),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=400.0,
        Rm=400.0,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_MHD_divfree",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=0.1,
        magnetic_amplitude=0.05,
        timestepper="IMEXRK222",
        prefer_numba=False,
        store_history=False,
        end_time=10.0,
    )
    if argv is not None:
        old_argv = sys.argv[:]
        sys.argv = [old_argv[0], *argv]
    try:
        args = _parse_args(defaults)
    finally:
        if argv is not None:
            sys.argv = old_argv

    params = defaults.copy()
    params["N"] = (
        _positive_int(args.nx, "nx"),
        _positive_int(args.ny, "ny"),
        _positive_int(args.nz, "nz"),
    )
    params["Re"] = args.Re
    params["Rm"] = args.Rm
    params["U_wall"] = args.U_wall
    params["dt"] = args.dt
    params["moderror"] = args.moderror
    params["modsave"] = args.modsave
    params["checkpoint"] = args.checkpoint
    params["family"] = args.family
    params["timestepper"] = args.timestepper
    params["perturbation_amplitude"] = args.perturbation_amplitude
    params["magnetic_amplitude"] = args.magnetic_amplitude
    params["filename"] = args.filename
    params["prefer_numba"] = args.prefer_numba
    params["store_history"] = args.store_history
    end_time = float(args.end_time)
    params.pop("end_time", None)

    if args.linear != "dns":
        return _run_linear_variant(args, params)

    if comm.Get_rank() == 0:
        print("Simulation parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print(f"  end_time: {end_time}")
        if args.max_divb_l2 is not None:
            print(f"  max_divb_l2: {args.max_divb_l2}")
        if args.max_divu_l2 is not None:
            print(f"  max_divu_l2: {args.max_divu_l2}")
        print()

    solver = PlaneCouetteMHDDivFree(**params)
    t, tstep = solver.initialize(from_checkpoint=False)
    try:
        final = solver.solve(
            t=t,
            tstep=tstep,
            end_time=end_time,
            max_divb_l2=args.max_divb_l2,
            max_divu_l2=args.max_divu_l2,
            assert_every_step=args.assert_every_step,
        )
    finally:
        if shenfun_cleanup is not None:
            shenfun_cleanup(vars(solver))

    if comm.Get_rank() == 0:
        print("Final diagnostics:")
        for key in (
            "t",
            "tstep",
            "Epert",
            "Etot",
            "Emag",
            "divu_l2",
            "divb_l2",
            "divu_rel",
            "divb_rel",
            "bmax",
        ):
            print(
                f"  {key}: {final[key]:.16e}"
                if isinstance(final[key], float)
                else f"  {key}: {final[key]}"
            )
    return final


if __name__ == "__main__":
    if comm.Get_size() > 1:
        print(f"Running on {comm.Get_size()} MPI processes")
    else:
        print("Running in serial mode")
    run()
