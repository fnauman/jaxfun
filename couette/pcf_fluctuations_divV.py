"""
Plane Couette flow (fluctuation form) with incompressibility diagnostics.

Focus: verify div(u)=0 to machine precision and monitor key PCF checks
at a modest resolution for quick runs.
"""

from __future__ import annotations

import argparse

import numpy as np
from ChannelFlow import KMM
from shenfun import *  # noqa
from shenfun import config
from shenfun.optimization import numba as shenfun_numba

# NumPy 2.0 compatibility for older shenfun numba kernels.
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item() if hasattr(a, "item") else np.asarray(a).item()

try:
    from shenfun.utilities import cleanup as shenfun_cleanup
except Exception:
    shenfun_cleanup = None

PREFER_NUMBA = True


def _select_backend_and_family(family):
    original_mode = config["optimization"]["mode"].lower()
    mode = original_mode
    if PREFER_NUMBA:
        if shenfun_numba is not None:
            config["optimization"]["mode"] = "numba"
            mode = "numba"
        else:
            config["optimization"]["mode"] = "python"
            mode = "python"
    return family, original_mode, mode


class PlaneCouetteFluctuationDiagnostics(KMM):
    def __init__(
        self,
        N=(32, 64, 32),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=400.0,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_fluct_divV",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=0.1,
        timestepper="IMEXRK222",
    ):
        self.Re = float(Re)
        self.U_wall = float(U_wall)
        self.perturbation_amplitude = float(perturbation_amplitude)

        family, original_mode, mode = _select_backend_and_family(family)
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

        # Base flow in streamwise direction and its constant shear.
        self.Ub = self.U_wall * self.X[0]
        self.x_1d = self.X[0][:, 0, 0]
        self.Ub_1d = self.U_wall * self.x_1d
        self.dUb_dx = self.U_wall
        self.ub_total = Array(self.BD)

        if comm.Get_rank() == 0:
            print("Plane Couette (fluctuation form) diagnostics initialized")
            print(f"  Re={self.Re:g}, U_wall={self.U_wall:g}, nu={nu:.6g}")
            print(f"  N={N}, domain={domain}, dt={dt}, stepper={timestepper}")
            if mode != original_mode:
                print(f"  optimization: {original_mode} -> {mode}")
            print()

    def initialize(self, from_checkpoint: bool = False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        X = self.X
        U = Array(self.BD)
        U[...] = 0.0
        if self.perturbation_amplitude > 0:
            if comm.Get_rank() == 0:
                print(f"Adding perturbations amp={self.perturbation_amplitude}")
            np.random.seed(42)
            wall = 1 - X[0] ** 2
            amp = self.perturbation_amplitude
            U[0] += (
                amp
                * wall
                * np.sin(2 * np.pi * X[1] / self.F1.domain[1])
                * np.cos(2 * np.pi * X[2] / self.F2.domain[1])
            )
            U[1] += (
                amp
                * wall
                * np.cos(2 * np.pi * X[1] / self.F1.domain[1])
                * np.sin(2 * np.pi * X[2] / self.F2.domain[1])
            )
            U[2] += (
                amp
                * wall
                * np.sin(4 * np.pi * X[1] / self.F1.domain[1])
                * np.cos(4 * np.pi * X[2] / self.F2.domain[1])
            )
        U.forward(self.u_)
        self.u_.mask_nyquist(self.mask)
        self.g_[:] = 1j * self.K[1] * self.u_[2] - 1j * self.K[2] * self.u_[1]
        if comm.Get_rank() == 0:
            print("Fluctuation field initialized (no base inserted)")
        return 0.0, 0

    def convection(self):
        if self.conv != 0:
            raise NotImplementedError("Only conv=0 implemented for fluctuations")
        H = self.H_.v
        up = self.u_.backward(padding_factor=self.padding_factor).v
        dudxp = self.dudx().backward(padding_factor=self.padding_factor).v
        dudyp = self.dudy().backward(padding_factor=self.padding_factor).v
        dudzp = self.dudz().backward(padding_factor=self.padding_factor).v
        dvdxp = self.dvdx().backward(padding_factor=self.padding_factor).v
        dvdyp = self.dvdy().backward(padding_factor=self.padding_factor).v
        dvdzp = self.dvdz().backward(padding_factor=self.padding_factor).v
        dwdxp = self.dwdx().backward(padding_factor=self.padding_factor).v
        dwdyp = self.dwdy().backward(padding_factor=self.padding_factor).v
        dwdzp = self.dwdz().backward(padding_factor=self.padding_factor).v
        n0 = up[0] * dudxp + up[1] * dudyp + up[2] * dudzp
        n1 = up[0] * dvdxp + up[1] * dvdyp + up[2] * dvdzp
        n2 = up[0] * dwdxp + up[1] * dwdyp + up[2] * dwdzp
        Ub = self.Ub_1d[:, None, None]
        n0 += Ub * dudyp
        n1 += Ub * dvdyp + up[0] * self.dUb_dx
        n2 += Ub * dwdyp
        H[0] = self.TDp.forward(n0, H[0])
        H[1] = self.TDp.forward(n1, H[1])
        H[2] = self.TDp.forward(n2, H[2])
        self.H_.mask_nyquist(self.mask)

    def total_velocity_physical_from(self, ubp):
        ubt = self.ub_total
        ubt[...] = ubp
        ubt.v[1] += self.Ub
        return ubt

    def _divergence_stats(self, divp, uprms):
        div_l2 = float(np.sqrt(inner(1, divp * divp)))
        div_linf = float(np.max(np.abs(divp)))
        div_rms = div_l2 / np.sqrt(self.volume)
        rel = div_rms / max(uprms, 1e-16)
        return div_l2, div_linf, div_rms, rel

    def print_diagnostics(self, t, tstep):
        if self.moderror <= 0 or tstep % self.moderror != 0:
            return
        ubp = self.u_.backward(self.ub)
        Epert = (
            inner(1, ubp[0] * ubp[0])
            + inner(1, ubp[1] * ubp[1])
            + inner(1, ubp[2] * ubp[2])
        )
        ubt = self.total_velocity_physical_from(ubp)
        Etot = (
            inner(1, ubt[0] * ubt[0])
            + inner(1, ubt[1] * ubt[1])
            + inner(1, ubt[2] * ubt[2])
        )
        uprms = float(np.sqrt(Epert / self.volume))
        divu = self.divu().backward()
        div_l2, div_linf, div_rms, div_rel = self._divergence_stats(divu, uprms)
        v_tot = ubt[1]
        top = float(np.mean(v_tot[-1, :, :]))
        bot = float(np.mean(v_tot[0, :, :]))
        dv_dx_fluct = self.dvdx().backward()
        mean_shear = float(np.mean(dv_dx_fluct + self.dUb_dx))
        wall_fluct = max(
            float(np.max(np.abs(ubp[:, 0, :, :]))),
            float(np.max(np.abs(ubp[:, -1, :, :]))),
        )
        if comm.Get_rank() == 0:
            print(
                f"t={t:7.3f} E'={Epert:10.3e} Etot={Etot:10.3e} "
                f"divL2={div_l2:8.2e} divLinf={div_linf:8.2e} "
                f"divRMS={div_rms:8.2e} rel={div_rel:6.2e} "
                f"u(1)~{top:+.4f} u(-1)~{bot:+.4f} "
                f"d<u>/dx~{mean_shear:.4f} wall|u'|={wall_fluct:.2e}"
            )

    def update(self, t, tstep):
        self.print_diagnostics(t, tstep)


def _parse_cli(params):
    parser = argparse.ArgumentParser(
        description="Plane Couette fluctuations div(u) diagnostics"
    )
    parser.add_argument("--nx", type=int, default=params["N"][0])
    parser.add_argument("--ny", type=int, default=params["N"][1])
    parser.add_argument("--nz", type=int, default=params["N"][2])
    parser.add_argument("--Re", type=float, default=params["Re"])
    parser.add_argument("--dt", type=float, default=params["dt"])
    parser.add_argument("--end-time", type=float, default=params["end_time"])
    parser.add_argument("--moderror", type=int, default=params["moderror"])
    parser.add_argument(
        "--perturbation-amplitude", type=float, default=params["perturbation_amplitude"]
    )
    parser.add_argument("--timestepper", type=str, default=params["timestepper"])
    return parser.parse_args()


def run_pcf_fluctuation_divV():
    params = dict(
        N=(32, 64, 32),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=400.0,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_fluct_divV",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=0.1,
        timestepper="IMEXRK222",
        end_time=10.0,
    )
    args = _parse_cli(params)
    params["N"] = (args.nx, args.ny, args.nz)
    params["Re"] = args.Re
    params["dt"] = args.dt
    params["moderror"] = args.moderror
    params["perturbation_amplitude"] = args.perturbation_amplitude
    params["timestepper"] = args.timestepper
    end_time = args.end_time
    params.pop("end_time", None)

    if comm.Get_rank() == 0:
        print("Simulation parameters:")
        for k, v in params.items():
            if k != "end_time":
                print(f"  {k}: {v}")
        print(f"  end_time: {end_time}")
        print()

    solver = PlaneCouetteFluctuationDiagnostics(**params)
    t, tstep = solver.initialize(from_checkpoint=False)
    try:
        solver.solve(t=t, tstep=tstep, end_time=end_time)
    finally:
        if shenfun_cleanup is not None:
            shenfun_cleanup(vars(solver))


if __name__ == "__main__":
    if comm.Get_size() > 1:
        print(f"Running on {comm.Get_size()} MPI processes")
    else:
        print("Running in serial mode")
    run_pcf_fluctuation_divV()
