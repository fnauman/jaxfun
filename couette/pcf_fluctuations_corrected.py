"""
Plane Couette Flow using fluctuations about the laminar base (Approach A).

Corrections in this version:
- Align base flow and shear production with KMM component ordering:
  component 0 = wall-normal (x), 1 = streamwise (y), 2 = spanwise (z).
  The base profile and production term now act on component 1.
- Diagnostics and plots now use the streamwise component for wall speeds
  and mean shear.

Notes:
- Base flow U_b(x) = U_wall*x is accounted for only in the convection term.
- Couette Re uses half-height h=1 for domain x in [-1, 1].
- Optional analysis outputs plane-averaged profiles, shear, RMS statistics, spectra,
  and SSP-style streak/roll diagnostics.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
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
    # Older shenfun releases may not expose cleanup in utilities.
    shenfun_cleanup = None

# Prefer Numba if available; otherwise use Python kernels. The requested
# basis family is never changed implicitly.
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


class PlaneCouetteFluctuation(KMM):
    def __init__(
        self,
        N=(64, 128, 64),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=600.0,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        modplot=50,
        modsave=500,
        moderror=10,
        modanalysis=50,
        modspectra=50,
        modssp=100,
        filename="PCF_fluct",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=250,
        perturbation_amplitude=0.05,
        timestepper="IMEXRK3",
        enable_live_plots=False,
        save_plots=True,
        save_analysis=True,
        save_spectra=True,
        save_ssp=False,
        ssp_stride=3,
        plot_format="png",
        plot_dpi=150,
    ):
        self.Re = float(Re)
        self.U_wall = float(U_wall)
        self.perturbation_amplitude = float(perturbation_amplitude)
        self.enable_live_plots = bool(enable_live_plots)
        self.save_plots = bool(save_plots)
        self.save_analysis = bool(save_analysis)
        self.save_spectra = bool(save_spectra)
        self.save_ssp = bool(save_ssp)
        self.modanalysis = int(modanalysis)
        self.modspectra = int(modspectra)
        self.modssp = int(modssp)
        self.ssp_stride = max(1, int(ssp_stride))
        self.plot_format = str(plot_format).lower()
        self.plot_dpi = int(plot_dpi)
        if self.plot_format not in ["png", "jpg", "jpeg", "pdf", "svg"]:
            raise ValueError("Unsupported plot format")

        # Select backend and basis family compatible with this environment.
        original_family = family
        family, original_mode, mode = _select_backend_and_family(family)

        # Couette viscosity: Re = U_wall * h / nu, with h = 1 for x in [-1, 1]
        nu = self.U_wall / self.Re

        # Initialize KMM with zero pressure gradient (PCF has no imposed dp/dy)
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
            modplot=modplot,
            modsave=modsave,
            moderror=moderror,
            checkpoint=checkpoint,
            timestepper=timestepper,
        )

        # KMM builds tensor-product spaces for velocity components:
        # - TB: wall-normal component (x) with wall BCs enforced
        # - TD: streamwise/spanwise components (y, z) with Dirichlet BCs
        # - BD: vector space [TB, TD, TD] in order (u_x, u_y, u_z)

        if comm.Get_rank() == 0:
            import os

            if self.save_plots:
                self.plot_dir = f"{filename}_plots"
                os.makedirs(self.plot_dir, exist_ok=True)
            if self.save_analysis or self.save_spectra or self.save_ssp:
                self.analysis_dir = f"{filename}_analysis"
                os.makedirs(self.analysis_dir, exist_ok=True)

        # Base flow in streamwise direction and its constant shear
        # U_b(x) = U_wall * x in the y-component
        self.Ub = self.U_wall * self.X[0]  # shape (Nx, Ny, Nz)
        self.x_1d = self.X[0][:, 0, 0]  # shape (Nx_local,)
        self.Ub_1d = self.U_wall * self.x_1d  # shape (Nx_local,)
        self.dUb_dx = self.U_wall  # constant dU_b/dx
        self.ub_total = Array(self.BD)  # total velocity work array

        if comm.Get_rank() == 0:
            print("Plane Couette (fluctuation form) initialized")
            print(f"  Re={self.Re:g}, U_wall={self.U_wall:g}, nu={nu:.6g}")
            print(f"  N={N}, domain={domain}, dt={dt}, stepper={timestepper}")
            if mode != original_mode:
                print(f"  optimization: {original_mode} -> {mode}")
            if family != original_family:
                print(f"  basis family: {original_family} -> {family}")
            print()

    # ---------------- init ----------------
    def initialize(self, from_checkpoint: bool = False):
        if from_checkpoint:
            return self.init_from_checkpoint()
        X = self.X
        U = Array(self.BD)
        U[...] = 0.0  # fluctuations only; base flow is added separately
        if self.perturbation_amplitude > 0:
            if comm.Get_rank() == 0:
                print(f"Adding perturbations amp={self.perturbation_amplitude}")
            # Deterministic perturbations with wall damping (1 - x^2)
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
        # Move to spectral space and initialize g = curl_x (used by KMM)
        U.forward(self.u_)
        self.u_.mask_nyquist(self.mask)
        self.g_[:] = 1j * self.K[1] * self.u_[2] - 1j * self.K[2] * self.u_[1]
        if comm.Get_rank() == 0:
            print("Fluctuation field initialized (no base inserted)")
        return 0.0, 0

    # --------------- convection ---------------
    def convection(self):
        if self.conv != 0:
            raise NotImplementedError("Only conv=0 implemented for fluctuations")
        H = self.H_.v
        up = self.u_.backward(padding_factor=self.padding_factor).v
        # Fluctuation gradients in physical space
        dudxp = self.dudx().backward(padding_factor=self.padding_factor).v
        dudyp = self.dudy().backward(padding_factor=self.padding_factor).v
        dudzp = self.dudz().backward(padding_factor=self.padding_factor).v
        dvdxp = self.dvdx().backward(padding_factor=self.padding_factor).v
        dvdyp = self.dvdy().backward(padding_factor=self.padding_factor).v
        dvdzp = self.dvdz().backward(padding_factor=self.padding_factor).v
        dwdxp = self.dwdx().backward(padding_factor=self.padding_factor).v
        dwdyp = self.dwdy().backward(padding_factor=self.padding_factor).v
        dwdzp = self.dwdz().backward(padding_factor=self.padding_factor).v
        # Nonlinear fluctuation term u' dot grad(u')
        n0 = up[0] * dudxp + up[1] * dudyp + up[2] * dudzp
        n1 = up[0] * dvdxp + up[1] * dvdyp + up[2] * dvdzp
        n2 = up[0] * dwdxp + up[1] * dwdyp + up[2] * dwdzp
        # Base-flow advection U_b * d/dy and shear production u'_x * dU_b/dx
        Ub = self.Ub_1d[:, None, None]
        n0 += Ub * dudyp
        n1 += Ub * dvdyp + up[0] * self.dUb_dx
        n2 += Ub * dwdyp
        # Transform back to spectral space for the RHS
        H[0] = self.TDp.forward(n0, H[0])
        H[1] = self.TDp.forward(n1, H[1])
        H[2] = self.TDp.forward(n2, H[2])
        self.H_.mask_nyquist(self.mask)

    # --------------- helpers ---------------
    def total_velocity_physical(self):
        return self.total_velocity_physical_from(self.u_.backward(self.ub))

    def total_velocity_physical_from(self, ubp):
        # Copy fluctuations into a separate buffer, then add base profile.
        ubt = self.ub_total
        ubt[...] = ubp
        ubt.v[1] += self.Ub
        return ubt

    # --------------- diagnostics ---------------
    def print_energy_and_divergence(self, t, tstep):
        if tstep % self.moderror != 0 or self.moderror <= 0:
            return
        # Perturbation and total kinetic energies
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
        # Divergence of fluctuations (base is divergence-free)
        divu = self.divu().backward()
        divL2 = np.sqrt(inner(1, divu * divu))
        # Wall speeds and mean shear based on streamwise component
        v_tot = ubt[1]
        top = float(np.mean(v_tot[-1, :, :]))
        bot = float(np.mean(v_tot[0, :, :]))
        dv_dx_fluct = self.dvdx().backward()  # streamwise fluctuation shear
        mean_shear = float(np.mean(dv_dx_fluct + self.dUb_dx))
        if comm.Get_rank() == 0:
            print(
                f"t={t:7.3f} Epert={Epert:10.6e} Etot={Etot:10.6e} div={divL2:8.2e} "
                f"u(1)~{top:+.4f} u(-1)~{bot:+.4f} d<u>/dx~{mean_shear:.4f}"
            )

    # --------------- plotting ---------------
    def init_plots(self):
        if comm.Get_rank() != 0:
            return
        if not self.enable_live_plots:
            plt.switch_backend("Agg")
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(12, 8)
        )
        self.time_hist, self.Epert_hist, self.Etot_hist = [], [], []
        self._setup_plot_axes()
        if self.enable_live_plots:
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)

    def _setup_plot_axes(self):
        if comm.Get_rank() != 0:
            return
        self.ax1.set_xlabel("Streamwise (y)")
        self.ax1.set_ylabel("Wall-normal (x)")
        self.ax1.set_title("v_total")
        self.ax2.set_xlabel("Streamwise (y)")
        self.ax2.set_ylabel("Wall-normal (x)")
        self.ax2.set_title("v'")
        self.ax3.set_xlabel("Spanwise (z)")
        self.ax3.set_ylabel("Wall-normal (x)")
        self.ax3.set_title("w'")
        self.ax4.set_xlabel("Time")
        self.ax4.set_ylabel("Energy")
        self.ax4.set_title("Energy Evolution")
        self.ax4.grid(True)

    def plot(self, t, tstep):
        if (not self.save_plots and not self.enable_live_plots) or self.modplot <= 0:
            return
        if not hasattr(self, "fig"):
            self.init_plots()
        if tstep % self.modplot != 0 or comm.Get_rank() != 0:
            return
        ubp = self.u_.backward(self.ub)
        ubt = self.total_velocity_physical_from(ubp)
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self._setup_plot_axes()
        # Streamwise velocity in an x-y plane
        self.ax1.contourf(
            self.X[1][:, :, 0], self.X[0][:, :, 0], ubt[1, :, :, 0], 50, cmap="RdBu_r"
        )
        # Streamwise fluctuation in the same plane
        self.ax2.contourf(
            self.X[1][:, :, 0], self.X[0][:, :, 0], ubp[1, :, :, 0], 50, cmap="RdBu_r"
        )
        # Spanwise fluctuation in an x-z plane at mid y
        mid_y = ubp.shape[2] // 2
        self.ax3.contourf(
            self.X[2][:, 0, :],
            self.X[0][:, 0, :],
            ubp[2, :, mid_y, :],
            50,
            cmap="RdBu_r",
        )
        # Energy history
        Epert = (
            inner(1, ubp[0] * ubp[0])
            + inner(1, ubp[1] * ubp[1])
            + inner(1, ubp[2] * ubp[2])
        )
        Etot = (
            inner(1, ubt[0] * ubt[0])
            + inner(1, ubt[1] * ubt[1])
            + inner(1, ubt[2] * ubt[2])
        )
        self.time_hist.append(t)
        self.Epert_hist.append(Epert)
        self.Etot_hist.append(Etot)
        self.ax4.plot(self.time_hist, self.Epert_hist, "g-", label="E' (pert)")
        self.ax4.plot(self.time_hist, self.Etot_hist, "b-", label="E_total")
        self.ax4.legend(loc="best")
        plt.tight_layout()
        if self.save_plots:
            fn = f"{self.plot_dir}/pcf_fluct_t{t:06.2f}_s{tstep:06d}.{self.plot_format}"
            try:
                self.fig.savefig(fn, dpi=self.plot_dpi, bbox_inches="tight")
                if (tstep // self.modplot) % 5 == 0:
                    print(f"Saved plot {fn}")
            except Exception as e:
                print(f"Plot save warning: {e}")
        if self.enable_live_plots:
            plt.draw()
            plt.pause(0.01)

    # --------------- analysis helpers ---------------
    def _gather_profile(self, x_local, profile_local):
        # Gather local x profiles to rank 0 for plotting/analysis.
        if comm.Get_size() == 1:
            return x_local.copy(), profile_local.copy()
        gathered = comm.gather((x_local.copy(), profile_local.copy()), root=0)
        if comm.Get_rank() != 0:
            return None, None
        x_all = np.concatenate([item[0] for item in gathered])
        p_all = np.concatenate([item[1] for item in gathered])
        order = np.argsort(x_all)
        return x_all[order], p_all[order]

    def init_analysis_plots(self):
        if comm.Get_rank() != 0:
            return
        if not self.enable_live_plots:
            plt.switch_backend("Agg")
        (
            self.fig_analysis,
            ((self.ax_mean, self.ax_shear), (self.ax_rms, self.ax_energy)),
        ) = plt.subplots(2, 2, figsize=(12, 8))
        self._setup_analysis_axes()
        if self.enable_live_plots:
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)

    def _setup_analysis_axes(self):
        if comm.Get_rank() != 0:
            return
        self.ax_mean.set_xlabel("Wall-normal (x)")
        self.ax_mean.set_ylabel("Mean streamwise velocity")
        self.ax_mean.set_title("Mean Profile")
        self.ax_mean.grid(True)
        self.ax_shear.set_xlabel("Wall-normal (x)")
        self.ax_shear.set_ylabel("Mean shear d<V>/dx")
        self.ax_shear.set_title("Mean Shear Profile")
        self.ax_shear.grid(True)
        self.ax_rms.set_xlabel("Wall-normal (x)")
        self.ax_rms.set_ylabel("RMS fluctuation")
        self.ax_rms.set_title("Velocity RMS")
        self.ax_rms.grid(True)
        self.ax_energy.set_xlabel("Time")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.set_title("Energy (analysis)")
        self.ax_energy.grid(True)

    def init_spectra_plots(self):
        if comm.Get_rank() != 0:
            return
        if not self.enable_live_plots:
            plt.switch_backend("Agg")
        self.fig_spectra, (self.ax_ky, self.ax_kz) = plt.subplots(1, 2, figsize=(12, 4))
        self._setup_spectra_axes()
        if self.enable_live_plots:
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)

    def _setup_spectra_axes(self):
        if comm.Get_rank() != 0:
            return
        self.ax_ky.set_xlabel("Streamwise wavenumber |k_y|")
        self.ax_ky.set_ylabel("Energy")
        self.ax_ky.set_title("Spectrum in y")
        self.ax_ky.grid(True)
        self.ax_kz.set_xlabel("Spanwise wavenumber |k_z|")
        self.ax_kz.set_ylabel("Energy")
        self.ax_kz.set_title("Spectrum in z")
        self.ax_kz.grid(True)

    def _fold_spectrum(self, k_local, energy_local, kmax, fold_negative):
        spec = np.zeros(kmax + 1, dtype=float)
        k_vals = np.abs(k_local) if fold_negative else k_local
        for k_val, e_val in zip(k_vals, energy_local):
            spec[int(k_val)] += float(e_val)
        return spec

    def _compute_spectra(self, uhat):
        # Spectrum from Fourier coefficients; sum over x and the other Fourier axis.
        energy_density = (
            np.abs(uhat[0]) ** 2 + np.abs(uhat[1]) ** 2 + np.abs(uhat[2]) ** 2
        )
        energy_ky_local = np.sum(energy_density, axis=(0, 2))
        energy_kz_local = np.sum(energy_density, axis=(0, 1))
        ky_full = self.F1.wavenumbers(bcast=False, scaled=False)
        kz_full = self.F2.wavenumbers(bcast=False, scaled=False)
        y_slice = self.TD.local_slice(True)[1]
        z_slice = self.TD.local_slice(True)[2]
        ky_local = ky_full[y_slice]
        kz_local = kz_full[z_slice]
        ky_max = int(np.max(np.abs(ky_full)))
        kz_max = int(np.max(kz_full))
        spec_ky_local = self._fold_spectrum(
            ky_local, energy_ky_local, ky_max, fold_negative=True
        )
        spec_kz_local = self._fold_spectrum(
            kz_local, energy_kz_local, kz_max, fold_negative=False
        )
        spec_ky = comm.allreduce(spec_ky_local)
        spec_kz = comm.allreduce(spec_kz_local)
        ky_axis = np.arange(ky_max + 1, dtype=float) * float(self.F1.domain_factor())
        kz_axis = np.arange(kz_max + 1, dtype=float) * float(self.F2.domain_factor())
        return ky_axis, spec_ky, kz_axis, spec_kz

    def init_ssp_plots(self):
        if comm.Get_rank() != 0:
            return
        if not self.enable_live_plots:
            plt.switch_backend("Agg")
        self.fig_ssp, (self.ax_streaks, self.ax_rolls) = plt.subplots(
            1, 2, figsize=(12, 4)
        )
        self._setup_ssp_axes()
        if self.enable_live_plots:
            plt.tight_layout()
            plt.draw()
            plt.show(block=False)

    def _setup_ssp_axes(self):
        if comm.Get_rank() != 0:
            return
        self.ax_streaks.set_xlabel("Spanwise (z)")
        self.ax_streaks.set_ylabel("Wall-normal (x)")
        self.ax_streaks.set_title("Streaks: <v_total>_y - U_b")
        self.ax_streaks.grid(True)
        self.ax_rolls.set_xlabel("Spanwise (z)")
        self.ax_rolls.set_ylabel("Wall-normal (x)")
        self.ax_rolls.set_title("Rolls: <u,w>_y with <omega_y>_y")
        self.ax_rolls.grid(True)

    def plot_ssp(self, t, tstep):
        if (
            self.conv != 0
            or (not self.save_ssp and not self.enable_live_plots)
            or self.modssp <= 0
            or tstep % self.modssp != 0
        ):
            return
        ubp = self.u_.backward(self.ub)
        ubt = self.total_velocity_physical_from(ubp)
        v_bar = np.mean(ubt.v[1], axis=1)
        streaks = v_bar - self.U_wall * self.x_1d[:, None]
        u_bar = np.mean(ubp.v[0], axis=1)
        w_bar = np.mean(ubp.v[2], axis=1)
        omega_y = self.dwdx().backward().v - self.dudz().backward().v
        omega_bar = np.mean(omega_y, axis=1)
        if comm.Get_rank() != 0:
            return
        if not hasattr(self, "fig_ssp"):
            self.init_ssp_plots()
        self.ax_streaks.clear()
        self.ax_rolls.clear()
        self._setup_ssp_axes()
        X = self.X[0][:, 0, :]
        Z = self.X[2][:, 0, :]
        self.ax_streaks.contourf(Z, X, streaks, 50, cmap="RdBu_r")
        self.ax_rolls.contourf(Z, X, omega_bar, 50, cmap="RdBu_r")
        s = self.ssp_stride
        self.ax_rolls.quiver(
            Z[::s, ::s],
            X[::s, ::s],
            w_bar[::s, ::s],
            u_bar[::s, ::s],
            color="k",
            scale=50,
        )
        plt.tight_layout()
        if self.save_ssp:
            fn = f"{self.analysis_dir}/pcf_ssp_t{t:06.2f}_s{tstep:06d}.{self.plot_format}"
            try:
                self.fig_ssp.savefig(fn, dpi=self.plot_dpi, bbox_inches="tight")
                if (tstep // self.modssp) % 5 == 0:
                    print(f"Saved SSP plot {fn}")
            except Exception as e:
                print(f"SSP plot save warning: {e}")
        if self.enable_live_plots:
            plt.draw()
            plt.pause(0.01)

    def analysis(self, t, tstep):
        if (
            (not self.save_analysis and not self.enable_live_plots)
            or self.modanalysis <= 0
            or tstep % self.modanalysis != 0
        ):
            return
        # Plane-averaged profiles over y,z and RMS fluctuations.
        ubp = self.u_.backward(self.ub)
        ubt = self.total_velocity_physical_from(ubp)
        v_tot = ubt.v[1]
        v_fluct = ubp.v[1]
        dv_dx_fluct = self.dvdx().backward().v
        mean_v_tot_local = np.mean(v_tot, axis=(1, 2))
        mean_vp_local = np.mean(v_fluct, axis=(1, 2))
        mean_shear_local = np.mean(dv_dx_fluct, axis=(1, 2)) + self.dUb_dx
        urms_local = np.sqrt(np.mean(ubp.v[0] ** 2, axis=(1, 2)))
        vrms_local = np.sqrt(np.mean(ubp.v[1] ** 2, axis=(1, 2)))
        wrms_local = np.sqrt(np.mean(ubp.v[2] ** 2, axis=(1, 2)))
        x_full, mean_v_tot = self._gather_profile(self.x_1d, mean_v_tot_local)
        _, mean_vp = self._gather_profile(self.x_1d, mean_vp_local)
        _, mean_shear = self._gather_profile(self.x_1d, mean_shear_local)
        _, urms = self._gather_profile(self.x_1d, urms_local)
        _, vrms = self._gather_profile(self.x_1d, vrms_local)
        _, wrms = self._gather_profile(self.x_1d, wrms_local)
        Epert = (
            inner(1, ubp[0] * ubp[0])
            + inner(1, ubp[1] * ubp[1])
            + inner(1, ubp[2] * ubp[2])
        )
        Etot = (
            inner(1, ubt[0] * ubt[0])
            + inner(1, ubt[1] * ubt[1])
            + inner(1, ubt[2] * ubt[2])
        )
        if comm.Get_rank() != 0:
            return
        if not hasattr(self, "analysis_time_hist"):
            self.analysis_time_hist = []
            self.analysis_Epert_hist = []
            self.analysis_Etot_hist = []
        if not hasattr(self, "analysis_count"):
            self.analysis_count = 0
            self.analysis_x = x_full
            self.mean_v_tot_avg = np.zeros_like(mean_v_tot)
            self.mean_shear_avg = np.zeros_like(mean_shear)
            self.urms_avg = np.zeros_like(urms)
            self.vrms_avg = np.zeros_like(vrms)
            self.wrms_avg = np.zeros_like(wrms)
        self.analysis_count += 1
        # Incremental time-average for profiles.
        weight = 1.0 / self.analysis_count
        self.mean_v_tot_avg += (mean_v_tot - self.mean_v_tot_avg) * weight
        self.mean_shear_avg += (mean_shear - self.mean_shear_avg) * weight
        self.urms_avg += (urms - self.urms_avg) * weight
        self.vrms_avg += (vrms - self.vrms_avg) * weight
        self.wrms_avg += (wrms - self.wrms_avg) * weight
        self.analysis_time_hist.append(t)
        self.analysis_Epert_hist.append(Epert)
        self.analysis_Etot_hist.append(Etot)
        if not hasattr(self, "fig_analysis"):
            self.init_analysis_plots()
        self.ax_mean.clear()
        self.ax_shear.clear()
        self.ax_rms.clear()
        self.ax_energy.clear()
        self._setup_analysis_axes()
        base_profile = self.U_wall * x_full
        self.ax_mean.plot(x_full, mean_v_tot, "b-", label="mean V")
        self.ax_mean.plot(x_full, base_profile, "k--", label="base U_b")
        self.ax_mean.plot(x_full, mean_vp, "m--", label="mean v'")
        if self.analysis_count > 1:
            self.ax_mean.plot(x_full, self.mean_v_tot_avg, "r:", label="time-avg V")
        self.ax_shear.plot(x_full, mean_shear, "b-", label="mean dV/dx")
        self.ax_shear.axhline(
            self.dUb_dx, color="k", linestyle="--", label="base shear"
        )
        if self.analysis_count > 1:
            self.ax_shear.plot(
                x_full, self.mean_shear_avg, "r:", label="time-avg shear"
            )
        self.ax_rms.plot(x_full, urms, "r-", label="u' rms")
        self.ax_rms.plot(x_full, vrms, "g-", label="v' rms")
        self.ax_rms.plot(x_full, wrms, "b-", label="w' rms")
        if self.analysis_count > 1:
            self.ax_rms.plot(x_full, self.urms_avg, "r:", label="u' rms avg")
            self.ax_rms.plot(x_full, self.vrms_avg, "g:", label="v' rms avg")
            self.ax_rms.plot(x_full, self.wrms_avg, "b:", label="w' rms avg")
        self.ax_energy.plot(
            self.analysis_time_hist, self.analysis_Epert_hist, "g-", label="E' (pert)"
        )
        self.ax_energy.plot(
            self.analysis_time_hist, self.analysis_Etot_hist, "b-", label="E_total"
        )
        self.ax_mean.legend(loc="best")
        self.ax_shear.legend(loc="best")
        self.ax_rms.legend(loc="best")
        self.ax_energy.legend(loc="best")
        plt.tight_layout()
        if self.save_analysis:
            fn = f"{self.analysis_dir}/pcf_analysis_t{t:06.2f}_s{tstep:06d}.{self.plot_format}"
            try:
                self.fig_analysis.savefig(fn, dpi=self.plot_dpi, bbox_inches="tight")
                if (tstep // self.modanalysis) % 5 == 0:
                    print(f"Saved analysis plot {fn}")
            except Exception as e:
                print(f"Analysis plot save warning: {e}")
        if self.enable_live_plots:
            plt.draw()
            plt.pause(0.01)

    def spectra(self, t, tstep):
        if (
            (not self.save_spectra and not self.enable_live_plots)
            or self.modspectra <= 0
            or tstep % self.modspectra != 0
        ):
            return
        spectra_data = self._compute_spectra(self.u_.v)
        if comm.Get_rank() != 0:
            return
        ky_axis, spec_ky, kz_axis, spec_kz = spectra_data
        if not hasattr(self, "fig_spectra"):
            self.init_spectra_plots()
        self.ax_ky.clear()
        self.ax_kz.clear()
        self._setup_spectra_axes()
        self.ax_ky.semilogy(ky_axis, np.maximum(spec_ky, 1e-30), "k-")
        self.ax_kz.semilogy(kz_axis, np.maximum(spec_kz, 1e-30), "k-")
        plt.tight_layout()
        if self.save_spectra:
            fn = f"{self.analysis_dir}/pcf_spectra_t{t:06.2f}_s{tstep:06d}.{self.plot_format}"
            try:
                self.fig_spectra.savefig(fn, dpi=self.plot_dpi, bbox_inches="tight")
                if (tstep // self.modspectra) % 5 == 0:
                    print(f"Saved spectra plot {fn}")
            except Exception as e:
                print(f"Spectra plot save warning: {e}")
        if self.enable_live_plots:
            plt.draw()
            plt.pause(0.01)

    def update(self, t, tstep):
        self.plot(t, tstep)
        self.print_energy_and_divergence(t, tstep)
        self.analysis(t, tstep)
        self.spectra(t, tstep)
        self.plot_ssp(t, tstep)


def _parse_cli(params):
    parser = argparse.ArgumentParser(
        description="Plane Couette fluctuations (corrected) diagnostics"
    )
    parser.add_argument(
        "--modplot",
        type=int,
        default=params["modplot"],
        help="Frequency for 2D fields plot",
    )
    parser.add_argument(
        "--modanalysis",
        type=int,
        default=params["modanalysis"],
        help="Frequency for profile analysis plot",
    )
    parser.add_argument(
        "--modspectra",
        type=int,
        default=params["modspectra"],
        help="Frequency for spectra plot",
    )
    parser.add_argument(
        "--modssp",
        type=int,
        default=params["modssp"],
        help="Frequency for SSP (rolls/streaks) plot",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=params["save_plots"],
    )
    parser.add_argument(
        "--save-analysis",
        action=argparse.BooleanOptionalAction,
        default=params["save_analysis"],
    )
    parser.add_argument(
        "--save-spectra",
        action=argparse.BooleanOptionalAction,
        default=params["save_spectra"],
    )
    parser.add_argument(
        "--save-ssp", action=argparse.BooleanOptionalAction, default=params["save_ssp"]
    )
    parser.add_argument(
        "--enable-live-plots",
        action=argparse.BooleanOptionalAction,
        default=params["enable_live_plots"],
    )
    parser.add_argument(
        "--ssp-stride",
        type=int,
        default=params["ssp_stride"],
        help="Quiver stride for SSP rolls",
    )
    parser.add_argument("--plot-format", type=str, default=params["plot_format"])
    parser.add_argument("--plot-dpi", type=int, default=params["plot_dpi"])
    parser.add_argument(
        "--linear",
        choices=["dns", "eigs", "nonmodal"],
        default="dns",
        help="run a linear eigenvalue or non-modal analysis instead of DNS",
    )
    parser.add_argument(
        "--linear-nx",
        type=int,
        default=params["N"][0],
        help="wall-normal collocation points for linear analysis",
    )
    parser.add_argument(
        "--ky", type=float, default=1.0, help="streamwise linear-analysis wavenumber"
    )
    parser.add_argument(
        "--kz", type=float, default=1.0, help="spanwise linear-analysis wavenumber"
    )
    parser.add_argument(
        "--linear-times",
        type=str,
        default="1,5,10,20",
        help="comma-separated times for non-modal transient growth",
    )
    parser.add_argument(
        "--linear-n-return",
        type=int,
        default=8,
        help="number of leading eigenvalues to print",
    )
    parser.add_argument(
        "--linear-n-modes",
        type=int,
        default=None,
        help="number of finite modes retained for non-modal analysis",
    )
    parser.add_argument(
        "--linear-finite-cap",
        type=float,
        default=1.0e8,
        help="discard generalized eigenvalues above this magnitude",
    )
    args = parser.parse_args()
    return vars(args)


def _run_linear_variant(params):
    from _linear_analysis import parse_times, print_eigenvalues, print_transient_growth
    from _pcf_linear import PlaneCouetteLinear

    lin = PlaneCouetteLinear.couette(
        nx=params["linear_nx"],
        Re=params["Re"],
        U_wall=params["U_wall"],
        mhd=False,
    )
    ky = params["ky"]
    kz = params["kz"]
    if params["linear"] == "eigs":
        w, _ = lin.eigs(
            ky,
            kz,
            n_return=params["linear_n_return"],
            finite_cap=params["linear_finite_cap"],
        )
        if comm.Get_rank() == 0:
            print(
                f"Plane Couette hydro linear eigenvalues: Re={params['Re']:g}, ky={ky:g}, kz={kz:g}"
            )
            print_eigenvalues(w)
        return {"eigenvalues": [(float(s.real), float(s.imag)) for s in w]}

    rows = lin.nonmodal_growth(
        ky,
        kz,
        parse_times(params["linear_times"]),
        n_modes=params["linear_n_modes"],
        finite_cap=params["linear_finite_cap"],
    )
    if comm.Get_rank() == 0:
        print(
            f"Plane Couette hydro non-modal growth: Re={params['Re']:g}, ky={ky:g}, kz={kz:g}"
        )
        print_transient_growth(rows)
    return {"transient_growth": rows}


def _strip_linear_keys(params):
    for key in (
        "linear",
        "linear_nx",
        "ky",
        "kz",
        "linear_times",
        "linear_n_return",
        "linear_n_modes",
        "linear_finite_cap",
    ):
        params.pop(key, None)


def run_pcf_fluctuation():
    # Default parameters for a modest-resolution PCF run
    params = dict(
        N=(32, 64, 32),
        domain=((-1, 1), (0, 4 * np.pi), (0, 2 * np.pi)),
        Re=400.0,
        U_wall=1.0,
        dt=0.01,
        conv=0,
        modplot=50,
        modsave=500,
        moderror=10,
        modanalysis=50,
        modspectra=50,
        modssp=100,
        filename="PCF_fluct_Re400",
        family="C",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=250,
        perturbation_amplitude=0.1,
        timestepper="IMEXRK222",
        enable_live_plots=False,
        save_plots=True,
        save_analysis=True,
        save_spectra=True,
        save_ssp=False,
        ssp_stride=3,
        plot_format="png",
        plot_dpi=150,
    )
    params.update(_parse_cli(params))
    if params["linear"] != "dns":
        return _run_linear_variant(params)
    _strip_linear_keys(params)
    if comm.Get_rank() == 0:
        print("Simulation parameters:")
        [print(f"  {k}: {v}") for k, v in params.items()]
        print()
        print("Plots will be saved" if params["save_plots"] else "No plot saving")
    solver = PlaneCouetteFluctuation(**params)
    t, tstep = solver.initialize(from_checkpoint=False)
    end_time = 50.0
    try:
        solver.solve(t=t, tstep=tstep, end_time=end_time)
        if comm.Get_rank() == 0:
            try:
                from mpi4py_fft import generate_xdmf

                generate_xdmf(f"{params['filename']}_U.h5")
                print("XDMF generated")
            except Exception:
                pass
    finally:
        if shenfun_cleanup is not None:
            shenfun_cleanup(vars(solver))
    return None


if __name__ == "__main__":
    if comm.Get_size() > 1:
        print(f"Running on {comm.Get_size()} MPI processes")
    else:
        print("Running in serial mode")
    run_pcf_fluctuation()
