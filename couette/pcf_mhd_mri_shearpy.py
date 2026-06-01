"""
Plane Couette MHD analogue of the shearpy MRI setup.

This extends ``pcf_mhd_divfree.py`` with the linear shearing-box terms used by
shearpy while keeping Shenfun's wall-bounded Plane Couette discretization:

    component 0 / coordinate x: radial, wall-normal, shear-gradient direction
    component 1 / coordinate y: azimuthal, streamwise, wall-motion direction
    component 2 / coordinate z: vertical, spanwise direction

    U_b(x) = -S*x*e_y
    du_x/dt += 2*Omega*u_y
    du_y/dt += (S - 2*Omega)*u_x
    dB_y/dt += -S*B_x

The last magnetic source is not inserted as a separate component update.  The
solver advances a vector potential and uses ``curl(U x B)`` through
``dA/dt = U x B + eta*lap(A)``, so the base-flow induction contribution follows
from ``B dot grad(U_b)``.  A uniform imposed field ``B0=(0, by, bz)`` is carried
separately from ``curl(A)`` to reproduce shearpy's net-field MRI cases without
breaking the compatible-space ``div(curl(A)) = 0`` invariant.

This is not a shearing-periodic box: the radial/shear direction is replaced by
no-slip Plane Couette walls.  The script is intended for testing the closest
wall-bounded PCF analogue of the shearpy MRI source terms and parameters.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
from shenfun import *  # noqa: F401,F403

from pcf_mhd_divfree import PlaneCouetteMHDDivFree, _positive_int, shenfun_cleanup


class PlaneCouetteMRIShearpy(PlaneCouetteMHDDivFree):
    """PCF MHD with shearpy-style rotation, shear, and imposed net field."""

    def __init__(
        self,
        N=(16, 32, 16),
        domain=((-2, 2), (0, 4.0), (0, 1.0)),
        Re=1000.0,
        Rm=1000.0,
        shear_rate=1.0,
        omega=2.0 / 3.0,
        by=0.0,
        bz=0.025,
        velocity_scale=1.0,
        dt=0.001,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_MHD_MRI_shearpy",
        family="L",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.0,
        timestepper="IMEXRK222",
        prefer_numba=False,
        store_history=False,
    ):
        self.shear_rate = float(shear_rate)
        self.omega = float(omega)
        self.background_b = np.array([0.0, float(by), float(bz)], dtype=float)
        self.velocity_scale = float(velocity_scale)
        self.x_bounds = (float(domain[0][0]), float(domain[0][1]))
        self.x_center = 0.5 * (self.x_bounds[0] + self.x_bounds[1])
        self.x_half_width = 0.5 * (self.x_bounds[1] - self.x_bounds[0])
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if self.x_half_width <= 0.0:
            raise ValueError("x domain must have positive width")

        super().__init__(
            N=N,
            domain=domain,
            Re=Re,
            Rm=Rm,
            U_wall=self.velocity_scale,
            dt=dt,
            conv=conv,
            moderror=moderror,
            modsave=modsave,
            filename=filename,
            family=family,
            padding_factor=padding_factor,
            checkpoint=checkpoint,
            perturbation_amplitude=perturbation_amplitude,
            magnetic_amplitude=magnetic_amplitude,
            timestepper=timestepper,
            prefer_numba=prefer_numba,
            store_history=store_history,
        )

        # Replace the base solver's PCF profile U_wall*x by the shearing-box
        # convention used in shearpy: U_b=-S*x*e_y.
        self.Ub = -self.shear_rate * self.X[0]
        self.Ub_1d = -self.shear_rate * self.x_1d
        self.dUb_dx = -self.shear_rate
        self.Ub_pad = -self.shear_rate * self.Xp[0]

        self.q_shear = self.shear_rate / self.omega if self.omega != 0.0 else math.inf
        self.kappa2 = 2.0 * self.omega * (2.0 * self.omega - self.shear_rate)

        if comm.Get_rank() == 0:
            print("MRI/rotating-shear extension enabled")
            print(
                f"  shear S={self.shear_rate:g}, Omega={self.omega:g}, "
                f"q=S/Omega={self.q_shear:g}, kappa^2={self.kappa2:g}"
            )
            print(
                "  base flow: U_b(x)=-S*x*e_y; "
                f"wall speeds y(top,bottom)=({-self.shear_rate*self.x_bounds[1]:+g}, "
                f"{-self.shear_rate*self.x_bounds[0]:+g})"
            )
            print(
                f"  imposed mean field B0=(0, {self.background_b[1]:g}, "
                f"{self.background_b[2]:g})"
            )
            print()

    def _total_b_components(self, bphys):
        return (
            bphys[0] + self.background_b[0],
            bphys[1] + self.background_b[1],
            bphys[2] + self.background_b[2],
        )

    def _wall_factor(self):
        xi = (self.X[0] - self.x_center) / self.x_half_width
        return 1.0 - xi**2

    # ------------------------------------------------------------------
    # Saturation-mechanism diagnostics (mean-shear feedback vs parasites)
    #
    # These implement the discriminating measurements called for in the
    # critical review of ``mri_3d_knobloch_julien_extension.tex``: the
    # x-resolved mean shear and stress profiles needed to test the JK
    # mean-shear-cancellation closure  nu d_x <V> = <u_x u_y - b_x b_y>,
    # the interior channel amplitude A(t), and the non-axisymmetric
    # (parasite) energy fraction.  All are serial-exact; the y,z reductions
    # are MPI-safe via allreduce, assuming the wall-normal x axis is not the
    # distributed one (as in the shipped runs).  The z-FFT channel/parasite
    # projections assume full (y,z) are local on each rank (serial / x-slab).
    # ------------------------------------------------------------------
    def _yz_mean(self, field):
        """y,z-average of a physical field -> 1D wall-normal (x) profile."""
        local_sum = np.asarray(field).sum(axis=(1, 2))
        if comm.Get_size() > 1:
            local_sum = comm.allreduce(local_sum)
        return local_sum / (self.F1.N * self.F2.N)

    def channel_vertical_mode(self):
        """Index n of the vertical Fourier mode closest to the optimal MRI
        channel wavenumber K0 = sqrt(15)/4 * Omega / v_A (Keplerian optimum)."""
        vA = max(abs(self.background_b[2]), 1e-12)
        K0 = math.sqrt(15.0) / 4.0 * self.omega / vA
        lz = self.domain_lengths[2]
        n = int(round(K0 * lz / (2.0 * math.pi)))
        return max(1, n), K0

    def mean_profiles(self):
        """Return x-resolved mean profiles used by the JK closure test.

        Vbar(x)   total horizontally-averaged azimuthal velocity
        Seff(x)   effective shear  d<V>/dx = -S + <d_x v'>  (code convention)
        reynolds  <u_x u_y>(x)
        maxwell   <-b_x b_y>(x)  using the TOTAL field (incl. imposed B0)
        total     reynolds + maxwell
        The JK mean-shear closure predicts  nu*Seff(x) ~ total(x) + const,
        with Seff -> 0 (marginal) in the core when the channel saturates.
        """
        self.update_B_from_A()
        ub = self.u_.backward(self.ub)
        bp = self.b_.backward(self.bb)
        bt0, bt1, _ = self._total_b_components(bp)
        dvdx = self.dvdx().backward()
        Vbar = self.Ub_1d + self._yz_mean(ub[1])
        Seff = self.dUb_dx + self._yz_mean(dvdx)
        reynolds = self._yz_mean(np.asarray(ub[0]) * np.asarray(ub[1]))
        maxwell = self._yz_mean(-(np.asarray(bt0) * np.asarray(bt1)))
        return {
            "x": self.x_1d.copy(),
            "Vbar": Vbar,
            "Seff": Seff,
            "reynolds": reynolds,
            "maxwell": maxwell,
            "total": reynolds + maxwell,
        }

    def channel_amplitude(self, interior_frac=0.6):
        """Interior channel amplitude A(t): RMS over the central interior of
        the k_y=0 velocity projected onto the dominant vertical channel mode.
        Returns (A, n) with n the vertical Fourier index used."""
        ub = self.u_.backward(self.ub)
        ux0 = np.asarray(ub[0]).mean(axis=1)        # k_y=0 part: (Nx, Nz)
        uy0 = np.asarray(ub[1]).mean(axis=1)
        nz = self.F2.N
        fx = np.fft.rfft(ux0, axis=1) / nz
        fy = np.fft.rfft(uy0, axis=1) / nz
        n, _ = self.channel_vertical_mode()
        n = min(n, fx.shape[1] - 1)
        amp_x = 2.0 * np.sqrt(np.abs(fx[:, n]) ** 2 + np.abs(fy[:, n]) ** 2)
        xi = (self.x_1d - self.x_center) / self.x_half_width
        mask = np.abs(xi) <= interior_frac
        if not np.any(mask):
            mask = np.ones_like(xi, dtype=bool)
        A = float(np.sqrt(np.mean(amp_x[mask] ** 2)))
        return A, n

    def parasite_diagnostic(self, interior_frac=0.6):
        """Identify the dominant non-axisymmetric (k_y!=0) mode and the fraction
        of perturbation energy it carries.  Returns dict."""
        ub = self.u_.backward(self.ub)
        xi = (self.x_1d - self.x_center) / self.x_half_width
        mask = np.abs(xi) <= interior_frac
        if not np.any(mask):
            mask = np.ones_like(xi, dtype=bool)
        ny, nz = self.F1.N, self.F2.N
        E = np.zeros((ny, nz))
        for c in range(3):
            f = np.fft.fft2(np.asarray(ub[c])[mask], axes=(1, 2)) / (ny * nz)
            E += (np.abs(f) ** 2).sum(axis=0)
        Etot = float(E.sum())
        Ena = E.copy()
        Ena[0, :] = 0.0                              # remove axisymmetric k_y=0 row
        i, j = np.unravel_index(int(np.argmax(Ena)), Ena.shape)
        ly, lz = self.domain_lengths[1], self.domain_lengths[2]
        ky = 2.0 * math.pi * (i if i <= ny // 2 else i - ny) / ly
        kz = 2.0 * math.pi * (j if j <= nz // 2 else j - nz) / lz
        return {
            "parasite_ky": float(ky),
            "parasite_kz": float(kz),
            "parasite_energy": float(Ena[i, j]),
            "nonaxi_fraction": float(Ena.sum() / max(Etot, 1e-30)),
        }

    def save_profiles(self, path):
        """Save the x-resolved mean/stress profiles to a .npz file (rank 0)."""
        prof = self.mean_profiles()
        if comm.Get_rank() == 0:
            np.savez(
                path,
                shear_rate=self.shear_rate,
                omega=self.omega,
                bz=self.background_b[2],
                nu=self.U_wall / self.Re,
                eta=self.eta,
                **prof,
            )
            print(f"Saved mean/stress profiles to {path}")

    def initialize(self, from_checkpoint: bool = False):
        if from_checkpoint:
            return self.init_from_checkpoint()

        X = self.X
        wall = self._wall_factor()

        U = Array(self.BD)
        U[...] = 0.0
        if self.perturbation_amplitude > 0:
            if comm.Get_rank() == 0:
                print(f"Adding velocity perturbations amp={self.perturbation_amplitude}")
            amp = self.perturbation_amplitude
            ly = self.F1.domain[1]
            lz = self.F2.domain[1]
            ky = 2 * np.pi / ly
            kz = 2 * np.pi / lz
            # Dominant axisymmetric (k_y = 0) channel-mode content drives the
            # MRI: the fastest growing mode is independent of the azimuthal y
            # and varies in the vertical z (the imposed-field direction) and
            # across the wall-normal x.  Seeding it directly removes the long
            # incubation that a purely non-axisymmetric seed needs before the
            # channel mode is fed nonlinearly.
            for n in (1, 2, 3):
                U[0] += amp * wall * np.cos(n * kz * X[2])
                U[1] += amp * wall * np.sin(n * kz * X[2])
            # A little non-axisymmetric content breaks exact symmetry and seeds
            # the 3D dynamics that take over once the channel mode saturates.
            U[0] += 0.1 * amp * wall * np.sin(ky * X[1]) * np.cos(kz * X[2])
            U[1] += 0.1 * amp * wall * np.cos(ky * X[1]) * np.sin(kz * X[2])
            U[2] += 0.1 * amp * wall * np.sin(2 * ky * X[1]) * np.cos(2 * kz * X[2])
        U.forward(self.u_)
        self.u_.mask_nyquist(self.mask)
        self.g_[:] = 1j * self.K[1] * self.u_[2] - 1j * self.K[2] * self.u_[1]

        A = Array(self.CD)
        A[...] = 0.0
        if self.magnetic_amplitude > 0:
            if comm.Get_rank() == 0:
                print(f"Adding magnetic perturbations through A amp={self.magnetic_amplitude}")
            ky = 2 * np.pi / self.F1.domain[1]
            kz = 2 * np.pi / self.F2.domain[1]
            amp = self.magnetic_amplitude
            A[0] += amp * wall * (1.0 / kz) * np.sin(ky * X[1]) * np.sin(kz * X[2])
        A.forward(self.a_)
        self.a_.mask_nyquist(self.mask)
        self.update_B_from_A()

        if comm.Get_rank() == 0:
            diag = self.compute_diagnostics(0.0, 0)
            print("Initial fields ready")
            print(f"Initial divB: L2={diag['divb_l2']:.3e} Linf={diag['divb_linf']:.3e}")
            print()
        return 0.0, 0

    def convection(self):
        if self.conv != 0:
            raise NotImplementedError("Only conv=0 is implemented for PCF MRI MHD")

        self.update_B_from_A()

        H = self.H_.v
        up = self.u_.backward(padding_factor=self.padding_factor)
        upv = up.v
        bpv = self.b_.backward(padding_factor=self.padding_factor).v
        bt0, bt1, bt2 = self._total_b_components(bpv)

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

        # KMM stores H=N-F because the velocity equations apply -H after the
        # pressure projection.  These two additions therefore produce the
        # shearpy source terms +2*Omega*u_y and -2*Omega*u_x.
        n0 += -2.0 * self.omega * upv[1]
        n1 += 2.0 * self.omega * upv[0]

        # Lorentz force J x B_total, with J=curl(curl(A)); the imposed uniform
        # field has no current but participates in the force.
        self.update_J_from_B()
        jbp = self.j_.backward(padding_factor=self.padding_factor).v
        l0 = jbp[1] * bt2 - jbp[2] * bt1
        l1 = jbp[2] * bt0 - jbp[0] * bt2
        l2 = jbp[0] * bt1 - jbp[1] * bt0
        n0 -= l0
        n1 -= l1
        n2 -= l2

        H[0] = self.TDp.forward(n0, H[0])
        H[1] = self.TDp.forward(n1, H[1])
        H[2] = self.TDp.forward(n2, H[2])
        self.H_.mask_nyquist(self.mask)

        # Vector-potential forcing from the total velocity and total magnetic
        # field.  This includes base-flow advection and -S*B_x*e_y induction.
        ubt = self.ub_total_pad
        ubt[...] = up
        ubt.v[1] += self.Ub_pad

        HA = self.HA_.v
        HA[0] = self.TDp.forward(ubt.v[1] * bt2 - ubt.v[2] * bt1, HA[0])
        HA[1] = self.TDp.forward(ubt.v[2] * bt0 - ubt.v[0] * bt2, HA[1])
        HA[2] = self.TDp.forward(ubt.v[0] * bt1 - ubt.v[1] * bt0, HA[2])
        self.HA_.mask_nyquist(self.mask)

    def compute_diagnostics(self, t, tstep, full=True):
        diag = super().compute_diagnostics(t, tstep)
        bp = self.b_.backward(self.bb)
        bt0, bt1, bt2 = self._total_b_components(bp)
        Emag_total = float(inner(1, bt0 * bt0) + inner(1, bt1 * bt1) + inner(1, bt2 * bt2))
        bmax_total = float(max(np.max(np.abs(bt0)), np.max(np.abs(bt1)), np.max(np.abs(bt2))))

        # Volume-averaged Reynolds/Maxwell stress and MRI transport alpha.
        ub = self.u_.backward(self.ub)
        reynolds = float(inner(1, ub[0] * ub[1])) / self.volume
        maxwell = -float(inner(1, bt0 * bt1)) / self.volume
        vA2 = float(self.background_b[2]) ** 2
        alpha = (reynolds + maxwell) / vA2 if vA2 > 0 else float("nan")

        # Interior effective-shear reduction (the JK mean-shear test).
        prof = self.mean_profiles()
        xi = (self.x_1d - self.x_center) / self.x_half_width
        core = np.abs(xi) <= 0.6
        seff_core = float(np.mean(prof["Seff"][core])) if np.any(core) else float("nan")
        # dUb_dx = -S, so no modification -> ratio 1, full cancellation -> 0.
        shear_reduction = 1.0 - seff_core / self.dUb_dx if self.dUb_dx != 0 else float("nan")

        diag.update(
            Emag_total=Emag_total,
            bmax_total=bmax_total,
            B0y=float(self.background_b[1]),
            B0z=float(self.background_b[2]),
            shear_rate=float(self.shear_rate),
            omega=float(self.omega),
            q_shear=float(self.q_shear),
            kappa2=float(self.kappa2),
            reynolds_stress=reynolds,
            maxwell_stress=maxwell,
            alpha=alpha,
            seff_core=seff_core,
            shear_reduction=shear_reduction,
        )

        if full:
            A, nK = self.channel_amplitude()
            par = self.parasite_diagnostic()
            diag.update(channel_amp=A, channel_kz_n=int(nK), **par)

        self.last_diagnostics = diag
        if self.store_history and self.history and self.history[-1]["tstep"] == int(tstep):
            self.history[-1] = diag.copy()
        return diag

    def print_diagnostics(self, t, tstep):
        if self.moderror <= 0 or tstep % self.moderror != 0:
            return
        diag = self.compute_diagnostics(t, tstep)
        if comm.Get_rank() == 0:
            print(
                f"t={diag['t']:7.3f} E'={diag['Epert']:10.3e} "
                f"EmagTot={diag['Emag_total']:10.3e} "
                f"A_ch={diag.get('channel_amp', float('nan')):.3e} "
                f"alpha={diag['alpha']:+.3e} "
                f"dShear={diag['shear_reduction']:+.3f} "
                f"Rey={diag['reynolds_stress']:+.2e} Max={diag['maxwell_stress']:+.2e} "
                f"f_na={diag.get('nonaxi_fraction', float('nan')):.2e} "
                f"divbL2={diag['divb_l2']:6.1e}"
            )


def _parse_args(defaults):
    p = argparse.ArgumentParser(description="Plane Couette analogue of shearpy MRI MHD")
    p.add_argument("--nx", type=int, default=defaults["N"][0])
    p.add_argument("--ny", type=int, default=defaults["N"][1])
    p.add_argument("--nz", type=int, default=defaults["N"][2])
    p.add_argument("--lx", type=float, default=defaults["domain"][0][1] - defaults["domain"][0][0])
    p.add_argument("--ly", type=float, default=defaults["domain"][1][1] - defaults["domain"][1][0])
    p.add_argument("--lz", type=float, default=defaults["domain"][2][1] - defaults["domain"][2][0])
    p.add_argument("--Re", type=float, default=defaults["Re"])
    p.add_argument("--Rm", type=float, default=defaults["Rm"])
    p.add_argument("--shear", type=float, default=defaults["shear_rate"])
    p.add_argument("--omega", type=float, default=defaults["omega"])
    p.add_argument("--by", type=float, default=defaults["by"])
    p.add_argument("--bz", type=float, default=defaults["bz"])
    p.add_argument("--velocity-scale", type=float, default=defaults["velocity_scale"])
    p.add_argument("--dt", type=float, default=defaults["dt"])
    p.add_argument("--end-time", type=float, default=defaults["end_time"])
    p.add_argument("--moderror", type=int, default=defaults["moderror"])
    p.add_argument("--modsave", type=int, default=defaults["modsave"])
    p.add_argument("--checkpoint", type=int, default=defaults["checkpoint"])
    p.add_argument("--family", choices=["C", "L"], default=defaults["family"])
    p.add_argument("--timestepper", type=str, default=defaults["timestepper"])
    p.add_argument("--perturbation-amplitude", type=float, default=defaults["perturbation_amplitude"])
    p.add_argument("--magnetic-amplitude", type=float, default=defaults["magnetic_amplitude"])
    p.add_argument("--filename", type=str, default=defaults["filename"])
    p.add_argument("--prefer-numba", action="store_true", default=defaults["prefer_numba"])
    p.add_argument("--store-history", action="store_true", default=defaults["store_history"])
    p.add_argument("--max-divb-l2", type=float, default=None)
    p.add_argument("--max-divu-l2", type=float, default=None)
    p.add_argument("--assert-every-step", action="store_true")
    p.add_argument("--save-profiles", type=str, default=None,
                   help="path to write x-resolved mean/stress profiles (.npz) at end")
    return p.parse_args()


def run(argv=None):
    defaults = dict(
        N=(16, 32, 16),
        domain=((-2, 2), (0, 4.0), (0, 1.0)),
        Re=1000.0,
        Rm=1000.0,
        shear_rate=1.0,
        omega=2.0 / 3.0,
        by=0.0,
        bz=0.025,
        velocity_scale=1.0,
        dt=0.001,
        conv=0,
        moderror=10,
        modsave=1000000,
        filename="PCF_MHD_MRI_shearpy",
        family="L",
        padding_factor=(1, 1.5, 1.5),
        checkpoint=500,
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.0,
        timestepper="IMEXRK222",
        prefer_numba=False,
        store_history=False,
        end_time=0.01,
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
    lx = float(args.lx)
    if lx <= 0.0:
        raise ValueError("lx must be positive")
    params["domain"] = ((-0.5 * lx, 0.5 * lx), (0, float(args.ly)), (0, float(args.lz)))
    params["Re"] = args.Re
    params["Rm"] = args.Rm
    params["shear_rate"] = args.shear
    params["omega"] = args.omega
    params["by"] = args.by
    params["bz"] = args.bz
    params["velocity_scale"] = args.velocity_scale
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

    solver = PlaneCouetteMRIShearpy(**params)
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
        if args.save_profiles is not None:
            solver.save_profiles(args.save_profiles)
    finally:
        if shenfun_cleanup is not None:
            shenfun_cleanup(vars(solver))

    if comm.Get_rank() == 0:
        print("Final diagnostics:")
        keys = (
            "t",
            "tstep",
            "Epert",
            "Etot",
            "Emag",
            "Emag_total",
            "divu_l2",
            "divb_l2",
            "divu_rel",
            "divb_rel",
            "bmax_total",
            "mean_shear",
            "shear_reduction",
            "seff_core",
            "channel_amp",
            "channel_kz_n",
            "reynolds_stress",
            "maxwell_stress",
            "alpha",
            "nonaxi_fraction",
            "parasite_ky",
            "q_shear",
            "kappa2",
        )
        for key in keys:
            print(f"  {key}: {final[key]:.16e}" if isinstance(final[key], float) else f"  {key}: {final[key]}")
    return final


if __name__ == "__main__":
    if comm.Get_size() > 1:
        print(f"Running on {comm.Get_size()} MPI processes")
    else:
        print("Running in serial mode")
    run()
