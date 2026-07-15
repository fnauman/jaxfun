"""Shearpy-style rotating Plane Couette MHD analogue in jaxfun.

This extends examples.pcf_mhd_jax with the linear shearing-box source terms and
imposed net magnetic field used by couette/pcf_mhd_mri_shearpy.py.
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import sympy as sp

try:
    from examples.pcf_mhd_jax import MHDState, PlaneCouetteMHDJax
except ModuleNotFoundError:  # direct script execution from examples/
    from pcf_mhd_jax import MHDState, PlaneCouetteMHDJax

from jaxfun.galerkin import InnerKind, TestFunction, TrialFunction, inner
from jaxfun.galerkin.inner import integrate
from jaxfun.integrators.nonlinear import physical_cross


class PlaneCouetteMRIShearpyJax(PlaneCouetteMHDJax):
    """Rotating-shear PCF MHD analogue following the shearpy MRI reference.

    Reference: couette/pcf_mhd_mri_shearpy.py:39-124 and :313-379.
    """

    def __init__(
        self,
        N: tuple[int, int, int] = (17, 16, 16),
        domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-1.0, 1.0),
            (0.0, 4.0 * float(sp.pi)),
            (0.0, 2.0 * float(sp.pi)),
        ),
        Re: float = 400.0,
        Rm: float | None = None,
        omega: float = 1.0,
        shear_rate: float = 1.0,
        background_b: tuple[float, float, float] = (0.0, 0.0, 0.025),
        dt: float = 0.01,
        family: str = "L",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        perturbation_amplitude: float = 0.05,
        magnetic_amplitude: float = 0.0,
        magnetic_seed: str = "ax_yz",
    ) -> None:
        self.omega = float(omega)
        self.shear_rate = float(shear_rate)
        self.background_b = tuple(float(v) for v in background_b)
        self.magnetic_seed = str(magnetic_seed)
        if self.magnetic_seed not in {"ax_yz", "sinusoidal_bz_x"}:
            raise ValueError("magnetic_seed must be one of {ax_yz, sinusoidal_bz_x}")
        self.x_bounds = (float(domain[0][0]), float(domain[0][1]))
        self.x_center = 0.5 * (self.x_bounds[0] + self.x_bounds[1])
        self.x_half_width = 0.5 * (self.x_bounds[1] - self.x_bounds[0])
        if self.x_half_width <= 0.0:
            raise ValueError("x domain must have positive width")
        super().__init__(
            N=N,
            domain=domain,
            Re=Re,
            Rm=Rm,
            U_wall=1.0,
            dt=dt,
            family=family,
            padding_factor=padding_factor,
            perturbation_amplitude=perturbation_amplitude,
            magnetic_amplitude=magnetic_amplitude,
        )
        self.Ub = -self.shear_rate * self.X[0]
        self.Ubp = -self.shear_rate * self.Xp[0]
        self.dUb_dx = -self.shear_rate
        self.q_shear = self.shear_rate / self.omega if self.omega != 0 else math.inf
        self.kappa2 = 2.0 * self.omega * (2.0 * self.omega - self.shear_rate)

    def _wall_factor(self):
        xi = (self.X[0] - self.x_center) / self.x_half_width
        return 1.0 - xi**2

    def initial_state(self) -> MHDState:
        """Return the shearpy MRI channel-mode seed used by the reference."""
        x, y, z = self.X
        wall = self._wall_factor()
        amp = self.perturbation_amplitude
        Ly = self.domain[1][1] - self.domain[1][0]
        Lz = self.domain[2][1] - self.domain[2][0]
        ky = 2.0 * jnp.pi / Ly
        kz = 2.0 * jnp.pi / Lz
        u0 = jnp.zeros(self.TD.num_quad_points)
        u1 = jnp.zeros_like(u0)
        u2 = jnp.zeros_like(u0)
        if amp != 0.0:
            for harmonic in (1, 2, 3):
                u0 = u0 + amp * wall * jnp.cos(harmonic * kz * z)
                u1 = u1 + amp * wall * jnp.sin(harmonic * kz * z)
            u0 = u0 + 0.1 * amp * wall * jnp.sin(ky * y) * jnp.cos(kz * z)
            u1 = u1 + 0.1 * amp * wall * jnp.cos(ky * y) * jnp.sin(kz * z)
            u2 = u2 + 0.1 * amp * wall * jnp.sin(2.0 * ky * y) * jnp.cos(2.0 * kz * z)
        flow = self.state_from_physical((u0, u1, u2))

        mag_amp = self.magnetic_amplitude
        ax = jnp.zeros(self.TD.num_quad_points)
        ay = jnp.zeros_like(ax)
        az = jnp.zeros_like(ax)
        if mag_amp != 0.0 and self.magnetic_seed == "ax_yz":
            ax = mag_amp * wall * (1.0 / kz) * jnp.sin(ky * y) * jnp.sin(kz * z)
        elif mag_amp != 0.0:
            # Standard zero-net-flux vertical field: Bz = d_x Ay =
            # mag_amp*sin(pi*x/h). Ay vanishes at both conducting walls.
            xi = (x - self.x_center) / self.x_half_width
            ay = ay - (mag_amp * self.x_half_width / jnp.pi) * (
                jnp.cos(jnp.pi * xi) + 1.0
            )
        return MHDState(flow=flow, A=self._A_state_from_physical((ax, ay, az)))

    def _total_B_physical(self, B, padded: bool = False):
        bp = self._backward_B(B, padded=padded)
        return tuple(bp[i] + self.background_b[i] for i in range(3))

    def _mhd_convection(self, state: MHDState):
        flow = state.flow
        up = self._backward_velocity(flow.u, padded=True)
        grads = self._velocity_gradients(flow.u)
        n = (
            up[0] * grads["dudx"] + up[1] * grads["dudy"] + up[2] * grads["dudz"],
            up[0] * grads["dvdx"] + up[1] * grads["dvdy"] + up[2] * grads["dvdz"],
            up[0] * grads["dwdx"] + up[1] * grads["dwdy"] + up[2] * grads["dwdz"],
        )
        n = self._add_base_convection(n, up, grads)
        n = (
            n[0] - 2.0 * self.omega * up[1],
            n[1] + 2.0 * self.omega * up[0],
            n[2],
        )

        B = self.update_B_from_A(state.A)
        J = self.update_J_from_B(B)
        bp_total = self._total_B_physical(B, padded=True)
        jp = self._backward_J(J, padded=True)
        lorentz = physical_cross(jp, bp_total)
        n = tuple(ni - li for ni, li in zip(n, lorentz, strict=True))
        H = tuple(self.TD.mask_nyquist(self.TDp.forward(ni)) for ni in n)

        utotal = (up[0], up[1] + self.Ubp, up[2])
        emf = physical_cross(utotal, bp_total)
        HA = self._A_forward_emf(emf)
        return H, HA

    def _dissipation_parts(self, state: MHDState, B) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Physical dissipation rates (nu * int |grad u|^2, eta * int |grad b|^2).

        These close the shearing-box budget d(E_phys)/dt = S V <total_stress> -
        dissipation between cadence rows (production/health.py).
        """
        u_spaces = (self.TB, self.TD, self.TD)
        b_spaces = self.b_coeff_spaces
        derivatives = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        diss_kinetic = jnp.asarray(0.0, dtype=jnp.real(state.flow.u[0]).dtype)
        for ui, space in zip(state.flow.u, u_spaces, strict=True):
            for deriv in derivatives:
                g = space.backward_primitive(ui, deriv)
                diss_kinetic = diss_kinetic + jnp.real(
                    integrate(jnp.conj(g) * g, self.TC)
                )
        diss_magnetic = jnp.asarray(0.0, dtype=diss_kinetic.dtype)
        for bi, space in zip(B, b_spaces, strict=True):
            for deriv in derivatives:
                g = space.backward_primitive(bi, deriv)
                diss_magnetic = diss_magnetic + jnp.real(
                    integrate(jnp.conj(g) * g, self.TC)
                )
        return self.nu * diss_kinetic, self.eta * diss_magnetic

    def _kz1_rms(self, fields) -> jnp.ndarray:
        """RMS amplitude of the axisymmetric fundamental vertical mode."""

        nz = int(fields[0].shape[2])
        if nz < 2:
            return jnp.asarray(0.0)
        wall_weights = jnp.asarray(self.TD.basespaces[0].integration_weights())
        wall_weights = wall_weights / jnp.sum(wall_weights)
        power = jnp.asarray(0.0, dtype=jnp.real(fields[0]).dtype)
        for field in fields:
            profile = jnp.mean(jnp.real(field), axis=1)
            coefficient = jnp.fft.fft(profile, axis=1)[:, 1] / nz
            power = power + 2.0 * jnp.sum(wall_weights * jnp.abs(coefficient) ** 2)
        return jnp.sqrt(power)

    def diagnostics(self, state: MHDState) -> dict[str, jnp.ndarray]:
        diag = super().diagnostics(state)
        up = self._backward_velocity(state.flow.u)
        B = self.update_B_from_A(state.A)
        bp = self._total_B_physical(B, padded=False)
        volume = (
            (self.domain[0][1] - self.domain[0][0])
            * (self.domain[1][1] - self.domain[1][0])
            * (self.domain[2][1] - self.domain[2][0])
        )
        reynolds = integrate(jnp.real(up[0] * jnp.conj(up[1])), self.TD) / volume
        maxwell = -integrate(jnp.real(bp[0] * jnp.conj(bp[1])), self.TD) / volume
        transport = reynolds + maxwell
        vA2 = self.background_b[2] ** 2
        # FJ-04: ZNF-safe. Always emit the shear-scale-normalized alpha; emit the
        # net-flux alpha only when an imposed vertical field exists (never NaN at B0=0).
        sh2 = (self.shear_rate * self.x_half_width) ** 2
        emag_total = jnp.asarray(0.0, dtype=up[0].real.dtype)
        for bi, space in zip(bp, self.b_coeff_spaces, strict=True):
            emag_total = emag_total + jnp.real(integrate(jnp.conj(bi) * bi, space))
        b_fluct = self._backward_B(B, padded=False)
        channel_kz1_velocity_rms = self._kz1_rms(up)
        channel_kz1_magnetic_rms = self._kz1_rms(b_fluct)
        diss_kinetic, diss_magnetic = self._dissipation_parts(state, B)
        bmax = jnp.max(jnp.asarray([jnp.max(jnp.abs(bi)) for bi in b_fluct]))
        bmax_total = jnp.max(jnp.asarray([jnp.max(jnp.abs(bi)) for bi in bp]))
        # FJ-04: mean magnetic flux + mean/fluctuating energy split of the TOTAL
        # field (imposed background included), so a net-flux run reports the
        # physical mean field (mean_bz == B0, matching shearpy) and a ZNF run
        # detects mean-flux contamination. The volume mean and the deviation from
        # it are orthogonal, so mag_energy_mean + mag_energy_fluct == Emag_total
        # holds exactly.
        # Convention: this family reports all E* diagnostics as the plain volume
        # integral of the squared field (2x the physical energy), matching its
        # shenfun references (couette/pcf_mhd_divfree.py, pcf_mhd_mri_shearpy.py)
        # and the inherited Emag/Epert. The primitive family reports the physical
        # 0.5*integral per ITS reference; the oracle stamps `energy_convention`
        # (and `box_volume`) so cross-code comparisons can convert.
        b_spaces = self.b_coeff_spaces
        mean_b = tuple(
            integrate(jnp.real(bti), sp) / volume
            for bti, sp in zip(bp, b_spaces, strict=True)
        )
        emag_fluct = sum(
            jnp.real(integrate(jnp.conj(bi) * bi, sp))
            for bi, sp in zip(b_fluct, b_spaces, strict=True)
        )
        mag_energy_mean = volume * sum(mb * mb for mb in mean_b)
        out = {
            **diag,
            "Emag_total": emag_total,
            "bmax": bmax,
            "bmax_total": bmax_total,
            "reynolds_xy": reynolds,
            "maxwell_xy": maxwell,
            "reynolds_stress": reynolds,
            "maxwell_stress": maxwell,
            "transport_xy": transport,
            "total_stress": transport,
            "alpha_Sh": transport / sh2 if sh2 > 0.0 else jnp.asarray(jnp.nan),
            "mean_bx": mean_b[0],
            "mean_by": mean_b[1],
            "mean_bz": mean_b[2],
            "mag_energy_fluct_total": emag_fluct,
            "mag_energy_mean": mag_energy_mean,
            "mag_energy_fluct": emag_total - mag_energy_mean,
            "dissipation_kinetic": diss_kinetic,
            "dissipation_magnetic": diss_magnetic,
            "q_shear": jnp.asarray(self.q_shear),
            "kappa2": jnp.asarray(self.kappa2),
            "channel_kz1_velocity_rms": channel_kz1_velocity_rms,
            "channel_kz1_magnetic_rms": channel_kz1_magnetic_rms,
            "channel_kz1_total_rms": jnp.sqrt(
                channel_kz1_velocity_rms**2 + channel_kz1_magnetic_rms**2
            ),
        }
        if vA2 > 0.0:
            alpha = transport / vA2
            out["alpha"] = alpha
            out["alpha_B0"] = alpha
        return out


class PlaneCouetteMRIShearpyInsulatingJax(PlaneCouetteMRIShearpyJax):
    """Rotating-shear PCF MHD with exact insulating (vacuum-matched) walls.

    The magnetic representation stays ``B = B0 + curl(A)`` (solenoidal to
    roundoff by construction), but the wall condition matches the fluctuation
    field to a decaying exterior vacuum potential per Fourier mode
    ``(ky, kz)``, ``k = sqrt(ky^2 + kz^2)``:

      ``b_x' = -+ k b_x``,  ``b_y = -+ (i ky / k) b_x``,
      ``b_z = -+ (i kz / k) b_x``   at ``x = +-1``.

    In vector-potential variables (gauge ``A_x = 0`` at the walls, evolution
    gauge ``dA/dt = u x B + eta lap(A)``) the two tangential matching rows
    decouple per mode under the rotation ``P = ky A_y + kz A_z``,
    ``Q = kz A_y - ky A_z``:

      ``P' = 0``  (Neumann)  and  ``Q' = -+ k Q``  (Robin)   at ``x = +-1``,

    and ``b_x = -i Q`` then satisfies the Robin row identically while the
    tangential components match the vacuum solution.  The ``k = 0`` mean mode
    reduces to Neumann rows for both potentials, i.e. the mean tangential
    fluctuation field vanishes at the walls (zero exterior field at infinity).

    Because the Robin coefficient is the physical ``k`` of each mode, the
    tangential potentials cannot live in a fixed composite basis; ``A_y`` and
    ``A_z`` are evolved as orthogonal-basis (``TC``) coefficients and each
    IMEX stage solves a per-mode bordered (tau) system: the last two Galerkin
    rows of the radial operator are replaced by the boundary rows.  ``A_x``
    keeps the Dirichlet (TD) treatment as the wall gauge choice.
    """

    @property
    def a_coeff_spaces(self):
        return (self.TD, self.TC, self.TC)

    @property
    def b_coeff_spaces(self):
        return (self.TC, self.TC, self.TC)

    @property
    def j_coeff_spaces(self):
        return (self.TC, self.TC, self.TC)

    def _build_A_operators(self) -> None:
        super()._build_A_operators()  # TD operators, reused for the A_x gauge
        self.TCp = self.TC.get_dealiased(self.padding_factor)

        # 1D radial mass / second-derivative matrices on the orthogonal basis,
        # assembled with jaxfun `inner` so quadrature scaling stays consistent.
        h1 = TestFunction(self.C0, name="hA1")
        a1 = TrialFunction(self.C0, name="aA1")
        (x1,) = self.C0.system.base_scalars()
        self._A_M1 = jnp.asarray(inner(h1 * a1, kind=InnerKind.BILINEAR).todense())
        self._A_D2 = jnp.asarray(
            inner(h1 * sp.diff(a1, x1, 2), kind=InnerKind.BILINEAR).todense()
        )

        # Physical per-mode wavenumbers (ky, kz) and the tangential rotation.
        ky2d, kz2d = jnp.broadcast_arrays(
            jnp.real(self.K[1][0]), jnp.real(self.K[2][0])
        )
        kmag = jnp.sqrt(ky2d**2 + kz2d**2)
        knz = jnp.where(kmag > 0.0, kmag, 1.0)
        self._A_kmag = kmag
        self._A_cy = jnp.where(kmag > 0.0, ky2d / knz, 1.0)
        self._A_cz = jnp.where(kmag > 0.0, kz2d / knz, 0.0)

        # Wall evaluation rows of the orthogonal basis (value and first
        # derivative), lower wall first.
        bounds = jnp.asarray(
            [self.domain[0][0], self.domain[0][1]], dtype=self._A_M1.dtype
        )
        ref_bounds = self.C0.map_reference_domain(bounds)
        self._A_d0 = self.C0.evaluate_basis_derivative(ref_bounds, 0)
        self._A_d1 = self.C0.evaluate_basis_derivative(ref_bounds, 1) * float(
            self.C0.domain_factor
        )

        n = int(self._A_M1.shape[0])
        n_modes = int(kmag.size)
        k2_flat = (kmag.reshape(n_modes) ** 2)[:, None, None]
        base = self._A_M1 - (self.dt * self._gamma) * self.eta * (
            self._A_D2 - k2_flat * self._A_M1
        )
        base = jnp.broadcast_to(base, (n_modes, n, n)).astype(complex)

        # P system: Neumann rows at both walls (mode independent).
        sp_rows = base.at[:, n - 2, :].set(self._A_d1[0][None, :])
        sp_rows = sp_rows.at[:, n - 1, :].set(self._A_d1[1][None, :])
        # Q system: Robin rows Q' - k Q = 0 (lower), Q' + k Q = 0 (upper).
        k_flat = kmag.reshape(n_modes)[:, None]
        sq_rows = base.at[:, n - 2, :].set(
            self._A_d1[0][None, :] - k_flat * self._A_d0[0][None, :]
        )
        sq_rows = sq_rows.at[:, n - 1, :].set(
            self._A_d1[1][None, :] + k_flat * self._A_d0[1][None, :]
        )
        self._A_P_lu = jax.vmap(jsp_linalg.lu_factor)(sp_rows)
        self._A_Q_lu = jax.vmap(jsp_linalg.lu_factor)(sq_rows)

    def _A_apply_radial(self, matrix, coeff):
        return jnp.einsum("ij,jkl->ikl", matrix, coeff)

    def _A_mass_rhs(self, A):
        return (
            self.MA @ A[0],
            self._A_apply_radial(self._A_M1, A[1]),
            self._A_apply_radial(self._A_M1, A[2]),
        )

    def _A_eta_lap(self, A):
        k2 = (self._A_kmag**2)[None, :, :]

        def lap(coeff):
            return self.eta * (
                self._A_apply_radial(self._A_D2, coeff)
                - k2 * self._A_apply_radial(self._A_M1, coeff)
            )

        return (self.LA @ A[0], lap(A[1]), lap(A[2]))

    def _A_forward_emf(self, emf):
        return (
            self.TD.mask_nyquist(self.TDp.forward(emf[0])),
            self.TC.mask_nyquist(self.TCp.forward(emf[1])),
            self.TC.mask_nyquist(self.TCp.forward(emf[2])),
        )

    def _A_bordered_solve(self, lu, rhs):
        """Solve the per-mode bordered radial systems for one potential."""
        n, ny, nz = rhs.shape
        modes = jnp.moveaxis(rhs.reshape(n, ny * nz), 0, 1)
        modes = modes.at[:, n - 2 :].set(0.0)  # homogeneous boundary rows
        lu_mat, piv = lu
        sol = jax.vmap(
            lambda lu_i, piv_i, b_i: jsp_linalg.lu_solve((lu_i, piv_i), b_i)
        )(lu_mat, piv, modes)
        return jnp.moveaxis(sol, 0, 1).reshape(n, ny, nz)

    def _A_solve(self, rhs):
        ax = self.TD.mask_nyquist(self._solve_prefactor(self.SA_factor, rhs[0]))
        cy = self._A_cy[None, :, :]
        cz = self._A_cz[None, :, :]
        rhs_p = cy * rhs[1] + cz * rhs[2]
        rhs_q = cz * rhs[1] - cy * rhs[2]
        p_new = self._A_bordered_solve(self._A_P_lu, rhs_p)
        q_new = self._A_bordered_solve(self._A_Q_lu, rhs_q)
        ay = self.TC.mask_nyquist(cy * p_new + cz * q_new)
        az = self.TC.mask_nyquist(cz * p_new - cy * q_new)
        return (ax, ay, az)

    def seed_linear_eigenmode(
        self,
        ky_mode: int = 0,
        kz_mode: int = 1,
        amp: float = 1.0e-6,
        which: int = 0,
        nx_linear: int = 96,
    ) -> tuple[MHDState, complex]:
        """Seed the real part of an insulating linear MHD eigenmode.

        The primitive-variable insulating eigenmode comes from
        :class:`examples.pcf_linear_jax.PlaneCouetteLinear` on a Chebyshev-
        Lobatto grid; the corresponding vector potential is reconstructed in
        closed form (gauge ``A_x = 0``): ``A_y = int b_z dx``,
        ``A_z = -int b_y dx`` with the integration constants fixed by matching
        ``b_x`` at the lower wall and the pure-gauge tangential constant
        ``P(-1) = ky A_y + kz A_z = 0``.  The reconstruction satisfies the
        solver's vacuum-matching rows identically because the eigenmode does.
        """
        import numpy as np

        from examples.pcf_linear_jax import PlaneCouetteLinear

        Ly = self.domain[1][1] - self.domain[1][0]
        Lz = self.domain[2][1] - self.domain[2][0]
        ky = 2.0 * np.pi * int(ky_mode) / Ly
        kz = 2.0 * np.pi * int(kz_mode) / Lz
        k = float(np.hypot(ky, kz))
        if k <= 0.0:
            raise ValueError("eigenmode seeding requires ky or kz nonzero")
        op = PlaneCouetteLinear(
            nx=int(nx_linear),
            nu=self.nu,
            eta=self.eta,
            Uprime=-self.shear_rate,
            Uoffset=0.0,
            omega=self.omega,
            by=self.background_b[1],
            bz=self.background_b[2],
            mhd=True,
            magnetic_bc="insulating",
        )
        w, V = op.eigs(ky, kz, n_return=which + 1)
        vec = V[:, which]
        n = op.nx
        blocks = op._blocks()

        def cheb_fit(values):
            fit_re = np.polynomial.chebyshev.Chebyshev.fit(
                op.x, values.real, n - 1, domain=[-1.0, 1.0]
            )
            fit_im = np.polynomial.chebyshev.Chebyshev.fit(
                op.x, values.imag, n - 1, domain=[-1.0, 1.0]
            )
            return fit_re, fit_im

        def block(name):
            i = blocks[name]
            return np.asarray(vec[i * n : (i + 1) * n])

        xq = np.asarray(jnp.squeeze(self.X[0]))

        def eval_at(fits, x):
            return fits[0](x) + 1j * fits[1](x)

        def integ(fits):
            return (fits[0].integ(lbnd=-1.0), fits[1].integ(lbnd=-1.0))

        fits_by = cheb_fit(block("by"))
        fits_bz = cheb_fit(block("bz"))
        bx_lower = complex(block("bx")[-1])  # Lobatto x[-1] = -1
        ay0 = kz * (1j * bx_lower) / k**2
        az0 = -ky * (1j * bx_lower) / k**2
        a_y = ay0 + eval_at(integ(fits_bz), xq)
        a_z = az0 - eval_at(integ(fits_by), xq)
        a_x = np.zeros_like(a_y)

        u_prof = [eval_at(cheb_fit(block(name)), xq) for name in ("ux", "uy", "uz")]

        phase = jnp.exp(1j * (ky * self.X[1] + kz * self.X[2]))

        def field3d(profile):
            return float(amp) * jnp.real(jnp.asarray(profile)[:, None, None] * phase)

        flow = self.state_from_physical(tuple(field3d(p) for p in u_prof))
        A = self._A_state_from_physical(tuple(field3d(p) for p in (a_x, a_y, a_z)))
        return MHDState(flow=flow, A=A), complex(w[which])

    def insulating_bc_residual(self, state: MHDState) -> jnp.ndarray:
        """Max wall residual of the vacuum-matching rows on b = curl(A).

        Evaluates, per mode and wall, ``b_x' -+ (-+ k) b_x``,
        ``b_y +- (i ky / k) b_x`` and ``b_z +- (i kz / k) b_x`` from the
        boundary rows; the tau rows enforce these exactly, so the residual is
        a roundoff-level witness of the imposed boundary condition.
        """
        B = self.update_B_from_A(state.A)
        d0, d1 = self._A_d0, self._A_d1

        def walls(coeff, rows):
            return jnp.einsum("wi,ikl->wkl", rows, coeff)

        bx0, bx1 = walls(B[0], d0), walls(B[0], d1)
        by0 = walls(B[1], d0)
        bz0 = walls(B[2], d0)
        kmag = self._A_kmag[None, :, :]
        knz = jnp.where(kmag > 0.0, kmag, 1.0)
        ky2d, kz2d = jnp.broadcast_arrays(
            jnp.real(self.K[1][0]), jnp.real(self.K[2][0])
        )
        sign = jnp.asarray([-1.0, 1.0])[:, None, None]  # outward normal (-, +)
        res_x = bx1 + sign * kmag * bx0
        res_y = by0 + sign * (1j * ky2d[None] / knz) * bx0
        res_z = bz0 + sign * (1j * kz2d[None] / knz) * bx0
        return jnp.max(
            jnp.asarray([jnp.max(jnp.abs(r)) for r in (res_x, res_y, res_z)])
        )

    def diagnostics(self, state: MHDState) -> dict[str, jnp.ndarray]:
        diag = super().diagnostics(state)
        diag["insulating_bc_residual"] = self.insulating_bc_residual(state)
        return diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs=3, default=(17, 16, 16))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--shear", type=float, default=1.0)
    parser.add_argument("--Bz", type=float, default=0.025)
    parser.add_argument(
        "--magnetic-bc", choices=("conducting", "insulating"), default="conducting"
    )
    args = parser.parse_args()

    solver_cls = (
        PlaneCouetteMRIShearpyInsulatingJax
        if args.magnetic_bc == "insulating"
        else PlaneCouetteMRIShearpyJax
    )
    solver = solver_cls(
        N=tuple(args.N),
        dt=args.dt,
        family=args.family,
        omega=args.omega,
        shear_rate=args.shear,
        background_b=(0.0, 0.0, args.Bz),
    )
    state = solver.solve(solver.initial_state(), args.steps)
    diag = solver.diagnostics(state)
    print(
        " ".join(
            f"{key}={float(diag[key]):.6e}"
            for key in ("divL2", "divB_L2", "alpha", "reynolds_xy", "maxwell_xy")
        )
    )


if __name__ == "__main__":
    main()
