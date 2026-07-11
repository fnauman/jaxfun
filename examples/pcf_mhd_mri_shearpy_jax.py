"""Shearpy-style rotating Plane Couette MHD analogue in jaxfun.

This extends examples.pcf_mhd_jax with the linear shearing-box source terms and
imposed net magnetic field used by couette/pcf_mhd_mri_shearpy.py.
"""

from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import sympy as sp

try:
    from examples.pcf_mhd_jax import MHDState, PlaneCouetteMHDJax
except ModuleNotFoundError:  # direct script execution from examples/
    from pcf_mhd_jax import MHDState, PlaneCouetteMHDJax

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
    ) -> None:
        self.omega = float(omega)
        self.shear_rate = float(shear_rate)
        self.background_b = tuple(float(v) for v in background_b)
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
            u2 = u2 + 0.1 * amp * wall * jnp.sin(2.0 * ky * y) * jnp.cos(
                2.0 * kz * z
            )
        flow = self.state_from_physical((u0, u1, u2))

        mag_amp = self.magnetic_amplitude
        ax = jnp.zeros(self.TD.num_quad_points)
        ay = jnp.zeros_like(ax)
        az = jnp.zeros_like(ax)
        if mag_amp != 0.0:
            ax = mag_amp * wall * (1.0 / kz) * jnp.sin(ky * y) * jnp.sin(kz * z)
        A = (
            self.TD.mask_nyquist(self.TD.forward(ax)),
            self.TD.mask_nyquist(self.TD.forward(ay)),
            self.TD.mask_nyquist(self.TD.forward(az)),
        )
        return MHDState(flow=flow, A=A)

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
        HA = tuple(self.TD.mask_nyquist(self.TDp.forward(ei)) for ei in emf)
        return H, HA

    def diagnostics(self, state: MHDState) -> dict[str, jnp.ndarray]:
        diag = super().diagnostics(state)
        up = self._backward_velocity(state.flow.u)
        bp = self._total_B_physical(self.update_B_from_A(state.A), padded=False)
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
        for bi, space in zip(bp, (self.TD, self.TC, self.TC), strict=True):
            emag_total = emag_total + jnp.real(integrate(jnp.conj(bi) * bi, space))
        b_fluct = self._backward_B(self.update_B_from_A(state.A), padded=False)
        bmax = jnp.max(jnp.asarray([jnp.max(jnp.abs(bi)) for bi in b_fluct]))
        bmax_total = jnp.max(jnp.asarray([jnp.max(jnp.abs(bi)) for bi in bp]))
        # FJ-04: mean magnetic flux + mean/fluctuating energy split of the evolved
        # (fluctuation) field, so a ZNF run through the curl workhorse can detect
        # mean-flux contamination and the FJ-04 tolerance can be generated here.
        # Convention: this family reports all E* diagnostics as the plain volume
        # integral of the squared field (2x the physical energy), matching its
        # shenfun references (couette/pcf_mhd_divfree.py, pcf_mhd_mri_shearpy.py)
        # and the inherited Emag/Epert, so Emag == mag_energy_mean + mag_energy_fluct
        # holds exactly. The primitive family reports the physical 0.5*integral per
        # ITS reference; the oracle stamps `energy_convention` to disambiguate.
        b_spaces = (self.TD, self.TC, self.TC)
        mean_b = tuple(
            integrate(jnp.real(bi), sp) / volume
            for bi, sp in zip(b_fluct, b_spaces, strict=True)
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
            "mag_energy_fluct": emag_fluct - mag_energy_mean,
            "q_shear": jnp.asarray(self.q_shear),
            "kappa2": jnp.asarray(self.kappa2),
        }
        if vA2 > 0.0:
            alpha = transport / vA2
            out["alpha"] = alpha
            out["alpha_B0"] = alpha
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs=3, default=(17, 16, 16))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--shear", type=float, default=1.0)
    parser.add_argument("--Bz", type=float, default=0.025)
    args = parser.parse_args()

    solver = PlaneCouetteMRIShearpyJax(
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
