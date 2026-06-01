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
        background_b: tuple[float, float, float] = (0.0, 0.0, 0.1),
        dt: float = 0.01,
        family: str = "L",
        perturbation_amplitude: float = 0.05,
        magnetic_amplitude: float = 0.05,
    ) -> None:
        self.omega = float(omega)
        self.shear_rate = float(shear_rate)
        self.background_b = tuple(float(v) for v in background_b)
        super().__init__(
            N=N,
            domain=domain,
            Re=Re,
            Rm=Rm,
            U_wall=1.0,
            dt=dt,
            family=family,
            perturbation_amplitude=perturbation_amplitude,
            magnetic_amplitude=magnetic_amplitude,
        )
        self.Ub = -self.shear_rate * self.X[0]
        self.Ubp = -self.shear_rate * self.Xp[0]
        self.dUb_dx = -self.shear_rate
        self.q_shear = self.shear_rate / self.omega if self.omega != 0 else math.inf
        self.kappa2 = 2.0 * self.omega * (2.0 * self.omega - self.shear_rate)

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
        return {
            **diag,
            "reynolds_xy": reynolds,
            "maxwell_xy": maxwell,
            "alpha": reynolds + maxwell,
            "q_shear": jnp.asarray(self.q_shear),
            "kappa2": jnp.asarray(self.kappa2),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs=3, default=(17, 16, 16))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--shear", type=float, default=1.0)
    parser.add_argument("--Bz", type=float, default=0.1)
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
