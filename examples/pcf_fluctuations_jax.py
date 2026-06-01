"""Plane Couette fluctuations using the jaxfun KMM solver.

This ports the solver-facing parts of couette/pcf_fluctuations_corrected.py:
fluctuations are evolved about the laminar base flow ``U_b(x)=U_wall*x`` and
component ordering is (wall-normal, streamwise, spanwise).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import sympy as sp
from jax import Array

try:
    from examples.channelflow_kmm import KMM, KMMState, Velocity
except ModuleNotFoundError:  # direct script execution from examples/
    from channelflow_kmm import KMM, KMMState, Velocity

from jaxfun.galerkin.inner import integrate


class PlaneCouetteFluctuationJax(KMM):
    """Plane Couette fluctuation solver following the shenfun reference.

    Reference: couette/pcf_fluctuations_corrected.py:58-212.
    """

    def __init__(
        self,
        N: tuple[int, int, int] = (17, 16, 16),
        domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-1.0, 1.0),
            (0.0, 4.0 * float(sp.pi)),
            (0.0, 2.0 * float(sp.pi)),
        ),
        Re: float = 600.0,
        U_wall: float = 1.0,
        dt: float = 0.01,
        family: str = "L",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        perturbation_amplitude: float = 0.05,
    ) -> None:
        self.Re = float(Re)
        self.U_wall = float(U_wall)
        self.perturbation_amplitude = float(perturbation_amplitude)
        nu = self.U_wall / self.Re
        super().__init__(
            N=N,
            domain=domain,
            nu=nu,
            dt=dt,
            family=family,
            padding_factor=padding_factor,
            dpdy=0.0,
        )
        self.Ub = self.U_wall * self.X[0]
        self.Ubp = self.U_wall * self.Xp[0]

    def initial_state(self) -> KMMState:
        """Return the deterministic fluctuation initial condition.

        Reference: couette/pcf_fluctuations_corrected.py:134-153.  The random
        seed in the reference is unused; this reproduces the analytic field.
        """
        x, y, z = self.X
        amp = self.perturbation_amplitude
        wall = 1.0 - x**2
        Ly = self.domain[1][1] - self.domain[1][0]
        Lz = self.domain[2][1] - self.domain[2][0]
        u0 = (
            amp * wall * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.cos(2.0 * jnp.pi * z / Lz)
        )
        u1 = (
            amp * wall * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.sin(2.0 * jnp.pi * z / Lz)
        )
        u2 = (
            amp * wall * jnp.sin(4.0 * jnp.pi * y / Ly) * jnp.cos(4.0 * jnp.pi * z / Lz)
        )
        return self.state_from_physical((u0, u1, u2))

    def _add_base_convection(
        self, n: Velocity, up: Velocity, grads: dict[str, Array]
    ) -> Velocity:
        n0, n1, n2 = n
        n0 = n0 + self.Ubp * grads["dudy"]
        n1 = n1 + self.Ubp * grads["dvdy"] + up[0] * self.U_wall
        n2 = n2 + self.Ubp * grads["dwdy"]
        return n0, n1, n2

    def total_velocity_physical(self, state: KMMState) -> Velocity:
        up = self._backward_velocity(state.u)
        return up[0], up[1] + self.Ub, up[2]

    def diagnostics(self, state: KMMState) -> dict[str, Array]:
        """Return the core PCF diagnostics used by the reference script."""
        up = self._backward_velocity(state.u)
        ut = self.total_velocity_physical(state)
        spaces = (self.TB, self.TD, self.TD)
        epert = jnp.asarray(0.0, dtype=up[0].real.dtype)
        etot = jnp.asarray(0.0, dtype=up[0].real.dtype)
        for ui, uti, space in zip(up, ut, spaces, strict=True):
            epert = epert + jnp.real(integrate(jnp.conj(ui) * ui, space))
            etot = etot + jnp.real(integrate(jnp.conj(uti) * uti, space))
        dv_dx = self.TD.backward_primitive(state.u[1], (1, 0, 0))
        return {
            "Epert": epert,
            "Etot": etot,
            "divL2": self.divergence_l2(state),
            "u_top": jnp.real(jnp.mean(ut[1][-1, :, :])),
            "u_bot": jnp.real(jnp.mean(ut[1][0, :, :])),
            "mean_shear": jnp.real(jnp.mean(dv_dx + self.U_wall)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs=3, default=(17, 16, 16))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Re", type=float, default=600.0)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    args = parser.parse_args()

    solver = PlaneCouetteFluctuationJax(
        N=tuple(args.N), dt=args.dt, Re=args.Re, family=args.family
    )
    state = solver.solve(solver.initial_state(), args.steps)
    diag = solver.diagnostics(state)
    print(" ".join(f"{key}={float(value):.6e}" for key, value in diag.items()))


if __name__ == "__main__":
    main()
