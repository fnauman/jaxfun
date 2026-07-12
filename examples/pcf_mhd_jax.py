"""Plane Couette MHD with divergence-free magnetic field construction.

This is the jaxfun counterpart of couette/pcf_mhd_divfree.py.  It evolves the
velocity fluctuations with the KMM solver and evolves a magnetic vector
potential A in TD^3, then recomputes B=curl(A) and J=curl(B) when needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

try:
    from examples.channelflow_kmm import KMMState, Velocity
    from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
except ModuleNotFoundError:  # direct script execution from examples/
    from channelflow_kmm import KMMState, Velocity
    from pcf_fluctuations_jax import PlaneCouetteFluctuationJax

from jaxfun.galerkin import TestFunction, TrialFunction, inner
from jaxfun.galerkin.inner import integrate
from jaxfun.integrators.cnab2 import scan_steps
from jaxfun.integrators.nonlinear import physical_cross


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MHDState:
    """Coefficient-space state for PCF MHD."""

    flow: KMMState
    A: Velocity

    def tree_flatten(self):
        return (self.flow, self.A), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        flow, A = children
        return cls(flow=flow, A=A)


class PlaneCouetteMHDJax(PlaneCouetteFluctuationJax):
    """Plane Couette MHD using A as the evolved magnetic unknown.

    References: couette/pcf_mhd_divfree.py:68-360 for spaces, seed fields,
    Lorentz forcing and vector-potential EMF forcing.
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
        U_wall: float = 1.0,
        dt: float = 0.01,
        family: str = "L",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        perturbation_amplitude: float = 0.1,
        magnetic_amplitude: float = 0.05,
    ) -> None:
        self.Rm = float(Re if Rm is None else Rm)
        self.eta = float(U_wall) / self.Rm
        self.magnetic_amplitude = float(magnetic_amplitude)
        super().__init__(
            N=N,
            domain=domain,
            Re=Re,
            U_wall=U_wall,
            dt=dt,
            family=family,
            padding_factor=padding_factor,
            perturbation_amplitude=perturbation_amplitude,
        )
        self._build_A_operators()

    def _build_A_operators(self) -> None:
        h = TestFunction(self.TD, name="hA")
        a = TrialFunction(self.TD, name="aA")
        coords = self.TD.system.base_scalars()
        lap_a = self._lap(a, coords)
        self.MA = inner(h * a, sparse=True)
        self.LA = inner(h * (self.eta * lap_a), sparse=True)
        self.SA = self.MA - (self.dt * self._gamma) * self.LA
        self.SA_factor = self.SA.lu_factor()

    def initial_state(self) -> MHDState:
        flow = super().initial_state()
        x, y, z = self.X
        amp = self.magnetic_amplitude
        ax = jnp.zeros(self.TD.num_quad_points)
        ay = jnp.zeros_like(ax)
        az = jnp.zeros_like(ax)
        if amp != 0.0:
            wall = 1.0 - x**2
            Ly = self.domain[1][1] - self.domain[1][0]
            Lz = self.domain[2][1] - self.domain[2][0]
            ky = 2.0 * jnp.pi / Ly
            kz = 2.0 * jnp.pi / Lz
            ax = amp * wall * (1.0 / kz) * jnp.sin(ky * y) * jnp.sin(kz * z)
        A = (
            self.TD.mask_nyquist(self.TD.forward(ax)),
            self.TD.mask_nyquist(self.TD.forward(ay)),
            self.TD.mask_nyquist(self.TD.forward(az)),
        )
        return MHDState(flow=flow, A=A)

    def update_B_from_A(self, A: Velocity) -> Velocity:
        """Compute B=curl(A) in [TD, TC, TC] coefficient spaces."""
        counts = self.TD.num_quad_points
        bx_phys = self.TD.backward_primitive(
            A[2], (0, 1, 0), N=counts
        ) - self.TD.backward_primitive(A[1], (0, 0, 1), N=counts)
        by_phys = self.TD.backward_primitive(
            A[0], (0, 0, 1), N=counts
        ) - self.TD.backward_primitive(A[2], (1, 0, 0), N=counts)
        bz_phys = self.TD.backward_primitive(
            A[1], (1, 0, 0), N=counts
        ) - self.TD.backward_primitive(A[0], (0, 1, 0), N=counts)
        return (
            self.TD.mask_nyquist(self.TD.forward(bx_phys)),
            self.TC.mask_nyquist(self.TC.forward(by_phys)),
            self.TC.mask_nyquist(self.TC.forward(bz_phys)),
        )

    def update_J_from_B(self, B: Velocity) -> Velocity:
        """Compute J=curl(B) in [TC, TD, TD] coefficient spaces."""
        counts = self.TD.num_quad_points
        jx_phys = self.TC.backward_primitive(
            B[2], (0, 1, 0), N=counts
        ) - self.TC.backward_primitive(B[1], (0, 0, 1), N=counts)
        jy_phys = self.TD.backward_primitive(
            B[0], (0, 0, 1), N=counts
        ) - self.TC.backward_primitive(B[2], (1, 0, 0), N=counts)
        jz_phys = self.TC.backward_primitive(
            B[1], (1, 0, 0), N=counts
        ) - self.TD.backward_primitive(B[0], (0, 1, 0), N=counts)
        return (
            self.TC.mask_nyquist(self.TC.forward(jx_phys)),
            self.TD.mask_nyquist(self.TD.forward(jy_phys)),
            self.TD.mask_nyquist(self.TD.forward(jz_phys)),
        )

    def _backward_B(self, B: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        return (
            self.TD.backward(B[0], N=counts),
            self.TC.backward(B[1], N=counts),
            self.TC.backward(B[2], N=counts),
        )

    def _backward_J(self, J: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        return (
            self.TC.backward(J[0], N=counts),
            self.TD.backward(J[1], N=counts),
            self.TD.backward(J[2], N=counts),
        )

    def _mhd_convection(self, state: MHDState) -> tuple[Velocity, Velocity]:
        flow = state.flow
        up = self._backward_velocity(flow.u, padded=True)
        grads = self._velocity_gradients(flow.u)
        n = (
            up[0] * grads["dudx"] + up[1] * grads["dudy"] + up[2] * grads["dudz"],
            up[0] * grads["dvdx"] + up[1] * grads["dvdy"] + up[2] * grads["dvdz"],
            up[0] * grads["dwdx"] + up[1] * grads["dwdy"] + up[2] * grads["dwdz"],
        )
        n = self._add_base_convection(n, up, grads)

        B = self.update_B_from_A(state.A)
        J = self.update_J_from_B(B)
        bp = self._backward_B(B, padded=True)
        jp = self._backward_J(J, padded=True)
        lorentz = physical_cross(jp, bp)
        n = tuple(ni - li for ni, li in zip(n, lorentz, strict=True))
        H = tuple(self.TD.mask_nyquist(self.TDp.forward(ni)) for ni in n)

        utotal = (up[0], up[1] + self.Ubp, up[2])
        emf = physical_cross(utotal, bp)
        HA = tuple(self.TD.mask_nyquist(self.TDp.forward(ei)) for ei in emf)
        return H, HA

    def step(self, state: MHDState) -> MHDState:
        a, b, _ = self.timestepper.stages()
        steps = self.timestepper.steps()
        flow0 = state.flow
        A0 = state.A
        u0_rhs = self.Mu @ flow0.u[0]
        g0_rhs = self.Mg @ flow0.g
        v00_rhs0 = self.M00 @ flow0.u[1][:, 0, 0]
        w00_rhs0 = self.M00 @ flow0.u[2][:, 0, 0]
        A0_rhs = tuple(self.MA @ Ai for Ai in A0)

        flow_stage = flow0
        A_stage = A0
        nonlinear_u: list[Array] = []
        nonlinear_g: list[Array] = []
        nonlinear_v: list[Array] = []
        nonlinear_w: list[Array] = []
        nonlinear_A: list[Velocity] = []
        linear_u: list[Array] = []
        linear_g: list[Array] = []
        linear_v: list[Array] = []
        linear_w: list[Array] = []
        linear_A: list[Velocity] = []

        for rk in range(steps):
            H, HA = self._mhd_convection(MHDState(flow=flow_stage, A=A_stage))
            Nu, Ng, Nv, Nw = self._nonlinear_rhs(H)
            nonlinear_u.append(Nu)
            nonlinear_g.append(Ng)
            nonlinear_v.append(Nv)
            nonlinear_w.append(Nw)
            nonlinear_A.append(tuple(self.MA @ hi for hi in HA))

            rhs_u = u0_rhs
            rhs_g = g0_rhs
            rhs_v = v00_rhs0
            rhs_w = w00_rhs0
            rhs_A = list(A0_rhs)
            for j in range(rk + 1):
                rhs_u = rhs_u + self.dt * b[rk + 1, j] * nonlinear_u[j]
                rhs_g = rhs_g + self.dt * b[rk + 1, j] * nonlinear_g[j]
                rhs_v = rhs_v + self.dt * b[rk + 1, j] * nonlinear_v[j]
                rhs_w = rhs_w + self.dt * b[rk + 1, j] * nonlinear_w[j]
                rhs_A = [
                    rhs_A[i] + self.dt * b[rk + 1, j] * nonlinear_A[j][i]
                    for i in range(3)
                ]

            if rk > 0:
                linear_u.append(self.Lu @ flow_stage.u[0])
                linear_g.append(self.Lg @ flow_stage.g)
                linear_v.append(self.L00 @ flow_stage.u[1][:, 0, 0])
                linear_w.append(self.L00 @ flow_stage.u[2][:, 0, 0])
                linear_A.append(tuple(self.LA @ Ai for Ai in A_stage))
                for j in range(rk):
                    rhs_u = rhs_u + self.dt * a[rk + 1, j + 1] * linear_u[j]
                    rhs_g = rhs_g + self.dt * a[rk + 1, j + 1] * linear_g[j]
                    rhs_v = rhs_v + self.dt * a[rk + 1, j + 1] * linear_v[j]
                    rhs_w = rhs_w + self.dt * a[rk + 1, j + 1] * linear_w[j]
                    rhs_A = [
                        rhs_A[i] + self.dt * a[rk + 1, j + 1] * linear_A[j][i]
                        for i in range(3)
                    ]

            u0_new = self._solve_prefactor(self.Su_factor, self.TB.mask_nyquist(rhs_u))
            g_new = self._solve_prefactor(self.Sg_factor, self.TD.mask_nyquist(rhs_g))
            v00_new = self._solve_prefactor(self.S00_factor, rhs_v)
            w00_new = self._solve_prefactor(self.S00_factor, rhs_w)
            u_new = self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new)
            A_stage = tuple(
                self.TD.mask_nyquist(self._solve_prefactor(self.SA_factor, rhs_A[i]))
                for i in range(3)
            )
            flow_stage = KMMState(u=u_new, g=g_new)

        return MHDState(flow=flow_stage, A=A_stage)

    def solve(self, state: MHDState, steps: int) -> MHDState:
        step = self.step if jax.device_count() > 1 else jax.checkpoint(self.step)
        return scan_steps(step, state, int(steps))

    def magnetic_divergence_l2(self, state: MHDState) -> Array:
        B = self.update_B_from_A(state.A)
        divb = (
            self.TD.backward_primitive(B[0], (1, 0, 0))
            + self.TC.backward_primitive(B[1], (0, 1, 0))
            + self.TC.backward_primitive(B[2], (0, 0, 1))
        )
        return jnp.sqrt(jnp.real(integrate(jnp.conj(divb) * divb, self.TC)))

    def fields_physical(self, state: MHDState) -> tuple[Array, ...]:
        """Physical (u_x, u_y, u_z, b_x, b_y, b_z) perturbation fields."""
        up = self._backward_velocity(state.flow.u)
        bp = self._backward_B(self.update_B_from_A(state.A))
        return (*up, *bp)

    def diagnostics(self, state: MHDState) -> dict[str, Array]:
        flow_diag = super().diagnostics(state.flow)
        B = self.update_B_from_A(state.A)
        bp = self._backward_B(B)
        magnetic_energy = jnp.asarray(0.0, dtype=bp[0].real.dtype)
        for bi, space in zip(bp, (self.TD, self.TC, self.TC), strict=True):
            magnetic_energy = magnetic_energy + jnp.real(
                integrate(jnp.conj(bi) * bi, space)
            )
        return {
            **flow_diag,
            "divB_L2": self.magnetic_divergence_l2(state),
            "Emag": magnetic_energy,
        }
