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
from jaxfun.integrators.cnab2 import variable_ab2_extrapolate
from jaxfun.integrators.nonlinear import physical_cross


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MHDState:
    """Coefficient-space state for PCF MHD with optional CNAB2 history."""

    flow: KMMState
    A: Velocity
    nonlinear_A_old: Velocity | None = None

    def tree_flatten(self):
        return (self.flow, self.A, self.nonlinear_A_old), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        flow, A, nonlinear_A_old = children
        return cls(flow=flow, A=A, nonlinear_A_old=nonlinear_A_old)


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
        family: str = "C",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        perturbation_amplitude: float = 0.1,
        magnetic_amplitude: float = 0.05,
        time_integrator: str | None = None,
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
            time_integrator=time_integrator,
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

    # Magnetic coefficient spaces.  The conducting family keeps A in TD^3 with
    # B=curl(A) in [TD, TC, TC] and J=curl(B) in [TC, TD, TD]; subclasses with
    # other wall conditions (e.g. insulating vacuum matching) override these
    # and the A-subsystem hooks below without touching the shared IMEX loop.
    @property
    def a_coeff_spaces(self):
        return (self.TD, self.TD, self.TD)

    @property
    def b_coeff_spaces(self):
        return (self.TD, self.TC, self.TC)

    @property
    def j_coeff_spaces(self):
        return (self.TC, self.TD, self.TD)

    def _A_mass_rhs(self, A: Velocity) -> Velocity:
        """Mass-matrix rows of the A subsystem (per component)."""
        return tuple(self.MA @ Ai for Ai in A)

    def _A_eta_lap(self, A: Velocity) -> Velocity:
        """eta*Laplacian Galerkin rows of the A subsystem (per component)."""
        return tuple(self.LA @ Ai for Ai in A)

    def _A_forward_emf(self, emf: Velocity) -> Velocity:
        """Forward-transform the physical EMF into per-component A forcings."""
        transformed = jax.vmap(
            lambda values: self.TD.mask_nyquist(self.TDp.forward(values))
        )(jnp.stack(emf))
        return transformed[0], transformed[1], transformed[2]

    def _A_solve(self, rhs: list, runtime_args=None) -> Velocity:
        """Solve the implicit A stage system from accumulated RHS rows."""
        runtime_args = self._A_runtime_args() if runtime_args is None else runtime_args
        return tuple(
            self.TD.mask_nyquist(
                self._solve_prefactor(self.SA_factor, rhs[i], runtime_args)
            )
            for i in range(3)
        )

    def _A_runtime_args(self):
        return self._factor_runtime_args(self.SA_factor)

    def _runtime_factor_args(self):
        return (*super()._runtime_factor_args(), self._A_runtime_args())

    def _A_state_from_physical(self, a_phys: Velocity) -> Velocity:
        """Forward-transform physical A samples into the family's A spaces."""
        return tuple(
            space.mask_nyquist(space.forward(ai))
            for space, ai in zip(self.a_coeff_spaces, a_phys, strict=True)
        )

    def _ensure_mhd_history(self, state: MHDState) -> MHDState:
        flow = self._ensure_flow_history(state.flow)
        if not self._cnab2:
            if flow is state.flow:
                return state
            return MHDState(flow=flow, A=state.A, nonlinear_A_old=state.nonlinear_A_old)
        nonlinear_A_old = state.nonlinear_A_old
        if nonlinear_A_old is None:
            nonlinear_A_old = tuple(jnp.zeros_like(Ai) for Ai in state.A)
        if flow is state.flow and nonlinear_A_old is state.nonlinear_A_old:
            return state
        return MHDState(flow=flow, A=state.A, nonlinear_A_old=nonlinear_A_old)

    def _new_mhd_state(self, flow: KMMState, A: Velocity) -> MHDState:
        return self._ensure_mhd_history(MHDState(flow=flow, A=A))

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
        return self._new_mhd_state(flow, self._A_state_from_physical((ax, ay, az)))

    def update_B_from_A(self, A: Velocity) -> Velocity:
        """Compute projected ``B=curl(A)`` without a physical round trip."""
        SA = self.a_coeff_spaces
        SB = self.b_coeff_spaces
        bx_orth = SA[2].derivative_orthogonal_coeffs(A[2], (0, 1, 0)) - SA[
            1
        ].derivative_orthogonal_coeffs(A[1], (0, 0, 1))
        by_orth = SA[0].derivative_orthogonal_coeffs(A[0], (0, 0, 1)) - SA[
            2
        ].derivative_orthogonal_coeffs(A[2], (1, 0, 0))
        bz_orth = SA[1].derivative_orthogonal_coeffs(A[1], (1, 0, 0)) - SA[
            0
        ].derivative_orthogonal_coeffs(A[0], (0, 1, 0))
        return (
            SB[0].mask_nyquist(SB[0].project_from_orthogonal(bx_orth)),
            SB[1].mask_nyquist(SB[1].project_from_orthogonal(by_orth)),
            SB[2].mask_nyquist(SB[2].project_from_orthogonal(bz_orth)),
        )

    def update_J_from_B(self, B: Velocity) -> Velocity:
        """Compute projected ``J=curl(B)`` without a physical round trip."""
        SB = self.b_coeff_spaces
        SJ = self.j_coeff_spaces
        jx_orth = SB[2].derivative_orthogonal_coeffs(B[2], (0, 1, 0)) - SB[
            1
        ].derivative_orthogonal_coeffs(B[1], (0, 0, 1))
        jy_orth = SB[0].derivative_orthogonal_coeffs(B[0], (0, 0, 1)) - SB[
            2
        ].derivative_orthogonal_coeffs(B[2], (1, 0, 0))
        jz_orth = SB[1].derivative_orthogonal_coeffs(B[1], (1, 0, 0)) - SB[
            0
        ].derivative_orthogonal_coeffs(B[0], (0, 1, 0))
        return (
            SJ[0].mask_nyquist(SJ[0].project_from_orthogonal(jx_orth)),
            SJ[1].mask_nyquist(SJ[1].project_from_orthogonal(jy_orth)),
            SJ[2].mask_nyquist(SJ[2].project_from_orthogonal(jz_orth)),
        )

    @staticmethod
    def _backward_vector(
        fields: Velocity, spaces: tuple, counts: tuple[int, ...] | None
    ) -> Velocity:
        if spaces[1] is spaces[2]:
            tangential = jax.vmap(
                lambda coefficients: spaces[1].backward(coefficients, N=counts)
            )(jnp.stack(fields[1:]))
            return (
                spaces[0].backward(fields[0], N=counts),
                tangential[0],
                tangential[1],
            )
        return tuple(
            space.backward(coefficients, N=counts)
            for space, coefficients in zip(spaces, fields, strict=True)
        )

    def _backward_B(self, B: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        return self._backward_vector(B, self.b_coeff_spaces, counts)

    def _backward_J(self, J: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        return self._backward_vector(J, self.j_coeff_spaces, counts)

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
        transformed = jax.vmap(
            lambda values: self.TD.mask_nyquist(self.TDp.forward(values))
        )(jnp.stack(n))
        H = (transformed[0], transformed[1], transformed[2])

        utotal = (up[0], up[1] + self.Ubp, up[2])
        emf = physical_cross(utotal, bp)
        HA = self._A_forward_emf(emf)
        return H, HA

    def _step_cnab2_mhd(self, state: MHDState, dt: Array, factor_args=None) -> MHDState:
        state = self._ensure_mhd_history(state)
        assert state.nonlinear_A_old is not None
        su_args, sg_args, s00_args, a_args = (
            self._runtime_factor_args() if factor_args is None else factor_args
        )
        H, HA = self._mhd_convection(state)
        current_A = self._A_mass_rhs(HA)
        extrapolated_A = variable_ab2_extrapolate(
            current_A,
            state.nonlinear_A_old,
            state.flow.have_old,
            dt,
            state.flow.previous_dt,
        )
        flow_new = self._cnab2_flow_update(
            state.flow, H, dt, (su_args, sg_args, s00_args)
        )
        half_dt = 0.5 * dt
        mass_A = self._A_mass_rhs(state.A)
        linear_A = self._A_eta_lap(state.A)
        rhs_A = [
            mass_A[i] + half_dt * linear_A[i] + dt * extrapolated_A[i] for i in range(3)
        ]
        return MHDState(
            flow=flow_new,
            A=self._A_solve(rhs_A, a_args),
            nonlinear_A_old=current_A,
        )

    def step(
        self, state: MHDState, dt: Array | None = None, factor_args=None
    ) -> MHDState:
        if self._cnab2:
            return self._step_cnab2_mhd(
                state, self._dt_array if dt is None else dt, factor_args
            )
        a, b, _ = self.timestepper.stages()
        if dt is None:
            dt = self._dt_array
        su_args, sg_args, s00_args, a_args = (
            self._runtime_factor_args() if factor_args is None else factor_args
        )
        steps = self.timestepper.steps()
        flow0 = state.flow
        A0 = state.A
        u0_rhs = self.Mu @ flow0.u[0]
        g0_rhs = self.Mg @ flow0.g
        v00_rhs0 = self.M00 @ flow0.u[1][:, 0, 0]
        w00_rhs0 = self.M00 @ flow0.u[2][:, 0, 0]
        A0_rhs = self._A_mass_rhs(A0)

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
            nonlinear_A.append(self._A_mass_rhs(HA))

            rhs_u = u0_rhs
            rhs_g = g0_rhs
            rhs_v = v00_rhs0
            rhs_w = w00_rhs0
            rhs_A = list(A0_rhs)
            for j in range(rk + 1):
                rhs_u = rhs_u + dt * b[rk + 1, j] * nonlinear_u[j]
                rhs_g = rhs_g + dt * b[rk + 1, j] * nonlinear_g[j]
                rhs_v = rhs_v + dt * b[rk + 1, j] * nonlinear_v[j]
                rhs_w = rhs_w + dt * b[rk + 1, j] * nonlinear_w[j]
                rhs_A = [
                    rhs_A[i] + dt * b[rk + 1, j] * nonlinear_A[j][i] for i in range(3)
                ]

            if rk > 0:
                linear_u.append(self.Lu @ flow_stage.u[0])
                linear_g.append(self.Lg @ flow_stage.g)
                linear_v.append(self.L00 @ flow_stage.u[1][:, 0, 0])
                linear_w.append(self.L00 @ flow_stage.u[2][:, 0, 0])
                linear_A.append(self._A_eta_lap(A_stage))
                for j in range(rk):
                    rhs_u = rhs_u + dt * a[rk + 1, j + 1] * linear_u[j]
                    rhs_g = rhs_g + dt * a[rk + 1, j + 1] * linear_g[j]
                    rhs_v = rhs_v + dt * a[rk + 1, j + 1] * linear_v[j]
                    rhs_w = rhs_w + dt * a[rk + 1, j + 1] * linear_w[j]
                    rhs_A = [
                        rhs_A[i] + dt * a[rk + 1, j + 1] * linear_A[j][i]
                        for i in range(3)
                    ]

            u0_new = self._solve_prefactor(
                self.Su_factor, self.TB.mask_nyquist(rhs_u), su_args
            )
            g_new = self._solve_prefactor(
                self.Sg_factor, self.TD.mask_nyquist(rhs_g), sg_args
            )
            v00_new = self._solve_prefactor(self.S00_factor, rhs_v, s00_args)
            w00_new = self._solve_prefactor(self.S00_factor, rhs_w, s00_args)
            u_new = self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new)
            A_stage = self._A_solve(rhs_A, a_args)
            flow_stage = KMMState(u=u_new, g=g_new)

        flow_stage = KMMState(
            u=flow_stage.u,
            g=flow_stage.g,
            nonlinear_old=state.flow.nonlinear_old,
            have_old=state.flow.have_old,
            previous_dt=state.flow.previous_dt,
        )
        return MHDState(
            flow=flow_stage,
            A=A_stage,
            nonlinear_A_old=state.nonlinear_A_old,
        )

    def set_dt(self, dt: float) -> None:
        super().set_dt(dt)
        self._build_A_operators()

    def solve(self, state: MHDState, steps: int) -> MHDState:
        return self._rollout_cache(self._ensure_mhd_history(state), int(steps))

    def magnetic_divergence_l2(self, state: MHDState) -> Array:
        B = self.update_B_from_A(state.A)
        SB = self.b_coeff_spaces
        divb = (
            SB[0].backward_primitive(B[0], (1, 0, 0))
            + SB[1].backward_primitive(B[1], (0, 1, 0))
            + SB[2].backward_primitive(B[2], (0, 0, 1))
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
        for bi, space in zip(bp, self.b_coeff_spaces, strict=True):
            magnetic_energy = magnetic_energy + jnp.real(
                integrate(jnp.conj(bi) * bi, space)
            )
        return {
            **flow_diag,
            "divB_L2": self.magnetic_divergence_l2(state),
            "Emag": magnetic_energy,
        }
