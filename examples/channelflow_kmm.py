"""Kim-Moin-Moser channel-flow solver built on jaxfun Galerkin spaces.

This is a JAX/Galerkin port of the core velocity-vorticity machinery in
couette/ChannelFlow.py. It evolves wall-normal velocity ``u`` on a clamped
biharmonic basis and wall-normal vorticity ``g`` on a Dirichlet basis, then
reconstructs the streamwise/spanwise velocity components from incompressibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import sympy as sp
from jax import Array

from jaxfun import Domain
from jaxfun.galerkin import (
    FunctionSpace,
    K_over_K2,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import integrate
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.integrators import IMEXRK222, PDEIMEXRK, ars_stage_rhs
from jaxfun.la.solvers import Biharmonic, Helmholtz

type Velocity = tuple[Array, Array, Array]


@dataclass(frozen=True)
class KMMState:
    """Coefficient-space KMM state."""

    u: Velocity
    g: Array


class KMM:
    """Velocity-vorticity channel-flow solver following couette/ChannelFlow.py.

    Reference: couette/ChannelFlow.py:44-251. Component convention matches the
    reference exactly: component 0 is wall-normal, 1 streamwise, 2 spanwise.
    """

    def __init__(
        self,
        N: tuple[int, int, int] = (17, 16, 16),
        domain: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (-1.0, 1.0),
            (0.0, 4.0 * float(sp.pi)),
            (0.0, 2.0 * float(sp.pi)),
        ),
        nu: float = 1.0 / 600.0,
        dt: float = 0.01,
        family: str = "L",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        dpdy: float = 0.0,
        timestepper: type[PDEIMEXRK] = IMEXRK222,
    ) -> None:
        self.N = tuple(int(n) for n in N)
        self.domain = domain
        self.nu = float(nu)
        self.dt = float(dt)
        self.padding_factor = padding_factor
        self.dpdy = float(dpdy)
        self.timestepper = timestepper
        if not issubclass(timestepper, PDEIMEXRK):
            raise NotImplementedError("KMM currently supports ARS PDEIMEXRK steppers")

        family_cls = self._family_class(family)
        self.B0 = FunctionSpace(
            self.N[0], family_cls, bc=(0, 0, 0, 0), domain=Domain(*domain[0]), name="B0"
        )
        self.D0 = FunctionSpace(
            self.N[0], family_cls, bc=(0, 0), domain=Domain(*domain[0]), name="D0"
        )
        self.C0 = FunctionSpace(
            self.N[0], family_cls, domain=Domain(*domain[0]), name="C0"
        )
        self.F1 = FunctionSpace(
            self.N[1], Fourier, domain=Domain(*domain[1]), name="F1"
        )
        self.F2 = FunctionSpace(
            self.N[2], Fourier, domain=Domain(*domain[2]), name="F2"
        )

        self.TB = TensorProduct(self.B0, self.F1, self.F2, name="TB")
        self.TD = TensorProduct(self.D0, self.F1, self.F2, name="TD")
        self.TC = TensorProduct(self.C0, self.F1, self.F2, name="TC")
        self.TBp = self.TB.get_dealiased(padding_factor)
        self.TDp = self.TD.get_dealiased(padding_factor)
        self.padding_counts = self.TDp.num_quad_points

        self.D00 = FunctionSpace(
            self.N[0], family_cls, bc=(0, 0), domain=Domain(*domain[0]), name="D00"
        )
        self.C00 = FunctionSpace(
            self.N[0], family_cls, domain=Domain(*domain[0]), name="C00"
        )

        self.K = self.TD.local_wavenumbers(scaled=True)
        self.K_over_K2 = K_over_K2(self.K, axes=(1, 2))
        self.X = self.TD.mesh()
        self.Xp = self.TDp.mesh()

        self._build_operators()

    @staticmethod
    def _family_class(family: str):
        family = family.upper()
        if family.startswith("L"):
            return Legendre
        if family.startswith("C"):
            return Chebyshev
        raise ValueError("family must be 'L' or 'C'")

    @staticmethod
    def _lap(expr: sp.Expr, coords: tuple[sp.Symbol, ...]) -> sp.Expr:
        return sum(sp.diff(expr, coord, 2) for coord in coords)

    def _build_operators(self) -> None:
        """Assemble KMM implicit operators.

        References: couette/ChannelFlow.py:145-163 and the ARS stage update in
        shenfun/shenfun/utilities/integrators.py:702-817.
        """
        a, _, _ = self.timestepper.stages()
        self._gamma = float(a[1, 1])

        ub = TrialFunction(self.TB, name="ub")
        vb = TestFunction(self.TB, name="vb")
        hb = TrialFunction(self.TD, name="hb")
        vh = TestFunction(self.TD, name="vh")
        coords = self.TB.system.base_scalars()
        lap_ub = self._lap(ub, coords)
        lap_hb = self._lap(hb, coords)

        self.Mu = inner(vb * lap_ub, sparse=True)
        self.Lu = inner(vb * (self.nu * self._lap(lap_ub, coords)), sparse=True)
        self.Mg = inner(vh * hb, sparse=True)
        self.Lg = inner(vh * (self.nu * lap_hb), sparse=True)
        self.Su = Biharmonic(
            vb,
            ub,
            coeff=self.dt * self._gamma,
            diffusivity=self.nu,
            coords=coords,
            sparse=True,
        )
        self.Sg = Helmholtz(
            vh,
            hb,
            coeff=self.dt * self._gamma,
            diffusivity=self.nu,
            coords=coords,
            sparse=True,
        )

        u0 = TrialFunction(self.D00, name="u0")
        v0 = TestFunction(self.D00, name="v0")
        (x0,) = self.D00.system.base_scalars()
        self.M00 = inner(v0 * u0, sparse=True)
        self.L00 = inner(v0 * (self.nu * sp.diff(u0, x0, 2)), sparse=True)
        self.S00 = self.M00 - (self.dt * self._gamma) * self.L00
        if self.dpdy != 0.0:
            ones = jnp.ones(self.C00.num_quad_points) * (-self.dpdy)
            self.dpdy_rhs = self.D00.scalar_product(ones)
        else:
            self.dpdy_rhs = jnp.zeros(self.D00.num_dofs)

    def zero_state(self) -> KMMState:
        u = (
            jnp.zeros(self.TB.num_dofs, dtype=complex),
            jnp.zeros(self.TD.num_dofs, dtype=complex),
            jnp.zeros(self.TD.num_dofs, dtype=complex),
        )
        return KMMState(u=u, g=jnp.zeros(self.TD.num_dofs, dtype=complex))

    def state_from_physical(self, u_phys: Velocity) -> KMMState:
        """Transform physical velocity samples into a KMM coefficient state."""
        u0 = self.TB.mask_nyquist(self.TB.forward(u_phys[0]))
        u1 = self.TD.mask_nyquist(self.TD.forward(u_phys[1]))
        u2 = self.TD.mask_nyquist(self.TD.forward(u_phys[2]))
        g = self.TD.mask_nyquist(1j * self.K[1] * u2 - 1j * self.K[2] * u1)
        return KMMState(u=(u0, u1, u2), g=g)

    def _backward_velocity(self, u: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        return (
            self.TB.backward(u[0], N=counts),
            self.TD.backward(u[1], N=counts),
            self.TD.backward(u[2], N=counts),
        )

    def _velocity_gradients(self, u: Velocity) -> dict[str, Array]:
        counts = self.padding_counts
        return {
            "dudx": self.TB.backward_primitive(u[0], (1, 0, 0), N=counts),
            "dudy": self.TB.backward_primitive(u[0], (0, 1, 0), N=counts),
            "dudz": self.TB.backward_primitive(u[0], (0, 0, 1), N=counts),
            "dvdx": self.TD.backward_primitive(u[1], (1, 0, 0), N=counts),
            "dvdy": self.TD.backward_primitive(u[1], (0, 1, 0), N=counts),
            "dvdz": self.TD.backward_primitive(u[1], (0, 0, 1), N=counts),
            "dwdx": self.TD.backward_primitive(u[2], (1, 0, 0), N=counts),
            "dwdy": self.TD.backward_primitive(u[2], (0, 1, 0), N=counts),
            "dwdz": self.TD.backward_primitive(u[2], (0, 0, 1), N=counts),
        }

    def _add_base_convection(
        self, n: Velocity, up: Velocity, grads: dict[str, Array]
    ) -> Velocity:
        return n

    def convection(self, state: KMMState) -> Velocity:
        """Return dealiased convection coefficients in the TD space.

        Reference: couette/ChannelFlow.py:199-225. This implements conv=0, the
        gradient form used by the Plane Couette fluctuation scripts.
        """
        up = self._backward_velocity(state.u, padded=True)
        g = self._velocity_gradients(state.u)
        n = (
            up[0] * g["dudx"] + up[1] * g["dudy"] + up[2] * g["dudz"],
            up[0] * g["dvdx"] + up[1] * g["dvdy"] + up[2] * g["dvdz"],
            up[0] * g["dwdx"] + up[1] * g["dwdy"] + up[2] * g["dwdz"],
        )
        n = self._add_base_convection(n, up, g)
        return tuple(self.TD.mask_nyquist(self.TDp.forward(ni)) for ni in n)

    def _nonlinear_rhs(self, H: Velocity) -> tuple[Array, Array, Array, Array]:
        counts = self.TD.num_quad_points
        Hu = (
            self.TD.backward_primitive(H[1], (1, 1, 0), N=counts)
            + self.TD.backward_primitive(H[2], (1, 0, 1), N=counts)
            - self.TD.backward_primitive(H[0], (0, 2, 0), N=counts)
            - self.TD.backward_primitive(H[0], (0, 0, 2), N=counts)
        )
        Hg = self.TD.backward_primitive(
            H[1], (0, 0, 1), N=counts
        ) - self.TD.backward_primitive(H[2], (0, 1, 0), N=counts)
        Nu = self.TB.scalar_product(Hu)
        Ng = self.TD.scalar_product(Hg)
        Nv00 = -(self.M00 @ H[1][:, 0, 0]) + self.dpdy_rhs
        Nw00 = -(self.M00 @ H[2][:, 0, 0])
        return Nu, Ng, Nv00, Nw00

    def _reconstruct_velocity(
        self, u0: Array, g: Array, v00: Array, w00: Array
    ) -> Velocity:
        dudx_phys = self.TB.backward_primitive(u0, (1, 0, 0), N=self.TD.num_quad_points)
        f = self.TD.forward(dudx_phys)
        u1 = 1j * (self.K_over_K2[0] * f + self.K_over_K2[1] * g)
        u2 = 1j * (self.K_over_K2[1] * f - self.K_over_K2[0] * g)
        u1 = u1.at[:, 0, 0].set(v00)
        u2 = u2.at[:, 0, 0].set(w00)
        return (
            self.TB.mask_nyquist(u0),
            self.TD.mask_nyquist(u1),
            self.TD.mask_nyquist(u2),
        )

    def step(self, state: KMMState) -> KMMState:
        """Advance one IMEX-RK step."""
        a, b, _ = self.timestepper.stages()
        steps = self.timestepper.steps()
        u0_initial = state.u[0]
        g_initial = state.g
        v00_initial = state.u[1][:, 0, 0]
        w00_initial = state.u[2][:, 0, 0]
        u0_rhs = self.Mu @ u0_initial
        g0_rhs = self.Mg @ g_initial
        v00_rhs0 = self.M00 @ v00_initial
        w00_rhs0 = self.M00 @ w00_initial

        u_stage = state.u
        g_stage = state.g
        base_rhs = (u0_rhs, g0_rhs, v00_rhs0, w00_rhs0)
        nonlinear_history: list[tuple[Array, Array, Array, Array]] = []
        linear_history: list[tuple[Array, Array, Array, Array]] = []

        for rk in range(steps):
            H = self.convection(KMMState(u=u_stage, g=g_stage))
            nonlinear_history.append(self._nonlinear_rhs(H))

            if rk > 0:
                linear_history.append(
                    (
                        self.Lu @ u_stage[0],
                        self.Lg @ g_stage,
                        self.L00 @ u_stage[1][:, 0, 0],
                        self.L00 @ u_stage[2][:, 0, 0],
                    )
                )

            rhs_u, rhs_g, rhs_v, rhs_w = ars_stage_rhs(
                base_rhs, nonlinear_history, linear_history, a, b, self.dt, rk
            )
            u0_new = self.Su.solve(self.TB.mask_nyquist(rhs_u))
            g_new = self.Sg.solve(self.TD.mask_nyquist(rhs_g))
            v00_new = self.S00.solve(rhs_v)
            w00_new = self.S00.solve(rhs_w)
            u_stage = self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new)
            g_stage = g_new

        return KMMState(u=u_stage, g=g_stage)

    def solve(self, state: KMMState, steps: int) -> KMMState:
        out = state
        for _ in range(int(steps)):
            out = self.step(out)
        return out

    def divergence_l2(self, state: KMMState) -> Array:
        divu = (
            self.TB.backward_primitive(state.u[0], (1, 0, 0))
            + self.TD.backward_primitive(state.u[1], (0, 1, 0))
            + self.TD.backward_primitive(state.u[2], (0, 0, 1))
        )
        return jnp.sqrt(jnp.real(integrate(jnp.conj(divu) * divu, self.TC)))

    def perturbation_energy(self, state: KMMState) -> Array:
        up = self._backward_velocity(state.u)
        spaces = (self.TB, self.TD, self.TD)
        total = jnp.asarray(0.0, dtype=up[0].real.dtype)
        for ui, space in zip(up, spaces, strict=True):
            total = total + jnp.real(integrate(jnp.conj(ui) * ui, space))
        return total
