"""Kim-Moin-Moser channel-flow solver built on jaxfun Galerkin spaces.

This is a JAX/Galerkin port of the core velocity-vorticity machinery in
couette/ChannelFlow.py. It evolves wall-normal velocity ``u`` on a clamped
biharmonic basis and wall-normal vorticity ``g`` on a Dirichlet basis, then
reconstructs the streamwise/spanwise velocity components from incompressibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
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
from jaxfun.integrators import IMEXRK3, IMEXRK222, PDEIMEXRK, ars_stage_rhs
from jaxfun.integrators.cnab2 import (
    ScanRolloutCache,
    ScanRolloutCacheInfo,
    variable_ab2_extrapolate,
)
from jaxfun.io import Cadence, run_with_cadence
from jaxfun.la.solvers import Biharmonic, Helmholtz

type Velocity = tuple[Array, Array, Array]
type KMMNonlinear = tuple[Array, Array, Array, Array]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class KMMState:
    """Coefficient-space KMM state with optional CNAB2 history."""

    u: Velocity
    g: Array
    nonlinear_old: KMMNonlinear | None = None
    have_old: float | Array = 0.0
    previous_dt: float | Array = 0.0

    def tree_flatten(self):
        return (
            self.u,
            self.g,
            self.nonlinear_old,
            self.have_old,
            self.previous_dt,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        u, g, nonlinear_old, have_old, previous_dt = children
        return cls(
            u=u,
            g=g,
            nonlinear_old=nonlinear_old,
            have_old=have_old,
            previous_dt=previous_dt,
        )


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
        family: str = "C",
        padding_factor: tuple[float, float, float] = (1.0, 1.5, 1.5),
        dpdy: float = 0.0,
        timestepper: type[PDEIMEXRK] | type[IMEXRK3] | None = None,
        time_integrator: str | None = None,
    ) -> None:
        self.N = tuple(int(n) for n in N)
        self.domain = domain
        self.nu = float(nu)
        self.dt = float(dt)
        # Keep the controller timestep as a dynamic JAX value in compiled
        # rollouts. ``self.dt`` remains the public Python scalar used when
        # rebuilding dt-dependent implicit operators and reporting time.
        self._dt_array = jnp.asarray(self.dt)
        self.padding_factor = padding_factor
        self.dpdy = float(dpdy)
        requested_integrator = None if time_integrator is None else str(time_integrator)
        if requested_integrator is not None and requested_integrator not in {
            "IMEXRK222",
            "IMEXRK3",
            "CNAB2",
        }:
            raise ValueError(
                "time_integrator must be one of {'IMEXRK222', 'IMEXRK3', 'CNAB2'}"
            )
        if timestepper is None:
            timestepper = IMEXRK3 if requested_integrator == "IMEXRK3" else IMEXRK222
        if not (issubclass(timestepper, PDEIMEXRK) or timestepper is IMEXRK3):
            raise NotImplementedError("KMM supports ARS PDEIMEXRK steppers and IMEXRK3")

        inferred_integrator = (
            "IMEXRK3"
            if timestepper is IMEXRK3
            else "IMEXRK222"
            if timestepper is IMEXRK222
            else timestepper.__name__
        )
        if requested_integrator == "CNAB2":
            if timestepper is not IMEXRK222:
                raise ValueError("CNAB2 requires the default IMEXRK222 timestepper")
        elif (
            requested_integrator is not None
            and requested_integrator != inferred_integrator
        ):
            raise ValueError(
                f"time_integrator={requested_integrator!r} conflicts with "
                f"timestepper={timestepper.__name__}"
            )

        self.timestepper = timestepper
        self.time_integrator = requested_integrator or inferred_integrator
        self._low_storage_imexrk3 = self.time_integrator == "IMEXRK3"
        self._cnab2 = self.time_integrator == "CNAB2"

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
        self.TYZ = TensorProduct(self.F1, self.F2, name="TYZ")
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
        self._rollout_cache = ScanRolloutCache(
            self._step_with_dt,
            dynamic_args=lambda: (self._dt_array, self._runtime_factor_args()),
        )
        self._pressure_cache = None

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
        a, b, _ = self.timestepper.stages()
        self._gamma = (
            None
            if self._low_storage_imexrk3
            else (0.5 if self._cnab2 else float(a[1, 1]))
        )

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
        if self._low_storage_imexrk3:
            gammas = tuple(float((a[rk] + b[rk]) * self.dt / 2.0) for rk in range(3))
            self.Su = tuple(
                Biharmonic(
                    vb,
                    ub,
                    coeff=gamma,
                    diffusivity=self.nu,
                    coords=coords,
                    sparse=True,
                )
                for gamma in gammas
            )
            self.Sg = tuple(
                Helmholtz(
                    vh,
                    hb,
                    coeff=gamma,
                    diffusivity=self.nu,
                    coords=coords,
                    sparse=True,
                )
                for gamma in gammas
            )
        else:
            assert self._gamma is not None
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
        if self._low_storage_imexrk3:
            self.S00 = tuple(
                Helmholtz(
                    v0,
                    u0,
                    coeff=gamma,
                    diffusivity=self.nu,
                    coords=(x0,),
                    sparse=True,
                )
                for gamma in gammas
            )
        else:
            assert self._gamma is not None
            self.S00 = Helmholtz(
                v0,
                u0,
                coeff=self.dt * self._gamma,
                diffusivity=self.nu,
                coords=(x0,),
                sparse=True,
            )
        if self.dpdy != 0.0:
            ones = jnp.ones(self.C00.num_quad_points) * (-self.dpdy)
            self.dpdy_rhs = self.D00.scalar_product(ones)
        else:
            self.dpdy_rhs = jnp.zeros(self.D00.num_dofs)
        self.Su_factor = self._prefactor_solver(self.Su)
        self.Sg_factor = self._prefactor_solver(self.Sg)
        self.S00_factor = self._prefactor_solver(self.S00)

    @staticmethod
    def _prefactor_solver(solver):
        if isinstance(solver, tuple):
            return tuple(item.lu_factor() for item in solver)
        return solver.lu_factor()

    @classmethod
    def _factor_runtime_args(cls, factor):
        if isinstance(factor, tuple):
            return tuple(cls._factor_runtime_args(item) for item in factor)
        runtime_args = getattr(factor, "runtime_args", None)
        return runtime_args() if runtime_args is not None else (factor,)

    def _runtime_factor_args(self):
        return (
            self._factor_runtime_args(self.Su_factor),
            self._factor_runtime_args(self.Sg_factor),
            self._factor_runtime_args(self.S00_factor),
        )

    @staticmethod
    def _solve_prefactor(factor, rhs, runtime_args=None):
        if runtime_args is None:
            return factor.solve(rhs)
        solve_runtime = getattr(factor, "solve_with_runtime_args", None)
        if solve_runtime is not None:
            return solve_runtime(rhs, runtime_args)
        if len(runtime_args) != 1:
            raise ValueError("non-wavenumber factor requires one runtime argument")
        return runtime_args[0].solve(rhs)

    def _build_pressure_cache(self) -> dict[str, Array]:
        """Assemble radial matrices for optional KMM pressure recovery."""
        pressure_test = FunctionSpace(
            self.N[0],
            self._family_class(self.B0.orthogonal.name),
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=Domain(*self.domain[0]),
            name="PN",
        )
        n = self.C0.num_dofs
        eye_p = jnp.eye(n, dtype=float)
        eye_t = jnp.eye(pressure_test.num_dofs, dtype=float)

        values = jax.vmap(self.C0.backward)(eye_p).T
        lap_values = jax.vmap(lambda c: self.C0.backward_primitive(c, 2))(eye_p).T
        tests = jax.vmap(pressure_test.backward)(eye_t).T
        # These are Galerkin pressure rows, so retain the orthogonality measure.
        weights = self.C0.quadrature_weights()

        stiffness = jnp.einsum("xi,xj,x->ij", tests, lap_values, weights)
        mass = jnp.einsum("xi,xj,x->ij", tests, values, weights)

        bounds = jnp.asarray([self.domain[0][0], self.domain[0][1]], dtype=float)
        ref_bounds = self.C0.map_reference_domain(bounds)
        d1_rows = self.C0.evaluate_basis_derivative(ref_bounds, 1) * float(
            self.C0.domain_factor
        )
        b0_d2_rows = (
            self.B0.evaluate_basis_derivative(self.B0.map_reference_domain(bounds), 2)
            * float(self.B0.domain_factor) ** 2
        )
        k2 = jnp.squeeze(self.K[1] * self.K[1] + self.K[2] * self.K[2], axis=0)
        return {
            "tests": tests,
            "weights": weights,
            "stiffness": stiffness,
            "mass": mass,
            "d1_rows": d1_rows,
            "b0_d2_rows": b0_d2_rows,
            "k2": k2,
        }

    def _pressure_solver_cache(self) -> dict[str, Array]:
        cache = self._pressure_cache
        if cache is None:
            cache = self._build_pressure_cache()
            self._pressure_cache = cache
        return cache

    def compute_pressure_coefficients(
        self, state: KMMState, H: Velocity | None = None
    ) -> Array:
        """Recover pressure coefficients from the KMM velocity state.

        This mirrors ``ChannelFlow.KMM.compute_pressure``: the Poisson RHS is
        ``-div(H)`` and the wall Neumann data are
        ``nu*d**2(u_wallnormal)/dx**2``.  The returned coefficients live in the
        unconstrained ``TC`` space; the zero transverse mode is pinned by setting
        the first radial coefficient to zero.
        """
        cache = self._pressure_solver_cache()
        H = self.convection(state) if H is None else H

        div_h = (
            self.TD.backward_primitive(H[0], (1, 0, 0))
            + self.TD.backward_primitive(H[1], (0, 1, 0))
            + self.TD.backward_primitive(H[2], (0, 0, 1))
        )
        rhs_phys = -div_h
        rhs_modes = jax.vmap(self.TYZ.forward, in_axes=0)(rhs_phys)
        rhs_galerkin = jnp.einsum(
            "xi,xkl,x->ikl", cache["tests"], rhs_modes, cache["weights"]
        )

        wall_neumann = self.nu * jnp.einsum(
            "bd,dkl->bkl", cache["b0_d2_rows"], state.u[0]
        )
        rhs = jnp.concatenate(
            (
                jnp.moveaxis(rhs_galerkin, 0, -1),
                jnp.moveaxis(wall_neumann, 0, -1),
            ),
            axis=-1,
        )

        galerkin = cache["stiffness"] - cache["k2"][..., None, None] * cache["mass"]
        bc_rows = jnp.broadcast_to(
            cache["d1_rows"], (*cache["k2"].shape, *cache["d1_rows"].shape)
        )
        matrices = jnp.concatenate((galerkin, bc_rows), axis=-2)
        matrices = matrices.at[0, 0, 0, :].set(0)
        matrices = matrices.at[0, 0, 0, 0].set(1)
        rhs = rhs.at[0, 0, 0].set(0)

        matrices = matrices.astype(rhs.dtype)
        coeff_modes = jnp.linalg.solve(matrices, rhs[..., None])[..., 0]
        coeff = jnp.moveaxis(coeff_modes, -1, 0)
        return self.TC.mask_nyquist(coeff)

    def compute_pressure(self, state: KMMState, H: Velocity | None = None) -> Array:
        """Return recovered pressure on the standard quadrature mesh."""
        return self.TC.backward(self.compute_pressure_coefficients(state, H))

    def _ensure_flow_history(self, state: KMMState) -> KMMState:
        if not self._cnab2 or state.nonlinear_old is not None:
            return state
        nonlinear_old = (
            jnp.zeros_like(state.u[0]),
            jnp.zeros_like(state.g),
            jnp.zeros_like(jnp.real(state.u[1][:, 0, 0])),
            jnp.zeros_like(jnp.real(state.u[2][:, 0, 0])),
        )
        return KMMState(
            u=state.u,
            g=state.g,
            nonlinear_old=nonlinear_old,
            have_old=jnp.zeros_like(state.have_old),
            previous_dt=jnp.asarray(0.0, dtype=jnp.real(state.g).dtype),
        )

    def zero_state(self) -> KMMState:
        u = (
            jnp.zeros(self.TB.num_dofs, dtype=complex),
            jnp.zeros(self.TD.num_dofs, dtype=complex),
            jnp.zeros(self.TD.num_dofs, dtype=complex),
        )
        state = KMMState(u=u, g=jnp.zeros(self.TD.num_dofs, dtype=complex))
        return self._ensure_flow_history(state)

    def state_from_physical(self, u_phys: Velocity) -> KMMState:
        """Transform physical velocity samples into a KMM coefficient state."""
        u0 = self.TB.mask_nyquist(self.TB.forward(u_phys[0]))
        u1 = self.TD.mask_nyquist(self.TD.forward(u_phys[1]))
        u2 = self.TD.mask_nyquist(self.TD.forward(u_phys[2]))
        g = self.TD.mask_nyquist(1j * self.K[1] * u2 - 1j * self.K[2] * u1)
        return self._ensure_flow_history(KMMState(u=(u0, u1, u2), g=g))

    def _backward_velocity(self, u: Velocity, padded: bool = False) -> Velocity:
        counts = self.padding_counts if padded else None
        transverse = jax.vmap(
            lambda coefficients: self.TD.backward(coefficients, N=counts)
        )(jnp.stack(u[1:]))
        return (
            self.TB.backward(u[0], N=counts),
            transverse[0],
            transverse[1],
        )

    def velocity_vorticity_physical(
        self, u: Velocity, *, padded: bool = False
    ) -> Velocity:
        """Return curl(u) on the requested physical quadrature mesh."""

        counts = self.padding_counts if padded else None
        spaces = (self.TB, self.TD, self.TD)

        def derivative(component: int, order: tuple[int, int, int]) -> Array:
            return spaces[component].backward_primitive(u[component], order, N=counts)

        return (
            derivative(2, (0, 1, 0)) - derivative(1, (0, 0, 1)),
            derivative(0, (0, 0, 1)) - derivative(2, (1, 0, 0)),
            derivative(1, (1, 0, 0)) - derivative(0, (0, 1, 0)),
        )

    def _velocity_gradients(self, u: Velocity) -> dict[str, Array]:
        counts = self.padding_counts
        transverse = jnp.stack(u[1:])

        def transverse_derivative(order: tuple[int, int, int]) -> Array:
            return jax.vmap(
                lambda coefficients: self.TD.backward_primitive(
                    coefficients, order, N=counts
                )
            )(transverse)

        dx = transverse_derivative((1, 0, 0))
        dy = transverse_derivative((0, 1, 0))
        dz = transverse_derivative((0, 0, 1))
        return {
            "dudx": self.TB.backward_primitive(u[0], (1, 0, 0), N=counts),
            "dudy": self.TB.backward_primitive(u[0], (0, 1, 0), N=counts),
            "dudz": self.TB.backward_primitive(u[0], (0, 0, 1), N=counts),
            "dvdx": dx[0],
            "dvdy": dy[0],
            "dvdz": dz[0],
            "dwdx": dx[1],
            "dwdy": dy[1],
            "dwdz": dz[1],
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
        transformed = jax.vmap(
            lambda values: self.TD.mask_nyquist(self.TDp.forward(values))
        )(jnp.stack(n))
        return transformed[0], transformed[1], transformed[2]

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
        Nv00 = -(self.M00 @ jnp.real(H[1][:, 0, 0])) + self.dpdy_rhs
        Nw00 = -(self.M00 @ jnp.real(H[2][:, 0, 0]))
        return Nu, Ng, jnp.real(Nv00), jnp.real(Nw00)

    def _reconstruct_velocity(
        self, u0: Array, g: Array, v00: Array, w00: Array
    ) -> Velocity:
        dudx_phys = self.TB.backward_primitive(u0, (1, 0, 0), N=self.TD.num_quad_points)
        f = self.TD.forward(dudx_phys)
        u1 = 1j * (self.K_over_K2[0] * f + self.K_over_K2[1] * g)
        u2 = 1j * (self.K_over_K2[1] * f - self.K_over_K2[0] * g)
        u1 = u1.at[:, 0, 0].set(jnp.real(v00))
        u2 = u2.at[:, 0, 0].set(jnp.real(w00))
        return (
            self.TB.mask_nyquist(u0),
            self.TD.mask_nyquist(u1),
            self.TD.mask_nyquist(u2),
        )

    def _step_imexrk3(self, state: KMMState, dt: Array, factor_args=None) -> KMMState:
        """Advance one Spalart low-storage IMEXRK3 step."""
        a, b, _ = self.timestepper.stages()
        su_args, sg_args, s00_args = (
            self._runtime_factor_args() if factor_args is None else factor_args
        )
        u_stage = state.u
        g_stage = state.g
        previous = (
            jnp.zeros_like(state.u[0]),
            jnp.zeros_like(state.g),
            jnp.zeros_like(jnp.real(state.u[1][:, 0, 0])),
            jnp.zeros_like(jnp.real(state.u[2][:, 0, 0])),
        )

        assert isinstance(self.Su_factor, tuple)
        assert isinstance(self.Sg_factor, tuple)
        assert isinstance(self.S00_factor, tuple)
        for rk in range(self.timestepper.steps()):
            H = self.convection(KMMState(u=u_stage, g=g_stage))
            current = self._nonlinear_rhs(H)
            gamma = (a[rk] + b[rk]) * dt / 2.0
            rhs_u = self.Mu @ u_stage[0] + gamma * (self.Lu @ u_stage[0])
            rhs_g = self.Mg @ g_stage + gamma * (self.Lg @ g_stage)
            rhs_v = self.M00 @ jnp.real(u_stage[1][:, 0, 0]) + gamma * (
                self.L00 @ jnp.real(u_stage[1][:, 0, 0])
            )
            rhs_w = self.M00 @ jnp.real(u_stage[2][:, 0, 0]) + gamma * (
                self.L00 @ jnp.real(u_stage[2][:, 0, 0])
            )
            rhs_u = rhs_u + dt * (a[rk] * current[0] + b[rk] * previous[0])
            rhs_g = rhs_g + dt * (a[rk] * current[1] + b[rk] * previous[1])
            rhs_v = rhs_v + dt * (a[rk] * current[2] + b[rk] * previous[2])
            rhs_w = rhs_w + dt * (a[rk] * current[3] + b[rk] * previous[3])

            u0_new = self._solve_prefactor(
                self.Su_factor[rk], self.TB.mask_nyquist(rhs_u), su_args[rk]
            )
            g_new = self.TD.mask_nyquist(
                self._solve_prefactor(
                    self.Sg_factor[rk], self.TD.mask_nyquist(rhs_g), sg_args[rk]
                )
            )
            v00_new = jnp.real(
                self._solve_prefactor(self.S00_factor[rk], rhs_v, s00_args[rk])
            )
            w00_new = jnp.real(
                self._solve_prefactor(self.S00_factor[rk], rhs_w, s00_args[rk])
            )
            u_stage = self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new)
            g_stage = g_new
            previous = current
        return KMMState(
            u=u_stage,
            g=g_stage,
            nonlinear_old=state.nonlinear_old,
            have_old=state.have_old,
            previous_dt=state.previous_dt,
        )

    def _cnab2_flow_update(
        self,
        state: KMMState,
        H: Velocity,
        dt: Array,
        factor_args=None,
    ) -> KMMState:
        state = self._ensure_flow_history(state)
        assert state.nonlinear_old is not None
        current = self._nonlinear_rhs(H)
        extrapolated = variable_ab2_extrapolate(
            current,
            state.nonlinear_old,
            state.have_old,
            dt,
            state.previous_dt,
        )
        su_args, sg_args, s00_args = (
            self._runtime_factor_args() if factor_args is None else factor_args
        )
        u0, g = state.u[0], state.g
        v00 = jnp.real(state.u[1][:, 0, 0])
        w00 = jnp.real(state.u[2][:, 0, 0])
        half_dt = 0.5 * dt
        rhs_u = self.Mu @ u0 + half_dt * (self.Lu @ u0) + dt * extrapolated[0]
        rhs_g = self.Mg @ g + half_dt * (self.Lg @ g) + dt * extrapolated[1]
        rhs_v = self.M00 @ v00 + half_dt * (self.L00 @ v00) + dt * extrapolated[2]
        rhs_w = self.M00 @ w00 + half_dt * (self.L00 @ w00) + dt * extrapolated[3]
        u0_new = self._solve_prefactor(
            self.Su_factor, self.TB.mask_nyquist(rhs_u), su_args
        )
        g_new = self.TD.mask_nyquist(
            self._solve_prefactor(self.Sg_factor, self.TD.mask_nyquist(rhs_g), sg_args)
        )
        v00_new = jnp.real(self._solve_prefactor(self.S00_factor, rhs_v, s00_args))
        w00_new = jnp.real(self._solve_prefactor(self.S00_factor, rhs_w, s00_args))
        return KMMState(
            u=self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new),
            g=g_new,
            nonlinear_old=current,
            have_old=jnp.ones_like(state.have_old),
            previous_dt=dt,
        )

    def _step_cnab2(self, state: KMMState, dt: Array, factor_args=None) -> KMMState:
        state = self._ensure_flow_history(state)
        return self._cnab2_flow_update(state, self.convection(state), dt, factor_args)

    def step(
        self, state: KMMState, dt: Array | None = None, factor_args=None
    ) -> KMMState:
        """Advance one second-order IMEX step."""
        if self._cnab2:
            return self._step_cnab2(
                state, self._dt_array if dt is None else dt, factor_args
            )
        if self._low_storage_imexrk3:
            return self._step_imexrk3(
                state, self._dt_array if dt is None else dt, factor_args
            )
        a, b, _ = self.timestepper.stages()
        if dt is None:
            dt = self._dt_array
        su_args, sg_args, s00_args = (
            self._runtime_factor_args() if factor_args is None else factor_args
        )
        steps = self.timestepper.steps()
        u0_initial = state.u[0]
        g_initial = state.g
        v00_initial = jnp.real(state.u[1][:, 0, 0])
        w00_initial = jnp.real(state.u[2][:, 0, 0])
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
                        self.L00 @ jnp.real(u_stage[1][:, 0, 0]),
                        self.L00 @ jnp.real(u_stage[2][:, 0, 0]),
                    )
                )

            rhs_u, rhs_g, rhs_v, rhs_w = ars_stage_rhs(
                base_rhs, nonlinear_history, linear_history, a, b, dt, rk
            )
            u0_new = self._solve_prefactor(
                self.Su_factor, self.TB.mask_nyquist(rhs_u), su_args
            )
            g_new = self.TD.mask_nyquist(
                self._solve_prefactor(
                    self.Sg_factor, self.TD.mask_nyquist(rhs_g), sg_args
                )
            )
            v00_new = jnp.real(self._solve_prefactor(self.S00_factor, rhs_v, s00_args))
            w00_new = jnp.real(self._solve_prefactor(self.S00_factor, rhs_w, s00_args))
            u_stage = self._reconstruct_velocity(u0_new, g_new, v00_new, w00_new)
            g_stage = g_new

        return KMMState(
            u=u_stage,
            g=g_stage,
            nonlinear_old=state.nonlinear_old,
            have_old=state.have_old,
            previous_dt=state.previous_dt,
        )

    def _step_with_dt(self, state: KMMState, dt: Array, factor_args) -> KMMState:
        """Advance one compiled rollout step with runtime dt and factors."""
        return self.step(state, dt, factor_args)

    def set_dt(self, dt: float) -> None:
        """Rebuild the dt-dependent implicit factorizations for a new step.

        Used by adaptive-CFL drivers; the spectral spaces and meshes are dt
        independent, so only the implicit operators are reassembled.
        """
        self.dt = float(dt)
        self._dt_array = jnp.asarray(self.dt)
        self._build_operators()

    def solve(self, state: KMMState, steps: int) -> KMMState:
        return self._rollout_cache(self._ensure_flow_history(state), int(steps))

    def rollout_cache_info(self) -> ScanRolloutCacheInfo:
        """Return bounded executable-cache lifecycle counters."""
        return self._rollout_cache.info()

    def solve_with_cadence(
        self,
        state: KMMState,
        steps: int,
        cadence: Cadence,
        *,
        block_size: int = 1,
        on_diagnostics=None,
        on_snapshot=None,
        on_checkpoint=None,
        should_stop=None,
        t0: float = 0.0,
        tstep0: int = 0,
    ) -> KMMState:
        return run_with_cadence(
            self.solve,
            state,
            steps=steps,
            dt=self.dt,
            cadence=cadence,
            block_size=block_size,
            diagnostics=getattr(self, "diagnostics", None),
            on_diagnostics=on_diagnostics,
            on_snapshot=on_snapshot,
            on_checkpoint=on_checkpoint,
            should_stop=should_stop,
            t0=t0,
            tstep0=tstep0,
        )

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
