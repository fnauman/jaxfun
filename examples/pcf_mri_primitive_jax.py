"""Primitive-b plane-Couette / MRI DNS in jaxfun.

This is the JAX counterpart of ``couette/pcf_mri_primitive.py`` for the
``ky=0`` channel-mode DNS goldens plus a full 3D ``ky`` extension.  It evolves
primitive velocity and magnetic field components directly, not a vector potential,
so the linear eigenmode from ``examples.pcf_linear_jax.PlaneCouetteLinear`` can be
injected block-for-block.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import sympy as sp
from jax import Array

from examples.pcf_linear_jax import PlaneCouetteLinear
from examples.taylor_couette_dns_jax import _positive_pivot_phase
from jaxfun import Domain, Dx
from jaxfun.diagnostics import quadratic_energy
from jaxfun.galerkin import (
    CoupledSpace,
    FunctionSpace,
    InnerKind,
    TensorProduct,
    TestFunction,
    TrialFunction,
    inner,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import integrate
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.integrators.cnab2 import cnab2_rhs, scan_steps
from jaxfun.io import Cadence, run_with_cadence
from jaxfun.la import TPMatrices, TPMatrix

type MHDFields = tuple[Array, Array, Array, Array, Array, Array]


def _dealias_tuple(value: Any, dimensions: int) -> tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        out = tuple(float(item) for item in value)
        if len(out) != dimensions:
            raise ValueError(f"expected {dimensions} dealias values, got {len(out)}")
        return out
    return (float(value),) * dimensions


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AxisymmetricPCFState:
    """Coefficient state for axisymmetric primitive PCF-MHD DNS."""

    x: MHDFields
    p: Array
    nonlinear_old: MHDFields
    have_old: bool | Array = False

    def tree_flatten(self):
        return (self.x, self.p, self.nonlinear_old, jnp.asarray(self.have_old)), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        x, p, nonlinear_old, have_old = children
        return cls(x=x, p=p, nonlinear_old=nonlinear_old, have_old=have_old)


def _bary_interp_matrix(x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=float)
    x_dst = np.asarray(x_dst, dtype=float)
    weights = np.ones(x_src.size)
    for j in range(x_src.size):
        diff = x_src[j] - x_src
        diff[j] = 1.0
        weights[j] = 1.0 / np.prod(diff)
    matrix = np.zeros((x_dst.size, x_src.size))
    for i, value in enumerate(x_dst):
        diff = value - x_src
        exact = np.where(np.abs(diff) < 1.0e-14)[0]
        if exact.size:
            matrix[i, exact[0]] = 1.0
            continue
        scaled = weights / diff
        matrix[i, :] = scaled / scaled.sum()
    return matrix


class AxisymmetricPCFMRIDNSJax:
    """Primitive-variable plane-Couette / shearbox MRI DNS for ``ky=0``.

    Coordinates are ``(z, x)`` in coefficient arrays: Fourier in the vertical
    spanwise direction and no-slip Chebyshev/Legendre in the wall-normal
    direction.  The evolved fields are ``(u_x,u_y,u_z,b_x,b_y,b_z)`` with one
    velocity-pressure saddle point and conducting magnetic walls.
    """

    def __init__(
        self,
        S: float = 1.0,
        omega: float = 2.0 / 3.0,
        B0: float = 0.1,
        nu: float = 1.0e-3,
        eta_mag: float = 1.0e-3,
        Nx: int = 40,
        Nz: int = 16,
        Lz: float = 1.0,
        dt: float = 2.0e-3,
        family: str = "C",
        dealias: float = 1.0,
    ) -> None:
        self.S = float(S)
        self.omega = float(omega)
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nx = int(Nx)
        self.Nz = int(Nz)
        self.Lz = float(Lz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = float(dealias)

        family_cls = self._family_class(self.family)
        dom = Domain(-1.0, 1.0)
        self.F = FunctionSpace(self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="F")
        self.SD = FunctionSpace(self.Nx, family_cls, bc=(0, 0), domain=dom, name="SD")
        self.S0 = FunctionSpace(self.Nx, family_cls, domain=dom, name="S0")
        self.SP = FunctionSpace(
            self.Nx, family_cls, domain=dom, num_dofs=self.Nx - 2, name="SP"
        )
        self.SN = FunctionSpace(
            self.Nx,
            family_cls,
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=dom,
            name="SN",
        )

        self.TD = TensorProduct(self.F, self.SD, name="TDpcf")
        self.T0 = TensorProduct(self.F, self.S0, name="T0pcf")
        self.TP = TensorProduct(self.F, self.SP, name="TPpcf")
        self.TN = TensorProduct(self.F, self.SN, name="TNpcf")
        self.VQ = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TP, self.TD, self.TN, self.TN),
            name="VQpcf",
        )
        self.VE = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN),
            name="VEpcf",
        )

        self.z, self.xcoord = self.TD.system.base_scalars()
        self.VQ_mode_indices = self._mode_indices(self.VQ)
        self.VE_mode_indices = self._mode_indices(self.VE)
        self.Z, self.X = self.T0.mesh()
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
        else:
            self.T0p = None
            self.padded_counts = None

        self.Limp, self.Lexp = self._build_operators()
        self.Limp_modes = self._extract_mode_matrices(self.Limp, self.VQ_mode_indices)
        self.Lexp_modes = self._extract_mode_matrices(self.Lexp, self.VE_mode_indices)
        self.Limp_lu = jax.vmap(jsp_linalg.lu_factor)(
            self._pin_pressure_modes(self.Limp_modes)
        )

    @staticmethod
    def _family_class(family: str):
        if family.startswith("L"):
            return Legendre
        if family.startswith("C"):
            return Chebyshev
        raise ValueError("family must be 'L' or 'C'")

    def _lap(self, u: sp.Expr) -> sp.Expr:
        return Dx(u, 1, 2) + Dx(u, 0, 2)

    @staticmethod
    def _dense(expr: sp.Expr) -> Array:
        mat = inner(expr, kind=InnerKind.BILINEAR)
        return jnp.asarray(mat.todense())

    @staticmethod
    def _mode_indices(space: CoupledSpace) -> Array:
        mode_shape = tuple(int(n) for n in space[0].num_dofs[:-1])
        per_mode_indices = []
        for flat_mode in range(int(np.prod(mode_shape)) if mode_shape else 1):
            mode = []
            for block, component in zip(space.block_slices, space, strict=True):
                radial_size = int(component.num_dofs[-1])
                start = int(block.start) + flat_mode * radial_size
                mode.append(jnp.arange(start, start + radial_size))
            per_mode_indices.append(jnp.concatenate(mode))
        return jnp.stack(per_mode_indices)

    @staticmethod
    def _extract_mode_matrices(A: Array, indices: Array) -> Array:
        return A[indices[:, :, None], indices[:, None, :]]

    @staticmethod
    def _scatter_modes(values: Array, indices: Array, size: int) -> Array:
        flat = jnp.zeros((size,), dtype=values.dtype)
        return flat.at[indices].set(values)

    @staticmethod
    def _put_block(A: Array, rows: slice, cols: slice, block: Array) -> Array:
        return A.at[rows, cols].add(block)

    def _add_form(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        i: int,
        j: int,
        expr: sp.Expr,
    ) -> Array:
        return self._put_block(
            A,
            test_space.block_slices[i],
            trial_space.block_slices[j],
            self._dense(expr),
        )

    def _linear_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        idx: dict[str, int],
        fields: dict[str, sp.Expr],
        tests: dict[str, sp.Expr],
        sign: float,
    ) -> Array:
        dz = lambda f: Dx(f, 0, 1)
        nu, eta, B0, S, omega = self.nu, self.eta_mag, self.B0, self.S, self.omega
        terms: list[tuple[str, str, sp.Expr]] = []

        def add(row: str, col: str, test: sp.Expr, coeff: float, expr: sp.Expr) -> None:
            factor = sign * float(coeff)
            if factor != 0.0:
                terms.append((row, col, test * (factor * expr)))

        add("ux", "ux", tests["ux"], nu, self._lap(fields["ux"]))
        add("ux", "uy", tests["ux"], 2.0 * omega, fields["uy"])
        add("ux", "bx", tests["ux"], B0, dz(fields["bx"]))
        add("uy", "uy", tests["uy"], nu, self._lap(fields["uy"]))
        add("uy", "ux", tests["uy"], S - 2.0 * omega, fields["ux"])
        add("uy", "by", tests["uy"], B0, dz(fields["by"]))
        add("uz", "uz", tests["uz"], nu, self._lap(fields["uz"]))
        add("uz", "bz", tests["uz"], B0, dz(fields["bz"]))
        add("bx", "bx", tests["bx"], eta, self._lap(fields["bx"]))
        add("bx", "ux", tests["bx"], B0, dz(fields["ux"]))
        add("by", "by", tests["by"], eta, self._lap(fields["by"]))
        add("by", "bx", tests["by"], -S, fields["bx"])
        add("by", "uy", tests["by"], B0, dz(fields["uy"]))
        add("bz", "bz", tests["bz"], eta, self._lap(fields["bz"]))
        add("bz", "uz", tests["bz"], B0, dz(fields["uz"]))
        for row, col, expr in terms:
            A = self._add_form(A, test_space, trial_space, idx[row], idx[col], expr)
        return A

    def _build_operators(self) -> tuple[Array, Array]:
        dt = self.dt
        dtype = jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))
        ux = TrialFunction(self.TD, name="ux")
        uy = TrialFunction(self.TD, name="uy")
        uz = TrialFunction(self.TD, name="uz")
        p = TrialFunction(self.TP, name="p")
        bx = TrialFunction(self.TD, name="bx")
        by = TrialFunction(self.TN, name="by")
        bz = TrialFunction(self.TN, name="bz")
        vx = TestFunction(self.TD, name="vx")
        vy = TestFunction(self.TD, name="vy")
        vz = TestFunction(self.TD, name="vz")
        q = TestFunction(self.TP, name="q")
        cx = TestFunction(self.TD, name="cx")
        cy = TestFunction(self.TN, name="cy")
        cz = TestFunction(self.TN, name="cz")
        idx_q = {"ux": 0, "uy": 1, "uz": 2, "p": 3, "bx": 4, "by": 5, "bz": 6}
        fields_q = {"ux": ux, "uy": uy, "uz": uz, "bx": bx, "by": by, "bz": bz}
        tests_q = {"ux": vx, "uy": vy, "uz": vz, "bx": cx, "by": cy, "bz": cz}

        Limp = jnp.zeros((self.VQ.dim, self.VQ.dim), dtype=dtype)
        for name in ("ux", "uy", "uz", "bx", "by", "bz"):
            Limp = self._add_form(
                Limp,
                self.VQ,
                self.VQ,
                idx_q[name],
                idx_q[name],
                tests_q[name] * fields_q[name] * (1.0 / dt),
            )
        Limp = self._linear_terms(
            Limp, self.VQ, self.VQ, idx_q, fields_q, tests_q, sign=-0.5
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 0, 3, vx * Dx(p, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ux, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 0, 1))

        eux = TrialFunction(self.TD, name="eux")
        euy = TrialFunction(self.TD, name="euy")
        euz = TrialFunction(self.TD, name="euz")
        ebx = TrialFunction(self.TD, name="ebx")
        eby = TrialFunction(self.TN, name="eby")
        ebz = TrialFunction(self.TN, name="ebz")
        tx = TestFunction(self.TD, name="tx")
        ty = TestFunction(self.TD, name="ty")
        tz = TestFunction(self.TD, name="tz")
        dx = TestFunction(self.TD, name="dx")
        dy = TestFunction(self.TN, name="dy")
        dz = TestFunction(self.TN, name="dz")
        idx_e = {"ux": 0, "uy": 1, "uz": 2, "bx": 3, "by": 4, "bz": 5}
        fields_e = {"ux": eux, "uy": euy, "uz": euz, "bx": ebx, "by": eby, "bz": ebz}
        tests_e = {"ux": tx, "uy": ty, "uz": tz, "bx": dx, "by": dy, "bz": dz}
        Lexp = jnp.zeros((self.VE.dim, self.VE.dim), dtype=dtype)
        for name in ("ux", "uy", "uz", "bx", "by", "bz"):
            Lexp = self._add_form(
                Lexp,
                self.VE,
                self.VE,
                idx_e[name],
                idx_e[name],
                tests_e[name] * fields_e[name] * (1.0 / dt),
            )
        Lexp = self._linear_terms(
            Lexp, self.VE, self.VE, idx_e, fields_e, tests_e, sign=0.5
        )
        return Limp, Lexp

    def _pin_pressure_modes(self, modes: Array) -> Array:
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        modes = modes.at[0, pressure_row, :].set(0)
        return modes.at[0, pressure_row, pressure_row].set(1)

    def _solve_limp(self, rhs: Array) -> Array:
        rhs_modes = rhs[self.VQ_mode_indices]
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        rhs_modes = rhs_modes.at[0, pressure_row].set(0)
        lu, piv = self.Limp_lu
        sol_modes = jax.vmap(
            lambda lu_i, piv_i, b_i: jsp_linalg.lu_solve((lu_i, piv_i), b_i)
        )(lu, piv, rhs_modes)
        return self._scatter_modes(sol_modes, self.VQ_mode_indices, self.VQ.dim)

    def zero_state(self) -> AxisymmetricPCFState:
        x = tuple(jnp.zeros(space.num_dofs, dtype=self.Limp.dtype) for space in self.VE)
        p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricPCFState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def state_from_physical(self, values: MHDFields) -> AxisymmetricPCFState:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        x = tuple(
            space.forward(value) for space, value in zip(spaces, values, strict=True)
        )
        p = jnp.zeros(self.TP.num_dofs, dtype=x[0].dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricPCFState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def _phys_mhd(self, coeff: Array, space) -> tuple[Array, Array, Array]:
        N = self.padded_counts
        value = space.backward(coeff, N=N)
        dx = space.backward_primitive(coeff, (0, 1), N=N)
        dz = space.backward_primitive(coeff, (1, 0), N=N)
        return value, dx, dz

    def _t0_coeff(self, values: Array) -> Array:
        if self.T0p is None:
            return self.T0.forward(values)
        return self.T0p.forward(values)

    def _dealias_to_standard(self, values: Array) -> Array:
        if self.T0p is None:
            return values
        coeff = self.T0p.forward(values)
        return self.T0.backward(coeff)

    def nonlinear(self, state: AxisymmetricPCFState) -> MHDFields:
        ux, uxx, uxz = self._phys_mhd(state.x[0], self.TD)
        uy, uyx, uyz = self._phys_mhd(state.x[1], self.TD)
        uz, uzx, uzz = self._phys_mhd(state.x[2], self.TD)
        bx, bxx, bxz = self._phys_mhd(state.x[3], self.TD)
        by, byx, byz = self._phys_mhd(state.x[4], self.TN)
        bz, bzx, bzz = self._phys_mhd(state.x[5], self.TN)
        nu_x = self._dealias_to_standard(ux * uxx + uz * uxz - bx * bxx - bz * bxz)
        nu_y = self._dealias_to_standard(ux * uyx + uz * uyz - bx * byx - bz * byz)
        nu_z = self._dealias_to_standard(ux * uzx + uz * uzz - bx * bzx - bz * bzz)

        eps_x = uy * bz - uz * by
        eps_y = uz * bx - ux * bz
        eps_z = ux * by - uy * bx
        ex_hat = self._t0_coeff(eps_x)
        ey_hat = self._t0_coeff(eps_y)
        ez_hat = self._t0_coeff(eps_z)
        nb_x = self.T0.backward_primitive(ey_hat, (1, 0))
        nb_y = -self.T0.backward_primitive(ex_hat, (1, 0)) + self.T0.backward_primitive(
            ez_hat, (0, 1)
        )
        nb_z = -self.T0.backward_primitive(ey_hat, (0, 1))
        return (
            self.TD.mask_nyquist(self.TD.scalar_product(nu_x)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_y)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_z)),
            self.TD.mask_nyquist(self.TD.scalar_product(nb_x)),
            self.TN.mask_nyquist(self.TN.scalar_product(nb_y)),
            self.TN.mask_nyquist(self.TN.scalar_product(nb_z)),
        )

    def _apply_lexp(self, x: MHDFields) -> MHDFields:
        flat = self.VE.flatten(x)
        modes = flat[self.VE_mode_indices]
        out_modes = jnp.einsum("kij,kj->ki", self.Lexp_modes, modes)
        out = self._scatter_modes(out_modes, self.VE_mode_indices, self.VE.dim)
        return self.VE.unflatten(out)  # ty: ignore[return-value]

    def step(self, state: AxisymmetricPCFState) -> AxisymmetricPCFState:
        n_hat = self.nonlinear(state)
        rhs_e = self._apply_lexp(state.x)
        rhs_x = cnab2_rhs(rhs_e, n_hat, state.nonlinear_old, state.have_old)
        rhs_p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        rhs = self.VQ.flatten((*rhs_x[:3], rhs_p, *rhs_x[3:]))
        sol = self.VQ.unflatten(self._solve_limp(rhs))
        x = (sol[0], sol[1], sol[2], sol[4], sol[5], sol[6])
        return AxisymmetricPCFState(x=x, p=sol[3], nonlinear_old=n_hat, have_old=True)

    def solve(self, state: AxisymmetricPCFState, steps: int) -> AxisymmetricPCFState:
        return scan_steps(self.step, state, steps)

    def solve_with_cadence(
        self,
        state: AxisymmetricPCFState,
        steps: int,
        cadence: Cadence,
        *,
        block_size: int = 1,
        on_diagnostics=None,
        on_snapshot=None,
        on_checkpoint=None,
        should_stop=None,
    ) -> AxisymmetricPCFState:
        return run_with_cadence(
            self.solve,
            state,
            steps=steps,
            dt=self.dt,
            cadence=cadence,
            block_size=block_size,
            diagnostics=self.diagnostics,
            on_diagnostics=on_diagnostics,
            on_snapshot=on_snapshot,
            on_checkpoint=on_checkpoint,
            should_stop=should_stop,
        )

    def _linear_operator(self) -> PlaneCouetteLinear:
        return PlaneCouetteLinear.shearpy(
            nx=self.Nx,
            Re=abs(self.S) / self.nu,
            Rm=abs(self.S) / self.eta_mag,
            shear_rate=self.S,
            omega=self.omega,
            by=0.0,
            bz=self.B0,
            velocity_scale=abs(self.S),
            magnetic_bc="conducting",
        )

    def _hydro_linear_operator(self) -> PlaneCouetteLinear:
        return PlaneCouetteLinear(
            nx=self.Nx,
            nu=self.nu,
            eta=self.eta_mag,
            Uprime=-self.S,
            omega=self.omega,
            mhd=False,
        )

    def _seed_from_linear(
        self,
        operator: PlaneCouetteLinear,
        names: tuple[str, ...],
        spaces: tuple[Any, ...],
        *,
        kz_mode: int,
        amp: float,
        which: int,
    ) -> tuple[AxisymmetricPCFState, complex]:
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        w, vectors = operator.eigs(0.0, kz, n_return=which + 1)
        vec = _positive_pivot_phase(vectors[:, which])
        blocks = operator._blocks()
        n = operator.nx
        x_dns = np.asarray(self.X[0, :]).ravel()
        interp = _bary_interp_matrix(operator.x, x_dns)
        z = np.asarray(self.Z[:, 0]).ravel()
        cos_z = np.cos(kz * z)
        sin_z = np.sin(kz * z)
        values = [jnp.zeros(self.T0.num_quad_points, dtype=self.Limp.real.dtype)] * 6
        name_to_slot = {"ux": 0, "uy": 1, "uz": 2, "bx": 3, "by": 4, "bz": 5}
        for name, _space in zip(names, spaces, strict=True):
            profile = np.asarray(vec[blocks[name] * n : (blocks[name] + 1) * n])
            profile_dns = interp @ profile
            phys = amp * (
                profile_dns.real[None, :] * cos_z[:, None]
                - profile_dns.imag[None, :] * sin_z[:, None]
            )
            values[name_to_slot[name]] = jnp.asarray(phys, dtype=self.Limp.real.dtype)
        return self.state_from_physical(tuple(values)), complex(w[which])

    def seed_linear_eigenmode(
        self, kz_mode: int = 1, amp: float = 1.0e-7, which: int = 0
    ) -> tuple[AxisymmetricPCFState, complex]:
        return self._seed_from_linear(
            self._linear_operator(),
            ("ux", "uy", "uz", "bx", "by", "bz"),
            (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN),
            kz_mode=kz_mode,
            amp=amp,
            which=which,
        )

    def seed_hydro_eigenmode(
        self, kz_mode: int = 1, amp: float = 1.0e-4, which: int = 0
    ) -> tuple[AxisymmetricPCFState, complex]:
        return self._seed_from_linear(
            self._hydro_linear_operator(),
            ("ux", "uy", "uz"),
            (self.TD, self.TD, self.TD),
            kz_mode=kz_mode,
            amp=amp,
            which=which,
        )

    def fields_physical(self, state: AxisymmetricPCFState) -> MHDFields:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        return tuple(
            space.backward(coeff) for space, coeff in zip(spaces, state.x, strict=True)
        )  # ty: ignore[return-value]

    def energy_parts(self, state: AxisymmetricPCFState) -> tuple[Array, Array]:
        fields = self.fields_physical(state)
        return (
            quadratic_energy(fields[:3], self.T0),
            quadratic_energy(fields[3:], self.T0),
        )

    def energy(self, state: AxisymmetricPCFState) -> Array:
        ek, em = self.energy_parts(state)
        return ek + em

    def velocity_divergence(self, state: AxisymmetricPCFState) -> Array:
        return self.TD.backward_primitive(
            state.x[0], (0, 1)
        ) + self.TD.backward_primitive(state.x[2], (1, 0))

    def magnetic_divergence(self, state: AxisymmetricPCFState) -> Array:
        return self.TD.backward_primitive(
            state.x[3], (0, 1)
        ) + self.TN.backward_primitive(state.x[5], (1, 0))

    def _l2(self, value: Array) -> Array:
        return jnp.sqrt(jnp.real(integrate(jnp.conj(value) * value, self.T0)))

    def diagnostics(self, state: AxisymmetricPCFState) -> dict[str, Array]:
        ek, em = self.energy_parts(state)
        return {
            "Ekin": ek,
            "Emag": em,
            "E": ek + em,
            "divu": self._l2(self.velocity_divergence(state)),
            "divb": self._l2(self.magnetic_divergence(state)),
        }

    def growth_rate(
        self, state: AxisymmetricPCFState, steps: int
    ) -> tuple[Array, AxisymmetricPCFState]:
        e0 = self.energy(state)
        out = self.solve(state, steps)
        e1 = self.energy(out)
        elapsed = int(steps) * self.dt
        return 0.5 * jnp.log(e1 / e0) / elapsed, out


class PCFMRIDNSJax:
    """Full 3D primitive-variable plane-Couette / shearbox MRI DNS.

    Field arrays are ordered ``(y, z, x)``: streamwise Fourier, vertical Fourier,
    and wall-normal Galerkin.  The evolved variables are
    ``(u_x,u_y,u_z,b_x,b_y,b_z)`` plus a pressure saddle point.  This is the 3D
    counterpart of :class:`AxisymmetricPCFMRIDNSJax` and follows the vendored
    shenfun ``PCFMRIDNS`` reference.
    """

    def __init__(
        self,
        S: float = 1.0,
        omega: float = 2.0 / 3.0,
        B0: float = 0.1,
        nu: float = 1.0e-3,
        eta_mag: float = 1.0e-3,
        Nx: int = 40,
        Ny: int = 8,
        Nz: int = 16,
        Ly: float = 2.0 * math.pi,
        Lz: float = 1.0,
        dt: float = 2.0e-3,
        family: str = "C",
        dealias: float | tuple[float, float, float] = 1.0,
    ) -> None:
        self.S = float(S)
        self.omega = float(omega)
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = _dealias_tuple(dealias, 3)

        family_cls = AxisymmetricPCFMRIDNSJax._family_class(self.family)
        dom = Domain(-1.0, 1.0)
        self.Fy = FunctionSpace(
            self.Ny, Fourier, domain=Domain(0.0, self.Ly), name="Fy"
        )
        self.Fz = FunctionSpace(
            self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="Fz"
        )
        self.SD = FunctionSpace(self.Nx, family_cls, bc=(0, 0), domain=dom, name="SD3")
        self.S0 = FunctionSpace(self.Nx, family_cls, domain=dom, name="S03")
        self.SP = FunctionSpace(
            self.Nx, family_cls, domain=dom, num_dofs=self.Nx - 2, name="SP3"
        )
        self.SN = FunctionSpace(
            self.Nx,
            family_cls,
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=dom,
            name="SN3",
        )

        self.TD = TensorProduct(self.Fy, self.Fz, self.SD, name="TDpcf3")
        self.T0 = TensorProduct(self.Fy, self.Fz, self.S0, name="T0pcf3")
        self.TP = TensorProduct(self.Fy, self.Fz, self.SP, name="TPpcf3")
        self.TN = TensorProduct(self.Fy, self.Fz, self.SN, name="TNpcf3")
        self.VQ = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TP, self.TD, self.TN, self.TN),
            name="VQpcf3",
        )
        self.VE = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN),
            name="VEpcf3",
        )

        self.ycoord, self.zcoord, self.xcoord = self.TD.system.base_scalars()
        self.VQ_mode_indices = AxisymmetricPCFMRIDNSJax._mode_indices(self.VQ)
        self.VE_mode_indices = AxisymmetricPCFMRIDNSJax._mode_indices(self.VE)
        self.Y, self.Z, self.X = self.T0.mesh()
        if any(value > 1.0 for value in self.dealias):
            self.T0p = self.T0.get_dealiased(self.dealias)
            self.padded_counts = self.T0p.num_quad_points
        else:
            self.T0p = None
            self.padded_counts = None

        self.Limp = jnp.zeros((0, 0), dtype=self._operator_dtype())
        self.Lexp = jnp.zeros((0, 0), dtype=self.Limp.dtype)
        self.Limp_modes, self.Lexp_modes = self._build_operator_modes()
        self.Limp_lu = jax.vmap(jsp_linalg.lu_factor)(
            self._pin_pressure_modes(self.Limp_modes)
        )

    @staticmethod
    def _operator_dtype() -> jnp.dtype:
        return jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))

    @staticmethod
    def _mode_block_slices(space: CoupledSpace) -> tuple[slice, ...]:
        starts = [0]
        sizes = [int(component.num_dofs[-1]) for component in space]
        for size in sizes[:-1]:
            starts.append(starts[-1] + size)
        return tuple(
            slice(start, start + size)
            for start, size in zip(starts, sizes, strict=True)
        )

    @staticmethod
    def _mode_size(space: CoupledSpace) -> int:
        return sum(int(component.num_dofs[-1]) for component in space)

    @staticmethod
    def _mode_shape(space: CoupledSpace) -> tuple[int, ...]:
        return tuple(int(n) for n in space[0].num_dofs[:-1])

    @staticmethod
    def _tp_terms(matrix: Any) -> list[TPMatrix]:
        if isinstance(matrix, TPMatrix):
            return [matrix]
        if isinstance(matrix, TPMatrices):
            return list(matrix.tpmats)
        raise TypeError(f"expected tensor-product matrix, got {type(matrix).__name__}")

    @staticmethod
    def _dense_factor(matrix: Any) -> np.ndarray:
        return np.asarray(matrix.todense())

    @classmethod
    def _mode_blocks_from_expr(
        cls, expr: sp.Expr, mode_shape: tuple[int, ...]
    ) -> Array:
        assembled = inner(expr, sparse=True, kind=InnerKind.BILINEAR)
        terms = cls._tp_terms(assembled)
        if not terms:
            raise ValueError("empty tensor-product operator")
        if len(mode_shape) != 2:
            raise ValueError(f"expected 2 Fourier mode axes, got {mode_shape}")

        first_radial = cls._dense_factor(terms[0].mats[-1])
        n_modes = int(np.prod(mode_shape))
        out = np.zeros(
            (n_modes, first_radial.shape[0], first_radial.shape[1]),
            dtype=np.result_type(first_radial, np.complex64),
        )
        for term in terms:
            if len(term.mats) != 3:
                raise ValueError(f"expected 3 tensor factors, got {len(term.mats)}")
            y_factor = cls._dense_factor(term.mats[0])
            z_factor = cls._dense_factor(term.mats[1])
            radial = cls._dense_factor(term.mats[2])
            for flat_mode, (iy, iz) in enumerate(np.ndindex(mode_shape)):
                coeff = (
                    np.asarray(term.coefficient) * y_factor[iy, iy] * z_factor[iz, iz]
                )
                if coeff != 0:
                    out[flat_mode] += coeff * radial
        return jnp.asarray(out)

    def _zero_mode_operator(self, space: CoupledSpace) -> Array:
        mode_shape = self._mode_shape(space)
        return jnp.zeros(
            (
                int(np.prod(mode_shape)),
                self._mode_size(space),
                self._mode_size(space),
            ),
            dtype=self.Limp.dtype,
        )

    def _put_mode_block(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        i: int,
        j: int,
        expr: sp.Expr,
    ) -> Array:
        test_mode_shape = self._mode_shape(test_space)
        if test_mode_shape != self._mode_shape(trial_space):
            raise ValueError("test and trial mode shapes must match")
        rows = self._mode_block_slices(test_space)[i]
        cols = self._mode_block_slices(trial_space)[j]
        block = self._mode_blocks_from_expr(expr, test_mode_shape)
        return A.at[:, rows, cols].add(block)

    def _lap(self, u: sp.Expr) -> sp.Expr:
        return Dx(u, 2, 2) + Dx(u, 0, 2) + Dx(u, 1, 2)

    def _linear_mode_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        idx: dict[str, int],
        fields: dict[str, sp.Expr],
        tests: dict[str, sp.Expr],
        sign: float,
    ) -> Array:
        dy = lambda f: Dx(f, 0, 1)
        dz = lambda f: Dx(f, 1, 1)
        adv = lambda f: self.S * self.xcoord * dy(f)
        nu, eta, B0, S, omega = self.nu, self.eta_mag, self.B0, self.S, self.omega
        terms: list[tuple[str, str, sp.Expr]] = []

        def add(row: str, col: str, test: sp.Expr, coeff: float, expr: sp.Expr) -> None:
            factor = sign * float(coeff)
            if factor != 0.0:
                terms.append((row, col, test * (factor * expr)))

        for name, test in (
            ("ux", tests["ux"]),
            ("uy", tests["uy"]),
            ("uz", tests["uz"]),
            ("bx", tests["bx"]),
            ("by", tests["by"]),
            ("bz", tests["bz"]),
        ):
            diffusivity = nu if name.startswith("u") else eta
            add(name, name, test, diffusivity, self._lap(fields[name]))
            add(name, name, test, 1.0, adv(fields[name]))

        add("ux", "uy", tests["ux"], 2.0 * omega, fields["uy"])
        add("ux", "bx", tests["ux"], B0, dz(fields["bx"]))
        add("uy", "ux", tests["uy"], S - 2.0 * omega, fields["ux"])
        add("uy", "by", tests["uy"], B0, dz(fields["by"]))
        add("uz", "bz", tests["uz"], B0, dz(fields["bz"]))
        add("bx", "ux", tests["bx"], B0, dz(fields["ux"]))
        add("by", "bx", tests["by"], -S, fields["bx"])
        add("by", "uy", tests["by"], B0, dz(fields["uy"]))
        add("bz", "uz", tests["bz"], B0, dz(fields["uz"]))
        for row, col, expr in terms:
            A = self._put_mode_block(
                A,
                test_space,
                trial_space,
                idx[row],
                idx[col],
                expr,
            )
        return A

    def _build_operator_modes(self) -> tuple[Array, Array]:
        dt = self.dt
        ux = TrialFunction(self.TD, name="ux3")
        uy = TrialFunction(self.TD, name="uy3")
        uz = TrialFunction(self.TD, name="uz3")
        p = TrialFunction(self.TP, name="p3")
        bx = TrialFunction(self.TD, name="bx3")
        by = TrialFunction(self.TN, name="by3")
        bz = TrialFunction(self.TN, name="bz3")
        vx = TestFunction(self.TD, name="vx3")
        vy = TestFunction(self.TD, name="vy3")
        vz = TestFunction(self.TD, name="vz3")
        q = TestFunction(self.TP, name="q3")
        cx = TestFunction(self.TD, name="cx3")
        cy = TestFunction(self.TN, name="cy3")
        cz = TestFunction(self.TN, name="cz3")
        idx_q = {"ux": 0, "uy": 1, "uz": 2, "p": 3, "bx": 4, "by": 5, "bz": 6}
        fields_q = {"ux": ux, "uy": uy, "uz": uz, "bx": bx, "by": by, "bz": bz}
        tests_q = {"ux": vx, "uy": vy, "uz": vz, "bx": cx, "by": cy, "bz": cz}

        Limp = self._zero_mode_operator(self.VQ)
        for name in ("ux", "uy", "uz", "bx", "by", "bz"):
            Limp = self._put_mode_block(
                Limp,
                self.VQ,
                self.VQ,
                idx_q[name],
                idx_q[name],
                tests_q[name] * fields_q[name] * (1.0 / dt),
            )
        Limp = self._linear_mode_terms(
            Limp, self.VQ, self.VQ, idx_q, fields_q, tests_q, sign=-0.5
        )
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 0, 3, vx * Dx(p, 2, 1))
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 1, 3, vy * Dx(p, 0, 1))
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 1, 1))
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ux, 2, 1))
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 3, 1, q * Dx(uy, 0, 1))
        Limp = self._put_mode_block(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 1, 1))

        eux = TrialFunction(self.TD, name="eux3")
        euy = TrialFunction(self.TD, name="euy3")
        euz = TrialFunction(self.TD, name="euz3")
        ebx = TrialFunction(self.TD, name="ebx3")
        eby = TrialFunction(self.TN, name="eby3")
        ebz = TrialFunction(self.TN, name="ebz3")
        tx = TestFunction(self.TD, name="tx3")
        ty = TestFunction(self.TD, name="ty3")
        tz = TestFunction(self.TD, name="tz3")
        dx = TestFunction(self.TD, name="dx3")
        dy = TestFunction(self.TN, name="dy3")
        dz = TestFunction(self.TN, name="dz3")
        idx_e = {"ux": 0, "uy": 1, "uz": 2, "bx": 3, "by": 4, "bz": 5}
        fields_e = {"ux": eux, "uy": euy, "uz": euz, "bx": ebx, "by": eby, "bz": ebz}
        tests_e = {"ux": tx, "uy": ty, "uz": tz, "bx": dx, "by": dy, "bz": dz}
        Lexp = self._zero_mode_operator(self.VE)
        for name in ("ux", "uy", "uz", "bx", "by", "bz"):
            Lexp = self._put_mode_block(
                Lexp,
                self.VE,
                self.VE,
                idx_e[name],
                idx_e[name],
                tests_e[name] * fields_e[name] * (1.0 / dt),
            )
        Lexp = self._linear_mode_terms(
            Lexp, self.VE, self.VE, idx_e, fields_e, tests_e, sign=0.5
        )
        return Limp, Lexp

    def _pin_pressure_modes(self, modes: Array) -> Array:
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        modes = modes.at[0, pressure_row, :].set(0)
        return modes.at[0, pressure_row, pressure_row].set(1)

    def _solve_limp(self, rhs: Array) -> Array:
        rhs_modes = rhs[self.VQ_mode_indices]
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        rhs_modes = rhs_modes.at[0, pressure_row].set(0)
        lu, piv = self.Limp_lu
        sol_modes = jax.vmap(
            lambda lu_i, piv_i, b_i: jsp_linalg.lu_solve((lu_i, piv_i), b_i)
        )(lu, piv, rhs_modes)
        return AxisymmetricPCFMRIDNSJax._scatter_modes(
            sol_modes, self.VQ_mode_indices, self.VQ.dim
        )

    def zero_state(self) -> AxisymmetricPCFState:
        x = tuple(jnp.zeros(space.num_dofs, dtype=self.Limp.dtype) for space in self.VE)
        p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricPCFState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def state_from_physical(self, values: MHDFields) -> AxisymmetricPCFState:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        x = tuple(
            space.forward(value) for space, value in zip(spaces, values, strict=True)
        )
        p = jnp.zeros(self.TP.num_dofs, dtype=x[0].dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricPCFState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def _phys_mhd(self, coeff: Array, space) -> tuple[Array, Array, Array, Array]:
        N = self.padded_counts
        value = space.backward(coeff, N=N)
        dx = space.backward_primitive(coeff, (0, 0, 1), N=N)
        dy = space.backward_primitive(coeff, (1, 0, 0), N=N)
        dz = space.backward_primitive(coeff, (0, 1, 0), N=N)
        return value, dx, dy, dz

    def _t0_coeff(self, values: Array) -> Array:
        if self.T0p is None:
            return self.T0.forward(values)
        return self.T0p.forward(values)

    def _dealias_to_standard(self, values: Array) -> Array:
        if self.T0p is None:
            return values
        coeff = self.T0p.forward(values)
        return self.T0.backward(coeff)

    def nonlinear(self, state: AxisymmetricPCFState) -> MHDFields:
        ux, uxx, uxy, uxz = self._phys_mhd(state.x[0], self.TD)
        uy, uyx, uyy, uyz = self._phys_mhd(state.x[1], self.TD)
        uz, uzx, uzy, uzz = self._phys_mhd(state.x[2], self.TD)
        bx, bxx, bxy, bxz = self._phys_mhd(state.x[3], self.TD)
        by, byx, byy, byz = self._phys_mhd(state.x[4], self.TN)
        bz, bzx, bzy, bzz = self._phys_mhd(state.x[5], self.TN)
        nu_x = self._dealias_to_standard(
            ux * uxx + uy * uxy + uz * uxz - bx * bxx - by * bxy - bz * bxz
        )
        nu_y = self._dealias_to_standard(
            ux * uyx + uy * uyy + uz * uyz - bx * byx - by * byy - bz * byz
        )
        nu_z = self._dealias_to_standard(
            ux * uzx + uy * uzy + uz * uzz - bx * bzx - by * bzy - bz * bzz
        )
        eps_x = uy * bz - uz * by
        eps_y = uz * bx - ux * bz
        eps_z = ux * by - uy * bx
        ex_hat = self._t0_coeff(eps_x)
        ey_hat = self._t0_coeff(eps_y)
        ez_hat = self._t0_coeff(eps_z)
        nb_x = self.T0.backward_primitive(
            ey_hat, (0, 1, 0)
        ) - self.T0.backward_primitive(ez_hat, (1, 0, 0))
        nb_y = self.T0.backward_primitive(
            ez_hat, (0, 0, 1)
        ) - self.T0.backward_primitive(ex_hat, (0, 1, 0))
        nb_z = self.T0.backward_primitive(
            ex_hat, (1, 0, 0)
        ) - self.T0.backward_primitive(ey_hat, (0, 0, 1))
        return (
            self.TD.mask_nyquist(self.TD.scalar_product(nu_x)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_y)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_z)),
            self.TD.mask_nyquist(self.TD.scalar_product(nb_x)),
            self.TN.mask_nyquist(self.TN.scalar_product(nb_y)),
            self.TN.mask_nyquist(self.TN.scalar_product(nb_z)),
        )

    def _apply_lexp(self, x: MHDFields) -> MHDFields:
        flat = self.VE.flatten(x)
        modes = flat[self.VE_mode_indices]
        out_modes = jnp.einsum("kij,kj->ki", self.Lexp_modes, modes)
        out = AxisymmetricPCFMRIDNSJax._scatter_modes(
            out_modes, self.VE_mode_indices, self.VE.dim
        )
        return self.VE.unflatten(out)  # ty: ignore[return-value]

    def step(self, state: AxisymmetricPCFState) -> AxisymmetricPCFState:
        n_hat = self.nonlinear(state)
        rhs_e = self._apply_lexp(state.x)
        rhs_x = cnab2_rhs(rhs_e, n_hat, state.nonlinear_old, state.have_old)
        rhs_p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        rhs = self.VQ.flatten((*rhs_x[:3], rhs_p, *rhs_x[3:]))
        sol = self.VQ.unflatten(self._solve_limp(rhs))
        x = (sol[0], sol[1], sol[2], sol[4], sol[5], sol[6])
        return AxisymmetricPCFState(x=x, p=sol[3], nonlinear_old=n_hat, have_old=True)

    def solve(self, state: AxisymmetricPCFState, steps: int) -> AxisymmetricPCFState:
        return scan_steps(self.step, state, steps)

    def solve_with_cadence(
        self,
        state: AxisymmetricPCFState,
        steps: int,
        cadence: Cadence,
        *,
        block_size: int = 1,
        on_diagnostics=None,
        on_snapshot=None,
        on_checkpoint=None,
        should_stop=None,
    ) -> AxisymmetricPCFState:
        return run_with_cadence(
            self.solve,
            state,
            steps=steps,
            dt=self.dt,
            cadence=cadence,
            block_size=block_size,
            diagnostics=self.diagnostics,
            on_diagnostics=on_diagnostics,
            on_snapshot=on_snapshot,
            on_checkpoint=on_checkpoint,
            should_stop=should_stop,
        )

    def _linear_operator(self) -> PlaneCouetteLinear:
        return PlaneCouetteLinear.shearpy(
            nx=self.Nx,
            Re=abs(self.S) / self.nu,
            Rm=abs(self.S) / self.eta_mag,
            shear_rate=self.S,
            omega=self.omega,
            by=0.0,
            bz=self.B0,
            velocity_scale=abs(self.S),
            magnetic_bc="conducting",
        )

    def seed_linear_eigenmode(
        self, ky_mode: int = 1, kz_mode: int = 1, amp: float = 1.0e-7, which: int = 0
    ) -> tuple[AxisymmetricPCFState, complex]:
        ky = 2.0 * math.pi * int(ky_mode) / self.Ly
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = self._linear_operator()
        w, vectors = lin.eigs(ky, kz, n_return=which + 1)
        vec = _positive_pivot_phase(vectors[:, which])
        blocks = lin._blocks()
        n = self.Nx
        x_dns = np.asarray(self.X[0, 0, :]).ravel()
        interp = _bary_interp_matrix(lin.x, x_dns)
        carg = ky * np.asarray(self.Y) + kz * np.asarray(self.Z)
        cos_mode = np.cos(carg)
        sin_mode = np.sin(carg)
        values = [jnp.zeros(self.T0.num_quad_points, dtype=self.Limp.real.dtype)] * 6
        name_to_slot = {"ux": 0, "uy": 1, "uz": 2, "bx": 3, "by": 4, "bz": 5}
        for name in ("ux", "uy", "uz", "bx", "by", "bz"):
            profile = np.asarray(vec[blocks[name] * n : (blocks[name] + 1) * n])
            profile_dns = interp @ profile
            phys = amp * (
                profile_dns.real[None, None, :] * cos_mode
                - profile_dns.imag[None, None, :] * sin_mode
            )
            values[name_to_slot[name]] = jnp.asarray(phys, dtype=self.Limp.real.dtype)
        return self.state_from_physical(tuple(values)), complex(w[which])

    def fields_physical(self, state: AxisymmetricPCFState) -> MHDFields:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.TN, self.TN)
        return tuple(
            space.backward(coeff) for space, coeff in zip(spaces, state.x, strict=True)
        )  # ty: ignore[return-value]

    def energy_parts(self, state: AxisymmetricPCFState) -> tuple[Array, Array]:
        fields = self.fields_physical(state)
        return (
            quadratic_energy(fields[:3], self.T0),
            quadratic_energy(fields[3:], self.T0),
        )

    def energy(self, state: AxisymmetricPCFState) -> Array:
        ek, em = self.energy_parts(state)
        return ek + em

    def velocity_divergence(self, state: AxisymmetricPCFState) -> Array:
        return (
            self.TD.backward_primitive(state.x[0], (0, 0, 1))
            + self.TD.backward_primitive(state.x[1], (1, 0, 0))
            + self.TD.backward_primitive(state.x[2], (0, 1, 0))
        )

    def magnetic_divergence(self, state: AxisymmetricPCFState) -> Array:
        return (
            self.TD.backward_primitive(state.x[3], (0, 0, 1))
            + self.TN.backward_primitive(state.x[4], (1, 0, 0))
            + self.TN.backward_primitive(state.x[5], (0, 1, 0))
        )

    def _l2(self, value: Array) -> Array:
        return jnp.sqrt(jnp.real(integrate(jnp.conj(value) * value, self.T0)))

    def stresses(self, state: AxisymmetricPCFState) -> tuple[Array, Array]:
        fields = self.fields_physical(state)
        ux, uy = fields[0], fields[1]
        bx, by = fields[3], fields[4]
        volume = self.Ly * self.Lz * 2.0
        reynolds = integrate(jnp.real(ux * jnp.conj(uy)), self.T0) / volume
        maxwell = -integrate(jnp.real(bx * jnp.conj(by)), self.T0) / volume
        return reynolds, maxwell

    def diagnostics(self, state: AxisymmetricPCFState) -> dict[str, Array]:
        ek, em = self.energy_parts(state)
        reynolds, maxwell = self.stresses(state)
        return {
            "Ekin": ek,
            "Emag": em,
            "E": ek + em,
            "divu": self._l2(self.velocity_divergence(state)),
            "divb": self._l2(self.magnetic_divergence(state)),
            "reynolds_stress": reynolds,
            "maxwell_stress": maxwell,
            "transport_alpha": (reynolds + maxwell) / (self.B0 * self.B0)
            if self.B0 != 0.0
            else jnp.asarray(jnp.nan),
        }

    def growth_rate(
        self, state: AxisymmetricPCFState, steps: int
    ) -> tuple[Array, AxisymmetricPCFState]:
        e0 = self.energy(state)
        out = self.solve(state, steps)
        e1 = self.energy(out)
        elapsed = int(steps) * self.dt
        return 0.5 * jnp.log(e1 / e0) / elapsed, out
