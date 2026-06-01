"""Axisymmetric Taylor-Couette DNS using jaxfun Galerkin blocks.

This is the JAX counterpart of ``couette/taylor_couette_dns.py`` for the
axisymmetric hydrodynamic perturbation equations.  It keeps the shenfun
formulation deliberately: Cartesian tensor-product spaces, explicit cylindrical
``1/r`` factors in the weak forms, Dirichlet velocity modes, and a truncated
orthogonal pressure space for the ``P_N/P_{N-2}`` pair.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import sympy as sp
from jax import Array

try:
    from examples.taylor_couette_linear_jax import (
        CircularCouette,
        TaylorCouetteLinearJax,
    )
except ModuleNotFoundError:  # direct script execution from examples/
    from taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax

from jaxfun import Domain, Dx
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

type Velocity = tuple[Array, Array, Array]


@dataclass(frozen=True)
class AxisymmetricTCState:
    """Coefficient state for the axisymmetric TC DNS solver."""

    u: Velocity
    p: Array
    nonlinear_old: Velocity
    have_old: bool = False


class AxisymmetricTCDNSJax:
    """Axisymmetric perturbation Taylor-Couette DNS.

    References:
      * ``couette/taylor_couette_dns.py:100-345`` for the shenfun solver.
      * ``COUETTE_IMPLEMENTATION_PLAN.md`` M6 for the JAX port requirements.

    The coupled block matrices are assembled from scalar jaxfun ``inner`` calls,
    extracted into independent axial Fourier-mode blocks, and solved by a
    batched pinned LU factorization.
    """

    def __init__(
        self,
        base: CircularCouette,
        nu: float = 1.0e-2,
        Nr: int = 32,
        Nz: int = 16,
        Lz: float | None = None,
        dt: float = 2.0e-3,
        family: str = "L",
        dealias: float = 1.5,
    ) -> None:
        self.base = base
        self.nu = float(nu)
        self.Nr = int(Nr)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.13 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu

        family_cls = self._family_class(self.family)
        dom = Domain(base.R1, base.R2)
        self.F = FunctionSpace(self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="F")
        self.SD = FunctionSpace(self.Nr, family_cls, bc=(0, 0), domain=dom, name="SD")
        self.S0 = FunctionSpace(self.Nr, family_cls, domain=dom, name="S0")
        self.SP = FunctionSpace(
            self.Nr, family_cls, domain=dom, num_dofs=self.Nr - 2, name="SP"
        )

        self.TD = TensorProduct(self.F, self.SD, name="TD")
        self.T0 = TensorProduct(self.F, self.S0, name="T0")
        self.TP = TensorProduct(self.F, self.SP, name="TP")
        self.VV = CoupledSpace((self.TD, self.TD, self.TD), name="VV")
        self.VQ = CoupledSpace((self.TD, self.TD, self.TD, self.TP), name="VQ")

        self.z, self.r = self.TD.system.base_scalars()
        self.pressure_pin = self.VQ.block_slices[3].start
        self.VQ_mode_indices = self._mode_indices(self.VQ)
        self.VV_mode_indices = self._mode_indices(self.VV)
        self.Z, self.R = self.T0.mesh()
        self.inv_r = 1.0 / self.R
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
            Zp, Rp = self.T0p.mesh()
            self.inv_r_p = 1.0 / Rp
        else:
            self.T0p = None
            self.padded_counts = None
            self.inv_r_p = self.inv_r

        self.Limp, self.Lexp = self._build_operators()
        self.Limp_modes = self._extract_mode_matrices(self.Limp, self.VQ_mode_indices)
        self.Lexp_modes = self._extract_mode_matrices(self.Lexp, self.VV_mode_indices)
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
        r = self.r
        return Dx(u, 1, 2) + (1 / r) * Dx(u, 1, 1) + Dx(u, 0, 2)

    @staticmethod
    def _dense(expr: sp.Expr) -> Array:
        mat = inner(expr, kind=InnerKind.BILINEAR)
        return jnp.asarray(mat.todense())

    @staticmethod
    def _mode_indices(space: CoupledSpace) -> Array:
        nz = int(space[0].num_dofs[0])
        per_mode_indices = []
        for k in range(nz):
            mode = []
            for block, component in zip(space.block_slices, space, strict=True):
                shape = component.num_dofs
                component_size = math.prod(shape[1:])
                start = int(block.start) + k * component_size
                mode.append(jnp.arange(start, start + component_size))
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

    def _build_operators(self) -> tuple[Array, Array]:
        """Assemble CNAB2 implicit/explicit matrices.

        Reference: ``couette/taylor_couette_dns.py:182-236``.
        """
        r = self.r
        dt = self.dt
        nu = self.nu
        a = self.base.a
        two_omega = 2 * (self.base.a + self.base.b / r**2)
        dtype = jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))

        ur = TrialFunction(self.TD, name="ur")
        ut = TrialFunction(self.TD, name="ut")
        uz = TrialFunction(self.TD, name="uz")
        p = TrialFunction(self.TP, name="p")
        vr = TestFunction(self.TD, name="vr")
        vt = TestFunction(self.TD, name="vt")
        vz = TestFunction(self.TD, name="vz")
        q = TestFunction(self.TP, name="q")

        Limp = jnp.zeros((self.VQ.dim, self.VQ.dim), dtype=dtype)
        for i, v, u in ((0, vr, ur), (1, vt, ut), (2, vz, uz)):
            Limp = self._add_form(Limp, self.VQ, self.VQ, i, i, v * u * (1.0 / dt))

        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 0, 0, vr * (-0.5 * nu * self._lap(ur))
        )
        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 0, 0, vr * (0.5 * nu * (1 / r**2) * ur)
        )
        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 0, 1, vr * (-0.5 * two_omega * ut)
        )
        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 1, 1, vt * (-0.5 * nu * self._lap(ut))
        )
        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 1, 1, vt * (0.5 * nu * (1 / r**2) * ut)
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 1, 0, vt * (-0.5 * (-2 * a) * ur))
        Limp = self._add_form(
            Limp, self.VQ, self.VQ, 2, 2, vz * (-0.5 * nu * self._lap(uz))
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 0, 3, vr * Dx(p, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ur, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * (1 / r) * ur)
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 0, 1))

        er = TrialFunction(self.TD, name="er")
        et = TrialFunction(self.TD, name="et")
        ez = TrialFunction(self.TD, name="ez")
        tr = TestFunction(self.TD, name="tr")
        tt = TestFunction(self.TD, name="tt")
        tz = TestFunction(self.TD, name="tz")
        Lexp = jnp.zeros((self.VV.dim, self.VV.dim), dtype=dtype)
        for i, v, u in ((0, tr, er), (1, tt, et), (2, tz, ez)):
            Lexp = self._add_form(Lexp, self.VV, self.VV, i, i, v * u * (1.0 / dt))

        Lexp = self._add_form(
            Lexp, self.VV, self.VV, 0, 0, tr * (0.5 * nu * self._lap(er))
        )
        Lexp = self._add_form(
            Lexp, self.VV, self.VV, 0, 0, tr * (-0.5 * nu * (1 / r**2) * er)
        )
        Lexp = self._add_form(Lexp, self.VV, self.VV, 0, 1, tr * (0.5 * two_omega * et))
        Lexp = self._add_form(
            Lexp, self.VV, self.VV, 1, 1, tt * (0.5 * nu * self._lap(et))
        )
        Lexp = self._add_form(
            Lexp, self.VV, self.VV, 1, 1, tt * (-0.5 * nu * (1 / r**2) * et)
        )
        Lexp = self._add_form(Lexp, self.VV, self.VV, 1, 0, tt * (0.5 * (-2 * a) * er))
        Lexp = self._add_form(
            Lexp, self.VV, self.VV, 2, 2, tz * (0.5 * nu * self._lap(ez))
        )
        return Limp, Lexp

    def _pin_pressure_modes(self, modes: Array) -> Array:
        """Pin the ``k=0`` pressure constant in the mode-local system."""
        pressure_row = sum(math.prod(space.num_dofs[1:]) for space in self.VQ[:3])
        modes = modes.at[0, pressure_row, :].set(0)
        return modes.at[0, pressure_row, pressure_row].set(1)

    def _solve_limp(self, rhs: Array) -> Array:
        rhs_modes = rhs[self.VQ_mode_indices]
        pressure_row = sum(math.prod(space.num_dofs[1:]) for space in self.VQ[:3])
        rhs_modes = rhs_modes.at[0, pressure_row].set(0)
        lu, piv = self.Limp_lu
        sol_modes = jax.vmap(
            lambda lu_i, piv_i, b_i: jsp_linalg.lu_solve((lu_i, piv_i), b_i)
        )(lu, piv, rhs_modes)
        return self._scatter_modes(sol_modes, self.VQ_mode_indices, self.VQ.dim)

    def zero_state(self) -> AxisymmetricTCState:
        u = tuple(jnp.zeros(space.num_dofs, dtype=self.Limp.dtype) for space in self.VV)
        p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        nold = tuple(jnp.zeros_like(ui) for ui in u)
        return AxisymmetricTCState(u=u, p=p, nonlinear_old=nold, have_old=False)

    def state_from_physical(self, values: Velocity) -> AxisymmetricTCState:
        u = tuple(self.TD.forward(value) for value in values)
        p = jnp.zeros(self.TP.num_dofs, dtype=u[0].dtype)
        nold = tuple(jnp.zeros_like(ui) for ui in u)
        return AxisymmetricTCState(u=u, p=p, nonlinear_old=nold, have_old=False)

    def initial_state(
        self, amp: float = 1.0e-3, kz_mode: int = 1
    ) -> AxisymmetricTCState:
        """Return the divergence-free streamfunction perturbation seed.

        Reference: ``couette/taylor_couette_dns.py:318-344``.
        """
        R1 = self.base.R1
        d = self.base.gap
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        arg = math.pi * (self.R - R1) / d
        g = jnp.sin(arg) ** 2
        gp = (2.0 * math.pi / d) * jnp.sin(arg) * jnp.cos(arg)
        ur = amp * (kz / self.R) * g * jnp.sin(kz * self.Z)
        uz = amp * (1.0 / self.R) * gp * jnp.cos(kz * self.Z)
        ut = amp * jnp.sin(arg) * jnp.cos(kz * self.Z)
        return self.state_from_physical((ur, ut, uz))

    def seed_linear_eigenmode(
        self, kz_mode: int = 1, amp: float = 1.0e-6, which: int = 0
    ) -> tuple[AxisymmetricTCState, complex]:
        """Seed the real part of a hydrodynamic linear eigenmode.

        The coefficients come from ``TaylorCouetteLinearJax`` on matching radial
        spaces.  In the full-complex Fourier layout we write both ``+k`` and
        ``-k`` coefficients so the physical perturbation is real-valued.
        """
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = TaylorCouetteLinearJax(
            self.base, nu=self.nu, N=self.Nr, family=self.family
        )
        w, vecs = lin.eigs(m=0, kz=kz, n_return=which + 1)
        vec = vecs[:, which]
        n = lin.n
        state = self.zero_state()
        comps = list(state.u)
        kpos = int(kz_mode) % self.Nz
        kneg = (-int(kz_mode)) % self.Nz
        for comp in range(3):
            block = jnp.asarray(vec[comp * n : (comp + 1) * n]) * amp
            arr = jnp.zeros_like(comps[comp])
            if kpos == kneg:
                arr = arr.at[kpos, :n].set(jnp.real(block))
            else:
                arr = arr.at[kpos, :n].set(0.5 * block)
                arr = arr.at[kneg, :n].set(0.5 * jnp.conj(block))
            comps[comp] = arr
        nold = tuple(jnp.zeros_like(ui) for ui in comps)
        return AxisymmetricTCState(tuple(comps), state.p, nold, False), complex(
            w[which]
        )

    def _dealias_to_standard(self, values: Array) -> Array:
        if self.T0p is None:
            return values
        coeff = self.T0p.forward(values)
        return self.T0.backward(coeff)

    def _phys(self, coeff: Array) -> tuple[Array, Array, Array]:
        N = self.padded_counts
        value = self.TD.backward(coeff, N=N)
        radial = self.TD.backward_primitive(coeff, (0, 1), N=N)
        axial = self.TD.backward_primitive(coeff, (1, 0), N=N)
        return value, radial, axial

    def nonlinear(self, state: AxisymmetricTCState) -> Velocity:
        """Return ``inner(v_i, N_i)`` for cylindrical perturbation advection.

        Reference: ``couette/taylor_couette_dns.py:251-272``.
        """
        ur, urr, urz = self._phys(state.u[0])
        ut, utr, _utz = self._phys(state.u[1])
        uz, uzr, uzz = self._phys(state.u[2])
        invr = self.inv_r_p
        n_r = ur * urr + uz * urz - ut * ut * invr
        n_t = ur * utr + uz * _utz + ur * ut * invr
        n_z = ur * uzr + uz * uzz
        return (
            self.TD.scalar_product(self._dealias_to_standard(n_r)),
            self.TD.scalar_product(self._dealias_to_standard(n_t)),
            self.TD.scalar_product(self._dealias_to_standard(n_z)),
        )

    def _apply_lexp(self, u: Velocity) -> Velocity:
        flat = self.VV.flatten(u)
        modes = flat[self.VV_mode_indices]
        out_modes = jnp.einsum("kij,kj->ki", self.Lexp_modes, modes)
        out = self._scatter_modes(out_modes, self.VV_mode_indices, self.VV.dim)
        return self.VV.unflatten(out)  # ty: ignore[return-value]

    def step(self, state: AxisymmetricTCState) -> AxisymmetricTCState:
        """Advance one CNAB2 step with an IMEX-Euler bootstrap."""
        n_hat = self.nonlinear(state)
        rhs_v = self._apply_lexp(state.u)
        rhs_u = []
        for rhs_i, n_i, old_i in zip(rhs_v, n_hat, state.nonlinear_old, strict=True):
            nonlinear_i = 1.5 * n_i - 0.5 * old_i if state.have_old else n_i
            rhs_u.append(rhs_i - nonlinear_i)
        rhs_p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        rhs = self.VQ.flatten((*rhs_u, rhs_p))
        sol = self.VQ.unflatten(self._solve_limp(rhs))
        return AxisymmetricTCState(
            u=(sol[0], sol[1], sol[2]),
            p=sol[3],
            nonlinear_old=n_hat,
            have_old=True,
        )

    def solve(self, state: AxisymmetricTCState, steps: int) -> AxisymmetricTCState:
        for _ in range(int(steps)):
            state = self.step(state)
        return state

    def velocity_physical(self, state: AxisymmetricTCState) -> Velocity:
        return tuple(self.TD.backward(ui) for ui in state.u)  # ty: ignore[return-value]

    def energy(self, state: AxisymmetricTCState) -> Array:
        ur, ut, uz = self.velocity_physical(state)
        density = 0.5 * (jnp.conj(ur) * ur + jnp.conj(ut) * ut + jnp.conj(uz) * uz)
        return jnp.real(integrate(density * self.R, self.T0))

    def divergence(self, state: AxisymmetricTCState) -> Array:
        dur_dr = self.TD.backward_primitive(state.u[0], (0, 1))
        duz_dz = self.TD.backward_primitive(state.u[2], (1, 0))
        ur = self.TD.backward(state.u[0])
        return dur_dr + ur * self.inv_r + duz_dz

    def divergence_linf(self, state: AxisymmetricTCState) -> Array:
        return jnp.max(jnp.abs(self.divergence(state)))

    def continuity_residual_l2(self, state: AxisymmetricTCState) -> Array:
        q = TestFunction(self.TP)
        ur = TrialFunction(self.TD)
        uz = TrialFunction(self.TD)
        dr = self._dense(q * Dx(ur, 1, 1))
        invr = self._dense(q * (1 / self.r) * ur)
        dz = self._dense(q * Dx(uz, 0, 1))
        residual = (dr + invr) @ state.u[0].ravel() + dz @ state.u[2].ravel()
        return jnp.linalg.norm(residual)

    def diagnostics(self, state: AxisymmetricTCState) -> dict[str, Array]:
        ur, ut, uz = self.velocity_physical(state)
        wall = jnp.max(
            jnp.array(
                [
                    jnp.max(jnp.abs(ur[:, 0])),
                    jnp.max(jnp.abs(ur[:, -1])),
                    jnp.max(jnp.abs(ut[:, 0])),
                    jnp.max(jnp.abs(ut[:, -1])),
                    jnp.max(jnp.abs(uz[:, 0])),
                    jnp.max(jnp.abs(uz[:, -1])),
                ]
            )
        )
        return {
            "E": self.energy(state),
            "div_linf": self.divergence_linf(state),
            "continuity_l2": self.continuity_residual_l2(state),
            "wall": wall,
            "Eth": jnp.real(integrate(jnp.conj(ut) * ut * self.R, self.T0)),
        }

    def growth_rate(
        self, state: AxisymmetricTCState, steps: int
    ) -> tuple[Array, AxisymmetricTCState]:
        e0 = self.energy(state)
        out = self.solve(state, steps)
        e1 = self.energy(out)
        elapsed = int(steps) * self.dt
        return 0.5 * jnp.log(e1 / e0) / elapsed, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nr", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--nu", type=float, default=1.0e-2)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    parser.add_argument("--dealias", type=float, default=1.5)
    args = parser.parse_args()

    solver = AxisymmetricTCDNSJax(
        CircularCouette(),
        nu=args.nu,
        Nr=args.Nr,
        Nz=args.Nz,
        dt=args.dt,
        family=args.family,
        dealias=args.dealias,
    )
    state = solver.solve(solver.initial_state(), args.steps)
    diag = solver.diagnostics(state)
    print(" ".join(f"{key}={float(value):.6e}" for key, value in diag.items()))


if __name__ == "__main__":
    main()
