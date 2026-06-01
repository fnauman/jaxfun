"""Axisymmetric Taylor-Couette DNS using jaxfun Galerkin blocks.

This is the JAX counterpart of ``couette/taylor_couette_dns.py`` for the
axisymmetric hydrodynamic perturbation equations.  It keeps the shenfun
formulation deliberately: Cartesian tensor-product spaces, explicit cylindrical
``1/r`` factors in the weak forms, Dirichlet velocity modes, and a truncated
orthogonal pressure space for the ``P_N/P_{N-2}`` pair.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
    from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
except ModuleNotFoundError:  # direct script execution from examples/
    from taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax
    from taylor_couette_mri_jax import TaylorCouetteMRIJax

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


def _require_resolved_m(m: int, ntheta: int) -> None:
    """Require distinct ``+/-m`` Fourier modes below the azimuthal Nyquist."""
    m = int(m)
    ntheta = int(ntheta)
    if 2 * abs(m) >= ntheta:
        raise ValueError(
            f"azimuthal mode |m|={abs(m)} is unresolved by Ntheta={ntheta}; "
            f"require 2|m| < Ntheta"
        )


@dataclass(frozen=True)
class AxisymmetricTCState:
    """Coefficient state for the axisymmetric TC DNS solver."""

    u: Velocity
    p: Array
    nonlinear_old: Velocity
    have_old: bool = False


MHDFields = tuple[Array, Array, Array, Array, Array, Array]


@dataclass(frozen=True)
class AxisymmetricMRIState:
    """Coefficient state for axisymmetric TC-MHD DNS."""

    x: MHDFields
    p: Array
    nonlinear_old: MHDFields
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
        """Return flat dof indices grouped by Fourier-mode tuple.

        Component tensor spaces are ordered with all Fourier axes first and the
        radial solve axis last, so each mode block is the contiguous radial
        slice for one ``(m, kz, ...)`` tuple.  This covers the axisymmetric
        ``(kz, r)`` solver and the 3D ``(theta, z, r)`` solver.
        """
        mode_shape = tuple(int(n) for n in space[0].num_dofs[:-1])
        if not mode_shape:
            mode_iter = [()]
        else:
            mode_iter = itertools.product(*(range(n) for n in mode_shape))
        per_mode_indices = []
        for mode_tuple in mode_iter:
            flat_mode = 0
            for value, extent in zip(mode_tuple, mode_shape, strict=True):
                flat_mode = flat_mode * extent + value
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


class TaylorCouetteDNSJax(AxisymmetricTCDNSJax):
    """Full 3D hydrodynamic perturbation Taylor-Couette DNS.

    Reference: ``couette/taylor_couette_dns.py:487-786``.  The unknowns depend
    on ``(theta, z, r)`` with complex Fourier axes in ``theta`` and ``z`` and a
    no-slip radial basis.  Each ``(m, kz)`` mode is solved as an independent
    radial velocity-pressure block.
    """

    def __init__(
        self,
        base: CircularCouette,
        nu: float = 1.0e-2,
        Nr: int = 24,
        Ntheta: int = 8,
        Nz: int = 16,
        Lz: float | None = None,
        dt: float = 2.0e-3,
        family: str = "L",
        dealias: float = 1.5,
    ) -> None:
        self.base = base
        self.nu = float(nu)
        self.Nr = int(Nr)
        self.Ntheta = int(Ntheta)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.13 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu

        family_cls = self._family_class(self.family)
        dom = Domain(base.R1, base.R2)
        self.Ft = FunctionSpace(
            self.Ntheta, Fourier, domain=Domain(0.0, 2.0 * math.pi), name="Ft"
        )
        self.Fz = FunctionSpace(
            self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="Fz"
        )
        self.SD = FunctionSpace(self.Nr, family_cls, bc=(0, 0), domain=dom, name="SD")
        self.S0 = FunctionSpace(self.Nr, family_cls, domain=dom, name="S0")
        self.SP = FunctionSpace(
            self.Nr, family_cls, domain=dom, num_dofs=self.Nr - 2, name="SP"
        )

        self.TD = TensorProduct(self.Ft, self.Fz, self.SD, name="TD3")
        self.T0 = TensorProduct(self.Ft, self.Fz, self.S0, name="T03")
        self.TP = TensorProduct(self.Ft, self.Fz, self.SP, name="TP3")
        self.VV = CoupledSpace((self.TD, self.TD, self.TD), name="VV3")
        self.VQ = CoupledSpace((self.TD, self.TD, self.TD, self.TP), name="VQ3")

        self.theta, self.z, self.r = self.TD.system.base_scalars()
        self.pressure_pin = self.VQ.block_slices[3].start
        self.VQ_mode_indices = self._mode_indices(self.VQ)
        self.VV_mode_indices = self._mode_indices(self.VV)
        self.Theta, self.Z, self.R = self.T0.mesh()
        self.inv_r = 1.0 / self.R
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
            _thp, _zp, Rp = self.T0p.mesh()
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

    def _lap(self, u: sp.Expr) -> sp.Expr:
        r = self.r
        return (
            Dx(u, 2, 2)
            + (1 / r) * Dx(u, 2, 1)
            + (1 / r**2) * Dx(u, 0, 2)
            + Dx(u, 1, 2)
        )

    def _add_avv_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        ur: sp.Expr,
        ut: sp.Expr,
        uz: sp.Expr,
        vr: sp.Expr,
        vt: sp.Expr,
        vz: sp.Expr,
        sign: float,
    ) -> Array:
        r = self.r
        nu = self.nu
        a = self.base.a
        omega = self.base.a + self.base.b / r**2
        terms = [
            (0, 0, vr * (sign * nu * self._lap(ur))),
            (0, 0, vr * (sign * (-nu) * (1 / r**2) * ur)),
            (0, 1, vr * (sign * (-nu) * (2 / r**2) * Dx(ut, 0, 1))),
            (0, 0, vr * (sign * (-omega) * Dx(ur, 0, 1))),
            (0, 1, vr * (sign * (2 * omega) * ut)),
            (1, 1, vt * (sign * nu * self._lap(ut))),
            (1, 1, vt * (sign * (-nu) * (1 / r**2) * ut)),
            (1, 0, vt * (sign * nu * (2 / r**2) * Dx(ur, 0, 1))),
            (1, 1, vt * (sign * (-omega) * Dx(ut, 0, 1))),
            (1, 0, vt * (sign * (-2 * a) * ur)),
            (2, 2, vz * (sign * nu * self._lap(uz))),
            (2, 2, vz * (sign * (-omega) * Dx(uz, 0, 1))),
        ]
        for i, j, expr in terms:
            A = self._add_form(A, test_space, trial_space, i, j, expr)
        return A

    def _build_operators(self) -> tuple[Array, Array]:
        r = self.r
        dt = self.dt
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
        Limp = self._add_avv_terms(
            Limp, self.VQ, self.VQ, ur, ut, uz, vr, vt, vz, sign=-0.5
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 0, 3, vr * Dx(p, 2, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 1, 3, vt * (1 / r) * Dx(p, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ur, 2, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * (1 / r) * ur)
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 1, q * (1 / r) * Dx(ut, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 1, 1))

        er = TrialFunction(self.TD, name="er")
        et = TrialFunction(self.TD, name="et")
        ez = TrialFunction(self.TD, name="ez")
        tr = TestFunction(self.TD, name="tr")
        tt = TestFunction(self.TD, name="tt")
        tz = TestFunction(self.TD, name="tz")
        Lexp = jnp.zeros((self.VV.dim, self.VV.dim), dtype=dtype)
        for i, v, u in ((0, tr, er), (1, tt, et), (2, tz, ez)):
            Lexp = self._add_form(Lexp, self.VV, self.VV, i, i, v * u * (1.0 / dt))
        Lexp = self._add_avv_terms(
            Lexp, self.VV, self.VV, er, et, ez, tr, tt, tz, sign=0.5
        )
        return Limp, Lexp

    def _phys(self, coeff: Array) -> tuple[Array, Array, Array, Array]:
        N = self.padded_counts
        value = self.TD.backward(coeff, N=N)
        radial = self.TD.backward_primitive(coeff, (0, 0, 1), N=N)
        theta = self.TD.backward_primitive(coeff, (1, 0, 0), N=N)
        axial = self.TD.backward_primitive(coeff, (0, 1, 0), N=N)
        return value, radial, theta, axial

    def nonlinear(self, state: AxisymmetricTCState) -> Velocity:
        ur, urr, urt, urz = self._phys(state.u[0])
        ut, utr, utt, utz = self._phys(state.u[1])
        uz, uzr, uzt, uzz = self._phys(state.u[2])
        invr = self.inv_r_p
        n_r = ur * urr + (ut * invr) * urt + uz * urz - ut * ut * invr
        n_t = ur * utr + (ut * invr) * utt + uz * utz + ur * ut * invr
        n_z = ur * uzr + (ut * invr) * uzt + uz * uzz
        return (
            self.TD.scalar_product(self._dealias_to_standard(n_r)),
            self.TD.scalar_product(self._dealias_to_standard(n_t)),
            self.TD.scalar_product(self._dealias_to_standard(n_z)),
        )

    def initial_state(
        self, amp: float = 1.0e-3, m: int = 1, kz_mode: int = 1
    ) -> AxisymmetricTCState:
        _require_resolved_m(m, self.Ntheta)
        R1 = self.base.R1
        d = self.base.gap
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        arg = math.pi * (self.R - R1) / d
        shape = jnp.sin(arg)
        phase = jnp.cos(int(m) * self.Theta) * jnp.cos(kz * self.Z)
        wall_field = amp * shape * phase
        return self.state_from_physical((wall_field, wall_field, wall_field))

    def seed_linear_eigenmode(
        self, m: int = 1, kz_mode: int = 1, amp: float = 1.0e-6, which: int = 0
    ) -> tuple[AxisymmetricTCState, complex]:
        _require_resolved_m(m, self.Ntheta)
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = TaylorCouetteLinearJax(
            self.base, nu=self.nu, N=self.Nr, family=self.family
        )
        w, vecs = lin.eigs(m=m, kz=kz, n_return=which + 1)
        vec = vecs[:, which]
        n = lin.n
        state = self.zero_state()
        comps = list(state.u)
        mpos = int(m) % self.Ntheta
        mneg = (-int(m)) % self.Ntheta
        kpos = int(kz_mode) % self.Nz
        kneg = (-int(kz_mode)) % self.Nz
        for comp in range(3):
            block = jnp.asarray(vec[comp * n : (comp + 1) * n]) * amp
            arr = jnp.zeros_like(comps[comp])
            if mpos == mneg and kpos == kneg:
                arr = arr.at[mpos, kpos, :n].set(jnp.real(block))
            else:
                arr = arr.at[mpos, kpos, :n].set(0.5 * block)
                arr = arr.at[mneg, kneg, :n].set(0.5 * jnp.conj(block))
            comps[comp] = arr
        nold = tuple(jnp.zeros_like(ui) for ui in comps)
        return AxisymmetricTCState(tuple(comps), state.p, nold, False), complex(
            w[which]
        )

    def velocity_physical(self, state: AxisymmetricTCState) -> Velocity:
        return tuple(self.TD.backward(ui) for ui in state.u)  # ty: ignore[return-value]

    def energy(self, state: AxisymmetricTCState) -> Array:
        ur, ut, uz = self.velocity_physical(state)
        density = 0.5 * (jnp.conj(ur) * ur + jnp.conj(ut) * ut + jnp.conj(uz) * uz)
        return jnp.real(integrate(density * self.R, self.T0))

    def divergence(self, state: AxisymmetricTCState) -> Array:
        dur_dr = self.TD.backward_primitive(state.u[0], (0, 0, 1))
        dut_dt = self.TD.backward_primitive(state.u[1], (1, 0, 0))
        duz_dz = self.TD.backward_primitive(state.u[2], (0, 1, 0))
        ur = self.TD.backward(state.u[0])
        return dur_dr + ur * self.inv_r + dut_dt * self.inv_r + duz_dz

    def continuity_residual_l2(self, state: AxisymmetricTCState) -> Array:
        q = TestFunction(self.TP)
        ur = TrialFunction(self.TD)
        ut = TrialFunction(self.TD)
        uz = TrialFunction(self.TD)
        dr = self._dense(q * Dx(ur, 2, 1))
        invr = self._dense(q * (1 / self.r) * ur)
        dt = self._dense(q * (1 / self.r) * Dx(ut, 0, 1))
        dz = self._dense(q * Dx(uz, 1, 1))
        residual = (
            (dr + invr) @ state.u[0].ravel()
            + dt @ state.u[1].ravel()
            + dz @ state.u[2].ravel()
        )
        return jnp.linalg.norm(residual)

    def diagnostics(self, state: AxisymmetricTCState) -> dict[str, Array]:
        ur, ut, uz = self.velocity_physical(state)
        wall = jnp.max(
            jnp.array(
                [
                    jnp.max(jnp.abs(ur[:, :, 0])),
                    jnp.max(jnp.abs(ur[:, :, -1])),
                    jnp.max(jnp.abs(ut[:, :, 0])),
                    jnp.max(jnp.abs(ut[:, :, -1])),
                    jnp.max(jnp.abs(uz[:, :, 0])),
                    jnp.max(jnp.abs(uz[:, :, -1])),
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


class AxisymmetricMRIDNSJax(AxisymmetricTCDNSJax):
    """Axisymmetric conducting-wall Taylor-Couette MHD/MRI DNS.

    Reference: ``couette/taylor_couette_dns.py:788-1190``.  The evolved state is
    ``(u_r,u_theta,u_z,b_r,b_theta,b_z)`` and the implicit solve uses the
    7-field total-pressure system ``(u_r,u_theta,u_z,Pi,b_r,b_theta,b_z)``.
    """

    def __init__(
        self,
        base: CircularCouette,
        B0: float = 0.1,
        nu: float = 1.0e-3,
        eta_mag: float = 1.0e-3,
        Nr: int = 24,
        Nz: int = 16,
        Lz: float | None = None,
        dt: float = 2.0e-3,
        family: str = "L",
        dealias: float = 1.5,
    ) -> None:
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nr = int(Nr)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.0 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        self.Rm = base.Omega1 * base.R1 * base.gap / self.eta_mag
        self.Pm = self.nu / self.eta_mag
        self.S = self.B0 * base.gap / self.eta_mag
        self.Jm = 0.5 * (base.R2 - base.R1)

        family_cls = self._family_class(self.family)
        dom = Domain(base.R1, base.R2)
        self.F = FunctionSpace(self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="F")
        self.SD = FunctionSpace(self.Nr, family_cls, bc=(0, 0), domain=dom, name="SD")
        self.S0 = FunctionSpace(self.Nr, family_cls, domain=dom, name="S0")
        self.SP = FunctionSpace(
            self.Nr, family_cls, domain=dom, num_dofs=self.Nr - 2, name="SP"
        )
        self.Sbt = FunctionSpace(
            self.Nr,
            family_cls,
            domain=dom,
            system=None,
            bc={
                "left": {"R": (base.R1 / self.Jm, 0)},
                "right": {"R": (base.R2 / self.Jm, 0)},
            },
            name="Sbt",
        )
        self.Sbz = FunctionSpace(
            self.Nr,
            family_cls,
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=dom,
            name="Sbz",
        )

        self.TD = TensorProduct(self.F, self.SD, name="TDm")
        self.T0 = TensorProduct(self.F, self.S0, name="T0m")
        self.TP = TensorProduct(self.F, self.SP, name="TPm")
        self.Tbt = TensorProduct(self.F, self.Sbt, name="Tbt")
        self.Tbz = TensorProduct(self.F, self.Sbz, name="Tbz")
        self.VQ = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TP, self.TD, self.Tbt, self.Tbz),
            name="VQm",
        )
        self.VE = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz),
            name="VEm",
        )

        self.z, self.r = self.TD.system.base_scalars()
        self.VQ_mode_indices = self._mode_indices(self.VQ)
        self.VE_mode_indices = self._mode_indices(self.VE)
        self.Z, self.R = self.T0.mesh()
        self.inv_r = 1.0 / self.R
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
            _Zp, Rp = self.T0p.mesh()
            self.inv_r_p = 1.0 / Rp
        else:
            self.T0p = None
            self.padded_counts = None
            self.inv_r_p = self.inv_r

        self.Limp, self.Lexp = self._build_operators()
        self.Limp_modes = self._extract_mode_matrices(self.Limp, self.VQ_mode_indices)
        self.Lexp_modes = self._extract_mode_matrices(self.Lexp, self.VE_mode_indices)
        self.Limp_lu = jax.vmap(jsp_linalg.lu_factor)(
            self._pin_pressure_modes(self.Limp_modes)
        )

    def _lap(self, u: sp.Expr) -> sp.Expr:
        r = self.r
        return Dx(u, 1, 2) + (1 / r) * Dx(u, 1, 1) + Dx(u, 0, 2)

    def _add_mhd_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        idx: dict[str, int],
        fields: dict[str, sp.Expr],
        tests: dict[str, sp.Expr],
        sign: float,
    ) -> Array:
        r = self.r
        nu, eta, B0 = self.nu, self.eta_mag, self.B0
        a = self.base.a
        omega = self.base.a + self.base.b / r**2
        r_omega_prime = -2 * self.base.b / r**2
        dz = lambda f: Dx(f, 0, 1)
        terms = [
            ("ur", "ur", tests["ur"] * (sign * nu * self._lap(fields["ur"]))),
            ("ur", "ur", tests["ur"] * (sign * (-nu) * (1 / r**2) * fields["ur"])),
            ("ur", "ut", tests["ur"] * (sign * (2 * omega) * fields["ut"])),
            ("ur", "br", tests["ur"] * (sign * B0 * dz(fields["br"]))),
            ("ut", "ut", tests["ut"] * (sign * nu * self._lap(fields["ut"]))),
            ("ut", "ut", tests["ut"] * (sign * (-nu) * (1 / r**2) * fields["ut"])),
            ("ut", "ur", tests["ut"] * (sign * (-2 * a) * fields["ur"])),
            ("ut", "bt", tests["ut"] * (sign * B0 * dz(fields["bt"]))),
            ("uz", "uz", tests["uz"] * (sign * nu * self._lap(fields["uz"]))),
            ("uz", "bz", tests["uz"] * (sign * B0 * dz(fields["bz"]))),
            ("br", "br", tests["br"] * (sign * eta * self._lap(fields["br"]))),
            ("br", "br", tests["br"] * (sign * (-eta) * (1 / r**2) * fields["br"])),
            ("br", "ur", tests["br"] * (sign * B0 * dz(fields["ur"]))),
            ("bt", "bt", tests["bt"] * (sign * eta * self._lap(fields["bt"]))),
            ("bt", "bt", tests["bt"] * (sign * (-eta) * (1 / r**2) * fields["bt"])),
            ("bt", "ut", tests["bt"] * (sign * B0 * dz(fields["ut"]))),
            ("bt", "br", tests["bt"] * (sign * r_omega_prime * fields["br"])),
            ("bz", "bz", tests["bz"] * (sign * eta * self._lap(fields["bz"]))),
            ("bz", "uz", tests["bz"] * (sign * B0 * dz(fields["uz"]))),
        ]
        for row, col, expr in terms:
            A = self._add_form(A, test_space, trial_space, idx[row], idx[col], expr)
        return A

    def _build_operators(self) -> tuple[Array, Array]:
        r = self.r
        dt = self.dt
        dtype = jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))

        ur = TrialFunction(self.TD, name="ur")
        ut = TrialFunction(self.TD, name="ut")
        uz = TrialFunction(self.TD, name="uz")
        p = TrialFunction(self.TP, name="Pi")
        br = TrialFunction(self.TD, name="br")
        bt = TrialFunction(self.Tbt, name="bt")
        bz = TrialFunction(self.Tbz, name="bz")
        vr = TestFunction(self.TD, name="vr")
        vt = TestFunction(self.TD, name="vt")
        vz = TestFunction(self.TD, name="vz")
        q = TestFunction(self.TP, name="q")
        cr = TestFunction(self.TD, name="cr")
        ct = TestFunction(self.Tbt, name="ct")
        cz = TestFunction(self.Tbz, name="cz")
        idx_q = {"ur": 0, "ut": 1, "uz": 2, "p": 3, "br": 4, "bt": 5, "bz": 6}
        fields_q = {"ur": ur, "ut": ut, "uz": uz, "br": br, "bt": bt, "bz": bz}
        tests_q = {"ur": vr, "ut": vt, "uz": vz, "br": cr, "bt": ct, "bz": cz}

        Limp = jnp.zeros((self.VQ.dim, self.VQ.dim), dtype=dtype)
        for name in ("ur", "ut", "uz", "br", "bt", "bz"):
            Limp = self._add_form(
                Limp,
                self.VQ,
                self.VQ,
                idx_q[name],
                idx_q[name],
                tests_q[name] * fields_q[name] * (1.0 / dt),
            )
        Limp = self._add_mhd_terms(
            Limp, self.VQ, self.VQ, idx_q, fields_q, tests_q, sign=-0.5
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 0, 3, vr * Dx(p, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ur, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * (1 / r) * ur)
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 0, 1))

        eur = TrialFunction(self.TD, name="eur")
        eut = TrialFunction(self.TD, name="eut")
        euz = TrialFunction(self.TD, name="euz")
        ebr = TrialFunction(self.TD, name="ebr")
        ebt = TrialFunction(self.Tbt, name="ebt")
        ebz = TrialFunction(self.Tbz, name="ebz")
        tur = TestFunction(self.TD, name="tur")
        tut = TestFunction(self.TD, name="tut")
        tuz = TestFunction(self.TD, name="tuz")
        tbr = TestFunction(self.TD, name="tbr")
        tbt = TestFunction(self.Tbt, name="tbt")
        tbz = TestFunction(self.Tbz, name="tbz")
        idx_e = {"ur": 0, "ut": 1, "uz": 2, "br": 3, "bt": 4, "bz": 5}
        fields_e = {"ur": eur, "ut": eut, "uz": euz, "br": ebr, "bt": ebt, "bz": ebz}
        tests_e = {"ur": tur, "ut": tut, "uz": tuz, "br": tbr, "bt": tbt, "bz": tbz}
        Lexp = jnp.zeros((self.VE.dim, self.VE.dim), dtype=dtype)
        for name in ("ur", "ut", "uz", "br", "bt", "bz"):
            Lexp = self._add_form(
                Lexp,
                self.VE,
                self.VE,
                idx_e[name],
                idx_e[name],
                tests_e[name] * fields_e[name] * (1.0 / dt),
            )
        Lexp = self._add_mhd_terms(
            Lexp, self.VE, self.VE, idx_e, fields_e, tests_e, sign=0.5
        )
        return Limp, Lexp

    def zero_state(self) -> AxisymmetricMRIState:
        x = tuple(jnp.zeros(space.num_dofs, dtype=self.Limp.dtype) for space in self.VE)
        p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricMRIState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def state_from_physical(self, values: MHDFields) -> AxisymmetricMRIState:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz)
        x = tuple(
            space.forward(value)
            for space, value in zip(spaces, values, strict=True)
        )
        p = jnp.zeros(self.TP.num_dofs, dtype=x[0].dtype)
        nold = tuple(jnp.zeros_like(xi) for xi in x)
        return AxisymmetricMRIState(x=x, p=p, nonlinear_old=nold, have_old=False)

    def _phys_mhd(self, coeff: Array, space) -> tuple[Array, Array, Array]:
        N = self.padded_counts
        value = space.backward(coeff, N=N)
        radial = space.backward_primitive(coeff, (0, 1), N=N)
        axial = space.backward_primitive(coeff, (1, 0), N=N)
        return value, radial, axial

    def _t0_coeff(self, values: Array) -> Array:
        if self.T0p is None:
            return self.T0.forward(values)
        return self.T0p.forward(values)

    def _standard_product(self, values: Array) -> Array:
        return self._dealias_to_standard(values)

    def nonlinear(self, state: AxisymmetricMRIState) -> MHDFields:
        ur, urr, urz = self._phys_mhd(state.x[0], self.TD)
        ut, utr, utz = self._phys_mhd(state.x[1], self.TD)
        uz, uzr, uzz = self._phys_mhd(state.x[2], self.TD)
        br, brr, brz = self._phys_mhd(state.x[3], self.TD)
        bt, btr, btz = self._phys_mhd(state.x[4], self.Tbt)
        bz, bzr, bzz = self._phys_mhd(state.x[5], self.Tbz)
        invr = self.inv_r_p
        au_r = ur * urr + uz * urz - ut * ut * invr
        au_t = ur * utr + uz * utz + ur * ut * invr
        au_z = ur * uzr + uz * uzz
        lb_r = br * brr + bz * brz - bt * bt * invr
        lb_t = br * btr + bz * btz + br * bt * invr
        lb_z = br * bzr + bz * bzz
        nu_r = self._standard_product(au_r - lb_r)
        nu_t = self._standard_product(au_t - lb_t)
        nu_z = self._standard_product(au_z - lb_z)

        eps_r = self._standard_product(ut * bz - uz * bt)
        eps_t = self._standard_product(uz * br - ur * bz)
        eps_z = self._standard_product(ur * bt - ut * br)
        er_hat = self._t0_coeff(eps_r)
        et_hat = self._t0_coeff(eps_t)
        ez_hat = self._t0_coeff(eps_z)
        nb_r = self.T0.backward_primitive(et_hat, (1, 0))
        nb_t = -self.T0.backward_primitive(er_hat, (1, 0)) + self.T0.backward_primitive(
            ez_hat, (0, 1)
        )
        nb_z = -self.T0.backward_primitive(et_hat, (0, 1)) - eps_t * self.inv_r
        return (
            self.TD.scalar_product(nu_r),
            self.TD.scalar_product(nu_t),
            self.TD.scalar_product(nu_z),
            self.TD.scalar_product(nb_r),
            self.Tbt.scalar_product(nb_t),
            self.Tbz.scalar_product(nb_z),
        )

    def _apply_lexp_mhd(self, x: MHDFields) -> MHDFields:
        flat = self.VE.flatten(x)
        modes = flat[self.VE_mode_indices]
        out_modes = jnp.einsum("kij,kj->ki", self.Lexp_modes, modes)
        out = self._scatter_modes(out_modes, self.VE_mode_indices, self.VE.dim)
        return self.VE.unflatten(out)  # ty: ignore[return-value]

    def step(self, state: AxisymmetricMRIState) -> AxisymmetricMRIState:
        n_hat = self.nonlinear(state)
        rhs_e = self._apply_lexp_mhd(state.x)
        rhs_x = []
        for rhs_i, n_i, old_i in zip(rhs_e, n_hat, state.nonlinear_old, strict=True):
            nonlinear_i = 1.5 * n_i - 0.5 * old_i if state.have_old else n_i
            rhs_x.append(rhs_i - nonlinear_i)
        rhs_p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp.dtype)
        rhs = self.VQ.flatten((*rhs_x[:3], rhs_p, *rhs_x[3:]))
        sol = self.VQ.unflatten(self._solve_limp(rhs))
        x = (sol[0], sol[1], sol[2], sol[4], sol[5], sol[6])
        return AxisymmetricMRIState(x=x, p=sol[3], nonlinear_old=n_hat, have_old=True)

    def solve(self, state: AxisymmetricMRIState, steps: int) -> AxisymmetricMRIState:
        for _ in range(int(steps)):
            state = self.step(state)
        return state

    def seed_linear_eigenmode(
        self, kz_mode: int = 1, amp: float = 1.0e-6, which: int = 0
    ) -> tuple[AxisymmetricMRIState, complex]:
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = TaylorCouetteMRIJax(
            self.base,
            B0=self.B0,
            nu=self.nu,
            eta_mag=self.eta_mag,
            N=self.Nr,
            family=self.family,
            magnetic_bc="conducting",
        )
        w, vecs = lin.eigs(m=0, kz=kz, n_return=which + 1)
        vec = vecs[:, which]
        n = lin.n
        state = self.zero_state()
        comps = list(state.x)
        kpos = int(kz_mode) % self.Nz
        kneg = (-int(kz_mode)) % self.Nz
        block_map = (0, 1, 2, 4, 5, 6)
        for comp, block_index in enumerate(block_map):
            block = jnp.asarray(vec[block_index * n : (block_index + 1) * n]) * amp
            arr = jnp.zeros_like(comps[comp])
            if kpos == kneg:
                arr = arr.at[kpos, :n].set(jnp.real(block))
            else:
                arr = arr.at[kpos, :n].set(0.5 * block)
                arr = arr.at[kneg, :n].set(0.5 * jnp.conj(block))
            comps[comp] = arr
        nold = tuple(jnp.zeros_like(xi) for xi in comps)
        return AxisymmetricMRIState(tuple(comps), state.p, nold, False), complex(
            w[which]
        )

    def fields_physical(self, state: AxisymmetricMRIState) -> MHDFields:
        spaces = (self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz)
        return tuple(
            space.backward(coeff) for space, coeff in zip(spaces, state.x, strict=True)
        )  # ty: ignore[return-value]

    def energy_parts(self, state: AxisymmetricMRIState) -> tuple[Array, Array]:
        ur, ut, uz, br, bt, bz = self.fields_physical(state)
        ek_density = (
            jnp.conj(ur) * ur + jnp.conj(ut) * ut + jnp.conj(uz) * uz
        )
        em_density = (
            jnp.conj(br) * br + jnp.conj(bt) * bt + jnp.conj(bz) * bz
        )
        ek = 0.5 * integrate(ek_density * self.R, self.T0)
        em = 0.5 * integrate(em_density * self.R, self.T0)
        return jnp.real(ek), jnp.real(em)

    def energy(self, state: AxisymmetricMRIState) -> Array:
        ek, em = self.energy_parts(state)
        return ek + em

    def velocity_divergence(self, state: AxisymmetricMRIState) -> Array:
        dur_dr = self.TD.backward_primitive(state.x[0], (0, 1))
        duz_dz = self.TD.backward_primitive(state.x[2], (1, 0))
        ur = self.TD.backward(state.x[0])
        return dur_dr + ur * self.inv_r + duz_dz

    def magnetic_divergence(self, state: AxisymmetricMRIState) -> Array:
        dbr_dr = self.TD.backward_primitive(state.x[3], (0, 1))
        dbz_dz = self.Tbz.backward_primitive(state.x[5], (1, 0))
        br = self.TD.backward(state.x[3])
        return dbr_dr + br * self.inv_r + dbz_dz

    def divergence_linf(self, state: AxisymmetricMRIState) -> tuple[Array, Array]:
        return (
            jnp.max(jnp.abs(self.velocity_divergence(state))),
            jnp.max(jnp.abs(self.magnetic_divergence(state))),
        )

    def diagnostics(self, state: AxisymmetricMRIState) -> dict[str, Array]:
        ek, em = self.energy_parts(state)
        divu, divb = self.divergence_linf(state)
        return {"Ekin": ek, "Emag": em, "E": ek + em, "divu": divu, "divb": divb}

    def growth_rate(
        self, state: AxisymmetricMRIState, steps: int
    ) -> tuple[Array, AxisymmetricMRIState]:
        e0 = self.energy(state)
        out = self.solve(state, steps)
        e1 = self.energy(out)
        elapsed = int(steps) * self.dt
        return 0.5 * jnp.log(e1 / e0) / elapsed, out


class TaylorCouetteMRIDNSJax(AxisymmetricMRIDNSJax):
    """Full 3D conducting-wall Taylor-Couette MHD/MRI DNS.

    Reference: ``couette/taylor_couette_dns.py:1196-1655``.  This combines the
    3D azimuthal Fourier machinery with the conducting-wall MHD block used by
    :class:`AxisymmetricMRIDNSJax`.
    """

    def __init__(
        self,
        base: CircularCouette,
        B0: float = 0.1,
        nu: float = 1.0e-3,
        eta_mag: float = 1.0e-3,
        Nr: int = 20,
        Ntheta: int = 8,
        Nz: int = 16,
        Lz: float | None = None,
        dt: float = 2.0e-3,
        family: str = "L",
        dealias: float = 1.5,
    ) -> None:
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nr = int(Nr)
        self.Ntheta = int(Ntheta)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self.family = family.upper()
        self.dealias = float(dealias)
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.0 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        self.Rm = base.Omega1 * base.R1 * base.gap / self.eta_mag
        self.Pm = self.nu / self.eta_mag
        self.S = self.B0 * base.gap / self.eta_mag
        self.Jm = 0.5 * (base.R2 - base.R1)

        family_cls = self._family_class(self.family)
        dom = Domain(base.R1, base.R2)
        self.Ft = FunctionSpace(
            self.Ntheta, Fourier, domain=Domain(0.0, 2.0 * math.pi), name="Ft"
        )
        self.Fz = FunctionSpace(
            self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="Fz"
        )
        self.SD = FunctionSpace(self.Nr, family_cls, bc=(0, 0), domain=dom, name="SD")
        self.S0 = FunctionSpace(self.Nr, family_cls, domain=dom, name="S0")
        self.SP = FunctionSpace(
            self.Nr, family_cls, domain=dom, num_dofs=self.Nr - 2, name="SP"
        )
        self.Sbt = FunctionSpace(
            self.Nr,
            family_cls,
            domain=dom,
            bc={
                "left": {"R": (base.R1 / self.Jm, 0)},
                "right": {"R": (base.R2 / self.Jm, 0)},
            },
            name="Sbt",
        )
        self.Sbz = FunctionSpace(
            self.Nr,
            family_cls,
            bc={"left": {"N": 0}, "right": {"N": 0}},
            domain=dom,
            name="Sbz",
        )

        self.TD = TensorProduct(self.Ft, self.Fz, self.SD, name="TDm3")
        self.T0 = TensorProduct(self.Ft, self.Fz, self.S0, name="T0m3")
        self.TP = TensorProduct(self.Ft, self.Fz, self.SP, name="TPm3")
        self.Tbt = TensorProduct(self.Ft, self.Fz, self.Sbt, name="Tbt3")
        self.Tbz = TensorProduct(self.Ft, self.Fz, self.Sbz, name="Tbz3")
        self.VQ = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TP, self.TD, self.Tbt, self.Tbz),
            name="VQm3",
        )
        self.VE = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TD, self.Tbt, self.Tbz),
            name="VEm3",
        )

        self.theta, self.z, self.r = self.TD.system.base_scalars()
        self.VQ_mode_indices = self._mode_indices(self.VQ)
        self.VE_mode_indices = self._mode_indices(self.VE)
        self.Theta, self.Z, self.R = self.T0.mesh()
        self.inv_r = 1.0 / self.R
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
            _Thp, _Zp, Rp = self.T0p.mesh()
            self.inv_r_p = 1.0 / Rp
        else:
            self.T0p = None
            self.padded_counts = None
            self.inv_r_p = self.inv_r

        self.Limp, self.Lexp = self._build_operators()
        self.Limp_modes = self._extract_mode_matrices(self.Limp, self.VQ_mode_indices)
        self.Lexp_modes = self._extract_mode_matrices(self.Lexp, self.VE_mode_indices)
        self.Limp_lu = jax.vmap(jsp_linalg.lu_factor)(
            self._pin_pressure_modes(self.Limp_modes)
        )

    def _lap(self, u: sp.Expr) -> sp.Expr:
        r = self.r
        return (
            Dx(u, 2, 2)
            + (1 / r) * Dx(u, 2, 1)
            + (1 / r**2) * Dx(u, 0, 2)
            + Dx(u, 1, 2)
        )

    def _add_mhd_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        idx: dict[str, int],
        fields: dict[str, sp.Expr],
        tests: dict[str, sp.Expr],
        sign: float,
    ) -> Array:
        r = self.r
        nu, eta, B0 = self.nu, self.eta_mag, self.B0
        a = self.base.a
        omega = self.base.a + self.base.b / r**2
        r_omega_prime = -2 * self.base.b / r**2
        dz = lambda f: Dx(f, 1, 1)
        dth = lambda f: Dx(f, 0, 1)
        terms = [
            ("ur", "ur", tests["ur"] * (sign * nu * self._lap(fields["ur"]))),
            ("ur", "ur", tests["ur"] * (sign * (-nu) * (1 / r**2) * fields["ur"])),
            ("ur", "ut", tests["ur"] * (sign * (-nu) * (2 / r**2) * dth(fields["ut"]))),
            ("ur", "ur", tests["ur"] * (sign * (-omega) * dth(fields["ur"]))),
            ("ur", "ut", tests["ur"] * (sign * (2 * omega) * fields["ut"])),
            ("ur", "br", tests["ur"] * (sign * B0 * dz(fields["br"]))),
            ("ut", "ut", tests["ut"] * (sign * nu * self._lap(fields["ut"]))),
            ("ut", "ut", tests["ut"] * (sign * (-nu) * (1 / r**2) * fields["ut"])),
            ("ut", "ur", tests["ut"] * (sign * nu * (2 / r**2) * dth(fields["ur"]))),
            ("ut", "ut", tests["ut"] * (sign * (-omega) * dth(fields["ut"]))),
            ("ut", "ur", tests["ut"] * (sign * (-2 * a) * fields["ur"])),
            ("ut", "bt", tests["ut"] * (sign * B0 * dz(fields["bt"]))),
            ("uz", "uz", tests["uz"] * (sign * nu * self._lap(fields["uz"]))),
            ("uz", "uz", tests["uz"] * (sign * (-omega) * dth(fields["uz"]))),
            ("uz", "bz", tests["uz"] * (sign * B0 * dz(fields["bz"]))),
            ("br", "br", tests["br"] * (sign * eta * self._lap(fields["br"]))),
            ("br", "br", tests["br"] * (sign * (-eta) * (1 / r**2) * fields["br"])),
            (
                "br",
                "bt",
                tests["br"] * (sign * (-eta) * (2 / r**2) * dth(fields["bt"])),
            ),
            ("br", "br", tests["br"] * (sign * (-omega) * dth(fields["br"]))),
            ("br", "ur", tests["br"] * (sign * B0 * dz(fields["ur"]))),
            ("bt", "bt", tests["bt"] * (sign * eta * self._lap(fields["bt"]))),
            ("bt", "bt", tests["bt"] * (sign * (-eta) * (1 / r**2) * fields["bt"])),
            ("bt", "br", tests["bt"] * (sign * eta * (2 / r**2) * dth(fields["br"]))),
            ("bt", "bt", tests["bt"] * (sign * (-omega) * dth(fields["bt"]))),
            ("bt", "br", tests["bt"] * (sign * r_omega_prime * fields["br"])),
            ("bt", "ut", tests["bt"] * (sign * B0 * dz(fields["ut"]))),
            ("bz", "bz", tests["bz"] * (sign * eta * self._lap(fields["bz"]))),
            ("bz", "bz", tests["bz"] * (sign * (-omega) * dth(fields["bz"]))),
            ("bz", "uz", tests["bz"] * (sign * B0 * dz(fields["uz"]))),
        ]
        for row, col, expr in terms:
            A = self._add_form(A, test_space, trial_space, idx[row], idx[col], expr)
        return A

    def _build_operators(self) -> tuple[Array, Array]:
        r = self.r
        dt = self.dt
        dtype = jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))
        ur = TrialFunction(self.TD, name="ur3")
        ut = TrialFunction(self.TD, name="ut3")
        uz = TrialFunction(self.TD, name="uz3")
        p = TrialFunction(self.TP, name="Pi3")
        br = TrialFunction(self.TD, name="br3")
        bt = TrialFunction(self.Tbt, name="bt3")
        bz = TrialFunction(self.Tbz, name="bz3")
        vr = TestFunction(self.TD, name="vr3")
        vt = TestFunction(self.TD, name="vt3")
        vz = TestFunction(self.TD, name="vz3")
        q = TestFunction(self.TP, name="q3")
        cr = TestFunction(self.TD, name="cr3")
        ct = TestFunction(self.Tbt, name="ct3")
        cz = TestFunction(self.Tbz, name="cz3")
        idx_q = {"ur": 0, "ut": 1, "uz": 2, "p": 3, "br": 4, "bt": 5, "bz": 6}
        fields_q = {"ur": ur, "ut": ut, "uz": uz, "br": br, "bt": bt, "bz": bz}
        tests_q = {"ur": vr, "ut": vt, "uz": vz, "br": cr, "bt": ct, "bz": cz}
        Limp = jnp.zeros((self.VQ.dim, self.VQ.dim), dtype=dtype)
        for name in ("ur", "ut", "uz", "br", "bt", "bz"):
            Limp = self._add_form(
                Limp,
                self.VQ,
                self.VQ,
                idx_q[name],
                idx_q[name],
                tests_q[name] * fields_q[name] * (1.0 / dt),
            )
        Limp = self._add_mhd_terms(
            Limp, self.VQ, self.VQ, idx_q, fields_q, tests_q, sign=-0.5
        )
        Limp = self._add_form(Limp, self.VQ, self.VQ, 0, 3, vr * Dx(p, 2, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 1, 3, vt * (1 / r) * Dx(p, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 2, 3, vz * Dx(p, 1, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * Dx(ur, 2, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 0, q * (1 / r) * ur)
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 1, q * (1 / r) * Dx(ut, 0, 1))
        Limp = self._add_form(Limp, self.VQ, self.VQ, 3, 2, q * Dx(uz, 1, 1))

        eur = TrialFunction(self.TD, name="eur3")
        eut = TrialFunction(self.TD, name="eut3")
        euz = TrialFunction(self.TD, name="euz3")
        ebr = TrialFunction(self.TD, name="ebr3")
        ebt = TrialFunction(self.Tbt, name="ebt3")
        ebz = TrialFunction(self.Tbz, name="ebz3")
        tur = TestFunction(self.TD, name="tur3")
        tut = TestFunction(self.TD, name="tut3")
        tuz = TestFunction(self.TD, name="tuz3")
        tbr = TestFunction(self.TD, name="tbr3")
        tbt = TestFunction(self.Tbt, name="tbt3")
        tbz = TestFunction(self.Tbz, name="tbz3")
        idx_e = {"ur": 0, "ut": 1, "uz": 2, "br": 3, "bt": 4, "bz": 5}
        fields_e = {"ur": eur, "ut": eut, "uz": euz, "br": ebr, "bt": ebt, "bz": ebz}
        tests_e = {"ur": tur, "ut": tut, "uz": tuz, "br": tbr, "bt": tbt, "bz": tbz}
        Lexp = jnp.zeros((self.VE.dim, self.VE.dim), dtype=dtype)
        for name in ("ur", "ut", "uz", "br", "bt", "bz"):
            Lexp = self._add_form(
                Lexp,
                self.VE,
                self.VE,
                idx_e[name],
                idx_e[name],
                tests_e[name] * fields_e[name] * (1.0 / dt),
            )
        Lexp = self._add_mhd_terms(
            Lexp, self.VE, self.VE, idx_e, fields_e, tests_e, sign=0.5
        )
        return Limp, Lexp

    def _phys_mhd(self, coeff: Array, space) -> tuple[Array, Array, Array, Array]:
        N = self.padded_counts
        value = space.backward(coeff, N=N)
        radial = space.backward_primitive(coeff, (0, 0, 1), N=N)
        theta = space.backward_primitive(coeff, (1, 0, 0), N=N)
        axial = space.backward_primitive(coeff, (0, 1, 0), N=N)
        return value, radial, theta, axial

    def nonlinear(self, state: AxisymmetricMRIState) -> MHDFields:
        ur, urr, urt, urz = self._phys_mhd(state.x[0], self.TD)
        ut, utr, utt, utz = self._phys_mhd(state.x[1], self.TD)
        uz, uzr, uzt, uzz = self._phys_mhd(state.x[2], self.TD)
        br, brr, brt, brz = self._phys_mhd(state.x[3], self.TD)
        bt, btr, btt, btz = self._phys_mhd(state.x[4], self.Tbt)
        bz, bzr, bzt, bzz = self._phys_mhd(state.x[5], self.Tbz)
        invr = self.inv_r_p
        au_r = ur * urr + (ut * invr) * urt + uz * urz - ut * ut * invr
        au_t = ur * utr + (ut * invr) * utt + uz * utz + ur * ut * invr
        au_z = ur * uzr + (ut * invr) * uzt + uz * uzz
        lb_r = br * brr + (bt * invr) * brt + bz * brz - bt * bt * invr
        lb_t = br * btr + (bt * invr) * btt + bz * btz + br * bt * invr
        lb_z = br * bzr + (bt * invr) * bzt + bz * bzz
        nu_r = self._standard_product(au_r - lb_r)
        nu_t = self._standard_product(au_t - lb_t)
        nu_z = self._standard_product(au_z - lb_z)

        eps_r = self._standard_product(ut * bz - uz * bt)
        eps_t = self._standard_product(uz * br - ur * bz)
        eps_z = self._standard_product(ur * bt - ut * br)
        er_hat = self._t0_coeff(eps_r)
        et_hat = self._t0_coeff(eps_t)
        ez_hat = self._t0_coeff(eps_z)
        nb_r = -self.inv_r * self.T0.backward_primitive(ez_hat, (1, 0, 0))
        nb_r = nb_r + self.T0.backward_primitive(et_hat, (0, 1, 0))
        nb_t = -self.T0.backward_primitive(er_hat, (0, 1, 0))
        nb_t = nb_t + self.T0.backward_primitive(ez_hat, (0, 0, 1))
        nb_z = -self.T0.backward_primitive(et_hat, (0, 0, 1)) - eps_t * self.inv_r
        nb_z = nb_z + self.inv_r * self.T0.backward_primitive(er_hat, (1, 0, 0))
        return (
            self.TD.scalar_product(nu_r),
            self.TD.scalar_product(nu_t),
            self.TD.scalar_product(nu_z),
            self.TD.scalar_product(nb_r),
            self.Tbt.scalar_product(nb_t),
            self.Tbz.scalar_product(nb_z),
        )

    def seed_linear_eigenmode(
        self, m: int = 1, kz_mode: int = 1, amp: float = 1.0e-6, which: int = 0
    ) -> tuple[AxisymmetricMRIState, complex]:
        _require_resolved_m(m, self.Ntheta)
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = TaylorCouetteMRIJax(
            self.base,
            B0=self.B0,
            nu=self.nu,
            eta_mag=self.eta_mag,
            N=self.Nr,
            family=self.family,
            magnetic_bc="conducting",
        )
        w, vecs = lin.eigs(m=m, kz=kz, n_return=which + 1)
        vec = vecs[:, which]
        n = lin.n
        state = self.zero_state()
        comps = list(state.x)
        mpos = int(m) % self.Ntheta
        mneg = (-int(m)) % self.Ntheta
        kpos = int(kz_mode) % self.Nz
        kneg = (-int(kz_mode)) % self.Nz
        block_map = (0, 1, 2, 4, 5, 6)
        for comp, block_index in enumerate(block_map):
            block = jnp.asarray(vec[block_index * n : (block_index + 1) * n]) * amp
            arr = jnp.zeros_like(comps[comp])
            if mpos == mneg and kpos == kneg:
                arr = arr.at[mpos, kpos, :n].set(jnp.real(block))
            else:
                arr = arr.at[mpos, kpos, :n].set(0.5 * block)
                arr = arr.at[mneg, kneg, :n].set(0.5 * jnp.conj(block))
            comps[comp] = arr
        nold = tuple(jnp.zeros_like(xi) for xi in comps)
        return AxisymmetricMRIState(tuple(comps), state.p, nold, False), complex(
            w[which]
        )

    def velocity_divergence(self, state: AxisymmetricMRIState) -> Array:
        dur_dr = self.TD.backward_primitive(state.x[0], (0, 0, 1))
        dut_dt = self.TD.backward_primitive(state.x[1], (1, 0, 0))
        duz_dz = self.TD.backward_primitive(state.x[2], (0, 1, 0))
        ur = self.TD.backward(state.x[0])
        return dur_dr + ur * self.inv_r + dut_dt * self.inv_r + duz_dz

    def magnetic_divergence(self, state: AxisymmetricMRIState) -> Array:
        dbr_dr = self.TD.backward_primitive(state.x[3], (0, 0, 1))
        dbt_dt = self.Tbt.backward_primitive(state.x[4], (1, 0, 0))
        dbz_dz = self.Tbz.backward_primitive(state.x[5], (0, 1, 0))
        br = self.TD.backward(state.x[3])
        return dbr_dr + br * self.inv_r + dbt_dt * self.inv_r + dbz_dz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nr", type=int, default=24)
    parser.add_argument("--Ntheta", type=int, default=0)
    parser.add_argument("--Nz", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--nu", type=float, default=1.0e-2)
    parser.add_argument("--eta-mag", type=float, default=1.0e-3)
    parser.add_argument("--B0", type=float, default=0.1)
    parser.add_argument("--family", choices=("L", "C"), default="L")
    parser.add_argument("--dealias", type=float, default=1.5)
    parser.add_argument("--mhd", action="store_true")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--kz-mode", type=int, default=1)
    parser.add_argument("--amp", type=float, default=1.0e-6)
    parser.add_argument("--seed-linear", action="store_true")
    args = parser.parse_args()

    eta = 0.5
    omega2 = eta**1.5 if args.mhd else 0.0
    base = CircularCouette(1.0, 2.0, 1.0, omega2)
    common = dict(
        base=base,
        nu=args.nu,
        Nr=args.Nr,
        Nz=args.Nz,
        dt=args.dt,
        family=args.family,
        dealias=args.dealias,
    )
    if args.mhd:
        mhd_common = dict(B0=args.B0, eta_mag=args.eta_mag)
        if args.Ntheta > 0:
            solver = TaylorCouetteMRIDNSJax(
                **common, Ntheta=args.Ntheta, **mhd_common
            )
            state = (
                solver.seed_linear_eigenmode(
                    m=args.m, kz_mode=args.kz_mode, amp=args.amp
                )[0]
                if args.seed_linear
                else solver.zero_state()
            )
        else:
            solver = AxisymmetricMRIDNSJax(**common, **mhd_common)
            state = (
                solver.seed_linear_eigenmode(kz_mode=args.kz_mode, amp=args.amp)[0]
                if args.seed_linear
                else solver.zero_state()
            )
    elif args.Ntheta > 0:
        solver = TaylorCouetteDNSJax(**common, Ntheta=args.Ntheta)
        state = (
            solver.seed_linear_eigenmode(
                m=args.m, kz_mode=args.kz_mode, amp=args.amp
            )[0]
            if args.seed_linear
            else solver.initial_state(amp=1.0e-3, m=args.m, kz_mode=args.kz_mode)
        )
    else:
        solver = AxisymmetricTCDNSJax(**common)
        state = (
            solver.seed_linear_eigenmode(kz_mode=args.kz_mode, amp=args.amp)[0]
            if args.seed_linear
            else solver.initial_state()
        )

    state = solver.solve(state, args.steps)
    diag = solver.diagnostics(state)
    print(" ".join(f"{key}={float(value):.6e}" for key, value in diag.items()))


if __name__ == "__main__":
    main()
