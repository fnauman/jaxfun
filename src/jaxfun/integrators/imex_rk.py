"""Additive IMEX Runge-Kutta integrators.

The tableaux mirror shenfun/shenfun/utilities/integrators.py classes
IMEXRK011, IMEXRK111, IMEXRK222, IMEXRK443 and IMEXRK3.  The
stage contract follows COUETTE_IMPLEMENTATION_PLAN.md section 10.1.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from jaxfun.typing import Array, Padding

from .base import BaseIntegrator


class PDEIMEXRK(BaseIntegrator):
    """ARS-style IMEX Runge-Kutta integrator for Galerkin systems."""

    A: ClassVar[tuple[tuple[float, ...], ...]]
    B: ClassVar[tuple[tuple[float, ...], ...]]
    C: ClassVar[tuple[float, ...]]

    _system_operator = None

    @classmethod
    def stages(cls) -> tuple[Array, Array, Array]:
        """Return (a, b, c) as JAX arrays."""
        return (
            jnp.asarray(cls.A, dtype=float),
            jnp.asarray(cls.B, dtype=float),
            jnp.asarray(cls.C, dtype=float),
        )

    @classmethod
    def steps(cls) -> int:
        return len(cls.C) - 1

    @classmethod
    def _active_diagonal(cls) -> float:
        a, _, _ = cls.stages()
        if cls.steps() == 0:
            return 0.0
        diag = jnp.diag(a)[1:]
        active = float(diag[0])
        if not bool(jnp.allclose(diag, active)):
            raise ValueError("ARS IMEX RK table requires one active DIRK diagonal")
        return active

    def setup(self, dt: float) -> None:
        """Precompute the single DIRK implicit operator for this step size."""
        self._system_operator = None
        gamma = self._active_diagonal()
        if gamma == 0 or self.linear_operator.is_zero:
            return
        operator = self.build_implicit_operator(gamma, dt)
        if operator is not None:
            self._system_operator = nnx.data(operator)

    def _solve_stage(self, rhs: Array) -> Array:
        if self._system_operator is not None:
            return self._system_operator.solve(rhs)
        return self.apply_mass_inverse(rhs)

    def _mask_rhs(self, rhs: Array) -> Array:
        mask_nyquist = getattr(self.functionspace, "mask_nyquist", None)
        return rhs if mask_nyquist is None else mask_nyquist(rhs)

    @jax.jit(static_argnums=(0, 3))
    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
        """Advance one IMEX-RK step in coefficient space."""
        a, b, _ = self.stages()
        steps = self.steps()
        u0_rhs = self.apply_mass(u_hat)
        u_stage = u_hat
        nonlinear = []
        linear = []

        for rk in range(steps):
            nonlinear.append(self.nonlinear_rhs_scalar_product(u_stage, N))
            rhs = u0_rhs
            for j in range(rk + 1):
                rhs = rhs + dt * b[rk + 1, j] * nonlinear[j]
            if rk > 0:
                linear.append(self.apply_linear_scalar_product(u_stage))
                for j in range(rk):
                    rhs = rhs + dt * a[rk + 1, j + 1] * linear[j]
            u_stage = self._solve_stage(self._mask_rhs(rhs))
        return u_stage


class IMEXRK3(BaseIntegrator):
    """Spalart low-storage third-order IMEX Runge-Kutta integrator.

    Reference: shenfun/shenfun/utilities/integrators.py:603-699.
    Unlike the ARS schemes above, IMEXRK3 has one implicit operator per stage.
    """

    A: ClassVar[tuple[float, ...]] = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
    B: ClassVar[tuple[float, ...]] = (0.0, -17.0 / 60.0, -5.0 / 12.0)
    C: ClassVar[tuple[float, ...]] = (0.0, 8.0 / 15.0, 2.0 / 3.0, 1.0)

    _system_operators = None

    @classmethod
    def stages(cls) -> tuple[Array, Array, Array]:
        return (
            jnp.asarray(cls.A, dtype=float),
            jnp.asarray(cls.B, dtype=float),
            jnp.asarray(cls.C, dtype=float),
        )

    @classmethod
    def steps(cls) -> int:
        return len(cls.A)

    def setup(self, dt: float) -> None:
        """Precompute the three stage-dependent implicit operators."""
        self._system_operators = None
        if self.linear_operator.is_zero:
            return

        a, b, _ = self.stages()
        operators = []
        for rk in range(self.steps()):
            operator = self.build_implicit_operator(float((a[rk] + b[rk]) / 2.0), dt)
            if operator is not None:
                operators.append(operator)
        self._system_operators = nnx.data(tuple(operators))

    def _solve_stage(self, rhs: Array, rk: int) -> Array:
        if self._system_operators is not None:
            return self._system_operators[rk].solve(rhs)
        return self.apply_mass_inverse(rhs)

    def _mask_rhs(self, rhs: Array) -> Array:
        mask_nyquist = getattr(self.functionspace, "mask_nyquist", None)
        return rhs if mask_nyquist is None else mask_nyquist(rhs)

    @jax.jit(static_argnums=(0, 3))
    def step(self, u_hat: Array, dt: float, N: Padding = None) -> Array:
        """Advance one IMEXRK3 step in coefficient space."""
        a, b, _ = self.stages()
        u_stage = u_hat
        w_prev = jnp.zeros_like(u_hat)

        for rk in range(self.steps()):
            w0 = self.nonlinear_rhs_scalar_product(u_stage, N)
            gamma = (a[rk] + b[rk]) * dt / 2.0
            rhs = self.apply_mass(u_stage) + gamma * (
                self.apply_linear_scalar_product(u_stage)
            )
            rhs = rhs + dt * (a[rk] * w0 + b[rk] * w_prev)
            u_stage = self._solve_stage(self._mask_rhs(rhs), rk)
            w_prev = w0
        return u_stage


class IMEXRK011(PDEIMEXRK):
    A = ((0.0, 0.0), (0.0, 0.0))
    B = ((0.0, 0.0), (1.0, 0.0))
    C = (1.0, 0.0)


class IMEXRK111(PDEIMEXRK):
    A = ((0.0, 0.0), (0.0, 1.0))
    B = ((0.0, 0.0), (1.0, 0.0))
    C = (0.0, 1.0)


class IMEXRK222(PDEIMEXRK):
    gamma = (2.0 - 2.0**0.5) / 2.0
    delta = 1.0 - 1.0 / (2.0 * gamma)
    A = ((0.0, 0.0, 0.0), (0.0, gamma, 0.0), (0.0, 1.0 - gamma, gamma))
    B = ((0.0, 0.0, 0.0), (gamma, 0.0, 0.0), (delta, 1.0 - delta, 0.0))
    C = (0.0, gamma, 1.0)


class IMEXRK443(PDEIMEXRK):
    A = (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0 / 2.0, 0.0, 0.0, 0.0),
        (0.0, 1.0 / 6.0, 1.0 / 2.0, 0.0, 0.0),
        (0.0, -1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 0.0),
        (0.0, 3.0 / 2.0, -3.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0),
    )
    B = (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0 / 2.0, 0.0, 0.0, 0.0, 0.0),
        (11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0),
        (5.0 / 6.0, -5.0 / 6.0, 1.0 / 2.0, 0.0, 0.0),
        (1.0 / 4.0, 7.0 / 4.0, 3.0 / 4.0, -7.0 / 4.0, 0.0),
    )
    C = (0.0, 1.0 / 2.0, 2.0 / 3.0, 1.0 / 2.0, 1.0)
