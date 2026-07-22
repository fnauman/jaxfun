"""Compact on-the-fly wavenumber prototype for the PCF ``Sg`` operator.

This research path stores the unfactored DIA rows per Fourier mode and creates
a dense pivoted solve inside the compiled call.  It is intentionally slower
than the production banded-LU path, but quantifies the persistent-memory floor
and provides a correctness oracle for a future ultraspherical recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxfun.la.diamatrix import DiaMatrix
from jaxfun.la.tpmatrix import (
    TPMatrices,
    TPMatricesWavenumberSolver,
    _dia_batch_to_dense,
)


@dataclass(frozen=True)
class StructuredPrototypeReport:
    shape: tuple[int, ...]
    compact_operator_bytes: int
    production_factor_bytes: int
    factor_to_compact_ratio: float
    relative_residual: float
    relative_solution_error: float


def _assemble_mode_data(operator: TPMatrices):
    terms = list(operator.tpmats)
    ndim = terms[0].dims
    diagonal_axes = [
        axis
        for axis in range(ndim)
        if all(set(cast(DiaMatrix, term.mats[axis]).offsets) == {0} for term in terms)
    ]
    polynomial_axes = [axis for axis in range(ndim) if axis not in diagonal_axes]
    if len(polynomial_axes) != 1:
        raise ValueError("prototype requires exactly one polynomial axis")
    poly_axis = polynomial_axes[0]
    shape = tuple(int(terms[0].mats[axis].shape[0]) for axis in range(ndim))
    n_poly = shape[poly_axis]
    dtype = jnp.result_type(*[term.mats[poly_axis].data.dtype for term in terms])
    weights = []
    for term in terms:
        weight = jnp.asarray(term.coefficient, dtype=dtype).reshape(1)
        for axis in diagonal_axes:
            diagonal = cast(DiaMatrix, term.mats[axis]).data[0]
            weight = jnp.outer(weight, diagonal).reshape(-1)
        weights.append(weight)
    offsets = tuple(
        sorted(
            {
                int(offset)
                for term in terms
                for offset in cast(DiaMatrix, term.mats[poly_axis]).offsets
            }
        )
    )
    polynomial_rows = []
    for term in terms:
        matrix = cast(DiaMatrix, term.mats[poly_axis])
        rows = [
            matrix.data[list(matrix.offsets).index(offset)]
            if offset in matrix.offsets
            else jnp.zeros(n_poly, dtype=dtype)
            for offset in offsets
        ]
        polynomial_rows.append(jnp.stack(rows))
    return (
        poly_axis,
        shape,
        offsets,
        jnp.stack(weights),
        jnp.stack(polynomial_rows),
    )


class CompactOnTheFlyWavenumberSolver:
    """Unfactored memory prototype; not intended as a production backend."""

    def __init__(self, operator: TPMatrices):
        self.operator = operator
        (
            self.poly_axis,
            self.shape,
            self.offsets,
            self.weights,
            self.polynomial_rows,
        ) = _assemble_mode_data(operator)
        self.fourier_axes = tuple(
            axis for axis in range(len(self.shape)) if axis != self.poly_axis
        )
        self.axes_order = (*self.fourier_axes, self.poly_axis)
        inverse = [0] * len(self.shape)
        for new, old in enumerate(self.axes_order):
            inverse[old] = new
        self.inverse_order = tuple(inverse)

    def solve(self, rhs: Array) -> Array:
        n_poly = self.shape[self.poly_axis]
        n_fourier = int(np.prod([self.shape[axis] for axis in self.fourier_axes]))
        mode_data = jnp.einsum("tf,tdp->fdp", self.weights, self.polynomial_rows)
        dense = _dia_batch_to_dense(mode_data, self.offsets, n_poly)
        rhs_modes = jnp.transpose(rhs, self.axes_order).reshape(n_fourier, n_poly)
        solution = jax.vmap(jnp.linalg.solve)(dense, rhs_modes)
        fourier_shape = tuple(self.shape[axis] for axis in self.fourier_axes)
        return jnp.transpose(
            solution.reshape((*fourier_shape, n_poly)), self.inverse_order
        )


def qualify_sg_prototype(solver, rhs: Array) -> StructuredPrototypeReport:
    """Compare the compact prototype with a solver's production ``Sg`` factor."""

    prototype = CompactOnTheFlyWavenumberSolver(solver.Sg)
    production = solver.Sg_factor
    if not isinstance(production, TPMatricesWavenumberSolver):
        raise TypeError("Sg must use TPMatricesWavenumberSolver")
    expected = production.solve(rhs)
    actual = prototype.solve(rhs)
    residual = solver.Sg @ actual - rhs
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs), 1.0e-300)
    solution_norm = jnp.maximum(jnp.linalg.norm(expected), 1.0e-300)
    compact_bytes = sum(
        int(value.size * value.dtype.itemsize)
        for value in (prototype.weights, prototype.polynomial_rows)
    )
    factor_bytes = sum(
        int(value.size * value.dtype.itemsize) for value in production.runtime_args()
    )
    return StructuredPrototypeReport(
        shape=prototype.shape,
        compact_operator_bytes=compact_bytes,
        production_factor_bytes=factor_bytes,
        factor_to_compact_ratio=factor_bytes / compact_bytes,
        relative_residual=float(jnp.linalg.norm(residual) / rhs_norm),
        relative_solution_error=float(
            jnp.linalg.norm(actual - expected) / solution_norm
        ),
    )
