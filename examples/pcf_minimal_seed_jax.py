"""Minimal-seed optimization helpers for Plane Couette flow.

The helpers here are intentionally small and operate on the existing KMMState
coefficient state. They provide the differentiable pieces needed by a direct
adjoint loop: fixed-energy normalization, perturbation gain, and tangent
projection on the initial-energy constraint.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from examples.channelflow_kmm import KMMState
from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax


def tree_scale(state: KMMState, scale: Array | float) -> KMMState:
    """Scale every coefficient array in a KMM state."""
    return jax.tree_util.tree_map(lambda x: scale * x, state)


def tree_add_scaled(
    state: KMMState, direction: KMMState, scale: Array | float
) -> KMMState:
    """Return ``state + scale * direction`` as a KMM state."""
    return jax.tree_util.tree_map(lambda x, dx: x + scale * dx, state, direction)


def tree_conj(state: KMMState) -> KMMState:
    """Return the elementwise complex conjugate of a KMM state."""
    return jax.tree_util.tree_map(jnp.conj, state)


def tree_real_vdot(left: KMMState, right: KMMState) -> Array:
    """Euclidean real inner product ``Re(sum(conj(left) * right))``."""
    return sum(
        jnp.real(jnp.vdot(x, y))
        for x, y in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    )


def tree_l2_norm(state: KMMState) -> Array:
    """Euclidean coefficient norm, used only as a numerical size diagnostic."""
    return jnp.sqrt(tree_real_vdot(state, state))


def _raise_if_not_positive_concrete(value: Array | float, name: str) -> None:
    """Raise for eager nonpositive scalars while staying compatible with grad."""
    try:
        concrete = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def jax_complex_directional_derivative(
    grad_state: KMMState, direction: KMMState
) -> Array:
    """Directional derivative pairing for real objectives with complex leaves.

    For a real-valued function of complex JAX arrays, ``jax.grad`` returns a
    gradient ``g`` such that the directional derivative along ``d`` is
    ``Re(sum(g * d))`` over all leaves.
    """
    return sum(
        jnp.real(jnp.sum(g * d))
        for g, d in zip(
            jax.tree_util.tree_leaves(grad_state),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )


def normalize_to_energy(
    solver: PlaneCouetteFluctuationJax,
    state: KMMState,
    target_energy: Array | float,
) -> KMMState:
    """Rescale an initial condition to a target perturbation energy."""
    energy = solver.perturbation_energy(state)
    target = jnp.asarray(target_energy, dtype=energy.dtype)
    _raise_if_not_positive_concrete(energy, "state energy")
    _raise_if_not_positive_concrete(target, "target energy")
    scale = jnp.sqrt(target / energy)
    return tree_scale(state, scale)


def perturbation_gain(
    solver: PlaneCouetteFluctuationJax,
    state: KMMState,
    steps: int,
) -> Array:
    """Return final perturbation energy divided by initial perturbation energy."""
    if int(steps) < 0:
        raise ValueError("steps must be non-negative")
    initial_energy = solver.perturbation_energy(state)
    _raise_if_not_positive_concrete(initial_energy, "initial energy")
    final_state = solver.solve(state, int(steps))
    return solver.perturbation_energy(final_state) / initial_energy


def project_to_energy_tangent(
    solver: PlaneCouetteFluctuationJax,
    state: KMMState,
    direction: KMMState,
) -> KMMState:
    """Project a direction onto the tangent space of constant initial energy."""
    energy_grad = jax.grad(solver.perturbation_energy)(state)
    energy_normal = tree_conj(energy_grad)
    numerator = jax_complex_directional_derivative(energy_grad, direction)
    denominator = tree_real_vdot(energy_normal, energy_normal)
    _raise_if_not_positive_concrete(denominator, "energy-gradient norm")
    return tree_add_scaled(direction, energy_normal, -numerator / denominator)


def gain_and_projected_gradient(
    solver: PlaneCouetteFluctuationJax,
    state: KMMState,
    steps: int,
) -> tuple[Array, KMMState]:
    """Return perturbation gain and its gradient projected to fixed energy."""

    def objective(initial_state: KMMState) -> Array:
        return perturbation_gain(solver, initial_state, steps)

    gain, gradient = jax.value_and_grad(objective)(state)
    return gain, project_to_energy_tangent(solver, state, gradient)
