"""Reusable spectral diagnostics for Couette-style solvers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import Array

from jaxfun.galerkin.inner import integrate


def quadratic_energy(
    components: Sequence[Array],
    space: Any,
    *,
    weight: Array | float = 1.0,
    factor: float = 0.5,
) -> Array:
    """Return ``factor * integral(sum(|u_i|^2) * weight)``.

    The explicit ``weight`` argument is used for cylindrical Taylor-Couette
    diagnostics where the volume element is ``r dr dz`` or ``r dr dz dtheta``.
    """
    if not components:
        raise ValueError("at least one component is required")
    density = jnp.zeros_like(jnp.real(components[0]))
    for component in components:
        density = density + jnp.real(jnp.conj(component) * component)
    return jnp.real(integrate(factor * density * weight, space))


def cylindrical_kinetic_energy(
    velocity: Sequence[Array], r: Array, space: Any
) -> Array:
    """Kinetic energy with explicit cylindrical volume element ``r``."""
    return quadratic_energy(velocity, space, weight=r, factor=0.5)


def cylindrical_magnetic_energy(
    magnetic: Sequence[Array], r: Array, space: Any
) -> Array:
    """Magnetic energy with explicit cylindrical volume element ``r``."""
    return quadratic_energy(magnetic, space, weight=r, factor=0.5)


def cylindrical_component_energy(component: Array, r: Array, space: Any) -> Array:
    """Unhalved component energy used by the TC azimuthal-energy diagnostic."""
    return quadratic_energy((component,), space, weight=r, factor=1.0)


def cylindrical_energy_parts(
    velocity: Sequence[Array], magnetic: Sequence[Array], r: Array, space: Any
) -> tuple[Array, Array]:
    """Return kinetic and magnetic cylindrical energies."""
    return (
        cylindrical_kinetic_energy(velocity, r, space),
        cylindrical_magnetic_energy(magnetic, r, space),
    )


def wall_linf(components: Sequence[Array], *, radial_axis: int = -1) -> Array:
    """Return the largest absolute boundary value on the radial walls."""
    if not components:
        raise ValueError("at least one component is required")
    maxima = []
    for component in components:
        maxima.append(jnp.max(jnp.abs(jnp.take(component, 0, axis=radial_axis))))
        maxima.append(jnp.max(jnp.abs(jnp.take(component, -1, axis=radial_axis))))
    return jnp.max(jnp.asarray(maxima))
