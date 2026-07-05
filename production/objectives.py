"""Differentiable production objectives for jaxfun solver states.

These helpers are intentionally thin: they call the solver's own ``solve`` /
``step`` and diagnostic methods so gradients flow through the same path used by
production runs. The ``steps`` arguments are Python-static by design.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from examples.pcf_minimal_seed_jax import (
    gain_and_projected_gradient,
    normalize_to_energy,
    perturbation_gain,
)


def final_energy_objective(solver: Any, state: Any, *, steps: int) -> Array:
    """Return final perturbation/total energy after a static number of steps."""

    _require_x64_enabled()
    return _state_energy(solver, _advance_state(solver, state, steps))


def time_integrated_energy_objective(solver: Any, state: Any, *, steps: int) -> Array:
    """Return trapezoidal time integral of solver energy over ``steps``."""

    _require_x64_enabled()
    steps_int = _validate_steps(steps)
    dt = _solver_dt(solver)
    current = state
    previous_energy = _state_energy(solver, current)
    total = jnp.zeros((), dtype=previous_energy.dtype)
    for _ in range(steps_int):
        current = _step_state(solver, current)
        next_energy = _state_energy(solver, current)
        total = total + 0.5 * (previous_energy + next_energy) * dt
        previous_energy = next_energy
    return total


def growth_rate_proxy_objective(solver: Any, state: Any, *, steps: int) -> Array:
    """Return ``0.5 * log(E_final / E_initial) / elapsed_time``."""

    _require_x64_enabled()
    steps_int = _validate_steps(steps, allow_zero=False)
    initial_energy = _state_energy(solver, state)
    final_energy = final_energy_objective(solver, state, steps=steps_int)
    _raise_if_not_positive_concrete(initial_energy, "initial energy")
    _raise_if_not_positive_concrete(final_energy, "final energy")
    elapsed = _solver_dt(solver) * steps_int
    return 0.5 * jnp.log(final_energy / initial_energy) / elapsed


def reynolds_stress_objective(
    solver: Any,
    state: Any,
    *,
    steps: int = 0,
    components: tuple[int, int] = (0, 1),
    subtract_mean: bool = False,
) -> Array:
    """Return the domain-mean Reynolds stress for a velocity component pair."""

    _require_x64_enabled()
    out = _advance_state(solver, state, steps)
    exact = _exact_solver_stress(
        solver, out, components=components, subtract_mean=subtract_mean, index=0
    )
    if exact is not None:
        return exact
    velocity = velocity_fields(solver, out)
    left = velocity[components[0]]
    right = velocity[components[1]]
    weights = _domain_weights(solver, left)
    if subtract_mean:
        left = left - _weighted_mean(left, weights)
        right = right - _weighted_mean(right, weights)
    return jnp.real(_weighted_mean(left * jnp.conj(right), weights))


def maxwell_stress_objective(
    solver: Any,
    state: Any,
    *,
    steps: int = 0,
    components: tuple[int, int] = (0, 1),
    subtract_mean: bool = False,
) -> Array:
    """Return ``-mean(B_i B_j)`` for a magnetic component pair."""

    _require_x64_enabled()
    out = _advance_state(solver, state, steps)
    exact = _exact_solver_stress(
        solver, out, components=components, subtract_mean=subtract_mean, index=1
    )
    if exact is not None:
        return exact
    magnetic = magnetic_fields(solver, out)
    left = magnetic[components[0]]
    right = magnetic[components[1]]
    weights = _domain_weights(solver, left)
    if subtract_mean:
        left = left - _weighted_mean(left, weights)
        right = right - _weighted_mean(right, weights)
    return -jnp.real(_weighted_mean(left * jnp.conj(right), weights))


def transport_alpha_objective(
    solver: Any,
    state: Any,
    *,
    steps: int = 0,
    pressure: Array | float = 1.0,
) -> Array:
    """Return Reynolds plus Maxwell stress normalized by pressure."""

    _require_x64_enabled()
    pressure_arr = jnp.asarray(pressure)
    _raise_if_not_zero_concrete(pressure_arr, "pressure")
    reynolds = reynolds_stress_objective(solver, state, steps=steps)
    maxwell = (
        maxwell_stress_objective(solver, state, steps=steps)
        if has_magnetic_fields(solver, state)
        else jnp.zeros((), dtype=reynolds.dtype)
    )
    return (reynolds + maxwell) / pressure_arr


def minimal_seed_gain_objective(
    solver: Any,
    state: Any,
    *,
    steps: int,
    target_energy: Array | float | None = None,
) -> Array:
    """Return PCF perturbation gain, optionally after fixed-energy normalization."""

    _require_x64_enabled()
    initial_state = state
    if target_energy is not None:
        initial_state = normalize_to_energy(solver, state, target_energy)
    return perturbation_gain(solver, initial_state, _validate_steps(steps))


def minimal_seed_value_and_projected_gradient(
    solver: Any,
    state: Any,
    *,
    steps: int,
    target_energy: Array | float | None = None,
) -> tuple[Array, Any]:
    """Return PCF gain and gradient projected onto the fixed-energy tangent."""

    _require_x64_enabled()
    initial_state = state
    if target_energy is not None:
        initial_state = normalize_to_energy(solver, state, target_energy)
    return gain_and_projected_gradient(solver, initial_state, _validate_steps(steps))


def _exact_solver_stress(
    solver: Any,
    state: Any,
    *,
    components: tuple[int, int],
    subtract_mean: bool,
    index: int,
) -> Array | None:
    if components != (0, 1) or subtract_mean or not hasattr(solver, "stresses"):
        return None
    stresses = tuple(solver.stresses(state))
    if len(stresses) <= index:
        return None
    return jnp.real(stresses[index])


def velocity_fields(solver: Any, state: Any) -> tuple[Array, ...]:
    """Return physical velocity components from supported solver interfaces."""

    if hasattr(solver, "velocity_physical"):
        return tuple(solver.velocity_physical(state))
    if hasattr(solver, "fields_physical"):
        fields = tuple(solver.fields_physical(state))
        if len(fields) >= 3:
            return fields[:3]
    if hasattr(solver, "total_velocity_physical"):
        return tuple(solver.total_velocity_physical(state))
    if hasattr(solver, "_backward_velocity") and hasattr(state, "u"):
        return tuple(solver._backward_velocity(state.u))
    raise AttributeError("solver does not expose physical velocity fields")


def magnetic_fields(solver: Any, state: Any) -> tuple[Array, ...]:
    """Return physical magnetic components from supported MHD solver interfaces."""

    if hasattr(solver, "fields_physical"):
        fields = tuple(solver.fields_physical(state))
        if len(fields) >= 6:
            return fields[3:6]
    raise AttributeError("solver does not expose physical magnetic fields")


def has_magnetic_fields(solver: Any, state: Any) -> bool:
    """Return whether the solver interface advertises magnetic fields.

    This probes only the declared field tuple shape. It deliberately does not
    catch errors raised while constructing fields, so broken MHD solvers fail
    loudly instead of being treated as hydrodynamic.
    """

    if not hasattr(solver, "fields_physical"):
        return False
    fields = tuple(solver.fields_physical(state))
    return len(fields) >= 6


def finite_difference_parameter_sensitivity(
    objective,
    parameter: float,
    *,
    step: float | None = None,
) -> float:
    """Return a central finite-difference sensitivity for static solver parameters.

    Parameter-dependent solver construction is intentionally outside JAX's traced
    state objective path. Use this helper for coarse production sensitivity checks
    where the callable rebuilds any solver objects from the supplied scalar.
    """

    _require_x64_enabled()
    value = float(parameter)
    delta = float(step) if step is not None else max(1.0e-6, abs(value) * 1.0e-6)
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError("finite-difference step must be positive and finite")
    return float((objective(value + delta) - objective(value - delta)) / (2.0 * delta))


def _require_x64_enabled() -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        raise RuntimeError("production objective helpers require jax_enable_x64=True")


def _domain_weights(solver: Any, field: Array) -> Array:
    if hasattr(solver, "X"):
        coords = tuple(solver.X)
        weights = jnp.ones_like(jnp.real(field))
        for axis, coord in enumerate(coords[: field.ndim]):
            axis_coord = _axis_coordinate(coord, axis)
            axis_weights = (
                _periodic_axis_weights(solver, axis_coord, axis)
                if _axis_is_periodic(solver, axis)
                else _trapezoid_axis_weights(axis_coord)
            )
            shape = [1] * field.ndim
            shape[axis] = axis_weights.shape[0]
            weights = weights * axis_weights.reshape(shape)
        return weights
    if hasattr(solver, "R") and hasattr(solver, "Z"):
        return _taylor_couette_domain_weights(solver, field)
    return jnp.ones_like(jnp.real(field))


def _axis_coordinate(coord: Any, axis: int) -> Array:
    arr = jnp.asarray(coord)
    if arr.ndim == 1:
        return arr
    index = [0] * arr.ndim
    index[axis] = slice(None)
    return arr[tuple(index)]


def _axis_is_periodic(solver: Any, axis: int) -> bool:
    domain = getattr(solver, "domain", None)
    if isinstance(domain, tuple | list) and axis < len(domain):
        return axis > 0 and _period_from_domain_entry(domain[axis]) is not None
    return False


def _period_from_domain_entry(entry: Any) -> float | None:
    if not isinstance(entry, tuple | list) or len(entry) != 2:
        return None
    try:
        lo = float(entry[0])
        hi = float(entry[1])
    except (TypeError, ValueError):
        return None
    period = hi - lo
    return period if math.isfinite(period) and period > 0.0 else None


def _periodic_axis_weights(solver: Any, coord: Array, axis: int) -> Array:
    x = jnp.asarray(coord)
    domain = getattr(solver, "domain", None)
    period = None
    if isinstance(domain, tuple | list) and axis < len(domain):
        period = _period_from_domain_entry(domain[axis])
    if period is None and x.shape[0] > 1:
        period = float((x[1] - x[0]) * x.shape[0])
    if period is None:
        return jnp.ones_like(x)
    return jnp.ones_like(x) * (period / x.shape[0])


def _taylor_couette_domain_weights(solver: Any, field: Array) -> Array:
    real_field = jnp.real(field)
    r_coord = _axis_coordinate(solver.R, 0)
    r_weights = _trapezoid_axis_weights(r_coord) * r_coord
    weights = r_weights.reshape((r_weights.shape[0],) + (1,) * (field.ndim - 1))
    if field.ndim >= 2:
        z_axis = 1
        z_coord = _axis_coordinate(solver.Z, z_axis)
        z_weights = _taylor_couette_z_weights(solver, z_coord)
        shape = [1] * field.ndim
        shape[z_axis] = z_weights.shape[0]
        weights = weights * z_weights.reshape(shape)
    return jnp.ones_like(real_field) * weights


def _taylor_couette_z_weights(solver: Any, z_coord: Array) -> Array:
    if hasattr(solver, "Lz"):
        length = float(solver.Lz)
        if math.isfinite(length) and length > 0.0:
            return jnp.ones_like(z_coord) * (length / z_coord.shape[0])
    return _trapezoid_axis_weights(z_coord)


def _trapezoid_axis_weights(coord: Array) -> Array:
    x = jnp.asarray(coord)
    if x.ndim != 1 or x.shape[0] < 2:
        return jnp.ones_like(x)
    dx = jnp.diff(x)
    interior = 0.5 * (dx[:-1] + dx[1:])
    return jnp.concatenate((0.5 * dx[:1], interior, 0.5 * dx[-1:]))


def _weighted_mean(values: Array, weights: Array) -> Array:
    real_weights = jnp.asarray(weights, dtype=jnp.real(values).dtype)
    return jnp.sum(values * real_weights) / jnp.sum(real_weights)


def _advance_state(solver: Any, state: Any, steps: int) -> Any:
    steps_int = _validate_steps(steps)
    if steps_int == 0:
        return state
    if hasattr(solver, "solve"):
        return solver.solve(state, steps_int)
    current = state
    for _ in range(steps_int):
        current = _step_state(solver, current)
    return current


def _step_state(solver: Any, state: Any) -> Any:
    if hasattr(solver, "step"):
        return solver.step(state)
    if hasattr(solver, "solve"):
        return solver.solve(state, 1)
    raise AttributeError("solver does not expose step or solve")


def _state_energy(solver: Any, state: Any) -> Array:
    if hasattr(solver, "perturbation_energy"):
        return solver.perturbation_energy(state)
    if hasattr(solver, "energy"):
        return solver.energy(state)
    if hasattr(solver, "diagnostics"):
        diagnostics = solver.diagnostics(state)
        for key in ("kinetic_energy", "Epert", "Ekin", "E"):
            if key in diagnostics:
                return diagnostics[key]
    raise AttributeError("solver does not expose an energy diagnostic")


def _solver_dt(solver: Any) -> Array:
    if not hasattr(solver, "dt"):
        raise AttributeError("solver does not expose dt")
    return jnp.asarray(solver.dt)


def _validate_steps(steps: int, *, allow_zero: bool = True) -> int:
    steps_int = int(steps)
    if steps_int < 0 or (steps_int == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"steps must be {comparator}")
    return steps_int


def _raise_if_not_positive_concrete(value: Array | float, name: str) -> None:
    try:
        concrete = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(concrete) or concrete <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def _raise_if_not_zero_concrete(value: Array | float, name: str) -> None:
    try:
        concrete = float(value)
    except (TypeError, ValueError):
        return
    if concrete == 0.0:
        raise ValueError(f"{name} must be nonzero")


__all__ = [
    "final_energy_objective",
    "growth_rate_proxy_objective",
    "finite_difference_parameter_sensitivity",
    "has_magnetic_fields",
    "magnetic_fields",
    "maxwell_stress_objective",
    "minimal_seed_gain_objective",
    "minimal_seed_value_and_projected_gradient",
    "reynolds_stress_objective",
    "time_integrated_energy_objective",
    "transport_alpha_objective",
    "velocity_fields",
]
