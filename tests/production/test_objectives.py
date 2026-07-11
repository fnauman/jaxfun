import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_minimal_seed_jax import (
    jax_complex_directional_derivative,
    minimal_seed_ascent,
    normalize_to_energy,
    tree_l2_norm,
)
from production.objectives import (
    _domain_weights,
    final_energy_objective,
    finite_difference_parameter_sensitivity,
    growth_rate_proxy_objective,
    maxwell_stress_objective,
    minimal_seed_gain_objective,
    minimal_seed_value_and_projected_gradient,
    reynolds_stress_objective,
    time_integrated_energy_objective,
    transport_alpha_objective,
)


def _pcf_initial_state_with_amp(solver: PlaneCouetteFluctuationJax, amp):
    x, y, z = solver.X
    wall = 1.0 - x**2
    ly = solver.domain[1][1] - solver.domain[1][0]
    lz = solver.domain[2][1] - solver.domain[2][0]
    u0 = amp * wall * jnp.sin(2.0 * jnp.pi * y / ly) * jnp.cos(2.0 * jnp.pi * z / lz)
    u1 = amp * wall * jnp.cos(2.0 * jnp.pi * y / ly) * jnp.sin(2.0 * jnp.pi * z / lz)
    u2 = amp * wall * jnp.sin(4.0 * jnp.pi * y / ly) * jnp.cos(4.0 * jnp.pi * z / lz)
    return solver.state_from_physical((u0, u1, u2))


def _central_difference(fun, x0: float, eps: float):
    return (fun(x0 + eps) - fun(x0 - eps)) / (2.0 * eps)


@pytest.fixture
def pcf_solver():
    return PlaneCouetteFluctuationJax(
        N=(7, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )


pytestmark = pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="production objective finite differences use x64",
)


def test_reynolds_stress_objective_uses_nonuniform_grid_weights():
    class FakeSolver:
        def __init__(self):
            x = jnp.asarray([0.0, 0.2, 1.0])
            y = jnp.asarray([0.0, 1.0])
            z = jnp.asarray([0.0, 1.0])
            self.X = jnp.meshgrid(x, y, z, indexing="ij")

        def velocity_physical(self, state):
            x = self.X[0]
            ones = jnp.ones_like(x)
            return (x, ones, jnp.zeros_like(x))

    value = reynolds_stress_objective(FakeSolver(), object(), steps=0)

    assert jnp.allclose(value, 0.5, rtol=1.0e-6, atol=1.0e-7)
    assert not jnp.allclose(value, 0.4, rtol=1.0e-6, atol=1.0e-7)


def test_periodic_axis_weights_span_full_domain():
    class FakeSolver:
        def __init__(self):
            x = jnp.asarray([0.0, 0.5, 1.0])
            y = jnp.asarray([0.0, 0.25, 0.5, 0.75])
            z = jnp.asarray([0.0, 1.0])
            self.domain = ((0.0, 1.0), (0.0, 1.0), (0.0, 2.0))
            self.X = jnp.meshgrid(x, y, z, indexing="ij")

    field = jnp.ones((3, 4, 2))
    weights = _domain_weights(FakeSolver(), field)

    assert jnp.allclose(jnp.sum(weights), 2.0, rtol=1.0e-12, atol=1.0e-12)


def test_taylor_couette_weights_use_radial_jacobian_without_x_mesh():
    class FakeTCSolver:
        def __init__(self):
            r = jnp.asarray([1.0, 2.0, 3.0])
            z = jnp.asarray([0.0, 0.5, 1.0, 1.5])
            self.R, self.Z = jnp.meshgrid(r, z, indexing="ij")
            self.Lz = 2.0

    solver = FakeTCSolver()
    weights = _domain_weights(solver, solver.R)
    weighted_mean_radius = jnp.sum(solver.R * weights) / jnp.sum(weights)

    assert jnp.allclose(weighted_mean_radius, 2.25, rtol=1.0e-12, atol=1.0e-12)
    assert not jnp.allclose(weighted_mean_radius, 2.0, rtol=1.0e-12, atol=1.0e-12)


def test_final_energy_objective_gradient_matches_finite_difference(pcf_solver):
    def objective(amp):
        state = _pcf_initial_state_with_amp(pcf_solver, amp)
        return final_energy_objective(pcf_solver, state, steps=1)

    amp0 = 0.02
    grad = jax.grad(objective)(amp0)
    fd = _central_difference(objective, amp0, 1.0e-5)

    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=2.0e-3, atol=1.0e-8)


def test_time_integrated_energy_objective_is_jittable_and_differentiable(pcf_solver):
    def objective(amp):
        state = _pcf_initial_state_with_amp(pcf_solver, amp)
        return time_integrated_energy_objective(pcf_solver, state, steps=2)

    amp0 = 0.02
    value = objective(amp0)
    jitted_value = jax.jit(objective)(amp0)
    grad = jax.grad(objective)(amp0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    assert jnp.allclose(value, jitted_value, rtol=1.0e-12, atol=1.0e-14)


def test_growth_rate_proxy_objective_is_finite(pcf_solver):
    state = _pcf_initial_state_with_amp(pcf_solver, 0.02)
    growth = growth_rate_proxy_objective(pcf_solver, state, steps=2)

    assert jnp.isfinite(growth)


def test_reynolds_stress_objective_gradient_matches_finite_difference(pcf_solver):
    def objective(amp):
        state = _pcf_initial_state_with_amp(pcf_solver, amp)
        return reynolds_stress_objective(pcf_solver, state, steps=1)

    amp0 = 0.02
    grad = jax.grad(objective)(amp0)
    fd = _central_difference(objective, amp0, 1.0e-5)
    state = _pcf_initial_state_with_amp(pcf_solver, amp0)
    alpha = transport_alpha_objective(pcf_solver, state, steps=1, pressure=2.0)

    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=5.0e-3, atol=1.0e-8)
    assert jnp.allclose(alpha, objective(amp0) / 2.0, rtol=1.0e-12, atol=1.0e-14)


def test_stress_objectives_prefer_solver_exact_quadrature():
    class ExactStressSolver:
        dt = 1.0

        def stresses(self, state):
            return (jnp.asarray(2.5), jnp.asarray(-0.75))

        def velocity_physical(self, state):
            field = jnp.ones((2, 2))
            return (field, field, field)

        def fields_physical(self, state):
            field = jnp.ones((2, 2))
            return (field, field, field, field, field, field)

    solver = ExactStressSolver()

    assert reynolds_stress_objective(solver, object()) == pytest.approx(2.5)
    assert maxwell_stress_objective(solver, object()) == pytest.approx(-0.75)


def test_minimal_seed_gain_and_projected_gradient_are_finite_and_tangent(pcf_solver):
    state = _pcf_initial_state_with_amp(pcf_solver, 0.02)
    target_energy = 1.0e-3
    normalized = normalize_to_energy(pcf_solver, state, target_energy)

    gain = minimal_seed_gain_objective(
        pcf_solver, state, steps=1, target_energy=target_energy
    )
    projected_gain, projected_grad = minimal_seed_value_and_projected_gradient(
        pcf_solver, state, steps=1, target_energy=target_energy
    )
    energy_grad = jax.grad(pcf_solver.perturbation_energy)(normalized)

    assert jnp.isfinite(gain)
    assert jnp.allclose(gain, projected_gain, rtol=1.0e-12, atol=1.0e-12)
    assert tree_l2_norm(projected_grad) > 0.0
    assert all(
        bool(jnp.isfinite(leaf).all())
        for leaf in jax.tree_util.tree_leaves(projected_grad)
    )
    assert jnp.allclose(
        jax_complex_directional_derivative(energy_grad, projected_grad),
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )


def test_transport_alpha_does_not_swallow_mhd_field_errors():
    class BrokenMHDSolver:
        dt = 1.0

        def velocity_physical(self, state):
            field = jnp.ones((2, 2))
            return (field, field, field)

        def fields_physical(self, state):
            raise AttributeError("internal field bug")

    with pytest.raises(AttributeError, match="internal field bug"):
        transport_alpha_objective(BrokenMHDSolver(), object(), pressure=1.0)


def test_finite_difference_parameter_sensitivity_is_explicit_static_contract():
    value = finite_difference_parameter_sensitivity(lambda nu: nu * nu, 3.0)

    assert value == pytest.approx(6.0, rel=1.0e-6, abs=1.0e-8)


def test_minimal_seed_ascent_keeps_energy_constraint(pcf_solver):
    state = _pcf_initial_state_with_amp(pcf_solver, 0.02)

    optimized, history = minimal_seed_ascent(
        pcf_solver,
        state,
        target_energy=1.0e-3,
        steps=1,
        iterations=2,
        step_size=1.0e-4,
    )

    assert history
    assert jnp.allclose(
        pcf_solver.perturbation_energy(optimized),
        1.0e-3,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert all(item.gain > 0.0 for item in history)
