import jax
import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_minimal_seed_jax import (
    jax_complex_directional_derivative,
    normalize_to_energy,
    tree_l2_norm,
)
from production.objectives import (
    final_energy_objective,
    growth_rate_proxy_objective,
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
