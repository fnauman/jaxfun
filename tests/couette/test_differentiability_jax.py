import jax
import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_minimal_seed_jax import (
    gain_and_projected_gradient,
    jax_complex_directional_derivative,
    normalize_to_energy,
    tree_add_scaled,
    tree_l2_norm,
)
from examples.taylor_couette_dns_jax import AxisymmetricTCDNSJax, CircularCouette


def _pcf_initial_state_with_amp(solver: PlaneCouetteFluctuationJax, amp):
    x, y, z = solver.X
    wall = 1.0 - x**2
    Ly = solver.domain[1][1] - solver.domain[1][0]
    Lz = solver.domain[2][1] - solver.domain[2][0]
    u0 = amp * wall * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.cos(
        2.0 * jnp.pi * z / Lz
    )
    u1 = amp * wall * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.sin(
        2.0 * jnp.pi * z / Lz
    )
    u2 = amp * wall * jnp.sin(4.0 * jnp.pi * y / Ly) * jnp.cos(
        4.0 * jnp.pi * z / Lz
    )
    return solver.state_from_physical((u0, u1, u2))


def _central_difference(fun, x0: float, eps: float):
    return (fun(x0 + eps) - fun(x0 - eps)) / (2.0 * eps)


def _pcf_direction_state(solver: PlaneCouetteFluctuationJax, amp):
    x, y, z = solver.X
    wall = 1.0 - x**2
    Ly = solver.domain[1][1] - solver.domain[1][0]
    Lz = solver.domain[2][1] - solver.domain[2][0]
    u0 = amp * wall * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.cos(2.0 * jnp.pi * z / Lz)
    u1 = amp * wall * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.sin(2.0 * jnp.pi * z / Lz)
    u2 = amp * wall * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.cos(2.0 * jnp.pi * z / Lz)
    return solver.state_from_physical((u0, u1, u2))


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Couette differentiability checks use x64 finite differences",
)
def test_pcf_energy_gradient_wrt_initial_amplitude_matches_finite_difference():
    solver = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=2.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )

    def final_energy(amp):
        state = _pcf_initial_state_with_amp(solver, amp)
        out = solver.step(state)
        return solver.diagnostics(out)["Epert"]

    amp0 = 0.02
    grad = jax.grad(final_energy)(amp0)
    fd = _central_difference(final_energy, amp0, 1.0e-5)

    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=2.0e-3, atol=1.0e-8)


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Couette differentiability checks use x64 finite differences",
)
def test_pcf_full_initial_state_gradient_matches_directional_fd():
    solver = PlaneCouetteFluctuationJax(
        N=(7, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )
    state = _pcf_initial_state_with_amp(solver, 0.02)
    direction = _pcf_direction_state(solver, 0.01)

    def final_energy(initial_state):
        out = solver.step(initial_state)
        return solver.perturbation_energy(out)

    grad_state = jax.grad(final_energy)(state)
    adjoint_directional = jax_complex_directional_derivative(grad_state, direction)
    fd = (
        final_energy(tree_add_scaled(state, direction, 1.0e-5))
        - final_energy(tree_add_scaled(state, direction, -1.0e-5))
    ) / 2.0e-5

    assert all(
        bool(jnp.isfinite(leaf).all()) for leaf in jax.tree_util.tree_leaves(grad_state)
    )
    assert jnp.allclose(adjoint_directional, fd, rtol=2.0e-3, atol=1.0e-8)


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Couette differentiability checks use x64 finite differences",
)
def test_pcf_zero_horizon_gain_has_zero_projected_gradient_on_energy_sphere():
    solver = PlaneCouetteFluctuationJax(
        N=(7, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )
    target_energy = 1.0e-3
    state = normalize_to_energy(
        solver, _pcf_initial_state_with_amp(solver, 0.02), target_energy
    )

    gain, projected_grad = gain_and_projected_gradient(solver, state, steps=0)
    energy_grad = jax.grad(solver.perturbation_energy)(state)

    assert jnp.allclose(
        solver.perturbation_energy(state), target_energy, rtol=1.0e-12, atol=1.0e-15
    )
    assert jnp.allclose(gain, 1.0, rtol=1.0e-12, atol=1.0e-12)
    assert tree_l2_norm(projected_grad) < 1.0e-10
    assert jnp.allclose(
        jax_complex_directional_derivative(energy_grad, projected_grad),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="Couette differentiability checks use x64 finite differences",
)
def test_axisymmetric_tc_energy_gradient_wrt_initial_amplitude_matches_fd():
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
    )

    def final_energy(amp):
        state = solver.initial_state(amp=amp)
        out = solver.solve(state, steps=2)
        return solver.energy(out)

    amp0 = 1.0e-4
    grad = jax.grad(final_energy)(amp0)
    fd = _central_difference(final_energy, amp0, 1.0e-6)

    assert jnp.isfinite(grad)
    assert jnp.allclose(grad, fd, rtol=2.0e-3, atol=1.0e-12)
