import jax
import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
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
