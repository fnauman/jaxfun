import jax
import jax.numpy as jnp
import pytest

from examples.taylor_couette_dns_jax import (
    AxisymmetricMRIDNSJax,
    AxisymmetricTCDNSJax,
    CircularCouette,
    TaylorCouetteDNSJax,
    TaylorCouetteMRIDNSJax,
    _require_resolved_m,
)


def test_tc_dns_zero_state_stays_zero() -> None:
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=10, Nz=8, dt=1.0e-3, dealias=1.0
    )
    state = solver.step(solver.zero_state())

    assert all(jnp.allclose(component, 0.0, atol=1.0e-7) for component in state.u)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-7)
    assert float(solver.continuity_residual_l2(state)) < 1.0e-8


def test_tc_dns_initial_one_step_is_finite_and_pinned() -> None:
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=10, Nz=8, dt=1.0e-3, dealias=1.5
    )
    state = solver.step(solver.initial_state(amp=1.0e-4))
    diag = solver.diagnostics(state)

    assert all(jnp.isfinite(component).all() for component in state.u)
    assert bool(jnp.isfinite(state.p).all())
    assert float(diag["E"]) > 0.0
    assert float(diag["continuity_l2"]) < 1.0e-7
    assert abs(complex(state.p[0, 0])) < 1.0e-7


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="TC DNS growth-rate validation uses x64",
)
def test_tc_dns_eigenmode_growth_matches_linear_solver_x64() -> None:
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=12, Nz=8, dt=1.0e-3, dealias=1.0
    )
    state, eig = solver.seed_linear_eigenmode(kz_mode=1, amp=1.0e-8)
    rate, out = solver.growth_rate(state, steps=100)

    assert jnp.allclose(rate, eig.real, rtol=1.0e-7, atol=1.0e-7)
    assert float(solver.continuity_residual_l2(out)) < 1.0e-18
    assert abs(complex(out.p[0, 0])) < 1.0e-18


def test_tc_dns_3d_zero_state_stays_zero() -> None:
    solver = TaylorCouetteDNSJax(
        CircularCouette(),
        nu=0.002,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state = solver.step(solver.zero_state())

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.u)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(solver.continuity_residual_l2(state)) < 1.0e-12


def test_tc_dns_3d_azimuthal_derivative_and_resolution_guard() -> None:
    solver = TaylorCouetteDNSJax(
        CircularCouette(),
        nu=0.002,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    coeff = jnp.zeros(solver.TD.num_dofs, dtype=complex).at[1, 0, 0].set(1.0)
    value = solver.TD.backward(coeff)
    dtheta = solver.TD.backward_primitive(coeff, (1, 0, 0))

    assert jnp.max(jnp.abs(dtheta - 1j * value)) < 1.0e-12
    with pytest.raises(ValueError):
        _require_resolved_m(2, solver.Ntheta)


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="3D TC DNS growth-rate validation uses x64",
)
def test_tc_dns_3d_eigenmode_growth_matches_linear_solver_x64() -> None:
    solver = TaylorCouetteDNSJax(
        CircularCouette(),
        nu=0.002,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state, eig = solver.seed_linear_eigenmode(m=1, kz_mode=1, amp=1.0e-8)
    rate, out = solver.growth_rate(state, steps=100)

    assert jnp.allclose(rate, eig.real, rtol=1.0e-6, atol=1.0e-6)
    assert float(solver.continuity_residual_l2(out)) < 1.0e-18
    assert abs(complex(out.p[0, 0, 0])) < 1.0e-18


def _keplerian_tc_base():
    eta = 0.5
    return CircularCouette(1.0, 2.0, 1.0, eta**1.5)


def test_tc_mri_dns_zero_state_stays_zero() -> None:
    solver = AxisymmetricMRIDNSJax(
        _keplerian_tc_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state = solver.step(solver.zero_state())
    diag = solver.diagnostics(state)

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.x)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(diag["E"]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diag["divu"]) < 1.0e-12
    assert float(diag["divb"]) < 1.0e-12


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="MRI DNS growth-rate validation uses x64",
)
def test_tc_mri_dns_eigenmode_growth_matches_linear_solver_x64() -> None:
    solver = AxisymmetricMRIDNSJax(
        _keplerian_tc_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state, eig = solver.seed_linear_eigenmode(kz_mode=1, amp=1.0e-8)
    rate, out = solver.growth_rate(state, steps=100)
    diag = solver.diagnostics(out)

    assert jnp.allclose(rate, eig.real, rtol=1.0e-6, atol=1.0e-6)
    assert float(diag["divu"]) < 1.0e-7
    assert float(diag["divb"]) < 1.0e-7
    assert abs(complex(out.p[0, 0])) < 1.0e-18


def test_tc_mri_dns_3d_zero_state_stays_zero() -> None:
    solver = TaylorCouetteMRIDNSJax(
        _keplerian_tc_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state = solver.step(solver.zero_state())
    diag = solver.diagnostics(state)

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.x)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(diag["E"]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diag["divu"]) < 1.0e-12
    assert float(diag["divb"]) < 1.0e-12


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="3D MRI DNS growth-rate validation uses x64",
)
def test_tc_mri_dns_3d_eigenmode_growth_matches_linear_solver_x64() -> None:
    solver = TaylorCouetteMRIDNSJax(
        _keplerian_tc_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state, eig = solver.seed_linear_eigenmode(m=1, kz_mode=1, amp=1.0e-8)
    rate, out = solver.growth_rate(state, steps=100)
    diag = solver.diagnostics(out)

    assert jnp.allclose(rate, eig.real, rtol=1.0e-6, atol=1.0e-6)
    assert float(diag["divu"]) < 1.0e-7
    assert float(diag["divb"]) < 1.0e-7
    assert abs(complex(out.p[0, 0, 0])) < 1.0e-18
