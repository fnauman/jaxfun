import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mri_primitive_jax import AxisymmetricPCFMRIDNSJax


def test_pcf_primitive_dns_zero_state_stays_zero() -> None:
    solver = AxisymmetricPCFMRIDNSJax(Nx=8, Nz=6, dt=1.0e-3, dealias=1.0)
    state = solver.step(solver.zero_state())
    diag = solver.diagnostics(state)

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.x)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(diag["E"]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diag["divu"]) < 1.0e-12
    assert float(diag["divb"]) < 1.0e-12


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="PCF primitive DNS growth-rate validation uses x64",
)
def test_pcf_primitive_hydro_dns_growth_matches_linear_solver_x64() -> None:
    solver = AxisymmetricPCFMRIDNSJax(
        S=1.0,
        omega=0.0,
        B0=0.0,
        nu=0.01,
        eta_mag=0.01,
        Nx=24,
        Nz=8,
        Lz=1.0,
        dt=2.0e-3,
        dealias=1.0,
    )
    state, eig = solver.seed_hydro_eigenmode(kz_mode=1, amp=1.0e-5)
    rate, out = solver.growth_rate(state, steps=50)
    diag = solver.diagnostics(out)

    assert jnp.allclose(rate, eig.real, rtol=1.0e-6, atol=1.0e-6)
    assert float(diag["divu"]) < 1.0e-10
    assert float(diag["divb"]) == pytest.approx(0.0, abs=1.0e-12)


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="PCF primitive MRI DNS growth-rate validation uses x64",
)
def test_pcf_primitive_mri_dns_growth_matches_linear_solver_x64() -> None:
    solver = AxisymmetricPCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nx=24,
        Nz=8,
        Lz=1.0,
        dt=2.0e-3,
        dealias=1.0,
    )
    state, eig = solver.seed_linear_eigenmode(kz_mode=1, amp=1.0e-7)
    rate, out = solver.growth_rate(state, steps=50)
    diag = solver.diagnostics(out)

    assert jnp.allclose(rate, eig.real, rtol=5.0e-6, atol=2.0e-6)
    assert float(diag["divu"]) < 1.0e-8
    assert float(diag["divb"]) < 1.0e-8
