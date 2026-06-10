import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mri_primitive_jax import AxisymmetricPCFMRIDNSJax, PCFMRIDNSJax
from jaxfun import Dx
from jaxfun.galerkin import InnerKind, TestFunction, TrialFunction, inner


def test_pcf_primitive_dns_zero_state_stays_zero() -> None:
    solver = AxisymmetricPCFMRIDNSJax(Nx=8, Nz=6, dt=1.0e-3, dealias=1.0)
    state = solver.step(solver.zero_state())
    diag = solver.diagnostics(state)

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.x)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(diag["E"]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diag["divu"]) < 1.0e-12
    assert float(diag["divb"]) < 1.0e-12


def test_pcf_primitive_3d_mode_blocks_match_dense_extraction() -> None:
    solver = PCFMRIDNSJax(Nx=8, Ny=4, Nz=4, dt=1.0e-3, dealias=1.0)
    u = TrialFunction(solver.TD, name="u_mode_check")
    v = TestFunction(solver.TD, name="v_mode_check")
    mode_shape = solver._mode_shape(solver.VE)
    radial_size = int(solver.TD.num_dofs[-1])
    scalar_mode_indices = jnp.stack(
        [
            jnp.arange(flat * radial_size, (flat + 1) * radial_size)
            for flat in range(int(jnp.prod(jnp.asarray(mode_shape))))
        ]
    )

    expressions = (
        v * u,
        v * Dx(u, 0, 1),
        v * solver.xcoord * Dx(u, 0, 1),
        v * Dx(u, 2, 2),
    )
    for expr in expressions:
        dense = jnp.asarray(inner(expr, kind=InnerKind.BILINEAR).todense())
        expected = dense[
            scalar_mode_indices[:, :, None], scalar_mode_indices[:, None, :]
        ]
        actual = solver._mode_blocks_from_expr(expr, mode_shape)
        assert jnp.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_pcf_primitive_3d_zero_state_stays_zero() -> None:
    solver = PCFMRIDNSJax(Nx=8, Ny=4, Nz=4, dt=1.0e-3, dealias=1.0)
    state = solver.step(solver.zero_state())
    diag = solver.diagnostics(state)

    assert all(jnp.allclose(component, 0.0, atol=1.0e-12) for component in state.x)
    assert jnp.allclose(state.p, 0.0, atol=1.0e-12)
    assert float(diag["E"]) == pytest.approx(0.0, abs=1.0e-12)
    assert float(diag["divu"]) < 1.0e-12
    assert float(diag["divb"]) < 1.0e-12


@pytest.mark.parametrize("dealias", [1.0, (1.0, 1.5, 1.5)])
def test_pcf_primitive_3d_seeded_mode_one_step_is_finite(
    dealias: float | tuple[float, float, float],
) -> None:
    solver = PCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nx=8,
        Ny=4,
        Nz=4,
        Ly=4.0,
        Lz=1.0,
        dt=1.0e-3,
        dealias=dealias,
    )
    state, eig = solver.seed_linear_eigenmode(ky_mode=1, kz_mode=1, amp=1.0e-7)
    out = solver.step(state)
    diag = solver.diagnostics(out)

    assert eig.real == pytest.approx(float(eig.real))
    assert all(jnp.isfinite(component).all() for component in out.x)
    assert jnp.isfinite(out.p).all()
    assert float(diag["E"]) > 0.0
    assert float(diag["divu"]) < 1.0e-7
    assert float(diag["divb"]) < 1.0e-7
    assert bool(jnp.isfinite(diag["transport_alpha"]))


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
