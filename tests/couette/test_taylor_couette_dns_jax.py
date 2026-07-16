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
from jaxfun.diagnostics import (
    coefficient_wall_linf,
    cylindrical_component_energy,
    cylindrical_energy_parts,
    cylindrical_kinetic_energy,
)
from jaxfun.galerkin.inner import integrate
from jaxfun.io import Cadence


def _assert_rollout_lifecycle(solver, state) -> None:
    for _ in range(2):
        state = solver.solve(state, 1)
    jax.block_until_ready(state)
    info = solver.rollout_cache_info()
    assert info.live_entries == 1
    assert info.misses == 1
    assert info.hits == 1

    solver.set_dt(0.5 * solver.dt)
    rebound = solver.rollout_cache_info()
    assert rebound.generation == 1
    assert rebound.live_entries == 0


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


def test_tc_hydro_rollout_caches_and_chebyshev_volumes() -> None:
    axisymmetric = AxisymmetricTCDNSJax(
        CircularCouette(), Nr=8, Nz=4, Lz=1.5, dt=1.0e-3, family="C", dealias=1.0
    )
    expected_axisymmetric = (
        axisymmetric.Lz
        * 0.5
        * (axisymmetric.base.R2**2 - axisymmetric.base.R1**2)
    )
    assert float(integrate(axisymmetric.R, axisymmetric.T0)) == pytest.approx(
        expected_axisymmetric
    )
    _assert_rollout_lifecycle(axisymmetric, axisymmetric.zero_state())

    full = TaylorCouetteDNSJax(
        CircularCouette(),
        Nr=8,
        Ntheta=4,
        Nz=4,
        Lz=1.5,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
    )
    expected_full = 2.0 * jnp.pi * full.Lz * 0.5 * (full.base.R2**2 - full.base.R1**2)
    assert float(integrate(full.R, full.T0)) == pytest.approx(float(expected_full))
    _assert_rollout_lifecycle(full, full.zero_state())


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="TC eigenmode seed comparison uses x64 eigensolves",
)
def test_tc_axisymmetric_and_3d_m0_eigenmode_seed_amplitudes_match() -> None:
    axisymmetric = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
    )
    full_3d = TaylorCouetteDNSJax(
        CircularCouette(),
        nu=0.002,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )

    state_axis, eig_axis = axisymmetric.seed_linear_eigenmode(kz_mode=1, amp=1.0e-5)
    state_3d, eig_3d = full_3d.seed_linear_eigenmode(m=0, kz_mode=1, amp=1.0e-5)

    assert eig_3d == pytest.approx(eig_axis, rel=1.0e-12, abs=1.0e-12)
    for axis_component, full_component in zip(
        axisymmetric.velocity_physical(state_axis),
        full_3d.velocity_physical(state_3d),
        strict=True,
    ):
        assert jnp.allclose(
            full_component[0], axis_component, rtol=1.0e-10, atol=1.0e-12
        )


def _deterministic_complex_rhs(dim: int, dtype) -> jnp.ndarray:
    i = jnp.arange(dim, dtype=jnp.float64)
    return (jnp.sin(0.13 * i) + 1j * jnp.cos(0.07 * i)).astype(dtype)


def _assert_pinned_limp_solve_residual(solver) -> None:
    rhs = _deterministic_complex_rhs(solver.VQ.dim, solver.Limp.dtype)
    sol = solver._solve_limp(rhs)

    indices = solver.VQ_mode_indices
    matrices = solver._pin_pressure_modes(solver.Limp_modes).astype(rhs.dtype)
    rhs_modes = rhs[indices]
    pressure_row = sum(int(space.num_dofs[-1]) for space in solver.VQ.tensorspaces[:3])
    rhs_modes = rhs_modes.at[0, pressure_row].set(0)
    sol_modes = sol[indices]

    residual = jnp.einsum("mij,mj->mi", matrices, sol_modes) - rhs_modes
    rel = jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs_modes), 1.0)
    assert float(rel) < 1.0e-11

    direct = jnp.linalg.solve(matrices, rhs_modes[..., None])[..., 0]
    assert jnp.allclose(sol_modes, direct, rtol=1.0e-11, atol=1.0e-11)


@pytest.mark.parametrize(
    "solver_factory",
    [
        lambda: AxisymmetricTCDNSJax(
            CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
        ),
        lambda: TaylorCouetteDNSJax(
            CircularCouette(),
            nu=0.002,
            Nr=8,
            Ntheta=4,
            Nz=6,
            dt=1.0e-3,
            dealias=1.0,
        ),
        lambda: AxisymmetricMRIDNSJax(
            _keplerian_tc_base(),
            B0=0.1,
            nu=0.001,
            eta_mag=0.001,
            Nr=8,
            Nz=6,
            dt=1.0e-3,
            dealias=1.0,
        ),
        lambda: TaylorCouetteMRIDNSJax(
            _keplerian_tc_base(),
            B0=0.1,
            nu=0.001,
            eta_mag=0.001,
            Nr=8,
            Ntheta=4,
            Nz=6,
            dt=1.0e-3,
            dealias=1.0,
        ),
    ],
    ids=[
        "axisymmetric-hydro",
        "full-3d-hydro",
        "axisymmetric-mhd",
        "full-3d-mhd",
    ],
)
def test_tc_dns_pinned_saddle_solves_match_dense_residuals(solver_factory) -> None:
    _assert_pinned_limp_solve_residual(solver_factory())


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
    # The pinned 3-D saddle-point solve satisfies continuity at float64
    # roundoff; its reduction floor is a few 1e-17 on current XLA backends.
    assert float(solver.continuity_residual_l2(out)) < 1.0e-15
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
    assert float(diag["wall_u"]) < 1.0e-18


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
    assert float(diag["wall_u"]) < 1.0e-18


def test_tc_mhd_rollout_caches_rebind_for_axisymmetric_and_3d() -> None:
    axisymmetric = AxisymmetricMRIDNSJax(
        _keplerian_tc_base(),
        Nr=8,
        Nz=4,
        Lz=1.5,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
    )
    _assert_rollout_lifecycle(axisymmetric, axisymmetric.zero_state())

    full = TaylorCouetteMRIDNSJax(
        _keplerian_tc_base(),
        Nr=8,
        Ntheta=4,
        Nz=4,
        Lz=1.5,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
    )
    _assert_rollout_lifecycle(full, full.zero_state())


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


def test_tc_diagnostics_helpers_match_solver_outputs() -> None:
    hydro = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
    )
    hstate = hydro.step(hydro.initial_state(amp=1.0e-4))
    velocity = hydro.velocity_physical(hstate)
    hdiag = hydro.diagnostics(hstate)

    assert jnp.allclose(
        hdiag["E"], cylindrical_kinetic_energy(velocity, hydro.R, hydro.T0)
    )
    assert jnp.allclose(
        hdiag["Eth"], cylindrical_component_energy(velocity[1], hydro.R, hydro.T0)
    )
    assert jnp.allclose(
        hdiag["wall"], coefficient_wall_linf(hstate.u, (hydro.TD,) * 3)
    )
    assert float(hdiag["wall"]) < 1.0e-18

    mri = AxisymmetricMRIDNSJax(
        _keplerian_tc_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    z, r = mri.T0.mesh()
    wall = (r - mri.base.R1) * (mri.base.R2 - r)
    values = (
        1.0e-4 * wall * jnp.cos(z),
        2.0e-4 * wall * jnp.sin(z),
        -1.0e-4 * wall * jnp.cos(2.0 * z),
        1.0e-5 * wall * jnp.sin(z),
        2.0e-5 * wall * jnp.cos(z),
        -1.0e-5 * wall * jnp.sin(2.0 * z),
    )
    mstate = mri.state_from_physical(values)
    fields = mri.fields_physical(mstate)
    ek, em = cylindrical_energy_parts(fields[:3], fields[3:], mri.R, mri.T0)
    mdiag = mri.diagnostics(mstate)

    assert jnp.allclose(mdiag["Ekin"], ek)
    assert jnp.allclose(mdiag["Emag"], em)
    assert float(mdiag["wall_u"]) < 1.0e-18
    assert jnp.allclose(mdiag["E"], ek + em)


def test_tc_solve_with_cadence_matches_direct_solve() -> None:
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
    )
    state0 = solver.initial_state(amp=1.0e-4)
    events = []

    out = solver.solve_with_cadence(
        state0,
        3,
        Cadence(diagnostics_every=2),
        block_size=2,
        on_diagnostics=lambda t, tstep, diag: events.append((t, tstep, diag["E"])),
    )
    direct = solver.solve(state0, 3)

    assert len(events) == 1
    assert events[0][1] == 2
    assert all(jnp.allclose(a, b) for a, b in zip(out.u, direct.u, strict=True))
    assert jnp.allclose(out.p, direct.p)
