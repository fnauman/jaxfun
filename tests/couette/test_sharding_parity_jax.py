import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.pcf_mri_primitive_jax import (
    AxisymmetricPCFMRIDNSJax,
    AxisymmetricPCFState,
)
from examples.taylor_couette_dns_jax import (
    AxisymmetricMRIDNSJax,
    AxisymmetricMRIState,
    AxisymmetricTCDNSJax,
    AxisymmetricTCState,
    CircularCouette,
    TaylorCouetteDNSJax,
    TaylorCouetteMRIDNSJax,
)
from jaxfun.sharding import spectral_sharding

pytestmark = pytest.mark.spmd


def _shard_tuple(values):
    return tuple(jax.device_put(component, spectral_sharding) for component in values)


def _shard_state(state):
    if isinstance(state, AxisymmetricTCState):
        return AxisymmetricTCState(
            u=_shard_tuple(state.u),
            p=jax.device_put(state.p, spectral_sharding),
            nonlinear_old=_shard_tuple(state.nonlinear_old),
            have_old=state.have_old,
        )
    if isinstance(state, AxisymmetricPCFState):
        return AxisymmetricPCFState(
            x=_shard_tuple(state.x),
            p=jax.device_put(state.p, spectral_sharding),
            nonlinear_old=_shard_tuple(state.nonlinear_old),
            have_old=state.have_old,
        )
    return AxisymmetricMRIState(
        x=_shard_tuple(state.x),
        p=jax.device_put(state.p, spectral_sharding),
        nonlinear_old=_shard_tuple(state.nonlinear_old),
        have_old=state.have_old,
    )


def _seed_coefficients(coeff, component: int):
    n = min(int(coeff.shape[-1]), 3)
    profile = (
        (component + 1)
        * 1.0e-5
        * (jnp.arange(n, dtype=jnp.float64) + 1.0)
        * (1.0 + 0.125j * (component + 1))
    )
    out = jnp.zeros_like(coeff)
    if coeff.ndim == 2:
        kpos = 1 % coeff.shape[0]
        kneg = (-1) % coeff.shape[0]
        out = out.at[kpos, :n].set(profile)
        out = out.at[kneg, :n].set(jnp.conj(profile))
        return out
    if coeff.ndim == 3:
        mpos = 1 % coeff.shape[0]
        mneg = (-1) % coeff.shape[0]
        kpos = 1 % coeff.shape[1]
        kneg = (-1) % coeff.shape[1]
        out = out.at[mpos, kpos, :n].set(0.5 * profile)
        out = out.at[mneg, kneg, :n].set(0.5 * jnp.conj(profile))
        return out
    raise AssertionError(f"unexpected coefficient rank {coeff.ndim}")


def _seeded_hydro_state(solver):
    state = solver.zero_state()
    u = tuple(_seed_coefficients(component, i) for i, component in enumerate(state.u))
    nold = tuple(jnp.zeros_like(component) for component in u)
    p = _seed_coefficients(state.p, len(u))
    return AxisymmetricTCState(u=u, p=p, nonlinear_old=nold, have_old=False)


def _seeded_mhd_state(solver):
    state = solver.zero_state()
    x = tuple(_seed_coefficients(component, i) for i, component in enumerate(state.x))
    nold = tuple(jnp.zeros_like(component) for component in x)
    p = _seed_coefficients(state.p, len(x))
    return AxisymmetricMRIState(x=x, p=p, nonlinear_old=nold, have_old=False)


def _seeded_pcf_primitive_state(solver):
    state = solver.zero_state()
    x = tuple(_seed_coefficients(component, i) for i, component in enumerate(state.x))
    nold = tuple(jnp.zeros_like(component) for component in x)
    p = _seed_coefficients(state.p, len(x))
    return AxisymmetricPCFState(x=x, p=p, nonlinear_old=nold, have_old=False)


def _assert_bit_identical(actual, expected, label):
    actual_np = np.asarray(jax.device_get(actual))
    expected_np = np.asarray(jax.device_get(expected))
    assert np.array_equal(actual_np, expected_np), label


def _assert_diagnostics_match(replicated, sharded):
    for key, value in replicated.items():
        _assert_bit_identical(sharded[key], value, key)


def _assert_physical_fields_match(replicated, sharded):
    for i, (replicated_field, sharded_field) in enumerate(
        zip(replicated, sharded, strict=True)
    ):
        _assert_bit_identical(sharded_field, replicated_field, f"field {i}")
        assert len(sharded_field.devices()) == jax.device_count()


def _assert_state_coefficients_match(replicated, sharded):
    if isinstance(replicated, AxisymmetricTCState):
        replicated_coeffs = (*replicated.u, replicated.p, *replicated.nonlinear_old)
        sharded_coeffs = (*sharded.u, sharded.p, *sharded.nonlinear_old)
    else:
        replicated_coeffs = (*replicated.x, replicated.p, *replicated.nonlinear_old)
        sharded_coeffs = (*sharded.x, sharded.p, *sharded.nonlinear_old)
    for replicated_coeff, sharded_coeff in zip(
        replicated_coeffs, sharded_coeffs, strict=True
    ):
        _assert_bit_identical(sharded_coeff, replicated_coeff, "state coefficient")
        assert len(sharded_coeff.devices()) == jax.device_count()


def test_tc_axisymmetric_sharded_transforms_and_diagnostics_match_replicated() -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
    )
    state = solver.initial_state(amp=1.0e-4)
    sharded_state = _shard_state(state)

    replicated_diag = solver.diagnostics(state)
    sharded_diag = solver.diagnostics(sharded_state)

    _assert_diagnostics_match(replicated_diag, sharded_diag)
    _assert_physical_fields_match(
        solver.velocity_physical(state), solver.velocity_physical(sharded_state)
    )


@pytest.mark.parametrize(
    ("solver_factory", "state_factory", "physical_method"),
    [
        (
            lambda: AxisymmetricTCDNSJax(
                CircularCouette(), nu=0.002, Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_hydro_state,
            "velocity_physical",
        ),
        (
            lambda: TaylorCouetteDNSJax(
                CircularCouette(),
                nu=0.002,
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_hydro_state,
            "velocity_physical",
        ),
        (
            lambda: AxisymmetricMRIDNSJax(
                CircularCouette(), Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_mhd_state,
            "fields_physical",
        ),
        (
            lambda: TaylorCouetteMRIDNSJax(
                CircularCouette(),
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_mhd_state,
            "fields_physical",
        ),
    ],
    ids=[
        "axisymmetric-hydro",
        "full-3d-hydro",
        "axisymmetric-mhd",
        "full-3d-mhd",
    ],
)
def test_taylor_couette_quadrants_sharded_transforms_and_diagnostics_match_replicated(
    solver_factory, state_factory, physical_method
) -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = solver_factory()
    state = state_factory(solver)
    sharded_state = _shard_state(state)

    _assert_diagnostics_match(
        solver.diagnostics(state), solver.diagnostics(sharded_state)
    )
    physical = getattr(solver, physical_method)
    _assert_physical_fields_match(physical(state), physical(sharded_state))


@pytest.mark.parametrize(
    ("solver_factory", "state_factory"),
    [
        (
            lambda: AxisymmetricTCDNSJax(
                CircularCouette(), nu=0.002, Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_hydro_state,
        ),
        (
            lambda: TaylorCouetteDNSJax(
                CircularCouette(),
                nu=0.002,
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_hydro_state,
        ),
        (
            lambda: AxisymmetricMRIDNSJax(
                CircularCouette(), Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_mhd_state,
        ),
        (
            lambda: TaylorCouetteMRIDNSJax(
                CircularCouette(),
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_mhd_state,
        ),
    ],
    ids=[
        "axisymmetric-hydro",
        "full-3d-hydro",
        "axisymmetric-mhd",
        "full-3d-mhd",
    ],
)
def test_taylor_couette_quadrants_sharded_one_step_matches_replicated(
    solver_factory, state_factory
) -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = solver_factory()
    state = state_factory(solver)
    replicated = solver.step(state)
    sharded = solver.step(_shard_state(state))

    _assert_state_coefficients_match(replicated, sharded)
    _assert_diagnostics_match(
        solver.diagnostics(replicated), solver.diagnostics(sharded)
    )


@pytest.mark.parametrize(
    ("solver_factory", "state_factory"),
    [
        (
            lambda: AxisymmetricTCDNSJax(
                CircularCouette(), nu=0.002, Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_hydro_state,
        ),
        (
            lambda: TaylorCouetteDNSJax(
                CircularCouette(),
                nu=0.002,
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_hydro_state,
        ),
        (
            lambda: AxisymmetricMRIDNSJax(
                CircularCouette(), Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
            ),
            _seeded_mhd_state,
        ),
        (
            lambda: TaylorCouetteMRIDNSJax(
                CircularCouette(),
                Nr=8,
                Ntheta=4,
                Nz=8,
                dt=1.0e-3,
                dealias=1.0,
            ),
            _seeded_mhd_state,
        ),
    ],
    ids=[
        "axisymmetric-hydro",
        "full-3d-hydro",
        "axisymmetric-mhd",
        "full-3d-mhd",
    ],
)
def test_taylor_couette_quadrants_sharded_five_step_rollout_matches_replicated(
    solver_factory, state_factory
) -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = solver_factory()
    state = state_factory(solver)
    replicated = solver.solve(state, 5)
    sharded = solver.solve(_shard_state(state), 5)

    _assert_state_coefficients_match(replicated, sharded)
    _assert_diagnostics_match(
        solver.diagnostics(replicated), solver.diagnostics(sharded)
    )


def test_pcf_primitive_sharded_five_step_rollout_matches_replicated() -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = AxisymmetricPCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nx=8,
        Nz=8,
        dt=1.0e-3,
        dealias=1.0,
    )
    state = _seeded_pcf_primitive_state(solver)
    replicated = solver.solve(state, 5)
    sharded = solver.solve(_shard_state(state), 5)

    _assert_state_coefficients_match(replicated, sharded)
    _assert_diagnostics_match(
        solver.diagnostics(replicated), solver.diagnostics(sharded)
    )
