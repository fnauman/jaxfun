import jax
import jax.numpy as jnp
import pytest

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


def _assert_diagnostics_match(replicated, sharded):
    for key, value in replicated.items():
        assert jnp.allclose(sharded[key], value, rtol=1.0e-12, atol=1.0e-12), key


def _assert_physical_fields_match(replicated, sharded):
    for replicated_field, sharded_field in zip(replicated, sharded, strict=True):
        assert jnp.allclose(sharded_field, replicated_field, rtol=1.0e-12, atol=1.0e-12)
        assert len(sharded_field.devices()) == jax.device_count()


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
