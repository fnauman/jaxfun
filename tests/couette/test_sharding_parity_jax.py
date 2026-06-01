import jax
import jax.numpy as jnp
import pytest

from examples.taylor_couette_dns_jax import (
    AxisymmetricTCDNSJax,
    AxisymmetricTCState,
    CircularCouette,
)
from jaxfun.sharding import spectral_sharding

pytestmark = pytest.mark.spmd


def test_tc_axisymmetric_sharded_transforms_and_diagnostics_match_replicated() -> None:
    if jax.device_count() < 2:
        pytest.skip("requires --num-devices=2")

    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=8, dt=1.0e-3, dealias=1.0
    )
    state = solver.initial_state(amp=1.0e-4)
    sharded_u = tuple(
        jax.device_put(component, spectral_sharding) for component in state.u
    )
    sharded_state = AxisymmetricTCState(
        u=sharded_u,
        p=jax.device_put(state.p, spectral_sharding),
        nonlinear_old=tuple(
            jax.device_put(component, spectral_sharding)
            for component in state.nonlinear_old
        ),
        have_old=state.have_old,
    )

    replicated_diag = solver.diagnostics(state)
    sharded_diag = solver.diagnostics(sharded_state)

    for key, value in replicated_diag.items():
        assert jnp.allclose(sharded_diag[key], value, rtol=1.0e-12, atol=1.0e-12), key

    for replicated, sharded in zip(
        solver.velocity_physical(state),
        solver.velocity_physical(sharded_state),
        strict=True,
    ):
        assert jnp.allclose(sharded, replicated, rtol=1.0e-12, atol=1.0e-12)
        assert len(sharded.devices()) == jax.device_count()
