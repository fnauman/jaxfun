import jax
import jax.numpy as jnp
import pytest

from examples.taylor_couette_linear_jax import CircularCouette
from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax
from jaxfun.galerkin.inner import integrate


@pytest.mark.parametrize("magnetic_bc", ["conducting", "insulating"])
def test_tc_vector_potential_cache_volume_and_exact_wall_trace(
    magnetic_bc: str,
) -> None:
    solver = TaylorCouetteVPMRIDNSJax(
        CircularCouette(),
        Nr=8,
        Ntheta=4,
        Nz=4,
        Lz=1.5,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
        magnetic_bc=magnetic_bc,
    )
    expected_volume = (
        2.0 * jnp.pi * solver.Lz * 0.5 * (solver.base.R2**2 - solver.base.R1**2)
    )
    assert float(integrate(solver.R, solver.T0)) == pytest.approx(
        float(expected_volume)
    )

    state = solver.zero_state()
    for _ in range(2):
        state = solver.solve(state, 1)
    jax.block_until_ready(state)
    info = solver.rollout_cache_info()
    assert info.live_entries == 1
    assert info.misses == 1
    assert info.hits == 1
    assert float(solver.diagnostics(state)["wall_u"]) < 1.0e-18

    solver.set_dt(0.5 * solver.dt)
    rebound = solver.rollout_cache_info()
    assert rebound.generation == 1
    assert rebound.live_entries == 0
