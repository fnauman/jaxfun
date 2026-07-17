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
    retained = solver.rollout_cache_info()
    assert retained.generation == 0
    assert retained.live_entries == 1
    expected = solver.step(state)
    state = solver.solve(state, 1)
    jax.block_until_ready((expected, state))
    assert all(
        bool(jnp.allclose(actual, reference, rtol=1.0e-12, atol=1.0e-12))
        for actual, reference in zip(
            jax.tree.leaves(state), jax.tree.leaves(expected), strict=True
        )
    )
    reused = solver.rollout_cache_info()
    assert reused.misses == 1
    assert reused.hits == 2


def test_trimmed_current_transforms_match_full_phys4_reference() -> None:
    solver = TaylorCouetteVPMRIDNSJax(
        CircularCouette(),
        Nr=8,
        Ntheta=4,
        Nz=4,
        Lz=1.5,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
    )
    size = 1
    for extent in solver.T0.num_dofs:
        size *= extent
    base = jnp.arange(size, dtype=jnp.float64).reshape(solver.T0.num_dofs)
    B = (
        (1.0e-4 + 2.0e-5j) * base,
        (-2.0e-4 + 1.0e-5j) * base,
        (3.0e-4 - 4.0e-5j) * base,
    )

    value, radial, theta, axial = jax.vmap(
        lambda coefficients: solver._phys4(coefficients, solver.T0)
    )(jnp.stack(B))
    expected = (
        solver.inv_r_p * theta[2] - axial[1],
        axial[0] - radial[2],
        radial[1] + solver.inv_r_p * value[1] - solver.inv_r_p * theta[0],
    )
    actual = solver._current_physical(B)

    for computed, reference in zip(actual, expected, strict=True):
        assert jnp.allclose(computed, reference, rtol=1.0e-13, atol=1.0e-13)
