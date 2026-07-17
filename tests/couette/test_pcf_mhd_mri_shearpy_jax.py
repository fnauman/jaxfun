import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_mri_shearpy_jax import (
    PlaneCouetteMRIShearpyInsulatingJax,
    PlaneCouetteMRIShearpyJax,
)


def test_pcf_mhd_mri_shearpy_one_step_is_finite() -> None:
    solver = PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
        background_b=(0.0, 0.0, 0.1),
    )
    state = solver.step(solver.initial_state())
    diag = solver.diagnostics(state)

    assert float(diag["divL2"]) < 1.0e-4
    assert float(diag["divB_L2"]) < 1.0e-5
    assert bool(jnp.isfinite(diag["alpha"]))
    assert bool(jnp.isfinite(diag["reynolds_xy"]))
    assert bool(jnp.isfinite(diag["maxwell_xy"]))
    assert float(diag["q_shear"]) == 1.0


def test_pcf_mhd_mri_shearpy_defaults_match_reference():
    solver = PlaneCouetteMRIShearpyJax(N=(9, 8, 8), family="L", dt=1.0e-3)

    assert solver.background_b == (0.0, 0.0, 0.025)
    assert solver.magnetic_amplitude == 0.0


@pytest.mark.integration
@pytest.mark.parametrize(
    "solver_cls",
    [PlaneCouetteMRIShearpyJax, PlaneCouetteMRIShearpyInsulatingJax],
)
def test_pcf_mhd_rollout_cache_reuses_blocks_across_dt_changes(solver_cls) -> None:
    solver = solver_cls(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.01,
        magnetic_amplitude=0.01,
        background_b=(0.0, 0.0, 0.1),
    )
    state = solver.initial_state()
    for _ in range(3):
        state = solver.solve(state, 1)
    jax.block_until_ready(state)

    info = solver.rollout_cache_info()
    assert info.live_entries == 1
    assert info.step_counts == (1,)
    assert info.misses == 1
    assert info.hits == 2

    solver.set_dt(5.0e-4)
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
    assert reused.live_entries == 1
    assert reused.misses == 1
    assert reused.hits == 3
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(state))


@pytest.mark.gpu
def test_pcf_mhd_mri_step_has_bounded_gpu_temporary_memory() -> None:
    assert jax.default_backend() == "gpu"
    solver = PlaneCouetteMRIShearpyJax(
        N=(17, 16, 16),
        domain=((-1.0, 1.0), (0.0, 4.0), (0.0, 1.0)),
        Re=2000.0,
        Rm=6000.0,
        omega=2.0 / 3.0,
        shear_rate=1.0,
        background_b=(0.0, 0.0, 0.0),
        dt=1.0e-3,
        family="L",
        padding_factor=(1.0, 1.5, 1.5),
        perturbation_amplitude=1.0e-3,
        magnetic_amplitude=0.1,
        magnetic_seed="sinusoidal_bz_x",
        solenoidal_velocity_seed=True,
    )
    state = solver.initial_state()
    compiled = jax.jit(solver.step).lower(state).compile()
    analysis = compiled.memory_analysis()

    assert analysis is not None
    assert analysis.temp_size_in_bytes < 64 * 1024**2
    out = compiled(state)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(out))


@pytest.mark.gpu
def test_pcf_mri_pallas_wavenumber_solver_matches_compact_jax(monkeypatch) -> None:
    assert jax.default_backend() == "gpu"
    kwargs = {
        "N": (17, 16, 16),
        "family": "C",
        "dt": 1.0e-3,
        "perturbation_amplitude": 1.0e-3,
        "magnetic_amplitude": 0.1,
        "background_b": (0.0, 0.0, 0.0),
        "magnetic_seed": "sinusoidal_bz_x",
        "solenoidal_velocity_seed": True,
    }
    monkeypatch.setenv("JAXFUN_WAVENUMBER_SOLVER", "jax")
    compact = PlaneCouetteMRIShearpyJax(**kwargs)
    monkeypatch.setenv("JAXFUN_WAVENUMBER_SOLVER", "pallas-triton")
    pallas = PlaneCouetteMRIShearpyJax(**kwargs)

    state = pallas.initial_state()
    expected = jax.jit(compact.step)(state)
    actual = jax.jit(pallas.step)(state)
    jax.block_until_ready((expected, actual))

    errors = []
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jnp.bool_):
            assert bool(jnp.array_equal(left, right))
        else:
            errors.append(jnp.max(jnp.abs(left - right)))
    assert float(jnp.max(jnp.stack(errors))) < 2.0e-12
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(actual))
    assert pallas.Su_factor.wavenumber_backend == "pallas-triton"
