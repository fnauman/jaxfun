import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax


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


@pytest.mark.gpu
def test_pcf_mhd_mri_step_has_bounded_gpu_temporary_memory() -> None:
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
