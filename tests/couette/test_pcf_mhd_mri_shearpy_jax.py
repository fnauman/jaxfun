import jax.numpy as jnp

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
