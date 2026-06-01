import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_jax import PlaneCouetteMHDJax


def test_pcf_mhd_initial_and_one_step_are_finite() -> None:
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
    )
    state0 = solver.initial_state()
    state1 = solver.step(state0)
    diag = solver.diagnostics(state1)

    assert all(jnp.isfinite(component).all() for component in state1.flow.u)
    assert all(jnp.isfinite(component).all() for component in state1.A)
    assert bool(jnp.isfinite(state1.flow.g).all())
    assert float(diag["Epert"]) > 0.0
    assert float(diag["Emag"]) > 0.0
    assert float(diag["divL2"]) < 1.0e-4
    assert float(diag["divB_L2"]) < 1.0e-5


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="magnetic invariant check uses float64"
)
def test_pcf_mhd_divergence_free_magnetic_field_float64() -> None:
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
    )
    state = solver.step(solver.initial_state())

    assert float(solver.magnetic_divergence_l2(state)) < 1.0e-12
