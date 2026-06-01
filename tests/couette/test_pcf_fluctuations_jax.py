import jax
import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax


def test_pcf_fluctuation_initialization_and_one_step_are_finite() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8), family="L", dt=1.0e-3, perturbation_amplitude=0.05
    )
    state0 = solver.initial_state()
    state1 = solver.step(state0)
    diag = solver.diagnostics(state1)

    assert all(jnp.isfinite(component).all() for component in state1.u)
    assert bool(jnp.isfinite(state1.g).all())
    assert float(diag["Epert"]) > 0.0
    assert float(diag["Etot"]) > float(diag["Epert"])
    assert float(diag["divL2"]) < 1.0e-4
    assert abs(float(diag["mean_shear"]) - 1.0) < 1.0e-4


def test_pcf_zero_state_stays_zero_for_fluctuations() -> None:
    solver = PlaneCouetteFluctuationJax(N=(9, 8, 8), family="L", dt=1.0e-3)
    state = solver.step(solver.zero_state())

    assert all(jnp.allclose(component, 0.0, atol=1.0e-7) for component in state.u)
    assert jnp.allclose(state.g, 0.0, atol=1.0e-7)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="shenfun parity reference uses float64"
)
def test_pcf_one_step_matches_shenfun_reference_diagnostics() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8), family="L", dt=1.0e-3, perturbation_amplitude=0.05
    )
    state = solver.step(solver.initial_state())
    diag = solver.diagnostics(state)

    expected = {
        "Epert": 0.21836099019180652,
        "Etot": 52.85625108205688,
        "divL2": 7.183953559387109e-17,
        "u_top": 0.968160239435768,
        "u_bot": -0.9681602394357679,
        "mean_shear": 1.0000000004699001,
    }
    for key, value in expected.items():
        atol = 1.0e-12 if key != "divL2" else 5.0e-15
        assert jnp.allclose(diag[key], value, rtol=1.0e-10, atol=atol), key
