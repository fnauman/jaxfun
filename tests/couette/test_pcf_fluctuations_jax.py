import jax
import jax.numpy as jnp
import pytest

from examples.channelflow_kmm import KMMState
from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from jaxfun.integrators import IMEXRK3, IMEXRK222
from jaxfun.io import Cadence


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


def test_pcf_pressure_recovery_is_finite_and_real() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        family="L",
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.05,
    )
    state = solver.step(solver.initial_state())

    coeff = solver.compute_pressure_coefficients(state)
    pressure = solver.compute_pressure(state)

    assert coeff.shape == solver.TC.num_dofs
    assert pressure.shape == solver.TC.num_quad_points
    assert bool(jnp.isfinite(coeff).all())
    assert bool(jnp.isfinite(pressure).all())
    assert float(jnp.max(jnp.abs(jnp.imag(pressure)))) < 1.0e-12
    assert float(jnp.linalg.norm(jnp.real(pressure))) > 0.0


def test_pcf_zero_state_stays_zero_for_fluctuations() -> None:
    solver = PlaneCouetteFluctuationJax(N=(9, 8, 8), family="L", dt=1.0e-3)
    state = solver.step(solver.zero_state())

    assert all(jnp.allclose(component, 0.0, atol=1.0e-7) for component in state.u)
    assert jnp.allclose(state.g, 0.0, atol=1.0e-7)


def test_pcf_mean_modes_are_forced_real_after_step() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )
    state0 = solver.initial_state()
    u = list(state0.u)
    imaginary_mean = 1.0e-4j * jnp.linspace(1.0, 2.0, u[1].shape[0])
    u[1] = u[1].at[:, 0, 0].add(imaginary_mean)
    u[2] = u[2].at[:, 0, 0].add(2.0 * imaginary_mean)

    state1 = solver.step(KMMState(u=tuple(u), g=state0.g))

    assert float(jnp.max(jnp.abs(jnp.imag(state1.u[1][:, 0, 0])))) < 1.0e-12
    assert float(jnp.max(jnp.abs(jnp.imag(state1.u[2][:, 0, 0])))) < 1.0e-12


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="recorded golden values use float64"
)
def test_pcf_one_step_matches_recorded_golden_diagnostics() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8), family="L", dt=1.0e-3, perturbation_amplitude=0.05
    )
    state = solver.step(solver.initial_state())
    diag = solver.diagnostics(state)

    expected = {
        "Epert": 0.21836099019180652,
        "Etot": 52.85625108205688,
        "divL2": 7.183953559387109e-17,
        "u_top": 1.0,
        "u_bot": -1.0,
        "mean_shear": 1.0,
    }
    for key, value in expected.items():
        atol = 1.0e-12 if key != "divL2" else 5.0e-15
        assert jnp.allclose(diag[key], value, rtol=1.0e-10, atol=atol), key


def test_pcf_imexrk3_one_step_is_finite() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
        timestepper=IMEXRK3,
    )
    state = solver.step(solver.initial_state())
    diag = solver.diagnostics(state)

    assert solver.timestepper is IMEXRK3
    assert all(jnp.isfinite(component).all() for component in state.u)
    assert bool(jnp.isfinite(state.g).all())
    assert float(diag["Epert"]) > 0.0
    assert float(diag["divL2"]) < 1.0e-4


def test_pcf_solve_with_cadence_matches_direct_solve() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )
    state0 = solver.initial_state()
    events = []

    out = solver.solve_with_cadence(
        state0,
        3,
        Cadence(diagnostics_every=2),
        block_size=2,
        on_diagnostics=lambda t, tstep, diag: events.append((t, tstep, diag["Epert"])),
    )
    direct = solver.solve(state0, 3)

    assert len(events) == 1
    assert events[0][1] == 2
    assert all(jnp.allclose(a, b) for a, b in zip(out.u, direct.u, strict=True))
    assert jnp.allclose(out.g, direct.g)


def test_pcf_time_integrator_string_dispatches_imexrk3() -> None:
    kwargs = {
        "N": (9, 4, 4),
        "Re": 200.0,
        "dt": 1.0e-3,
        "padding_factor": (1.0, 1.0, 1.0),
    }
    by_name = PlaneCouetteFluctuationJax(**kwargs, time_integrator="IMEXRK3")
    by_class = PlaneCouetteFluctuationJax(**kwargs, timestepper=IMEXRK3)

    assert by_name.time_integrator == "IMEXRK3"
    assert by_name.timestepper is IMEXRK3
    named_state = by_name.step(by_name.initial_state())
    class_state = by_class.step(by_class.initial_state())
    for actual, expected in zip(
        (*named_state.u, named_state.g),
        (*class_state.u, class_state.g),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize(
    ("timestepper", "time_integrator"),
    [
        (IMEXRK3, "IMEXRK222"),
        (IMEXRK222, "IMEXRK3"),
        (IMEXRK3, "CNAB2"),
    ],
)
def test_pcf_rejects_conflicting_timestepper_and_integrator(
    timestepper, time_integrator
) -> None:
    with pytest.raises(ValueError, match="conflicts|requires"):
        PlaneCouetteFluctuationJax(
            N=(9, 4, 4),
            timestepper=timestepper,
            time_integrator=time_integrator,
        )
