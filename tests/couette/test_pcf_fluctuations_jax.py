import jax
import jax.numpy as jnp
import pytest

from examples.channelflow_kmm import KMMState
from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from jaxfun.integrators import IMEXRK3, IMEXRK222
from jaxfun.io import Cadence


def _random_complex(key, shape):
    real_key, imag_key = jax.random.split(key)
    return jax.random.normal(real_key, shape) + 1j * jax.random.normal(imag_key, shape)


@pytest.mark.parametrize("family", ["C", "L"], ids=["chebyshev", "legendre"])
def test_kmm_coefficient_rhs_and_reconstruction_match_transform_oracles(
    family,
) -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(13, 12, 12),
        family=family,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.0,
        coefficient_path="optimized",
    )
    keys = jax.random.split(jax.random.PRNGKey(1729), 7)
    H = tuple(
        solver.TD.mask_nyquist(
            1.0e-3 * _random_complex(keys[index], solver.TD.num_dofs)
        )
        for index in range(3)
    )

    actual_rhs = solver._nonlinear_rhs(H)
    expected_rhs = solver._nonlinear_rhs_physical_reference(H)
    for actual, expected in zip(actual_rhs, expected_rhs, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-11, atol=2.0e-12), float(
            jnp.max(jnp.abs(actual - expected))
        )

    u0 = solver.TB.mask_nyquist(1.0e-3 * _random_complex(keys[3], solver.TB.num_dofs))
    g = solver.TD.mask_nyquist(1.0e-3 * _random_complex(keys[4], solver.TD.num_dofs))
    v00 = jax.random.normal(keys[5], (solver.D00.num_dofs,))
    w00 = jax.random.normal(keys[6], (solver.D00.num_dofs,))
    actual_u = solver._reconstruct_velocity(u0, g, v00, w00)
    expected_u = solver._reconstruct_velocity_physical_reference(u0, g, v00, w00)
    for actual, expected in zip(actual_u, expected_u, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-11, atol=2.0e-12), float(
            jnp.max(jnp.abs(actual - expected))
        )


def test_kmm_coefficient_hot_path_avoids_physical_transform_round_trips(
    monkeypatch,
) -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8),
        family="C",
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.0,
        coefficient_path="optimized",
    )
    H = tuple(jnp.zeros(solver.TD.num_dofs, dtype=complex) for _ in range(3))
    u0 = jnp.zeros(solver.TB.num_dofs, dtype=complex)
    g = jnp.zeros(solver.TD.num_dofs, dtype=complex)
    mean = jnp.zeros(solver.D00.num_dofs)

    to_orthogonal = type(solver.TD).to_orthogonal
    conversions = 0

    def counted_to_orthogonal(space, coefficients):
        nonlocal conversions
        conversions += 1
        return to_orthogonal(space, coefficients)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("coefficient hot path performed a physical transform")

    tensor_space_type = type(solver.TD)
    monkeypatch.setattr(tensor_space_type, "to_orthogonal", counted_to_orthogonal)
    monkeypatch.setattr(tensor_space_type, "backward_primitive", forbidden)
    monkeypatch.setattr(tensor_space_type, "forward", forbidden)
    monkeypatch.setattr(tensor_space_type, "scalar_product", forbidden)

    solver._nonlinear_rhs(H)
    assert conversions == 3
    solver._reconstruct_velocity(u0, g, mean, mean)


def test_pcf_defaults_to_transform_reference_path() -> None:
    solver = PlaneCouetteFluctuationJax(N=(9, 4, 4))

    assert solver.coefficient_path == "transform"


def test_pcf_rejects_unknown_nonlinear_form() -> None:
    with pytest.raises(ValueError, match="nonlinear_form"):
        PlaneCouetteFluctuationJax(N=(9, 4, 4), nonlinear_form="advective")


@pytest.mark.parametrize("family", ["C", "L"], ids=["chebyshev", "legendre"])
def test_pcf_rotational_form_preserves_laminar_state_and_matches_gradient_step(
    family,
) -> None:
    kwargs = dict(
        N=(13, 12, 12),
        family=family,
        dt=1.0e-4,
        padding_factor=(1.0, 1.5, 1.5),
        perturbation_amplitude=0.02,
        time_integrator="CNAB2",
    )
    gradient = PlaneCouetteFluctuationJax(**kwargs, nonlinear_form="gradient")
    rotational = PlaneCouetteFluctuationJax(**kwargs, nonlinear_form="rotational")

    laminar = rotational.step(rotational.zero_state())
    assert all(jnp.allclose(value, 0.0, atol=2.0e-13) for value in laminar.u)
    assert jnp.allclose(laminar.g, 0.0, atol=2.0e-13)

    state = gradient.initial_state()
    grad_rhs = gradient._nonlinear_rhs(gradient.convection(state))
    rotational_state = rotational.initial_state()
    rot_rhs = rotational._nonlinear_rhs(rotational.convection(rotational_state))
    relative = jnp.linalg.norm(grad_rhs[0] - rot_rhs[0]) / jnp.linalg.norm(grad_rhs[0])
    assert float(relative) < 1.0e-4
    assert jnp.allclose(rot_rhs[1], grad_rhs[1], rtol=2.0e-11, atol=2.0e-12)

    grad_step = gradient.step(state)
    rot_step = rotational.step(rotational_state)
    for actual, expected in zip(
        (*rot_step.u, rot_step.g), (*grad_step.u, grad_step.g), strict=True
    ):
        assert jnp.allclose(actual, expected, rtol=2.0e-7, atol=2.0e-10)


def test_pcf_rotational_term_has_zero_pointwise_self_work() -> None:
    solver = PlaneCouetteFluctuationJax(
        N=(13, 12, 12),
        family="C",
        nonlinear_form="rotational",
        perturbation_amplitude=0.02,
    )
    state = solver.initial_state()
    n, up = solver._flow_convection_physical(state)
    omega = solver.velocity_vorticity_physical(state.u, padded=True)
    utotal, _omega_total = solver._add_base_rotational_fields(up, omega)
    self_work = sum(ui * ni for ui, ni in zip(utotal, n, strict=True))
    assert float(jnp.max(jnp.abs(self_work))) < 2.0e-13


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
