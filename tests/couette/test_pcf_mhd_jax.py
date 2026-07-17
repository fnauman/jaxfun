import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from examples.pcf_mhd_mri_shearpy_jax import (
    PlaneCouetteMRIShearpyInsulatingJax,
)


def test_pcf_mhd_cnab2_evaluates_nonlinearity_once_per_step() -> None:
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        Re=200.0,
        Rm=200.0,
        dt=1.0e-3,
        time_integrator="CNAB2",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.01,
        magnetic_amplitude=0.005,
    )
    state = solver.initial_state()
    original = solver._mhd_convection
    calls = 0

    def counted(current):
        nonlocal calls
        calls += 1
        return original(current)

    solver._mhd_convection = counted
    out = solver.step(state)

    assert calls == 1
    assert bool(jnp.asarray(out.flow.have_old))
    assert out.nonlinear_A_old is not None


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


def _physical_projection_curl(fields, source_spaces, target_spaces, counts):
    curl_physical = (
        source_spaces[2].backward_primitive(fields[2], (0, 1, 0), N=counts)
        - source_spaces[1].backward_primitive(fields[1], (0, 0, 1), N=counts),
        source_spaces[0].backward_primitive(fields[0], (0, 0, 1), N=counts)
        - source_spaces[2].backward_primitive(fields[2], (1, 0, 0), N=counts),
        source_spaces[1].backward_primitive(fields[1], (1, 0, 0), N=counts)
        - source_spaces[0].backward_primitive(fields[0], (0, 1, 0), N=counts),
    )
    return tuple(
        target.mask_nyquist(target.forward(value))
        for target, value in zip(target_spaces, curl_physical, strict=True)
    )


@pytest.mark.parametrize("family", ["C", "L"], ids=["chebyshev", "legendre"])
@pytest.mark.parametrize(
    "solver_type",
    [PlaneCouetteMHDJax, PlaneCouetteMRIShearpyInsulatingJax],
    ids=["conducting", "insulating"],
)
def test_coefficient_curl_projection_matches_physical_reference(
    family, solver_type
) -> None:
    solver = solver_type(
        N=(13, 12, 12),
        family=family,
        dt=1.0e-3,
        perturbation_amplitude=0.0,
        magnetic_amplitude=0.0,
    )
    x, y, z = solver.X
    wall = 1.0 - x**2
    physical_A = (
        wall * jnp.sin(y) * jnp.cos(z),
        wall**2 * jnp.cos(2.0 * y) * jnp.sin(z),
        wall * jnp.sin(y) * jnp.sin(2.0 * z),
    )
    A = solver._A_state_from_physical(physical_A)
    counts = solver.TD.num_quad_points

    expected_B = _physical_projection_curl(
        A, solver.a_coeff_spaces, solver.b_coeff_spaces, counts
    )
    actual_B = solver.update_B_from_A(A)
    for actual, expected in zip(actual_B, expected_B, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12), float(
            jnp.max(jnp.abs(actual - expected))
        )

    expected_J = _physical_projection_curl(
        expected_B, solver.b_coeff_spaces, solver.j_coeff_spaces, counts
    )
    actual_J = solver.update_J_from_B(actual_B)
    for actual, expected in zip(actual_J, expected_J, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-11, atol=2.0e-11), float(
            jnp.max(jnp.abs(actual - expected))
        )


def test_backward_vector_uses_each_heterogeneous_tangential_space() -> None:
    class OffsetSpace:
        def __init__(self, offset):
            self.offset = offset

        def backward(self, coefficients, *, N=None):
            assert N == (3, 4, 5)
            return coefficients + self.offset

    fields = tuple(jnp.asarray([value]) for value in (1.0, 2.0, 3.0))
    spaces = tuple(OffsetSpace(offset) for offset in (10.0, 20.0, 30.0))

    transformed = PlaneCouetteMHDJax._backward_vector(fields, spaces, (3, 4, 5))

    assert [float(value[0]) for value in transformed] == [11.0, 22.0, 33.0]
