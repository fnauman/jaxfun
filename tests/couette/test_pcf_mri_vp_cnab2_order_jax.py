"""Temporal-order gate for the vector-potential PCF CNAB2 workhorse."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax

pytestmark = pytest.mark.slow


def _state_l2(a, b) -> float:
    left = (*a.flow.u, a.flow.g, *a.A)
    right = (*b.flow.u, b.flow.g, *b.A)
    total = sum(
        float(jnp.vdot(x - y, x - y).real) for x, y in zip(left, right, strict=True)
    )
    return math.sqrt(total)


def _run(dt: float, steps: int):
    solver = PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        Re=200.0,
        Rm=200.0,
        omega=2.0 / 3.0,
        shear_rate=1.0,
        background_b=(0.0, 0.0, 0.05),
        dt=dt,
        time_integrator="CNAB2",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.01,
        magnetic_amplitude=0.005,
        solenoidal_velocity_seed=True,
    )
    return solver.solve(solver.initial_state(), steps)


def test_pcf_mri_vector_potential_cnab2_is_second_order() -> None:
    jax.config.update("jax_enable_x64", True)
    final_time = 0.04
    base_steps = 16
    coarse = _run(final_time / base_steps, base_steps)
    medium = _run(final_time / (2 * base_steps), 2 * base_steps)
    fine = _run(final_time / (4 * base_steps), 4 * base_steps)

    error_coarse = _state_l2(coarse, medium)
    error_fine = _state_l2(medium, fine)
    assert error_fine > 0.0
    order = math.log2(error_coarse / error_fine)
    assert 1.8 <= order <= 2.2, (
        f"observed order {order:.3f}; "
        f"errors were {error_coarse:.3e} and {error_fine:.3e}"
    )
