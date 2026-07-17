"""Restart and variable-step history regressions for PCF vector-potential CNAB2."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax
from production.oracles import _checkpoint_payload, _state_from_checkpoint_payload


def _solver(dt: float) -> PlaneCouetteMRIShearpyJax:
    return PlaneCouetteMRIShearpyJax(
        N=(9, 4, 4),
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


def test_cnab2_checkpoint_preserves_variable_step_history_exactly() -> None:
    parent = _solver(1.0e-3)
    checkpoint_state = parent.solve(parent.initial_state(), 3)
    payload = _checkpoint_payload(checkpoint_state)
    restored = _state_from_checkpoint_payload(
        payload, state_kind="pcf_vector_potential_mhd_saturation"
    )

    parent.set_dt(7.0e-4)
    expected = parent.solve(checkpoint_state, 2)
    resumed_solver = _solver(7.0e-4)
    actual = resumed_solver.solve(restored, 2)
    jax.block_until_ready((expected, actual))

    assert bool(jnp.asarray(actual.flow.have_old))
    assert float(actual.flow.previous_dt) == 7.0e-4
    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
        )
    )
