"""Third-order one-evaluation SBDF3 regressions for the PCF workhorses."""

from __future__ import annotations

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from examples.pcf_mhd_mri_shearpy_jax import (
    PlaneCouetteMRIShearpyInsulatingJax,
    PlaneCouetteMRIShearpyJax,
)
from production.oracles import _checkpoint_payload, _state_from_checkpoint_payload


def _hydro(dt: float = 1.0e-3) -> PlaneCouetteFluctuationJax:
    return PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=dt,
        time_integrator="SBDF3",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.01,
    )


def _mhd(solver_type=PlaneCouetteMRIShearpyJax, dt: float = 1.0e-3):
    kwargs = {}
    if issubclass(solver_type, PlaneCouetteMRIShearpyJax):
        # The generic analytic seed is not KMM-consistent until its first
        # projection, which introduces an O(dt) mismatch in self-convergence.
        kwargs["solenoidal_velocity_seed"] = True
    return solver_type(
        N=(9, 4, 4),
        Re=200.0,
        Rm=200.0,
        dt=dt,
        time_integrator="SBDF3",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.01,
        magnetic_amplitude=0.005,
        **kwargs,
    )


def _assert_tree_close(left, right, *, rtol=2.0e-12, atol=2.0e-13) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    assert all(
        bool(jnp.allclose(a, b, rtol=rtol, atol=atol))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _kmm_consistent_initial_state(solver):
    state = solver.initial_state()
    means = (
        jnp.real(state.u[1][:, 0, 0]),
        jnp.real(state.u[2][:, 0, 0]),
    )
    return replace(
        state,
        u=solver._reconstruct_velocity(state.u[0], state.g, *means),
    )


def test_hydro_sbdf3_startup_scan_and_one_evaluation(monkeypatch) -> None:
    solver = _hydro()
    initial = solver.initial_state()
    first = solver.step(initial)
    second = solver.step(first)
    assert int(first.history_steps) == 1
    assert int(second.history_steps) == 2
    assert second.solution_old is not None
    assert second.solution_older is not None
    assert second.nonlinear_old is not None
    assert second.nonlinear_older is not None

    original = solver.convection
    calls = 0

    def counted(state):
        nonlocal calls
        calls += 1
        return original(state)

    monkeypatch.setattr(solver, "convection", counted)
    third = solver.step(second)
    assert calls == 1
    assert int(third.history_steps) == 2

    scan_solver = _hydro()
    scanned = scan_solver.solve(scan_solver.initial_state(), 4)
    eager_solver = _hydro()
    eager = eager_solver.initial_state()
    for _ in range(4):
        eager = eager_solver.step(eager)
    _assert_tree_close(scanned, eager)

    # Retain an explicit compiled-lax.scan gate across the dynamic startup
    # branch even though production bootstraps before its steady-only cache.
    transition_solver = _hydro()
    transitioned = transition_solver._rollout_cache(
        transition_solver.initial_state(), 4
    )
    _assert_tree_close(transitioned, eager)


def test_sbdf3_rejects_changed_dt_clearly() -> None:
    solver = _hydro()
    solver.set_dt(solver.dt)
    solver.set_dt(np.nextafter(solver.dt, math.inf))
    with pytest.raises(NotImplementedError, match="fixed dt"):
        solver.set_dt(0.5 * solver.dt)


@pytest.mark.parametrize("problem", ["hydro", "mhd"])
def test_sbdf3_does_not_mask_type_errors_from_step_assembly(
    problem, monkeypatch
) -> None:
    solver = _hydro() if problem == "hydro" else _mhd()
    state = solver.solve(solver.initial_state(), 2)

    def broken(*_args, **_kwargs):
        raise TypeError("assembly bug")

    if problem == "hydro":
        monkeypatch.setattr(solver, "_solve_prefactor_many", broken)
    else:
        monkeypatch.setattr(solver, "_step_sbdf3_steady", broken)
    with pytest.raises(TypeError, match="assembly bug"):
        solver.step(state)


@pytest.mark.parametrize(
    "solver_type",
    [PlaneCouetteMHDJax, PlaneCouetteMRIShearpyInsulatingJax],
    ids=["conducting", "insulating"],
)
def test_mhd_sbdf3_steady_step_evaluates_once_and_preserves_constraints(
    solver_type, monkeypatch
) -> None:
    solver = _mhd(solver_type)
    state = solver.solve(solver.initial_state(), 2)
    original = solver._mhd_convection
    calls = 0

    def counted(current):
        nonlocal calls
        calls += 1
        return original(current)

    monkeypatch.setattr(solver, "_mhd_convection", counted)
    state = solver.step(state)
    assert calls == 1
    assert int(state.flow.history_steps) == 2
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(state))
    assert float(solver.divergence_l2(state.flow)) < 1.0e-11
    assert float(solver.magnetic_divergence_l2(state)) < 1.0e-11
    if isinstance(solver, PlaneCouetteMRIShearpyInsulatingJax):
        assert float(solver.insulating_bc_residual(state)) < 1.0e-10


def test_sbdf3_checkpoint_preserves_two_level_history_exactly() -> None:
    parent = _mhd()
    checkpoint_state = parent.solve(parent.initial_state(), 4)
    payload = _checkpoint_payload(checkpoint_state)
    restored = _state_from_checkpoint_payload(
        payload, state_kind="pcf_vector_potential_mhd_saturation"
    )
    assert int(restored.flow.history_steps) == 2

    expected = parent.solve(checkpoint_state, 2)
    resumed = _mhd().solve(restored, 2)
    jax.block_until_ready((expected, resumed))
    assert all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(
            jax.tree.leaves(expected), jax.tree.leaves(resumed), strict=True
        )
    )


def _state_l2(left, right, *, mhd: bool) -> float:
    a = (*left.flow.u, left.flow.g, *left.A) if mhd else (*left.u, left.g)
    b = (*right.flow.u, right.flow.g, *right.A) if mhd else (*right.u, right.g)
    return math.sqrt(
        sum(float(jnp.vdot(x - y, x - y).real) for x, y in zip(a, b, strict=True))
    )


@pytest.mark.slow
@pytest.mark.parametrize("problem", ["hydro", "mhd"])
def test_pcf_sbdf3_has_third_order_self_convergence(problem) -> None:
    final_time = 0.04
    step_counts = (8, 16, 32, 64)
    solutions = []
    for steps in step_counts:
        dt = final_time / steps
        if problem == "hydro":
            solver = _hydro(dt)
            initial = _kmm_consistent_initial_state(solver)
        else:
            solver = _mhd(dt=dt)
            initial = solver.initial_state()
        solutions.append(solver.solve(initial, steps))

    errors = [
        _state_l2(solutions[i], solutions[i + 1], mhd=problem == "mhd")
        for i in range(3)
    ]
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(2.7 <= order <= 3.3 for order in orders), (errors, orders)


def test_hydro_sbdf3_agrees_with_small_dt_imexrk3_reference() -> None:
    final_time = 0.01
    sbdf = _hydro(final_time / 20)
    sbdf_initial = _kmm_consistent_initial_state(sbdf)
    actual = sbdf.solve(sbdf_initial, 20)

    reference = PlaneCouetteFluctuationJax(
        N=(9, 4, 4),
        Re=200.0,
        dt=final_time / 160,
        time_integrator="IMEXRK3",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.01,
    )
    reference_initial = _kmm_consistent_initial_state(reference)
    expected = reference.solve(reference_initial, 160)
    assert _state_l2(actual, expected, mhd=False) < 2.0e-10


@pytest.mark.slow
def test_kmm_diffusion_only_mode_has_third_order_accuracy() -> None:
    final_time = 0.2
    errors = []
    for steps in (8, 16, 32, 64):
        solver = PlaneCouetteFluctuationJax(
            N=(9, 4, 4),
            Re=20.0,
            dt=final_time / steps,
            time_integrator="SBDF3",
            padding_factor=(1.0, 1.0, 1.0),
            perturbation_amplitude=0.0,
        )
        state = solver.zero_state()
        initial_mean = jnp.linspace(0.01, 0.02, state.u[1].shape[0])
        velocity = list(state.u)
        velocity[1] = velocity[1].at[:, 0, 0].set(initial_mean)
        state = replace(state, u=tuple(velocity))
        actual = solver.solve(state, steps).u[1][:, 0, 0]

        generator = np.linalg.solve(
            np.asarray(solver.M00.todense()), np.asarray(solver.L00.todense())
        )
        expected = expm(generator * final_time) @ np.asarray(initial_mean)
        errors.append(float(jnp.linalg.norm(actual - expected)))

    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(3)]
    assert all(2.7 <= order <= 3.3 for order in orders), (errors, orders)
