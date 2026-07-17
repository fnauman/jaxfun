"""FJ-12: benchmark harness + cost model (synthetic; no GPU/solver required)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from production.benchmark import (
    benchmark_step,
    fit_cost_model,
    predicted_gpu_hours,
)


def test_cost_model_recovers_power_law():
    # times = 1e-6 * dof^1.0
    dofs = [1e3, 1e4, 1e5, 1e6]
    times = [1e-6 * d for d in dofs]
    model = fit_cost_model(dofs, times)
    assert model.b == pytest.approx(1.0, abs=1e-6)
    assert model.predict(1e7) == pytest.approx(1e-6 * 1e7, rel=1e-3)


def test_cost_model_validates_on_held_out_grid_within_tolerance():
    dofs = [1e3, 1e4, 1e5]
    times = [3e-7 * d**1.1 for d in dofs]
    model = fit_cost_model(dofs, times)
    held_out_dof = 1e6
    observed = 3e-7 * held_out_dof**1.1
    assert model.relative_error(held_out_dof, observed) < 0.2  # FJ-12 gate


def test_fit_requires_two_samples():
    with pytest.raises(ValueError):
        fit_cost_model([1e3], [1e-3])


def test_predicted_gpu_hours():
    # 1 ms/step, dt=0.005, 300 shear times -> 300/0.005 = 60000 steps * 1ms = 60 s
    hours = predicted_gpu_hours(1e-3, shear_times=300.0, dt=0.005)
    assert hours == pytest.approx(60.0 / 3600.0, rel=1e-6)
    # 20% I/O overhead
    hours_io = predicted_gpu_hours(
        1e-3, shear_times=300.0, dt=0.005, io_overhead_frac=0.2
    )
    assert hours_io == pytest.approx(hours * 1.2, rel=1e-6)


class _MockSolver:
    dt = 0.01

    def zero_state(self):
        return 0

    def step(self, state):
        return state + 1


class _MockProductionSolver(_MockSolver):
    def __init__(self):
        self.solve_calls = 0

    def step(self, state):  # pragma: no cover - production rollout must be used
        raise AssertionError("benchmark bypassed solver.solve")

    def solve(self, state, steps):
        self.solve_calls += 1
        return state + steps


@dataclass(frozen=True)
class _MockCacheInfo:
    hits: int
    misses: int
    live_entries: int


class _MockCachedProductionSolver(_MockProductionSolver):
    def __init__(self):
        super().__init__()
        self.hits = 0
        self.misses = 0
        self.live_entries = 0

    def solve(self, state, steps):
        self.solve_calls += 1
        if self.live_entries:
            self.hits += 1
        else:
            self.misses += 1
            self.live_entries = 1
        return state + steps

    def set_dt(self, dt):
        self.dt = float(dt)

    def rollout_cache_info(self):
        return _MockCacheInfo(self.hits, self.misses, self.live_entries)


def test_benchmark_step_separates_compile_and_warm():
    timing = benchmark_step(_MockSolver, label="mock", warmup_steps=1, timed_steps=5)
    assert timing.label == "mock"
    assert timing.timed_steps == 5
    assert timing.dt == 0.01
    assert timing.warm_step_s >= 0.0
    assert timing.cost_per_shear_time_s == pytest.approx(timing.warm_step_s / 0.01)
    d = timing.to_dict()
    assert "cost_per_shear_time_s" in d


def test_benchmark_step_times_compiled_production_rollout():
    built = []

    def build():
        solver = _MockProductionSolver()
        built.append(solver)
        return solver

    timing = benchmark_step(
        build,
        label="production-mock",
        warmup_steps=2,
        timed_steps=4,
        rollout_steps=7,
    )

    assert built[0].solve_calls == 1 + 2 + 4
    assert timing.rollout_steps == 7
    assert timing.to_dict()["timed_blocks"] == 4
    assert timing.to_dict()["total_timed_steps"] == 28


def test_benchmark_rejects_nonpositive_rollout_length():
    with pytest.raises(ValueError, match="rollout_steps"):
        benchmark_step(_MockSolver, label="bad", rollout_steps=0)


def test_benchmark_probes_dt_transitions_without_new_rollout_variant():
    timing = benchmark_step(
        _MockCachedProductionSolver,
        label="cached-production-mock",
        warmup_steps=0,
        timed_steps=1,
        rollout_steps=2,
        dt_transition_probes=3,
    )

    probe = timing.dt_transition_probe
    assert probe is not None
    assert probe["rollout_cache_hits_delta"] == 3
    assert probe["rollout_cache_misses_delta"] == 0
    assert probe["reused_compiled_variant"] is True
    assert timing.rollout_cache_info == {"hits": 4, "misses": 1, "live_entries": 1}


def test_benchmark_rejects_negative_dt_transition_probes():
    with pytest.raises(ValueError, match="dt_transition_probes"):
        benchmark_step(
            _MockSolver,
            label="bad-dt-probe",
            dt_transition_probes=-1,
        )
