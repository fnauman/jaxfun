"""Adaptive-CFL time stepping for the production DNS runners.

Spectral IMEX solvers in this repository prefactorize their implicit
operators for a fixed ``dt``, so the production adaptation is chunked: the
run advances in compiled blocks at fixed ``dt``, and between blocks the
driver measures the explicit CFL number from the family's health scalars and,
when it leaves the configured band, adopts a new ``dt`` via ``solver.set_dt``
(which rebuilds the factorizations) before the next block.

Contracts kept honest:

* Elapsed time is accumulated exactly (``t += n * dt`` per block with the dt
  actually used), never inferred as ``tstep * dt``.
* Multistep (CNAB2) solvers restart with their IMEX-Euler bootstrap after a
  dt change: the Adams-Bashforth history belongs to the old step size, so the
  driver clears ``have_old``.
* Shrinking is immediate when the CFL exceeds the target; growth is damped
  (one ``growth_cap`` factor per check) and only engages when the measured
  CFL sits below ``grow_when_below * target``, giving hysteresis.
* Every dt decision is recorded; the runner reports the full adaptation
  history so a run cannot silently hide step-size churn.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

__all__ = ["AdaptiveCFLConfig", "adaptive_cfl_from_spec", "run_adaptive_cfl"]


@dataclass(frozen=True)
class AdaptiveCFLConfig:
    """Configuration of the chunked adaptive-CFL driver.

    ``target`` is the explicit-CFL setpoint (the health gate aborts at 1.0,
    so production targets should sit well below).  ``check_every`` is the
    compiled-block length in steps between CFL measurements.
    """

    target: float = 0.5
    safety: float = 0.9
    dt_min: float = 1.0e-8
    dt_max: float = 1.0
    check_every: int = 25
    growth_cap: float = 1.5
    grow_when_below: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 < self.target < 1.0):
            raise ValueError("adaptive CFL target must sit in (0, 1)")
        if not (0.0 < self.safety <= 1.0):
            raise ValueError("safety must sit in (0, 1]")
        if self.dt_min <= 0.0 or self.dt_max <= self.dt_min:
            raise ValueError("require 0 < dt_min < dt_max")
        if int(self.check_every) < 1:
            raise ValueError("check_every must be a positive step count")
        if self.growth_cap <= 1.0:
            raise ValueError("growth_cap must exceed 1")
        if not (0.0 < self.grow_when_below < 1.0):
            raise ValueError("grow_when_below must sit in (0, 1)")


def adaptive_cfl_from_spec(spec: dict[str, Any]) -> AdaptiveCFLConfig | None:
    """Return the adaptive config from ``spec['time']['adaptive_cfl']``.

    Absent block (or explicit false) means fixed-dt semantics.  ``true``
    selects all defaults; a dict overrides individual fields.
    """

    block = spec.get("time", {}).get("adaptive_cfl")
    if not block:
        return None
    if block is True:
        return AdaptiveCFLConfig()
    if not isinstance(block, dict):
        raise ValueError("time.adaptive_cfl must be a boolean or an object")
    return AdaptiveCFLConfig(**{str(k): v for k, v in block.items()})


def _reset_multistep_history(state: Any) -> Any:
    """Clear the AB2 history after a dt change (IMEX-Euler bootstrap)."""
    if dataclasses.is_dataclass(state) and hasattr(state, "have_old"):
        return dataclasses.replace(state, have_old=False)
    return state


def _proposed_dt(
    dt: float, cfl_total: float, config: AdaptiveCFLConfig
) -> float | None:
    """Return a new dt or None when the current one stays."""
    if not math.isfinite(cfl_total) or cfl_total <= 0.0:
        return None
    # cfl scales linearly with dt, so dt_at_target is exact for the measured state.
    dt_at_target = dt * config.target / cfl_total
    if cfl_total > config.target:
        return max(config.dt_min, config.safety * dt_at_target)
    if cfl_total < config.grow_when_below * config.target:
        grown = min(config.growth_cap * dt, config.safety * dt_at_target)
        grown = min(grown, config.dt_max)
        if grown > dt * (1.0 + 1.0e-12):
            return grown
    return None


def run_adaptive_cfl(
    solver: Any,
    state: Any,
    *,
    elapsed_target: float,
    config: AdaptiveCFLConfig,
    health_scalars_fn: Any,
    on_block: Any | None = None,
    t0: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Advance exactly ``elapsed_target`` time units with CFL-targeted dt.

    The horizon is a *time* target, not a step count: dt changes alter the
    number of steps, and the final step is clipped so the run lands exactly
    on ``t0 + elapsed_target`` (a fixed step count would overshoot the
    requested final time under growth and stop early under shrinkage).

    The CFL is measured on the state *before* every compiled block (the
    post-block measurement of one block doubles as the pre-block measurement
    of the next), so an unsafe initial or newly evolved dt is shrunk before
    any stepping instead of tripping the production health gate mid-block.

    ``health_scalars_fn(solver, state)`` must expose ``cfl_total`` (the
    family health contract).  ``on_block(t, tstep, state, health)`` runs
    after every compiled block with the exact accumulated time and the
    evolved state's health scalars.  Returns the final state and the
    adaptation record.
    """

    elapsed_target = float(elapsed_target)
    if elapsed_target < 0.0:
        raise ValueError("elapsed_target must be non-negative")
    if solver.dt > config.dt_max or solver.dt < config.dt_min:
        raise ValueError(
            f"initial dt={solver.dt:g} lies outside "
            f"[{config.dt_min:g}, {config.dt_max:g}]"
        )

    def measure(current_state: Any) -> tuple[dict[str, float], float]:
        values = {
            str(key): float(value)
            for key, value in health_scalars_fn(solver, current_state).items()
        }
        return values, float(values.get("cfl_total", math.nan))

    def maybe_adapt(current_state: Any, cfl_total: float, done: int, t: float) -> Any:
        new_dt = _proposed_dt(float(solver.dt), cfl_total, config)
        if new_dt is None:
            return current_state
        changes.append(
            {
                "tstep": float(done),
                "t": float(t),
                "dt_old": float(solver.dt),
                "dt_new": float(new_dt),
                "cfl_total": float(cfl_total),
            }
        )
        solver.set_dt(new_dt)
        nonlocal dt_used_min, dt_used_max
        dt_used_min = min(dt_used_min, float(new_dt))
        dt_used_max = max(dt_used_max, float(new_dt))
        return _reset_multistep_history(current_state)

    t = float(t0)
    done = 0
    changes: list[dict[str, float]] = []
    dt_used_min = float(solver.dt)
    dt_used_max = float(solver.dt)
    cfl_max = 0.0
    final_step_clipped = False
    time_left = elapsed_target
    # Round-off guard: treat anything below this fraction of dt_min as "done".
    tiny = 1.0e-12 * max(elapsed_target, float(solver.dt))

    # Pre-flight: adapt to the *initial* state before any stepping, so an
    # unsafe starting dt never evolves a block.
    health, cfl_total = measure(state)
    cfl_max = max(cfl_max, cfl_total) if math.isfinite(cfl_total) else cfl_max
    state = maybe_adapt(state, cfl_total, done, t)

    working_dt = float(solver.dt)
    while time_left > tiny:
        working_dt = float(solver.dt)
        if time_left < working_dt * (1.0 - 1.0e-12):
            # Clip the final step so the run lands exactly on the target time
            # (the working dt is kept for the record; the run ends here).
            solver.set_dt(time_left)
            state = _reset_multistep_history(state)
            final_step_clipped = True
            n = 1
        else:
            n = min(int(config.check_every), int(time_left / working_dt + 1.0e-12))
            n = max(n, 1)
        state = solver.solve(state, n)
        done += n
        advanced = n * float(solver.dt)
        t += advanced
        time_left -= advanced
        health, cfl_total = measure(state)
        cfl_max = max(cfl_max, cfl_total) if math.isfinite(cfl_total) else cfl_max
        if on_block is not None:
            on_block(t, done, state, health)
        if time_left > tiny:
            # The post-block measurement doubles as the next pre-block check.
            state = maybe_adapt(state, cfl_total, done, t)

    record = {
        "adaptive_cfl_target": float(config.target),
        "dt_final": float(working_dt),
        "dt_min_used": float(dt_used_min),
        "dt_max_used": float(dt_used_max),
        "dt_changes": changes,
        "n_dt_changes": len(changes),
        "steps_taken": int(done),
        "final_step_clipped": bool(final_step_clipped),
        "elapsed_time": float(t - t0),
        "cfl_total_max_observed": float(cfl_max),
    }
    return state, record
