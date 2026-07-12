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
    steps: int,
    config: AdaptiveCFLConfig,
    health_scalars_fn: Any,
    on_block: Any | None = None,
    t0: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Advance ``steps`` with chunked CFL-targeted dt adaptation.

    ``health_scalars_fn(solver, state)`` must expose ``cfl_total`` (the
    family health contract).  ``on_block(t, tstep, state, health)`` runs
    after every compiled block with the exact accumulated time.  Returns the
    final state and the adaptation record.
    """

    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if solver.dt > config.dt_max or solver.dt < config.dt_min:
        raise ValueError(
            f"initial dt={solver.dt:g} lies outside "
            f"[{config.dt_min:g}, {config.dt_max:g}]"
        )

    t = float(t0)
    done = 0
    changes: list[dict[str, float]] = []
    dt_used_min = float(solver.dt)
    dt_used_max = float(solver.dt)
    cfl_max = 0.0
    while done < steps:
        n = min(int(config.check_every), steps - done)
        state = solver.solve(state, n)
        done += n
        t += n * float(solver.dt)
        health = {
            str(key): float(value)
            for key, value in health_scalars_fn(solver, state).items()
        }
        cfl_total = float(health.get("cfl_total", math.nan))
        cfl_max = max(cfl_max, cfl_total) if math.isfinite(cfl_total) else cfl_max
        if on_block is not None:
            on_block(t, done, state, health)
        if done >= steps:
            break
        new_dt = _proposed_dt(float(solver.dt), cfl_total, config)
        if new_dt is not None:
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
            state = _reset_multistep_history(state)
            dt_used_min = min(dt_used_min, float(new_dt))
            dt_used_max = max(dt_used_max, float(new_dt))
    record = {
        "adaptive_cfl_target": float(config.target),
        "dt_final": float(solver.dt),
        "dt_min_used": float(dt_used_min),
        "dt_max_used": float(dt_used_max),
        "dt_changes": changes,
        "n_dt_changes": len(changes),
        "elapsed_time": float(t - t0),
        "cfl_total_max_observed": float(cfl_max),
    }
    return state, record
