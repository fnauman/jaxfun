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

    time_spec = spec.get("time", {})
    if "adaptive_cfl" not in time_spec:
        return None
    block = time_spec["adaptive_cfl"]
    if block is False:
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
    of the next).  An unsafe initial dt is shrunk before any stepping.  If an
    evolved block crosses the hard CFL ceiling, the run aborts because that
    already-completed block cannot be repaired by shrinking the next one.

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
    if 0.0 < elapsed_target < config.dt_min:
        raise ValueError(
            f"elapsed_target={elapsed_target:g} is smaller than "
            f"dt_min={config.dt_min:g}; the exact horizon cannot be reached "
            "without violating the configured timestep floor"
        )

    def measure(current_state: Any) -> tuple[dict[str, float], float]:
        values = {
            str(key): float(value)
            for key, value in health_scalars_fn(solver, current_state).items()
        }
        return values, float(values.get("cfl_total", math.nan))

    def require_finite_cfl(cfl_total: float, *, done: int, t: float) -> None:
        if not math.isfinite(cfl_total) or cfl_total < 0.0:
            raise FloatingPointError(
                f"adaptive CFL measurement is invalid at tstep={done} "
                f"t={t:g}: cfl_total={cfl_total!r}"
            )

    def record_dt_change(
        *,
        done: int,
        t: float,
        dt_old: float,
        dt_new: float,
        cfl_total: float,
        cfl_total_projected: float,
        reason: str,
    ) -> None:
        changes.append(
            {
                "tstep": float(done),
                "t": float(t),
                "dt_old": float(dt_old),
                "dt_new": float(dt_new),
                "cfl_total": float(cfl_total),
                "cfl_total_projected": float(cfl_total_projected),
                "reason": reason,
            }
        )

    def maybe_adapt(current_state: Any, cfl_total: float, done: int, t: float) -> Any:
        old_dt = float(solver.dt)
        new_dt = _proposed_dt(old_dt, cfl_total, config)
        if new_dt is None:
            return current_state
        # At dt_min a requested shrink can collapse to the current timestep.
        # Do not report a no-op as a change, and never advance when the CFL
        # safety ceiling cannot be met at the configured floor.
        if math.isclose(new_dt, old_dt, rel_tol=1.0e-12, abs_tol=0.0):
            if cfl_total > 1.0:
                raise RuntimeError(
                    "adaptive CFL cannot satisfy the safety ceiling at "
                    f"dt_min={config.dt_min:g}: cfl_total={cfl_total:g}"
                )
            return current_state
        projected_cfl = cfl_total * float(new_dt) / old_dt
        if projected_cfl > 1.0:
            raise RuntimeError(
                "adaptive CFL cannot satisfy the safety ceiling at "
                f"dt={new_dt:g}: projected cfl_total={projected_cfl:g}"
            )
        record_dt_change(
            done=done,
            t=t,
            dt_old=old_dt,
            dt_new=float(new_dt),
            cfl_total=cfl_total,
            cfl_total_projected=projected_cfl,
            reason="cfl",
        )
        solver.set_dt(new_dt)
        return _reset_multistep_history(current_state)

    t = float(t0)
    done = 0
    changes: list[dict[str, Any]] = []
    # These bounds describe timesteps that actually advanced the state.  In
    # particular, an unsafe configured dt removed by the pre-flight check was
    # never "used" and must not pollute dt_max_used.
    dt_used_min = math.inf
    dt_used_max = 0.0
    dt_last_used: float | None = None
    cfl_max = 0.0
    final_step_clipped = False
    time_left = elapsed_target
    # Round-off guard: treat anything below this fraction of dt_min as "done".
    tiny = 1.0e-12 * max(elapsed_target, float(solver.dt))

    # Pre-flight: adapt to the *initial* state before any stepping, so an
    # unsafe starting dt never evolves a block.
    health, cfl_total = measure(state)
    require_finite_cfl(cfl_total, done=done, t=t)
    cfl_max = max(cfl_max, cfl_total) if math.isfinite(cfl_total) else cfl_max
    state = maybe_adapt(state, cfl_total, done, t)

    working_dt = float(solver.dt)
    while time_left > tiny:
        working_dt = float(solver.dt)
        restore_dt: float | None = None
        n_full = int(time_left / working_dt + 1.0e-12)
        remainder = time_left - n_full * working_dt
        if (
            n_full <= int(config.check_every)
            and remainder > tiny
            and remainder < config.dt_min * (1.0 - 1.0e-12)
        ):
            # A naive final clip would violate dt_min.  Redistribute the
            # remaining horizon over equal endpoint steps.  Prefer one extra,
            # smaller step (never worsens CFL); when that would fall below the
            # floor, use fewer slightly larger steps only if the projected CFL
            # and dt_max constraints remain safe.
            n_endpoint = n_full + 1
            endpoint_dt = time_left / n_endpoint
            if endpoint_dt < config.dt_min * (1.0 - 1.0e-12):
                if n_full < 1:
                    raise RuntimeError(
                        "exact final-time landing would require a timestep below "
                        f"dt_min={config.dt_min:g}"
                    )
                n_endpoint = n_full
                endpoint_dt = time_left / n_endpoint
            endpoint_cfl = cfl_total * endpoint_dt / working_dt
            if endpoint_dt > config.dt_max * (1.0 + 1.0e-12):
                raise RuntimeError(
                    "exact final-time redistribution would exceed "
                    f"dt_max={config.dt_max:g}: dt={endpoint_dt:g}"
                )
            if endpoint_cfl > 1.0:
                raise RuntimeError(
                    "exact final-time redistribution would exceed the CFL "
                    f"safety ceiling: projected cfl_total={endpoint_cfl:g}"
                )
            record_dt_change(
                done=done,
                t=t,
                dt_old=working_dt,
                dt_new=endpoint_dt,
                cfl_total=cfl_total,
                cfl_total_projected=endpoint_cfl,
                reason="final_time_redistribution",
            )
            restore_dt = working_dt
            solver.set_dt(endpoint_dt)
            state = _reset_multistep_history(state)
            final_step_clipped = True
            n = n_endpoint
        elif time_left < working_dt * (1.0 - 1.0e-12):
            # Clip the final step so the run lands exactly on the target time
            # and record the actual endpoint timestep just like a CFL change.
            # Endpoint redistribution above guarantees this clip respects the
            # configured dt_min floor.
            clipped_dt = float(time_left)
            if clipped_dt < config.dt_min * (1.0 - 1.0e-12):
                raise RuntimeError(
                    "exact final-time landing would require "
                    f"dt={clipped_dt:g} below dt_min={config.dt_min:g}"
                )
            clipped_cfl = cfl_total * clipped_dt / working_dt
            record_dt_change(
                done=done,
                t=t,
                dt_old=working_dt,
                dt_new=clipped_dt,
                cfl_total=cfl_total,
                cfl_total_projected=clipped_cfl,
                reason="final_time_clip",
            )
            restore_dt = working_dt
            solver.set_dt(clipped_dt)
            state = _reset_multistep_history(state)
            final_step_clipped = True
            n = 1
        else:
            n = min(int(config.check_every), int(time_left / working_dt + 1.0e-12))
            n = max(n, 1)
        step_dt = float(solver.dt)
        dt_used_min = min(dt_used_min, step_dt)
        dt_used_max = max(dt_used_max, step_dt)
        dt_last_used = step_dt
        state = solver.solve(state, n)
        done += n
        advanced = n * step_dt
        t += advanced
        time_left -= advanced
        health, cfl_total = measure(state)
        require_finite_cfl(cfl_total, done=done, t=t)
        cfl_max = max(cfl_max, cfl_total) if math.isfinite(cfl_total) else cfl_max
        if cfl_total > 1.0:
            raise RuntimeError(
                "adaptive CFL safety ceiling exceeded after an evolved block "
                f"at tstep={done} t={t:g}: cfl_total={cfl_total:g}; the "
                "completed block cannot be repaired by shrinking the next dt"
            )
        if on_block is not None:
            on_block(t, done, state, health)
        if restore_dt is not None:
            # The endpoint step is an exact-horizon implementation detail, not
            # the controller's next working dt.  Restore the controller and
            # clear AB2 history so the returned solver/state pair is honest.
            solver.set_dt(restore_dt)
            state = _reset_multistep_history(state)
        if time_left > tiny:
            # The post-block measurement doubles as the next pre-block check.
            state = maybe_adapt(state, cfl_total, done, t)

    if dt_last_used is None:
        # A zero-horizon run advances no step; retain finite, unsurprising
        # reporting without claiming the pre-flight configured dt was used.
        dt_last_used = float(solver.dt)
        dt_used_min = dt_last_used
        dt_used_max = dt_last_used

    record = {
        "adaptive_cfl_target": float(config.target),
        "dt_final": float(solver.dt),
        "dt_last_used": float(dt_last_used),
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
