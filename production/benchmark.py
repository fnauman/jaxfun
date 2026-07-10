"""3-D performance benchmark harness + cost model (FJ-12).

Correctness gates do not make a campaign affordable. This harness measures the cost
of a solver so a campaign's GPU-hours can be predicted and enforced. It deliberately
separates:

* **compile/factorization** time (first traced step / operator build),
* **warm steady-state** time per step (median over a timed window),
* **checkpoint/diagnostic I/O** (measured by the caller and passed in),
* **peak memory** (best-effort from the JAX live-buffer stats).

and reports **cost per simulated shear time** rather than only per step. The workhorse
*decision* remains gated on an authorized GPU run -- this module only produces the
measurements and a fitted cost model; it does not select a solver.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class StepTiming:
    label: str
    compile_s: float
    warm_step_s: float
    warm_step_p50_s: float
    warm_step_p90_s: float
    timed_steps: int
    dt: float
    peak_bytes: int | None

    @property
    def cost_per_shear_time_s(self) -> float:
        """Wall-seconds to advance one shear time ``S*t = 1`` (dt in shear units)."""
        if self.dt <= 0.0:
            return float("nan")
        return self.warm_step_s / self.dt

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_per_shear_time_s"] = self.cost_per_shear_time_s
        return d


def _block_until_ready(state: Any) -> None:
    try:
        import jax

        jax.block_until_ready(state)
    except Exception:  # pragma: no cover - non-JAX states
        pass


def _peak_bytes() -> int | None:
    try:
        import jax

        for device in jax.devices():
            stats = device.memory_stats() if hasattr(device, "memory_stats") else None
            if stats and "peak_bytes_in_use" in stats:
                return int(stats["peak_bytes_in_use"])
    except Exception:  # pragma: no cover
        return None
    return None


def benchmark_step(
    build_solver: Callable[[], Any],
    *,
    label: str,
    warmup_steps: int = 2,
    timed_steps: int = 10,
    seed_state: Callable[[Any], Any] | None = None,
) -> StepTiming:
    """Benchmark a solver's compile vs warm-cache steady-state per-step cost.

    ``build_solver()`` returns a solver exposing ``step(state)`` and ``dt``. If given,
    ``seed_state(solver)`` builds the initial state; otherwise ``solver.zero_state()``.
    """

    t0 = time.perf_counter()
    solver = build_solver()
    state = seed_state(solver) if seed_state is not None else solver.zero_state()
    # First step includes tracing/compilation and operator factorization.
    state = solver.step(state)
    _block_until_ready(state)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup_steps)):
        state = solver.step(state)
    _block_until_ready(state)

    per_step: list[float] = []
    for _ in range(max(1, timed_steps)):
        s = time.perf_counter()
        state = solver.step(state)
        _block_until_ready(state)
        per_step.append(time.perf_counter() - s)

    arr = np.asarray(per_step)
    return StepTiming(
        label=label,
        compile_s=float(compile_s),
        warm_step_s=float(np.median(arr)),
        warm_step_p50_s=float(np.percentile(arr, 50)),
        warm_step_p90_s=float(np.percentile(arr, 90)),
        timed_steps=int(arr.size),
        dt=float(getattr(solver, "dt", float("nan"))),
        peak_bytes=_peak_bytes(),
    )


@dataclass(frozen=True)
class CostModel:
    """Power-law fit ``warm_step_s ~ a * dof^b`` over degrees of freedom."""

    a: float
    b: float
    dofs: list[float]
    times: list[float]

    def predict(self, dof: float) -> float:
        return float(self.a * dof**self.b)

    def relative_error(self, dof: float, observed: float) -> float:
        pred = self.predict(dof)
        return abs(pred - observed) / observed if observed else float("nan")


def fit_cost_model(dofs: list[float], warm_step_times: list[float]) -> CostModel:
    """Fit a log-log power law of per-step time vs total degrees of freedom."""

    d = np.asarray(dofs, dtype=float)
    t = np.asarray(warm_step_times, dtype=float)
    if d.size < 2 or np.any(d <= 0) or np.any(t <= 0):
        raise ValueError("need >=2 positive (dof, time) samples to fit a cost model")
    b, log_a = np.polyfit(np.log(d), np.log(t), 1)
    return CostModel(a=float(np.exp(log_a)), b=float(b), dofs=list(d), times=list(t))


def predicted_gpu_hours(
    warm_step_s: float, *, shear_times: float, dt: float, io_overhead_frac: float = 0.0
) -> float:
    """Predicted GPU-hours to advance ``shear_times`` at step cost ``warm_step_s``."""

    if dt <= 0.0:
        return float("nan")
    steps = shear_times / dt
    seconds = steps * warm_step_s * (1.0 + io_overhead_frac)
    return seconds / 3600.0
