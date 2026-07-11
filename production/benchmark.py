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


# ---------------------------------------------------------------------------
# Real-measurement path (review blocker 6: the harness must measure the actual
# production solvers, not only synthetic stand-ins).
# ---------------------------------------------------------------------------


def _solver_and_seed_builders(spec: dict[str, Any]):
    """Return ``(build_solver, seed_state)`` for a spec's production solver.

    Uses the same construction helpers as the oracles so the measured cost is the
    production solver's, byte-identical configuration included.
    """

    representation = spec.get("representation")
    geometry = spec.get("geometry")
    physics = spec.get("physics")
    if geometry == "pcf" and representation == "vector_potential":
        from production.oracles import _curl_solver_from_spec

        return (
            lambda: _curl_solver_from_spec(spec),
            lambda solver: solver.initial_state(),
        )
    if geometry == "pcf" and physics in {"mhd", "mri"}:
        from production.oracles import (
            _pcf_mhd_perturbation_state,
            _pcf_mri_packet_state,
            _primitive_solver_from_spec,
        )

        def seed(solver: Any) -> Any:
            if physics == "mri":
                return _pcf_mri_packet_state(solver, spec)[0]
            return _pcf_mhd_perturbation_state(solver, spec)[0]

        return (lambda: _primitive_solver_from_spec(spec)), seed
    raise ValueError(
        f"no benchmark solver factory for {spec.get('problem_id')!r} "
        f"(geometry={geometry!r}, physics={physics!r}, representation={representation!r})"
    )


def _spec_dof(spec: dict[str, Any]) -> float:
    """Nominal degrees of freedom: grid size x number of evolved fields."""

    from production.oracles import _selected_resolution

    resolution = _selected_resolution(spec)
    cells = 1.0
    for key in ("Nx", "Ny", "Nz", "N", "Nr"):
        value = resolution.get(key)
        if value is not None:
            cells *= float(int(value))
    fields = max(1, len(spec.get("evolved_variables", [])))
    return cells * fields


def measure_spec(
    config_path: str | Any,
    *,
    tiers: tuple[str, ...] = ("smoke",),
    timed_steps: int = 10,
    warmup_steps: int = 2,
    shear_times: float | None = None,
) -> dict[str, Any]:
    """FJ-12: measure the real production solver at materialized resolution tiers.

    Returns an artifact dict with per-tier :class:`StepTiming` measurements, the
    fitted cost model when >= 2 tiers were measured, and predicted GPU-hours for
    the spec's configured horizon. CPU measurements bound the campaign shape; the
    workhorse *decision* still requires the authorized GPU run of this same CLI.
    """

    from production.adapters import load_config

    measurements: list[dict[str, Any]] = []
    dofs: list[float] = []
    times: list[float] = []
    spec: dict[str, Any] = {}
    for tier in tiers:
        config = load_config(config_path, resolution_tier=tier)
        spec = config.spec
        build_solver, seed_state = _solver_and_seed_builders(spec)
        timing = benchmark_step(
            build_solver,
            label=f"{spec['problem_id']}@{tier}",
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            seed_state=seed_state,
        )
        horizon = (
            float(shear_times)
            if shear_times is not None
            else float(spec["time"]["final_time"])
        )
        dof = _spec_dof(spec)
        measurements.append(
            {
                "tier": tier,
                "dof": dof,
                **timing.to_dict(),
                "predicted_hours_full_horizon": predicted_gpu_hours(
                    timing.warm_step_s, shear_times=horizon, dt=timing.dt
                ),
            }
        )
        dofs.append(dof)
        times.append(timing.warm_step_s)

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "problem_id": spec.get("problem_id"),
        "spec_hash": spec.get("spec_hash"),
        "backend": _backend_name(),
        "timed_steps": int(timed_steps),
        "warmup_steps": int(warmup_steps),
        "measurements": measurements,
        "provenance": _provenance_safe(),
    }
    if len(measurements) >= 2:
        model = fit_cost_model(dofs, times)
        artifact["cost_model"] = {
            "a": model.a,
            "b": model.b,
            "dofs": model.dofs,
            "times": model.times,
        }
    else:
        artifact["cost_model"] = None
        artifact["cost_model_note"] = (
            "need >= 2 measured tiers to fit the power-law cost model"
        )
    return artifact


def _backend_name() -> str:
    try:
        import jax

        return str(jax.default_backend())
    except Exception:  # pragma: no cover
        return "unknown"


def _provenance_safe() -> dict[str, Any]:
    try:
        from production.provenance import capture_provenance

        return capture_provenance()
    except Exception:  # pragma: no cover - provenance must never break a benchmark
        return {}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="FJ-12: benchmark the real production solver per resolution tier."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tiers", default="smoke", help="comma-separated tiers")
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--shear-times", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    artifact = measure_spec(
        args.config,
        tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
        timed_steps=args.timed_steps,
        warmup_steps=args.warmup_steps,
        shear_times=args.shear_times,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    for row in artifact["measurements"]:
        print(
            f"{row['label']}: dof={row['dof']:.0f} compile={row['compile_s']:.2f}s "
            f"warm_step={row['warm_step_s'] * 1000.0:.2f}ms "
            f"full_horizon={row['predicted_hours_full_horizon']:.2f}h"
        )
    print(f"wrote benchmark artifact -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
