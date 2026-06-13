"""Compare production diagnostics from two device-specific runner subprocesses."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScalarComparison:
    key: str
    left: Any | None
    right: Any | None
    atol: float
    rtol: float
    passed: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "left": self.left,
            "right": self.right,
            "atol": self.atol,
            "rtol": self.rtol,
            "passed": self.passed,
            "message": self.message,
        }


def run_device_comparison(
    *,
    config: str | Path,
    out: str | Path,
    device_a: str = "cpu",
    device_b: str = "auto",
    steps: int | None = None,
    resolution_tier: str | None = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-6,
    timeout_seconds: float = 1800.0,
    python: str | Path = sys.executable,
) -> dict[str, Any]:
    """Run a config twice in subprocesses and compare final numeric diagnostics."""

    out_root = Path(out)
    out_root.mkdir(parents=True, exist_ok=True)
    left_dir = out_root / f"a_{_safe_label(device_a)}"
    right_dir = out_root / f"b_{_safe_label(device_b)}"
    left = _run_problem_subprocess(
        config=config,
        out=left_dir,
        device=device_a,
        steps=steps,
        resolution_tier=resolution_tier,
        timeout_seconds=timeout_seconds,
        python=python,
    )
    right = _run_problem_subprocess(
        config=config,
        out=right_dir,
        device=device_b,
        steps=steps,
        resolution_tier=resolution_tier,
        timeout_seconds=timeout_seconds,
        python=python,
    )
    comparisons: list[ScalarComparison] = []
    if left["returncode"] == 0 and right["returncode"] == 0:
        comparisons = compare_final_diagnostics(
            left_dir, right_dir, atol=atol, rtol=rtol
        )
    report = {
        "schema_version": 1,
        "config": str(config),
        "devices": {"left": device_a, "right": device_b},
        "run_options": {"steps": steps, "resolution_tier": resolution_tier},
        "runs": {"left": left, "right": right},
        "timing": _timing_report(left, right),
        "summary": _summary(
            comparisons,
            subprocess_ok=left["returncode"] == 0 and right["returncode"] == 0,
        ),
        "comparisons": [comparison.to_dict() for comparison in comparisons],
    }
    report_path = out_root / "device_comparison.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def compare_final_diagnostics(
    left_run: str | Path,
    right_run: str | Path,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-6,
) -> list[ScalarComparison]:
    """Compare final numeric scalar diagnostics from two run directories."""

    left = _comparable_scalars(_load_final_row(Path(left_run) / "diagnostics.jsonl"))
    right = _comparable_scalars(_load_final_row(Path(right_run) / "diagnostics.jsonl"))
    comparisons: list[ScalarComparison] = []
    for key in sorted(set(left) | set(right)):
        if key not in left:
            comparisons.append(
                ScalarComparison(
                    key, None, right[key], atol, rtol, False, "missing on left"
                )
            )
            continue
        if key not in right:
            comparisons.append(
                ScalarComparison(
                    key, left[key], None, atol, rtol, False, "missing on right"
                )
            )
            continue
        comparisons.append(_compare_scalar(key, left[key], right[key], atol, rtol))
    return comparisons


def _run_problem_subprocess(
    *,
    config: str | Path,
    out: Path,
    device: str,
    steps: int | None,
    resolution_tier: str | None,
    timeout_seconds: float,
    python: str | Path,
) -> dict[str, Any]:
    cmd = [
        str(python),
        "production/run_problem.py",
        "--config",
        str(config),
        "--out",
        str(out),
        "--device",
        device,
    ]
    if steps is not None:
        cmd.extend(["--steps", str(int(steps))])
    if resolution_tier is not None:
        cmd.extend(["--resolution-tier", resolution_tier])
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("JAXFUN_PRODUCTION_DTYPE", "float32")
    if device == "auto":
        env.pop("JAX_PLATFORMS", None)
    started = True
    wall_start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "out_dir": str(out),
            "command": cmd,
            "returncode": 124,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "started": started,
            "wall_time_seconds": time.perf_counter() - wall_start,
        }
    return {
        "out_dir": str(out),
        "command": cmd,
        "returncode": completed.returncode,
        "timed_out": False,
        "timeout_seconds": timeout_seconds,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started": started,
        "wall_time_seconds": time.perf_counter() - wall_start,
    }


def _timing_report(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_time = left.get("wall_time_seconds")
    right_time = right.get("wall_time_seconds")
    speedup_left_over_right = None
    if left_time and right_time and float(right_time) > 0.0:
        speedup_left_over_right = float(left_time) / float(right_time)
    return {
        "left_wall_time_seconds": left_time,
        "right_wall_time_seconds": right_time,
        "left_over_right_speedup": speedup_left_over_right,
    }


def _load_final_row(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no diagnostics rows in {path}")
    return rows[-1]


def _comparable_scalars(row: dict[str, Any]) -> dict[str, float | bool]:
    out: dict[str, float | bool] = {}
    for key, value in row.items():
        if key == "t":
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int | float):
            out[key] = float(value)
    return out


def _compare_scalar(
    key: str, left: float | bool, right: float | bool, atol: float, rtol: float
) -> ScalarComparison:
    if isinstance(left, bool) or isinstance(right, bool):
        passed = isinstance(left, bool) and isinstance(right, bool) and left == right
        message = "" if passed else "boolean scalar mismatch"
        return ScalarComparison(key, left, right, atol, rtol, passed, message)
    tolerance = atol + rtol * abs(right)
    diff = abs(left - right)
    passed = math.isfinite(left) and math.isfinite(right) and diff <= tolerance
    message = "" if passed else f"abs diff {diff:.6e} > tolerance {tolerance:.6e}"
    return ScalarComparison(key, left, right, atol, rtol, passed, message)


def _summary(
    comparisons: list[ScalarComparison], *, subprocess_ok: bool
) -> dict[str, int]:
    if not subprocess_ok:
        return {"passed": 0, "failed": 1, "skipped": 0, "compared": 0}
    if not comparisons:
        return {"passed": 0, "failed": 1, "skipped": 0, "compared": 0}
    return {
        "passed": sum(1 for comparison in comparisons if comparison.passed),
        "failed": sum(1 for comparison in comparisons if not comparison.passed),
        "skipped": 0,
        "compared": len(comparisons),
    }


def _safe_label(device: str) -> str:
    return device.replace("/", "_").replace(" ", "_")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--device-a", default="cpu", choices=["auto", "cpu", "cuda", "gpu"]
    )
    parser.add_argument(
        "--device-b", default="auto", choices=["auto", "cpu", "cuda", "gpu"]
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--resolution-tier",
        choices=["smoke", "start", "production"],
        help="Materialize a nested production resolution tier for both runs.",
    )
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_device_comparison(
        config=args.config,
        out=args.out,
        device_a=args.device_a,
        device_b=args.device_b,
        steps=args.steps,
        resolution_tier=args.resolution_tier,
        atol=args.atol,
        rtol=args.rtol,
        timeout_seconds=args.timeout_seconds,
        python=args.python,
    )
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
