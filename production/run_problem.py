"""Production runner entrypoint for jaxfun problem specs.

This module wires validation, device capture, metadata, and golden resolution.
Actual solver execution is intentionally explicit: until a solver factory is
registered for a spec, non-validate-only runs fail with a clear error instead of
claiming parity.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .adapters import ProductionConfig, load_config
    from .compare_goldens import resolve_golden
    from .device import capture_device_record
    from .problem_spec import ProblemSpecError, UnsupportedSpecError
except ImportError:  # pragma: no cover - direct script mode
    from adapters import ProductionConfig, load_config  # type: ignore
    from compare_goldens import resolve_golden  # type: ignore
    from device import capture_device_record  # type: ignore
    from problem_spec import ProblemSpecError, UnsupportedSpecError  # type: ignore


class SolverExecutionNotImplementedError(RuntimeError):
    """Raised until a production solver path is wired for a spec."""


def run_problem(
    *,
    config_path: str | Path,
    out: str | Path,
    compare_golden: bool = False,
    shenfun_golden: str | Path | None = None,
    write_golden: bool = False,
    device: str = "auto",
    steps: int | None = None,
    checkpoint_every: int | None = None,
    validate_only: bool = False,
    capture_device: bool = True,
) -> dict[str, Any]:
    """Validate a config, write metadata, and eventually execute its solver."""

    config = load_config(config_path)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_record = capture_device_record(device) if capture_device else {"capture_skipped": True}
    metadata = build_metadata(
        config,
        config_path=Path(config_path),
        out_dir=out_dir,
        device_record=device_record,
        compare_golden=compare_golden,
        shenfun_golden=Path(shenfun_golden) if shenfun_golden is not None else None,
        write_golden=write_golden,
        requested_device=device,
        steps=steps,
        checkpoint_every=checkpoint_every,
        validate_only=validate_only,
    )
    _write_json(out_dir / "metadata.json", metadata)

    if validate_only:
        return metadata

    raise SolverExecutionNotImplementedError(
        f"production solver execution is not wired yet for {config.problem_id}; "
        "contract validation metadata was written, but no DNS or golden comparison was run"
    )


def build_metadata(
    config: ProductionConfig,
    *,
    config_path: Path,
    out_dir: Path,
    device_record: dict[str, Any],
    compare_golden: bool,
    shenfun_golden: Path | None,
    write_golden: bool,
    requested_device: str,
    steps: int | None,
    checkpoint_every: int | None,
    validate_only: bool,
) -> dict[str, Any]:
    golden_resolution = _golden_resolution_metadata(config.problem_id, shenfun_golden)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "problem_id": config.problem_id,
        "artifact_id": config.artifact_id,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "spec_hash": config.spec["spec_hash"],
        "geometry": config.geometry,
        "physics": config.physics,
        "support_state": config.spec["support_state"],
        "adapter": {
            **config.metadata,
            "solver_args": config.solver_args,
            "source_files": list(config.source_files),
        },
        "device": device_record,
        "run_options": {
            "requested_device": requested_device,
            "steps_override": steps,
            "checkpoint_every": checkpoint_every,
            "compare_golden": compare_golden,
            "write_golden": write_golden,
            "validate_only": validate_only,
        },
        "golden_resolution": golden_resolution,
        "execution": {
            "status": "validated" if validate_only else "not_started",
            "solver_execution_wired": False,
        },
    }


def _golden_resolution_metadata(problem_id: str, explicit_golden: Path | None) -> dict[str, Any]:
    if explicit_golden is not None:
        return {
            "policy": "explicit",
            "golden_path": str(explicit_golden),
            "exists": explicit_golden.exists(),
        }
    try:
        resolution = resolve_golden(problem_id)
    except FileNotFoundError as exc:
        return {
            "policy": "missing",
            "golden_path": None,
            "exists": False,
            "message": str(exc),
        }
    return {
        "policy": resolution.policy,
        "root": str(resolution.root),
        "golden_path": str(resolution.golden_path),
        "spec_path": str(resolution.spec_path),
        "exists": True,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(data), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare-golden", action="store_true")
    parser.add_argument("--shenfun-golden")
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_problem(
            config_path=args.config,
            out=args.out,
            compare_golden=args.compare_golden,
            shenfun_golden=args.shenfun_golden,
            write_golden=args.write_golden,
            device=args.device,
            steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            validate_only=args.validate_only,
        )
    except (ProblemSpecError, UnsupportedSpecError) as exc:
        print(f"spec rejected: {exc}", file=sys.stderr)
        return 1
    except SolverExecutionNotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
