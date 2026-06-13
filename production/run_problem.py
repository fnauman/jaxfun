"""Production runner entrypoint for jaxfun problem specs.

This module wires validation, device capture, metadata, and golden resolution.
Actual solver execution is intentionally explicit: until a solver factory is
registered for a spec, non-validate-only runs fail with a clear error instead of
claiming parity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .adapters import ProductionConfig, load_config
    from .compare_goldens import (
        compare_problem,
        compare_to_golden,
        load_golden,
        resolve_golden,
        scalar_hash,
    )
    from .device import capture_device_record
    from .oracles import ProductionOracleNotImplementedError, run_supported_spec
    from .problem_spec import ProblemSpecError, UnsupportedSpecError
except ImportError:  # pragma: no cover - direct script mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from production.adapters import ProductionConfig, load_config  # type: ignore
    from production.compare_goldens import (
        compare_problem,
        compare_to_golden,
        load_golden,
        resolve_golden,
        scalar_hash,
    )  # type: ignore
    from production.device import capture_device_record  # type: ignore
    from production.oracles import (
        ProductionOracleNotImplementedError,
        run_supported_spec,
    )  # type: ignore
    from production.problem_spec import (  # type: ignore
        ProblemSpecError,
        UnsupportedSpecError,
    )


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
    resolution_tier: str | None = None,
    validate_only: bool = False,
    capture_device: bool = True,
) -> dict[str, Any]:
    """Validate a config, write metadata, and eventually execute its solver."""

    config = load_config(config_path, resolution_tier=resolution_tier)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_record = (
        capture_device_record(device) if capture_device else {"capture_skipped": True}
    )
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
        resolution_tier=resolution_tier,
        validate_only=validate_only,
    )
    _write_json(out_dir / "metadata.json", metadata)

    if validate_only:
        return metadata

    solver_started_at = _utc_timestamp()
    solver_start = time.perf_counter()
    try:
        diagnostics = run_supported_spec(
            config.spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            device_record=device_record,
        )
    except ProductionOracleNotImplementedError as exc:
        metadata["timing"] = _solver_timing(solver_started_at, solver_start)
        _write_json(out_dir / "metadata.json", metadata)
        raise SolverExecutionNotImplementedError(
            f"{exc}; contract validation metadata was written, but no DNS "
            "or golden comparison was run"
        ) from exc
    except Exception as exc:
        metadata["timing"] = _solver_timing(solver_started_at, solver_start)
        metadata["execution"] = {
            "status": "failed",
            "solver_execution_wired": True,
            "execution_kind": _execution_kind(config.spec),
            "failure_reason": _exception_message(exc),
        }
        _write_json(out_dir / "metadata.json", metadata)
        raise
    metadata["timing"] = _solver_timing(solver_started_at, solver_start)

    _write_json(out_dir / "spec.json", config.spec)
    _write_diagnostics(out_dir / "diagnostics.jsonl", diagnostics)
    metadata["execution"] = {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": _execution_kind(config.spec),
    }
    metadata["validation_scope"] = _validation_scope_metadata(
        config.spec,
        diagnostics,
        device_record=device_record,
        compare_golden=compare_golden,
        steps=steps,
        resolution_tier=resolution_tier,
    )
    metadata["saturation_checks"] = _saturation_check_metadata(
        diagnostics, validation_scope=metadata["validation_scope"]
    )
    metadata["validation_floor"] = _validation_floor_metadata(
        diagnostics, validation_scope=metadata["validation_scope"]
    )
    metadata["diagnostics_path"] = str(out_dir / "diagnostics.jsonl")
    _write_json(out_dir / "metadata.json", metadata)
    _assert_validation_floor_checks(metadata)
    _assert_required_saturation_checks(metadata)
    checkpoint_path = out_dir / "checkpoints" / "checkpoints.h5"
    if checkpoint_path.exists():
        metadata["checkpoint_path"] = str(checkpoint_path)

    if compare_golden:
        result = _compare_diagnostics(
            config,
            diagnostics,
            explicit_golden=Path(shenfun_golden)
            if shenfun_golden is not None
            else None,
        )
        metadata["comparison_passed"] = result.passed
        metadata["comparisons"] = [item.to_dict() for item in result.comparisons]
        metadata["observables_compared"] = [item.key for item in result.comparisons]
        metadata["golden_resolution"].update(result.metadata)
        if not result.passed:
            _write_json(out_dir / "metadata.json", metadata)
            raise RuntimeError(f"golden comparison failed for {config.problem_id}")

    if write_golden:
        golden_path = _write_golden(
            out_dir / "golden" / "golden.json", config, diagnostics, device_record
        )
        metadata["written_golden"] = str(golden_path)

    _write_json(out_dir / "metadata.json", metadata)
    return metadata


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
    resolution_tier: str | None,
    validate_only: bool,
) -> dict[str, Any]:
    golden_resolution = _golden_resolution_metadata(config.problem_id, shenfun_golden)
    return {
        "schema_version": 1,
        "generated_at_utc": _utc_timestamp(),
        "problem_id": config.problem_id,
        "artifact_id": config.artifact_id,
        "config_path": str(config_path),
        "out_dir": str(out_dir),
        "spec_hash": config.spec["spec_hash"],
        "base_spec_hash": config.metadata.get(
            "base_spec_hash", config.spec["spec_hash"]
        ),
        "geometry": config.geometry,
        "physics": config.physics,
        "support_state": config.spec["support_state"],
        "expected_oracle": config.spec["expected_oracle"],
        "diagnostics": config.spec["diagnostics"],
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
            "resolution_tier": resolution_tier,
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


def _execution_kind(spec: dict[str, Any]) -> str:
    oracle_type = spec["expected_oracle"]["type"]
    if oracle_type in {
        "tc_hydro_saturation_ladder",
        "tc_mri_saturation_ladder",
        "mri_saturation_ladder",
        "gpu_generated_saturated_dns",
    }:
        return "dns-saturation"
    if oracle_type in {
        "circular_couette_dns_growth",
        "tc_mri_dns_growth",
        "pcf_hydro_dns_decay",
        "pcf_mri_dns_growth",
    }:
        return "dns-linear-window"
    if "linear" in oracle_type or oracle_type in {
        "circular_couette_base_flow",
        "local_ideal_mri",
        "plane_couette_laminar",
    }:
        return "linear-oracle"
    return "analytic-oracle"


def _validation_scope_metadata(
    spec: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    device_record: dict[str, Any],
    compare_golden: bool,
    steps: int | None,
    resolution_tier: str | None,
) -> dict[str, Any]:
    expected_oracle = spec.get("expected_oracle", {})
    fallback_rungs = expected_oracle.get("fallback_rungs", [])
    mode = device_record.get("mode")
    execution_kind = _execution_kind(spec)
    scalar_keys = sorted(diagnostics.get("scalars", {}).keys())
    bounded_smoke = _is_bounded_smoke_run(steps=steps, resolution_tier=resolution_tier)
    common = {
        "checked_observables": scalar_keys,
        "steps_override": steps,
        "resolution_tier": resolution_tier,
        "bounded_smoke": bounded_smoke,
    }

    if compare_golden:
        return {
            **common,
            "kind": "golden_comparison",
            "reason": "compared diagnostics against the resolved committed golden",
        }
    if (
        mode == "cpu_smoke"
        and execution_kind == "dns-saturation"
        and fallback_rungs == [3]
        and bounded_smoke
    ):
        return {
            **common,
            "kind": "cpu_smoke_finiteness_divergence_only",
            "reason": (
                "rung-3-only saturated run has no committed nonlinear-state "
                "golden; CPU smoke checks solver completion, finite diagnostics, "
                "and emitted divergence diagnostics, not production parity"
            ),
        }
    if mode == "cpu_smoke" and execution_kind == "dns-saturation" and bounded_smoke:
        return {
            **common,
            "kind": "cpu_smoke_fallback_oracle",
            "reason": (
                "CPU smoke for saturated run with analytic or linear-DNS fallback "
                "rungs available"
            ),
        }
    if execution_kind == "dns-saturation" and bounded_smoke:
        return {
            **common,
            "kind": "bounded_saturation_smoke",
            "reason": (
                "executed a step-limited or reduced-resolution saturation smoke "
                "run; generated artifacts are smoke diagnostics, not a full "
                "production saturation golden"
            ),
        }
    if execution_kind == "dns-saturation":
        return {
            **common,
            "kind": "generated_saturated_golden",
            "reason": "executed saturated production run and generated diagnostics",
        }
    return {
        **common,
        "kind": "oracle_execution",
        "reason": "executed configured analytic, linear, or DNS oracle path",
    }


def _is_bounded_smoke_run(*, steps: int | None, resolution_tier: str | None) -> bool:
    return steps is not None or resolution_tier in {"smoke", "start"}


def _saturation_check_metadata(
    diagnostics: dict[str, Any], *, validation_scope: dict[str, Any]
) -> dict[str, Any]:
    scalars = diagnostics.get("scalars", {})
    has_passed_key = "saturation_check_passed" in scalars
    raw_passed = scalars.get("saturation_check_passed")
    required = validation_scope.get("kind") == "generated_saturated_golden"
    return {
        "required": required,
        "present": has_passed_key,
        "passed": None if raw_passed is None else bool(raw_passed),
        "energy_growth_factor": scalars.get("energy_growth_factor"),
        "magnetic_energy_growth_factor": scalars.get("magnetic_energy_growth_factor"),
    }


_VALIDATION_FLOOR_SCOPES = {
    "bounded_saturation_smoke",
    "cpu_smoke_finiteness_divergence_only",
    "cpu_smoke_fallback_oracle",
}
_SMOKE_DIVERGENCE_LIMIT = 1.0e-2


def _validation_floor_metadata(
    diagnostics: dict[str, Any], *, validation_scope: dict[str, Any]
) -> dict[str, Any]:
    kind = validation_scope.get("kind")
    required = kind in _VALIDATION_FLOOR_SCOPES
    numeric_values = list(_iter_numeric_diagnostics(diagnostics))
    nonfinite = [key for key, value in numeric_values if not math.isfinite(value)]
    divergence_values = [
        (key, abs(value))
        for key, value in _iter_final_numeric_diagnostics(diagnostics)
        if _is_divergence_diagnostic(key)
    ]
    divergence_present = bool(divergence_values)
    max_divergence = max((value for _, value in divergence_values), default=None)
    divergence_failed = (
        not divergence_present
        or max_divergence is None
        or not math.isfinite(max_divergence)
        or max_divergence > _SMOKE_DIVERGENCE_LIMIT
    )
    passed = not required or (not nonfinite and not divergence_failed)
    return {
        "required": required,
        "passed": passed,
        "nonfinite_diagnostics": nonfinite,
        "divergence_diagnostics": [key for key, _ in divergence_values],
        "divergence_present": divergence_present,
        "max_divergence": max_divergence,
        "divergence_limit": _SMOKE_DIVERGENCE_LIMIT,
    }


def _iter_numeric_diagnostics(diagnostics: dict[str, Any]):
    for section, values in (
        ("scalars", diagnostics.get("scalars", {})),
        ("time_series", diagnostics.get("time_series", [])),
    ):
        if isinstance(values, dict):
            iterable = [("", values)]
        else:
            iterable = [(str(index), row) for index, row in enumerate(values)]
        for prefix, row in iterable:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    label = (
                        f"{section}.{prefix}.{key}" if prefix else f"{section}.{key}"
                    )
                    yield label, float(value)


def _iter_final_numeric_diagnostics(diagnostics: dict[str, Any]):
    rows = [("scalars", diagnostics.get("scalars", {}))]
    series = diagnostics.get("time_series") or []
    if series:
        rows.append(("time_series.final", series[-1]))
    for prefix, row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                yield f"{prefix}.{key}", float(value)


def _is_divergence_diagnostic(key: str) -> bool:
    name = key.rsplit(".", 1)[-1]
    return (
        name.startswith("divergence")
        or name.startswith("divu")
        or name.startswith("divb")
        or name.startswith("continuity")
    )


def _assert_validation_floor_checks(metadata: dict[str, Any]) -> None:
    checks = metadata.get("validation_floor", {})
    if not checks.get("required") or checks.get("passed") is True:
        return
    details = []
    nonfinite = checks.get("nonfinite_diagnostics") or []
    if nonfinite:
        details.append("nonfinite=" + ",".join(nonfinite))
    if not checks.get("divergence_present"):
        details.append("divergence=missing")
    else:
        details.append(
            f"max_divergence={checks.get('max_divergence')} "
            f"> {checks.get('divergence_limit')}"
        )
    message = "validation floor failed"
    if details:
        message = f"{message}: {'; '.join(details)}"
    metadata["execution"] = {
        **metadata.get("execution", {}),
        "status": "failed",
        "failure_reason": message,
    }
    _write_json(Path(metadata["out_dir"]) / "metadata.json", metadata)
    raise RuntimeError(message)


def _assert_required_saturation_checks(metadata: dict[str, Any]) -> None:
    checks = metadata.get("saturation_checks", {})
    if not checks.get("required"):
        return
    if checks.get("present") and checks.get("passed") is True:
        return

    details = []
    for key in ("energy_growth_factor", "magnetic_energy_growth_factor"):
        value = checks.get(key)
        if value is not None:
            details.append(f"{key}={value}")
    state = (
        "missing" if not checks.get("present") else str(checks.get("passed")).lower()
    )
    message = f"full saturation check failed: saturation_check_passed is {state}"
    if details:
        message = f"{message} ({', '.join(details)})"
    metadata["execution"] = {
        **metadata.get("execution", {}),
        "status": "failed",
        "failure_reason": message,
    }
    _write_json(Path(metadata["out_dir"]) / "metadata.json", metadata)
    raise RuntimeError(message)


def _exception_message(exc: Exception) -> str:
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _golden_resolution_metadata(
    problem_id: str, explicit_golden: Path | None
) -> dict[str, Any]:
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


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _solver_timing(started_at_utc: str, start: float) -> dict[str, Any]:
    return {
        "solver_started_at_utc": started_at_utc,
        "solver_finished_at_utc": _utc_timestamp(),
        "solver_wall_time_seconds": time.perf_counter() - start,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_diagnostics(path: Path, diagnostics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalars = diagnostics["scalars"]
    series = diagnostics.get("time_series")
    if series:
        rows = [dict(row) for row in series]
        rows[-1] = {**scalars, **rows[-1]}
    else:
        rows = [{"t": 0.0, **scalars}]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


def _compare_diagnostics(
    config: ProductionConfig,
    diagnostics: dict[str, Any],
    *,
    explicit_golden: Path | None,
):
    convention_metadata = {
        "canonical_axes": config.canonical_axes,
        "native_axes": config.native_axes,
        "axis_conventions": config.axis_conventions,
        "source_files": list(config.source_files),
    }
    if explicit_golden is not None:
        golden = load_golden(explicit_golden)
        return compare_to_golden(
            diagnostics["scalars"],
            golden,
            golden_path=explicit_golden,
            require_all_golden_scalars=True,
            convention_metadata=convention_metadata,
        )
    return compare_problem(
        config.problem_id,
        diagnostics["scalars"],
        require_all_golden_scalars=True,
        convention_metadata=convention_metadata,
    )


def _write_golden(
    path: Path,
    config: ProductionConfig,
    diagnostics: dict[str, Any],
    device_record: dict[str, Any],
) -> Path:
    diagnostics_ready = _json_ready(diagnostics)
    scalars = diagnostics_ready["scalars"]
    data = {
        "schema_version": 1,
        "artifact_id": config.artifact_id,
        "problem_id": config.problem_id,
        "spec_hash": config.spec["spec_hash"],
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "environment": {
            "interpreter": sys.executable,
            "jax": device_record,
        },
        "git": {},
        "source_anchors": config.spec["expected_oracle"].get("source_anchors", []),
        "tolerance_model": config.spec["tolerance_model"],
        "diagnostics": diagnostics_ready,
        "comparison_fields": {
            "scalars_sha256": scalar_hash(scalars),
        },
    }
    _write_json(path, data)
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
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
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"]
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument(
        "--resolution-tier",
        choices=["smoke", "start", "production"],
        help="Materialize a nested resolution tier before execution.",
    )
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
            resolution_tier=args.resolution_tier,
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
