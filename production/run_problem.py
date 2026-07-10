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
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from .adapters import ProductionConfig, load_config
    from .compare_goldens import (
        assert_golden_not_quarantined,
        compare_problem,
        compare_to_golden,
        load_golden,
        resolve_golden,
        validate_golden,
        scalar_hash,
    )
    from .device import capture_device_record, configure_production_dtype
    from .oracles import (
        ProductionOracleNotImplementedError,
        load_resume_checkpoint,
        run_supported_spec,
        validate_resume_checkpoint,
    )
    from .problem_spec import ProblemSpecError, UnsupportedSpecError, load_spec
    from .provenance import ReleaseCleanlinessError
    from .quench import QuenchError, burn_in_horizon, validate_quench
except ImportError:  # pragma: no cover - direct script mode
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from production.adapters import ProductionConfig, load_config  # type: ignore
    from production.compare_goldens import (
        assert_golden_not_quarantined,
        compare_problem,
        compare_to_golden,
        load_golden,
        resolve_golden,
        validate_golden,
        scalar_hash,
    )  # type: ignore
    from production.device import (  # type: ignore
        capture_device_record,
        configure_production_dtype,
    )
    from production.oracles import (
        ProductionOracleNotImplementedError,
        load_resume_checkpoint,
        run_supported_spec,
        validate_resume_checkpoint,
    )  # type: ignore
    from production.problem_spec import (  # type: ignore
        ProblemSpecError,
        UnsupportedSpecError,
    )
    from production.provenance import ReleaseCleanlinessError  # type: ignore
    from production.problem_spec import load_spec  # type: ignore
    from production.quench import (  # type: ignore
        QuenchError,
        burn_in_horizon,
        validate_quench,
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
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    resolution_tier: str | None = None,
    validate_only: bool = False,
    capture_device: bool = True,
    resume: str | Path | None = None,
    require_clean: bool = False,
    allow_dirty: bool = False,
    wandb: bool = False,
    wandb_project: str | None = None,
    wandb_offline: bool = False,
    quench_from: str | Path | None = None,
    burn_in_steps: int = 0,
) -> dict[str, Any]:
    """Validate a config, write metadata, and eventually execute its solver."""

    config = load_config(config_path, resolution_tier=resolution_tier)
    resume_record, quench_metadata = _resolve_resume_or_quench(
        config, resume=resume, quench_from=quench_from, burn_in_steps=burn_in_steps
    )
    quench_mode = quench_metadata is not None
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    release_gate = _enforce_release_gate(
        out_dir, require_clean=require_clean, allow_dirty=allow_dirty
    )
    compilation_cache, restore_compilation_cache = _configure_compilation_cache(out_dir)
    metadata: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        effective_diagnostics_every = _effective_diagnostics_every(
            config.spec,
            diagnostics_every=diagnostics_every,
            steps=steps,
            resolution_tier=resolution_tier,
        )

        device_record = (
            capture_device_record(device) if capture_device else {"capture_skipped": True}
        )
        _assert_precision_matches_spec(config.spec, device_record)
        if resume_record is not None:
            validate_resume_checkpoint(
                resume_record, config.spec, device_record, quench=quench_mode
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
            snapshot_every=snapshot_every,
            diagnostics_every=effective_diagnostics_every,
            resolution_tier=resolution_tier,
            validate_only=validate_only,
            resume=Path(resume) if resume is not None else None,
            compilation_cache=compilation_cache,
        )
        if release_gate is not None and isinstance(metadata.get("provenance"), dict):
            metadata["provenance"]["release_gate"] = release_gate
        if quench_metadata is not None:
            metadata["quench"] = quench_metadata
        _write_json(out_dir / "metadata.json", metadata)

        if validate_only:
            return metadata

        solver_started_at = _utc_timestamp()
        solver_start = time.perf_counter()
        solver_steps = _executed_solver_steps(
            config.spec, steps=steps, resume_record=resume_record
        )
        try:
            diagnostics = run_supported_spec(
                config.spec,
                steps=steps,
                out_dir=out_dir,
                checkpoint_every=checkpoint_every,
                snapshot_every=snapshot_every,
                diagnostics_every=effective_diagnostics_every,
                device_record=device_record,
                resume_checkpoint=resume_record,
                quench=quench_mode,
            )
        except ProductionOracleNotImplementedError as exc:
            metadata["timing"] = _solver_timing(
                solver_started_at, solver_start, solver_steps=solver_steps
            )
            _write_json(out_dir / "metadata.json", metadata)
            raise SolverExecutionNotImplementedError(
                f"{exc}; contract validation metadata was written, but no DNS "
                "or golden comparison was run"
            ) from exc
        except Exception as exc:
            metadata["timing"] = _solver_timing(
                solver_started_at, solver_start, solver_steps=solver_steps
            )
            metadata["execution"] = {
                # FJ-06: distinguish nan_inf / blew_up / walltime from a generic failure.
                "status": _operational_status(exc),
                "solver_execution_wired": True,
                "execution_kind": _execution_kind(config.spec),
                "failure_reason": _exception_message(exc),
            }
            _write_json(out_dir / "metadata.json", metadata)
            raise
        metadata["timing"] = _solver_timing(
            solver_started_at, solver_start, solver_steps=solver_steps
        )

        _write_json(out_dir / "spec.json", config.spec)
        _write_diagnostics(
            out_dir / "diagnostics.jsonl",
            diagnostics,
            append=resume_record is not None and (out_dir / "diagnostics.jsonl").exists(),
        )
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
        metadata["classification"] = _classification_metadata(diagnostics)
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
        snapshot_path = out_dir / "snapshots" / "snapshots.h5"
        if snapshot_path.exists():
            metadata["snapshot_path"] = str(snapshot_path)
            xdmf_path = snapshot_path.with_suffix(".xdmf")
            if xdmf_path.exists():
                metadata["snapshot_xdmf_path"] = str(xdmf_path)

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
                out_dir / "golden" / "golden.json",
                config,
                diagnostics,
                device_record,
                metadata=metadata,
            )
            metadata["written_golden"] = str(golden_path)

        _write_json(out_dir / "metadata.json", metadata)
        return metadata
    finally:
        restore_compilation_cache()
        # FJ-07: mirror to W&B exactly once on every path that executed a solver --
        # success, early stop, comparison failure, or crash -- so the failed run and
        # its operational status are recorded. `diagnostics` is {} if the solver never
        # produced any (the sink degrades gracefully); skipped for validate-only.
        if wandb and not validate_only and metadata:
            _mirror_to_wandb(
                config,
                diagnostics,
                metadata,
                enabled=wandb,
                project=wandb_project,
                offline=wandb_offline,
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
    snapshot_every: int | None,
    diagnostics_every: int | None,
    resolution_tier: str | None,
    validate_only: bool,
    resume: Path | None,
    compilation_cache: dict[str, Any],
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
        "numerics_contract_version": config.spec.get("numerics_contract_version"),
        "provenance": _capture_provenance_safe(),
        "resolved_physics": _resolved_physics_metadata(
            config.spec, precision=device_record.get("production_run_dtype")
        ),
        "integrator": _integrator_provenance(config.spec),
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
        "compilation_cache": compilation_cache,
        "run_options": {
            "requested_device": requested_device,
            "steps_override": steps,
            "checkpoint_every": checkpoint_every,
            "snapshot_every": snapshot_every,
            "diagnostics_every": diagnostics_every,
            "resolution_tier": resolution_tier,
            "resume": None if resume is None else str(resume),
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


def _effective_diagnostics_every(
    spec: dict[str, Any],
    *,
    diagnostics_every: int | None,
    steps: int | None,
    resolution_tier: str | None,
) -> int | None:
    if diagnostics_every is not None:
        return diagnostics_every
    if _execution_kind(spec) != "dns-saturation":
        return None
    if _is_bounded_smoke_run(steps=steps, resolution_tier=resolution_tier):
        return None
    total_steps = _steps_from_spec_metadata(spec, steps=steps)
    return max(1, min(100, max(1, total_steps // 16)))


def _steps_from_spec_metadata(spec: dict[str, Any], *, steps: int | None) -> int:
    if steps is not None:
        return int(steps)
    time_spec = spec["time"]
    return int(round(float(time_spec["final_time"]) / float(time_spec["dt"])))


def _saturation_check_metadata(
    diagnostics: dict[str, Any], *, validation_scope: dict[str, Any]
) -> dict[str, Any]:
    scalars = diagnostics.get("scalars", {})
    has_passed_key = "saturation_check_passed" in scalars
    raw_passed = scalars.get("saturation_check_passed")
    required = validation_scope.get("kind") == "generated_saturated_golden"
    type_valid = isinstance(raw_passed, bool)
    return {
        "required": required,
        "present": has_passed_key,
        "type_valid": type_valid if has_passed_key else None,
        "passed": raw_passed if type_valid else None,
        "energy_growth_factor": scalars.get("energy_growth_factor"),
        "magnetic_energy_growth_factor": scalars.get("magnetic_energy_growth_factor"),
        "stationarity_check_passed": scalars.get("stationarity_check_passed"),
        "stationarity_relative_change": scalars.get("stationarity_relative_change"),
    }


_VALIDATION_FLOOR_SCOPES = {
    "generated_saturated_golden",
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


_DIVERGENCE_DIAGNOSTIC_RE = re.compile(
    r"^(?:"
    r"divergence(?:[_-](?:u|b))?(?:[_-]?(?:l2|linf|norm|rms|max))?"
    r"|div(?:u|b)(?:[_-]?(?:l2|linf|norm|rms|max))?"
    r"|div(?:[_-]?(?:l2|linf|norm|rms|max))"
    r"|continuity(?:[_-]?(?:residual|l2|linf|norm|rms|max))?"
    r")$"
)


def _is_divergence_diagnostic(key: str) -> bool:
    name = key.rsplit(".", 1)[-1].lower()
    return _DIVERGENCE_DIAGNOSTIC_RE.match(name) is not None


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
    if (
        checks.get("present")
        and checks.get("type_valid") is True
        and checks.get("passed") is True
        and checks.get("stationarity_check_passed") is True
    ):
        return

    details = []
    for key in (
        "energy_growth_factor",
        "magnetic_energy_growth_factor",
        "stationarity_relative_change",
    ):
        value = checks.get(key)
        if value is not None:
            details.append(f"{key}={value}")
    if checks.get("stationarity_check_passed") is not True:
        details.append(
            "stationarity_check_passed="
            f"{checks.get('stationarity_check_passed')}"
        )
    if not checks.get("present"):
        state = "missing"
    elif checks.get("type_valid") is not True:
        state = "non-boolean"
    else:
        state = str(checks.get("passed")).lower()
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


def _solver_timing(
    started_at_utc: str, start: float, *, solver_steps: int | None = None
) -> dict[str, Any]:
    elapsed = time.perf_counter() - start
    timing: dict[str, Any] = {
        "solver_started_at_utc": started_at_utc,
        "solver_finished_at_utc": _utc_timestamp(),
        "solver_wall_time_seconds": elapsed,
    }
    if solver_steps is not None:
        steps = max(0, int(solver_steps))
        timing["solver_steps"] = steps
        if steps > 0 and elapsed > 0.0:
            timing["seconds_per_step"] = elapsed / steps
            timing["ms_per_step"] = 1000.0 * elapsed / steps
            timing["steps_per_second"] = steps / elapsed
    return timing


def _executed_solver_steps(
    spec: dict[str, Any], *, steps: int | None, resume_record: Any | None
) -> int | None:
    if _execution_kind(spec) not in {"dns-saturation", "dns-linear-window"}:
        return None
    target_steps = _steps_from_spec_metadata(spec, steps=steps)
    start_step = (
        int(getattr(resume_record, "tstep", 0))
        if resume_record is not None
        else 0
    )
    return max(0, target_steps - start_step)


def _configure_compilation_cache(out_dir: Path):
    configured = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    cache_dir = (
        Path(configured) if configured else out_dir.parent / "_jax_compilation_cache"
    )
    record: dict[str, Any] = {
        "requested": True,
        "source": "env" if configured else "run_parent_default",
        "path": str(cache_dir),
        "enabled": False,
        "restores_process_config": True,
    }
    restore = lambda: None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        import jax

        restore_state = _capture_compilation_cache_config(jax)

        def restore() -> None:
            _restore_compilation_cache_config(jax, restore_state)

        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        try:
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        except Exception as exc:  # pragma: no cover - depends on JAX version
            record["min_compile_time_config_error"] = _exception_message(exc)
        record["enabled"] = True
    except Exception as exc:  # pragma: no cover - cache support is best effort
        record["error"] = _exception_message(exc)
    return record, restore


def _capture_compilation_cache_config(jax_module: Any) -> dict[str, Any]:
    return {
        "jax_compilation_cache_dir": getattr(
            jax_module.config, "jax_compilation_cache_dir", None
        ),
        "jax_persistent_cache_min_compile_time_secs": getattr(
            jax_module.config, "jax_persistent_cache_min_compile_time_secs", None
        ),
    }


def _restore_compilation_cache_config(
    jax_module: Any, restore_state: dict[str, Any]
) -> None:
    for key, value in restore_state.items():
        try:
            jax_module.config.update(key, value)
        except Exception:
            pass


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(data), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_diagnostics(
    path: Path, diagnostics: dict[str, Any], *, append: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalars = diagnostics["scalars"]
    series = diagnostics.get("time_series")
    if series:
        rows = [dict(row) for row in series]
        rows[-1] = {**scalars, **rows[-1]}
    else:
        rows = [{"t": 0.0, **scalars}]
    mode = "a" if append else "w"
    if append and len(rows) > 1:
        rows = rows[1:]
    with path.open(mode, encoding="utf-8") as fh:
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
        golden = validate_golden(explicit_golden, spec=config.spec)
        assert_golden_not_quarantined(golden, config.problem_id)
        return compare_to_golden(
            diagnostics["scalars"],
            golden,
            golden_path=explicit_golden,
            require_all_golden_scalars=True,
            convention_metadata=convention_metadata,
        )
    resolution = resolve_golden(config.problem_id)
    assert_golden_not_quarantined(
        load_golden(resolution.golden_path), config.problem_id
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
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    diagnostics_ready = _json_ready(diagnostics)
    scalars = diagnostics_ready["scalars"]
    _assert_golden_divergence_ok(config.problem_id, scalars)
    tolerance_model = _golden_tolerance_model(config.spec["tolerance_model"], scalars)
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
        "git": _capture_provenance_safe(),
        "source_anchors": config.spec["expected_oracle"].get("source_anchors", []),
        "tolerance_model": tolerance_model,
        "diagnostics": diagnostics_ready,
        "generation": {
            "run_options": (metadata or {}).get("run_options", {}),
            "validation_scope": (metadata or {}).get("validation_scope", {}),
        },
        "comparison_fields": {
            "scalars_sha256": scalar_hash(scalars),
            "tolerance_model_sha256": scalar_hash(tolerance_model),
        },
    }
    _write_json(path, data)
    return path


def _resolve_resume_or_quench(
    config: ProductionConfig,
    *,
    resume: str | Path | None,
    quench_from: str | Path | None,
    burn_in_steps: int,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Return ``(resume_record, quench_metadata)`` for resume-exact or quench (FJ-05)."""

    if quench_from is not None and resume is not None:
        raise ProblemSpecError("--resume and --quench are mutually exclusive")
    if quench_from is not None:
        source = Path(quench_from)
        record = load_resume_checkpoint(source)
        parent_spec = load_spec(source / "spec.json")
        diff = validate_quench(parent_spec, config.spec)  # raises on illegal change
        tstep0 = int(getattr(record, "tstep", 0))
        meta = {
            "mode": "quench",
            "parent_run_dir": str(source),
            "parent_spec_hash": parent_spec.get("spec_hash"),
            "child_spec_hash": config.spec["spec_hash"],
            "mutable_diff": {k: list(v) for k, v in diff["changed"].items()},
            **burn_in_horizon(tstep0=tstep0, burn_in_steps=burn_in_steps),
        }
        return record, meta
    if resume is not None:
        return load_resume_checkpoint(resume), None
    return None, None


def _enforce_release_gate(
    out_dir: Path, *, require_clean: bool, allow_dirty: bool
) -> dict[str, Any] | None:
    """FJ-13: enforce the clean-worktree gate when a production launch requests it."""

    if not require_clean:
        return None
    try:
        from .provenance import assert_release_clean
    except ImportError:  # pragma: no cover - direct script mode
        from production.provenance import assert_release_clean  # type: ignore
    prov = assert_release_clean(out_dir, allow_dirty=allow_dirty)
    return prov.get("release_gate")


_INTEGRATOR_FORMAL_ORDER = {
    "CNAB2": 2,
    "IMEXRK222": 2,
    "IMEXRK443": 3,
    "analytic": None,
    "linear_eigenproblem": None,
}

# The wired primitive PCF MHD/MRI saturation solver advances the coupled block with
# hard-coded CNAB2 regardless of the requested-but-inert time.integrator label.
_PRIMITIVE_SATURATION_ORACLES = {
    "gpu_generated_saturated_dns",
    "mri_saturation_ladder",
}


def _integrator_provenance(spec: dict[str, Any]) -> dict[str, Any]:
    """FJ-08: record the integrator actually used, its formal order, and dt."""

    requested = spec.get("time", {}).get("integrator")
    oracle = spec.get("expected_oracle", {}).get("type")
    representation = spec.get("representation")
    actual = requested
    if representation == "vector_potential":
        # curl family dispatches to PlaneCouetteMRIShearpyJax, which runs IMEXRK222.
        actual = "IMEXRK222"
    elif (
        oracle in _PRIMITIVE_SATURATION_ORACLES
        and spec.get("physics") in {"mhd", "mri"}
    ):
        actual = "CNAB2"  # hard-coded in PCFMRIDNSJax
    return {
        "requested": requested,
        "actual": actual,
        "formal_order": _INTEGRATOR_FORMAL_ORDER.get(actual),
        "dt": spec.get("time", {}).get("dt"),
        "order_regression_test": (
            "tests/couette/test_pcf_mri_cnab2_order_jax.py"
            if actual == "CNAB2"
            else None
        ),
    }


def _mirror_to_wandb(
    config: ProductionConfig,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
    *,
    enabled: bool,
    project: str | None,
    offline: bool,
) -> None:
    """FJ-07: mirror the cadence stream + run summary to W&B (host-side, post-solve).

    A no-op when disabled or when wandb is uninstalled. Never invoked inside JAX
    tracing -- it reads the already-materialized diagnostics.
    """

    if not enabled:
        return
    try:
        from .wandb_sink import WandbSink
    except ImportError:  # pragma: no cover - direct script mode
        from production.wandb_sink import WandbSink  # type: ignore

    resolved = metadata.get("resolved_physics") or {}
    sink = WandbSink(
        enabled=True,
        project=project or "jaxfun-production",
        group=f"{config.geometry}/{config.physics}",
        run_id=f"{config.problem_id}-{config.spec['spec_hash'][:12]}",
        config={
            "problem_id": config.problem_id,
            "spec_hash": config.spec["spec_hash"],
            **{k: resolved.get(k) for k in ("Re_h", "Rm_h", "Pm", "B0", "nu", "eta")},
        },
        mode="offline" if offline else None,
    )
    with sink:
        for row in diagnostics.get("time_series") or []:
            sink.log_cadence(row)
        classification = metadata.get("classification") or {}
        execution = metadata.get("execution") or {}
        scalars = diagnostics.get("scalars") or {}
        summary = {
            "operational_status": execution.get("status"),
            "scientific_class": classification.get("scientific_class"),
            "final_time": (diagnostics.get("time_series") or [{}])[-1].get("t"),
        }
        summary.update({k: v for k, v in scalars.items() if _is_number(v)})
        sink.log_summary(summary)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _assert_precision_matches_spec(
    spec: dict[str, Any], device_record: dict[str, Any]
) -> None:
    """FJ-07/FJ-08: a spec's declared precision must match the active run dtype.

    A sweep can set ``spec['precision']`` (float32/float64); the process dtype is set
    from ``JAXFUN_PRODUCTION_DTYPE``. If they disagree the run would be archived under a
    mislabeled precision, so fail loudly instead of silently downgrading.
    """

    declared = spec.get("precision")
    if declared is None:
        return
    active = device_record.get("production_run_dtype")
    if active is None:
        return
    alias = {"single": "float32", "fp32": "float32", "double": "float64", "fp64": "float64"}
    declared_norm = alias.get(str(declared).lower(), str(declared).lower())
    if declared_norm != str(active).lower():
        raise ProblemSpecError(
            f"spec precision {declared!r} does not match the active production dtype "
            f"{active!r}; set JAXFUN_PRODUCTION_DTYPE={declared_norm} before running "
            "so the materialized precision is actually honored."
        )


def _operational_status(exc: BaseException) -> str:
    try:
        from .classify import operational_status_from_exception
    except ImportError:  # pragma: no cover - direct script mode
        from production.classify import operational_status_from_exception  # type: ignore
    return operational_status_from_exception(exc).value


def _classification_metadata(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """FJ-06: separate scientific class from operational status."""

    try:
        from .classify import classify_scientific
    except ImportError:  # pragma: no cover - direct script mode
        from production.classify import classify_scientific  # type: ignore

    series = diagnostics.get("time_series") or []
    scalars = diagnostics.get("scalars") or {}
    stationary = scalars.get("stationarity_check_passed")
    # Noise floor: a tiny fraction of the peak fluctuation energy in the series.
    peak = 0.0
    for row in series:
        for key in ("mag_energy_fluct", "total_energy"):
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                peak = max(peak, abs(float(value)))
    noise_floor = 1.0e-10 * peak
    return classify_scientific(
        series,
        noise_floor=noise_floor,
        stationary=bool(stationary) if stationary is not None else None,
    )


def _resolved_physics_metadata(
    spec: dict[str, Any], precision: str | None = None
) -> dict[str, Any] | None:
    """FJ-00: record the single resolved-physics object in the manifest.

    ``precision`` should be the *actual* run dtype (from the device record) so the
    canonical physics block agrees with ``metadata.device.production_run_dtype``.
    """

    if spec.get("geometry") not in {"pcf", "channel"}:
        return None
    groups = spec.get("nondimensional_groups", {})
    if groups.get("nu") is None and groups.get("Re") is None:
        return None
    try:
        from .physics import resolve_physics
    except ImportError:  # pragma: no cover - direct script mode
        from production.physics import resolve_physics  # type: ignore
    try:
        if precision is not None:
            return resolve_physics(spec, precision=str(precision)).to_metadata()
        return resolve_physics(spec).to_metadata()
    except Exception:  # pragma: no cover - already validated upstream
        return None


def _capture_provenance_safe() -> dict[str, Any]:
    try:
        from .provenance import capture_provenance
    except ImportError:  # pragma: no cover - direct script mode
        from production.provenance import capture_provenance  # type: ignore
    try:
        return capture_provenance()
    except Exception:  # pragma: no cover - provenance must never break a run
        return {}


def _assert_golden_divergence_ok(problem_id: str, scalars: dict[str, Any]) -> None:
    """FJ-03: refuse to promote a golden whose solenoidality violates the guard.

    A physically invalid nonlinear state (``div_u``/``div_b`` above the runtime
    divergence guard) must never become a committed reference.
    """

    try:
        from .oracles import _DIVERGENCE_GUARD_LIMIT, _is_divergence_key
    except ImportError:  # pragma: no cover - direct script mode
        from production.oracles import (  # type: ignore
            _DIVERGENCE_GUARD_LIMIT,
            _is_divergence_key,
        )

    offenders = []
    for key, value in scalars.items():
        if not _is_divergence_key(str(key)):
            continue
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(magnitude) or magnitude > _DIVERGENCE_GUARD_LIMIT:
            offenders.append(f"{key}={value}")
    if offenders:
        raise RuntimeError(
            f"refusing to write golden for {problem_id!r}: solenoidality guard "
            f"violated ({', '.join(offenders)} > {_DIVERGENCE_GUARD_LIMIT:g}). "
            "This state is not a valid nonlinear reference (FJ-03)."
        )


def _golden_tolerance_model(
    tolerance_model: dict[str, Any], scalars: dict[str, Any]
) -> dict[str, Any]:
    model = json.loads(json.dumps(tolerance_model))
    scalar_tolerances = model.setdefault("scalars", {})
    stationarity_tol = scalars.get("stationarity_relative_tolerance", 5.0e-2)
    for key, value in scalars.items():
        if key in scalar_tolerances or not key.startswith("stationarity_"):
            continue
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if key == "stationarity_relative_change":
            scalar_tolerances[key] = float(stationarity_tol)
        elif key in {"stationarity_previous_mean", "stationarity_current_mean"}:
            scalar_tolerances[key] = max(abs(float(value)) * float(stationarity_tol), 1.0e-30)
        elif key == "stationarity_window_samples":
            scalar_tolerances[key] = 1.0
        else:
            scalar_tolerances[key] = 0.0
    return model


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
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--compare-golden", action="store_true")
    parser.add_argument("--shenfun-golden")
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"]
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--snapshot-every", type=int)
    parser.add_argument("--diagnostics-every", type=int)
    parser.add_argument(
        "--resolution-tier",
        choices=["smoke", "start", "production"],
        help="Materialize a nested resolution tier before execution.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", help="Resume from a prior run directory.")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="FJ-13: refuse to run from a dirty/untagged/unpushed worktree.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a discovery-only run from a dirty tree; archives the diff.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="FJ-07: mirror diagnostics to Weights & Biases (optional; local files "
        "stay the source of truth). No-op if wandb is uninstalled.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Use WANDB_MODE=offline; sync later with `wandb sync`.",
    )
    parser.add_argument(
        "--quench",
        dest="quench_from",
        default=None,
        help="FJ-05: continue from a parent run dir, changing only nu/eta (Re/Rm).",
    )
    parser.add_argument("--burn-in-steps", type=int, default=0)
    return parser


def _peek_spec_precision(config_path: Path) -> str | None:
    """Read a materialized spec's `precision` field before configuring JAX."""

    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    precision = data.get("precision")
    return str(precision) if precision is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    resume_dir = Path(args.resume) if args.resume else None
    config_path = Path(args.config) if args.config else None
    out_dir = Path(args.out) if args.out else None
    if resume_dir is not None:
        config_path = config_path or (resume_dir / "spec.json")
        out_dir = out_dir or resume_dir
    if config_path is None or out_dir is None:
        parser.error("--config and --out are required unless --resume is used")

    env_keys = ("JAXFUN_PRODUCTION_DTYPE", "JAXFUN_ENABLE_X64", "JAX_ENABLE_X64")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    previous_x64 = None
    jax_module = None
    try:
        import jax as jax_module  # type: ignore[no-redef]

        previous_x64 = bool(jax_module.config.read("jax_enable_x64"))
    except Exception:
        jax_module = None

    # FJ-07/FJ-08: honor a materialized spec's `precision` field by driving the process
    # dtype from it (a swept float64 spec then actually runs at float64). An explicit
    # JAXFUN_PRODUCTION_DTYPE in the environment still wins.
    spec_precision = _peek_spec_precision(config_path)
    dtype_override = (
        spec_precision if os.environ.get("JAXFUN_PRODUCTION_DTYPE") is None else None
    )
    configure_production_dtype(dtype=dtype_override, apply_to_process=True)
    try:
        try:
            run_problem(
                config_path=config_path,
                out=out_dir,
                compare_golden=args.compare_golden,
                shenfun_golden=args.shenfun_golden,
                write_golden=args.write_golden,
                device=args.device,
                steps=args.steps,
                checkpoint_every=args.checkpoint_every,
                snapshot_every=args.snapshot_every,
                diagnostics_every=args.diagnostics_every,
                resolution_tier=args.resolution_tier,
                validate_only=args.validate_only,
                resume=resume_dir,
                # FJ-13: promoting a committed golden from the CLI is a production
                # action -> require a clean/pushed/immutable-ref tree (unless the run
                # is an explicit --allow-dirty discovery run).
                require_clean=args.require_clean or args.write_golden,
                allow_dirty=args.allow_dirty,
                wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_offline=args.wandb_offline,
                quench_from=args.quench_from,
                burn_in_steps=args.burn_in_steps,
            )
        except (ProblemSpecError, UnsupportedSpecError) as exc:
            print(f"spec rejected: {exc}", file=sys.stderr)
            return 1
        except QuenchError as exc:
            print(f"quench rejected: {exc}", file=sys.stderr)
            return 1
        except ReleaseCleanlinessError as exc:
            print(f"release gate: {exc}", file=sys.stderr)
            return 3
        except SolverExecutionNotImplementedError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if jax_module is not None and previous_x64 is not None:
            jax_module.config.update("jax_enable_x64", previous_x64)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
