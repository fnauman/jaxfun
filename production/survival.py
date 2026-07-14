"""First-passage survival analysis for transition-regime quench ensembles.

The solver remains responsible only for emitting authenticated quench provenance
and canonical diagnostic cadence. This module converts those artifacts into
dwell-qualified decay observations on a quench-age clock, then builds grouped
Kaplan-Meier estimates with log-log Greenwood intervals and optional parent-
cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from production.observables import energy_convention_for_spec

SURVIVAL_SCHEMA_VERSION = 1
SURVIVAL_GROUPING_KEYS = (
    "problem_id",
    "geometry",
    "physics",
    "representation",
    "magnetic_bc",
    "numerics_contract_version",
    "child_spec_hash",
    "parent_spec_hash",
    "mutable_diff_hash",
    "resolution_tier",
    "precision",
    "integrator",
    "dt",
    "energy_convention",
    "energy_key",
    "decay_threshold",
    "dwell_time",
    "analysis_start_age",
)


class SurvivalAnalysisError(ValueError):
    """Raised when a survival input or ensemble contract is malformed."""


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SurvivalAnalysisError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SurvivalAnalysisError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise SurvivalAnalysisError(f"{label} must be a finite number")
    return number


def _validate_decay_controls(
    *,
    threshold: Any,
    dwell_time: Any,
    analysis_start_age: Any,
) -> tuple[float, float, float]:
    threshold_value = _finite_number(threshold, "decay threshold")
    dwell_value = _finite_number(dwell_time, "dwell time")
    start_value = _finite_number(analysis_start_age, "analysis start age")
    if threshold_value < 0.0:
        raise SurvivalAnalysisError("decay threshold must be nonnegative")
    if dwell_value < 0.0:
        raise SurvivalAnalysisError("dwell time must be nonnegative")
    if start_value < 0.0:
        raise SurvivalAnalysisError("analysis start age must be nonnegative")
    return threshold_value, dwell_value, start_value


def _canonical_group(group: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(group, dict):
        raise SurvivalAnalysisError("survival group must be a JSON object")
    required = set(SURVIVAL_GROUPING_KEYS)
    supplied = set(group)
    missing = sorted(required - supplied)
    extra = sorted(supplied - required)
    if missing or extra:
        raise SurvivalAnalysisError(
            "survival group must contain the complete canonical key set; "
            f"missing={missing}, extra={extra}"
        )
    return {key: group[key] for key in SURVIVAL_GROUPING_KEYS}


def _canonical_quench_series(
    rows: list[dict[str, Any]],
    *,
    parent_time: float,
    energy_key: str,
    analysis_start_age: float,
) -> list[tuple[float, float]]:
    if not isinstance(rows, list) or not rows:
        raise SurvivalAnalysisError("diagnostic series must be a non-empty list")
    if not isinstance(energy_key, str) or not energy_key:
        raise SurvivalAnalysisError("energy_key must be a non-empty string")

    canonical: list[tuple[float, float]] = []
    last_time: float | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SurvivalAnalysisError(f"diagnostic row {index} must be an object")
        if "t" not in row or energy_key not in row:
            raise SurvivalAnalysisError(
                f"diagnostic row {index} requires t and {energy_key!r}"
            )
        absolute_time = _finite_number(row["t"], f"diagnostic row {index} time")
        energy = _finite_number(row[energy_key], f"diagnostic row {index} {energy_key}")
        if energy < 0.0:
            raise SurvivalAnalysisError(
                f"diagnostic row {index} {energy_key} must be nonnegative"
            )
        if absolute_time < parent_time - 1.0e-12:
            raise SurvivalAnalysisError(
                "diagnostic time precedes the authenticated parent checkpoint"
            )
        if last_time is not None and absolute_time <= last_time:
            raise SurvivalAnalysisError("diagnostic times must be strictly increasing")
        last_time = absolute_time
        age = max(0.0, absolute_time - parent_time)
        if age >= analysis_start_age - 1.0e-12:
            canonical.append((age, energy))

    if not canonical:
        raise SurvivalAnalysisError(
            "no diagnostic samples remain after the declared analysis start age"
        )
    return canonical


def quench_first_passage(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    parent_cluster_id: str,
    group: dict[str, Any],
    parent_time: float,
    energy_key: str,
    threshold: float,
    dwell_time: float,
    analysis_start_age: float = 0.0,
    operational_status: str = "completed",
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical dwell-qualified decay/right-censoring observation.

    The event time is the first sampled below-threshold quench age whose state
    later remains below threshold for at least ``dwell_time``. The qualification
    age is stored separately, so the dwell rule does not shift the reported
    first-passage lifetime. If no event qualifies, the run is right-censored at
    its last canonical diagnostic age.
    """

    if not isinstance(run_id, str) or not run_id:
        raise SurvivalAnalysisError("run_id must be a non-empty string")
    if not isinstance(parent_cluster_id, str) or not parent_cluster_id:
        raise SurvivalAnalysisError("parent_cluster_id must be a non-empty string")
    group_payload = _canonical_group(group)
    parent = _finite_number(parent_time, "parent checkpoint time")
    threshold_value, dwell_value, start_value = _validate_decay_controls(
        threshold=threshold,
        dwell_time=dwell_time,
        analysis_start_age=analysis_start_age,
    )
    if group_payload["energy_key"] != energy_key:
        raise SurvivalAnalysisError(
            "survival group energy_key does not match the analysis controls"
        )
    group_controls = {
        "decay_threshold": threshold_value,
        "dwell_time": dwell_value,
        "analysis_start_age": start_value,
    }
    for key, expected in group_controls.items():
        if _finite_number(group_payload[key], f"group {key}") != expected:
            raise SurvivalAnalysisError(
                f"survival group {key} does not match the analysis controls"
            )
    canonical = _canonical_quench_series(
        rows,
        parent_time=parent,
        energy_key=energy_key,
        analysis_start_age=start_value,
    )

    below_start: float | None = None
    event_age: float | None = None
    qualification_age: float | None = None
    for age, energy in canonical:
        if energy <= threshold_value:
            if below_start is None:
                below_start = age
            if age - below_start >= dwell_value - 1.0e-12:
                event_age = below_start
                qualification_age = age
                break
        else:
            below_start = None

    last_observed_age = canonical[-1][0]
    observed = event_age is not None
    duration = event_age if observed else last_observed_age
    group_hash = _payload_hash(group_payload)
    payload = {
        "schema_version": SURVIVAL_SCHEMA_VERSION,
        "run_id": run_id,
        "parent_cluster_id": parent_cluster_id,
        "group": group_payload,
        "group_hash": group_hash,
        "operational_status": str(operational_status),
        "clock": {
            "origin": "parent_checkpoint",
            "parent_time": parent,
            "analysis_start_age": start_value,
            "last_observed_age": last_observed_age,
        },
        "event": {
            "type": "dwell_qualified_decay",
            "energy_key": energy_key,
            "threshold": threshold_value,
            "dwell_time": dwell_value,
            "observed": observed,
            "first_passage_age": event_age,
            "qualification_age": qualification_age,
            "right_censor_age": None if observed else last_observed_age,
        },
        "duration": duration,
        "event_observed": observed,
        "source": dict(source or {}),
    }
    return {**payload, "observation_hash": _payload_hash(payload)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SurvivalAnalysisError(f"cannot read {path}") from exc
    if not isinstance(loaded, dict):
        raise SurvivalAnalysisError(f"{path} must contain a JSON object")
    return loaded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SurvivalAnalysisError(
                    f"{path}:{lineno} must contain a JSON object"
                )
            rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise SurvivalAnalysisError(f"cannot read {path}") from exc
    if not rows:
        raise SurvivalAnalysisError(f"{path} contains no diagnostic rows")
    return rows


def _load_run_spec(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    archived = run_dir / "spec.json"
    if archived.exists():
        return _read_json(archived)
    config_path = metadata.get("config_path")
    if isinstance(config_path, str) and Path(config_path).exists():
        return _read_json(Path(config_path))
    raise SurvivalAnalysisError(
        f"{run_dir} has neither an archived spec.json nor a readable config_path"
    )


def _diagnostics_source(run_dir: Path, metadata: dict[str, Any]) -> Path:
    execution = metadata.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    status = execution.get("status")
    canonical = run_dir / "diagnostics.jsonl"
    if status == "completed":
        if canonical.exists():
            return canonical
        raise SurvivalAnalysisError(
            f"{run_dir} is completed but lacks diagnostics.jsonl"
        )

    declared = execution.get("partial_diagnostics_path")
    if not isinstance(declared, str) or not declared:
        raise SurvivalAnalysisError(
            f"{run_dir} is non-completed but does not declare its partial diagnostics"
        )
    if Path(declared).name != "diagnostics.partial.jsonl":
        raise SurvivalAnalysisError(
            f"{run_dir} declares an unsupported partial diagnostics artifact"
        )
    partial = run_dir / "diagnostics.partial.jsonl"
    if not partial.exists():
        raise SurvivalAnalysisError(
            f"{run_dir} declares diagnostics.partial.jsonl but the file is missing"
        )
    return partial


def _magnetic_bc(spec: dict[str, Any]) -> Any:
    conditions = spec.get("boundary_conditions")
    if not isinstance(conditions, dict):
        return None
    magnetic = conditions.get("magnetic")
    return magnetic.get("type") if isinstance(magnetic, dict) else None


def _analysis_start_age(
    metadata: dict[str, Any],
    *,
    parent_step: int,
    dt: float,
) -> float:
    quench = metadata["quench"]
    if "classification_valid_after_tstep" not in quench:
        raise SurvivalAnalysisError(
            "quench metadata must declare classification_valid_after_tstep explicitly"
        )
    valid_after = quench["classification_valid_after_tstep"]
    if isinstance(valid_after, bool):
        raise SurvivalAnalysisError(
            "classification_valid_after_tstep must be an integer"
        )
    try:
        valid_step = int(valid_after)
    except (TypeError, ValueError) as exc:
        raise SurvivalAnalysisError(
            "classification_valid_after_tstep must be an integer"
        ) from exc
    if valid_step != valid_after or valid_step < parent_step:
        raise SurvivalAnalysisError(
            "classification_valid_after_tstep precedes the parent checkpoint"
        )
    return (valid_step - parent_step) * dt


def _survival_group(
    metadata: dict[str, Any],
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    energy_key: str,
    threshold: float,
    dwell_time: float,
    analysis_start_age: float,
) -> dict[str, Any]:
    quench = metadata["quench"]
    run_options = metadata.get("run_options")
    run_options = run_options if isinstance(run_options, dict) else {}
    device = metadata.get("device")
    device = device if isinstance(device, dict) else {}
    integrator = metadata.get("integrator")
    integrator = integrator if isinstance(integrator, dict) else {}
    convention = next(
        (
            row.get("energy_convention")
            for row in reversed(rows)
            if isinstance(row.get("energy_convention"), str)
        ),
        energy_convention_for_spec(spec),
    )
    group = {
        "problem_id": metadata.get("problem_id"),
        "geometry": metadata.get("geometry"),
        "physics": metadata.get("physics"),
        "representation": spec.get("representation"),
        "magnetic_bc": _magnetic_bc(spec),
        "numerics_contract_version": metadata.get("numerics_contract_version"),
        "child_spec_hash": metadata.get("spec_hash"),
        "parent_spec_hash": quench.get("parent_spec_hash"),
        "mutable_diff_hash": _payload_hash(quench.get("mutable_diff", {})),
        "resolution_tier": run_options.get("resolution_tier"),
        "precision": device.get("production_run_dtype", spec.get("precision")),
        "integrator": integrator.get("actual", spec.get("time", {}).get("integrator")),
        "dt": integrator.get("dt", spec.get("time", {}).get("dt")),
        "energy_convention": convention,
        "energy_key": energy_key,
        "decay_threshold": threshold,
        "dwell_time": dwell_time,
        "analysis_start_age": analysis_start_age,
    }
    if tuple(group) != SURVIVAL_GROUPING_KEYS:
        raise AssertionError("survival grouping-key implementation drift")
    missing = [
        key
        for key in (
            "problem_id",
            "geometry",
            "physics",
            "representation",
            "numerics_contract_version",
            "child_spec_hash",
            "parent_spec_hash",
            "integrator",
            "dt",
        )
        if group[key] is None
    ]
    if missing:
        raise SurvivalAnalysisError(
            f"survival grouping metadata is missing required keys: {missing}"
        )
    return group


def load_quench_observation(
    run_dir: str | Path,
    *,
    energy_key: str,
    threshold: float,
    dwell_time: float,
) -> dict[str, Any]:
    """Load one run directory into a canonical first-passage observation."""

    path = Path(run_dir)
    metadata = _read_json(path / "metadata.json")
    quench = metadata.get("quench")
    if not isinstance(quench, dict) or quench.get("mode") != "quench":
        raise SurvivalAnalysisError(f"{path} is not an explicit quench run")
    duration = quench.get("duration")
    duration = duration if isinstance(duration, dict) else {}
    parent_checkpoint = duration.get("parent_checkpoint")
    if not isinstance(parent_checkpoint, dict):
        raise SurvivalAnalysisError(
            f"{path} lacks authenticated quench parent_checkpoint metadata"
        )
    parent_time = _finite_number(
        parent_checkpoint.get("time"), "parent checkpoint time"
    )
    parent_step_raw = parent_checkpoint.get("step")
    if isinstance(parent_step_raw, bool):
        raise SurvivalAnalysisError("parent checkpoint step must be an integer")
    try:
        parent_step = int(parent_step_raw)
    except (TypeError, ValueError) as exc:
        raise SurvivalAnalysisError(
            "parent checkpoint step must be an integer"
        ) from exc
    if parent_step != parent_step_raw:
        raise SurvivalAnalysisError("parent checkpoint step must be an integer")

    spec = _load_run_spec(path, metadata)
    diagnostics_path = _diagnostics_source(path, metadata)
    rows = _read_jsonl(diagnostics_path)
    integrator = metadata.get("integrator")
    integrator = integrator if isinstance(integrator, dict) else {}
    spec_time = spec.get("time")
    spec_time = spec_time if isinstance(spec_time, dict) else {}
    dt = _finite_number(integrator.get("dt", spec_time.get("dt")), "child dt")
    if dt <= 0.0:
        raise SurvivalAnalysisError("child dt must be positive")
    start_age = _analysis_start_age(metadata, parent_step=parent_step, dt=dt)
    threshold_value, dwell_value, start_age = _validate_decay_controls(
        threshold=threshold,
        dwell_time=dwell_time,
        analysis_start_age=start_age,
    )
    group = _survival_group(
        metadata,
        spec,
        rows,
        energy_key=energy_key,
        threshold=threshold_value,
        dwell_time=dwell_value,
        analysis_start_age=start_age,
    )
    parent_run_dir = quench.get("parent_run_dir")
    if not isinstance(parent_run_dir, str) or not parent_run_dir:
        raise SurvivalAnalysisError("quench metadata lacks parent_run_dir")
    parent_cluster_id = _payload_hash(
        {
            "parent_run_dir": parent_run_dir,
            "parent_spec_hash": quench.get("parent_spec_hash"),
        }
    )
    run_id = _payload_hash(
        {
            "out_dir": metadata.get("out_dir", str(path)),
            "generated_at_utc": metadata.get("generated_at_utc"),
            "child_spec_hash": metadata.get("spec_hash"),
            "parent_checkpoint_sha256": quench.get("parent_checkpoint_sha256"),
        }
    )
    execution = metadata.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    return quench_first_passage(
        rows,
        run_id=run_id,
        parent_cluster_id=parent_cluster_id,
        group=group,
        parent_time=parent_time,
        energy_key=energy_key,
        threshold=threshold_value,
        dwell_time=dwell_value,
        analysis_start_age=start_age,
        operational_status=str(execution.get("status", "unknown")),
        source={
            "run_dir": str(path),
            "diagnostics_path": str(diagnostics_path),
            "child_spec_hash": metadata.get("spec_hash"),
            "parent_spec_hash": quench.get("parent_spec_hash"),
            "parent_checkpoint_sha256": quench.get("parent_checkpoint_sha256"),
        },
    )


def _validate_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(observations, list) or not observations:
        raise SurvivalAnalysisError("Kaplan-Meier input must be non-empty")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise SurvivalAnalysisError(f"observation {index} must be an object")
        supplied_hash = observation.get("observation_hash")
        payload = {
            key: value
            for key, value in observation.items()
            if key != "observation_hash"
        }
        if supplied_hash != _payload_hash(payload):
            raise SurvivalAnalysisError(
                f"observation {index} hash does not authenticate its payload"
            )
        run_id = observation.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in seen:
            raise SurvivalAnalysisError(
                "survival observations require unique non-empty run_id values"
            )
        seen.add(run_id)
        group = observation.get("group")
        if not isinstance(group, dict):
            raise SurvivalAnalysisError(f"observation {index} lacks group metadata")
        canonical_group = _canonical_group(group)
        if observation.get("group_hash") != _payload_hash(canonical_group):
            raise SurvivalAnalysisError(
                f"observation {index} group hash does not match its grouping keys"
            )
        duration = _finite_number(
            observation.get("duration"), f"observation {index} duration"
        )
        if duration < 0.0:
            raise SurvivalAnalysisError(
                f"observation {index} duration must be nonnegative"
            )
        if not isinstance(observation.get("event_observed"), bool):
            raise SurvivalAnalysisError(
                f"observation {index} event_observed must be boolean"
            )
        cluster = observation.get("parent_cluster_id")
        if not isinstance(cluster, str) or not cluster:
            raise SurvivalAnalysisError(f"observation {index} lacks parent_cluster_id")
        validated.append(observation)
    return validated


def _loglog_interval(
    survival: float,
    greenwood_sum: float,
    *,
    confidence_level: float,
) -> tuple[float, float]:
    if survival <= 0.0:
        return 0.0, 0.0
    if survival >= 1.0 or greenwood_sum <= 0.0:
        return survival, survival
    if not math.isfinite(greenwood_sum):
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + 0.5 * confidence_level)
    log_survival = math.log(survival)
    transformed = math.log(-log_survival)
    standard_error = math.sqrt(greenwood_sum) / abs(log_survival)
    lower = math.exp(-math.exp(transformed + z * standard_error))
    upper = math.exp(-math.exp(transformed - z * standard_error))
    return max(0.0, lower), min(1.0, upper)


def _km_curve(
    observations: list[dict[str, Any]],
    *,
    confidence_level: float,
) -> list[dict[str, Any]]:
    times = sorted({float(item["duration"]) for item in observations})
    at_risk = len(observations)
    survival = 1.0
    greenwood_sum = 0.0
    curve: list[dict[str, Any]] = []
    for time in times:
        events = sum(
            item["event_observed"] and float(item["duration"]) == time
            for item in observations
        )
        censored = sum(
            (not item["event_observed"]) and float(item["duration"]) == time
            for item in observations
        )
        if events:
            survival *= 1.0 - events / at_risk
            if at_risk - events > 0:
                greenwood_sum += events / (at_risk * (at_risk - events))
            else:
                greenwood_sum = math.inf
        lower, upper = _loglog_interval(
            survival,
            greenwood_sum,
            confidence_level=confidence_level,
        )
        curve.append(
            {
                "time": time,
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
                "survival": survival,
                "greenwood_loglog_lower": lower,
                "greenwood_loglog_upper": upper,
                "cluster_bootstrap_lower": None,
                "cluster_bootstrap_upper": None,
            }
        )
        at_risk -= events + censored
    return curve


def _survival_at(curve: list[dict[str, Any]], time: float) -> float:
    survival = 1.0
    for row in curve:
        if float(row["time"]) > time:
            break
        survival = float(row["survival"])
    return survival


def _cluster_bootstrap(
    observations: list[dict[str, Any]],
    curve: list[dict[str, Any]],
    *,
    confidence_level: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        clusters.setdefault(observation["parent_cluster_id"], []).append(observation)
    cluster_ids = sorted(clusters)
    if samples <= 0:
        return {
            "method": "disabled",
            "samples": 0,
            "seed": seed,
            "parent_clusters": len(cluster_ids),
        }
    if len(cluster_ids) < 2:
        return {
            "method": "unavailable",
            "reason": "at least two parent clusters are required",
            "samples": samples,
            "seed": seed,
            "parent_clusters": len(cluster_ids),
        }

    rng = np.random.default_rng(seed)
    times = [float(row["time"]) for row in curve]
    estimates = np.empty((samples, len(times)), dtype=float)
    for sample_index in range(samples):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample_observations = [
            observation
            for cluster_id in selected
            for observation in clusters[str(cluster_id)]
        ]
        sample_curve = _km_curve(
            sample_observations,
            confidence_level=confidence_level,
        )
        estimates[sample_index, :] = [
            _survival_at(sample_curve, time) for time in times
        ]

    alpha = 0.5 * (1.0 - confidence_level)
    lowers = np.quantile(estimates, alpha, axis=0)
    uppers = np.quantile(estimates, 1.0 - alpha, axis=0)
    for row, lower, upper in zip(curve, lowers, uppers, strict=True):
        row["cluster_bootstrap_lower"] = float(lower)
        row["cluster_bootstrap_upper"] = float(upper)
    return {
        "method": "parent_cluster_percentile_bootstrap",
        "samples": samples,
        "seed": seed,
        "parent_clusters": len(cluster_ids),
    }


def kaplan_meier(
    observations: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    cluster_bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate one grouped survival curve with uncertainty intervals."""

    validated = _validate_observations(observations)
    confidence = _finite_number(confidence_level, "confidence level")
    if not 0.0 < confidence < 1.0:
        raise SurvivalAnalysisError(
            "confidence level must lie strictly between 0 and 1"
        )
    if (
        isinstance(cluster_bootstrap_samples, bool)
        or not isinstance(cluster_bootstrap_samples, int)
        or cluster_bootstrap_samples < 0
    ):
        raise SurvivalAnalysisError(
            "cluster_bootstrap_samples must be a nonnegative integer"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SurvivalAnalysisError("bootstrap seed must be an integer")
    group_hashes = {item["group_hash"] for item in validated}
    if len(group_hashes) != 1:
        raise SurvivalAnalysisError(
            "Kaplan-Meier observations must share one complete grouping key"
        )

    curve = _km_curve(validated, confidence_level=confidence)
    bootstrap = _cluster_bootstrap(
        validated,
        curve,
        confidence_level=confidence,
        samples=cluster_bootstrap_samples,
        seed=seed,
    )
    median = next(
        (float(row["time"]) for row in curve if float(row["survival"]) <= 0.5),
        None,
    )
    event_count = sum(item["event_observed"] for item in validated)
    clusters = {item["parent_cluster_id"] for item in validated}
    return {
        "schema_version": SURVIVAL_SCHEMA_VERSION,
        "group": validated[0]["group"],
        "group_hash": validated[0]["group_hash"],
        "runs": len(validated),
        "events": event_count,
        "right_censored": len(validated) - event_count,
        "parent_clusters": len(clusters),
        "median_survival": median,
        "confidence_level": confidence,
        "uncertainty": {
            "pointwise": "Greenwood variance with complementary log-log transform",
            "clustered": bootstrap,
        },
        "curve": curve,
    }


def survival_ensembles(
    observations: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    cluster_bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Group authenticated observations and build deterministic KM ensembles."""

    validated = _validate_observations(observations)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for observation in validated:
        grouped.setdefault(observation["group_hash"], []).append(observation)
    ensembles = [
        kaplan_meier(
            sorted(grouped[group_hash], key=lambda item: item["run_id"]),
            confidence_level=confidence_level,
            cluster_bootstrap_samples=cluster_bootstrap_samples,
            seed=seed,
        )
        for group_hash in sorted(grouped)
    ]
    ordered_observations = sorted(
        validated, key=lambda item: (item["group_hash"], item["run_id"])
    )
    payload = {
        "schema_version": SURVIVAL_SCHEMA_VERSION,
        "grouping_keys": list(SURVIVAL_GROUPING_KEYS),
        "observations": ordered_observations,
        "ensembles": ensembles,
    }
    return {**payload, "analysis_hash": _payload_hash(payload)}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build dwell-qualified quench survival ensembles."
    )
    parser.add_argument("run_dirs", nargs="+", help="Quench run directories.")
    parser.add_argument("--energy-key", default="mag_energy_fluct")
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--dwell-time", required=True, type=float)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--cluster-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    observations = [
        load_quench_observation(
            run_dir,
            energy_key=args.energy_key,
            threshold=args.threshold,
            dwell_time=args.dwell_time,
        )
        for run_dir in args.run_dirs
    ]
    analysis = survival_ensembles(
        observations,
        confidence_level=args.confidence,
        cluster_bootstrap_samples=args.cluster_bootstrap,
        seed=args.seed,
    )
    _write_json_atomic(args.out, analysis)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "analysis_hash": analysis["analysis_hash"],
                "observations": len(analysis["observations"]),
                "ensembles": len(analysis["ensembles"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
