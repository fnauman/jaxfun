"""Adaptive, uncertainty-aware scientific frontier refinement (issue #10).

The Cartesian sweep executor remains the mechanism that materializes and runs
points. This module adds a deterministic controller around it: normalize runner
metadata, locate a decayed/non-decayed bracket, bisect it, and persist a
hash-linked lineage whose endpoints carry immutable run and spec identities.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .problem_spec import load_spec

FRONTIER_SCHEMA_VERSION = 1
_NEGATIVE_CLASSES = frozenset({"decayed"})
_POSITIVE_CLASSES = frozenset({"growing", "sustained"})
_BRACKET_CLASSES = _NEGATIVE_CLASSES | _POSITIVE_CLASSES
FRONTIER_TERMINAL_STATUSES = frozenset(
    {
        "converged",
        "max_refinements",
        "nonmonotonic",
        "uncertain_endpoints",
        "uncertain_points",
        "unbracketed",
    }
)


class FrontierRefinementError(ValueError):
    """Raised when a frontier request or lineage is malformed."""


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_sweep_result(
    metadata: dict[str, Any] | None,
    *,
    fallback_status: str,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Normalize one runner result into stable sweep-index fields."""

    metadata = metadata if isinstance(metadata, dict) else {}
    execution = metadata.get("execution")
    execution = execution if isinstance(execution, dict) else {}
    classification = metadata.get("classification")
    classification = classification if isinstance(classification, dict) else {}
    fit = classification.get("fit")
    fit = fit if isinstance(fit, dict) else {}

    operational = str(execution.get("status") or fallback_status)
    scientific = classification.get("scientific_class")
    scientific = None if scientific is None else str(scientific)
    reason = failure_reason or execution.get("failure_reason")
    return {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "operational_status": operational,
        "failure_reason": None if reason is None else str(reason),
        "scientific_class": scientific,
        "classification_reason": classification.get("reason"),
        "fit_slope": _finite_or_none(fit.get("slope")),
        "fit_stderr": _finite_or_none(fit.get("stderr")),
        "classification_eligible": bool(
            operational == "completed" and scientific in _BRACKET_CLASSES
        ),
    }


def _point_from_entry(entry: dict[str, Any], axis: str) -> dict[str, Any] | None:
    overrides = entry.get("overrides")
    if not isinstance(overrides, dict):
        return None
    value = _finite_or_none(overrides.get(axis))
    if value is None:
        return None
    result = entry.get("result")
    result = result if isinstance(result, dict) else {}
    run_id = entry.get("run_id")
    spec_hash = entry.get("spec_hash")
    eligible = bool(result.get("classification_eligible", False))
    if eligible and (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(spec_hash, str)
        or not spec_hash
    ):
        raise FrontierRefinementError(
            "eligible frontier points require run_id and spec_hash"
        )
    return {
        "value": value,
        "run_id": run_id,
        "spec_hash": spec_hash,
        "operational_status": result.get("operational_status", entry.get("status")),
        "scientific_class": result.get("scientific_class"),
        "fit_slope": _finite_or_none(result.get("fit_slope")),
        "fit_stderr": _finite_or_none(result.get("fit_stderr")),
        "classification_eligible": eligible,
    }


def _side(point: dict[str, Any]) -> int:
    scientific = point.get("scientific_class")
    if not point.get("classification_eligible"):
        return 0
    if scientific in _NEGATIVE_CLASSES:
        return -1
    if scientific in _POSITIVE_CLASSES:
        return 1
    return 0


def _endpoint_confident(point: dict[str, Any], confidence_z: float) -> bool:
    if confidence_z == 0.0 or point.get("scientific_class") == "sustained":
        return True
    slope = point.get("fit_slope")
    stderr = point.get("fit_stderr")
    if slope is None or stderr is None or stderr < 0.0:
        return False
    if point.get("scientific_class") == "growing":
        return slope - confidence_z * stderr > 0.0
    if point.get("scientific_class") == "decayed":
        return slope + confidence_z * stderr < 0.0
    return False


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["lineage_hash"] = _payload_hash(sealed)
    return sealed


def frontier_decision(
    entries: list[dict[str, Any]],
    *,
    axis: str,
    abs_tolerance: float,
    rel_tolerance: float = 0.0,
    confidence_z: float = 1.96,
    parent_lineage_hash: str | None = None,
    request_hash: str | None = None,
) -> dict[str, Any]:
    """Return a sealed next-step decision from canonical sweep-index entries."""

    if not axis:
        raise FrontierRefinementError("frontier axis must be non-empty")
    for label, value in (
        ("abs_tolerance", abs_tolerance),
        ("rel_tolerance", rel_tolerance),
        ("confidence_z", confidence_z),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise FrontierRefinementError(f"{label} must be finite and nonnegative")
    if abs_tolerance == 0.0 and rel_tolerance == 0.0:
        raise FrontierRefinementError(
            "at least one frontier tolerance must be positive"
        )

    points = sorted(
        (
            point
            for entry in entries
            if (point := _point_from_entry(entry, axis)) is not None
        ),
        key=lambda point: point["value"],
    )
    values = [point["value"] for point in points]
    if len(values) != len(set(values)):
        raise FrontierRefinementError(
            f"frontier axis {axis!r} has duplicate values; fix all other sweep "
            "controls so each frontier coordinate identifies exactly one run"
        )

    base = {
        "schema_version": FRONTIER_SCHEMA_VERSION,
        "axis": axis,
        "parent_lineage_hash": parent_lineage_hash,
        "request_hash": request_hash,
        "abs_tolerance": float(abs_tolerance),
        "rel_tolerance": float(rel_tolerance),
        "confidence_z": float(confidence_z),
        "sampled_points": points,
        "bracket": None,
        "next_value": None,
    }
    eligible = [(index, point, _side(point)) for index, point in enumerate(points)]
    eligible = [item for item in eligible if item[2] != 0]
    if len(eligible) < 2:
        return _seal({**base, "status": "incomplete"})

    transitions = [
        (left, right)
        for left, right in zip(eligible, eligible[1:], strict=False)
        if left[2] != right[2]
    ]
    if not transitions:
        return _seal({**base, "status": "unbracketed"})
    if len(transitions) > 1:
        return _seal({**base, "status": "nonmonotonic"})

    left, right = transitions[0]
    if left[2] != -1 or right[2] != 1:
        return _seal({**base, "status": "nonmonotonic"})
    if any(_side(point) == 0 for point in points[left[0] + 1 : right[0]]):
        return _seal({**base, "status": "uncertain_points"})

    low = left[1]
    high = right[1]
    width = high["value"] - low["value"]
    tolerance = max(
        float(abs_tolerance),
        float(rel_tolerance) * max(abs(low["value"]), abs(high["value"])),
    )
    bracket = {
        "low": low,
        "high": high,
        "width": width,
        "tolerance": tolerance,
    }
    decision = {**base, "bracket": bracket}
    if not (
        _endpoint_confident(low, float(confidence_z))
        and _endpoint_confident(high, float(confidence_z))
    ):
        return _seal({**decision, "status": "uncertain_endpoints"})
    if width <= tolerance:
        return _seal({**decision, "status": "converged"})

    midpoint = low["value"] + 0.5 * width
    if midpoint in {low["value"], high["value"]}:
        return _seal({**decision, "status": "converged"})
    return _seal({**decision, "status": "refine", "next_value": midpoint})


def verify_frontier_lineage(records: list[dict[str, Any]]) -> bool:
    """Verify every record hash and parent link in a frontier lineage."""

    parent: str | None = None
    for index, record in enumerate(records):
        if record.get("parent_lineage_hash") != parent:
            raise FrontierRefinementError(
                f"frontier lineage parent mismatch at record {index}"
            )
        supplied = record.get("lineage_hash")
        payload = {key: value for key, value in record.items() if key != "lineage_hash"}
        if supplied != _payload_hash(payload):
            raise FrontierRefinementError(
                f"frontier lineage hash mismatch at record {index}"
            )
        parent = str(supplied)
    return True


def _write_lineage(path: Path, records: list[dict[str, Any]]) -> None:
    verify_frontier_lineage(records)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", "utf-8")
    tmp.replace(path)


def _frontier_summary(
    *,
    decision: dict[str, Any],
    sweep_index_path: Path,
    lineage_path: Path,
) -> dict[str, Any]:
    return {
        "status": decision["status"],
        "axis": decision["axis"],
        "bracket": decision.get("bracket"),
        "next_value": decision.get("next_value"),
        "sampled_points": len(decision.get("sampled_points", [])),
        "lineage_hash": decision["lineage_hash"],
        "sweep_index_path": str(sweep_index_path),
        "lineage_path": str(lineage_path),
    }


def execute_frontier_sweep(
    base_spec_path: str | Path,
    *,
    axis: str,
    bounds: tuple[float, float] | list[float],
    out_dir: str | Path,
    fixed_overrides: dict[str, Any] | None = None,
    abs_tolerance: float,
    rel_tolerance: float = 0.0,
    confidence_z: float = 1.96,
    max_refinements: int = 8,
    runner: Any | None = None,
    resolution_tier: str | None = None,
    steps: int | None = None,
    wandb: bool = False,
) -> dict[str, Any]:
    """Execute/resume a one-axis bisection until a safe terminal decision."""

    from .sweep import apply_overrides, execute_sweep, run_id_for

    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise FrontierRefinementError("frontier bounds must contain exactly two values")
    low, high = (float(value) for value in bounds)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise FrontierRefinementError("frontier bounds must be finite and increasing")
    if isinstance(max_refinements, bool) or int(max_refinements) != max_refinements:
        raise FrontierRefinementError("max_refinements must be a nonnegative integer")
    if max_refinements < 0:
        raise FrontierRefinementError("max_refinements must be nonnegative")
    fixed = dict(fixed_overrides or {})
    if axis in fixed:
        raise FrontierRefinementError("frontier axis cannot also be a fixed override")
    base_spec = load_spec(base_spec_path)
    request_hash = _payload_hash(
        {
            "schema_version": FRONTIER_SCHEMA_VERSION,
            "base_spec_hash": base_spec["spec_hash"],
            "axis": axis,
            "bounds": [low, high],
            "fixed_overrides": fixed,
            "abs_tolerance": float(abs_tolerance),
            "rel_tolerance": float(rel_tolerance),
            "confidence_z": float(confidence_z),
            "max_refinements": int(max_refinements),
            "resolution_tier": resolution_tier,
            "steps": steps,
        }
    )

    expected_override_keys = {axis, *fixed}

    def requested_entries(
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select only points authenticated for this base spec and control slice."""

        selected: list[dict[str, Any]] = []
        for entry in entries:
            overrides = entry.get("overrides")
            if (
                not isinstance(overrides, dict)
                or set(overrides) != expected_override_keys
            ):
                continue
            value = _finite_or_none(overrides.get(axis))
            if value is None or not low <= value <= high:
                continue
            if any(overrides.get(key) != expected for key, expected in fixed.items()):
                continue
            resolved = apply_overrides(base_spec, {**fixed, axis: value})
            if (
                entry.get("run_id") != run_id_for(resolved)
                or entry.get("spec_hash") != resolved["spec_hash"]
            ):
                continue
            selected.append(entry)
        return selected

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sweep_index_path = out_path / "sweep_index.json"
    lineage_path = out_path / "frontier_index.json"
    records: list[dict[str, Any]] = []
    if lineage_path.exists():
        loaded = json.loads(lineage_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise FrontierRefinementError("frontier_index.json must contain a list")
        records = loaded
        verify_frontier_lineage(records)
        if records and records[0].get("request_hash") != request_hash:
            raise FrontierRefinementError(
                "frontier resume request does not match the authenticated lineage"
            )
        if records and records[-1].get("status") in FRONTIER_TERMINAL_STATUSES:
            return _frontier_summary(
                decision=records[-1],
                sweep_index_path=sweep_index_path,
                lineage_path=lineage_path,
            )

    values = {low, high}
    if sweep_index_path.exists():
        existing = json.loads(sweep_index_path.read_text(encoding="utf-8"))
        for entry in requested_entries(existing):
            values.add(float(entry["overrides"][axis]))

    parent = records[-1]["lineage_hash"] if records else None
    refinements = sum(record.get("status") == "refine" for record in records)
    if records and records[-1].get("status") == "refine":
        values.add(float(records[-1]["next_value"]))

    while True:
        grid = {key: [value] for key, value in fixed.items()}
        grid[axis] = sorted(values)
        execute_sweep(
            base_spec_path,
            grid,
            out_path,
            execute=True,
            runner=runner,
            resolution_tier=resolution_tier,
            steps=steps,
            wandb=wandb,
        )
        entries = requested_entries(
            json.loads(sweep_index_path.read_text(encoding="utf-8"))
        )
        decision = frontier_decision(
            entries,
            axis=axis,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
            confidence_z=confidence_z,
            parent_lineage_hash=parent,
            request_hash=request_hash,
        )
        if decision["status"] == "refine" and refinements >= max_refinements:
            payload = {
                key: value for key, value in decision.items() if key != "lineage_hash"
            }
            payload["status"] = "max_refinements"
            payload["next_value"] = None
            decision = _seal(payload)

        records.append(decision)
        _write_lineage(lineage_path, records)
        if decision["status"] != "refine":
            return _frontier_summary(
                decision=decision,
                sweep_index_path=sweep_index_path,
                lineage_path=lineage_path,
            )
        values.add(float(decision["next_value"]))
        parent = decision["lineage_hash"]
        refinements += 1
