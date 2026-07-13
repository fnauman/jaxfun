"""Explicit quench / continue-from validation and checkpoint banks (FJ-05).

Strict same-spec resume (``resume-exact``) requires an identical ``spec_hash``. A
*quench* is a separate, explicit continuation that lowers ``Rm`` (or ``Re``) from a
selected plateau state while keeping everything else immutable. This module defines
the mutable-field allowlist and validates that a child spec differs from its parent
only within that allowlist, and it builds a checkpoint-bank manifest entry recording
provenance for each banked plateau state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

# Only the resistive/viscous coefficients (and their nondimensional labels) may change
# in a quench. Everything else -- geometry, basis, resolution, state layout, wall
# conditions, coordinate order, representation, numerics contract -- is immutable.
MUTABLE_QUENCH_FIELDS: frozenset[str] = frozenset(
    {
        "nondimensional_groups.nu",
        "nondimensional_groups.eta_mag",
        "nondimensional_groups.eta",
        "nondimensional_groups.Re",
        "nondimensional_groups.Rm",
        "nondimensional_groups.Re_h",
        "nondimensional_groups.Rm_h",
        "nondimensional_groups.Re_TC",
        "nondimensional_groups.Rm_TC",
        "nondimensional_groups.Pm",
    }
)

# Derived / bookkeeping keys that are ignored when diffing parent vs child.
_IGNORED_FIELDS: frozenset[str] = frozenset({"spec_hash"})

# Keys that must be present and unchanged for a quench to be admissible.
_IMMUTABLE_REQUIRED: tuple[str, ...] = (
    "geometry",
    "physics",
    "numerics_contract_version",
    "representation",
)

_QUENCH_DECREASING_GROUPS: frozenset[str] = frozenset(
    {
        "nondimensional_groups.Re",
        "nondimensional_groups.Rm",
        "nondimensional_groups.Re_h",
        "nondimensional_groups.Rm_h",
        "nondimensional_groups.Re_TC",
        "nondimensional_groups.Rm_TC",
    }
)
_QUENCH_INCREASING_DIFFUSIVITIES: frozenset[str] = frozenset(
    {
        "nondimensional_groups.nu",
        "nondimensional_groups.eta",
        "nondimensional_groups.eta_mag",
    }
)


class QuenchError(ValueError):
    """Raised when a quench continuation violates the immutability contract."""


QUENCH_DURATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FixedQuenchDuration:
    """Resolved fixed-step horizon for one explicit quench continuation."""

    request_kind: str
    requested_additional_time: float | None
    requested_additional_steps: int | None
    additional_time: float
    additional_steps: int
    parent_time: float
    parent_step: int
    target_time: float
    target_step: int
    child_dt: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": QUENCH_DURATION_SCHEMA_VERSION,
            "stepping": "fixed",
            "request_kind": self.request_kind,
            "child_dt": self.child_dt,
            "parent_checkpoint": {
                "time": self.parent_time,
                "step": self.parent_step,
            },
            "requested": {
                "additional_time": self.requested_additional_time,
                "additional_steps": self.requested_additional_steps,
            },
            "absolute_target": {
                "time": self.target_time,
                "step": self.target_step,
            },
            "attained": {
                "final_time": None,
                "final_step": None,
                "additional_time": None,
                "additional_steps": None,
                "target_reached": None,
            },
        }


def validate_quench_duration_request(
    *,
    quench: bool,
    steps: int | None,
    additional_time: float | None,
    additional_steps: int | None,
) -> None:
    """Validate the API/CLI shape before any output or solver work begins."""

    supplied = int(additional_time is not None) + int(additional_steps is not None)
    if not quench:
        if supplied:
            raise QuenchError(
                "additional_time/additional_steps require an explicit quench"
            )
        return
    if steps is not None:
        raise QuenchError(
            "steps is an absolute fresh/resume target and cannot be used for a "
            "quench; use exactly one of additional_time or additional_steps"
        )
    if supplied != 1:
        raise QuenchError(
            "quench requires exactly one of additional_time or additional_steps; "
            "the child spec final_time alone is not a quench horizon"
        )
    if additional_time is not None and (
        isinstance(additional_time, bool)
        or not isinstance(additional_time, Real)
        or not math.isfinite(float(additional_time))
        or float(additional_time) <= 0.0
    ):
        raise QuenchError("additional_time must be finite and strictly positive")
    if additional_steps is not None and (
        isinstance(additional_steps, bool)
        or not isinstance(additional_steps, Integral)
        or int(additional_steps) <= 0
    ):
        raise QuenchError("additional_steps must be a strictly positive integer")


def validate_quench_output_options(
    *, quench: bool, compare_golden: bool, write_golden: bool
) -> None:
    """Reject golden workflows whose semantics do not apply to a quench."""

    if not quench:
        return
    requested = [
        name
        for name, enabled in (
            ("compare_golden", compare_golden),
            ("write_golden", write_golden),
        )
        if enabled
    ]
    if requested:
        raise QuenchError(
            f"{', '.join(requested)} cannot be used with a quench; transition "
            "growth, decay, or saturation is a scientific outcome"
        )


def validate_burn_in_request(
    *,
    quench: bool,
    burn_in_steps: int,
    resolved_additional_steps: int | None = None,
) -> None:
    """Validate burn-in shape and, once known, its quench-horizon bound."""

    if isinstance(burn_in_steps, bool) or not isinstance(burn_in_steps, Integral):
        raise QuenchError("burn_in_steps must be an integer, not a boolean")
    burn_in = int(burn_in_steps)
    if not quench:
        if burn_in != 0:
            raise QuenchError("burn_in_steps must be 0 outside an explicit quench")
        return
    if burn_in < 0:
        raise QuenchError("burn_in_steps must be non-negative for a quench")
    if resolved_additional_steps is None:
        return
    if (
        isinstance(resolved_additional_steps, bool)
        or not isinstance(resolved_additional_steps, Integral)
        or int(resolved_additional_steps) <= 0
    ):
        raise QuenchError("resolved quench additional_steps must be a positive integer")
    additional_steps = int(resolved_additional_steps)
    if burn_in >= additional_steps:
        raise QuenchError(
            "burn_in_steps must be strictly less than the resolved quench "
            f"additional_steps ({burn_in} >= {additional_steps})"
        )


def validate_quench_runner_preflight(spec: dict[str, Any], *, quench: bool) -> None:
    """Allow quenching only through runners with an explicit continuation path."""

    if not quench:
        return
    geometry = spec.get("geometry")
    physics = spec.get("physics")
    representation = spec.get("representation")
    oracle = (spec.get("expected_oracle") or {}).get("type")
    pcf_primitive = (
        geometry == "pcf"
        and representation == "primitive"
        and (
            (physics == "mhd" and oracle == "gpu_generated_saturated_dns")
            or (physics == "mri" and oracle == "mri_saturation_ladder")
        )
    )
    pcf_vector_potential = (
        geometry == "pcf"
        and physics in {"mhd", "mri"}
        and representation == "vector_potential"
        and oracle in {"gpu_generated_saturated_dns", "mri_saturation_ladder"}
    )
    tc_vector_potential = (
        geometry == "taylor_couette"
        and physics in {"mhd", "mri"}
        and representation == "vector_potential"
        and oracle == "tc_mri_saturation_ladder"
    )
    if pcf_primitive or pcf_vector_potential or tc_vector_potential:
        return
    raise QuenchError(
        "quench runner is not implemented for "
        f"geometry={geometry!r}, physics={physics!r}, "
        f"representation={representation!r}, expected_oracle.type={oracle!r}; "
        "supported quench runners are PCF primitive MHD/MRI saturation, PCF "
        "vector-potential MHD/MRI saturation, and Taylor-Couette "
        "vector-potential MHD/MRI saturation"
    )


def resolve_fixed_quench_duration(
    *,
    additional_time: float | None,
    additional_steps: int | None,
    child_dt: float,
    parent_time: float,
    parent_step: int,
) -> FixedQuenchDuration:
    """Resolve an explicit additional request against a selected parent state."""

    validate_quench_duration_request(
        quench=True,
        steps=None,
        additional_time=additional_time,
        additional_steps=additional_steps,
    )
    dt = float(child_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise QuenchError("fixed-step quench requires a finite positive child dt")
    t0 = float(parent_time)
    if not math.isfinite(t0):
        raise QuenchError("parent checkpoint time must be finite")
    if isinstance(parent_step, bool) or not isinstance(parent_step, Integral):
        raise QuenchError("parent checkpoint step must be an integer")
    step0 = int(parent_step)
    if step0 < 0:
        raise QuenchError("parent checkpoint step must be non-negative")

    if additional_time is not None:
        requested_time = float(additional_time)
        step_ratio = requested_time / dt
        resolved_steps = int(round(step_ratio))
        if resolved_steps <= 0 or not math.isclose(
            step_ratio,
            float(resolved_steps),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise QuenchError(
                f"additional_time={requested_time:g} must be an integer multiple "
                f"of child dt={dt:g} for fixed-step quenching"
            )
        request_kind = "additional_time"
        requested_steps = None
    else:
        requested_time = None
        resolved_steps = int(additional_steps)
        requested_steps = resolved_steps
        request_kind = "additional_steps"

    elapsed = float(resolved_steps) * dt
    return FixedQuenchDuration(
        request_kind=request_kind,
        requested_additional_time=requested_time,
        requested_additional_steps=requested_steps,
        additional_time=elapsed,
        additional_steps=resolved_steps,
        parent_time=t0,
        parent_step=step0,
        target_time=t0 + elapsed,
        target_step=step0 + resolved_steps,
        child_dt=dt,
    )


def validate_bank_checkpoint_record(
    entry: dict[str, Any],
    record: Any,
    *,
    parent_spec_hash: str | None = None,
) -> None:
    """Require the selected manifest entry and loaded checkpoint to agree."""

    try:
        entry_time = float(entry["state_time"])
        record_time = float(record.t)
    except (KeyError, TypeError, ValueError) as exc:
        raise QuenchError(
            "selected checkpoint-bank entry has invalid time/step metadata"
        ) from exc
    entry_step_raw = entry.get("tstep")
    record_step_raw = getattr(record, "tstep", None)
    if (
        isinstance(entry_step_raw, bool)
        or not isinstance(entry_step_raw, Integral)
        or isinstance(record_step_raw, bool)
        or not isinstance(record_step_raw, Integral)
    ):
        raise QuenchError(
            "selected checkpoint-bank entry and checkpoint tstep must be integers"
        )
    entry_step = int(entry_step_raw)
    record_step = int(record_step_raw)
    if entry_step != record_step:
        raise QuenchError(
            "selected checkpoint-bank entry does not match the loaded checkpoint: "
            f"tstep {entry_step} != {record_step}"
        )
    tolerance = 1.0e-12 * max(1.0, abs(entry_time), abs(record_time))
    if not (
        math.isfinite(entry_time)
        and math.isfinite(record_time)
        and math.isclose(entry_time, record_time, rel_tol=0.0, abs_tol=tolerance)
    ):
        raise QuenchError(
            "selected checkpoint-bank entry does not match the loaded checkpoint: "
            f"time {entry_time!r} != {record_time!r}"
        )

    attrs = getattr(record, "attrs", {})
    entry_hash = str(entry.get("spec_hash"))
    record_hash = str(attrs.get("spec_hash"))
    if entry_hash != record_hash:
        raise QuenchError(
            "selected checkpoint-bank entry spec_hash does not match the loaded "
            "checkpoint"
        )
    if parent_spec_hash is not None and entry_hash != str(parent_spec_hash):
        raise QuenchError(
            "selected checkpoint-bank entry spec_hash does not match parent spec.json"
        )
    try:
        entry_version = int(entry["numerics_contract_version"])
        record_version = int(attrs.get("numerics_contract_version"))
    except (KeyError, TypeError, ValueError) as exc:
        raise QuenchError(
            "selected checkpoint-bank entry has invalid numerics-contract metadata"
        ) from exc
    if entry_version != record_version:
        raise QuenchError(
            "selected checkpoint-bank entry numerics_contract_version does not "
            "match the loaded checkpoint"
        )


def finalize_fixed_quench_duration(
    duration: dict[str, Any], *, final_time: float, final_step: int
) -> dict[str, Any]:
    """Fill the attained fixed-step horizon from the solver's emitted endpoint."""

    data = copy.deepcopy(duration)
    parent = data["parent_checkpoint"]
    target = data["absolute_target"]
    dt = float(data["child_dt"])
    t0 = float(parent["time"])
    step0 = int(parent["step"])
    tf = float(final_time)
    if not math.isfinite(tf):
        raise QuenchError("quench solver endpoint time must be finite")
    if isinstance(final_step, bool) or not isinstance(final_step, Integral):
        raise QuenchError("quench solver endpoint step must be an integer")
    stepf = int(final_step)
    elapsed = tf - t0
    taken = stepf - step0
    target_time = float(target["time"])
    step_consistent_time = t0 + float(taken) * dt
    time_scale = max(1.0, abs(tf), abs(target_time), abs(step_consistent_time))
    # The integer endpoint is authoritative for fixed stepping. Repeated time
    # additions can accumulate O(n * ulp) drift on a long run, so retain a
    # fail-closed time guard without rejecting an exact requested step count.
    time_tolerance = max(
        1.0e-10 * time_scale,
        8.0 * max(1, abs(taken)) * math.ulp(time_scale),
    )
    reached = (
        stepf == int(target["step"])
        and math.isclose(
            tf,
            step_consistent_time,
            rel_tol=0.0,
            abs_tol=time_tolerance,
        )
        and math.isclose(
            step_consistent_time,
            target_time,
            rel_tol=0.0,
            abs_tol=8.0 * math.ulp(time_scale),
        )
    )
    data["attained"] = {
        "final_time": tf,
        "final_step": stepf,
        "additional_time": elapsed,
        "additional_steps": taken,
        "target_reached": bool(reached),
    }
    return data


def _canonicalize_tc_quench_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a TC spec with every coupled local/native alias materialized.

    Direct run specs historically carried only ``Re``/``Rm`` while swept specs
    now also carry ``Re_h``/``Rm_h`` and explicit ``Re_TC``/``Rm_TC``. Quench
    comparison must canonicalize both sides before diffing, otherwise adding a
    truthful derived alias looks like a state-incompatible physics change.
    """

    if spec.get("geometry") != "taylor_couette":
        return spec
    try:
        from .physics import resolve_physics

        resolved = resolve_physics(spec)
    except (KeyError, TypeError, ValueError) as exc:
        raise QuenchError(f"invalid Taylor-Couette quench physics: {exc}") from exc

    data = copy.deepcopy(spec)
    groups = data["nondimensional_groups"]
    groups.update(
        {
            "nu": resolved.nu,
            "Re_h": resolved.Re_h,
            "Re_TC": resolved.Re_TC,
            "Re": resolved.Re_TC,
        }
    )
    if resolved.eta is not None:
        groups.update(
            {
                "eta_mag": resolved.eta,
                "Rm_h": resolved.Rm_h,
                "Rm_TC": resolved.Rm_TC,
                "Rm": resolved.Rm_TC,
                "Pm": resolved.Pm,
            }
        )
        groups.pop("eta", None)
    return data


def _validate_tc_coupled_changes(
    parent_spec: dict[str, Any], changed: dict[str, tuple[Any, Any]]
) -> None:
    """Require each TC coefficient and all of its aliases to move together."""

    if parent_spec.get("geometry") != "taylor_couette":
        return
    coupled_sets = (
        frozenset(
            {
                "nondimensional_groups.nu",
                "nondimensional_groups.Re",
                "nondimensional_groups.Re_h",
                "nondimensional_groups.Re_TC",
            }
        ),
        frozenset(
            {
                "nondimensional_groups.eta_mag",
                "nondimensional_groups.Rm",
                "nondimensional_groups.Rm_h",
                "nondimensional_groups.Rm_TC",
            }
        ),
    )
    changed_keys = changed.keys()
    for coupled in coupled_sets:
        touched = coupled & changed_keys
        if touched and touched != coupled:
            raise QuenchError(
                "Taylor-Couette quench controls are coupled; changing "
                f"{sorted(touched)} also requires {sorted(coupled - touched)}"
            )


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, full))
    else:
        out[prefix] = data
    return out


def validate_quench(
    parent_spec: dict[str, Any],
    child_spec: dict[str, Any],
    *,
    allow: frozenset[str] = MUTABLE_QUENCH_FIELDS,
) -> dict[str, Any]:
    """Validate a quench child against its parent; return the resolved diff.

    Raises :class:`QuenchError` if any field outside ``allow`` differs, if an
    immutable required field changed or is missing, or if the numerics contract
    versions differ.
    """

    parent_spec = _canonicalize_tc_quench_spec(parent_spec)
    child_spec = _canonicalize_tc_quench_spec(child_spec)
    parent = _flatten(parent_spec)
    child = _flatten(child_spec)

    for key in _IMMUTABLE_REQUIRED:
        pv = parent_spec.get(key)
        cv = child_spec.get(key)
        if pv is None or cv is None:
            raise QuenchError(f"quench requires immutable field {key!r} on both specs")
        if pv != cv:
            raise QuenchError(
                f"quench cannot change immutable field {key!r}: {pv!r} -> {cv!r}"
            )

    changed: dict[str, tuple[Any, Any]] = {}
    illegal: list[str] = []
    for key in set(parent) | set(child):
        if key in _IGNORED_FIELDS:
            continue
        if parent.get(key) != child.get(key):
            changed[key] = (parent.get(key), child.get(key))
            if key not in allow:
                illegal.append(key)

    if illegal:
        raise QuenchError(
            "quench may change only "
            f"{sorted(allow)}; illegal changes to {sorted(illegal)}"
        )
    if not changed:
        raise QuenchError(
            "quench child is identical to parent; use resume-exact instead"
        )
    _validate_tc_coupled_changes(parent_spec, changed)
    _validate_quench_direction(changed)
    return {
        "changed": changed,
        "mutable_allowlist": sorted(allow),
        "direction_policy": (
            "Re/Rm and local/native TC aliases nonincreasing; nu/eta nondecreasing"
        ),
    }


def _validate_quench_direction(changed: dict[str, tuple[Any, Any]]) -> None:
    """Reject continuations that move toward less dissipative parameters."""

    directional_keys = [
        *sorted(_QUENCH_DECREASING_GROUPS & changed.keys()),
        *sorted(_QUENCH_INCREASING_DIFFUSIVITIES & changed.keys()),
    ]
    for key in directional_keys:
        parent_value, child_value = changed[key]
        try:
            parent_number = float(parent_value)
            child_number = float(child_value)
        except (TypeError, ValueError) as exc:
            raise QuenchError(
                f"quench direction requires numeric values for {key!r}: "
                f"{parent_value!r} -> {child_value!r}"
            ) from exc
        if not (math.isfinite(parent_number) and math.isfinite(child_number)):
            raise QuenchError(
                f"quench direction requires finite values for {key!r}: "
                f"{parent_value!r} -> {child_value!r}"
            )
        if key in _QUENCH_DECREASING_GROUPS and child_number > parent_number:
            raise QuenchError(
                f"quench cannot increase {key.rsplit('.', 1)[-1]}: "
                f"{parent_number:g} -> {child_number:g}"
            )
        if key in _QUENCH_INCREASING_DIFFUSIVITIES and child_number < parent_number:
            raise QuenchError(
                f"quench cannot decrease {key.rsplit('.', 1)[-1]}: "
                f"{parent_number:g} -> {child_number:g}"
            )


def checkpoint_bank_entry(
    *,
    parent_run_id: str,
    child_run_id: str | None,
    t: float,
    tstep: int,
    spec_hash: str,
    representation: str,
    numerics_contract_version: int,
    checkpoint_path: str,
    plateau_stats: dict[str, Any] | None = None,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Build one checkpoint-bank manifest entry (FJ-05)."""

    stats = plateau_stats or {}
    qualified = stats.get("plateau_qualified") is True

    return {
        "parent_run_id": parent_run_id,
        "child_run_id": child_run_id,
        "state_time": float(t),
        "tstep": int(tstep),
        "spec_hash": spec_hash,
        "representation": representation,
        "numerics_contract_version": int(numerics_contract_version),
        "checkpoint_path": checkpoint_path,
        "file_sha256": file_sha256,
        "plateau_qualified": qualified,
        "selection_status": "eligible" if qualified else "quarantined",
        "plateau_window_stats": stats,
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def burn_in_horizon(*, tstep0: int, burn_in_steps: int) -> dict[str, Any]:
    """Return the post-quench burn-in window during which inherited growth/class
    history is quarantined (FJ-05)."""

    return {
        "burn_in_steps": int(burn_in_steps),
        "burn_in_until_tstep": int(tstep0) + int(burn_in_steps),
        "classification_valid_after_tstep": int(tstep0) + int(burn_in_steps),
    }


def stable_manifest_json(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, sort_keys=True, indent=2)
