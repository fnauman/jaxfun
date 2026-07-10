"""Explicit quench / continue-from validation and checkpoint banks (FJ-05).

Strict same-spec resume (``resume-exact``) requires an identical ``spec_hash``. A
*quench* is a separate, explicit continuation that lowers ``Rm`` (or ``Re``) from a
selected plateau state while keeping everything else immutable. This module defines
the mutable-field allowlist and validates that a child spec differs from its parent
only within that allowlist, and it builds a checkpoint-bank manifest entry recording
provenance for each banked plateau state.
"""

from __future__ import annotations

import hashlib
import json
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


class QuenchError(ValueError):
    """Raised when a quench continuation violates the immutability contract."""


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
    return {"changed": changed, "mutable_allowlist": sorted(allow)}


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
        "plateau_window_stats": plateau_stats or {},
    }


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def burn_in_horizon(
    *, tstep0: int, burn_in_steps: int
) -> dict[str, Any]:
    """Return the post-quench burn-in window during which inherited growth/class
    history is quarantined (FJ-05)."""

    return {
        "burn_in_steps": int(burn_in_steps),
        "burn_in_until_tstep": int(tstep0) + int(burn_in_steps),
        "classification_valid_after_tstep": int(tstep0) + int(burn_in_steps),
    }


def stable_manifest_json(entries: list[dict[str, Any]]) -> str:
    return json.dumps(entries, sort_keys=True, indent=2)
