"""Sweep-safe semantic override interface (FJ-07).

A campaign launches many runs from one base spec by applying *validated semantic
overrides* (``Re_h``, ``Rm_h``, ``B0``, ``Ly``, ``Lz``, resolution, BC, horizon,
precision, seed). Every override must either change the fully *resolved* spec or fail
validation -- a sweep can never silently relabel identical physics. The materialized
per-run spec (with derived ``nu``/``eta`` written back so the solver and diagnostics
share one source) is archived before launch, keyed by a collision-resistant run id.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .physics import resolve_physics
from .problem_spec import ProblemSpecError, load_spec, spec_hash, validate_spec

# Semantic override -> how it maps onto the spec.
_SUPPORTED_OVERRIDES = {
    "Re_h", "Rm_h", "B0", "Ly", "Lz", "horizon", "dt", "seed", "bc",
    "precision", "resolution",
}


class SweepOverrideError(ProblemSpecError):
    """Raised for an unknown or invalid sweep override."""


def apply_overrides(base_spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a validated, physics-resolved spec with ``overrides`` applied.

    Coefficient overrides (``Re_h``/``Rm_h``) drop the stale ``nu``/``eta``/``Pm`` so
    they are re-derived from the new group; the resolved values are then written back
    so the solver reads exactly what was reported. Raises on unknown overrides or on
    an override that yields an inconsistent spec.
    """

    unknown = set(overrides) - _SUPPORTED_OVERRIDES
    if unknown:
        raise SweepOverrideError(
            f"unknown sweep override(s) {sorted(unknown)}; "
            f"supported: {sorted(_SUPPORTED_OVERRIDES)}"
        )

    spec = copy.deepcopy(base_spec)
    groups = spec.setdefault("nondimensional_groups", {})

    if "Re_h" in overrides:
        groups["Re"] = float(overrides["Re_h"])
        groups.pop("nu", None)
        groups.pop("Pm", None)
    if "Rm_h" in overrides:
        groups["Rm"] = float(overrides["Rm_h"])
        groups.pop("eta_mag", None)
        groups.pop("eta", None)
        groups.pop("Pm", None)
    if "B0" in overrides:
        b0 = float(overrides["B0"])
        groups["B0"] = b0
        forcing = spec.get("forcing")
        if isinstance(forcing, dict) and "B0" in forcing and not isinstance(
            forcing["B0"], (list, tuple)
        ):
            forcing["B0"] = b0
    if "Ly" in overrides:
        spec["domain"]["y_period"] = float(overrides["Ly"])
    if "Lz" in overrides:
        spec["domain"]["z_period"] = float(overrides["Lz"])
    if "horizon" in overrides:
        spec["time"]["final_time"] = float(overrides["horizon"])
    if "dt" in overrides:
        spec["time"]["dt"] = float(overrides["dt"])
    if "seed" in overrides:
        spec.setdefault("initial_condition", {})["random_seed"] = overrides["seed"]
    if "bc" in overrides:
        magnetic = spec.setdefault("boundary_conditions", {}).setdefault("magnetic", {})
        if isinstance(magnetic, dict):
            magnetic["type"] = str(overrides["bc"])
        else:
            spec["boundary_conditions"]["magnetic"] = {"type": str(overrides["bc"])}
    if "precision" in overrides:
        spec["precision"] = str(overrides["precision"])
    if "resolution" in overrides:
        res = overrides["resolution"]
        if not isinstance(res, dict):
            raise SweepOverrideError("resolution override must be an object")
        spec.setdefault("resolution", {}).update(res)

    # Re-derive nu/eta from the (possibly changed) groups and write them back so the
    # solver and the reported nondimensional numbers cannot drift.
    if spec.get("geometry") in {"pcf", "channel"} and (
        groups.get("nu") is not None or groups.get("Re") is not None
    ):
        resolved = resolve_physics(spec)
        groups["nu"] = resolved.nu
        if resolved.eta is not None:
            groups["eta_mag"] = resolved.eta
        if resolved.Pm is not None:
            groups["Pm"] = resolved.Pm

    return validate_spec(spec)  # raises ProblemSpecError on any inconsistency


def run_id_for(spec: dict[str, Any]) -> str:
    """Collision-resistant run id grouped by problem_id + resolved spec hash."""

    return f"{spec['problem_id']}-{spec_hash(spec)[:12]}"


def materialize_run_spec(
    base_spec_path: str | Path,
    overrides: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Materialize + archive a per-run spec; return {run_id, spec_path, spec_hash}."""

    base = load_spec(base_spec_path)
    resolved = apply_overrides(base, overrides)
    run_id = run_id_for(resolved)
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    spec_path.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "spec_path": str(spec_path),
        "spec_hash": resolved["spec_hash"],
        "overrides": overrides,
    }


def _parse_override(token: str) -> tuple[str, Any]:
    if "=" not in token:
        raise SweepOverrideError(f"override {token!r} must be key=value")
    key, _, raw = token.partition("=")
    key = key.strip()
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize a swept run spec (FJ-07).")
    parser.add_argument("--base", required=True, help="Base spec JSON path.")
    parser.add_argument("--out", required=True, help="Output directory for run specs.")
    parser.add_argument(
        "--set", action="append", default=[], metavar="key=value",
        help="Semantic override, e.g. --set Re_h=1600 --set B0=0.0125",
    )
    args = parser.parse_args(argv)
    overrides = dict(_parse_override(token) for token in args.set)
    record = materialize_run_spec(args.base, overrides, args.out)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
