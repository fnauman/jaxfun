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
# NOTE: "seed" is intentionally excluded -- the wired PCF saturation IC builders use
# fixed analytic eigenmodes and never read initial_condition.random_seed, so a seed
# override would only change the run id/hash without changing the trajectory (a
# silent relabel). Re-add it only once a solver actually consumes the seed.
_SUPPORTED_OVERRIDES = {
    "Re_h",
    "Rm_h",
    "B0",
    "Ly",
    "Lz",
    "horizon",
    "dt",
    "bc",
    "precision",
    "resolution",
}


class SweepOverrideError(ProblemSpecError):
    """Raised for an unknown or invalid sweep override."""


def apply_overrides(
    base_spec: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
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
        # Some specs (e.g. the ideal-MRI shearbox) express the imposed field via the
        # vertical component Bz that the linear oracle reads directly; keep it in sync
        # so a B0 sweep cannot silently run the unchanged field.
        if "Bz" in groups:
            groups["Bz"] = b0
        forcing = spec.get("forcing")
        if isinstance(forcing, dict):
            if "B0" in forcing and not isinstance(forcing["B0"], (list, tuple)):
                forcing["B0"] = b0
            if "bz" in forcing and not isinstance(forcing["bz"], (list, tuple)):
                forcing["bz"] = b0
    if "Ly" in overrides:
        spec["domain"]["y_period"] = float(overrides["Ly"])
    if "Lz" in overrides:
        spec["domain"]["z_period"] = float(overrides["Lz"])
    if "horizon" in overrides:
        spec["time"]["final_time"] = float(overrides["horizon"])
    if "dt" in overrides:
        spec["time"]["dt"] = float(overrides["dt"])
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


def cartesian_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """All override combinations of a ``{key: [values, ...]}`` grid, stably ordered."""

    import itertools

    if not grid:
        return []
    keys = sorted(grid)
    for key in keys:
        if key not in _SUPPORTED_OVERRIDES:
            raise SweepOverrideError(
                f"unknown sweep override {key!r}; supported: "
                f"{sorted(_SUPPORTED_OVERRIDES)}"
            )
        if not isinstance(grid[key], (list, tuple)) or not grid[key]:
            raise SweepOverrideError(f"grid axis {key!r} must be a non-empty list")
    return [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(grid[key] for key in keys))
    ]


def execute_sweep(
    base_spec_path: str | Path,
    grid: dict[str, list[Any]],
    out_dir: str | Path,
    *,
    execute: bool = False,
    runner: Any | None = None,
    resolution_tier: str | None = None,
    steps: int | None = None,
    wandb: bool = False,
    skip_completed: bool = True,
) -> dict[str, Any]:
    """Cartesian sweep lifecycle: materialize every grid point, optionally run it.

    Each point is materialized (validated + archived spec keyed by run id) and,
    with ``execute=True``, run through ``production.run_problem.run_problem``
    serially; per-point status lands in ``sweep_index.json`` after every point
    (atomic rewrite), so a killed campaign resumes where it stopped. With
    ``skip_completed`` a re-invocation -- including one with a widened grid --
    skips already-completed run ids: the basic continuation/frontier workflow.
    A failing point is recorded and does not abort the remaining grid.
    Adaptive refinement (choosing the next points from results) is tracked in
    production/KNOWN_ISSUES.md.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    index_path = out_path / "sweep_index.json"
    entries: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        for item in json.loads(index_path.read_text(encoding="utf-8")):
            entries[str(item.get("run_id"))] = item

    def flush() -> None:
        ordered = [entries[key] for key in sorted(entries)]
        tmp = index_path.with_name(f".{index_path.name}.tmp")
        tmp.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", "utf-8")
        tmp.replace(index_path)

    if runner is None and execute:
        from .run_problem import run_problem as runner  # type: ignore[no-redef]

    completed = failed = skipped = 0
    for overrides in cartesian_grid(grid):
        record = materialize_run_spec(base_spec_path, overrides, out_path)
        run_id = record["run_id"]
        prior = entries.get(run_id)
        if skip_completed and prior is not None and prior.get("status") == "completed":
            skipped += 1
            continue
        entry = {**record, "status": "materialized"}
        entries[run_id] = entry
        flush()
        if not execute:
            continue
        try:
            runner(
                config_path=record["spec_path"],
                out=out_path / run_id / "run",
                resolution_tier=resolution_tier,
                steps=steps,
                wandb=wandb,
            )
        except Exception as exc:  # record and continue with the rest of the grid
            entry["status"] = "failed"
            entry["failure_reason"] = f"{type(exc).__name__}: {exc}"
            failed += 1
        else:
            entry["status"] = "completed"
            completed += 1
        flush()
    return {
        "points": len(cartesian_grid(grid)),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "index_path": str(index_path),
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
    parser = argparse.ArgumentParser(
        description="Materialize (and optionally execute) swept run specs (FJ-07)."
    )
    parser.add_argument("--base", required=True, help="Base spec JSON path.")
    parser.add_argument("--out", required=True, help="Output directory for run specs.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Semantic override, e.g. --set Re_h=1600 --set B0=0.0125",
    )
    parser.add_argument(
        "--grid",
        default=None,
        help='Cartesian grid as JSON, e.g. \'{"Rm_h": [400, 800], "B0": [0.025]}\'. '
        "Materializes every combination; with --execute, runs each point "
        "serially and records per-point status in sweep_index.json "
        "(re-invocation skips completed points).",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resolution-tier", choices=["smoke", "start", "production"], default=None
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)

    if args.grid is not None:
        grid = json.loads(args.grid)
        if not isinstance(grid, dict):
            raise SweepOverrideError("--grid must be a JSON object of lists")
        summary = execute_sweep(
            args.base,
            grid,
            args.out,
            execute=args.execute,
            resolution_tier=args.resolution_tier,
            steps=args.steps,
            wandb=args.wandb,
        )
        print(json.dumps(summary, indent=2))
        return 1 if summary["failed"] else 0

    overrides = dict(_parse_override(token) for token in args.set)
    record = materialize_run_spec(args.base, overrides, args.out)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
