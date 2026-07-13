"""Sweep-safe semantic override interface (FJ-07).

A campaign launches many runs from one base spec by applying *validated semantic
overrides* (compatible physics controls, domain periods, resolution, horizon, and
precision). Availability is geometry/physics/oracle-capability aware: for
example, a full-annulus Taylor-Couette run rejects Cartesian ``Ly`` instead of
silently storing an unused ``y_period``. Every override must either change the fully
*resolved* spec or fail validation. The materialized per-run spec (with derived
``nu``/``eta`` and local/native control numbers written back) is archived before
launch, keyed by a collision-resistant run id.

Magnetic wall families are deliberately not sweepable: each requires a separate
base spec carrying the matching problem, oracle, and golden-artifact identity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from .physics import resolve_physics
from .problem_spec import ProblemSpecError, load_spec, spec_hash, validate_spec

# Semantic override -> how it maps onto the spec.
# NOTE: "seed" is intentionally excluded -- the wired PCF saturation IC builders use
# fixed analytic eigenmodes and never read initial_condition.random_seed, so a seed
# override would only change the run id/hash without changing the trajectory (a
# silent relabel). Re-add it only once a solver actually consumes the seed.
_KNOWN_OVERRIDES = {
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

_STATIC_ORACLES = frozenset(
    {
        "plane_poiseuille_laminar",
        "plane_couette_laminar",
        "circular_couette_base_flow",
        "pcf_mhd_linear_conducting",
        "local_ideal_mri",
        "tc_mhd_linear_conducting",
        "tc_mhd_linear_insulating",
    }
)

_RESOLUTION_KEYS_BY_ORACLE: dict[str, frozenset[str]] = {
    "plane_poiseuille_laminar": frozenset({"nx", "ny", "nz", "family"}),
    "plane_couette_laminar": frozenset({"nx"}),
    "circular_couette_base_flow": frozenset({"N", "family"}),
    "pcf_mhd_linear_conducting": frozenset({"nx"}),
    "local_ideal_mri": frozenset({"nx"}),
    "tc_mhd_linear_conducting": frozenset({"N", "family"}),
    "tc_mhd_linear_insulating": frozenset({"N", "family"}),
    "pcf_hydro_dns_decay": frozenset({"Nx", "Nz", "family", "dealias"}),
    "pcf_mri_dns_growth": frozenset({"Nx", "Nz", "family", "dealias"}),
    "circular_couette_dns_growth": frozenset({"Nr", "Nz", "family"}),
    "tc_mri_dns_growth": frozenset({"Nr", "Nz", "family"}),
    "tc_hydro_saturation_ladder": frozenset({"Nr", "Nz", "family", "dealias"}),
}

_PCF_3D_ORACLES = frozenset({"gpu_generated_saturated_dns", "mri_saturation_ladder"})
_PCF_AXISYMMETRIC_ORACLES = frozenset({"pcf_hydro_dns_decay", "pcf_mri_dns_growth"})
_TC_NONLINEAR_ORACLES = frozenset(
    {
        "circular_couette_dns_growth",
        "tc_mri_dns_growth",
        "tc_hydro_saturation_ladder",
        "tc_mri_saturation_ladder",
    }
)
_RE_H_ORACLES = frozenset(
    {
        "plane_poiseuille_laminar",
        "plane_couette_laminar",
        "circular_couette_base_flow",
        "pcf_mhd_linear_conducting",
        "local_ideal_mri",
        "tc_mhd_linear_conducting",
        "tc_mhd_linear_insulating",
        "pcf_hydro_dns_decay",
        "pcf_mri_dns_growth",
        "circular_couette_dns_growth",
        "tc_mri_dns_growth",
        "tc_hydro_saturation_ladder",
        "gpu_generated_saturated_dns",
        "mri_saturation_ladder",
        "tc_mri_saturation_ladder",
    }
)
_RM_H_ORACLES = frozenset(
    {
        "pcf_mhd_linear_conducting",
        "local_ideal_mri",
        "tc_mhd_linear_conducting",
        "tc_mhd_linear_insulating",
        "pcf_mri_dns_growth",
        "tc_mri_dns_growth",
        "gpu_generated_saturated_dns",
        "mri_saturation_ladder",
        "tc_mri_saturation_ladder",
    }
)
_B0_ORACLES = _RM_H_ORACLES | {"local_ideal_mri"}


class SweepOverrideError(ProblemSpecError):
    """Raised for an unknown or invalid sweep override."""


def _oracle_type(spec: dict[str, Any]) -> str:
    return str(spec.get("expected_oracle", {}).get("type", ""))


def _resolution_keys_for_spec(spec: dict[str, Any]) -> frozenset[str]:
    oracle = _oracle_type(spec)
    if oracle in _RESOLUTION_KEYS_BY_ORACLE:
        return _RESOLUTION_KEYS_BY_ORACLE[oracle]
    if oracle == "gpu_generated_saturated_dns":
        return frozenset({"Nx", "Ny", "Nz", "family", "dealias"})
    if oracle == "mri_saturation_ladder":
        return frozenset({"Nx", "Ny", "Nz", "family", "dealias"})
    if oracle == "tc_mri_saturation_ladder":
        if spec.get("representation") == "vector_potential":
            return frozenset({"Nr", "Ntheta", "Nz", "family", "dealias"})
        return frozenset({"Nr", "Nz", "family", "dealias"})
    # Analytic pipe oracles do not allocate a spectral discretization.
    return frozenset()


def _has_consumed_resolution_control(spec: dict[str, Any]) -> bool:
    allowed = _resolution_keys_for_spec(spec)
    resolution = spec.get("resolution", {})
    if not isinstance(resolution, dict):
        return False
    if allowed & resolution.keys():
        return True
    return any(
        isinstance(resolution.get(tier), dict)
        and bool(allowed & resolution[tier].keys())
        for tier in ("smoke", "start", "production")
    )


def _consumes_ly(spec: dict[str, Any]) -> bool:
    oracle = _oracle_type(spec)
    return oracle == "plane_poiseuille_laminar" or (
        spec.get("geometry") == "pcf" and oracle in _PCF_3D_ORACLES
    )


def _consumes_lz(spec: dict[str, Any]) -> bool:
    oracle = _oracle_type(spec)
    if oracle in {"plane_poiseuille_laminar", "hagen_poiseuille", "pipe_womersley"}:
        return True
    if spec.get("geometry") == "pcf":
        return oracle in (_PCF_3D_ORACLES | _PCF_AXISYMMETRIC_ORACLES)
    if spec.get("geometry") == "taylor_couette":
        return oracle in _TC_NONLINEAR_ORACLES
    return False


def _validate_resolution_override(spec: dict[str, Any], value: Any) -> None:
    if not isinstance(value, dict):
        raise SweepOverrideError("resolution override must be an object")
    allowed = _resolution_keys_for_spec(spec)
    resolution = spec.get("resolution", {})
    if not allowed or not isinstance(resolution, dict):
        raise SweepOverrideError(
            f"selected oracle {_oracle_type(spec)!r} consumes no resolution override"
        )

    def validate_block(
        override: dict[str, Any],
        base: dict[str, Any],
        context: str,
        *,
        inherited: dict[str, Any] | None = None,
    ) -> None:
        available = base.keys() | (inherited.keys() if inherited is not None else set())
        for key in override:
            if key not in allowed or key not in available:
                raise SweepOverrideError(
                    f"resolution key {context}.{key} is ignored by selected oracle "
                    f"{_oracle_type(spec)!r}; consumed keys present in this block: "
                    f"{sorted(allowed & available)}"
                )

    tiers = {key for key in ("smoke", "start", "production") if key in resolution}
    for key, item in value.items():
        if key in tiers:
            if not isinstance(item, dict) or not isinstance(resolution[key], dict):
                raise SweepOverrideError(f"resolution.{key} override must be an object")
            validate_block(
                item,
                resolution[key],
                f"resolution.{key}",
                inherited=resolution,
            )
        else:
            if key in {"smoke", "start", "production"}:
                raise SweepOverrideError(
                    f"resolution tier {key!r} is not defined by the base spec"
                )
            validate_block({key: item}, resolution, "resolution")


def _validate_override_values(
    base_spec: dict[str, Any], overrides: dict[str, Any]
) -> None:
    if "resolution" in overrides:
        _validate_resolution_override(base_spec, overrides["resolution"])
    if "B0" in overrides:
        try:
            b0 = float(overrides["B0"])
        except (TypeError, ValueError) as exc:
            raise SweepOverrideError("B0 amplitude must be numeric") from exc
        if not math.isfinite(b0) or b0 < 0.0:
            raise SweepOverrideError("B0 amplitude must be finite and nonnegative")


def _rescale_component_vector(value: Any, amplitude: float) -> list[float]:
    components = [float(component) for component in value]
    if not all(math.isfinite(component) for component in components):
        raise SweepOverrideError("forcing.B0 components must be finite")
    norm = math.sqrt(sum(component * component for component in components))
    if amplitude == 0.0:
        return [0.0 for _ in components]
    if norm == 0.0:
        raise SweepOverrideError(
            "cannot apply a positive B0 amplitude to a zero component vector; "
            "state the intended direction in the base spec"
        )
    return [amplitude * component / norm for component in components]


def _rescaled_solver_components(
    groups: dict[str, Any], forcing_vector: Any | None, amplitude: float
) -> list[float] | None:
    """Rescale the imposed-field direction consumed by component-based solvers.

    The PCF linear families read ``By``/``Bz`` from nondimensional groups while
    the archived forcing records ``[Bx, By, Bz]``. When either representation is
    component-based, update both from one direction vector so metadata and the
    assembled operator cannot diverge.
    """

    component_keys = ("Bx", "By", "Bz")
    present = {key for key in component_keys if key in groups}
    if forcing_vector is not None:
        components = [float(component) for component in forcing_vector]
        if present and len(components) != len(component_keys):
            raise SweepOverrideError(
                "forcing.B0 must have three [Bx, By, Bz] components when the "
                "selected solver consumes component groups"
            )
        if present:
            for index, key in enumerate(component_keys):
                if key not in present and not math.isclose(
                    components[index], 0.0, rel_tol=0.0, abs_tol=1.0e-15
                ):
                    raise SweepOverrideError(
                        f"forcing.B0 component {key} is nonzero but is not consumed "
                        "by the selected solver"
                    )
        return _rescale_component_vector(components, amplitude)
    if present:
        components = [float(groups.get(key, 0.0)) for key in component_keys]
        return _rescale_component_vector(components, amplitude)
    return None


def supported_overrides_for_spec(base_spec: dict[str, Any]) -> frozenset[str]:
    """Return semantic overrides actually consumed by this solver family.

    This is intentionally conservative. A capability should be added only when
    the selected production path consumes the changed value; changing a JSON hash
    is not evidence that the PDE changed.
    """

    physics = str(base_spec.get("physics"))
    integrator = str(base_spec.get("time", {}).get("integrator"))
    oracle = _oracle_type(base_spec)
    allowed = {"precision"}
    if _has_consumed_resolution_control(base_spec):
        allowed.add("resolution")

    if oracle in _RE_H_ORACLES:
        allowed.add("Re_h")
    if _consumes_ly(base_spec):
        allowed.add("Ly")

    # Static eigensolvers consume explicit wavenumbers and no time horizon even
    # when a legacy spec happens to label its integrator as IMEX.
    is_time_dependent = oracle not in _STATIC_ORACLES and integrator not in {
        "analytic",
        "linear_eigenproblem",
    }
    if _consumes_lz(base_spec):
        allowed.add("Lz")
    if is_time_dependent:
        allowed.update({"dt", "horizon"})

    if physics in {"mhd", "mri"} and oracle in _RM_H_ORACLES:
        allowed.add("Rm_h")
    if physics in {"mhd", "mri"} and oracle in _B0_ORACLES:
        allowed.add("B0")
    return frozenset(allowed)


def _validate_override_capabilities(
    base_spec: dict[str, Any], overrides: dict[str, Any]
) -> None:
    unknown = set(overrides) - _KNOWN_OVERRIDES
    if unknown:
        raise SweepOverrideError(
            f"unknown sweep override(s) {sorted(unknown)}; "
            f"known: {sorted(_KNOWN_OVERRIDES)}"
        )
    if "bc" in overrides:
        raise SweepOverrideError(
            f"bc is not a sweepable control for selected oracle "
            f"{_oracle_type(base_spec)!r}; choose a separate base spec with its "
            "own problem_id, expected_oracle, and golden artifact"
        )
    supported = supported_overrides_for_spec(base_spec)
    unsupported = set(overrides) - supported
    if unsupported:
        raise SweepOverrideError(
            f"sweep override(s) {sorted(unsupported)} are not consumed by "
            f"geometry={base_spec.get('geometry')!r}, "
            f"physics={base_spec.get('physics')!r}, "
            f"integrator={base_spec.get('time', {}).get('integrator')!r}; "
            f"supported: {sorted(supported)}"
        )


def apply_overrides(
    base_spec: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """Return a validated, physics-resolved spec with ``overrides`` applied.

    Coefficient overrides (``Re_h``/``Rm_h``) drop the stale ``nu``/``eta``/``Pm`` so
    they are re-derived from the new group; the resolved values are then written back
    so the solver reads exactly what was reported. Raises on unknown overrides or on
    an override that yields an inconsistent spec.
    """

    _validate_override_capabilities(base_spec, overrides)
    _validate_override_values(base_spec, overrides)

    spec = copy.deepcopy(base_spec)
    groups = spec.setdefault("nondimensional_groups", {})
    is_tc = spec.get("geometry") == "taylor_couette"

    if "Re_h" in overrides:
        if is_tc:
            groups["Re_h"] = float(overrides["Re_h"])
            groups.pop("Re", None)
            groups.pop("Re_TC", None)
        else:
            groups["Re"] = float(overrides["Re_h"])
            groups.pop("Re_h", None)
        groups.pop("nu", None)
        groups.pop("Pm", None)
    if "Rm_h" in overrides:
        if is_tc:
            groups["Rm_h"] = float(overrides["Rm_h"])
            groups.pop("Rm", None)
            groups.pop("Rm_TC", None)
        else:
            groups["Rm"] = float(overrides["Rm_h"])
            groups.pop("Rm_h", None)
        groups.pop("eta_mag", None)
        groups.pop("eta", None)
        groups.pop("Pm", None)
    if "B0" in overrides:
        b0 = float(overrides["B0"])
        forcing = spec.get("forcing")
        forcing_vector = None
        if isinstance(forcing, dict) and isinstance(forcing.get("B0"), (list, tuple)):
            forcing_vector = forcing["B0"]
        components = _rescaled_solver_components(groups, forcing_vector, b0)
        groups["B0"] = b0
        if components is not None:
            for index, key in enumerate(("Bx", "By", "Bz")):
                if key in groups:
                    groups[key] = components[index]
        if isinstance(forcing, dict):
            if "B0" in forcing:
                if forcing_vector is not None:
                    forcing["B0"] = components
                else:
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
    if "precision" in overrides:
        spec["precision"] = str(overrides["precision"])
    if "resolution" in overrides:
        res = overrides["resolution"]
        resolution = spec.setdefault("resolution", {})
        for key, value in res.items():
            if key in {"smoke", "start", "production"}:
                resolution[key].update(value)
            else:
                resolution[key] = value

    # Re-derive nu/eta from the (possibly changed) groups and write them back so the
    # solver and the reported nondimensional numbers cannot drift.
    if spec.get("geometry") in {"pcf", "channel", "taylor_couette"} and any(
        groups.get(key) is not None for key in ("nu", "Re", "Re_h", "Re_TC")
    ):
        resolved = resolve_physics(spec)
        groups["nu"] = resolved.nu
        if resolved.eta is not None:
            groups["eta_mag"] = resolved.eta
        if resolved.Pm is not None:
            groups["Pm"] = resolved.Pm
        if is_tc:
            groups["Re_h"] = resolved.Re_h
            groups["Re_TC"] = resolved.Re_TC
            # ``Re``/``Rm`` remain compatibility aliases for the historical TC
            # native convention; the explicit names remove any ambiguity.
            groups["Re"] = resolved.Re_TC
            if resolved.Rm_h is not None:
                groups["Rm_h"] = resolved.Rm_h
                groups["Rm_TC"] = resolved.Rm_TC
                groups["Rm"] = resolved.Rm_TC

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


def cartesian_grid(
    grid: dict[str, list[Any]], *, base_spec: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """All override combinations of a ``{key: [values, ...]}`` grid, stably ordered."""

    import itertools

    if not grid:
        return []
    keys = sorted(grid)
    for key in keys:
        if key not in _KNOWN_OVERRIDES:
            raise SweepOverrideError(
                f"unknown sweep override {key!r}; supported: {sorted(_KNOWN_OVERRIDES)}"
            )
        if not isinstance(grid[key], (list, tuple)) or not grid[key]:
            raise SweepOverrideError(f"grid axis {key!r} must be a non-empty list")
    points = [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(grid[key] for key in keys))
    ]
    if base_spec is not None:
        _validate_override_capabilities(base_spec, {key: None for key in keys})
    return points


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

    base_spec = load_spec(base_spec_path)
    points = cartesian_grid(grid, base_spec=base_spec)
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
    for overrides in points:
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
        "points": len(points),
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
