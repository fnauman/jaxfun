"""Immutable production-release promotion contract (issue #25).

A production run is not promotable merely because its final row looks healthy.
This module audits the complete persisted horizon, verifies that the run came from
the exact commit pinned by a matching remote tag, checks the dependency lock and
real test reports, and builds a non-overwritable evidence bundle with content hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import tomllib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .health import CFL_LIMIT, MODE_OCCUPANCY_LIMIT, SPECTRAL_TAIL_LIMIT
from .provenance import _head_release_ref, capture_provenance
from .release_check import CONVENTIONS, _annotate_test_summary

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_VERSION = "production-promotion-v1"
_BUDGET_RESIDUAL_LIMIT = 1.0e-2
_GIT_COMMIT_RE = re.compile(r"#[0-9a-fA-F]{40}$")


class PromotionError(RuntimeError):
    """Raised when release evidence does not satisfy the promotion contract."""


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PromotionError(f"required evidence is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PromotionError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"JSON evidence must be an object: {path}")
    return value


def _read_diagnostics(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise PromotionError(f"required evidence is missing: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromotionError(
                f"invalid diagnostics row {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise PromotionError(
                f"diagnostics row {path}:{line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise PromotionError(f"diagnostics evidence is empty: {path}")
    return rows


def _fail(problems: Iterable[str]) -> None:
    items = list(problems)
    if items:
        raise PromotionError("promotion refused:\n- " + "\n- ".join(items))


def audit_dependency_lock(repo_root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Require a complete uv lock whose remote git sources pin full commits."""

    lock_path = repo_root / "uv.lock"
    project_path = repo_root / "pyproject.toml"
    problems: list[str] = []
    for path in (lock_path, project_path):
        if not path.is_file():
            problems.append(f"missing dependency contract {path.name}")
    _fail(problems)

    try:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PromotionError(f"cannot parse uv.lock: {exc}") from exc

    packages = lock.get("package")
    if not isinstance(packages, list) or not packages:
        raise PromotionError("uv.lock has no locked packages")

    git_sources: list[dict[str, str]] = []
    for package in packages:
        if not isinstance(package, dict):
            problems.append("uv.lock contains a non-object package entry")
            continue
        name = str(package.get("name", "<unnamed>"))
        source = package.get("source") or {}
        if not isinstance(source, dict):
            problems.append(f"{name}: invalid lock source")
            continue
        git_source = source.get("git")
        if git_source is not None:
            value = str(git_source)
            if _GIT_COMMIT_RE.search(value) is None:
                problems.append(
                    f"{name}: remote git source is not pinned to a full commit"
                )
            else:
                git_sources.append({"package": name, "source": value})
        elif "registry" in source and not package.get("version"):
            problems.append(f"{name}: registry dependency has no locked version")
    _fail(problems)
    return {
        "lock_format_version": lock.get("version"),
        "package_count": len(packages),
        "remote_git_sources": git_sources,
        "files": {
            "uv.lock": _fingerprint(lock_path),
            "pyproject.toml": _fingerprint(project_path),
        },
    }


def audit_test_summary(path: Path) -> dict[str, Any]:
    """Require a passing summary whose live-Shenfun tier ran without skips."""

    summary = _read_json(path)
    problems: list[str] = []
    for key in ("failed", "errors"):
        value = summary.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"test summary {key!r} must be an integer")
    live = summary.get("live_shenfun")
    if not isinstance(live, dict):
        problems.append("test summary must contain a live_shenfun object")
    else:
        passed = live.get("passed")
        skipped = live.get("skipped")
        if not isinstance(passed, int) or isinstance(passed, bool) or passed <= 0:
            problems.append("live_shenfun.passed must be a positive integer")
        if not isinstance(skipped, int) or isinstance(skipped, bool) or skipped != 0:
            problems.append("live_shenfun.skipped must be zero")
    _fail(problems)

    annotated = _annotate_test_summary(summary)
    if annotated.get("release_test_gate_passed") is not True:
        raise PromotionError("test summary reports failures, errors, or skipped parity")
    return annotated


def _numeric_values(row: dict[str, Any]) -> Iterable[tuple[str, float]]:
    for key, value in row.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            yield str(key), float(value)


def _constraint_values(row: dict[str, Any]) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for key, value in _numeric_values(row):
        name = key.lower()
        is_constraint = (
            "divergence" in name
            or re.match(r"^div[_-]?[ub]?(?:[_-]|$)", name) is not None
            or name.startswith("continuity")
            or name.endswith("_bc_residual")
            or name.endswith("_constraint_residual")
        )
        if is_constraint:
            values.append((key, abs(value)))
    return values


def audit_run(run_dir: Path) -> dict[str, Any]:
    """Audit complete run evidence, never only the final diagnostics row."""

    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "metadata.json")
    spec = _read_json(run_dir / "spec.json")
    rows = _read_diagnostics(run_dir / "diagnostics.jsonl")
    golden = _read_json(run_dir / "golden" / "golden.json")
    problems: list[str] = []

    if (metadata.get("execution") or {}).get("status") != "completed":
        problems.append("execution.status is not completed")
    scope = metadata.get("validation_scope") or {}
    if scope.get("kind") != "generated_saturated_golden":
        problems.append("validation scope is not a full generated saturation run")
    if scope.get("bounded_smoke") is not False:
        problems.append("bounded/smoke evidence cannot be promoted")
    if metadata.get("spec_hash") != spec.get("spec_hash"):
        problems.append("metadata and archived spec hashes disagree")
    if golden.get("spec_hash") != metadata.get("spec_hash"):
        problems.append("golden and run spec hashes disagree")
    if golden.get("artifact_id") != metadata.get("artifact_id"):
        problems.append("golden and run artifact IDs disagree")
    if golden.get("problem_id") != metadata.get("problem_id"):
        problems.append("golden and run problem IDs disagree")

    release_gate = (metadata.get("provenance") or {}).get("release_gate") or {}
    if release_gate.get("passed") is not True:
        problems.append("the run did not pass its strict release gate")
    run_ref = release_gate.get("release_ref") or {}
    if run_ref.get("remote_verified") is not True:
        problems.append("the run is not tied to a verified remote tag")

    floor = metadata.get("validation_floor") or {}
    if floor.get("required") is not True or floor.get("passed") is not True:
        problems.append("validation floor did not pass")
    constraint_limit = floor.get("divergence_limit")
    if (
        not isinstance(constraint_limit, int | float)
        or isinstance(constraint_limit, bool)
        or not math.isfinite(float(constraint_limit))
        or float(constraint_limit) <= 0.0
    ):
        problems.append("validation floor has no finite positive constraint limit")
        constraint_limit = 0.0

    saturation = metadata.get("saturation_checks") or {}
    if not (
        saturation.get("required") is True
        and saturation.get("present") is True
        and saturation.get("type_valid") is True
        and saturation.get("passed") is True
        and saturation.get("stationarity_check_passed") is True
    ):
        problems.append("saturation and stationarity checks did not pass")

    classification = metadata.get("classification") or {}
    fit = classification.get("fit") or {}
    if classification.get("scientific_class") != "sustained":
        problems.append("scientific classification is not sustained")
    if classification.get("underresolved") is not False:
        problems.append("classification does not explicitly clear underresolution")
    if classification.get("stationary") is not True:
        problems.append("classification is not stationary")
    if classification.get("persistent_stress") is not True:
        problems.append("classification lacks persistent stress")
    if classification.get("independently_sampled") is not True:
        problems.append("classification lacks enough independent samples")
    if not isinstance(fit.get("samples"), int) or fit.get("samples", 0) < 4:
        problems.append("classification fit has fewer than four samples")

    times: list[float] = []
    constraint_maxima: dict[str, float] = {}
    for index, row in enumerate(rows):
        nonfinite = [
            key for key, value in _numeric_values(row) if not math.isfinite(value)
        ]
        if nonfinite:
            problems.append(
                f"diagnostics row {index} has nonfinite values: {', '.join(nonfinite)}"
            )
        value = row.get("t")
        if isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(f"diagnostics row {index} has no numeric time")
        else:
            times.append(float(value))
        constraints = _constraint_values(row)
        if not constraints:
            problems.append(f"diagnostics row {index} has no constraint evidence")
        for key, magnitude in constraints:
            constraint_maxima[key] = max(constraint_maxima.get(key, 0.0), magnitude)
            if not math.isfinite(magnitude) or magnitude > float(constraint_limit):
                problems.append(
                    f"diagnostics row {index} violates {key}: "
                    f"{magnitude:g} > {float(constraint_limit):g}"
                )

    if len(times) != len(rows):
        pass
    elif len(times) < 2:
        problems.append("whole-horizon evidence requires at least two cadence rows")
    else:
        if any(
            second <= first for first, second in zip(times, times[1:], strict=False)
        ):
            problems.append("diagnostic times are not strictly increasing")
        time_spec = spec.get("time") or {}
        final_time = time_spec.get("final_time")
        dt = time_spec.get("dt")
        if not isinstance(final_time, int | float) or not isinstance(dt, int | float):
            problems.append("spec has no numeric final_time and dt")
        else:
            tolerance = max(abs(float(dt)), 1.0e-12)
            if times[0] > tolerance:
                problems.append("diagnostics do not start at the initial horizon")
            if abs(times[-1] - float(final_time)) > tolerance:
                problems.append(
                    f"diagnostics end at t={times[-1]:g}, "
                    f"expected {float(final_time):g}"
                )

    final = rows[-1]
    health_limits = {
        "cfl_total": CFL_LIMIT,
        "spectral_tail_max": SPECTRAL_TAIL_LIMIT,
        "mode_occupancy": MODE_OCCUPANCY_LIMIT,
    }
    health_values: dict[str, float] = {}
    for key, limit in health_limits.items():
        value = final.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(f"whole-horizon health aggregate {key} is missing")
            continue
        number = float(value)
        health_values[key] = number
        if not math.isfinite(number) or number > limit:
            problems.append(f"health aggregate {key}={number:g} exceeds {limit:g}")

    budget = final.get("energy_budget_residual")
    if isinstance(budget, bool) or not isinstance(budget, int | float):
        problems.append("energy_budget_residual is missing")
        budget_value = None
    else:
        budget_value = float(budget)
        if not math.isfinite(budget_value) or budget_value > _BUDGET_RESIDUAL_LIMIT:
            problems.append(
                f"energy budget residual {budget_value:g} exceeds "
                f"{_BUDGET_RESIDUAL_LIMIT:g}"
            )

    _fail(problems)
    return {
        "problem_id": metadata.get("problem_id"),
        "artifact_id": metadata.get("artifact_id"),
        "spec_hash": metadata.get("spec_hash"),
        "row_count": len(rows),
        "horizon": {"start": times[0], "end": times[-1]},
        "constraint_maxima": constraint_maxima,
        "health_maxima": health_values,
        "energy_budget_residual": budget_value,
        "classification": classification,
    }


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _fingerprint(destination)


def build_release_bundle(
    *,
    run_dir: Path,
    test_summary: Path,
    test_artifacts: Iterable[Path],
    out_dir: Path,
    remote: str = "origin",
    repo_root: Path = _REPO_ROOT,
) -> Path:
    """Build a content-addressed, non-overwritable production evidence bundle."""

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise PromotionError(
            f"release output already exists and is immutable: {out_dir}"
        )

    artifacts = [Path(path) for path in test_artifacts]
    if not artifacts:
        raise PromotionError("at least one actual test artifact is required")
    missing = [str(path) for path in artifacts if not path.is_file()]
    _fail(f"test artifact is missing: {path}" for path in missing)
    empty = [str(path) for path in artifacts if path.stat().st_size == 0]
    _fail(
        f"test artifact is empty and is not execution evidence: {path}"
        for path in empty
    )
    dependency_audit = audit_dependency_lock(repo_root)
    tests = audit_test_summary(Path(test_summary))
    run_audit = audit_run(Path(run_dir))
    release_ref = _head_release_ref(remote)
    provenance = capture_provenance()
    problems: list[str] = []
    if release_ref.get("remote_verified") is not True:
        problems.append(
            "current HEAD is not pinned by the same exact tag on the remote"
        )
    if provenance.get("dirty"):
        problems.append("current worktree is dirty")
    if provenance.get("unpushed_commits"):
        problems.append("current checkout has unpushed commits")

    metadata = _read_json(Path(run_dir) / "metadata.json")
    run_provenance = metadata.get("provenance") or {}
    if run_provenance.get("commit") != provenance.get("commit"):
        problems.append("run commit differs from the release commit")
    if run_provenance.get("lockfile_sha256") != {
        key: value["sha256"] for key, value in dependency_audit["files"].items()
    }:
        problems.append("run dependency hashes differ from the release checkout")
    run_ref = (run_provenance.get("release_gate") or {}).get("release_ref") or {}
    if run_ref.get("remote_tag_commit") != release_ref.get("remote_tag_commit"):
        problems.append("run and release remote tags resolve to different commits")
    _fail(problems)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
    files: dict[str, dict[str, Any]] = {}
    try:
        required_run_files = (
            Path(run_dir) / "metadata.json",
            Path(run_dir) / "spec.json",
            Path(run_dir) / "diagnostics.jsonl",
            Path(run_dir) / "golden" / "golden.json",
        )
        for source in required_run_files:
            if not source.is_file():
                raise PromotionError(f"required run artifact is missing: {source}")
            relative = Path("run") / source.relative_to(Path(run_dir))
            files[str(relative)] = _copy_file(source, stage / relative)

        for name in ("uv.lock", "pyproject.toml"):
            source = repo_root / name
            relative = Path("dependencies") / name
            files[str(relative)] = _copy_file(source, stage / relative)

        summary_relative = Path("tests") / "summary.json"
        files[str(summary_relative)] = _copy_file(
            Path(test_summary), stage / summary_relative
        )
        for index, source in enumerate(artifacts, start=1):
            relative = Path("tests") / f"{index:03d}-{source.name}"
            files[str(relative)] = _copy_file(source, stage / relative)

        identity_payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
        release_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        manifest = {
            "schema_version": 1,
            "contract_version": _CONTRACT_VERSION,
            "release_id": release_id,
            "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "release_ref": release_ref,
            "provenance": provenance,
            "dependencies": dependency_audit,
            "tests": tests,
            "run_audit": run_audit,
            "conventions": CONVENTIONS,
            "files": files,
        }
        (stage / "release.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage.replace(out_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return out_dir


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit and build an immutable production release bundle."
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--test-summary", required=True, type=Path)
    parser.add_argument(
        "--test-artifact",
        required=True,
        action="append",
        type=Path,
        help="actual JUnit/log/coverage artifact to archive (repeatable)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args(argv)
    try:
        written = build_release_bundle(
            run_dir=args.run,
            test_summary=args.test_summary,
            test_artifacts=args.test_artifact,
            out_dir=args.out,
            remote=args.remote,
        )
    except PromotionError as exc:
        parser.error(str(exc))
    print(f"wrote immutable production release -> {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
