"""Build machine-readable summaries for production validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def record_from_metadata(
    metadata: dict[str, Any], *, metadata_path: str | Path | None = None
) -> dict[str, Any]:
    execution = metadata.get("execution", {})
    device = metadata.get("device", {})
    status = execution.get("status", "unknown")
    solver_wired = bool(execution.get("solver_execution_wired", False))
    if not solver_wired and status in {"validated", "not_started"}:
        outcome = "skipped"
        reason = "metadata validation only; solver execution was not requested"
    elif status == "completed":
        outcome = "passed"
        reason = ""
    else:
        outcome = "failed"
        reason = f"execution status {status!r}"

    expected_oracle = metadata.get("expected_oracle", {})
    validation_scope = metadata.get("validation_scope", {})
    validation_scope_kind = validation_scope.get("kind")
    validation_scope_reason = validation_scope.get("reason", "")
    if (
        status == "completed"
        and not reason
        and validation_scope_kind
        in {"cpu_smoke_finiteness_divergence_only", "bounded_saturation_smoke"}
    ):
        reason = validation_scope_reason
    timing = metadata.get("timing", {})
    return {
        "problem_id": metadata.get("problem_id"),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "out_dir": metadata.get("out_dir"),
        "geometry": metadata.get("geometry"),
        "physics": metadata.get("physics"),
        "mode": device.get("mode"),
        "backend": device.get("default_backend"),
        "degraded": device.get("degraded"),
        "execution_status": status,
        "solver_execution_wired": solver_wired,
        "outcome": outcome,
        "reason": reason,
        "fallback_rungs": expected_oracle.get("fallback_rungs", []),
        "validation_scope": validation_scope_kind,
        "validation_scope_reason": validation_scope_reason,
        "checked_observables": validation_scope.get("checked_observables", []),
        "golden_resolution": metadata.get("golden_resolution", {}),
        "observables_compared": metadata.get("observables_compared", []),
        "comparisons": metadata.get("comparisons", []),
        "solver_wall_time_seconds": timing.get("solver_wall_time_seconds"),
        "solver_started_at_utc": timing.get("solver_started_at_utc"),
        "solver_finished_at_utc": timing.get("solver_finished_at_utc"),
    }


def skipped_record(
    problem_id: str,
    reason: str,
    *,
    geometry: str | None = None,
    physics: str | None = None,
) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "metadata_path": None,
        "out_dir": None,
        "geometry": geometry,
        "physics": physics,
        "mode": None,
        "backend": None,
        "degraded": None,
        "execution_status": "skipped",
        "solver_execution_wired": False,
        "outcome": "skipped",
        "reason": reason,
        "fallback_rungs": [],
        "validation_scope": None,
        "validation_scope_reason": "",
        "checked_observables": [],
        "golden_resolution": {},
        "observables_compared": [],
        "comparisons": [],
        "solver_wall_time_seconds": None,
        "solver_started_at_utc": None,
        "solver_finished_at_utc": None,
    }


def build_report(
    metadata_paths: list[str | Path],
    *,
    skipped_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = [
        record_from_metadata(load_metadata(path), metadata_path=path)
        for path in metadata_paths
    ]
    records.extend(skipped_records or [])
    summary = {
        "passed": sum(1 for record in records if record["outcome"] == "passed"),
        "failed": sum(1 for record in records if record["outcome"] == "failed"),
        "skipped": sum(1 for record in records if record["outcome"] == "skipped"),
    }
    return {"schema_version": 1, "summary": summary, "runs": records}


def find_metadata_files(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    return sorted(
        path for path in root.glob("*/*/metadata.json") if "_report" not in path.parts
    )


def write_report(
    metadata_paths: list[str | Path],
    out_dir: str | Path,
    *,
    skipped_records: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(metadata_paths, skipped_records=skipped_records)
    json_path = out / "results.json"
    md_path = out / "results.md"
    json_path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Production Validation Report",
        "",
        "Passed: {passed}  Failed: {failed}  Skipped: {skipped}".format(
            passed=summary["passed"],
            failed=summary["failed"],
            skipped=summary["skipped"],
        ),
        "",
        (
            "| problem_id | outcome | mode | backend | wall_time_s | "
            "fallback_rungs | validation_scope | observables | reason |"
        ),
        "|---|---|---|---|---:|---|---|---|---|",
    ]
    for record in report["runs"]:
        lines.append(
            (
                "| {problem_id} | {outcome} | {mode} | {backend} | "
                "{wall_time} | {fallback_rungs} | {validation_scope} | "
                "{observables} | {reason} |"
            ).format(
                problem_id=_markdown_cell(record.get("problem_id")),
                outcome=_markdown_cell(record.get("outcome")),
                mode=_markdown_cell(record.get("mode")),
                backend=_markdown_cell(record.get("backend")),
                wall_time=_format_wall_time(record.get("solver_wall_time_seconds")),
                fallback_rungs=_markdown_cell(
                    _join_values(record.get("fallback_rungs", []))
                ),
                validation_scope=_markdown_cell(record.get("validation_scope")),
                observables=_markdown_cell(_join_values(_display_observables(record))),
                reason=_markdown_cell(record.get("reason", "")),
            )
        )
    comparison_records = [
        record for record in report["runs"] if record.get("comparisons")
    ]
    if comparison_records:
        lines.extend(["", "## Comparison Details", ""])
        for record in comparison_records:
            lines.extend(
                [
                    "### {problem_id}".format(
                        problem_id=_markdown_text(record.get("problem_id"))
                    ),
                    "",
                    "| observable | passed | expected | actual | tolerance | message |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for item in record.get("comparisons", []):
                lines.append(
                    (
                        "| {key} | {passed} | {expected} | {actual} | "
                        "{tolerance} | {message} |"
                    ).format(
                        key=_markdown_cell(item.get("key")),
                        passed=_markdown_cell(item.get("passed")),
                        expected=_markdown_cell(_format_scalar(item.get("expected"))),
                        actual=_markdown_cell(_format_scalar(item.get("actual"))),
                        tolerance=_markdown_cell(_format_scalar(item.get("tolerance"))),
                        message=_markdown_cell(item.get("message", "")),
                    )
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def _display_observables(record: dict[str, Any]) -> Any:
    compared = record.get("observables_compared")
    if compared:
        return compared
    return record.get("checked_observables", [])


def _format_wall_time(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _join_values(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return ", ".join(str(value) for value in values)
    return str(values)


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _markdown_cell(value: Any) -> str:
    return _markdown_text(value).replace("\n", " ").replace("|", "\\|")


def _markdown_text(value: Any) -> str:
    return "" if value is None else str(value)


def _parse_skip_arg(value: str) -> dict[str, Any]:
    problem_id, separator, reason = value.partition("=")
    if not separator or not problem_id.strip() or not reason.strip():
        raise argparse.ArgumentTypeError("--skip must use the form problem_id=reason")
    return skipped_record(problem_id.strip(), reason.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/_report")
    parser.add_argument("--print", dest="print_path")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        type=_parse_skip_arg,
        metavar="PROBLEM_ID=REASON",
        help="Append an explicit skipped run record to the report.",
    )
    args = parser.parse_args(argv)

    if args.print_path:
        print(Path(args.print_path).read_text(encoding="utf-8"), end="")
        return 0

    write_report(
        find_metadata_files(args.runs_root), args.out, skipped_records=args.skip
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
