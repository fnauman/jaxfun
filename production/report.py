"""Build machine-readable summaries for production validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def record_from_metadata(metadata: dict[str, Any], *, metadata_path: str | Path | None = None) -> dict[str, Any]:
    execution = metadata.get("execution", {})
    device = metadata.get("device", {})
    status = execution.get("status", "unknown")
    solver_wired = bool(execution.get("solver_execution_wired", False))
    if not solver_wired and status in {"validated", "not_started"}:
        outcome = "skipped"
        reason = "solver execution not wired; metadata validation only"
    elif status == "completed":
        outcome = "passed"
        reason = ""
    else:
        outcome = "failed"
        reason = f"execution status {status!r}"

    expected_oracle = metadata.get("expected_oracle", {})
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
        "golden_resolution": metadata.get("golden_resolution", {}),
        "observables_compared": metadata.get("observables_compared", []),
        "comparisons": metadata.get("comparisons", []),
    }


def build_report(metadata_paths: list[str | Path]) -> dict[str, Any]:
    records = [record_from_metadata(load_metadata(path), metadata_path=path) for path in metadata_paths]
    summary = {
        "passed": sum(1 for record in records if record["outcome"] == "passed"),
        "failed": sum(1 for record in records if record["outcome"] == "failed"),
        "skipped": sum(1 for record in records if record["outcome"] == "skipped"),
    }
    return {"schema_version": 1, "summary": summary, "runs": records}


def find_metadata_files(runs_root: str | Path) -> list[Path]:
    root = Path(runs_root)
    return sorted(path for path in root.glob("*/*/metadata.json") if "_report" not in path.parts)


def write_report(metadata_paths: list[str | Path], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(metadata_paths)
    json_path = out / "results.json"
    md_path = out / "results.md"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_summary(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _markdown_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Production Validation Report",
        "",
        f"Passed: {summary['passed']}  Failed: {summary['failed']}  Skipped: {summary['skipped']}",
        "",
        "| problem_id | outcome | mode | backend | reason |",
        "|---|---|---|---|---|",
    ]
    for record in report["runs"]:
        lines.append(
            "| {problem_id} | {outcome} | {mode} | {backend} | {reason} |".format(
                problem_id=record.get("problem_id"),
                outcome=record.get("outcome"),
                mode=record.get("mode"),
                backend=record.get("backend"),
                reason=record.get("reason", ""),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="runs/_report")
    parser.add_argument("--print", dest="print_path")
    args = parser.parse_args(argv)

    if args.print_path:
        print(Path(args.print_path).read_text(encoding="utf-8"), end="")
        return 0

    write_report(find_metadata_files(args.runs_root), args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
