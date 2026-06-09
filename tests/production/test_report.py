import json
from pathlib import Path

from production.report import build_report, write_report
from production.run_problem import run_problem

ROOT = Path(__file__).resolve().parents[2]


def test_report_marks_validate_only_runner_metadata_as_skipped(tmp_path):
    out = tmp_path / "runs" / "tc_supercritical_saturation" / "stamp"
    run_problem(
        config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
        out=out,
        validate_only=True,
        capture_device=False,
    )
    report = build_report([out / "metadata.json"])
    assert report["summary"] == {"passed": 0, "failed": 0, "skipped": 1}
    record = report["runs"][0]
    assert record["problem_id"] == "tc_supercritical_saturation"
    assert record["outcome"] == "skipped"
    assert record["fallback_rungs"] == [2, 3]
    assert (
        record["reason"]
        == "metadata validation only; solver execution was not requested"
    )


def test_report_marks_completed_channel_oracle_as_passed(tmp_path):
    out = tmp_path / "runs" / "channel_poiseuille_hydro_v1" / "stamp"
    run_problem(
        config_path=ROOT
        / "production"
        / "examples"
        / "channel_poiseuille_hydro_v1.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    report = build_report([out / "metadata.json"])
    assert report["summary"] == {"passed": 1, "failed": 0, "skipped": 0}
    record = report["runs"][0]
    assert record["problem_id"] == "channel_poiseuille_hydro_v1"
    assert record["outcome"] == "passed"
    assert record["observables_compared"] == [
        "divergence_l2",
        "flow_rate",
        "kinetic_energy",
        "pressure_gradient",
    ]
    assert record["solver_wall_time_seconds"] is not None
    assert record["solver_started_at_utc"]
    assert record["solver_finished_at_utc"]


def test_write_report_outputs_json_and_markdown(tmp_path):
    out = tmp_path / "runs" / "tc_supercritical_saturation" / "stamp"
    run_problem(
        config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
        out=out,
        validate_only=True,
        capture_device=False,
    )
    paths = write_report([out / "metadata.json"], tmp_path / "_report")
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    data = json.loads(paths["json"].read_text())
    assert data["summary"]["skipped"] == 1
    markdown = paths["markdown"].read_text()
    assert "wall_time_s" in markdown
    assert "fallback_rungs" in markdown
    assert "2, 3" in markdown


def test_write_report_markdown_includes_observable_comparisons(tmp_path):
    out = tmp_path / "runs" / "channel_poiseuille_hydro_v1" / "stamp"
    run_problem(
        config_path=ROOT
        / "production"
        / "examples"
        / "channel_poiseuille_hydro_v1.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )

    paths = write_report([out / "metadata.json"], tmp_path / "_report")

    markdown = paths["markdown"].read_text()
    assert "observables" in markdown
    assert "flow_rate" in markdown
    assert "pressure_gradient" in markdown
    assert "## Comparison Details" in markdown
    assert "| flow_rate | True | 1.33299739649 | 1.33299739649 | 1e-10 |  |" in markdown
