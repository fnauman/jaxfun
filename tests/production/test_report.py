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
