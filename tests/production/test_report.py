import json
from pathlib import Path

from production.report import (
    build_report,
    main,
    record_from_metadata,
    skipped_record,
    write_report,
)
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


def test_report_labels_rung3_cpu_smoke_as_finiteness_only(tmp_path):
    reason = (
        "rung-3-only saturated run has no committed nonlinear-state golden; "
        "CPU smoke checks solver completion, finite diagnostics, and emitted "
        "divergence diagnostics, not production parity"
    )
    metadata = {
        "problem_id": "pcf_mhd_divfree",
        "out_dir": "runs/pcf_mhd_divfree/stamp",
        "geometry": "pcf",
        "physics": "mhd",
        "expected_oracle": {"fallback_rungs": [3]},
        "device": {
            "mode": "cpu_smoke",
            "default_backend": "cpu",
            "degraded": True,
        },
        "execution": {
            "status": "completed",
            "solver_execution_wired": True,
            "execution_kind": "dns-saturation",
        },
        "validation_scope": {
            "kind": "cpu_smoke_finiteness_divergence_only",
            "reason": reason,
            "checked_observables": [
                "divergence_u_l2",
                "divergence_b_l2",
                "magnetic_energy",
            ],
        },
        "timing": {"solver_wall_time_seconds": 1.25},
    }

    record = record_from_metadata(metadata, metadata_path="runs/x/metadata.json")

    assert record["outcome"] == "passed"
    assert record["validation_scope"] == "cpu_smoke_finiteness_divergence_only"
    assert record["checked_observables"] == [
        "divergence_u_l2",
        "divergence_b_l2",
        "magnetic_energy",
    ]
    assert record["reason"] == reason

    metadata_path = tmp_path / "runs" / "pcf_mhd_divfree" / "stamp" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    paths = write_report([metadata_path], tmp_path / "_report")

    markdown = paths["markdown"].read_text()
    assert "validation_scope" in markdown
    assert "cpu_smoke_finiteness_divergence_only" in markdown
    assert "divergence_u_l2, divergence_b_l2, magnetic_energy" in markdown
    assert "not production parity" in markdown


def test_report_labels_bounded_gpu_smoke_as_not_full_saturation(tmp_path):
    reason = (
        "executed a step-limited or reduced-resolution saturation smoke run; "
        "generated artifacts are smoke diagnostics, not a full production "
        "saturation golden"
    )
    metadata = {
        "problem_id": "pcf_mhd_divfree",
        "out_dir": "runs/pcf_mhd_divfree/stamp",
        "geometry": "pcf",
        "physics": "mhd",
        "expected_oracle": {"fallback_rungs": [3]},
        "device": {
            "mode": "gpu",
            "default_backend": "gpu",
            "degraded": False,
        },
        "execution": {
            "status": "completed",
            "solver_execution_wired": True,
            "execution_kind": "dns-saturation",
        },
        "validation_scope": {
            "kind": "bounded_saturation_smoke",
            "reason": reason,
            "checked_observables": ["divergence_b_l2", "magnetic_energy"],
            "steps_override": 2,
            "resolution_tier": "smoke",
            "bounded_smoke": True,
        },
        "timing": {"solver_wall_time_seconds": 1.25},
    }

    record = record_from_metadata(metadata, metadata_path="runs/x/metadata.json")

    assert record["outcome"] == "passed"
    assert record["validation_scope"] == "bounded_saturation_smoke"
    assert record["reason"] == reason

    metadata_path = tmp_path / "runs" / "pcf_mhd_divfree" / "stamp" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    paths = write_report([metadata_path], tmp_path / "_report")

    markdown = paths["markdown"].read_text()
    assert "bounded_saturation_smoke" in markdown
    assert "not a full production saturation golden" in markdown
    assert "divergence_b_l2, magnetic_energy" in markdown


def test_report_uses_runner_failure_reason_for_failed_execution():
    metadata = {
        "problem_id": "tc_mri_nonlinear_saturation",
        "geometry": "taylor_couette",
        "physics": "mri",
        "expected_oracle": {"fallback_rungs": [1, 2, 3]},
        "execution": {
            "status": "failed",
            "solver_execution_wired": True,
            "execution_kind": "dns-saturation",
            "failure_reason": (
                "full saturation check failed: saturation_check_passed is false"
            ),
        },
        "device": {"mode": "gpu", "default_backend": "gpu", "degraded": False},
    }

    report = build_report([])
    report["runs"] = [record_from_metadata(metadata)]
    report["summary"] = {"passed": 0, "failed": 1, "skipped": 0}

    record = report["runs"][0]
    assert record["outcome"] == "failed"
    assert (
        record["reason"]
        == "full saturation check failed: saturation_check_passed is false"
    )


def test_report_marks_failed_golden_comparison_as_failed(tmp_path):
    metadata = {
        "problem_id": "channel_poiseuille_hydro_v1",
        "out_dir": "runs/channel/stamp",
        "geometry": "channel",
        "physics": "hydro",
        "expected_oracle": {},
        "device": {
            "mode": "cpu_smoke",
            "default_backend": "cpu",
            "degraded": True,
        },
        "execution": {
            "status": "completed",
            "solver_execution_wired": True,
            "execution_kind": "analytic-oracle",
        },
        "comparison_passed": False,
        "observables_compared": ["flow_rate"],
        "comparisons": [
            {
                "key": "flow_rate",
                "expected": 1.0,
                "actual": 1.25,
                "tolerance": 1.0e-10,
                "passed": False,
                "message": "abs diff 0.25 exceeds tolerance 1e-10",
            }
        ],
        "validation_scope": {
            "kind": "golden_comparison",
            "reason": "compared diagnostics against the resolved committed golden",
            "checked_observables": ["flow_rate"],
        },
        "timing": {"solver_wall_time_seconds": 0.1},
    }

    metadata_path = tmp_path / "runs" / "channel" / "stamp" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_report([metadata_path])

    assert report["summary"] == {"passed": 0, "failed": 1, "skipped": 0}
    record = report["runs"][0]
    assert record["outcome"] == "failed"
    assert record["reason"] == "golden comparison failed"
    assert record["comparisons"][0]["expected"] == 1.0
    assert record["comparisons"][0]["actual"] == 1.25
    assert record["comparisons"][0]["tolerance"] == 1.0e-10

    paths = write_report([metadata_path], tmp_path / "_report")
    markdown = paths["markdown"].read_text()
    assert "Failed: 1" in markdown
    assert "golden comparison failed" in markdown
    expected_row = (
        "| flow_rate | False | 1 | 1.25 | 1e-10 | "
        "abs diff 0.25 exceeds tolerance 1e-10 |"
    )
    assert expected_row in markdown


def test_report_accepts_explicit_skipped_records():
    reason = "pipe hydro is parity_pending until the axis-regular radial basis lands"

    report = build_report(
        [],
        skipped_records=[
            skipped_record(
                "pipe_hagen_poiseuille_v1",
                reason,
                geometry="pipe",
                physics="hydro",
            )
        ],
    )

    assert report["summary"] == {"passed": 0, "failed": 0, "skipped": 1}
    record = report["runs"][0]
    assert record["problem_id"] == "pipe_hagen_poiseuille_v1"
    assert record["geometry"] == "pipe"
    assert record["physics"] == "hydro"
    assert record["outcome"] == "skipped"
    assert record["reason"] == reason


def test_report_cli_writes_explicit_pipe_skip_rows(tmp_path):
    reason = "pipe hydro is parity_pending until the axis-regular radial basis lands"

    code = main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--out",
            str(tmp_path / "_report"),
            "--skip",
            f"pipe_hagen_poiseuille_v1={reason}",
            "--skip",
            f"pipe_womersley_v1={reason}",
        ]
    )

    assert code == 0
    data = json.loads((tmp_path / "_report" / "results.json").read_text())
    assert data["summary"] == {"passed": 0, "failed": 0, "skipped": 2}
    assert {record["problem_id"] for record in data["runs"]} == {
        "pipe_hagen_poiseuille_v1",
        "pipe_womersley_v1",
    }
    markdown = (tmp_path / "_report" / "results.md").read_text()
    assert "pipe_hagen_poiseuille_v1" in markdown
    assert reason in markdown


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
