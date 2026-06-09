import json
from pathlib import Path

import pytest

from production.problem_spec import UnsupportedSpecError
from production.run_problem import SolverExecutionNotImplementedError, main, run_problem


ROOT = Path(__file__).resolve().parents[2]


def test_validate_only_writes_metadata_without_claiming_solver_execution(tmp_path):
    out = tmp_path / "run"
    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
        out=out,
        validate_only=True,
        capture_device=False,
    )
    written = json.loads((out / "metadata.json").read_text())
    assert written["problem_id"] == "tc_supercritical_saturation"
    assert written["execution"] == {
        "solver_execution_wired": False,
        "status": "validated",
    }
    assert metadata["adapter"]["axis_conventions"]["axis_0"] == "r radial"


def test_pipe_spec_rejected_before_output_directory_is_created(tmp_path):
    out = tmp_path / "pipe"
    with pytest.raises(UnsupportedSpecError):
        run_problem(
            config_path=ROOT / "production" / "examples" / "pipe_hagen_poiseuille_v1.json",
            out=out,
            validate_only=True,
            capture_device=False,
        )
    assert not out.exists()


def test_non_validate_run_fails_explicitly_until_solver_is_wired(tmp_path):
    with pytest.raises(SolverExecutionNotImplementedError, match="solver execution is not wired"):
        run_problem(
            config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
            out=tmp_path / "run",
            capture_device=False,
        )


def test_cli_validate_only_returns_success(tmp_path):
    code = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "tc_supercritical_saturation.json"),
            "--out",
            str(tmp_path / "run"),
            "--validate-only",
        ]
    )
    assert code == 0


def test_cli_non_validate_returns_not_implemented_status(tmp_path):
    code = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "tc_supercritical_saturation.json"),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 2
