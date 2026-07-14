"""Issue #10: adaptive scientific frontier refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import production.sweep as sweep_module
from production.frontier import (
    FrontierRefinementError,
    execute_frontier_sweep,
    frontier_decision,
    verify_frontier_lineage,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"


def _metadata(scientific_class: str, slope: float, stderr: float):
    return {
        "execution": {"status": "completed"},
        "classification": {
            "scientific_class": scientific_class,
            "reason": "test classification",
            "fit": {"slope": slope, "stderr": stderr},
        },
    }


def _entry(
    value: float,
    scientific_class: str,
    slope: float,
    stderr: float,
    *,
    eligible: bool = True,
):
    return {
        "run_id": f"run-{value:g}",
        "spec_hash": f"hash-{value:g}",
        "status": "completed",
        "overrides": {"Rm_h": value},
        "result": {
            "operational_status": "completed",
            "scientific_class": scientific_class,
            "fit_slope": slope,
            "fit_stderr": stderr,
            "classification_eligible": eligible,
        },
    }


def test_frontier_decision_bisects_and_authenticates_lineage():
    entries = [
        _entry(400.0, "decayed", -0.2, 0.01),
        _entry(800.0, "growing", 0.2, 0.01),
    ]
    first = frontier_decision(entries, axis="Rm_h", abs_tolerance=25.0)
    assert first["status"] == "refine"
    assert first["next_value"] == pytest.approx(600.0)
    assert first["bracket"]["low"]["spec_hash"] == "hash-400"
    assert verify_frontier_lineage([first]) is True

    entries.append(_entry(600.0, "growing", 0.1, 0.01))
    second = frontier_decision(
        entries,
        axis="Rm_h",
        abs_tolerance=25.0,
        parent_lineage_hash=first["lineage_hash"],
    )
    assert second["next_value"] == pytest.approx(500.0)
    assert verify_frontier_lineage([first, second]) is True

    tampered = json.loads(json.dumps([first, second]))
    tampered[0]["next_value"] = 601.0
    with pytest.raises(FrontierRefinementError, match="hash mismatch"):
        verify_frontier_lineage(tampered)


def test_frontier_refuses_uncertain_or_marginal_evidence():
    uncertain = [
        _entry(400.0, "decayed", -0.2, 0.01),
        _entry(800.0, "growing", 0.01, 0.02),
    ]
    decision = frontier_decision(
        uncertain, axis="Rm_h", abs_tolerance=25.0, confidence_z=1.96
    )
    assert decision["status"] == "uncertain_endpoints"
    assert decision["next_value"] is None

    marginal = [
        _entry(400.0, "decayed", -0.2, 0.01),
        _entry(600.0, "marginal", 0.0, 0.1, eligible=False),
        _entry(800.0, "growing", 0.2, 0.01),
    ]
    decision = frontier_decision(marginal, axis="Rm_h", abs_tolerance=25.0)
    assert decision["status"] == "uncertain_points"


def test_frontier_rejects_reversed_transition_direction():
    reversed_transition = [
        _entry(400.0, "growing", 0.2, 0.01),
        _entry(800.0, "decayed", -0.2, 0.01),
    ]
    decision = frontier_decision(reversed_transition, axis="Rm_h", abs_tolerance=25.0)
    assert decision["status"] == "nonmonotonic"
    assert decision["bracket"] is None


def test_frontier_rejects_duplicate_coordinates():
    entries = [
        _entry(400.0, "decayed", -0.2, 0.01),
        {
            **_entry(400.0, "growing", 0.2, 0.01),
            "run_id": "other-fixed-controls",
        },
    ]
    with pytest.raises(FrontierRefinementError, match="duplicate values"):
        frontier_decision(entries, axis="Rm_h", abs_tolerance=25.0)


def test_frontier_requires_immutable_endpoint_identities():
    missing_identity = _entry(400.0, "decayed", -0.2, 0.01)
    missing_identity.pop("spec_hash")
    with pytest.raises(FrontierRefinementError, match="require run_id and spec_hash"):
        frontier_decision(
            [missing_identity, _entry(800.0, "growing", 0.2, 0.01)],
            axis="Rm_h",
            abs_tolerance=25.0,
        )


def test_incomplete_frontier_retries_failed_endpoint(tmp_path):
    attempts: dict[float, int] = {}

    def transient_runner(*, config_path, **kwargs):
        spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
        value = float(spec["nondimensional_groups"]["Rm"])
        attempts[value] = attempts.get(value, 0) + 1
        if value == 800.0 and attempts[value] == 1:
            raise RuntimeError("transient runner failure")
        if value < 600.0:
            return _metadata("decayed", -0.1, 0.01)
        return _metadata("growing", 0.1, 0.01)

    options = {
        "axis": "Rm_h",
        "bounds": [400.0, 800.0],
        "out_dir": tmp_path,
        "abs_tolerance": 25.0,
        "max_refinements": 0,
        "runner": transient_runner,
    }
    first = execute_frontier_sweep(BASE, **options)
    assert first["status"] == "incomplete"

    second = execute_frontier_sweep(BASE, **options)
    assert second["status"] == "max_refinements"
    assert attempts == {400.0: 1, 800.0: 2}
    lineage = json.loads(Path(second["lineage_path"]).read_text(encoding="utf-8"))
    assert verify_frontier_lineage(lineage) is True


def test_frontier_ignores_stale_sweep_rows_from_other_requests(tmp_path):
    stale = [
        {
            **_entry(400.0, "growing", 0.2, 0.01),
            "overrides": {"Rm_h": 400.0, "B0": 0.05},
        },
        {
            **_entry(600.0, "growing", 0.2, 0.01),
            "overrides": {"Rm_h": 600.0, "B0": 0.025},
        },
        {
            **_entry(900.0, "growing", 0.2, 0.01),
            "overrides": {"Rm_h": 900.0, "B0": 0.025},
        },
    ]
    (tmp_path / "sweep_index.json").write_text(json.dumps(stale), encoding="utf-8")

    def classified_runner(*, config_path, **kwargs):
        spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
        value = float(spec["nondimensional_groups"]["Rm"])
        if value < 600.0:
            return _metadata("decayed", -0.1, 0.01)
        return _metadata("growing", 0.1, 0.01)

    summary = execute_frontier_sweep(
        BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path,
        fixed_overrides={"B0": 0.025},
        abs_tolerance=25.0,
        max_refinements=0,
        runner=classified_runner,
    )
    assert summary["status"] == "max_refinements"
    assert summary["sampled_points"] == 2
    lineage = json.loads(Path(summary["lineage_path"]).read_text(encoding="utf-8"))
    assert [point["value"] for point in lineage[-1]["sampled_points"]] == [
        400.0,
        800.0,
    ]


def test_execute_frontier_sweep_refines_resumes_and_persists_results(tmp_path):
    calls: list[float] = []

    def classified_runner(*, config_path, out, **kwargs):
        spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
        value = float(spec["nondimensional_groups"]["Rm"])
        calls.append(value)
        Path(out).mkdir(parents=True, exist_ok=True)
        if value < 550.0:
            return _metadata("decayed", -0.1, 0.01)
        return _metadata("growing", 0.1, 0.01)

    summary = execute_frontier_sweep(
        BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path,
        fixed_overrides={"B0": 0.025},
        abs_tolerance=50.0,
        runner=classified_runner,
    )
    assert summary["status"] == "converged"
    assert summary["bracket"]["low"]["value"] == pytest.approx(500.0)
    assert summary["bracket"]["high"]["value"] == pytest.approx(550.0)
    assert sorted(calls) == [400.0, 500.0, 550.0, 600.0, 800.0]

    index = json.loads(Path(summary["sweep_index_path"]).read_text(encoding="utf-8"))
    assert len(index) == 5
    assert all(entry["result"]["schema_version"] == 1 for entry in index)
    assert all(entry["result"]["classification_eligible"] for entry in index)

    lineage = json.loads(Path(summary["lineage_path"]).read_text(encoding="utf-8"))
    assert len(lineage) == 4
    assert verify_frontier_lineage(lineage) is True
    assert lineage[-1]["status"] == "converged"

    calls.clear()
    resumed = execute_frontier_sweep(
        BASE,
        axis="Rm_h",
        bounds=[400.0, 800.0],
        out_dir=tmp_path,
        fixed_overrides={"B0": 0.025},
        abs_tolerance=50.0,
        runner=classified_runner,
    )
    assert resumed["lineage_hash"] == summary["lineage_hash"]
    assert calls == []

    with pytest.raises(FrontierRefinementError, match="does not match"):
        execute_frontier_sweep(
            BASE,
            axis="Rm_h",
            bounds=[400.0, 800.0],
            out_dir=tmp_path,
            fixed_overrides={"B0": 0.05},
            abs_tolerance=50.0,
            runner=classified_runner,
        )

    with pytest.raises(FrontierRefinementError, match="does not match"):
        execute_frontier_sweep(
            BASE,
            axis="Rm_h",
            bounds=[400.0, 800.0],
            out_dir=tmp_path,
            fixed_overrides={"B0": 0.025},
            abs_tolerance=50.0,
            max_refinements=9,
            runner=classified_runner,
        )


@pytest.mark.parametrize(
    ("status", "expected_exit"),
    [
        ("converged", 0),
        ("max_refinements", 0),
        ("nonmonotonic", 0),
        ("uncertain_endpoints", 0),
        ("uncertain_points", 0),
        ("unbracketed", 0),
        ("incomplete", 1),
    ],
)
def test_frontier_cli_distinguishes_scientific_stops_from_incomplete_runs(
    monkeypatch, tmp_path, status, expected_exit
):
    def fake_frontier(*_args, **_kwargs):
        return {"status": status}

    monkeypatch.setattr(sweep_module, "execute_frontier_sweep", fake_frontier)
    frontier_json = json.dumps(
        {"axis": "Rm_h", "bounds": [400, 800], "abs_tolerance": 25}
    )
    exit_code = sweep_module.main(
        [
            "--base",
            str(BASE),
            "--out",
            str(tmp_path),
            "--execute",
            "--frontier",
            frontier_json,
        ]
    )
    assert exit_code == expected_exit
