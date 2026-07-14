"""Issue #17: first-passage survival ensembles and uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.survival import (
    SURVIVAL_GROUPING_KEYS,
    SurvivalAnalysisError,
    kaplan_meier,
    load_quench_observation,
    main,
    quench_first_passage,
    survival_ensembles,
)


def _group(
    label: str = "same",
    *,
    dwell_time: float = 0.0,
    analysis_start_age: float = 0.0,
) -> dict:
    return {
        "problem_id": "transition-child",
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "magnetic_bc": "insulating",
        "numerics_contract_version": 2,
        "child_spec_hash": f"child-{label}",
        "parent_spec_hash": "parent-spec",
        "mutable_diff_hash": "mutable-diff",
        "resolution_tier": "testing",
        "precision": "float64",
        "integrator": "IMEXRK222",
        "dt": 0.5,
        "energy_convention": "volume_mean_total",
        "energy_key": "energy",
        "decay_threshold": 1.0,
        "dwell_time": dwell_time,
        "analysis_start_age": analysis_start_age,
    }


def _observation(
    run_id: str,
    cluster: str,
    duration: float,
    *,
    event: bool,
    group: dict | None = None,
):
    rows = [{"t": 0.0, "energy": 2.0}]
    rows.append({"t": duration, "energy": 0.5 if event else 2.0})
    return quench_first_passage(
        rows,
        run_id=run_id,
        parent_cluster_id=cluster,
        group=group or _group(),
        parent_time=0.0,
        energy_key="energy",
        threshold=1.0,
        dwell_time=0.0,
    )


def test_first_passage_uses_quench_age_and_requires_continuous_dwell():
    rows = [
        {"t": 10.0, "energy": 2.0},
        {"t": 11.0, "energy": 0.5},
        {"t": 12.0, "energy": 2.0},
        {"t": 13.0, "energy": 0.4},
        {"t": 14.0, "energy": 0.3},
        {"t": 15.0, "energy": 0.2},
    ]
    observation = quench_first_passage(
        rows,
        run_id="run-event",
        parent_cluster_id="parent-a",
        group=_group("transition", dwell_time=2.0),
        parent_time=10.0,
        energy_key="energy",
        threshold=1.0,
        dwell_time=2.0,
    )

    assert observation["event_observed"] is True
    assert observation["duration"] == pytest.approx(3.0)
    assert observation["event"]["first_passage_age"] == pytest.approx(3.0)
    assert observation["event"]["qualification_age"] == pytest.approx(5.0)
    assert observation["clock"]["last_observed_age"] == pytest.approx(5.0)


def test_unqualified_decay_is_right_censored_after_burn_in():
    rows = [
        {"t": 10.0, "energy": 0.5},
        {"t": 11.0, "energy": 0.4},
        {"t": 12.0, "energy": 2.0},
        {"t": 13.0, "energy": 0.5},
        {"t": 14.0, "energy": 0.4},
    ]
    observation = quench_first_passage(
        rows,
        run_id="run-censored",
        parent_cluster_id="parent-a",
        group=_group("transition", dwell_time=2.0, analysis_start_age=2.0),
        parent_time=10.0,
        energy_key="energy",
        threshold=1.0,
        dwell_time=2.0,
        analysis_start_age=2.0,
        operational_status="walltime",
    )

    assert observation["event_observed"] is False
    assert observation["duration"] == pytest.approx(4.0)
    assert observation["event"]["right_censor_age"] == pytest.approx(4.0)
    assert observation["operational_status"] == "walltime"

    with pytest.raises(SurvivalAnalysisError, match="strictly increasing"):
        quench_first_passage(
            [{"t": 10.0, "energy": 2.0}, {"t": 10.0, "energy": 1.0}],
            run_id="bad-clock",
            parent_cluster_id="parent-a",
            group=_group("transition"),
            parent_time=10.0,
            energy_key="energy",
            threshold=1.0,
            dwell_time=0.0,
        )


def test_kaplan_meier_risk_sets_greenwood_and_cluster_bootstrap_are_deterministic():
    observations = [
        _observation("event-1", "parent-a", 1.0, event=True),
        _observation("censor-2", "parent-a", 2.0, event=False),
        _observation("event-3", "parent-b", 3.0, event=True),
    ]
    estimate = kaplan_meier(
        observations,
        cluster_bootstrap_samples=100,
        seed=7,
    )

    assert estimate["runs"] == 3
    assert estimate["events"] == 2
    assert estimate["right_censored"] == 1
    assert estimate["parent_clusters"] == 2
    assert estimate["median_survival"] == pytest.approx(3.0)
    assert [row["at_risk"] for row in estimate["curve"]] == [3, 2, 1]
    assert [row["survival"] for row in estimate["curve"]] == pytest.approx(
        [2.0 / 3.0, 2.0 / 3.0, 0.0]
    )
    for row in estimate["curve"]:
        assert (
            row["greenwood_loglog_lower"]
            <= row["survival"]
            <= row["greenwood_loglog_upper"]
        )
        assert row["cluster_bootstrap_lower"] is not None
        assert row["cluster_bootstrap_upper"] is not None

    repeated = kaplan_meier(
        observations,
        cluster_bootstrap_samples=100,
        seed=7,
    )
    assert repeated == estimate


def test_observation_hashes_and_complete_groups_prevent_silent_mixing():
    first = _observation("run-a", "parent-a", 1.0, event=True)
    with pytest.raises(SurvivalAnalysisError, match="complete canonical"):
        _observation(
            "incomplete",
            "parent-a",
            1.0,
            event=True,
            group={"campaign": "incomplete"},
        )

    mismatched = _group()
    mismatched["dwell_time"] = 1.0
    with pytest.raises(SurvivalAnalysisError, match="analysis controls"):
        _observation(
            "mismatch",
            "parent-a",
            1.0,
            event=True,
            group=mismatched,
        )

    second = _observation(
        "run-b",
        "parent-b",
        2.0,
        event=False,
        group=_group("different"),
    )
    analysis = survival_ensembles([second, first])
    assert len(analysis["ensembles"]) == 2
    assert analysis["analysis_hash"]
    assert (
        analysis["observations"][0]["group_hash"]
        < analysis["observations"][1]["group_hash"]
    )

    tampered = json.loads(json.dumps(first))
    tampered["duration"] = 99.0
    with pytest.raises(SurvivalAnalysisError, match="hash"):
        kaplan_meier([tampered])


def _write_quench_run(
    root: Path,
    name: str,
    *,
    parent_run_dir: str,
    values: list[float],
    status: str = "completed",
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    spec = {
        "problem_id": "transition-child",
        "spec_hash": "child-spec-hash",
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "precision": "float64",
        "boundary_conditions": {"magnetic": {"type": "conducting"}},
        "time": {"integrator": "IMEXRK222", "dt": 0.5},
    }
    metadata = {
        "schema_version": 1,
        "generated_at_utc": f"2026-07-14T00:00:0{name[-1]}Z",
        "problem_id": spec["problem_id"],
        "spec_hash": spec["spec_hash"],
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "out_dir": str(run_dir),
        "run_options": {"resolution_tier": "production"},
        "device": {"production_run_dtype": "float64"},
        "integrator": {"actual": "IMEXRK222", "dt": 0.5},
        "execution": {"status": status},
        "quench": {
            "mode": "quench",
            "parent_run_dir": parent_run_dir,
            "parent_spec_hash": "parent-spec-hash",
            "child_spec_hash": spec["spec_hash"],
            "classification_valid_after_tstep": 102,
            "parent_checkpoint_sha256": f"checkpoint-{name}",
            "mutable_diff": {"nondimensional_groups.Rm": [100.0, 80.0]},
            "duration": {
                "parent_checkpoint": {"time": 10.0, "step": 100},
            },
        },
    }
    rows = [
        {
            "t": 10.0 + 0.5 * index,
            "mag_energy_fluct": value,
            "energy_convention": "integral_abs2",
        }
        for index, value in enumerate(values)
    ]
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    diagnostics_name = (
        "diagnostics.jsonl" if status == "completed" else "diagnostics.partial.jsonl"
    )
    (run_dir / diagnostics_name).write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


def test_run_loader_groups_parent_clusters_and_censors_partial_runs(tmp_path):
    event_dir = _write_quench_run(
        tmp_path,
        "run1",
        parent_run_dir="/parents/a",
        values=[2.0, 2.0, 2.0, 0.5, 0.4],
    )
    censored_dir = _write_quench_run(
        tmp_path,
        "run2",
        parent_run_dir="/parents/a",
        values=[2.0, 2.0, 2.0, 0.5],
        status="walltime",
    )
    event = load_quench_observation(
        event_dir,
        energy_key="mag_energy_fluct",
        threshold=1.0,
        dwell_time=0.5,
    )
    censored = load_quench_observation(
        censored_dir,
        energy_key="mag_energy_fluct",
        threshold=1.0,
        dwell_time=0.5,
    )

    assert event["event"]["first_passage_age"] == pytest.approx(1.5)
    assert event["event"]["qualification_age"] == pytest.approx(2.0)
    assert censored["event_observed"] is False
    assert censored["duration"] == pytest.approx(1.5)
    assert censored["operational_status"] == "walltime"
    assert event["parent_cluster_id"] == censored["parent_cluster_id"]
    assert tuple(event["group"]) == SURVIVAL_GROUPING_KEYS


def test_cli_writes_clustered_survival_analysis(tmp_path):
    first = _write_quench_run(
        tmp_path,
        "run1",
        parent_run_dir="/parents/a",
        values=[2.0, 2.0, 2.0, 0.5, 0.4],
    )
    second = _write_quench_run(
        tmp_path,
        "run2",
        parent_run_dir="/parents/b",
        values=[2.0, 2.0, 2.0, 2.0, 2.0],
    )
    output = tmp_path / "survival.json"
    exit_code = main(
        [
            str(first),
            str(second),
            "--threshold",
            "1.0",
            "--dwell-time",
            "0.5",
            "--cluster-bootstrap",
            "50",
            "--seed",
            "3",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["analysis_hash"]
    assert len(analysis["observations"]) == 2
    assert len(analysis["ensembles"]) == 1
    assert analysis["ensembles"][0]["parent_clusters"] == 2
    assert (
        analysis["ensembles"][0]["uncertainty"]["clustered"]["method"]
        == "parent_cluster_percentile_bootstrap"
    )
