"""Review round 3, blocker 5: Cartesian sweep executor lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.sweep import SweepOverrideError, cartesian_grid, execute_sweep

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"


def test_cartesian_grid_orders_and_validates():
    points = cartesian_grid({"Rm_h": [400, 800], "B0": [0.025]})
    assert points == [
        {"B0": 0.025, "Rm_h": 400},
        {"B0": 0.025, "Rm_h": 800},
    ]
    with pytest.raises(SweepOverrideError, match="unknown sweep override"):
        cartesian_grid({"seed": [1, 2]})
    with pytest.raises(SweepOverrideError, match="non-empty list"):
        cartesian_grid({"Rm_h": []})


def test_execute_sweep_records_status_and_skips_completed(tmp_path):
    calls: list[str] = []

    def fake_runner(*, config_path, out, **kwargs):
        calls.append(str(config_path))
        Path(out).mkdir(parents=True, exist_ok=True)

    summary = execute_sweep(
        BASE,
        {"Rm_h": [400.0, 800.0]},
        tmp_path,
        execute=True,
        runner=fake_runner,
    )
    assert summary["points"] == 2
    assert summary["completed"] == 2
    assert summary["failed"] == 0
    index = json.loads(Path(summary["index_path"]).read_text(encoding="utf-8"))
    assert len(index) == 2
    assert all(entry["status"] == "completed" for entry in index)
    # Each materialized spec resolved the swept Rm into eta_mag before launch.
    for entry in index:
        spec = json.loads(Path(entry["spec_path"]).read_text(encoding="utf-8"))
        assert spec["nondimensional_groups"]["Rm"] in (400.0, 800.0)
        assert spec["nondimensional_groups"]["eta_mag"] == pytest.approx(
            1.0 / spec["nondimensional_groups"]["Rm"]
        )

    # Re-invocation (the widened-grid frontier workflow) skips completed points.
    calls.clear()
    summary = execute_sweep(
        BASE,
        {"Rm_h": [400.0, 800.0]},
        tmp_path,
        execute=True,
        runner=fake_runner,
    )
    assert summary["skipped"] == 2 and calls == []


def test_execute_sweep_records_failure_and_continues(tmp_path):
    seen: list[str] = []

    def flaky_runner(*, config_path, out, **kwargs):
        seen.append(str(config_path))
        if len(seen) == 1:
            raise FloatingPointError("nonfinite state")
        Path(out).mkdir(parents=True, exist_ok=True)

    summary = execute_sweep(
        BASE,
        {"Rm_h": [400.0, 800.0]},
        tmp_path,
        execute=True,
        runner=flaky_runner,
    )
    assert summary["failed"] == 1 and summary["completed"] == 1
    index = json.loads(Path(summary["index_path"]).read_text(encoding="utf-8"))
    statuses = sorted(entry["status"] for entry in index)
    assert statuses == ["completed", "failed"]
    failed = next(e for e in index if e["status"] == "failed")
    assert "FloatingPointError" in failed["failure_reason"]
