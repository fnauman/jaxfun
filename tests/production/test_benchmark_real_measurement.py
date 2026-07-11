"""FJ-12 review blocker 6: the benchmark harness must measure the real solvers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import pytest

from production.benchmark import main, measure_spec

ROOT = Path(__file__).resolve().parents[2]
VP_SPEC = ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"


def test_measure_spec_times_the_production_curl_solver():
    jax.config.update("jax_enable_x64", True)
    artifact = measure_spec(VP_SPEC, tiers=("smoke",), timed_steps=3, warmup_steps=1)

    assert artifact["problem_id"] == "exp_pcf_mri_vector_potential"
    assert artifact["backend"]
    (row,) = artifact["measurements"]
    assert row["tier"] == "smoke"
    # Smoke tier: 9x8x8 grid x 6 evolved fields.
    assert row["dof"] == 9 * 8 * 8 * 6
    assert row["compile_s"] > 0.0
    assert row["warm_step_s"] > 0.0
    assert row["timed_steps"] == 3
    assert math.isfinite(row["cost_per_shear_time_s"])
    assert math.isfinite(row["predicted_hours_full_horizon"])
    # A single tier cannot fit the power law; that must be explicit, not silent.
    assert artifact["cost_model"] is None
    assert "cost_model_note" in artifact


def test_benchmark_cli_writes_artifact(tmp_path):
    jax.config.update("jax_enable_x64", True)
    out = tmp_path / "bench.json"
    rc = main(
        [
            "--config",
            str(VP_SPEC),
            "--tiers",
            "smoke",
            "--timed-steps",
            "2",
            "--warmup-steps",
            "1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["measurements"][0]["label"] == "exp_pcf_mri_vector_potential@smoke"
    assert "provenance" in artifact


def test_measure_spec_holdout_validation(monkeypatch):
    """Review round 3: the cost model must be validated on a held-out tier."""
    import production.benchmark as bm

    def fake_builders(spec):
        return (lambda: None), (lambda solver: None)

    def fake_benchmark_step(build_solver, *, label, warmup_steps, timed_steps, seed_state):
        tier = label.split("@")[1]
        dof = {"smoke": 9 * 8 * 8 * 6, "start": 17 * 16 * 16 * 6,
               "production": 33 * 32 * 32 * 6}[tier]
        warm = 1e-8 * dof**1.2  # exact power law -> holdout error ~ 0
        return bm.StepTiming(
            label=label, compile_s=1.0, warm_step_s=warm, warm_step_p50_s=warm,
            warm_step_p90_s=warm, timed_steps=timed_steps, dt=0.005, peak_bytes=None,
        )

    monkeypatch.setattr(bm, "_solver_and_seed_builders", fake_builders)
    monkeypatch.setattr(bm, "benchmark_step", fake_benchmark_step)

    artifact = bm.measure_spec(
        VP_SPEC,
        tiers=("smoke", "start", "production"),
        timed_steps=2,
        holdout_tier="production",
    )
    validation = artifact["holdout_validation"]
    assert validation["tier"] == "production"
    assert validation["relative_error"] < 1e-9
    assert artifact["cost_model"]["b"] == pytest.approx(1.2, rel=1e-6)

    with pytest.raises(ValueError, match="holdout"):
        bm.measure_spec(
            VP_SPEC, tiers=("smoke", "start"), timed_steps=2, holdout_tier="production"
        )
