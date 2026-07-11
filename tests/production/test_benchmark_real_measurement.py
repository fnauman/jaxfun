"""FJ-12 review blocker 6: the benchmark harness must measure the real solvers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax

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
