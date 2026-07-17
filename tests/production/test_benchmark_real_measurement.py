"""FJ-12 review blocker 6: the benchmark harness must measure the real solvers."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import jax
import numpy as np
import pytest

from production.adapters import load_config
from production.benchmark import _solver_and_seed_builders, main, measure_spec
from production.oracles import ProductionOracleNotImplementedError

ROOT = Path(__file__).resolve().parents[2]
VP_SPEC = ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"


def test_measure_spec_times_the_production_curl_solver():
    jax.config.update("jax_enable_x64", True)
    artifact = measure_spec(
        VP_SPEC,
        tiers=("smoke",),
        timed_steps=3,
        warmup_steps=1,
        rollout_steps=2,
    )

    assert artifact["problem_id"] == "exp_pcf_mri_vector_potential"
    assert artifact["backend"]
    (row,) = artifact["measurements"]
    assert row["tier"] == "smoke"
    # Smoke tier: 9x8x8 grid x 6 evolved fields.
    assert row["dof"] == 9 * 8 * 8 * 6
    assert row["compile_s"] > 0.0
    assert row["warm_step_s"] > 0.0
    assert row["timed_steps"] == 3
    assert row["rollout_steps"] == 2
    assert row["total_timed_steps"] == 6
    assert row["dt_transition_probe"]["rollout_cache_misses_delta"] == 0
    assert row["dt_transition_probe"]["reused_compiled_variant"] is True
    assert "compilation_cache_info" in row
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
            "--rollout-steps",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["measurements"][0]["label"] == "exp_pcf_mri_vector_potential@smoke"
    assert "provenance" in artifact
    assert artifact["provenance"]["jax_config"]["jax_use_simplified_jaxpr_constants"]


def test_measure_spec_holdout_validation(monkeypatch):
    """Review round 3: the cost model must be validated on a held-out tier."""
    import production.benchmark as bm

    def fake_builders(spec):
        return (lambda: None), (lambda solver: None)

    def fake_benchmark_step(
        build_solver,
        *,
        label,
        warmup_steps,
        timed_steps,
        seed_state,
        rollout_steps,
        dt_transition_probes,
    ):
        tier = label.split("@")[1]
        dof = {
            "smoke": 9 * 8 * 8 * 6,
            "start": 17 * 16 * 16 * 6,
            "production": 33 * 32 * 32 * 6,
        }[tier]
        warm = 1e-8 * dof**1.2  # exact power law -> holdout error ~ 0
        return bm.StepTiming(
            label=label,
            compile_s=1.0,
            warm_step_s=warm,
            warm_step_p50_s=warm,
            warm_step_p90_s=warm,
            timed_steps=timed_steps,
            dt=0.005,
            peak_bytes=None,
            rollout_steps=rollout_steps,
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


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "expected_class", "ntheta"),
    [
        ("pcf_fluct_re400.json", "PlaneCouetteFluctuationJax", None),
        ("pcf_mhd_divfree.json", "PCFMRIDNSJax", None),
        ("exp_pcf_mri_vector_potential.json", "PlaneCouetteMRIShearpyJax", None),
        ("tc_supercritical_saturation.json", "AxisymmetricTCDNSJax", None),
        ("tc_supercritical_saturation.json", "TaylorCouetteDNSJax", 4),
        ("tc_mri_nonlinear_saturation.json", "AxisymmetricMRIDNSJax", None),
        ("tc_mri_nonlinear_saturation.json", "TaylorCouetteMRIDNSJax", 4),
        ("exp_tc_mri_vector_potential.json", "TaylorCouetteVPMRIDNSJax", None),
    ],
)
def test_benchmark_factory_covers_wall_bounded_production_paths(
    filename, expected_class, ntheta
):
    config = load_config(
        ROOT / "production" / "runs" / filename, resolution_tier="smoke"
    )
    spec = copy.deepcopy(config.spec)
    if ntheta is not None:
        spec["resolution"]["Ntheta"] = ntheta

    build_solver, seed_state = _solver_and_seed_builders(spec)
    solver = build_solver()
    assert type(solver).__name__ == expected_class

    state = seed_state(solver)
    leaves = jax.tree_util.tree_leaves(state)
    assert leaves
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


@pytest.mark.parametrize("integrator", ["analytic", "linear_eigenproblem"])
def test_benchmark_rejects_non_time_stepping_pcf_hydro_integrator(
    integrator,
) -> None:
    config = load_config(
        ROOT / "production" / "runs" / "pcf_fluct_re400.json",
        resolution_tier="smoke",
    )
    spec = copy.deepcopy(config.spec)
    spec["time"]["integrator"] = integrator

    with pytest.raises(
        ProductionOracleNotImplementedError,
        match=r"PCF hydrodynamic KMM benchmark requires a time-stepping integrator",
    ):
        _solver_and_seed_builders(spec)
