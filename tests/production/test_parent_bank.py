"""Review round 3, blocker 3: the ZNF parent-bank quench workflow."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest

from production.oracles import (
    load_checkpoint_bank_index,
    load_resume_checkpoint,
    run_supported_spec,
)

ROOT = Path(__file__).resolve().parents[2]


def _vp_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vp_bank_smoke",
        "spec_hash": "vp-bank-spec-hash",
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "conducting"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0, "Omega": 2.0 / 3.0, "nu": 2e-2, "eta_mag": 2e-2,
            "Re": 50.0, "Rm": 50.0, "Pm": 1.0, "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1e-3, "final_time": 0.01},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {"artifact_id": "vp-bank", "regeneration_command": "test-only"},
    }
    spec["nondimensional_groups"].update(groups)
    spec["forcing"]["B0"] = spec["nondimensional_groups"]["B0"]
    return spec


def test_checkpoint_bank_retains_multiple_plateau_times(tmp_path):
    """The latest checkpoint is O(1)-rewritten; the bank must retain each
    interval with a provenance manifest so multiple plateau times survive."""
    jax.config.update("jax_enable_x64", True)
    run_dir = tmp_path / "parent"
    run_supported_spec(
        _vp_spec(), steps=4, out_dir=run_dir, checkpoint_every=2, checkpoint_bank=True
    )

    bank_dir = run_dir / "checkpoints" / "bank"
    assert (bank_dir / "checkpoint_00000002.h5").exists()
    assert (bank_dir / "checkpoint_00000004.h5").exists()
    entries = load_checkpoint_bank_index(run_dir)
    assert [entry["tstep"] for entry in entries] == [2, 4]
    for entry in entries:
        assert entry["spec_hash"] == "vp-bank-spec-hash"
        assert entry["representation"] == "vector_potential"
        assert entry["file_sha256"]
        assert Path(entry["checkpoint_path"]).exists()

    # The latest file still resumes to the final step; step= selects a banked
    # plateau instead.
    assert load_resume_checkpoint(run_dir).tstep == 4
    assert load_resume_checkpoint(run_dir, step=2).tstep == 2


def test_quench_from_banked_plateau_applies_burn_in_to_fits(tmp_path):
    """A quench can select an earlier plateau and the burn-in window is
    excluded from the fitted history, not just recorded in metadata."""
    jax.config.update("jax_enable_x64", True)
    run_dir = tmp_path / "parent"
    run_supported_spec(
        _vp_spec(), steps=4, out_dir=run_dir, checkpoint_every=2, checkpoint_bank=True
    )
    record = load_resume_checkpoint(run_dir, step=2)
    assert record.tstep == 2

    child = _vp_spec(Rm=40.0, eta_mag=2.5e-2, Pm=0.8)
    child["spec_hash"] = "vp-bank-child-hash"
    out = run_supported_spec(
        child,
        steps=8,
        resume_checkpoint=record,
        quench=True,
        diagnostics_every=1,
        burn_in_steps=2,
    )
    sc = out["scalars"]
    dt = child["time"]["dt"]
    assert sc["analysis_burn_in_steps"] == 2
    assert sc["analysis_t_start"] == pytest.approx((2 + 2) * dt)
    # The recorded series still starts at the parent plateau time...
    assert out["time_series"][0]["t"] == pytest.approx(2 * dt)
    # ...but the stationarity fit window excluded the burn-in rows: its sample
    # count matches the post-burn-in rows only.
    fit_rows = [
        r
        for r in out["time_series"]
        if r["t"] >= sc["analysis_t_start"] - 1e-12
    ]
    assert sc["stationarity_window_samples"] <= len(fit_rows)


def test_cli_parent_bank_quench_workflow(tmp_path):
    """End-to-end: parent runs with --checkpoint-bank, child quenches from a
    selected plateau step with a burn-in horizon."""
    jax.config.update("jax_enable_x64", True)
    from production.run_problem import main

    parent_dir = tmp_path / "parent"
    rc = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json"),
            "--out",
            str(parent_dir),
            "--resolution-tier",
            "smoke",
            "--steps",
            "4",
            "--checkpoint-every",
            "2",
            "--checkpoint-bank",
        ]
    )
    assert rc == 0
    assert (parent_dir / "checkpoints" / "bank" / "checkpoint_00000002.h5").exists()

    # Child spec: the same materialized smoke spec with only Rm/eta/Pm changed.
    parent_spec = json.loads((parent_dir / "spec.json").read_text(encoding="utf-8"))
    child_spec = json.loads(json.dumps(parent_spec))
    child_spec["nondimensional_groups"]["Rm"] = 800.0
    child_spec["nondimensional_groups"]["eta_mag"] = 1.0 / 800.0
    child_spec["nondimensional_groups"]["Pm"] = 800.0 / 1000.0
    child_spec.pop("spec_hash", None)
    child_path = tmp_path / "child_spec.json"
    child_path.write_text(json.dumps(child_spec), encoding="utf-8")

    child_dir = tmp_path / "child"
    rc = main(
        [
            "--config",
            str(child_path),
            "--out",
            str(child_dir),
            "--steps",
            "8",
            "--diagnostics-every",
            "1",
            "--quench",
            str(parent_dir),
            "--quench-step",
            "2",
            "--burn-in-steps",
            "2",
        ]
    )
    assert rc == 0
    metadata = json.loads((child_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["quench"]["mode"] == "quench"
    assert metadata["quench"]["selected_tstep"] == 2
    assert metadata["quench"]["requested_quench_step"] == 2
    assert metadata["quench"]["burn_in_steps"] == 2
    assert metadata["quench"]["classification_valid_after_tstep"] == 4
