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
    select_qualified_parent_checkpoint,
)
from production.quench import QuenchError, file_sha256

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
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 2e-2,
            "eta_mag": 2e-2,
            "Re": 50.0,
            "Rm": 50.0,
            "Pm": 1.0,
            "B0": 0.05,
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
        assert (run_dir / entry["checkpoint_path"]).exists()
        assert entry["plateau_qualified"] is False
        assert entry["selection_status"] == "quarantined"
        assert entry["plateau_window_stats"]["qualification_reasons"]

    with pytest.raises(QuenchError, match="not plateau-qualified"):
        select_qualified_parent_checkpoint(run_dir, step=2)

    # The latest file still resumes to the final step; step= selects a banked
    # plateau instead.
    assert load_resume_checkpoint(run_dir).tstep == 4
    assert load_resume_checkpoint(run_dir, step=2).tstep == 2


def test_qualified_relative_bank_path_resolves_from_parent_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "parent"
    bank_dir = run_dir / "checkpoints" / "bank"
    bank_dir.mkdir(parents=True)
    target = bank_dir / "checkpoint_00000002.h5"
    target.write_bytes(b"immutable checkpoint payload")
    stats = {
        "plateau_qualified": True,
        "diagnostics_current": True,
        "stationary": True,
        "persistent_stress": True,
        "checkpoint_health_underresolved": False,
        "correlation_time_total_stress": 1.0,
        "effective_independent_samples": 5.0,
        "required_independent_samples": 5.0,
        "qualification_reasons": [],
    }
    entry = {
        "tstep": 2,
        "plateau_qualified": True,
        "checkpoint_path": "checkpoints/bank/checkpoint_00000002.h5",
        "file_sha256": file_sha256(str(target)),
        "plateau_window_stats": stats,
    }
    (bank_dir / "index.json").write_text(json.dumps([entry]), encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    selected = select_qualified_parent_checkpoint(run_dir, step=2)
    assert selected["tstep"] == 2


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
        additional_steps=6,
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
        r for r in out["time_series"] if r["t"] >= sc["analysis_t_start"] - 1e-12
    ]
    assert sc["stationarity_window_samples"] <= len(fit_rows)


def test_cli_parent_bank_quench_workflow(tmp_path, monkeypatch):
    """End-to-end: parent runs with --checkpoint-bank, child quenches from a
    selected plateau step with a burn-in horizon."""
    jax.config.update("jax_enable_x64", True)
    monkeypatch.setenv("JAXFUN_PRODUCTION_DTYPE", "float64")
    from production.run_problem import main

    parent_config = json.loads(
        (ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json").read_text(
            encoding="utf-8"
        )
    )
    parent_config["resolution"] = {
        "Nx": 17,
        "Ny": 8,
        "Nz": 16,
        "family": "L",
        "dealias": {"x": 1.0, "y": 1.0, "z": 1.0},
    }
    parent_config["time"]["dt"] = 1e-3
    parent_config_path = tmp_path / "parent_config.json"
    parent_config_path.write_text(json.dumps(parent_config), encoding="utf-8")

    parent_dir = tmp_path / "parent"
    rc = main(
        [
            "--config",
            str(parent_config_path),
            "--out",
            str(parent_dir),
            "--steps",
            "4",
            "--checkpoint-every",
            "2",
            "--checkpoint-bank",
        ]
    )
    assert rc == 0
    assert (parent_dir / "checkpoints" / "bank" / "checkpoint_00000002.h5").exists()

    # A four-step smoke cannot establish a plateau.  Promote the synthetic test
    # entry explicitly so this test exercises the positive CLI/provenance path;
    # the rejection path is covered above against the unmodified manifest.
    index_path = parent_dir / "checkpoints" / "bank" / "index.json"
    bank = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in bank if item["tstep"] == 2)
    entry["plateau_qualified"] = True
    entry["selection_status"] = "eligible"
    entry["plateau_window_stats"] = {
        "plateau_qualified": True,
        "diagnostics_current": True,
        "effective_independent_samples": 5.0,
        "required_independent_samples": 5.0,
        "correlation_time_total_stress": 1.0,
        "stationary": True,
        "persistent_stress": True,
        "checkpoint_health_underresolved": False,
        "qualification_reasons": [],
    }
    index_path.write_text(json.dumps(bank), encoding="utf-8")

    # Child spec: the same materialized smoke spec with only Rm/eta/Pm changed.
    parent_spec = json.loads((parent_dir / "spec.json").read_text(encoding="utf-8"))
    child_spec = json.loads(json.dumps(parent_spec))
    child_spec["nondimensional_groups"]["Rm"] = 800.0
    child_spec["nondimensional_groups"]["eta_mag"] = 1.0 / 800.0
    child_spec["nondimensional_groups"]["Pm"] = 800.0 / 1000.0
    child_spec.pop("spec_hash", None)
    child_path = tmp_path / "child_spec.json"
    child_path.write_text(json.dumps(child_spec), encoding="utf-8")

    validate_dir = tmp_path / "child-validate"
    rc = main(
        [
            "--config",
            str(child_path),
            "--out",
            str(validate_dir),
            "--additional-steps",
            "6",
            "--quench",
            str(parent_dir),
            "--quench-step",
            "2",
            "--validate-only",
        ]
    )
    assert rc == 0
    validated = json.loads((validate_dir / "metadata.json").read_text(encoding="utf-8"))
    assert validated["quench"]["duration"]["attained"] == {
        "final_time": None,
        "final_step": None,
        "additional_time": None,
        "additional_steps": None,
        "target_reached": None,
    }

    child_dir = tmp_path / "child"
    rc = main(
        [
            "--config",
            str(child_path),
            "--out",
            str(child_dir),
            "--additional-steps",
            "6",
            "--diagnostics-every",
            "1",
            "--quench",
            str(parent_dir),
            "--quench-step",
            "2",
            "--burn-in-steps",
            "2",
            "--allow-dirty",
        ]
    )
    assert rc == 0
    metadata = json.loads((child_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["quench"]["mode"] == "quench"
    assert metadata["quench"]["selected_tstep"] == 2
    assert metadata["quench"]["requested_quench_step"] == 2
    assert metadata["quench"]["parent_plateau_qualified"] is True
    assert metadata["quench"]["parent_plateau_window_stats"]["stationary"] is True
    assert metadata["quench"]["burn_in_steps"] == 2
    assert metadata["quench"]["classification_valid_after_tstep"] == 4
    duration = metadata["quench"]["duration"]
    assert duration["schema_version"] == 1
    assert duration["stepping"] == "fixed"
    assert duration["request_kind"] == "additional_steps"
    assert duration["parent_checkpoint"] == {"time": 0.002, "step": 2}
    assert duration["requested"] == {
        "additional_time": None,
        "additional_steps": 6,
    }
    assert duration["absolute_target"] == {"time": 0.008, "step": 8}
    assert duration["attained"] == {
        "final_time": 0.008,
        "final_step": 8,
        "additional_time": 0.006,
        "additional_steps": 6,
        "target_reached": True,
    }
