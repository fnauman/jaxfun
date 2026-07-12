"""FJ-05: quench continuation validation + checkpoint banks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from production.quench import (
    QuenchError,
    burn_in_horizon,
    checkpoint_bank_entry,
    validate_quench,
)
from production.sweep import apply_overrides

ROOT = Path(__file__).resolve().parents[2]
TC_BASE = ROOT / "production" / "runs" / "exp_tc_mri_vector_potential.json"


def _parent():
    return {
        "geometry": "pcf",
        "physics": "mri",
        "representation": "primitive",
        "numerics_contract_version": 2,
        "nondimensional_groups": {
            "Re": 1000.0,
            "Rm": 1000.0,
            "nu": 1e-3,
            "eta_mag": 1e-3,
        },
        "resolution": {"production": {"Nx": 32, "Ny": 32, "Nz": 32}},
        "boundary_conditions": {"magnetic": {"type": "conducting"}},
    }


def test_quench_allows_lowering_Rm_and_eta():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["nondimensional_groups"]["Rm"] = 500.0
    child["nondimensional_groups"]["eta_mag"] = 2e-3
    diff = validate_quench(parent, child)
    assert "nondimensional_groups.Rm" in diff["changed"]
    assert "nondimensional_groups.eta_mag" in diff["changed"]


def test_quench_rejects_raising_Rm_and_lowering_eta():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["nondimensional_groups"]["Rm"] = 2000.0
    child["nondimensional_groups"]["eta_mag"] = 5e-4
    with pytest.raises(QuenchError, match="cannot increase Rm"):
        validate_quench(parent, child)


def test_quench_rejects_raising_Re_and_lowering_nu():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["nondimensional_groups"]["Re"] = 2000.0
    child["nondimensional_groups"]["nu"] = 5e-4
    with pytest.raises(QuenchError, match="cannot increase Re"):
        validate_quench(parent, child)


def test_quench_rejects_resolution_change():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["resolution"]["production"]["Nx"] = 64
    with pytest.raises(QuenchError, match="illegal"):
        validate_quench(parent, child)


def test_quench_rejects_bc_change():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["boundary_conditions"]["magnetic"]["type"] = "pseudo_vacuum"
    with pytest.raises(QuenchError):
        validate_quench(parent, child)


def test_quench_rejects_representation_change():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["representation"] = "vector_potential"
    with pytest.raises(QuenchError, match="immutable field"):
        validate_quench(parent, child)


def test_quench_rejects_contract_version_change():
    parent = _parent()
    child = copy.deepcopy(parent)
    child["numerics_contract_version"] = 3
    with pytest.raises(QuenchError, match="immutable field"):
        validate_quench(parent, child)


def test_quench_rejects_identical_spec():
    parent = _parent()
    with pytest.raises(QuenchError, match="identical"):
        validate_quench(parent, copy.deepcopy(parent))


def test_checkpoint_bank_entry_records_provenance():
    entry = checkpoint_bank_entry(
        parent_run_id="run-A",
        child_run_id=None,
        t=32.0,
        tstep=6400,
        spec_hash="abc",
        representation="primitive",
        numerics_contract_version=2,
        checkpoint_path="checkpoints/step_000006400.h5",
        plateau_stats={"mean_Emag": 0.02},
    )
    assert entry["parent_run_id"] == "run-A"
    assert entry["state_time"] == 32.0
    assert entry["plateau_window_stats"]["mean_Emag"] == 0.02


def test_burn_in_horizon_quarantines_inherited_history():
    h = burn_in_horizon(tstep0=6400, burn_in_steps=1000)
    assert h["classification_valid_after_tstep"] == 7400


def _tc_parent(*, materialized: bool = True):
    spec = json.loads(TC_BASE.read_text())
    return apply_overrides(spec, {}) if materialized else spec


def test_tc_quench_allows_coupled_local_native_and_legacy_rm_aliases():
    parent = _tc_parent()
    child = apply_overrides(
        parent,
        {"Rm_h": 0.5 * parent["nondimensional_groups"]["Rm_h"]},
    )

    result = validate_quench(parent, child)
    changed = result["changed"]
    for key in ("Rm_h", "Rm_TC", "Rm", "eta_mag"):
        assert f"nondimensional_groups.{key}" in changed
    assert changed["nondimensional_groups.Rm_h"] == pytest.approx(
        (
            parent["nondimensional_groups"]["Rm_h"],
            child["nondimensional_groups"]["Rm_h"],
        )
    )


def test_tc_quench_canonicalizes_legacy_parent_before_alias_diff():
    parent = _tc_parent(materialized=False)
    child = apply_overrides(parent, {"Rm_h": 100.0})

    result = validate_quench(parent, child)
    assert "nondimensional_groups.Rm_h" in result["changed"]
    assert "nondimensional_groups.Rm_TC" in result["changed"]


def test_tc_quench_rejects_inconsistent_coupled_alias():
    parent = _tc_parent()
    child = apply_overrides(parent, {"Rm_h": 100.0})
    child["nondimensional_groups"]["Rm_TC"] *= 0.9

    with pytest.raises(QuenchError, match="invalid Taylor-Couette quench physics"):
        validate_quench(parent, child)


def test_tc_quench_rejects_increasing_local_and_native_reynolds_numbers():
    parent = _tc_parent()
    child = apply_overrides(
        parent,
        {"Re_h": 2.0 * parent["nondimensional_groups"]["Re_h"]},
    )

    with pytest.raises(QuenchError, match="cannot increase"):
        validate_quench(parent, child)
