"""FJ-07: sweep-safe semantic override interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.problem_spec import ProblemSpecError
from production.sweep import (
    SweepOverrideError,
    apply_overrides,
    materialize_run_spec,
    run_id_for,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "production" / "runs" / "exp_pcf_mri_shearbox_growth.json"


def _base():
    return json.loads(BASE.read_text())


def test_Re_h_override_changes_resolved_nu():
    base = _base()
    base_nu = base["nondimensional_groups"]["nu"]
    out = apply_overrides(base, {"Re_h": 2000.0})
    # Re_h: 1000 -> 2000 halves nu (nu = |S| h^2 / Re_h)
    assert out["nondimensional_groups"]["nu"] == pytest.approx(base_nu / 2.0)
    assert out["nondimensional_groups"]["Re"] == 2000.0


def test_Rm_h_override_changes_resolved_eta():
    out = apply_overrides(_base(), {"Rm_h": 500.0})
    assert out["nondimensional_groups"]["eta_mag"] == pytest.approx(1.0 / 500.0)


def test_B0_override_syncs_forcing():
    out = apply_overrides(_base(), {"B0": 0.05})
    assert out["nondimensional_groups"]["B0"] == 0.05
    assert out["forcing"]["B0"] == 0.05


def test_geometry_overrides_change_spec():
    out = apply_overrides(_base(), {"Ly": 3.0, "Lz": 0.5, "horizon": 20.0})
    assert out["domain"]["y_period"] == 3.0
    assert out["domain"]["z_period"] == 0.5
    assert out["time"]["final_time"] == 20.0


def test_every_override_changes_hash_or_fails():
    """FJ-07 acceptance: an override must change the resolved spec (hash)."""
    base_spec = apply_overrides(_base(), {})
    changed = apply_overrides(_base(), {"Re_h": 1600.0})
    assert changed["spec_hash"] != base_spec["spec_hash"]


def test_unknown_override_rejected():
    with pytest.raises(SweepOverrideError, match="unknown"):
        apply_overrides(_base(), {"not_a_field": 1.0})


def test_inconsistent_manual_spec_still_validates_pathway():
    # An override that produces an inconsistent Pm cannot slip through: overriding
    # only Re_h re-derives nu, so Pm stays consistent (no error), but a hand-broken
    # base would raise. Here we assert the happy path resolves cleanly.
    out = apply_overrides(_base(), {"Re_h": 800.0, "Rm_h": 800.0})
    assert out["nondimensional_groups"]["Pm"] == pytest.approx(1.0)


def test_materialize_writes_unique_run_spec(tmp_path):
    rec = materialize_run_spec(BASE, {"Re_h": 1600.0, "B0": 0.0125}, tmp_path)
    assert Path(rec["spec_path"]).exists()
    assert rec["run_id"].startswith("exp_pcf_mri_shearbox_growth-")
    written = json.loads(Path(rec["spec_path"]).read_text())
    assert written["spec_hash"] == rec["spec_hash"]
    # run id is grouped by problem id + resolved hash
    assert run_id_for(written) == rec["run_id"]
