"""FJ-05: quench continuation validation + checkpoint banks."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from production.quench import (
    QuenchError,
    burn_in_horizon,
    checkpoint_bank_entry,
    finalize_fixed_quench_duration,
    resolve_fixed_quench_duration,
    validate_bank_checkpoint_record,
    validate_burn_in_request,
    validate_quench,
    validate_quench_duration_request,
    validate_quench_output_options,
    validate_quench_runner_preflight,
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quench": True}, "requires exactly one"),
        (
            {"quench": True, "steps": 10, "additional_steps": 2},
            "cannot be used for a quench",
        ),
        (
            {"quench": True, "additional_time": 0.1, "additional_steps": 2},
            "requires exactly one",
        ),
        (
            {"quench": False, "additional_steps": 2},
            "require an explicit quench",
        ),
    ],
)
def test_quench_duration_request_rejects_ambiguous_modes(kwargs, message):
    options = {
        "quench": False,
        "steps": None,
        "additional_time": None,
        "additional_steps": None,
        **kwargs,
    }
    with pytest.raises(QuenchError, match=message):
        validate_quench_duration_request(**options)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_quench_additional_time_must_be_finite_and_positive(value):
    with pytest.raises(QuenchError, match="finite and strictly positive"):
        validate_quench_duration_request(
            quench=True,
            steps=None,
            additional_time=value,
            additional_steps=None,
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_quench_additional_steps_must_be_a_positive_integer(value):
    with pytest.raises(QuenchError, match="positive integer"):
        validate_quench_duration_request(
            quench=True,
            steps=None,
            additional_time=None,
            additional_steps=value,
        )


def test_burn_in_contract_requires_integer_and_zero_outside_quench():
    validate_burn_in_request(quench=False, burn_in_steps=0)
    for invalid in (True, 1.5):
        with pytest.raises(QuenchError, match="must be an integer"):
            validate_burn_in_request(quench=True, burn_in_steps=invalid)
    for nonzero in (-1, 1):
        with pytest.raises(QuenchError, match="must be 0 outside"):
            validate_burn_in_request(quench=False, burn_in_steps=nonzero)


def test_burn_in_contract_is_strictly_inside_resolved_quench_horizon():
    validate_burn_in_request(quench=True, burn_in_steps=0, resolved_additional_steps=4)
    validate_burn_in_request(quench=True, burn_in_steps=3, resolved_additional_steps=4)
    with pytest.raises(QuenchError, match="must be non-negative"):
        validate_burn_in_request(
            quench=True, burn_in_steps=-1, resolved_additional_steps=4
        )
    for invalid in (4, 5):
        with pytest.raises(QuenchError, match="strictly less"):
            validate_burn_in_request(
                quench=True,
                burn_in_steps=invalid,
                resolved_additional_steps=4,
            )


def test_quench_golden_options_are_explicitly_unsupported():
    validate_quench_output_options(quench=False, compare_golden=True, write_golden=True)
    for option in ("compare_golden", "write_golden"):
        values = {"compare_golden": False, "write_golden": False, option: True}
        with pytest.raises(QuenchError, match=option):
            validate_quench_output_options(quench=True, **values)


@pytest.mark.parametrize(
    "path",
    [
        "production/runs/pcf_mhd_divfree.json",
        "production/runs/exp_pcf_mri_pseudo_vacuum.json",
        "production/runs/exp_pcf_mri_vector_potential.json",
        "production/runs/exp_tc_mri_vector_potential.json",
    ],
)
def test_quench_runner_preflight_accepts_only_wired_saturation_families(path):
    spec = json.loads((Path(__file__).resolve().parents[2] / path).read_text())
    validate_quench_runner_preflight(spec, quench=True)


def test_quench_runner_preflight_rejects_unwired_saturation_runner():
    spec = json.loads(
        (
            Path(__file__).resolve().parents[2] / "production/runs/pcf_fluct_re400.json"
        ).read_text()
    )
    with pytest.raises(QuenchError, match="quench runner is not implemented"):
        validate_quench_runner_preflight(spec, quench=True)


def test_fixed_quench_duration_resolves_absolute_target_from_parent():
    duration = resolve_fixed_quench_duration(
        additional_time=0.3,
        additional_steps=None,
        child_dt=0.1,
        parent_time=1.25,
        parent_step=7,
    )

    assert duration.additional_steps == 3
    assert duration.additional_time == pytest.approx(0.3)
    assert duration.target_step == 10
    assert duration.target_time == pytest.approx(1.55)
    metadata = duration.to_metadata()
    assert metadata["schema_version"] == 1
    assert metadata["request_kind"] == "additional_time"
    assert metadata["requested"] == {
        "additional_time": 0.3,
        "additional_steps": None,
    }
    assert metadata["attained"]["target_reached"] is None


def test_fixed_quench_duration_rejects_nonintegral_or_substep_time():
    for additional_time in (0.15, 1.0e-20):
        with pytest.raises(QuenchError, match="integer multiple"):
            resolve_fixed_quench_duration(
                additional_time=additional_time,
                additional_steps=None,
                child_dt=0.1,
                parent_time=0.0,
                parent_step=0,
            )


def test_fixed_quench_duration_rejects_nonintegral_parent_step():
    with pytest.raises(QuenchError, match="parent checkpoint step must be an integer"):
        resolve_fixed_quench_duration(
            additional_time=None,
            additional_steps=2,
            child_dt=0.1,
            parent_time=0.0,
            parent_step=3.5,
        )


def _bank_record(*, t=1.25, tstep=7, spec_hash="parent", version=2):
    return SimpleNamespace(
        t=t,
        tstep=tstep,
        attrs={
            "spec_hash": spec_hash,
            "numerics_contract_version": version,
        },
    )


def _bank_entry(*, state_time=1.25, tstep=7, spec_hash="parent", version=2):
    return {
        "state_time": state_time,
        "tstep": tstep,
        "spec_hash": spec_hash,
        "numerics_contract_version": version,
    }


def test_bank_entry_must_match_loaded_checkpoint_contract():
    validate_bank_checkpoint_record(
        _bank_entry(), _bank_record(), parent_spec_hash="parent"
    )
    with pytest.raises(QuenchError, match="tstep 8 != 7"):
        validate_bank_checkpoint_record(_bank_entry(tstep=8), _bank_record())
    with pytest.raises(QuenchError, match="time"):
        validate_bank_checkpoint_record(_bank_entry(state_time=1.5), _bank_record())
    with pytest.raises(QuenchError, match="spec_hash"):
        validate_bank_checkpoint_record(_bank_entry(spec_hash="other"), _bank_record())
    with pytest.raises(QuenchError, match="must be integers"):
        validate_bank_checkpoint_record(_bank_entry(tstep=7.5), _bank_record())


def test_attained_duration_uses_actual_solver_time_and_step():
    duration = resolve_fixed_quench_duration(
        additional_time=None,
        additional_steps=3,
        child_dt=0.1,
        parent_time=1.25,
        parent_step=7,
    ).to_metadata()
    attained = finalize_fixed_quench_duration(duration, final_time=1.55, final_step=10)
    assert attained["attained"] == {
        "final_time": 1.55,
        "final_step": 10,
        "additional_time": pytest.approx(0.3),
        "additional_steps": 3,
        "target_reached": True,
    }

    wrong_step = finalize_fixed_quench_duration(duration, final_time=1.55, final_step=9)
    assert wrong_step["attained"]["final_step"] == 9
    assert wrong_step["attained"]["target_reached"] is False


def test_attained_duration_allows_step_scaled_long_run_time_drift():
    additional_steps = 2_000_000
    duration = resolve_fixed_quench_duration(
        additional_time=None,
        additional_steps=additional_steps,
        child_dt=1.0e-6,
        parent_time=1.25,
        parent_step=7,
    ).to_metadata()
    target_time = duration["absolute_target"]["time"]
    scale = max(1.0, abs(target_time))
    accumulated_drift = 4.0 * additional_steps * math.ulp(scale)

    attained = finalize_fixed_quench_duration(
        duration,
        final_time=target_time + accumulated_drift,
        final_step=duration["absolute_target"]["step"],
    )

    assert attained["attained"]["target_reached"] is True


def test_attained_duration_rejects_time_error_beyond_roundoff_envelope():
    duration = resolve_fixed_quench_duration(
        additional_time=None,
        additional_steps=2_000_000,
        child_dt=1.0e-6,
        parent_time=1.25,
        parent_step=7,
    ).to_metadata()

    attained = finalize_fixed_quench_duration(
        duration,
        final_time=duration["absolute_target"]["time"] + 1.0e-3,
        final_step=duration["absolute_target"]["step"],
    )

    assert attained["attained"]["target_reached"] is False


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
