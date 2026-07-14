from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from production.comparison_manifest import (
    BUILDER_VERSION,
    RELATION_PCF_TC,
    RELATION_SHEARBOX_PCF,
    SCHEMA_VERSION,
    build_comparison_manifest,
    canonical_sha256,
    main,
)
from production.physics import resolve_physics

ROOT = Path(__file__).resolve().parents[2]
JAXFUN_REPOSITORY = "https://github.com/fnauman/jaxfun"
SHEARPY_REPOSITORY = "https://github.com/fnauman/shearpy-jimenez"
JAXFUN_COMMIT = "a" * 40
SHEARPY_COMMIT = "b" * 40


def _spec(name: str) -> dict:
    return json.loads((ROOT / "production" / "runs" / name).read_text("utf-8"))


def _shearpy_manifest() -> dict:
    return {
        "schema_version": "shearpy.run_manifest.v2",
        "campaign_preset": "mri_production_v1",
        "domain": "shearing_periodic",
        "evolution": "mhd",
        "shear": 1.0,
        "omega": 2.0 / 3.0,
        "q": 1.5,
        "re": 1000.0,
        "rm": 1000.0,
        "pm": 1.0,
        "nu": 0.001,
        "eta": 0.001,
        "box_lengths": [2.0, 4.0, 1.0],
        "mean_magnetic_field": [0.0, 0.0, 0.025],
    }


def _shearbox_pcf(**overrides) -> dict:
    kwargs = {
        "relation": RELATION_SHEARBOX_PCF,
        "left": _shearpy_manifest(),
        "right": _spec("exp_pcf_mri_vector_potential.json"),
        "left_repository": SHEARPY_REPOSITORY,
        "left_commit": SHEARPY_COMMIT,
        "right_repository": JAXFUN_REPOSITORY,
        "right_commit": JAXFUN_COMMIT,
    }
    kwargs.update(overrides)
    return build_comparison_manifest(**kwargs)


def test_shearbox_pcf_manifest_has_stable_ids_and_exact_conventions():
    manifest = _shearbox_pcf()

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["builder_version"] == BUILDER_VERSION
    assert manifest["comparison_id"].startswith("comparison-sha256:")
    assert manifest["pair_id"].startswith("pair-sha256:")
    assert manifest["mapping"]["background_flow"] == "U_y(x)=-S*x"
    assert manifest["mapping"]["signed_shear"] == "S=-d(U_y)/dx"
    assert "not asserted to be equivalent" in manifest["mapping"]["scope"]

    shearbox, pcf = manifest["endpoints"]
    assert shearbox["kind"] == "shearing_box"
    assert pcf["kind"] == "pcf"
    assert shearbox["controls"]["h"] == pytest.approx(1.0)
    assert shearbox["controls"]["Re_h"] == pytest.approx(1000.0)
    assert shearbox["controls"]["B0_over_U0"] == pytest.approx(0.025)
    assert pcf["controls"]["q"] == pytest.approx(1.5)
    assert pcf["controls"]["Ly_over_h"] == pytest.approx(4.0)

    # shearpy is already a physical volume mean. The PCF vector-potential
    # diagnostic is integral(|field|^2), so physical mean energy is E/(2V).
    assert shearbox["observable_adapter"]["raw_energy_to_normalized"] == 1.0
    source_keys = shearbox["observable_adapter"]["source_keys"]
    assert source_keys["kinetic_energy_fluct"] == "energy.kin_fluct"
    assert source_keys["magnetic_energy_fluct"] == "energy.mag_fluct"
    assert source_keys["growth_rate_mag_fluct"] == "derive:0.5*growth.mag_fluct"
    assert pcf["observable_adapter"]["volume"] == pytest.approx(8.0)
    assert pcf["observable_adapter"]["raw_energy_to_normalized"] == pytest.approx(
        0.5 / 8.0
    )
    assert pcf["observable_adapter"]["source_keys"]["maxwell_stress"] == (
        "maxwell_stress_xy"
    )
    assert manifest["observables"]["observables"]["alpha_Sh"] == "alpha_Sh=Txy_star"


def test_manifest_is_invariant_to_json_object_key_order():
    left = _shearpy_manifest()
    right = _spec("exp_pcf_mri_vector_potential.json")
    reordered_left = dict(reversed(list(left.items())))
    reordered_right = dict(reversed(list(right.items())))

    first = _shearbox_pcf(left=left, right=right)
    second = _shearbox_pcf(left=reordered_left, right=reordered_right)

    assert first == second


def test_pair_id_changes_with_inputs_but_comparison_id_is_contract_stable():
    first = _shearbox_pcf()
    changed = _shearpy_manifest()
    changed["mean_magnetic_field"] = [0.0, 0.0, 0.05]
    second = _shearbox_pcf(left=changed)

    assert first["comparison_id"] == second["comparison_id"]
    assert first["pair_id"] != second["pair_id"]
    assert first["endpoints"][0]["provenance"]["input_sha256"] == canonical_sha256(
        _shearpy_manifest()
    )


def test_local_pcf_tc_controls_come_from_authoritative_physics_resolver():
    pcf_spec = _spec("exp_pcf_mri_vector_potential.json")
    tc_spec = _spec("exp_tc_mri_vector_potential.json")
    manifest = build_comparison_manifest(
        relation=RELATION_PCF_TC,
        left=pcf_spec,
        right=tc_spec,
        left_repository=JAXFUN_REPOSITORY,
        left_commit=JAXFUN_COMMIT,
        right_repository=JAXFUN_REPOSITORY,
        right_commit=JAXFUN_COMMIT,
    )

    _, tc = manifest["endpoints"]
    resolved = resolve_physics(tc_spec)
    controls = tc["controls"]
    assert controls["h"] == pytest.approx((resolved.R2 - resolved.R1) / 2.0)
    assert controls["S"] == pytest.approx(resolved.S_mid)
    assert controls["Omega"] == pytest.approx(resolved.Omega_mid)
    assert controls["q"] == pytest.approx(resolved.q_mid)
    assert controls["Re_h"] == pytest.approx(resolved.Re_h)
    assert controls["Rm_h"] == pytest.approx(resolved.Rm_h)
    assert controls["curvature"] == pytest.approx(resolved.h / resolved.r_mid)
    assert controls["Ly_over_h"] == pytest.approx(
        resolved.theta_period * resolved.r_mid / resolved.h
    )
    expected_volume = (
        0.5 * resolved.theta_period * (resolved.R2**2 - resolved.R1**2) * resolved.Lz
    )
    assert tc["observable_adapter"]["volume"] == pytest.approx(expected_volume)
    assert tc["observable_adapter"]["raw_energy_to_normalized"] == pytest.approx(
        1.0 / expected_volume / resolved.U0**2
    )
    assert manifest["mapping"]["axes"]["pcf_y"] == "r_mid*theta"


@pytest.mark.parametrize("commit", ["main", "abc123", "g" * 40, "a" * 39])
def test_provenance_requires_a_full_object_id(commit):
    with pytest.raises(ValueError, match="full 40- or 64-hex"):
        _shearbox_pcf(left_commit=commit)


def test_inconsistent_shearpy_q_is_rejected():
    left = _shearpy_manifest()
    left["q"] = 2.0
    with pytest.raises(ValueError, match="q=S/Omega"):
        _shearbox_pcf(left=left)


def test_cli_output_is_byte_deterministic(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    left.write_text(json.dumps(_shearpy_manifest()), "utf-8")
    right.write_text(json.dumps(_spec("exp_pcf_mri_vector_potential.json")), "utf-8")
    args = [
        "--relation",
        RELATION_SHEARBOX_PCF,
        "--left",
        str(left),
        "--right",
        str(right),
        "--left-repository",
        SHEARPY_REPOSITORY,
        "--left-commit",
        SHEARPY_COMMIT,
        "--right-repository",
        JAXFUN_REPOSITORY,
        "--right-commit",
        JAXFUN_COMMIT,
    ]

    assert main([*args, "--out", str(first)]) == 0
    assert main([*args, "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")


def test_nonfinite_input_is_rejected_before_hashing():
    left = copy.deepcopy(_shearpy_manifest())
    left["nu"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _shearbox_pcf(left=left)
