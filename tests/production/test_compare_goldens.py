import json
from pathlib import Path

import pytest

from production.compare_goldens import (
    compare_problem,
    compare_to_golden,
    load_golden,
    resolve_golden,
    scalar_hash,
    validate_golden,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "production" / "goldens"


def test_resolves_vendored_golden_without_shenfun_checkout():
    resolution = resolve_golden("pcf_hydro_laminar_v1")
    assert resolution.policy == "vendored"
    assert (
        resolution.golden_path
        == GOLDENS / "pcf_hydro_laminar_v1" / "golden" / "golden.json"
    )


@pytest.mark.parametrize(
    "problem_id",
    [
        "pcf_hydro_laminar_v1",
        "pcf_fluct_re400",
        "pcf_mhd_conducting_v1",
        "pcf_mhd_divfree",
        "pcf_mri_primitive_dns_v1",
        "exp_pcf_mri_shearbox_growth",
        "tc_supercritical_saturation",
        "tc_mri_nonlinear_saturation",
        "taylor_couette_hydro_dns_v1",
        "pipe_hagen_poiseuille_v1",
        "pipe_womersley_v1",
    ],
)
def test_vendored_golden_hash_validates(problem_id):
    path = GOLDENS / problem_id / "golden" / "golden.json"
    golden = validate_golden(path)
    assert (
        scalar_hash(golden["diagnostics"]["scalars"])
        == golden["comparison_fields"]["scalars_sha256"]
    )


def test_comparison_uses_per_observable_tolerances_and_passes_exact_golden_scalars():
    golden = load_golden(GOLDENS / "pcf_mhd_conducting_v1" / "golden" / "golden.json")
    result = compare_problem("pcf_mhd_conducting_v1", golden["diagnostics"]["scalars"])
    assert result.passed
    compared = {item.key for item in result.comparisons}
    assert compared == {
        "divergence_b_l2",
        "divergence_u_l2",
        "eigenvalue_imag",
        "eigenvalue_real",
        "growth_rate",
        "kinetic_energy",
        "magnetic_bc",
        "magnetic_energy",
        "maxwell_stress_xy",
        "total_energy",
    }


def test_missing_numeric_golden_tolerance_fails(tmp_path):
    golden = load_golden(GOLDENS / "pcf_hydro_laminar_v1" / "golden" / "golden.json")
    golden["tolerance_model"]["scalars"].pop("kinetic_energy")
    golden["comparison_fields"]["scalars_sha256"] = scalar_hash(
        golden["diagnostics"]["scalars"]
    )
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    with pytest.raises(ValueError, match="missing tolerance"):
        validate_golden(path)

    result = compare_to_golden(golden["diagnostics"]["scalars"], golden)
    assert not result.passed
    failure = next(item for item in result.comparisons if item.key == "kinetic_energy")
    assert failure.message == "numeric scalar missing tolerance"


def test_invalid_numeric_golden_tolerance_fails(tmp_path):
    golden = load_golden(GOLDENS / "pcf_hydro_laminar_v1" / "golden" / "golden.json")
    golden["tolerance_model"]["scalars"]["kinetic_energy"] = -1.0
    golden["comparison_fields"]["scalars_sha256"] = scalar_hash(
        golden["diagnostics"]["scalars"]
    )
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid tolerance"):
        validate_golden(path)


def test_tolerance_model_hash_tampering_fails(tmp_path):
    golden = load_golden(GOLDENS / "pcf_hydro_laminar_v1" / "golden" / "golden.json")
    golden["comparison_fields"]["tolerance_model_sha256"] = "not-the-hash"
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden), encoding="utf-8")

    with pytest.raises(ValueError, match="tolerance hash"):
        validate_golden(path)


def test_bool_numeric_scalar_mismatch_fails_type_check():
    golden = load_golden(GOLDENS / "pcf_mhd_divfree" / "golden" / "golden.json")
    actual = dict(golden["diagnostics"]["scalars"])
    actual["saturation_check_passed"] = 0.0

    result = compare_to_golden(actual, golden, require_all_golden_scalars=True)

    assert not result.passed
    failure = next(
        item for item in result.comparisons if item.key == "saturation_check_passed"
    )
    assert failure.message == "scalar type mismatch"


@pytest.mark.parametrize(
    ("problem_id", "growth_scalar", "minimum_growth"),
    [
        ("pcf_fluct_re400", "energy_growth_factor", 2.0),
        ("exp_pcf_mri_shearbox_growth", "magnetic_energy_growth_factor", 2.0),
        ("tc_supercritical_saturation", "energy_growth_factor", 2.0),
        ("tc_mri_nonlinear_saturation", "magnetic_energy_growth_factor", 2.0),
    ],
)
def test_promoted_saturation_golden_validates_against_run_spec(
    problem_id, growth_scalar, minimum_growth
):
    root = GOLDENS / problem_id
    spec = json.loads((root / "spec.json").read_text())
    golden = validate_golden(root / "golden" / "golden.json", spec=spec)
    scalars = golden["diagnostics"]["scalars"]

    assert scalars["saturation_check_passed"] is True
    assert scalars[growth_scalar] > minimum_growth
    assert golden["environment"]["jax"]["default_backend"] == "gpu"

    result = compare_problem(
        problem_id,
        scalars,
        require_all_golden_scalars=True,
    )
    assert result.passed


def test_decaying_pcf_mhd_divfree_candidate_is_not_promoted():
    root = GOLDENS / "pcf_mhd_divfree"
    spec = json.loads((root / "spec.json").read_text())
    golden = validate_golden(root / "golden" / "golden.json", spec=spec)
    scalars = golden["diagnostics"]["scalars"]
    metadata = json.loads((root / "metadata.json").read_text())

    assert scalars["magnetic_energy_growth_factor"] < 1.0
    assert scalars["saturation_check_passed"] is False
    assert metadata["saturation_checks"]["passed"] is False
    assert metadata["validation_scope"] == "failed_generated_saturated_candidate"

    actual = dict(scalars)
    actual["saturation_check_passed"] = True
    result = compare_problem(
        "pcf_mhd_divfree",
        actual,
        require_all_golden_scalars=True,
    )
    assert not result.passed
    failure = next(
        item for item in result.comparisons if item.key == "saturation_check_passed"
    )
    assert failure.message == "non-numeric scalar mismatch"


def test_failed_comparison_reports_expected_actual_and_tolerance():
    golden = load_golden(
        GOLDENS / "channel_poiseuille_hydro_v1" / "golden" / "golden.json"
    )
    actual = dict(golden["diagnostics"]["scalars"])
    actual["flow_rate"] += 1.0
    result = compare_problem("channel_poiseuille_hydro_v1", actual)
    assert not result.passed
    failure = next(item for item in result.comparisons if item.key == "flow_rate")
    assert failure.expected == golden["diagnostics"]["scalars"]["flow_rate"]
    assert failure.actual == actual["flow_rate"]
    assert failure.tolerance == golden["tolerance_model"]["scalars"]["flow_rate"]
    assert "exceeds tolerance" in failure.message


def test_missing_actual_scalar_fails_with_key_name():
    golden = load_golden(GOLDENS / "taylor_couette_hydro_v1" / "golden" / "golden.json")
    actual = dict(golden["diagnostics"]["scalars"])
    actual.pop("growth_rate")
    result = compare_problem("taylor_couette_hydro_v1", actual)
    assert not result.passed
    failure = next(item for item in result.comparisons if item.key == "growth_rate")
    assert failure.message == "actual scalar missing"


def test_validate_golden_detects_scalar_hash_tampering(tmp_path):
    golden = load_golden(GOLDENS / "pcf_hydro_laminar_v1" / "golden" / "golden.json")
    golden["diagnostics"]["scalars"]["growth_rate"] += 1.0
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    with pytest.raises(ValueError, match="scalar hash"):
        validate_golden(path)
