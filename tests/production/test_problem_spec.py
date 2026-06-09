import json
from pathlib import Path

import pytest

from production.problem_spec import (
    UnsupportedSpecError,
    load_spec,
    spec_hash,
    validate_spec,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "production" / "examples"
GOLDENS = ROOT / "production" / "goldens"
PROMOTIONS = ROOT / "production" / "promotions"


NON_PIPE_ACCEPTED = [
    "pcf_hydro_laminar_v1",
    "channel_poiseuille_hydro_v1",
    "pcf_mhd_conducting_v1",
    "pcf_mri_shearbox_v1",
    "pcf_hydro_primitive_dns_v1",
    "pcf_mri_primitive_dns_v1",
    "taylor_couette_hydro_v1",
    "taylor_couette_hydro_dns_v1",
    "taylor_couette_mhd_conducting_v1",
    "taylor_couette_mhd_insulating_v1",
    "taylor_couette_mhd_dns_v1",
]


@pytest.mark.parametrize("problem_id", NON_PIPE_ACCEPTED)
def test_non_pipe_specs_validate_and_match_vendored_golden_hash(problem_id):
    spec = load_spec(EXAMPLES / f"{problem_id}.json")
    assert spec["problem_id"] == problem_id
    golden = json.loads((GOLDENS / problem_id / "golden" / "golden.json").read_text())
    assert spec["spec_hash"] == golden["spec_hash"]


def test_pipe_hydro_rejected_until_axis_regular_basis_lands():
    with pytest.raises(
        UnsupportedSpecError,
        match=(
            "pipe hydro is parity_pending.*axis-regularity.*"
            "pipe_hagen_poiseuille_v1.*pipe_womersley_v1"
        ),
    ):
        load_spec(EXAMPLES / "pipe_hagen_poiseuille_v1.json")

    record = (PROMOTIONS / "pipe_hydro_axis_regular_basis.md").read_text()
    assert "axis-regular radial basis" in record
    assert "singular weighted-Galerkin penalties" in record
    assert "pipe_hagen_poiseuille_v1" in record
    assert "pipe_womersley_v1" in record


def test_pipe_hash_can_still_be_computed_for_vendored_golden_validation():
    spec = load_spec(
        EXAMPLES / "pipe_hagen_poiseuille_v1.json", allow_unimplemented=True
    )
    golden = json.loads(
        (GOLDENS / "pipe_hagen_poiseuille_v1" / "golden" / "golden.json").read_text()
    )
    assert spec["spec_hash"] == golden["spec_hash"]


def test_pipe_mhd_rejection_matches_shenfun_contract():
    with pytest.raises(UnsupportedSpecError, match="pipe MHD/MRI is unsupported"):
        load_spec(EXAMPLES / "unsupported" / "pipe_mhd_unsupported_v1.json")


def test_tc_insulating_m1_rejection_names_axisymmetric_requirement():
    with pytest.raises(UnsupportedSpecError, match="axisymmetric m=0"):
        load_spec(
            EXAMPLES
            / "unsupported"
            / "taylor_couette_mhd_insulating_m1_unsupported_v1.json"
        )


def test_pm_equals_rm_over_re_invariant_is_enforced():
    spec = json.loads((EXAMPLES / "pcf_mhd_conducting_v1.json").read_text())
    spec["nondimensional_groups"]["Pm"] = 2.0
    with pytest.raises(ValueError, match="Pm must equal Rm/Re"):
        validate_spec(spec)


def test_spec_hash_ignores_existing_hash_field():
    spec = json.loads((EXAMPLES / "pcf_hydro_laminar_v1.json").read_text())
    expected = spec_hash(spec)
    spec["spec_hash"] = "not-used"
    assert spec_hash(spec) == expected
