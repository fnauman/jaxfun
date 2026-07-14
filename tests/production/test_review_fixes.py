"""Regression tests for the review-fix pass (F3, F4, F6, F7, F8)."""

from __future__ import annotations

import pytest

from production.problem_spec import ProblemSpecError
from production.run_problem import (
    _assert_precision_matches_spec,
    _integrator_provenance,
    _resolved_physics_metadata,
)


def test_integrator_provenance_records_imexrk222_for_vector_potential():
    """F8: the curl path runs IMEXRK222, not the primitive CNAB2."""
    spec = {
        "representation": "vector_potential",
        "physics": "mri",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "time": {"integrator": "IMEXRK222", "dt": 0.005},
    }
    prov = _integrator_provenance(spec)
    assert prov["actual"] == "IMEXRK222"
    assert prov["formal_order"] == 2


def test_integrator_provenance_records_cnab2_for_primitive():
    spec = {
        "representation": "primitive",
        "physics": "mri",
        "expected_oracle": {"type": "mri_saturation_ladder"},
        "time": {"integrator": "IMEXRK222", "dt": 0.005},
    }
    prov = _integrator_provenance(spec)
    assert prov["actual"] == "CNAB2"  # hard-coded in the primitive solver
    assert prov["formal_order"] == 2


def test_resolved_physics_metadata_records_actual_precision():
    """F4: resolved physics precision must match the active run dtype."""
    spec = {
        "geometry": "pcf",
        "physics": "mri",
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 1e-3,
            "eta_mag": 1e-3,
            "Re": 1000.0,
            "Rm": 1000.0,
            "Pm": 1.0,
            "B0": 0.025,
        },
        "boundary_conditions": {"velocity": {}, "magnetic": {"type": "conducting"}},
        "forcing": {"B0": 0.025},
    }
    meta = _resolved_physics_metadata(spec, precision="float32")
    assert meta["precision"] == "float32"
    # default (no precision passed) still resolves
    assert _resolved_physics_metadata(spec)["precision"] == "float64"


def test_precision_mismatch_is_rejected():
    """F3: a spec precision that disagrees with the active dtype must fail loudly."""
    spec = {"precision": "float64"}
    with pytest.raises(ProblemSpecError, match="precision"):
        _assert_precision_matches_spec(spec, {"production_run_dtype": "float32"})
    # matching precision is fine
    _assert_precision_matches_spec(
        {"precision": "float32"}, {"production_run_dtype": "float32"}
    )
    # no declared precision -> no constraint
    _assert_precision_matches_spec({}, {"production_run_dtype": "float32"})


def test_release_ref_and_untracked_archive(tmp_path):
    """F6/F7: the strict gate reports the release-ref status and archives untracked."""
    from production.provenance import _head_release_ref, assert_release_clean

    ref = _head_release_ref()
    assert set(ref) == {
        "exact_tag",
        "tag_commit",
        "remote",
        "remote_tag_commit",
        "remote_verified",
        "is_immutable_ref",
    }
    assert ref["is_immutable_ref"] is ref["remote_verified"]

    # The working tree during development is dirty/unpinned, so allow_dirty archives.
    prov = assert_release_clean(tmp_path, allow_dirty=True)
    gate = prov["release_gate"]
    if gate["discovery_only"]:
        assert "diff_sha256" in gate
        assert "untracked_archived" in gate  # F7: untracked list is captured
