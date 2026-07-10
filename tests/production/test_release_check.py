"""FJ-13: release provenance + convention artifact."""

from __future__ import annotations

import json

from production.release_check import (
    CONVENTIONS,
    release_manifest,
    write_release_artifact,
)


def test_conventions_state_axis_orders_and_traps():
    assert CONVENTIONS["coordinate_order"]["pcf_primitive"] == ["y", "z", "x"]
    assert CONVENTIONS["coordinate_order"]["pcf_kmm"] == ["x", "y", "z"]
    assert "S=+1" in CONVENTIONS["signed_shear"]
    assert "half-gap h = 1" in CONVENTIONS["gap"]
    assert "2 q Omega^2" in CONVENTIONS["ideal_mri_cutoff"]


def test_release_manifest_has_provenance_and_conventions():
    manifest = release_manifest()
    assert "provenance" in manifest and "commit" in manifest["provenance"]
    assert "conventions" in manifest
    assert manifest["reference_interpreter"].endswith("python") or "python" in manifest[
        "reference_interpreter"
    ]


def test_skipped_live_parity_is_not_a_pass():
    # live tier skipped (0 passed) must NOT count as a release pass
    m = release_manifest(
        {"failed": 0, "errors": 0, "live_shenfun": {"passed": 0, "skipped": 24}}
    )
    assert m["tests"]["live_shenfun_ran_not_skipped"] is False
    assert m["tests"]["release_test_gate_passed"] is False


def test_ran_live_parity_with_no_failures_passes_gate():
    m = release_manifest(
        {"failed": 0, "errors": 0, "live_shenfun": {"passed": 24, "skipped": 0}}
    )
    assert m["tests"]["release_test_gate_passed"] is True


def test_write_release_artifact(tmp_path):
    out = write_release_artifact(tmp_path / "release.json")
    data = json.loads(out.read_text())
    assert data["conventions"]["coordinate_order"]["pcf_primitive"] == ["y", "z", "x"]
