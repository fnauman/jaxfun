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


def test_release_gate_default_covers_production_scale_runs(tmp_path):
    """Review blocker 6: release enforcement must not be opt-in for the runs
    that mint release artifacts -- production-scale runs of production DNS specs
    and golden promotion gate by default; bounded dev runs stay permissive."""
    import json

    from production.run_problem import _build_parser, _requires_release_gate

    parser = _build_parser()
    prod_spec = tmp_path / "prod.json"
    prod_spec.write_text(
        json.dumps(
            {
                "support_state": "production",
                "expected_oracle": {"type": "gpu_generated_saturated_dns"},
            }
        ),
        encoding="utf-8",
    )
    exp_spec = tmp_path / "exp.json"
    exp_spec.write_text(
        json.dumps(
            {
                "support_state": "experimental",
                "expected_oracle": {"type": "mri_saturation_ladder"},
            }
        ),
        encoding="utf-8",
    )
    laminar_spec = tmp_path / "laminar.json"
    laminar_spec.write_text(
        json.dumps(
            {
                "support_state": "production",
                "expected_oracle": {"type": "plane_couette_laminar"},
            }
        ),
        encoding="utf-8",
    )
    base = ["--config", "x", "--out", "y"]

    # production-scale run of a production DNS spec -> gate on by default
    assert _requires_release_gate(parser.parse_args(base), prod_spec) is True
    # bounded smoke/dev runs stay permissive
    assert (
        _requires_release_gate(parser.parse_args([*base, "--steps", "2"]), prod_spec)
        is False
    )
    assert (
        _requires_release_gate(
            parser.parse_args([*base, "--resolution-tier", "smoke"]), prod_spec
        )
        is False
    )
    # experimental specs stay permissive unless explicitly gated
    assert _requires_release_gate(parser.parse_args(base), exp_spec) is False
    assert (
        _requires_release_gate(
            parser.parse_args([*base, "--require-clean"]), exp_spec
        )
        is True
    )
    # golden promotion always gates, even on bounded runs
    assert (
        _requires_release_gate(
            parser.parse_args([*base, "--steps", "2", "--write-golden"]), exp_spec
        )
        is True
    )
    # analytic/linear production oracles are not DNS release artifacts
    assert _requires_release_gate(parser.parse_args(base), laminar_spec) is False
