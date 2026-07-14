"""Strict immutable-release promotion contract regressions (issue #25)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from production import promotion
from production.promotion import PromotionError

_COMMIT = "a" * 40


def _write_project(root: Path, *, git_ref: str = _COMMIT) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "\n".join(
            (
                "version = 1",
                "revision = 3",
                'requires-python = ">=3.12"',
                "",
                "[[package]]",
                'name = "demo"',
                'version = "1.0.0"',
                'source = { registry = "https://pypi.org/simple" }',
                "",
                "[[package]]",
                'name = "remote-demo"',
                'version = "1.0.0"',
                f'source = {{ git = "https://example.invalid/demo#{git_ref}" }}',
                "",
            )
        ),
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_ref() -> dict[str, object]:
    return {
        "exact_tag": "production-v1",
        "tag_commit": _COMMIT,
        "remote": "origin",
        "remote_tag_commit": _COMMIT,
        "remote_verified": True,
        "is_immutable_ref": True,
    }


def _write_run(root: Path, lock_root: Path) -> Path:
    run = root / "run"
    (run / "golden").mkdir(parents=True)
    spec = {
        "problem_id": "qualified",
        "spec_hash": "spec-123",
        "time": {"dt": 1.0, "final_time": 2.0},
    }
    metadata = {
        "problem_id": "qualified",
        "artifact_id": "qualified",
        "spec_hash": "spec-123",
        "execution": {"status": "completed"},
        "validation_scope": {
            "kind": "generated_saturated_golden",
            "bounded_smoke": False,
        },
        "validation_floor": {
            "required": True,
            "passed": True,
            "divergence_limit": 1.0e-2,
        },
        "saturation_checks": {
            "required": True,
            "present": True,
            "type_valid": True,
            "passed": True,
            "stationarity_check_passed": True,
        },
        "classification": {
            "scientific_class": "sustained",
            "underresolved": False,
            "stationary": True,
            "persistent_stress": True,
            "independently_sampled": True,
            "fit": {"samples": 8},
        },
        "provenance": {
            "commit": _COMMIT,
            "lockfile_sha256": {
                "uv.lock": _file_hash(lock_root / "uv.lock"),
                "pyproject.toml": _file_hash(lock_root / "pyproject.toml"),
            },
            "release_gate": {
                "passed": True,
                "release_ref": _release_ref(),
            },
        },
    }
    rows = [
        {"t": 0.0, "divergence_b_l2": 1.0e-13},
        {"t": 1.0, "divergence_b_l2": 2.0e-13},
        {
            "t": 2.0,
            "divergence_b_l2": 3.0e-13,
            "cfl_total": 0.4,
            "spectral_tail_max": 2.0e-4,
            "mode_occupancy": 0.8,
            "energy_budget_residual": 2.0e-3,
        },
    ]
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run / "golden" / "golden.json").write_text(
        json.dumps(
            {
                "problem_id": "qualified",
                "artifact_id": "qualified",
                "spec_hash": "spec-123",
            }
        ),
    )
    return run


def _write_test_evidence(root: Path) -> tuple[Path, Path]:
    summary = root / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "passed": 120,
                "failed": 0,
                "errors": 0,
                "live_shenfun": {"passed": 4, "skipped": 0},
            }
        ),
        encoding="utf-8",
    )
    junit = root / "junit.xml"
    junit.write_text(
        '<testsuite tests="124" failures="0" errors="0"/>\n', encoding="utf-8"
    )
    return summary, junit


def test_head_release_ref_requires_matching_remote_tag(monkeypatch):
    from production import provenance

    responses = {
        ("rev-parse", "HEAD"): _COMMIT,
        ("describe", "--exact-match", "--tags", "HEAD"): "production-v1",
        ("rev-list", "-n", "1", "production-v1"): _COMMIT,
        (
            "ls-remote",
            "--tags",
            "origin",
            "refs/tags/production-v1",
            "refs/tags/production-v1^{}",
        ): f"{_COMMIT}\trefs/tags/production-v1",
    }
    monkeypatch.setattr(provenance, "_git", lambda *args: responses.get(args))
    assert provenance._head_release_ref()["remote_verified"] is True

    responses[
        (
            "ls-remote",
            "--tags",
            "origin",
            "refs/tags/production-v1",
            "refs/tags/production-v1^{}",
        )
    ] = f"{'b' * 40}\trefs/tags/production-v1"
    ref = provenance._head_release_ref()
    assert ref["remote_verified"] is False
    assert ref["is_immutable_ref"] is False


def test_dependency_audit_rejects_movable_git_source(tmp_path):
    _write_project(tmp_path, git_ref="main")
    with pytest.raises(PromotionError, match="full commit"):
        promotion.audit_dependency_lock(tmp_path)


def test_run_audit_checks_mid_horizon_constraints(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write_project(project)
    run = _write_run(tmp_path, project)

    audit = promotion.audit_run(run)
    assert audit["horizon"] == {"start": 0.0, "end": 2.0}
    assert audit["classification"]["scientific_class"] == "sustained"

    rows = [
        json.loads(line)
        for line in (run / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["divergence_b_l2"] = 0.02
    (run / "diagnostics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(PromotionError, match="row 1 violates"):
        promotion.audit_run(run)


def test_run_audit_requires_whole_horizon_health_budget_and_classification(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    _write_project(project)
    run = _write_run(tmp_path, project)

    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metadata["classification"]["independently_sampled"] = False
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PromotionError, match="independent samples"):
        promotion.audit_run(run)


def test_test_summary_requires_live_parity_to_run(tmp_path):
    summary, _ = _write_test_evidence(tmp_path)
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["live_shenfun"] = {"passed": 0, "skipped": 4}
    summary.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PromotionError, match="live_shenfun"):
        promotion.audit_test_summary(summary)


def test_release_bundle_archives_and_hashes_real_evidence(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    _write_project(project)
    run = _write_run(tmp_path, project)
    summary, junit = _write_test_evidence(tmp_path)
    out = tmp_path / "release"

    monkeypatch.setattr(promotion, "_head_release_ref", lambda _remote: _release_ref())
    monkeypatch.setattr(
        promotion,
        "capture_provenance",
        lambda: {
            "commit": _COMMIT,
            "dirty": False,
            "unpushed_commits": [],
            "remote_url": "git@example.invalid:demo.git",
        },
    )

    written = promotion.build_release_bundle(
        run_dir=run,
        test_summary=summary,
        test_artifacts=[junit],
        out_dir=out,
        repo_root=project,
    )
    assert written == out
    manifest = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert manifest["contract_version"] == "production-promotion-v1"
    assert manifest["release_ref"]["remote_verified"] is True
    assert manifest["run_audit"]["row_count"] == 3
    assert (out / "tests" / "001-junit.xml").read_bytes() == junit.read_bytes()
    for relative, expected in manifest["files"].items():
        assert _file_hash(out / relative) == expected["sha256"]

    with pytest.raises(PromotionError, match="already exists"):
        promotion.build_release_bundle(
            run_dir=run,
            test_summary=summary,
            test_artifacts=[junit],
            out_dir=out,
            repo_root=project,
        )
