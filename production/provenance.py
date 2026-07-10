"""Run provenance and release-cleanliness gate (FJ-13).

Every production manifest must record enough to reproduce the run from an
immutable commit: the git commit, branch (informational), release tag, remote URL,
dirty-state flag, dependency-lock hash, and JAX/CUDA versions. :func:`capture_provenance`
collects these; :func:`assert_release_clean` refuses to start a production run from a
dirty, untagged, or unpushed worktree unless an explicit discovery-only override is
given, in which case the exact diff and its SHA256 are archived with the run.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseCleanlinessError(RuntimeError):
    """Raised when a production run is attempted from an unclean/unpinned worktree."""


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _lockfile_sha256() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("uv.lock", "pyproject.toml"):
        path = _REPO_ROOT / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _cuda_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    try:
        import jax

        versions["jax"] = getattr(jax, "__version__", None)
        try:
            import jaxlib

            versions["jaxlib"] = getattr(jaxlib, "__version__", None)
        except Exception:  # pragma: no cover - jaxlib always present with jax
            pass
        try:
            versions["jax_backend"] = jax.default_backend()
        except Exception:  # pragma: no cover
            versions["jax_backend"] = None
    except Exception:  # pragma: no cover - jax always present in prod
        pass
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        versions["nvidia_driver"] = smi.splitlines()[0].strip()
    return versions


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _dirty_files() -> list[str]:
    status = _git("status", "--porcelain")
    if not status:
        return []
    return [line for line in status.splitlines() if line.strip()]


def capture_provenance() -> dict[str, Any]:
    """Return a JSON-ready provenance block for a manifest (FJ-13)."""

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    describe = _git("describe", "--tags", "--always", "--dirty")
    remote = _git("config", "--get", "remote.origin.url")
    dirty_files = _dirty_files()
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    unpushed = _git("rev-list", "@{u}..HEAD") if upstream else None
    return {
        "commit": commit,
        "branch": branch,
        "describe": describe,
        "remote_url": remote,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "upstream": upstream,
        "unpushed_commits": (
            [c for c in unpushed.splitlines() if c] if unpushed else []
        ),
        "lockfile_sha256": _lockfile_sha256(),
        "versions": _cuda_versions(),
    }


def assert_release_clean(
    out_dir: Path, *, allow_dirty: bool = False
) -> dict[str, Any]:
    """Enforce the FJ-13 gate: no run from a dirty/untagged/unpushed commit.

    With ``allow_dirty`` the run is permitted as *discovery-only*: the exact diff and
    its SHA256 are archived into ``out_dir`` and the returned provenance is flagged.
    """

    prov = capture_provenance()
    problems: list[str] = []
    if prov["commit"] is None:
        problems.append("not a git worktree (no commit)")
    if prov["dirty"]:
        problems.append(f"dirty worktree ({len(prov['dirty_files'])} changed files)")
    if prov["unpushed_commits"]:
        problems.append(f"{len(prov['unpushed_commits'])} unpushed commit(s)")
    if prov.get("upstream") is None:
        problems.append("no upstream/remote-tracking branch (commit not pushed)")

    if problems and not allow_dirty:
        raise ReleaseCleanlinessError(
            "production run refused: "
            + "; ".join(problems)
            + ". Run from a clean, pushed, tagged commit, or pass --allow-dirty "
            "for a discovery-only run (its diff will be archived)."
        )

    prov["release_gate"] = {
        "passed": not problems,
        "problems": problems,
        "discovery_only": bool(problems and allow_dirty),
    }
    if problems and allow_dirty:
        diff = _git("diff", "HEAD") or ""
        out_dir.mkdir(parents=True, exist_ok=True)
        diff_path = out_dir / "worktree_diff.patch"
        diff_path.write_text(diff, encoding="utf-8")
        prov["release_gate"]["diff_archive"] = str(diff_path)
        prov["release_gate"]["diff_sha256"] = hashlib.sha256(
            diff.encode("utf-8")
        ).hexdigest()
    return prov
