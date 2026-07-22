"""Run provenance and release-cleanliness gate (FJ-13).

Every production manifest must record enough to reproduce the run from an
immutable commit: the git commit, branch (informational), release tag, remote
URL, dirty-state flag, dependency-lock hash, and JAX/CUDA versions.
:func:`capture_provenance` collects these; :func:`assert_release_clean` refuses
to start a production run from a dirty, untagged, or unpushed worktree unless
an explicit discovery-only override is given, in which case the exact diff and
its SHA256 are archived with the run.
"""

from __future__ import annotations

import hashlib
import os
import platform
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


def _head_release_ref(remote: str = "origin") -> dict[str, Any]:
    """Verify that HEAD is pinned by the same exact tag on ``remote``.

    A branch -- including ``main`` -- is a movable ref and therefore cannot be a
    production release identity. Annotated tags are compared through their peeled
    commit; lightweight tags compare directly.
    """

    commit = _git("rev-parse", "HEAD")
    exact_tag = _git("describe", "--exact-match", "--tags", "HEAD")
    tag_commit = _git("rev-list", "-n", "1", exact_tag) if exact_tag else None
    remote_tag_commit = None
    if exact_tag:
        listing = _git(
            "ls-remote",
            "--tags",
            remote,
            f"refs/tags/{exact_tag}",
            f"refs/tags/{exact_tag}^{{}}",
        )
        direct = None
        peeled = None
        for line in (listing or "").splitlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            if fields[1].endswith("^{}"):
                peeled = fields[0]
            else:
                direct = fields[0]
        remote_tag_commit = peeled or direct
    verified = bool(
        commit and exact_tag and tag_commit == commit and remote_tag_commit == commit
    )
    return {
        "exact_tag": exact_tag,
        "tag_commit": tag_commit,
        "remote": remote,
        "remote_tag_commit": remote_tag_commit,
        "remote_verified": verified,
        "is_immutable_ref": verified,
    }


def _untracked_files() -> list[str]:
    listing = _git("ls-files", "--others", "--exclude-standard")
    return [line for line in (listing or "").splitlines() if line.strip()]


def _lockfile_sha256() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("uv.lock", "pyproject.toml"):
        path = _REPO_ROOT / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _cuda_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": platform.python_version()}
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
        try:
            device = jax.devices()[0]
            versions["gpu"] = getattr(device, "device_kind", None)
            versions["cuda_runtime"] = getattr(device.client, "platform_version", None)
        except Exception:  # pragma: no cover
            pass
    except Exception:  # pragma: no cover - jax always present in prod
        pass
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        versions["nvidia_driver"] = smi.splitlines()[0].strip()

    return versions


def _jax_runtime_config() -> dict[str, Any]:
    """Capture compiler settings that affect executable identity and caching."""
    try:
        import jax

        simplified = bool(
            getattr(jax.config, "jax_use_simplified_jaxpr_constants", False)
        )
        return {
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "jax_use_simplified_jaxpr_constants": simplified,
            "jax_use_simplified_jaxpr_constants_env": os.environ.get(
                "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"
            ),
            "jaxfun_use_simplified_jaxpr_constants": os.environ.get(
                "JAXFUN_USE_SIMPLIFIED_JAXPR_CONSTANTS"
            ),
            "jaxfun_wavenumber_solver": os.environ.get(
                "JAXFUN_WAVENUMBER_SOLVER", "jax"
            ),
        }
    except Exception:  # pragma: no cover - provenance must remain best effort
        return {}


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
        "jax_config": _jax_runtime_config(),
    }


def assert_release_clean(
    out_dir: Path,
    *,
    allow_dirty: bool = False,
    remote: str = "origin",
) -> dict[str, Any]:
    """Enforce the FJ-13 gate: no run from a dirty/untagged/unpushed commit.

    With ``allow_dirty`` the run is permitted as *discovery-only*: the exact diff and
    its SHA256 are archived into ``out_dir`` and the returned provenance is flagged.
    """

    prov = capture_provenance()
    release_ref = _head_release_ref(remote)
    prov["release_ref"] = release_ref
    problems: list[str] = []
    if prov["commit"] is None:
        problems.append("not a git worktree (no commit)")
    if prov["dirty"]:
        problems.append(f"dirty worktree ({len(prov['dirty_files'])} changed files)")
    if prov["unpushed_commits"]:
        problems.append(f"{len(prov['unpushed_commits'])} unpushed commit(s)")
    if prov.get("upstream") is None:
        problems.append("no upstream/remote-tracking branch (commit not pushed)")
    if not release_ref["is_immutable_ref"]:
        problems.append(
            "HEAD is not an immutable release ref: an exact local tag resolving to "
            f"HEAD must exist at the same commit on {remote} "
            "(a branch, including main, is a movable ref)"
        )

    if problems and not allow_dirty:
        raise ReleaseCleanlinessError(
            "production run refused: "
            + "; ".join(problems)
            + ". Run from a clean commit pinned by an exact pushed tag, or "
            "pass --allow-dirty for a discovery-only run (its diff will be archived)."
        )

    prov["release_gate"] = {
        "passed": not problems,
        "problems": problems,
        "discovery_only": bool(problems and allow_dirty),
        "release_ref": release_ref,
    }
    if problems and allow_dirty:
        prov["release_gate"].update(_archive_worktree(out_dir))
    return prov


def _archive_worktree(out_dir: Path) -> dict[str, Any]:
    """Archive the exact tracked diff AND untracked files for a discovery-only run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    diff = _git("diff", "HEAD") or ""
    diff_path = out_dir / "worktree_diff.patch"
    diff_path.write_text(diff, encoding="utf-8")

    # git diff HEAD omits untracked files even when they made the run dirty; capture
    # them explicitly so the archive can reproduce the executed checkout.
    hasher = hashlib.sha256()
    hasher.update(diff.encode("utf-8"))
    untracked = _untracked_files()
    archived_untracked: list[str] = []
    if untracked:
        archive_dir = out_dir / "worktree_untracked"
        for rel in sorted(untracked):
            src = _REPO_ROOT / rel
            if not src.is_file():
                continue
            data = src.read_bytes()
            dst = archive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            hasher.update(f"\0untracked:{rel}\0".encode())
            hasher.update(data)
            archived_untracked.append(rel)
    return {
        "diff_archive": str(diff_path),
        "untracked_archived": archived_untracked,
        "diff_sha256": hasher.hexdigest(),
    }
