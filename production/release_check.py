"""Release provenance + convention artifact (FJ-13 / FJ-03.8).

Assembles the artifact that must accompany a production release: full run provenance
plus a *convention-explicit* manifest recording the coordinate order per solver family,
the signed-shear convention, the half/full-gap convention, the reference interpreter,
and the vendored Shenfun commit. The live-Shenfun parity tier must be run and shown to
have executed (not skipped) on the release commit -- this module records the test
summary and flags a skipped-but-green tier as NOT a pass.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .axes import SOLVER_NATIVE_AXES

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The convention traps the release artifact must state explicitly (FJ-03.8/FJ-10.4).
CONVENTIONS: dict[str, Any] = {
    "coordinate_order": {
        "pcf_primitive": list(SOLVER_NATIVE_AXES["pcf_primitive"]),  # (y, z, x)
        "pcf_kmm": list(SOLVER_NATIVE_AXES["pcf_kmm"]),  # (x, y, z)
        "pcf_vector_potential": list(SOLVER_NATIVE_AXES["pcf_vector_potential"]),
        "taylor_couette": ["r", "theta", "z"],
    },
    "signed_shear": "shearing box S=+1 with U'(x) = -S (shearpy convention)",
    "gap": "wall-normal domain [-1, 1]; half-gap h = 1 (full box width = 2)",
    "atlas_note": "the h=1 operator is built directly; do NOT use an a=0.5 atlas helper",
    "imposed_field": "B0 = bz (vertical Alfven speed, rho = mu0 = 1)",
    "ideal_mri_cutoff": "(k_z v_A)^2 < 2 q Omega^2 (Balbus-Hawley vertical field)",
}


def _shenfun_commit() -> str | None:
    provenance = _REPO_ROOT / "production" / "goldens" / "PROVENANCE.json"
    if not provenance.exists():
        return None
    try:
        data = json.loads(provenance.read_text())
    except json.JSONDecodeError:
        return None
    # best-effort: look for a commit-like field anywhere in the record
    for key in ("shenfun_commit", "commit", "source_commit"):
        if key in data:
            return str(data[key])
    return None


def release_manifest(test_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the full release artifact (provenance + conventions + test summary)."""

    from .provenance import capture_provenance

    manifest = {
        "provenance": capture_provenance(),
        "conventions": CONVENTIONS,
        "reference_interpreter": sys.executable,
        "shenfun_commit": _shenfun_commit(),
        "shenfun_env": {
            "SHENFUN_PYTHON": os.environ.get("SHENFUN_PYTHON"),
            "SHENFUN_SOURCE_ROOT": os.environ.get("SHENFUN_SOURCE_ROOT"),
        },
    }
    if test_summary is not None:
        manifest["tests"] = _annotate_test_summary(test_summary)
    return manifest


def _annotate_test_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flag the FJ-13 gate: a skipped live-parity tier does NOT count as a pass."""

    out = dict(summary)
    live = summary.get("live_shenfun", {})
    ran = int(live.get("passed", 0)) > 0 and int(live.get("skipped", 0)) == 0
    out["live_shenfun_ran_not_skipped"] = ran
    out["release_test_gate_passed"] = bool(
        ran
        and int(summary.get("failed", 0)) == 0
        and int(summary.get("errors", 0)) == 0
    )
    return out


def write_release_artifact(
    out_path: str | Path, *, test_summary: dict[str, Any] | None = None
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(release_manifest(test_summary), indent=2) + "\n", "utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write the release artifact (FJ-13).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    written = write_release_artifact(args.out)
    print(f"wrote release artifact -> {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
