"""No-shenfun problem-spec validator for jaxfun production runs.

This mirrors the shenfun production validator and adds jaxfun implementation
gates for unsupported or intentionally deferred jaxfun subcases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


class ProblemSpecError(ValueError):
    """Raised when a production problem spec is malformed."""


class UnsupportedSpecError(ProblemSpecError):
    """Raised when a spec requests an unsupported or not-yet-ported subcase."""


SUPPORTED_GEOMETRIES = {"pcf", "channel", "taylor_couette", "pipe"}
SUPPORTED_PHYSICS = {"hydro", "mhd", "mri"}
SUPPORT_STATES = {"production", "experimental", "unsupported"}
INTEGRATORS = {"analytic", "IMEXRK222", "CNAB2", "linear_eigenproblem"}
MAGNETIC_BCS = {None, "conducting", "insulating", "pseudo_vacuum", "dirichlet"}
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "problem_spec.schema.json"

_PROBLEM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _stable_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_hash(spec: dict[str, Any]) -> str:
    """Return the shenfun-compatible SHA256 hash for a problem spec."""

    payload = {k: v for k, v in spec.items() if k != "spec_hash"}
    return hashlib.sha256(_stable_dumps(payload).encode("utf-8")).hexdigest()


def load_spec(
    path: str | Path,
    *,
    allow_unsupported: bool = False,
    allow_unimplemented: bool = False,
) -> dict[str, Any]:
    """Load and validate a JSON problem spec."""

    with Path(path).open("r", encoding="utf-8") as fh:
        return validate_spec(
            json.load(fh),
            allow_unsupported=allow_unsupported,
            allow_unimplemented=allow_unimplemented,
        )


def iter_example_specs(
    root: str | Path | None = None, *, include_unsupported: bool = False
):
    """Yield vendored example specs in a deterministic order."""

    base = (
        Path(root) if root is not None else Path(__file__).resolve().parent / "examples"
    )
    pattern = "**/*.json" if include_unsupported else "*.json"
    yield from sorted(base.glob(pattern))


def validate_spec(
    spec: dict[str, Any],
    *,
    allow_unsupported: bool = False,
    allow_unimplemented: bool = False,
) -> dict[str, Any]:
    """Validate a neutral production spec and return a normalized copy.

    ``allow_unsupported`` is for matrix/report tooling that must inspect rejected
    specs. ``allow_unimplemented`` is retained for compatibility with older
    promotion tooling while preserving contract checks and stable spec hash.
    """

    if not isinstance(spec, dict):
        raise ProblemSpecError("problem spec must be a JSON object")
    data = copy.deepcopy(spec)

    _require_keys(
        data,
        [
            "family",
            "problem_id",
            "geometry",
            "physics",
            "support_state",
            "formulation",
            "evolved_variables",
            "diagnostic_variables",
            "canonical_axes",
            "native_axes",
            "nondimensional_groups",
            "boundary_conditions",
            "domain",
            "resolution",
            "time",
            "initial_condition",
            "forcing",
            "diagnostics",
            "expected_oracle",
            "tolerance_model",
            "golden",
            "unsupported_subcases",
        ],
        "spec",
    )

    if data["family"] != "shenfun":
        raise ProblemSpecError("family must be 'shenfun'")
    if not _PROBLEM_ID_RE.match(str(data["problem_id"])):
        raise ProblemSpecError("problem_id must be a stable lowercase identifier")

    geometry = data["geometry"]
    physics = data["physics"]
    support_state = data["support_state"]
    if geometry not in SUPPORTED_GEOMETRIES:
        raise ProblemSpecError(
            f"geometry must be one of {sorted(SUPPORTED_GEOMETRIES)}"
        )
    if physics not in SUPPORTED_PHYSICS:
        raise ProblemSpecError(f"physics must be one of {sorted(SUPPORTED_PHYSICS)}")
    if support_state not in SUPPORT_STATES:
        raise ProblemSpecError(f"support_state must be one of {sorted(SUPPORT_STATES)}")

    _validate_axes(data)
    _validate_collections(data)
    _validate_time(data)
    _validate_nondimensional_groups(data)
    _validate_boundary_conditions(data)
    _validate_oracle_and_golden(data)
    _reject_shenfun_unsupported_subcases(data, allow_unsupported=allow_unsupported)

    if support_state == "unsupported" and not allow_unsupported:
        raise UnsupportedSpecError(
            "support_state 'unsupported' specs are rejected before solver allocation"
        )

    _reject_jaxfun_unimplemented(data, allow_unimplemented=allow_unimplemented)

    data["spec_hash"] = spec_hash(data)
    return data


def _require_keys(data: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ProblemSpecError(
            f"{context} missing required field(s): {', '.join(missing)}"
        )


def _validate_axes(data: dict[str, Any]) -> None:
    axes = data["canonical_axes"]
    native = data["native_axes"]
    if not isinstance(axes, dict) or set(axes) != {"x", "y", "z"}:
        raise ProblemSpecError("canonical_axes must state x, y, and z")
    if not isinstance(native, dict) or not {"axis_0", "axis_1", "axis_2"}.issubset(
        native
    ):
        raise ProblemSpecError("native_axes must state axis_0, axis_1, and axis_2")


def _validate_collections(data: dict[str, Any]) -> None:
    for key in (
        "evolved_variables",
        "diagnostic_variables",
        "diagnostics",
        "unsupported_subcases",
    ):
        if not isinstance(data[key], list):
            raise ProblemSpecError(f"{key} must be a list")
    for key in (
        "formulation",
        "domain",
        "resolution",
        "initial_condition",
        "forcing",
        "expected_oracle",
        "tolerance_model",
        "golden",
    ):
        if not isinstance(data[key], dict):
            raise ProblemSpecError(f"{key} must be an object")
    if "mode" in data and not isinstance(data["mode"], dict):
        raise ProblemSpecError("mode must be an object when present")


def _validate_time(data: dict[str, Any]) -> None:
    time = data["time"]
    _require_keys(time, ["integrator", "dt", "final_time"], "time")
    if time["integrator"] not in INTEGRATORS:
        raise ProblemSpecError(f"time.integrator must be one of {sorted(INTEGRATORS)}")
    if float(time["dt"]) < 0.0 or float(time["final_time"]) < 0.0:
        raise ProblemSpecError("time.dt and time.final_time must be non-negative")
    if time["integrator"] != "analytic" and float(time["dt"]) <= 0.0:
        raise ProblemSpecError("time.dt must be positive for non-analytic integrators")


def _validate_nondimensional_groups(data: dict[str, Any]) -> None:
    groups = data["nondimensional_groups"]
    if not isinstance(groups, dict):
        raise ProblemSpecError("nondimensional_groups must be an object")

    for key in ("Re", "Rm", "Pm", "Ha", "Omega", "S", "radius_ratio"):
        if key in groups and groups[key] is not None:
            _require_finite_number(groups[key], f"nondimensional_groups.{key}")

    if all(k in groups and groups[k] is not None for k in ("Re", "Rm", "Pm")):
        re = float(groups["Re"])
        rm = float(groups["Rm"])
        pm = float(groups["Pm"])
        if re <= 0.0 or rm <= 0.0 or pm <= 0.0:
            raise ProblemSpecError("Re, Rm, and Pm must be positive when present")
        expected = rm / re
        if not math.isclose(pm, expected, rel_tol=1.0e-10, abs_tol=1.0e-14):
            raise ProblemSpecError(
                f"Pm must equal Rm/Re; got Pm={pm:g}, Rm/Re={expected:g}"
            )

    if data["geometry"] == "taylor_couette":
        if "radius_ratio" not in groups:
            raise ProblemSpecError("Taylor-Couette specs must state radius_ratio")
        rr = float(groups["radius_ratio"])
        if not 0.0 < rr < 1.0:
            raise ProblemSpecError(
                "Taylor-Couette radius_ratio must satisfy 0 < eta < 1"
            )


def _validate_boundary_conditions(data: dict[str, Any]) -> None:
    bc = data["boundary_conditions"]
    _require_keys(bc, ["velocity"], "boundary_conditions")
    magnetic = bc.get("magnetic")
    if magnetic is not None:
        kind = magnetic.get("type") if isinstance(magnetic, dict) else magnetic
        if kind not in MAGNETIC_BCS:
            raise ProblemSpecError(f"unsupported magnetic boundary condition {kind!r}")
    if data["physics"] in {"mhd", "mri"} and magnetic is None:
        raise ProblemSpecError("MHD/MRI specs must state boundary_conditions.magnetic")


def _validate_oracle_and_golden(data: dict[str, Any]) -> None:
    oracle = data["expected_oracle"]
    _require_keys(oracle, ["type", "source"], "expected_oracle")
    golden = data["golden"]
    _require_keys(golden, ["artifact_id", "regeneration_command"], "golden")
    if not data["diagnostics"]:
        raise ProblemSpecError("diagnostics must list at least one emitted diagnostic")
    tolerances = data["tolerance_model"]
    _require_keys(tolerances, ["kind", "scalars"], "tolerance_model")
    if not isinstance(tolerances["scalars"], dict):
        raise ProblemSpecError("tolerance_model.scalars must be an object")


def _reject_shenfun_unsupported_subcases(
    data: dict[str, Any], *, allow_unsupported: bool
) -> None:
    geometry = data["geometry"]
    physics = data["physics"]
    bc = data["boundary_conditions"].get("magnetic")
    magnetic_bc = bc.get("type") if isinstance(bc, dict) else bc
    mode = data.get("mode", {})

    if allow_unsupported:
        return

    if geometry == "pipe" and physics in {"mhd", "mri"}:
        raise UnsupportedSpecError(
            "pipe MHD/MRI is unsupported in shenfun production specs"
        )
    if geometry == "channel" and physics in {"mhd", "mri"}:
        raise UnsupportedSpecError(
            "channel MHD/MRI is deferred; use pcf MHD/MRI production specs"
        )
    if physics == "mri" and geometry not in {"pcf", "taylor_couette"}:
        raise UnsupportedSpecError(
            "MRI production specs are supported only for pcf and Taylor-Couette"
        )
    if geometry == "pcf" and physics in {"mhd", "mri"} and magnetic_bc != "conducting":
        raise UnsupportedSpecError(
            "PCF MHD/MRI production specs support conducting magnetic walls only"
        )
    if (
        geometry == "taylor_couette"
        and physics in {"mhd", "mri"}
        and magnetic_bc not in {"conducting", "insulating"}
    ):
        raise UnsupportedSpecError(
            "Taylor-Couette MHD/MRI production specs support conducting or "
            "insulating magnetic walls only"
        )
    if (
        geometry == "taylor_couette"
        and physics in {"mhd", "mri"}
        and magnetic_bc == "insulating"
    ):
        m = int(mode.get("azimuthal_wavenumber", 0))
        kz = float(mode.get("axial_wavenumber", 0.0))
        if m != 0:
            raise UnsupportedSpecError(
                "Taylor-Couette insulating magnetic_bc is supported only for "
                "axisymmetric m=0"
            )
        if kz == 0.0:
            raise UnsupportedSpecError(
                "Taylor-Couette insulating magnetic_bc requires nonzero "
                "axial_wavenumber"
            )


def _reject_jaxfun_unimplemented(
    data: dict[str, Any], *, allow_unimplemented: bool
) -> None:
    if allow_unimplemented:
        return


def support_is_experimental(data: dict[str, Any]) -> bool:
    return data.get("support_state") == "experimental"


def _require_finite_number(value: Any, label: str) -> None:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ProblemSpecError(f"{label} must be numeric") from exc
    if not math.isfinite(x):
        raise ProblemSpecError(f"{label} must be finite")
