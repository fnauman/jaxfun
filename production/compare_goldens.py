"""Golden resolution, validation, and scalar comparison for jaxfun production."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .problem_spec import validate_spec

REQUIRED_GOLDEN_FIELDS = [
    "schema_version",
    "artifact_id",
    "problem_id",
    "spec_hash",
    "generated_at_utc",
    "environment",
    "git",
    "source_anchors",
    "tolerance_model",
    "diagnostics",
    "comparison_fields",
]


@dataclass(frozen=True)
class GoldenResolution:
    problem_id: str
    policy: str
    root: Path
    golden_path: Path
    spec_path: Path


@dataclass(frozen=True)
class ScalarComparison:
    key: str
    expected: Any
    actual: Any
    tolerance: float | None
    passed: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "expected": self.expected,
            "actual": self.actual,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "message": self.message,
        }


@dataclass(frozen=True)
class ComparisonResult:
    problem_id: str
    golden_path: Path
    passed: bool
    comparisons: tuple[ScalarComparison, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "golden_path": str(self.golden_path),
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
            "metadata": self.metadata,
        }


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def scalar_hash(scalars: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(scalars).encode("utf-8")).hexdigest()


def vendored_golden_root() -> Path:
    return Path(__file__).resolve().parent / "goldens"


def fallback_shenfun_golden_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parents[1] / "fn_shenfun" / "demo" / "production" / "goldens"


def resolve_golden(
    problem_id: str, golden_root: str | Path | None = None
) -> GoldenResolution:
    if golden_root is not None:
        root = Path(golden_root)
        policy = "explicit"
    else:
        vendored = vendored_golden_root()
        if vendored.exists():
            root = vendored
            policy = "vendored"
        else:
            root = Path(
                os.environ.get("SHENFUN_GOLDENS_ROOT", fallback_shenfun_golden_root())
            )
            policy = (
                "env-var" if "SHENFUN_GOLDENS_ROOT" in os.environ else "sibling-default"
            )

    golden_path = root / problem_id / "golden" / "golden.json"
    spec_path = root / problem_id / "spec.json"
    if not golden_path.exists():
        raise FileNotFoundError(f"golden for {problem_id!r} not found at {golden_path}")
    return GoldenResolution(
        problem_id=problem_id,
        policy=policy,
        root=root,
        golden_path=golden_path,
        spec_path=spec_path,
    )


def load_golden(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


class QuarantinedGoldenError(RuntimeError):
    """Raised when a quarantined (pre-FJ-01/invalid) golden is used for production."""


def assert_golden_not_quarantined(golden: dict[str, Any], problem_id: str) -> None:
    """FJ-03: refuse to seed/validate production against a quarantined golden.

    Quarantined goldens are retained only as regression evidence; they must never
    be a live comparison reference. Inspection helpers (load_golden/validate_golden)
    still read them, but the production comparison path calls this guard.
    """

    quarantine = golden.get("quarantined")
    if quarantine and quarantine.get("forbidden_from_seeding_production", True):
        raise QuarantinedGoldenError(
            f"golden for {problem_id!r} is quarantined and cannot validate "
            f"production: {quarantine.get('reason', 'pre-FJ-01 artifact')}"
        )


def validate_golden(
    path: str | Path, spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = load_golden(path)
    missing = [key for key in REQUIRED_GOLDEN_FIELDS if key not in data]
    if missing:
        raise ValueError(f"golden missing required field(s): {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError(
            f"unsupported golden schema_version {data['schema_version']!r}"
        )
    if "scalars" not in data["diagnostics"]:
        raise ValueError("golden diagnostics must contain scalars")
    if not data["diagnostics"]["scalars"]:
        raise ValueError("golden diagnostics.scalars must not be empty")
    expected_hash = scalar_hash(data["diagnostics"]["scalars"])
    stored_hash = data["comparison_fields"].get("scalars_sha256")
    if stored_hash != expected_hash:
        raise ValueError("golden scalar hash does not match diagnostics.scalars")
    missing_tolerances = _numeric_scalars_missing_tolerance(data)
    if missing_tolerances:
        raise ValueError(
            "golden numeric scalar(s) missing tolerance: "
            + ", ".join(missing_tolerances)
        )
    invalid_tolerances = _invalid_numeric_tolerances(data)
    if invalid_tolerances:
        raise ValueError(
            "golden numeric scalar(s) have invalid tolerance: "
            + ", ".join(invalid_tolerances)
        )
    stored_tolerance_hash = data["comparison_fields"].get("tolerance_model_sha256")
    if stored_tolerance_hash is not None and stored_tolerance_hash != scalar_hash(
        data["tolerance_model"]
    ):
        raise ValueError("golden tolerance hash does not match tolerance_model")
    if spec is not None:
        validated = validate_spec(
            spec, allow_unsupported=False, allow_unimplemented=False
        )
        if data["spec_hash"] != validated["spec_hash"]:
            raise ValueError("golden spec_hash does not match spec")
    return data


def load_actual_scalars(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if "diagnostics" in data and isinstance(data["diagnostics"], dict):
        return data["diagnostics"].get("scalars", data["diagnostics"])
    if "scalars" in data and isinstance(data["scalars"], dict):
        return data["scalars"]
    return data


def compare_to_golden(
    actual_scalars: dict[str, Any],
    golden: dict[str, Any],
    *,
    golden_path: str | Path | None = None,
    require_all_golden_scalars: bool = True,
    convention_metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    validate_scalar_hash(golden)
    golden_scalars = golden["diagnostics"]["scalars"]
    tolerances = golden["tolerance_model"]["scalars"]
    keys = sorted(golden_scalars) if require_all_golden_scalars else sorted(tolerances)
    comparisons = tuple(
        _compare_one(
            key,
            expected=golden_scalars.get(key),
            actual=actual_scalars.get(key, _MISSING),
            tolerance=tolerances.get(key),
            key_in_golden=key in golden_scalars,
        )
        for key in keys
    )
    return ComparisonResult(
        problem_id=golden["problem_id"],
        golden_path=Path(golden_path) if golden_path is not None else Path("<memory>"),
        passed=all(item.passed for item in comparisons),
        comparisons=comparisons,
        metadata={"conventions": convention_metadata or {}},
    )


def compare_problem(
    problem_id: str,
    actual_scalars: dict[str, Any],
    *,
    golden_root: str | Path | None = None,
    require_all_golden_scalars: bool = True,
    convention_metadata: dict[str, Any] | None = None,
) -> ComparisonResult:
    resolution = resolve_golden(problem_id, golden_root=golden_root)
    spec = None
    if resolution.spec_path.exists():
        with resolution.spec_path.open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
    golden = validate_golden(resolution.golden_path, spec=spec)
    result = compare_to_golden(
        actual_scalars,
        golden,
        golden_path=resolution.golden_path,
        require_all_golden_scalars=require_all_golden_scalars,
        convention_metadata=convention_metadata,
    )
    result.metadata.update(
        {
            "golden_resolution_policy": resolution.policy,
            "golden_root": str(resolution.root),
        }
    )
    return result


def validate_scalar_hash(golden: dict[str, Any]) -> None:
    expected_hash = scalar_hash(golden["diagnostics"]["scalars"])
    stored_hash = golden["comparison_fields"].get("scalars_sha256")
    if stored_hash != expected_hash:
        raise ValueError("golden scalar hash does not match diagnostics.scalars")


def _compare_one(
    key: str,
    *,
    expected: Any,
    actual: Any,
    tolerance: Any,
    key_in_golden: bool,
) -> ScalarComparison:
    if not key_in_golden:
        return ScalarComparison(
            key,
            expected,
            actual,
            _as_optional_float(tolerance),
            False,
            "key not present in golden scalars",
        )
    if actual is _MISSING:
        return ScalarComparison(
            key,
            expected,
            None,
            _as_optional_float(tolerance),
            False,
            "actual scalar missing",
        )

    if isinstance(expected, bool) != isinstance(actual, bool):
        return ScalarComparison(
            key,
            expected,
            actual,
            _as_optional_float(tolerance),
            False,
            "scalar type mismatch",
        )

    if _is_number(expected) or _is_number(actual):
        if not (_is_number(expected) and _is_number(actual)):
            return ScalarComparison(
                key,
                expected,
                actual,
                _as_optional_float(tolerance),
                False,
                "scalar type mismatch",
            )
        if tolerance is None:
            return ScalarComparison(
                key,
                float(expected),
                float(actual),
                None,
                False,
                "numeric scalar missing tolerance",
            )
        tol = float(tolerance)
        if not math.isfinite(tol) or tol < 0.0:
            return ScalarComparison(
                key,
                float(expected),
                float(actual),
                tol,
                False,
                "numeric scalar has invalid tolerance",
            )
        diff = abs(float(actual) - float(expected))
        passed = math.isfinite(diff) and diff <= tol
        msg = "" if passed else f"abs diff {diff:g} exceeds tolerance {tol:g}"
        return ScalarComparison(key, float(expected), float(actual), tol, passed, msg)

    passed = actual == expected
    msg = "" if passed else "non-numeric scalar mismatch"
    return ScalarComparison(
        key, expected, actual, _as_optional_float(tolerance), passed, msg
    )


def _numeric_scalars_missing_tolerance(golden: dict[str, Any]) -> list[str]:
    scalars = golden["diagnostics"]["scalars"]
    tolerances = golden.get("tolerance_model", {}).get("scalars", {})
    return sorted(
        key
        for key, value in scalars.items()
        if _is_number(value) and key not in tolerances
    )


def _invalid_numeric_tolerances(golden: dict[str, Any]) -> list[str]:
    scalars = golden["diagnostics"]["scalars"]
    tolerances = golden.get("tolerance_model", {}).get("scalars", {})
    invalid = []
    for key, value in scalars.items():
        if not _is_number(value) or key not in tolerances:
            continue
        tolerance = tolerances[key]
        if isinstance(tolerance, bool):
            invalid.append(key)
            continue
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            invalid.append(key)
            continue
        if not math.isfinite(tol) or tol < 0.0:
            invalid.append(key)
    return sorted(invalid)


def _as_optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class _Missing:
    pass


_MISSING = _Missing()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument(
        "--actual",
        required=True,
        help=(
            "JSON file containing scalars, diagnostics.scalars, or a golden-like object"
        ),
    )
    parser.add_argument("--golden-root")
    parser.add_argument("--require-all-golden-scalars", action="store_true")
    args = parser.parse_args(argv)

    result = compare_problem(
        args.problem_id,
        load_actual_scalars(args.actual),
        golden_root=args.golden_root,
        require_all_golden_scalars=args.require_all_golden_scalars,
    )
    print(stable_json(result.to_dict()), end="")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
