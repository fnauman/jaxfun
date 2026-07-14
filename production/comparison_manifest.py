"""Deterministic cross-repository comparison manifests.

The manifest is intentionally data-only: shearpy can emit its existing
``shearpy.run_manifest.v2`` and consume the resulting JSON without importing
jaxfun.  Jaxfun problem specs are resolved through :mod:`production.physics`,
so comparison controls cannot drift from the values consumed by the solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .observables import energy_convention_for_spec
from .physics import ResolvedPhysics, resolve_physics

SCHEMA_VERSION = "spectraldns.cross_repository_comparison.v1"
BUILDER_VERSION = "cross_repository_mapping_builder.v1"
MAPPING_VERSION = "shearbox_pcf_tc_mapping.v1"
OBSERVABLES_VERSION = "normalized_mhd_observables.v1"

RELATION_SHEARBOX_PCF = "shearbox_to_pcf"
RELATION_PCF_TC = "local_pcf_to_taylor_couette"
RELATIONS = (RELATION_SHEARBOX_PCF, RELATION_PCF_TC)

_COMMIT_RE = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


def canonical_json(value: Any) -> str:
    """Return the canonical byte representation used for every identifier."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except ValueError as exc:
        raise ValueError("canonical JSON numbers must be finite") from exc


def canonical_sha256(value: Any) -> str:
    """Hash one JSON value with the manifest's canonical serialization."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(f"{label} is inconsistent: {left!r} != {right!r}")


def _source_provenance(
    *, repository: str, commit: str, input_data: Mapping[str, Any], input_id: str
) -> dict[str, Any]:
    repository = str(repository).strip()
    commit = str(commit).strip().lower()
    if not repository:
        raise ValueError("repository provenance must be non-empty")
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("commit provenance must be a full 40- or 64-hex object ID")
    return {
        "repository": repository,
        "commit": commit,
        "input_id": input_id,
        "input_sha256": canonical_sha256(input_data),
    }


def _volume(physics: ResolvedPhysics) -> float:
    if physics.geometry == "taylor_couette":
        assert physics.R1 is not None
        assert physics.R2 is not None
        assert physics.theta_period is not None
        return 0.5 * physics.theta_period * (physics.R2**2 - physics.R1**2) * physics.Lz
    return 2.0 * physics.h * physics.Ly * physics.Lz


def _energy_raw_to_physical_mean(convention: str, volume: float) -> float:
    factors = {
        "integral_abs2": 0.5 / volume,
        "half_integral_abs2": 1.0 / volume,
        "half_integral_abs2_annulus": 1.0 / volume,
        "physical_volume_mean_half_abs2": 1.0,
    }
    try:
        return factors[convention]
    except KeyError as exc:
        raise ValueError(f"unsupported energy convention {convention!r}") from exc


def _observable_adapter(
    *,
    energy_convention: str,
    volume: float,
    shear: float,
    h: float,
    source_keys: Mapping[str, str],
) -> dict[str, Any]:
    shear_scale = abs(shear)
    if shear_scale == 0.0:
        raise ValueError("comparison endpoints require nonzero shear")
    velocity_scale = shear_scale * h
    energy_to_mean = _energy_raw_to_physical_mean(energy_convention, volume)
    return {
        "energy_convention": energy_convention,
        "volume": volume,
        "raw_energy_to_normalized": energy_to_mean / velocity_scale**2,
        "raw_volume_mean_stress_to_normalized": 1.0 / velocity_scale**2,
        "raw_mean_field_to_normalized": 1.0 / velocity_scale,
        "raw_time_to_normalized": shear_scale,
        "raw_growth_rate_to_normalized": 1.0 / shear_scale,
        "raw_wavenumber_to_normalized": h,
        "source_keys": dict(source_keys),
    }


def _comparison_b0(spec: Mapping[str, Any], resolved_b0: float) -> tuple[float, str]:
    """Return the imposed-field magnitude, including legacy PCF components."""

    groups = spec.get("nondimensional_groups", {})
    legacy_keys = ("Bx", "By", "Bz")
    if not any(key in groups for key in legacy_keys):
        return resolved_b0, "resolved_physics"
    components = tuple(_finite(groups.get(key, 0.0), key) for key in legacy_keys)
    legacy_b0 = math.sqrt(sum(component**2 for component in components))
    if resolved_b0 != 0.0:
        _close(resolved_b0, legacy_b0, "legacy Bx/By/Bz magnitude vs B0")
        return resolved_b0, "resolved_physics+legacy_components"
    return legacy_b0, "legacy_components"


def _jaxfun_endpoint(
    spec: Mapping[str, Any], *, expected_geometry: str, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    spec_dict = dict(spec)
    physics = resolve_physics(
        spec_dict, precision=str(spec_dict.get("precision", "float64"))
    )
    if physics.geometry != expected_geometry:
        raise ValueError(
            f"expected {expected_geometry!r} endpoint, got {physics.geometry!r}"
        )
    if physics.eta is None or physics.Rm_h is None or physics.Pm is None:
        raise ValueError("comparison endpoints must resolve resistive MHD controls")
    energy_convention = energy_convention_for_spec(spec_dict)
    if energy_convention is None:
        raise ValueError(
            "jaxfun endpoint does not declare a supported energy convention"
        )
    omega = physics.Omega
    q = physics.S / omega if omega != 0.0 else None
    groups = spec_dict.get("nondimensional_groups", {})
    declared_q = groups.get("q_shear", groups.get("q"))
    if declared_q is not None:
        if q is None:
            raise ValueError("jaxfun q is declared while Omega is zero")
        _close(_finite(declared_q, "jaxfun q"), q, "jaxfun q=S/Omega")
    volume = _volume(physics)
    b0, b0_source = _comparison_b0(spec_dict, physics.B0)
    controls = {
        "h": physics.h,
        "S": physics.S,
        "Omega": omega,
        "q": q,
        "U0": physics.U0,
        "nu": physics.nu,
        "eta": physics.eta,
        "Re_h": physics.Re_h,
        "Rm_h": physics.Rm_h,
        "Pm": physics.Pm,
        "B0": b0,
        "B0_over_U0": b0 / physics.U0,
        "B0_source": b0_source,
        "Ly_over_h": physics.Ly / physics.h,
        "Lz_over_h": physics.Lz / physics.h,
        "curvature": physics.curvature or 0.0,
    }
    if physics.geometry == "taylor_couette":
        controls.update(
            R1=physics.R1,
            R2=physics.R2,
            Omega1=physics.Omega1,
            Omega2=physics.Omega2,
            r_mid=physics.r_mid,
            theta_period=physics.theta_period,
            Re_TC=physics.Re_TC,
            Rm_TC=physics.Rm_TC,
        )
    if physics.geometry == "taylor_couette":
        magnetic_source_keys = {
            "magnetic_energy_fluct": "magnetic_energy",
            "magnetic_energy_mean": "derive:0.5*volume*mean_bz^2",
            "magnetic_energy_total": ("derive:magnetic_energy+0.5*volume*mean_bz^2"),
            "growth_rate_mag_fluct": ("derive:0.5*d(log(magnetic_energy))/dt"),
        }
    else:
        magnetic_source_keys = {
            "magnetic_energy_fluct": "mag_energy_fluct",
            "magnetic_energy_mean": "mag_energy_mean",
            "magnetic_energy_total": "magnetic_energy_total",
            "growth_rate_mag_fluct": ("derive:0.5*d(log(mag_energy_fluct))/dt"),
        }
    return {
        "kind": expected_geometry,
        "problem_id": str(spec_dict.get("problem_id", "unknown")),
        "provenance": dict(provenance),
        "controls": controls,
        "observable_adapter": _observable_adapter(
            energy_convention=energy_convention,
            volume=volume,
            shear=physics.S,
            h=physics.h,
            source_keys={
                "time": "t",
                "kinetic_energy_fluct": "kinetic_energy",
                **magnetic_source_keys,
                "reynolds_stress": "reynolds_stress",
                "maxwell_stress": (
                    "maxwell_stress_rt"
                    if physics.geometry == "taylor_couette"
                    else "maxwell_stress_xy"
                ),
                "total_stress": "total_stress",
                "mean_magnetic_z": "mean_bz",
            },
        ),
    }


def _shearpy_endpoint(
    manifest: Mapping[str, Any], *, reference_h: float, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("schema_version") != "shearpy.run_manifest.v2":
        raise ValueError("shearbox endpoint must use shearpy.run_manifest.v2")
    if manifest.get("domain") != "shearing_periodic":
        raise ValueError("shearpy endpoint must be a shearing_periodic run")
    if manifest.get("evolution") != "mhd":
        raise ValueError("shearpy endpoint must declare evolution='mhd'")
    lengths_raw = manifest.get("box_lengths")
    if not isinstance(lengths_raw, Sequence) or isinstance(lengths_raw, (str, bytes)):
        raise ValueError("shearpy box_lengths must contain three numbers")
    lengths = tuple(_positive(v, "shearpy box length") for v in lengths_raw)
    if len(lengths) != 3:
        raise ValueError("shearpy box_lengths must contain three numbers")
    magnetic_raw = manifest.get("mean_magnetic_field")
    if not isinstance(magnetic_raw, Sequence) or isinstance(magnetic_raw, (str, bytes)):
        raise ValueError("shearpy mean_magnetic_field must contain three numbers")
    magnetic = tuple(_finite(v, "shearpy mean magnetic field") for v in magnetic_raw)
    if len(magnetic) != 3:
        raise ValueError("shearpy mean_magnetic_field must contain three numbers")

    h = _positive(reference_h, "PCF reference half-gap")
    shear = _finite(manifest.get("shear"), "shearpy shear")
    if shear == 0.0:
        raise ValueError("comparison endpoints require nonzero shear")
    omega = _finite(manifest.get("omega"), "shearpy omega")
    q = None if omega == 0.0 else shear / omega
    manifest_q = manifest.get("q")
    if manifest_q is not None and q is not None:
        _close(_finite(manifest_q, "shearpy q"), q, "shearpy q=S/Omega")
    if manifest_q is not None and q is None:
        raise ValueError("shearpy q is declared while omega is zero")
    nu = _positive(manifest.get("nu"), "shearpy nu")
    eta = _positive(manifest.get("eta"), "shearpy eta")
    re_value = _positive(manifest.get("re"), "shearpy re")
    rm_value = _positive(manifest.get("rm"), "shearpy rm")
    pm_value = _positive(manifest.get("pm"), "shearpy pm")
    _close(re_value, 1.0 / nu, "shearpy re=1/nu")
    _close(rm_value, 1.0 / eta, "shearpy rm=1/eta")
    _close(pm_value, nu / eta, "shearpy pm=nu/eta")
    scale = abs(shear) * h**2
    velocity_scale = abs(shear) * h
    volume = math.prod(lengths)
    b0 = math.sqrt(sum(component**2 for component in magnetic))
    return {
        "kind": "shearing_box",
        "problem_id": str(manifest.get("campaign_preset") or "shearpy-run"),
        "provenance": dict(provenance),
        "controls": {
            "h": h,
            "S": shear,
            "Omega": omega,
            "q": q,
            "U0": velocity_scale,
            "nu": nu,
            "eta": eta,
            "Re_shearpy": re_value,
            "Rm_shearpy": rm_value,
            "Re_h": scale / nu,
            "Rm_h": scale / eta,
            "Pm": nu / eta,
            "B0": b0,
            "B0_over_U0": b0 / velocity_scale,
            "Lx_over_h": lengths[0] / h,
            "Ly_over_h": lengths[1] / h,
            "Lz_over_h": lengths[2] / h,
            "curvature": 0.0,
        },
        "observable_adapter": _observable_adapter(
            energy_convention="physical_volume_mean_half_abs2",
            volume=volume,
            shear=shear,
            h=h,
            source_keys={
                "time": "t",
                "kinetic_energy_fluct": "energy.kin_fluct",
                "magnetic_energy_total": "energy.mag_total",
                "magnetic_energy_mean": "energy.mag_mean",
                "magnetic_energy_fluct": "energy.mag_fluct",
                "growth_rate_mag_fluct": "derive:0.5*growth.mag_fluct",
                "reynolds_stress": "stress.reynolds",
                "maxwell_stress": "stress.maxwell",
                "total_stress": "stress.total",
                "mean_magnetic_z": "mean_b.z",
            },
        ),
    }


def _observable_contract() -> dict[str, Any]:
    return {
        "version": OBSERVABLES_VERSION,
        "canonical_scales": {
            "length": "h",
            "time": "1/abs(S)",
            "velocity_and_alfven_speed": "U0=abs(S)*h",
        },
        "observables": {
            "time": "tau=abs(S)*t",
            "wavenumber": "k_star=k*h",
            "mean_magnetic_field": "mean(B)_star=mean(B)/U0",
            "kinetic_energy_fluct": "K_star=(0.5*volume_mean(|u|^2))/U0^2",
            "magnetic_energy_total": ("Mtot_star=(0.5*volume_mean(|B|^2))/U0^2"),
            "magnetic_energy_mean": ("Mmean_star=(0.5*|volume_mean(B)|^2)/U0^2"),
            "magnetic_energy_fluct": (
                "Mfluct_star=(0.5*volume_mean(|B-volume_mean(B)|^2))/U0^2"
            ),
            "growth_rate_mag_fluct": (
                "gamma_mag_fluct_star=(0.5*d(log(Mfluct))/dt)/abs(S)"
            ),
            "reynolds_stress": "R_star=volume_mean(u_x*u_y)/U0^2",
            "maxwell_stress": "Mxy_star=-volume_mean(B_x*B_y)/U0^2",
            "total_stress": "Txy_star=R_star+Mxy_star",
            "alpha_Sh": "alpha_Sh=Txy_star",
        },
        "total_field_rule": (
            "magnetic means and energies include the imposed background field; "
            "mean and fluctuating magnetic energies must sum to total magnetic energy"
        ),
    }


def _mapping_contract(relation: str) -> dict[str, Any]:
    common = {
        "version": MAPPING_VERSION,
        "control_keys": [
            "h",
            "S",
            "Omega",
            "q",
            "Re_h",
            "Rm_h",
            "Pm",
            "B0_over_U0",
            "Ly_over_h",
            "Lz_over_h",
            "curvature",
        ],
    }
    if relation == RELATION_SHEARBOX_PCF:
        return {
            **common,
            "relation": relation,
            "scope": (
                "local rotating-shear bulk mapping; periodic shearbox and PCF wall "
                "boundary conditions are not asserted to be equivalent"
            ),
            "axes": {"x": "x", "y": "y", "z": "z"},
            "components": {
                "u_x": "u_x",
                "u_y": "u_y",
                "u_z": "u_z",
                "B_x": "B_x",
                "B_y": "B_y",
                "B_z": "B_z",
            },
            "background_flow": "U_y(x)=-S*x",
            "signed_shear": "S=-d(U_y)/dx",
            "length_anchor": "shearbox reference h equals the paired PCF half-gap",
            "shearpy_native_controls": {
                "Re_shearpy": "1/nu",
                "Rm_shearpy": "1/eta",
                "Re_h": "abs(S)*h^2/nu",
                "Rm_h": "abs(S)*h^2/eta",
            },
        }
    if relation == RELATION_PCF_TC:
        return {
            **common,
            "relation": relation,
            "scope": (
                "mid-gap local map from PCF Cartesian controls to circular Couette flow"
            ),
            "axes": {"pcf_x": "r-r_mid", "pcf_y": "r_mid*theta", "pcf_z": "z"},
            "components": {
                "u_x": "u_r",
                "u_y": "u_theta",
                "u_z": "u_z",
                "B_x": "B_r",
                "B_y": "B_theta",
                "B_z": "B_z",
            },
            "tc_local_controls": {
                "h": "(R2-R1)/2",
                "r_mid": "(R1+R2)/2",
                "Omega_mid": "a+b/r_mid^2",
                "S_mid": "-r_mid*d(Omega)/dr=2*b/r_mid^2",
                "q_mid": "S_mid/Omega_mid",
                "curvature": "h/r_mid",
                "azimuthal_length": "r_mid*theta_period",
            },
        }
    raise ValueError(f"unknown comparison relation {relation!r}")


def _assemble(relation: str, endpoints: list[dict[str, Any]]) -> dict[str, Any]:
    mapping = _mapping_contract(relation)
    observables = _observable_contract()
    campaign_identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "mapping": mapping,
        "observables": observables,
    }
    comparison_id = f"comparison-sha256:{canonical_sha256(campaign_identity)}"
    pair_identity = {
        "comparison_id": comparison_id,
        "relation": relation,
        "endpoints": endpoints,
    }
    pair_id = f"pair-sha256:{canonical_sha256(pair_identity)}"
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "comparison_id": comparison_id,
        "pair_id": pair_id,
        "relation": relation,
        "mapping": mapping,
        "observables": observables,
        "endpoints": endpoints,
        "contract_hashes": {
            "mapping_sha256": canonical_sha256(mapping),
            "observables_sha256": canonical_sha256(observables),
        },
    }


def build_comparison_manifest(
    *,
    relation: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_repository: str,
    left_commit: str,
    right_repository: str,
    right_commit: str,
) -> dict[str, Any]:
    """Build one deterministic shearbox-PCF or local PCF-TC pair manifest."""

    left_id = str(left.get("problem_id") or left.get("campaign_preset") or "left")
    right_id = str(right.get("problem_id") or right.get("campaign_preset") or "right")
    left_provenance = _source_provenance(
        repository=left_repository,
        commit=left_commit,
        input_data=left,
        input_id=left_id,
    )
    right_provenance = _source_provenance(
        repository=right_repository,
        commit=right_commit,
        input_data=right,
        input_id=right_id,
    )
    if relation == RELATION_SHEARBOX_PCF:
        right_endpoint = _jaxfun_endpoint(
            right, expected_geometry="pcf", provenance=right_provenance
        )
        left_endpoint = _shearpy_endpoint(
            left,
            reference_h=right_endpoint["controls"]["h"],
            provenance=left_provenance,
        )
        endpoints = [left_endpoint, right_endpoint]
    elif relation == RELATION_PCF_TC:
        endpoints = [
            _jaxfun_endpoint(left, expected_geometry="pcf", provenance=left_provenance),
            _jaxfun_endpoint(
                right, expected_geometry="taylor_couette", provenance=right_provenance
            ),
        ]
    else:
        raise ValueError(f"relation must be one of {RELATIONS!r}")
    return _assemble(relation, endpoints)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write canonical, byte-stable JSON without timestamps or local paths."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(manifest) + "\n", "utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation", choices=RELATIONS, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-repository", required=True)
    parser.add_argument("--left-commit", required=True)
    parser.add_argument("--right-repository", required=True)
    parser.add_argument("--right-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_comparison_manifest(
        relation=args.relation,
        left=_read_json(args.left),
        right=_read_json(args.right),
        left_repository=args.left_repository,
        left_commit=args.left_commit,
        right_repository=args.right_repository,
        right_commit=args.right_commit,
    )
    write_manifest(args.out, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
