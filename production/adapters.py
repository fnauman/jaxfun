"""Spec-to-jaxfun convention adapters.

The functions here stop at validated metadata. Solver allocation is intentionally
left to the production runner so unsupported specs fail before JAX compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .problem_spec import load_spec, validate_spec


SOLVER_SOURCE_FILES: dict[tuple[str, str], list[str]] = {
    ("pcf", "hydro"): [
        "examples/pcf_fluctuations_jax.py",
        "examples/pcf_linear_jax.py",
    ],
    ("channel", "hydro"): ["examples/channelflow_kmm.py"],
    ("pcf", "mhd"): [
        "examples/pcf_mhd_jax.py",
        "examples/pcf_mri_primitive_jax.py",
    ],
    ("pcf", "mri"): ["examples/pcf_mri_primitive_jax.py"],
    ("taylor_couette", "hydro"): [
        "examples/taylor_couette_dns_jax.py",
        "examples/taylor_couette_linear_jax.py",
    ],
    ("taylor_couette", "mhd"): [
        "examples/taylor_couette_dns_jax.py",
        "examples/taylor_couette_mri_jax.py",
    ],
    ("taylor_couette", "mri"): [
        "examples/taylor_couette_dns_jax.py",
        "examples/taylor_couette_mri_jax.py",
    ],
}

GEOMETRY_AXIS_CONVENTIONS: dict[str, dict[str, str]] = {
    "pcf": {"axis_0": "x wall-normal", "axis_1": "y streamwise", "axis_2": "z spanwise"},
    "channel": {"axis_0": "x wall-normal", "axis_1": "y streamwise", "axis_2": "z spanwise"},
    "taylor_couette": {"axis_0": "r radial", "axis_1": "theta azimuthal", "axis_2": "z axial"},
    "pipe": {"axis_0": "r radial", "axis_1": "theta azimuthal", "axis_2": "z axial"},
}


@dataclass(frozen=True)
class ProductionConfig:
    spec: dict[str, Any]
    problem_id: str
    geometry: str
    physics: str
    artifact_id: str
    source_files: tuple[str, ...]
    native_axes: dict[str, Any]
    canonical_axes: dict[str, Any]
    axis_conventions: dict[str, str]
    solver_args: dict[str, Any]
    metadata: dict[str, Any]


def load_config(path: str | Path) -> ProductionConfig:
    """Load a spec and return validated adapter metadata."""

    return config_from_spec(load_spec(path))


def config_from_spec(spec: dict[str, Any]) -> ProductionConfig:
    validated = validate_spec(spec)
    geometry = validated["geometry"]
    physics = validated["physics"]
    source_files = tuple(SOLVER_SOURCE_FILES.get((geometry, physics), ()))
    solver_args = solver_arguments_from_spec(validated)
    metadata = {
        "canonical_axes": validated["canonical_axes"],
        "native_axes": validated["native_axes"],
        "axis_conventions": GEOMETRY_AXIS_CONVENTIONS[geometry],
        "nondimensional_groups": validated["nondimensional_groups"],
        "boundary_conditions": validated["boundary_conditions"],
        "solver_source_files": list(source_files),
        "spec_hash": validated["spec_hash"],
    }
    return ProductionConfig(
        spec=validated,
        problem_id=validated["problem_id"],
        geometry=geometry,
        physics=physics,
        artifact_id=validated["golden"]["artifact_id"],
        source_files=source_files,
        native_axes=validated["native_axes"],
        canonical_axes=validated["canonical_axes"],
        axis_conventions=GEOMETRY_AXIS_CONVENTIONS[geometry],
        solver_args=solver_args,
        metadata=metadata,
    )


def solver_arguments_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Extract constructor-oriented arguments without instantiating a solver."""

    groups = spec["nondimensional_groups"]
    time = spec["time"]
    args: dict[str, Any] = {
        "geometry": spec["geometry"],
        "physics": spec["physics"],
        "resolution": spec["resolution"],
        "domain": spec["domain"],
        "dt": time["dt"],
        "final_time": time["final_time"],
        "integrator": time["integrator"],
        "boundary_conditions": spec["boundary_conditions"],
        "initial_condition": spec["initial_condition"],
        "forcing": spec["forcing"],
    }
    for key in ("Re", "Rm", "Pm", "Ha", "Omega", "S", "B0", "R1", "R2", "Omega1", "Omega2", "nu", "eta_mag"):
        if key in groups:
            args[key] = groups[key]
    if "mode" in spec:
        args["mode"] = spec["mode"]
    return args


def shenfun_rfft_coeff_layout(coeff: Any, *, radial_n: int, spanwise_n: int) -> np.ndarray:
    """Convert shenfun's conjugate-symmetric Fourier coefficients to rfft layout."""

    coeff_np = np.asarray(coeff)
    _assert_conjugate_symmetric(coeff_np, periodic_axes=(1, 2))
    out = np.zeros((radial_n, coeff_np.shape[1], spanwise_n // 2 + 1), dtype=complex)
    out[: coeff_np.shape[0], :, :] = coeff_np[:, :, : spanwise_n // 2 + 1]
    return out


def _assert_conjugate_symmetric(coeff: np.ndarray, periodic_axes: tuple[int, ...]) -> None:
    axis_shape = tuple(coeff.shape[axis] for axis in periodic_axes)
    for mode in np.ndindex(axis_shape):
        src = [slice(None)] * coeff.ndim
        dst = [slice(None)] * coeff.ndim
        for axis, index in zip(periodic_axes, mode, strict=True):
            src[axis] = index
            dst[axis] = (-index) % coeff.shape[axis]
        src_values = coeff[tuple(src)]
        dst_values = coeff[tuple(dst)]
        if max(np.max(np.abs(src_values)), np.max(np.abs(dst_values))) < 1.0e-8:
            continue
        if not np.allclose(dst_values, np.conj(src_values), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError("shenfun Fourier coefficients are not conjugate symmetric")
