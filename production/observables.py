"""Canonical observable helpers and jaxfun diagnostic name mapping."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


DNS_DIVERGENCE_KEYS: dict[str, tuple[str, ...]] = {
    "pcf_hydro_primitive_dns_v1": ("divergence_u",),
    "pcf_mri_primitive_dns_v1": ("divergence_u", "divergence_b"),
    "taylor_couette_hydro_dns_v1": ("divergence_linf",),
    "taylor_couette_mhd_dns_v1": ("divergence_u", "divergence_b"),
}

CANONICAL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "kinetic_energy": ("kinetic_energy", "Epert", "Ekin", "energy"),
    "magnetic_energy": ("magnetic_energy", "Emag"),
    "total_energy": ("total_energy", "Etot"),
    "divergence_l2": ("divergence_l2", "divL2", "div_l2"),
    "divergence_u_l2": ("divergence_u_l2", "divu_l2", "divL2"),
    "divergence_b_l2": ("divergence_b_l2", "divb_l2", "divB_L2"),
    "divergence_u": ("divergence_u", "divu", "divu_l2", "divL2"),
    "divergence_b": ("divergence_b", "divb", "divb_l2", "divB_L2"),
    "divergence_linf": ("divergence_linf", "divLinf", "div_linf"),
    "reynolds_stress": ("reynolds_stress",),
    "maxwell_stress_xy": ("maxwell_stress_xy", "maxwell_stress"),
    "transport_alpha": ("transport_alpha", "alpha"),
    "flow_rate": ("flow_rate",),
    "flow_rate_exact": ("flow_rate_exact",),
    "wall_flux": ("wall_flux",),
    "torque": ("torque",),
    "wall_shear_lower": ("wall_shear_lower", "bottom_wall_shear"),
    "wall_shear_upper": ("wall_shear_upper", "top_wall_shear"),
    "growth_rate": ("growth_rate",),
    "growth_rate_linear": ("growth_rate_linear",),
    "growth_rate_from_energy": ("growth_rate_from_energy",),
    "q_shear": ("q_shear",),
    "local_mri_growth": ("local_mri_growth",),
    "local_mri_smax_over_omega": ("local_mri_smax_over_omega",),
    "eigenvalue_real": ("eigenvalue_real",),
    "eigenvalue_imag": ("eigenvalue_imag",),
    "magnetic_bc": ("magnetic_bc",),
    "rayleigh_stable": ("rayleigh_stable",),
    "pressure_gradient": ("pressure_gradient", "dpdy"),
}


def expected_divergence_keys(
    *,
    geometry: str,
    physics: str,
    artifact_id: str | None = None,
) -> tuple[str, ...]:
    """Return the exact divergence scalar names expected for a spec/golden."""

    if artifact_id in DNS_DIVERGENCE_KEYS:
        return DNS_DIVERGENCE_KEYS[artifact_id]
    if physics == "hydro":
        return ("divergence_l2",)
    if geometry == "pcf" and physics in {"mhd", "mri"}:
        return ("divergence_u_l2", "divergence_b_l2")
    if geometry == "taylor_couette" and physics in {"mhd", "mri"}:
        return ("divergence_b_l2",)
    if geometry == "pipe" and physics in {"mhd", "mri"}:
        return ("divergence_b_l2",)
    return ("divergence_l2",)


def canonicalize_scalars(
    internal: dict[str, Any],
    *,
    geometry: str,
    physics: str,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    """Map jaxfun/internal diagnostic names onto shenfun golden scalar names."""

    out: dict[str, Any] = {}
    expected = set(expected_divergence_keys(geometry=geometry, physics=physics, artifact_id=artifact_id))
    expected.update(
        key
        for key, candidates in CANONICAL_CANDIDATES.items()
        if not key.startswith("divergence_")
        and any(candidate in internal for candidate in candidates)
    )

    for key in sorted(expected):
        value = _first_present(internal, CANONICAL_CANDIDATES.get(key, (key,)))
        if value is not _MISSING:
            out[key] = _scalarize_value(value)

    for key, value in internal.items():
        if key in CANONICAL_CANDIDATES and key not in out:
            out[key] = _scalarize_value(value)
    return out


def as_components(field: Iterable[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(field, np.ndarray) and field.ndim >= 1:
        return [np.asarray(field[i]) for i in range(field.shape[0])]
    return [np.asarray(comp) for comp in field]


def integral(values: Any, weights: Any = None) -> float:
    arr = np.asarray(values)
    if weights is None:
        raise ValueError("quadrature weights are required for canonical integrals")
    return float(np.real(np.sum(arr * np.asarray(weights))))


def trapezoid_weights(coordinate: Any) -> np.ndarray:
    x = np.asarray(coordinate, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("trapezoid_weights requires a 1-D coordinate with at least two points")
    dx = np.diff(x)
    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]
    if x.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return weights


def kinetic_energy(velocity: Any, weights: Any = None) -> float:
    comps = as_components(velocity)
    density = sum(np.abs(c) ** 2 for c in comps)
    return 0.5 * integral(density, weights)


def magnetic_energy(magnetic_field: Any, weights: Any = None) -> float:
    comps = as_components(magnetic_field)
    density = sum(np.abs(c) ** 2 for c in comps)
    return 0.5 * integral(density, weights)


def total_energy(velocity: Any = None, magnetic_field: Any = None, weights: Any = None) -> float:
    total = 0.0
    if velocity is not None:
        total += kinetic_energy(velocity, weights)
    if magnetic_field is not None:
        total += magnetic_energy(magnetic_field, weights)
    return total


def divergence_l2(divergence: Any, weights: Any = None) -> float:
    div = np.asarray(divergence)
    return math.sqrt(integral(np.abs(div) ** 2, weights))


def divergence_linf(divergence: Any) -> float:
    return float(np.max(np.abs(np.asarray(divergence))))


def reynolds_stress(velocity: Any, pair: tuple[int, int] = (0, 1), weights: Any = None) -> float:
    comps = as_components(velocity)
    return integral(np.real(comps[pair[0]] * np.conjugate(comps[pair[1]])), weights)


def maxwell_stress(magnetic_field: Any, pair: tuple[int, int] = (0, 1), weights: Any = None) -> float:
    comps = as_components(magnetic_field)
    return -integral(np.real(comps[pair[0]] * np.conjugate(comps[pair[1]])), weights)


def transport_alpha(reynolds: float, maxwell: float = 0.0, pressure: float = 1.0) -> float:
    if pressure == 0:
        raise ValueError("pressure must be nonzero for alpha normalization")
    return float((reynolds + maxwell) / pressure)


def flow_rate(radial: Any, axial_velocity: Any, *, geometry: str = "pipe") -> float:
    r = np.asarray(radial)
    u = np.asarray(axial_velocity)
    trapz = getattr(np, "trapezoid", np.trapz)
    if geometry == "pipe":
        return float(2.0 * math.pi * trapz(u * r, r))
    if geometry in {"channel", "pcf"}:
        return float(trapz(u, r))
    raise ValueError("flow_rate supports pipe, channel, and pcf geometries")


def wall_flux(coordinate: Any, scalar: Any, *, side: str = "upper") -> float:
    x = np.asarray(coordinate)
    f = np.asarray(scalar)
    if x.size < 2:
        raise ValueError("at least two samples are required")
    if side == "upper":
        return float((f[-1] - f[-2]) / (x[-1] - x[-2]))
    if side == "lower":
        return float((f[1] - f[0]) / (x[1] - x[0]))
    raise ValueError("side must be 'upper' or 'lower'")


def torque(radius: float, shear_stress: float) -> float:
    return float(2.0 * math.pi * radius**2 * shear_stress)


def growth_rate_from_energy(times: Any, energies: Any) -> float:
    t = np.asarray(times, dtype=float)
    e = np.asarray(energies, dtype=float)
    if t.size < 2 or e.size != t.size:
        raise ValueError("times and energies must have matching length >= 2")
    if np.any(e <= 0.0):
        raise ValueError("energies must be positive")
    return float(0.5 * np.polyfit(t, np.log(e), 1)[0])


class _Missing:
    pass


_MISSING = _Missing()


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return _MISSING


def _scalarize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value
