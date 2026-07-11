"""Single resolved-physics contract (FJ-00).

Solver coefficients (``nu``, ``eta``, imposed ``B0``) and the reported
nondimensional groups (``Re``, ``Rm``, ``Pm``) had independent sources: a spec
could carry ``Re=1000`` alongside ``nu=0.001`` while only ``nu`` reached the
operator, so a sweep that varied only ``Re`` would silently relabel identical
physics. :class:`ResolvedPhysics` makes the dimensional coefficients canonical and
derives the nondimensional groups *once*, cross-checking any redundant inputs and
raising :class:`ProblemSpecError` **before** JAX compilation when they disagree.

Canonical PCF-MRI units use the half-gap ``h`` (domain ``x`` in ``[-h, h]``) and
the shear rate ``S`` as the velocity scale, ``U0 = |S| h``, so

    Re_h = |S| h^2 / nu,   Rm_h = |S| h^2 / eta,   Pm = nu / eta = Rm_h / Re_h.

Plain plane-Couette hydro/MHD uses the wall speed ``U_wall`` as the velocity
scale; for the comparison campaign ``U_wall = |S| h`` so the two conventions
coincide. The convention actually used is recorded in
:attr:`ResolvedPhysics.reynolds_convention`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .problem_spec import ProblemSpecError

# Relative tolerance when cross-checking a directly-supplied coefficient against
# the value derived from a supplied nondimensional group.
_CONSISTENCY_RTOL = 1.0e-9
_CONSISTENCY_ATOL = 1.0e-12


@dataclass(frozen=True)
class ResolvedPhysics:
    """The one physics object every solver and diagnostic consumes."""

    # --- canonical dimensional inputs ---
    h: float
    S: float
    Omega: float
    nu: float
    eta: float | None
    B0: float
    Ly: float
    Lz: float
    U0: float  # velocity scale = |S| h (canonical PCF)
    # --- resolved nondimensional groups ---
    Re_h: float
    Rm_h: float | None
    Pm: float | None
    # --- labels / provenance ---
    velocity_scale: str  # "shear" or "wall"
    reynolds_convention: str
    magnetic_bc: str | None
    precision: str
    raw_inputs: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-ready dict of both raw and resolved values."""

        data = asdict(self)
        return data


def _finite(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ProblemSpecError(f"{label} must be numeric") from exc
    if not math.isfinite(out):
        raise ProblemSpecError(f"{label} must be finite")
    return out


def _positive(value: float, label: str) -> float:
    if not value > 0.0:
        raise ProblemSpecError(f"{label} must be positive, got {value}")
    return value


def _half_gap(spec: dict[str, Any]) -> float:
    """Return the half-gap ``h`` from the wall-normal domain interval."""

    domain = spec["domain"]
    if "x" in domain:  # pcf / channel
        x0, x1 = (_finite(v, "domain.x") for v in domain["x"])
        return 0.5 * (x1 - x0)
    if "r" in domain:  # taylor_couette / pipe -- gap width
        r0, r1 = (_finite(v, "domain.r") for v in domain["r"])
        return 0.5 * (r1 - r0)
    raise ProblemSpecError("domain must provide a wall-normal interval (x or r)")


def _b0_magnitude(value: Any, label: str) -> float:
    """Return an imposed-field magnitude from a scalar or a component vector."""

    if isinstance(value, (list, tuple)):
        comps = [_finite(v, label) for v in value]
        return math.sqrt(sum(c * c for c in comps))
    return abs(_finite(value, label))


def _single_b0(spec: dict[str, Any]) -> float:
    """Return one imposed-field value, rejecting divergent group/forcing sources.

    ``forcing.B0`` may be a scalar or a component vector (e.g. ``[0, 0, bz]``); the
    magnitude is used and cross-checked against a scalar ``nondimensional_groups.B0``.
    """

    groups = spec["nondimensional_groups"]
    forcing = spec.get("forcing", {})
    group_b0 = groups.get("B0")
    forcing_b0 = forcing.get("B0") if isinstance(forcing, dict) else None
    if group_b0 is not None and forcing_b0 is not None:
        gb = _b0_magnitude(group_b0, "nondimensional_groups.B0")
        fb = _b0_magnitude(forcing_b0, "forcing.B0")
        if not math.isclose(
            gb, fb, rel_tol=_CONSISTENCY_RTOL, abs_tol=_CONSISTENCY_ATOL
        ):
            raise ProblemSpecError(
                f"imposed field disagrees: nondimensional_groups.B0={gb:g} vs "
                f"forcing.B0={fb:g}; state a single value"
            )
        return gb
    if group_b0 is not None:
        return _b0_magnitude(group_b0, "nondimensional_groups.B0")
    if forcing_b0 is not None:
        return _b0_magnitude(forcing_b0, "forcing.B0")
    return 0.0


def _resolve_coefficient(
    *,
    supplied: float | None,
    group: float | None,
    scale: float,
    coeff_label: str,
    group_label: str,
) -> tuple[float, float]:
    """Resolve a diffusivity and its nondimensional group from ``coeff = scale/group``.

    Returns ``(coeff, group_value)``. Rejects an over-specified inconsistent pair.
    """

    if supplied is None and group is None:
        raise ProblemSpecError(f"must supply {coeff_label} or {group_label}")
    if supplied is not None and group is not None:
        coeff = _positive(supplied, coeff_label)
        grp = _positive(group, group_label)
        derived = scale / coeff
        if not math.isclose(grp, derived, rel_tol=1.0e-6, abs_tol=1.0e-9):
            raise ProblemSpecError(
                f"over-specified inconsistent inputs: {group_label}={grp:g} but "
                f"{coeff_label}={coeff:g} implies {group_label}={derived:g} "
                f"(scale |S| h^2 = {scale:g})"
            )
        return coeff, grp
    if supplied is not None:
        coeff = _positive(supplied, coeff_label)
        return coeff, scale / coeff
    grp = _positive(group, group_label)  # type: ignore[arg-type]
    return scale / grp, grp


def resolve_physics(
    spec: dict[str, Any], *, precision: str = "float64"
) -> ResolvedPhysics:
    """Resolve one canonical physics object from a validated spec.

    Raises :class:`ProblemSpecError` on inconsistent or under-specified inputs so
    the failure happens before any JAX compilation.
    """

    groups = spec["nondimensional_groups"]
    physics = spec["physics"]
    geometry = spec["geometry"]
    h = _positive(_half_gap(spec), "half-gap h")

    Omega = _finite(groups.get("Omega", 0.0), "Omega")
    B0 = _single_b0(spec)

    domain = spec["domain"]
    Ly = _finite(domain.get("y_period", domain.get("theta_period", 0.0)), "y_period")
    Lz = _finite(domain.get("z_period", 0.0), "z_period")

    nu_in = groups.get("nu")
    Re_in = groups.get("Re", groups.get("Re_h"))
    eta_in = groups.get("eta_mag", groups.get("eta"))
    Rm_in = groups.get("Rm", groups.get("Rm_h"))

    # Velocity scale U0 = |S| h. S may be given (MRI); otherwise derive it from the
    # wall convention (U0 = U_wall, S = U_wall/h) via Re and nu when available.
    S_in = groups.get("S")
    if S_in is not None:
        S = _finite(S_in, "S")
        U0 = abs(S) * h
        velocity_scale = "shear"
        reynolds_convention = "Re_h = |S| h^2 / nu"
    else:
        # plain PCF: U0 = U_wall; derive from Re*nu/h when both present, else 1.
        U_wall = groups.get("U_wall")
        if U_wall is not None:
            U0 = abs(_finite(U_wall, "U_wall"))
        elif Re_in is not None and nu_in is not None:
            U0 = _positive(_finite(Re_in, "Re"), "Re") * _finite(nu_in, "nu") / h
        else:
            U0 = 1.0
        S = math.copysign(U0 / h, 1.0)
        velocity_scale = "wall"
        reynolds_convention = "Re = U_wall h / nu (U_wall = |S| h)"

    scale = U0 * h  # = |S| h^2
    nu, Re_h = _resolve_coefficient(
        supplied=None if nu_in is None else _finite(nu_in, "nu"),
        group=None if Re_in is None else _finite(Re_in, "Re"),
        scale=scale,
        coeff_label="nu",
        group_label="Re_h",
    )

    magnetic = spec.get("boundary_conditions", {}).get("magnetic")
    magnetic_bc = magnetic.get("type") if isinstance(magnetic, dict) else magnetic

    if physics in {"mhd", "mri"}:
        eta, Rm_h = _resolve_coefficient(
            supplied=None if eta_in is None else _finite(eta_in, "eta"),
            group=None if Rm_in is None else _finite(Rm_in, "Rm"),
            scale=scale,
            coeff_label="eta",
            group_label="Rm_h",
        )
        Pm = nu / eta
        if "Pm" in groups and groups["Pm"] is not None:
            pm_in = _finite(groups["Pm"], "Pm")
            if not math.isclose(pm_in, Pm, rel_tol=1.0e-6, abs_tol=1.0e-9):
                raise ProblemSpecError(
                    f"over-specified inconsistent Pm={pm_in:g}; nu/eta={Pm:g}"
                )
    else:
        eta = None
        Rm_h = None
        Pm = None

    raw_inputs = {
        "nondimensional_groups": dict(groups),
        "geometry": geometry,
        "physics": physics,
        "domain_x": domain.get("x"),
    }

    return ResolvedPhysics(
        h=h,
        S=S,
        Omega=Omega,
        nu=nu,
        eta=eta,
        B0=B0,
        Ly=Ly,
        Lz=Lz,
        U0=U0,
        Re_h=Re_h,
        Rm_h=Rm_h,
        Pm=Pm,
        velocity_scale=velocity_scale,
        reynolds_convention=reynolds_convention,
        magnetic_bc=magnetic_bc,
        precision=precision,
        raw_inputs=raw_inputs,
    )
