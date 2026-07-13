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

Taylor-Couette keeps its traditional inner-cylinder Reynolds number distinct
from the midpoint-local shear Reynolds number used for cross-geometry
comparisons::

    Re_TC = |Omega1| R1 (R2 - R1) / nu,
    Re_h  = |S_mid| h^2 / nu,  S_mid = -r_mid dOmega/dr.

Both are derived from the same coefficient and recorded in the run metadata.
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

    geometry: str
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
    # --- Taylor-Couette geometry and native controls (None for Cartesian flows) ---
    R1: float | None = None
    R2: float | None = None
    Omega1: float | None = None
    Omega2: float | None = None
    gap: float | None = None
    r_mid: float | None = None
    curvature: float | None = None
    Omega_mid: float | None = None
    S_mid: float | None = None
    q_mid: float | None = None
    theta_period: float | None = None
    Re_TC: float | None = None
    Rm_TC: float | None = None

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
    amplitude = _finite(value, label)
    if amplitude < 0.0:
        raise ProblemSpecError(
            f"{label} is a field amplitude and must be nonnegative; "
            "use an explicit component vector for signed direction"
        )
    return amplitude


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


def _consistent_alias(
    groups: dict[str, Any], primary: str, legacy: str
) -> float | None:
    """Return one value for a named group and its legacy alias."""

    primary_value = groups.get(primary)
    legacy_value = groups.get(legacy)
    if primary_value is None and legacy_value is None:
        return None
    if primary_value is None:
        return _finite(legacy_value, legacy)
    value = _finite(primary_value, primary)
    if legacy_value is not None:
        alias = _finite(legacy_value, legacy)
        if not math.isclose(value, alias, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ProblemSpecError(
                f"{primary}={value:g} disagrees with legacy alias {legacy}={alias:g}"
            )
    return value


def _resolve_tc_coefficient(
    *,
    supplied: float | None,
    local_group: float | None,
    native_group: float | None,
    local_scale: float,
    native_scale: float,
    coeff_label: str,
    local_label: str,
    native_label: str,
) -> tuple[float, float, float]:
    """Resolve one TC diffusivity against local and native Reynolds numbers."""

    if supplied is not None:
        coeff = _positive(supplied, coeff_label)
    elif local_group is not None:
        group = _positive(local_group, local_label)
        if local_scale <= 0.0:
            raise ProblemSpecError(
                f"cannot derive {coeff_label} from {local_label}: "
                "midpoint shear is zero"
            )
        coeff = local_scale / group
    elif native_group is not None:
        group = _positive(native_group, native_label)
        if native_scale <= 0.0:
            raise ProblemSpecError(
                f"cannot derive {coeff_label} from {native_label}: inner-cylinder "
                "velocity scale is zero"
            )
        coeff = native_scale / group
    else:
        raise ProblemSpecError(
            f"must supply {coeff_label}, {local_label}, or {native_label}"
        )

    resolved_local = local_scale / coeff
    resolved_native = native_scale / coeff
    for supplied_group, resolved_group, label in (
        (local_group, resolved_local, local_label),
        (native_group, resolved_native, native_label),
    ):
        if supplied_group is not None:
            value = _positive(supplied_group, label)
            if not math.isclose(value, resolved_group, rel_tol=1.0e-6, abs_tol=1.0e-9):
                raise ProblemSpecError(
                    f"over-specified inconsistent inputs: {label}={value:g} but "
                    f"{coeff_label}={coeff:g} implies {label}={resolved_group:g}"
                )
    return coeff, resolved_local, resolved_native


def _resolve_taylor_couette_physics(
    spec: dict[str, Any], *, precision: str
) -> ResolvedPhysics:
    """Resolve TC native and midpoint-local controls from one coefficient set."""

    groups = spec["nondimensional_groups"]
    physics = spec["physics"]
    domain = spec["domain"]
    r_lo, r_hi = (_finite(value, "domain.r") for value in domain["r"])
    R1 = _positive(_finite(groups.get("R1"), "R1"), "R1")
    R2 = _positive(_finite(groups.get("R2"), "R2"), "R2")
    if not R2 > R1:
        raise ProblemSpecError("Taylor-Couette requires R2 > R1")
    if not (
        math.isclose(r_lo, R1, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(r_hi, R2, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ProblemSpecError(
            f"domain.r={[r_lo, r_hi]} must match cylinder radii R1={R1:g}, R2={R2:g}"
        )
    expected_ratio = R1 / R2
    if groups.get("radius_ratio") is not None:
        radius_ratio = _positive(
            _finite(groups["radius_ratio"], "radius_ratio"), "radius_ratio"
        )
        if not math.isclose(
            radius_ratio, expected_ratio, rel_tol=1.0e-9, abs_tol=1.0e-12
        ):
            raise ProblemSpecError(
                f"radius_ratio={radius_ratio:g} but R1/R2={expected_ratio:g}"
            )

    Omega1 = _finite(groups.get("Omega1"), "Omega1")
    Omega2 = _finite(groups.get("Omega2"), "Omega2")
    gap = R2 - R1
    h = 0.5 * gap
    r_mid = 0.5 * (R1 + R2)
    curvature = h / r_mid
    denominator = R2**2 - R1**2
    profile_a = (Omega2 * R2**2 - Omega1 * R1**2) / denominator
    profile_b = (Omega1 - Omega2) * R1**2 * R2**2 / denominator
    Omega_mid = profile_a + profile_b / r_mid**2
    S_mid = 2.0 * profile_b / r_mid**2
    q_mid = S_mid / Omega_mid if abs(Omega_mid) > 0.0 else None
    local_scale = abs(S_mid) * h**2
    native_scale = abs(Omega1) * R1 * gap

    nu_in = groups.get("nu")
    Re_h_in = groups.get("Re_h")
    Re_TC_in = _consistent_alias(groups, "Re_TC", "Re")
    nu, Re_h, Re_TC = _resolve_tc_coefficient(
        supplied=None if nu_in is None else _finite(nu_in, "nu"),
        local_group=None if Re_h_in is None else _finite(Re_h_in, "Re_h"),
        native_group=Re_TC_in,
        local_scale=local_scale,
        native_scale=native_scale,
        coeff_label="nu",
        local_label="Re_h",
        native_label="Re_TC",
    )

    if physics in {"mhd", "mri"}:
        eta_in = _consistent_alias(groups, "eta_mag", "eta")
        Rm_h_in = groups.get("Rm_h")
        Rm_TC_in = _consistent_alias(groups, "Rm_TC", "Rm")
        eta, Rm_h, Rm_TC = _resolve_tc_coefficient(
            supplied=None if eta_in is None else _finite(eta_in, "eta"),
            local_group=None if Rm_h_in is None else _finite(Rm_h_in, "Rm_h"),
            native_group=Rm_TC_in,
            local_scale=local_scale,
            native_scale=native_scale,
            coeff_label="eta",
            local_label="Rm_h",
            native_label="Rm_TC",
        )
        Pm = nu / eta
        if groups.get("Pm") is not None:
            pm_in = _finite(groups["Pm"], "Pm")
            if not math.isclose(pm_in, Pm, rel_tol=1.0e-6, abs_tol=1.0e-9):
                raise ProblemSpecError(
                    f"over-specified inconsistent Pm={pm_in:g}; nu/eta={Pm:g}"
                )
    else:
        eta = None
        Rm_h = None
        Rm_TC = None
        Pm = None

    magnetic = spec.get("boundary_conditions", {}).get("magnetic")
    magnetic_bc = magnetic.get("type") if isinstance(magnetic, dict) else magnetic
    theta_period = _positive(
        _finite(domain["theta_period"], "theta_period"), "theta_period"
    )
    Lz = _positive(_finite(domain["z_period"], "z_period"), "z_period")
    U0 = abs(S_mid) * h
    return ResolvedPhysics(
        geometry="taylor_couette",
        h=h,
        S=S_mid,
        Omega=Omega_mid,
        nu=nu,
        eta=eta,
        B0=_single_b0(spec),
        Ly=theta_period * r_mid,
        Lz=Lz,
        U0=U0,
        Re_h=Re_h,
        Rm_h=Rm_h,
        Pm=Pm,
        velocity_scale="midpoint_shear",
        reynolds_convention=(
            "Re_h = |S_mid| h^2 / nu; Re_TC = |Omega1| R1 (R2-R1) / nu"
        ),
        magnetic_bc=magnetic_bc,
        precision=precision,
        raw_inputs={
            "nondimensional_groups": dict(groups),
            "geometry": "taylor_couette",
            "physics": physics,
            "domain_r": list(domain["r"]),
        },
        R1=R1,
        R2=R2,
        Omega1=Omega1,
        Omega2=Omega2,
        gap=gap,
        r_mid=r_mid,
        curvature=curvature,
        Omega_mid=Omega_mid,
        S_mid=S_mid,
        q_mid=q_mid,
        theta_period=theta_period,
        Re_TC=Re_TC,
        Rm_TC=Rm_TC,
    )


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
    if geometry == "taylor_couette":
        return _resolve_taylor_couette_physics(spec, precision=precision)
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
        geometry=geometry,
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
