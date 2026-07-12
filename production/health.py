"""Resolution / stability health contract (review round 3, blocker 2).

Quantifies whether a production run is resolved and stable enough for its
numbers to be trusted, so a "lowest sensible resolution" campaign has an
objective floor:

* **CFL decomposition** -- advective (total velocity including the base
  shear), Alfven (total field including the imposed B0), per direction, plus
  the explicit-diffusion number reported separately (both PCF families
  integrate diffusion implicitly, so it bounds resolution, not stability).
* **Spectral tail fractions** -- per canonical axis, the coefficient-energy
  fraction carried by the top third of retained modes (the classic
  under-resolution detector for spectral methods).
* **Retained-mode occupancy** -- fraction of modes above a relative floor; an
  occupancy near 1 means the spectrum has no decaying headroom.
* **Correlation time** -- integrated autocorrelation of a cadence series key,
  for judging whether an averaging window holds independent samples.
* **Energy-budget residual** (curl workhorse) -- relative closure of
  d(E_phys)/dt = S * V * total_stress - dissipation between cadence rows.

Wall-BC enforcement is exact by construction in these Galerkin bases
(boundary-adapted composite spaces), so there is no numeric wall residual to
report; a PDE-residual-at-wall diagnostic is tracked in
production/KNOWN_ISSUES.md.

``underresolved_from_scalars`` maps the health scalars onto the FJ-06
classifier's ``underresolved`` quarantine flag.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# A run whose top-third modes carry more than this fraction of coefficient
# energy, or whose explicit CFL exceeds CFL_LIMIT, is quarantined from
# scientific-class inference (classified inconclusive) instead of trusted.
SPECTRAL_TAIL_LIMIT = 1.0e-3
CFL_LIMIT = 1.0
MODE_OCCUPANCY_LIMIT = 0.99
# Runtime aborts target catastrophic underresolution.  The stricter limits
# above still quarantine completed results from scientific inference; using a
# separate abort ceiling avoids killing a seeded startup transient that is
# actively shedding its high-mode content.
EARLY_ABORT_SPECTRAL_TAIL_LIMIT = 1.0e-1
EARLY_ABORT_MODE_OCCUPANCY_LIMIT = 0.999
EARLY_ABORT_STARTUP_STEPS = 50
# Mode counts as occupied when its |c|^2 exceeds this fraction of the peak.
OCCUPANCY_RELATIVE_FLOOR = 1.0e-20

# A stationary-looking trace is not a plateau unless its averaging window spans
# several correlation times.  These constants are shared by parent-bank
# qualification and the sustained-state classifier.
MIN_CORRELATION_SAMPLES = 8
MIN_INDEPENDENT_SAMPLES = 5.0
PLATEAU_STATIONARITY_TOLERANCE = 5.0e-2

# Native coefficient-axis kinds per PCF solver family (see production/axes.py
# for the native orders; jaxfun keeps both periodic axes as full complex
# Fourier, verified against the solver arrays).
_CURL_AXIS_KINDS = ("chebyshev", "fourier_full", "fourier_full")  # (x, y, z)
_PRIMITIVE_AXIS_KINDS = ("fourier_full", "fourier_full", "chebyshev")  # (y, z, x)


def _tail_mask(size: int, kind: str) -> np.ndarray:
    """Boolean mask selecting the top third of retained modes along one axis."""

    idx = np.arange(size)
    if kind == "fourier_full":
        k = np.minimum(idx, size - idx)
        kmax = max(1, size // 2)
        return k >= max(1, int(math.ceil(2.0 * kmax / 3.0)))
    # Chebyshev/Legendre composite (or any monotone mode ordering).
    return idx >= int(math.ceil(2.0 * size / 3.0))


def spectral_tail_fractions(
    coefficients: Sequence[Any], axis_kinds: Sequence[str]
) -> tuple[float, ...]:
    """Per-axis tail coefficient-energy fraction, summed over all fields."""

    arrays = [np.asarray(c) for c in coefficients]
    n_axes = len(axis_kinds)
    tail = [0.0] * n_axes
    total = 0.0
    for arr in arrays:
        power = np.abs(arr) ** 2
        total += float(power.sum())
        for axis, kind in enumerate(axis_kinds):
            mask = _tail_mask(arr.shape[axis], kind)
            selector = [slice(None)] * arr.ndim
            selector[axis] = mask
            tail[axis] += float(power[tuple(selector)].sum())
    if total <= 0.0:
        return tuple(0.0 for _ in range(n_axes))
    return tuple(t / total for t in tail)


def mode_occupancy(coefficients: Sequence[Any]) -> float:
    """Fraction of retained modes above the relative floor, over all fields."""

    active = 0
    count = 0
    for c in coefficients:
        power = np.abs(np.asarray(c)) ** 2
        count += power.size
        peak = float(power.max()) if power.size else 0.0
        if peak > 0.0:
            active += int((power > OCCUPANCY_RELATIVE_FLOOR * peak).sum())
    return active / count if count else 0.0


def _max_abs(values: Any) -> float:
    return float(np.max(np.abs(np.asarray(values))))


def _cfl_scalars(
    *,
    u_max: tuple[float, float, float],
    b_max: tuple[float, float, float],
    spacings: tuple[float, float, float],
    dt: float,
    nu: float,
    eta: float,
    rotation_rate: float,
    shear_rate: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    directional_total = 0.0
    for axis, label in enumerate(("x", "y", "z")):
        adv = u_max[axis] * dt / spacings[axis]
        alf = b_max[axis] * dt / spacings[axis]
        out[f"cfl_advective_{label}"] = adv
        out[f"cfl_alfven_{label}"] = alf
        directional_total = max(directional_total, adv + alf)
    out["cfl_diffusive"] = max(nu, eta) * dt / min(spacings) ** 2
    # Rotation and the linear shear-source coupling are rates rather than
    # grid-crossing speeds, but they constrain an explicit step in the same
    # nondimensional way.  Keep them separate so timestep calibration can see
    # which part of the operator is limiting.
    out["cfl_rotation"] = 2.0 * abs(rotation_rate) * dt
    out["cfl_shear_source"] = abs(shear_rate) * dt
    # Explicit-terms CFL; diffusion is implicit in both families and reported
    # separately above.
    out["cfl_total"] = max(
        directional_total,
        out["cfl_rotation"],
        out["cfl_shear_source"],
    )
    return out


def curl_health_scalars(solver: Any, state: Any) -> dict[str, float]:
    """Health scalars for the vector-potential (curl) workhorse family."""

    up = solver.total_velocity_physical(state.flow)
    B = solver.update_B_from_A(state.A)
    b_total = solver._total_B_physical(B, padded=False)

    x1d = np.sort(np.unique(np.asarray(solver.X[0])))
    dx = float(np.diff(x1d).min())
    Ly = float(solver.domain[1][1] - solver.domain[1][0])
    Lz = float(solver.domain[2][1] - solver.domain[2][0])
    ny = int(np.asarray(state.flow.u[0]).shape[1])
    nz = int(np.asarray(state.flow.u[0]).shape[2])
    spacings = (dx, Ly / ny, Lz / nz)

    scalars = _cfl_scalars(
        u_max=tuple(_max_abs(c) for c in up),
        b_max=tuple(_max_abs(c) for c in b_total),
        spacings=spacings,
        dt=float(solver.dt),
        nu=float(solver.nu),
        eta=float(solver.eta),
        rotation_rate=float(solver.omega),
        shear_rate=float(solver.shear_rate),
    )
    coefficients = list(state.flow.u) + list(B)
    tail_x, tail_y, tail_z = spectral_tail_fractions(coefficients, _CURL_AXIS_KINDS)
    scalars.update(
        {
            "spectral_tail_x": tail_x,
            "spectral_tail_y": tail_y,
            "spectral_tail_z": tail_z,
            "spectral_tail_max": max(tail_x, tail_y, tail_z),
            "mode_occupancy": mode_occupancy(coefficients),
        }
    )
    return scalars


def primitive_health_scalars(solver: Any, state: Any) -> dict[str, float]:
    """Health scalars for the primitive-b 3-D family (native order (y, z, x))."""

    fields = solver.fields_physical(state)
    # Total-velocity bound: the base flow U_y = -S x contributes |S| h.
    u_max = (
        _max_abs(fields[0]),
        _max_abs(fields[1]) + abs(float(solver.S)) * 1.0,
        _max_abs(fields[2]),
    )
    b_max = (
        _max_abs(fields[3]),
        _max_abs(fields[4]),
        _max_abs(fields[5]) + abs(float(solver.B0)),
    )
    x1d = np.sort(np.unique(np.asarray(solver.X)))
    dx = float(np.diff(x1d).min())
    spacings = (
        dx,
        float(solver.Ly) / int(solver.Ny),
        float(solver.Lz) / int(solver.Nz),
    )
    scalars = _cfl_scalars(
        u_max=u_max,
        b_max=b_max,
        spacings=spacings,
        dt=float(solver.dt),
        nu=float(solver.nu),
        eta=float(solver.eta_mag),
        rotation_rate=float(solver.omega),
        shear_rate=float(solver.S),
    )
    coefficients = list(state.x)
    tail_y, tail_z, tail_x = spectral_tail_fractions(
        coefficients, _PRIMITIVE_AXIS_KINDS
    )
    scalars.update(
        {
            "spectral_tail_x": tail_x,
            "spectral_tail_y": tail_y,
            "spectral_tail_z": tail_z,
            "spectral_tail_max": max(tail_x, tail_y, tail_z),
            "mode_occupancy": mode_occupancy(coefficients),
        }
    )
    return scalars


def correlation_time(
    rows: Sequence[dict[str, Any]], *, key: str = "total_stress"
) -> float | None:
    """Integrated autocorrelation time of ``key`` over uniform cadence rows.

    Returns ``None`` with fewer than 8 usable samples. Uses the standard
    initial-positive-sequence estimator: tau = dt * (1 + 2 * sum(rho_k)) up to
    the first non-positive autocorrelation.
    """

    samples = [
        (float(row["t"]), float(row[key]))
        for row in rows
        if key in row and "t" in row and math.isfinite(float(row[key]))
    ]
    if len(samples) < MIN_CORRELATION_SAMPLES:
        return None
    times = np.asarray([s[0] for s in samples])
    values = np.asarray([s[1] for s in samples])
    dt_rows = np.diff(times)
    dt_row = float(np.median(dt_rows))
    if dt_row <= 0.0:
        return None
    centered = values - values.mean()
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        return None
    acf_sum = 0.0
    n = centered.size
    for lag in range(1, n - 1):
        rho = float(np.dot(centered[:-lag], centered[lag:])) / variance
        if rho <= 0.0:
            break
        acf_sum += rho
    return dt_row * (1.0 + 2.0 * acf_sum)


def effective_independent_samples(
    rows: Sequence[dict[str, Any]],
    *,
    correlation: float | None,
    key: str = "total_stress",
) -> float | None:
    """Number of correlation-time-sized samples covered by a cadence window."""

    if (
        correlation is None
        or not math.isfinite(float(correlation))
        or correlation <= 0.0
    ):
        return None
    times = [
        float(row["t"])
        for row in rows
        if key in row
        and "t" in row
        and math.isfinite(float(row[key]))
        and math.isfinite(float(row["t"]))
    ]
    if len(times) < 2:
        return None
    duration = max(times) - min(times)
    if duration <= 0.0:
        return None
    return float(duration / float(correlation))


def plateau_window_stats(
    rows: Sequence[dict[str, Any]],
    *,
    checkpoint_time: float,
    energy_key: str = "total_energy",
    stress_key: str = "total_stress",
    max_samples: int = 512,
    min_independent_samples: float = MIN_INDEPENDENT_SAMPLES,
    stationarity_tolerance: float = PLATEAU_STATIONARITY_TOLERANCE,
) -> dict[str, Any]:
    """Qualify a retained checkpoint from its trailing canonical diagnostics.

    A checkpoint is selectable as a quench parent only when its diagnostic
    window reaches the checkpoint, has enough samples for a finite stress
    correlation estimate, spans the declared number of independent samples,
    and has stationary energy and persistent finite stress.
    """

    usable = [
        row
        for row in rows
        if all(key in row for key in ("t", energy_key, stress_key))
        and all(math.isfinite(float(row[key])) for key in ("t", energy_key, stress_key))
        and float(row["t"]) <= float(checkpoint_time) + 1.0e-12
    ][-max(1, int(max_samples)) :]
    reasons: list[str] = []
    samples = len(usable)
    current = bool(
        usable and abs(float(usable[-1]["t"]) - float(checkpoint_time)) <= 1.0e-12
    )
    if not current:
        reasons.append("no diagnostic sample at checkpoint time")
    if samples < MIN_CORRELATION_SAMPLES:
        reasons.append(
            f"need at least {MIN_CORRELATION_SAMPLES} plateau samples, got {samples}"
        )

    energies = [float(row[energy_key]) for row in usable]
    relative_change: float | None = None
    stationary = False
    if samples >= 4:
        quarter = max(1, samples // 4)
        previous_mean = float(np.mean(energies[-2 * quarter : -quarter]))
        current_mean = float(np.mean(energies[-quarter:]))
        denom = max(abs(previous_mean), abs(current_mean), 1.0e-300)
        relative_change = abs(current_mean - previous_mean) / denom
        stationary = relative_change <= float(stationarity_tolerance)
        if not stationary:
            reasons.append(
                f"energy window is not stationary ({relative_change:.3e} > "
                f"{stationarity_tolerance:.3e})"
            )
    else:
        reasons.append("insufficient samples for stationarity check")

    tau = correlation_time(usable, key=stress_key)
    independent = effective_independent_samples(usable, correlation=tau, key=stress_key)
    if tau is None:
        reasons.append("finite stress correlation time unavailable")
    if independent is None or independent < float(min_independent_samples):
        actual = "unavailable" if independent is None else f"{independent:.3g}"
        reasons.append(
            f"need {min_independent_samples:g} independent stress samples, got {actual}"
        )

    mean_energy = float(np.mean(energies)) if energies else None
    stresses = [float(row[stress_key]) for row in usable]
    mean_stress = float(np.mean(stresses)) if stresses else None
    persistent_stress = mean_stress is not None and abs(mean_stress) > 0.0
    if not persistent_stress:
        reasons.append("persistent nonzero stress unavailable")

    return {
        "samples": samples,
        "window_start_t": float(usable[0]["t"]) if usable else None,
        "window_end_t": float(usable[-1]["t"]) if usable else None,
        "checkpoint_time": float(checkpoint_time),
        "diagnostics_current": current,
        "mean_total_energy": mean_energy,
        "mean_total_stress": mean_stress,
        "stationarity_relative_change": relative_change,
        "stationarity_tolerance": float(stationarity_tolerance),
        "stationary": stationary,
        "correlation_time_total_stress": tau,
        "effective_independent_samples": independent,
        "required_independent_samples": float(min_independent_samples),
        "persistent_stress": persistent_stress,
        "plateau_qualified": not reasons,
        "qualification_reasons": reasons,
    }


def energy_budget_residual(
    rows: Sequence[dict[str, Any]], *, shear_rate: float, volume: float
) -> float | None:
    """Median relative closure of the shearing-box energy budget across rows.

    Physical energies: E_phys = 0.5 * total_energy (the family's integral_abs2
    scalars), injection P = S * V * total_stress (volume-mean stresses), and
    dissipation from the ``dissipation_kinetic``/``dissipation_magnetic`` row
    keys (already physical: nu * integral |grad u|^2 etc.). Imposed-field
    exchange terms cancel in the (kinetic + magnetic) sum for periodic y/z and
    no-slip walls, so the residual measures discretization/commutation error.
    """

    required = (
        "t",
        "total_energy",
        "total_stress",
        "dissipation_kinetic",
        "dissipation_magnetic",
    )
    usable = [row for row in rows if all(k in row for k in required)]
    if len(usable) < 2:
        return None
    residuals = []
    for first, second in zip(usable[:-1], usable[1:], strict=False):
        dt_pair = float(second["t"]) - float(first["t"])
        if dt_pair <= 0.0:
            continue
        lhs = (
            0.5
            * (float(second["total_energy"]) - float(first["total_energy"]))
            / dt_pair
        )
        stress_mid = 0.5 * (
            float(first["total_stress"]) + float(second["total_stress"])
        )
        dissipation_mid = 0.5 * (
            float(first["dissipation_kinetic"])
            + float(first["dissipation_magnetic"])
            + float(second["dissipation_kinetic"])
            + float(second["dissipation_magnetic"])
        )
        rhs = float(shear_rate) * float(volume) * stress_mid - dissipation_mid
        scale = max(abs(lhs), abs(rhs), dissipation_mid, 1.0e-300)
        residuals.append(abs(lhs - rhs) / scale)
    if not residuals:
        return None
    return float(np.median(residuals))


def underresolved_from_scalars(scalars: dict[str, Any]) -> bool | None:
    """FJ-06 quarantine flag from the health scalars (None when absent)."""

    tail = scalars.get("spectral_tail_max")
    cfl = scalars.get("cfl_total")
    occupancy = scalars.get("mode_occupancy")
    if tail is None and cfl is None and occupancy is None:
        return None
    flagged = False
    if tail is not None and float(tail) > SPECTRAL_TAIL_LIMIT:
        flagged = True
    if cfl is not None and float(cfl) > CFL_LIMIT:
        flagged = True
    if occupancy is not None and float(occupancy) > MODE_OCCUPANCY_LIMIT:
        flagged = True
    return flagged
