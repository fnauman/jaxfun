"""Run health classification (FJ-06).

Separates an **operational status** (did the integration run cleanly?) from a
**scientific class** (what did the physics do?). The scientific class is inferred
from a late-window least-squares log-slope of the fluctuation energy above a
declared noise floor -- never from a two-point ratio or "alive at final time".
``sustained`` additionally requires persistent stress and a stationarity criterion,
and an under-resolved run is quarantined from threshold inference.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import numpy as np

from .health import MIN_INDEPENDENT_SAMPLES, effective_independent_samples


class OperationalStatus(str, Enum):
    """Did the integration run cleanly?"""

    COMPLETED = "completed"
    EARLY_STOP = "early_stop"
    NAN_INF = "nan_inf"
    BLEW_UP = "blew_up"
    WALLTIME = "walltime"
    FAILED = "failed"
    UNDERRESOLVED = "underresolved"


class ScientificClass(str, Enum):
    """What did the physics do?"""

    GROWING = "growing"
    SUSTAINED = "sustained"
    MARGINAL = "marginal"
    DECAYED = "decayed"
    INCONCLUSIVE = "inconclusive"


def fit_late_window_log_slope(
    times: list[float],
    values: list[float],
    *,
    window_fraction: float = 0.5,
    noise_floor: float = 0.0,
    min_samples: int = 4,
) -> dict[str, Any]:
    """Least-squares slope of ``log(value)`` vs ``time`` over the trailing window.

    Only strictly-positive samples above ``noise_floor`` in the trailing
    ``window_fraction`` of the series are fitted. Returns slope, its standard
    error, the sample count, the coefficient of determination, and the fit window.
    """

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.size != v.size or t.size < min_samples:
        return _empty_fit()
    t0 = t[int((1.0 - window_fraction) * t.size)]
    mask = (t >= t0) & (v > noise_floor) & (v > 0.0) & np.isfinite(v)
    tw = t[mask]
    vw = v[mask]
    if tw.size < min_samples or np.ptp(tw) <= 0.0:
        return _empty_fit()
    logv = np.log(vw)
    slope, intercept = np.polyfit(tw, logv, 1)
    pred = slope * tw + intercept
    ss_res = float(np.sum((logv - pred) ** 2))
    ss_tot = float(np.sum((logv - np.mean(logv)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    n = tw.size
    if n > 2:
        residual_var = ss_res / (n - 2)
        t_var = float(np.sum((tw - np.mean(tw)) ** 2))
        stderr = math.sqrt(residual_var / t_var) if t_var > 0.0 else float("inf")
    else:
        stderr = float("inf")
    return {
        "slope": float(slope),
        "stderr": float(stderr),
        "samples": int(n),
        "r_squared": float(r2),
        "window_start_t": float(tw[0]),
        "window_end_t": float(tw[-1]),
        "noise_floor": float(noise_floor),
    }


def _empty_fit() -> dict[str, Any]:
    return {
        "slope": None,
        "stderr": None,
        "samples": 0,
        "r_squared": None,
        "window_start_t": None,
        "window_end_t": None,
        "noise_floor": None,
    }


def classify_scientific(
    series: list[dict[str, Any]],
    *,
    energy_key: str = "mag_energy_fluct",
    fallback_energy_key: str = "total_energy",
    stress_key: str = "total_stress",
    noise_floor: float = 0.0,
    growth_tol: float = 1.0e-3,
    stationary: bool | None = None,
    stress_floor: float = 0.0,
    underresolved: bool = False,
    correlation_time: float | None = None,
    min_independent_samples: float = MIN_INDEPENDENT_SAMPLES,
) -> dict[str, Any]:
    """Classify the physics of a run from its cadence series (FJ-06)."""

    if underresolved:
        return {
            "scientific_class": ScientificClass.INCONCLUSIVE.value,
            "reason": "under-resolved: quarantined from threshold inference",
            "fit": _empty_fit(),
        }

    key = (
        energy_key if any(energy_key in row for row in series) else fallback_energy_key
    )
    times = [float(row["t"]) for row in series if key in row and "t" in row]
    values = [float(row[key]) for row in series if key in row and "t" in row]
    fit = fit_late_window_log_slope(times, values, noise_floor=noise_floor)
    slope = fit["slope"]

    if slope is None:
        return {
            "scientific_class": ScientificClass.INCONCLUSIVE.value,
            "reason": "insufficient positive fluctuation-energy samples to fit a slope",
            "fit": fit,
            "energy_key": key,
        }

    late_stress = _late_window_mean(series, stress_key)
    persistent_stress = late_stress is not None and abs(late_stress) > stress_floor
    late_start = int(0.5 * len(series))
    independent_samples = effective_independent_samples(
        series[late_start:], correlation=correlation_time, key=stress_key
    )
    independently_sampled = (
        independent_samples is not None
        and independent_samples >= float(min_independent_samples)
    )

    if slope > growth_tol:
        cls = ScientificClass.GROWING
        reason = f"late-window log-slope {slope:.3e} > +{growth_tol:g}"
    elif slope < -growth_tol:
        cls = ScientificClass.DECAYED
        reason = f"late-window log-slope {slope:.3e} < -{growth_tol:g}"
    else:
        # Marginal band: sustained only with persistent stress, stationarity,
        # and an averaging window spanning the required correlation times.
        if persistent_stress and stationary and independently_sampled:
            cls = ScientificClass.SUSTAINED
            reason = (
                f"slope {slope:.3e} within +/-{growth_tol:g}, persistent stress "
                f"({late_stress:.3e}), stationary, and {independent_samples:.3g} "
                "independent samples"
            )
        else:
            cls = ScientificClass.MARGINAL
            missing = []
            if not persistent_stress:
                missing.append("no persistent stress")
            if not stationary:
                missing.append("not stationary")
            if not independently_sampled:
                actual = (
                    "unavailable"
                    if independent_samples is None
                    else f"{independent_samples:.3g}"
                )
                missing.append(
                    f"only {actual} independent samples (need "
                    f"{min_independent_samples:g})"
                )
            reason = f"slope {slope:.3e} within +/-{growth_tol:g} but " + ", ".join(
                missing
            )
    return {
        "scientific_class": cls.value,
        "reason": reason,
        "fit": fit,
        "energy_key": key,
        "late_window_stress": late_stress,
        "persistent_stress": bool(persistent_stress),
        "stationary": stationary,
        "correlation_time_total_stress": correlation_time,
        "effective_independent_samples": independent_samples,
        "required_independent_samples": float(min_independent_samples),
        "independently_sampled": independently_sampled,
    }


def _late_window_mean(
    series: list[dict[str, Any]], key: str, *, window_fraction: float = 0.5
) -> float | None:
    vals = [float(row[key]) for row in series if key in row]
    if not vals:
        return None
    start = int((1.0 - window_fraction) * len(vals))
    window = vals[start:]
    return float(np.mean(window)) if window else None


def operational_status_from_exception(exc: BaseException) -> OperationalStatus:
    """Map a solver exception to an operational status (FJ-06)."""

    message = str(exc).lower()
    if "nonfinite" in message or "nan" in message or "inf" in message:
        return OperationalStatus.NAN_INF
    if (
        "underresolved" in message
        or "spectral_tail" in message
        or "cfl_total" in message
        or "mode_occupancy" in message
    ):
        return OperationalStatus.UNDERRESOLVED
    if "runaway" in message or "ceiling" in message or "blew up" in message:
        return OperationalStatus.BLEW_UP
    if "divergence" in message:
        return OperationalStatus.BLEW_UP
    if "walltime" in message or "timeout" in message:
        return OperationalStatus.WALLTIME
    return OperationalStatus.FAILED
