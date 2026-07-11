"""FJ-06: health classifier tests."""

from __future__ import annotations

import math

from production.classify import (
    OperationalStatus,
    ScientificClass,
    classify_scientific,
    fit_late_window_log_slope,
    operational_status_from_exception,
)


def _exp_series(rate, n=20, e0=1e-6, dt=0.1, stress=0.0):
    return [
        {
            "t": i * dt,
            "mag_energy_fluct": e0 * math.exp(rate * i * dt),
            "total_stress": stress,
        }
        for i in range(n)
    ]


def test_log_slope_recovers_growth_rate():
    series = _exp_series(0.5)
    fit = fit_late_window_log_slope(
        [r["t"] for r in series], [r["mag_energy_fluct"] for r in series]
    )
    assert fit["slope"] == __import__("pytest").approx(0.5, rel=1e-6)
    assert fit["samples"] >= 4
    assert fit["r_squared"] > 0.999


def test_growing_run_classified_growing():
    out = classify_scientific(_exp_series(0.5, stress=1.0))
    assert out["scientific_class"] == ScientificClass.GROWING.value


def test_decaying_run_classified_decayed():
    out = classify_scientific(_exp_series(-0.3, stress=1.0))
    assert out["scientific_class"] == ScientificClass.DECAYED.value


def test_flat_with_stress_and_stationarity_is_sustained():
    out = classify_scientific(
        _exp_series(0.0, stress=1e-2),
        stationary=True,
        stress_floor=1e-6,
        correlation_time=0.1,
    )
    assert out["scientific_class"] == ScientificClass.SUSTAINED.value


def test_finite_correlation_time_without_enough_independent_samples_is_not_sustained():
    out = classify_scientific(
        _exp_series(0.0, stress=1e-2),
        stationary=True,
        stress_floor=1e-6,
        correlation_time=10.0,
    )
    assert out["scientific_class"] == ScientificClass.MARGINAL.value
    assert out["independently_sampled"] is False
    assert "independent samples" in out["reason"]


def test_flat_without_stress_is_marginal():
    out = classify_scientific(
        _exp_series(0.0, stress=0.0), stationary=True, stress_floor=1e-6
    )
    assert out["scientific_class"] == ScientificClass.MARGINAL.value


def test_flat_alive_but_not_stationary_is_marginal_not_sustained():
    out = classify_scientific(
        _exp_series(0.0, stress=1.0), stationary=False, stress_floor=1e-6
    )
    assert out["scientific_class"] == ScientificClass.MARGINAL.value


def test_underresolved_is_inconclusive():
    out = classify_scientific(_exp_series(0.5, stress=1.0), underresolved=True)
    assert out["scientific_class"] == ScientificClass.INCONCLUSIVE.value


def test_insufficient_data_is_inconclusive():
    out = classify_scientific([{"t": 0.0, "mag_energy_fluct": 1e-6}])
    assert out["scientific_class"] == ScientificClass.INCONCLUSIVE.value


def test_operational_status_mapping():
    assert (
        operational_status_from_exception(FloatingPointError("nonfinite solver state"))
        == OperationalStatus.NAN_INF
    )
    assert (
        operational_status_from_exception(RuntimeError("divergence drift exceeded"))
        == OperationalStatus.BLEW_UP
    )
    assert (
        operational_status_from_exception(RuntimeError("energy runaway ceiling"))
        == OperationalStatus.BLEW_UP
    )
    assert (
        operational_status_from_exception(RuntimeError("cfl_total health guard"))
        == OperationalStatus.UNDERRESOLVED
    )
    assert (
        operational_status_from_exception(ValueError("something else"))
        == OperationalStatus.FAILED
    )
