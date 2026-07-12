"""Review round 3, blocker 2: the resolution/stability health contract."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from production import health
from production.oracles import run_supported_spec


def _vp_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vp_health_smoke",
        "spec_hash": "vp-health-spec-hash",
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "conducting"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 2e-2,
            "eta_mag": 2e-2,
            "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1e-3, "final_time": 0.04},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {"artifact_id": "x", "regeneration_command": "x"},
    }
    spec["nondimensional_groups"].update(groups)
    return spec


def test_tail_mask_selects_top_third_modes():
    # Monotone ordering (Chebyshev): last third of indices.
    mask = health._tail_mask(9, "chebyshev")
    assert list(np.nonzero(mask)[0]) == [6, 7, 8]
    # Full Fourier: |k| above two thirds of kmax, on both index wings.
    mask = health._tail_mask(8, "fourier_full")
    k = np.minimum(np.arange(8), 8 - np.arange(8))
    assert set(np.nonzero(mask)[0]) == set(np.nonzero(k >= 3)[0])


def test_spectral_tail_fractions_flag_energy_in_tail():
    low = np.zeros((8, 8), dtype=complex)
    low[0, 0] = 1.0
    assert health.spectral_tail_fractions([low], ("chebyshev", "fourier_full")) == (
        0.0,
        0.0,
    )
    high = np.zeros((8, 8), dtype=complex)
    high[7, 4] = 1.0  # top Chebyshev degree, Nyquist-adjacent Fourier mode
    tails = health.spectral_tail_fractions([high], ("chebyshev", "fourier_full"))
    assert tails == (1.0, 1.0)


def test_mode_occupancy_counts_active_fraction():
    arr = np.zeros(10, dtype=complex)
    arr[:3] = 1.0
    assert health.mode_occupancy([arr]) == pytest.approx(0.3)


def test_correlation_time_of_oscillation_is_finite_and_positive():
    t = np.arange(64) * 0.1
    rows = [{"t": float(ti), "total_stress": math.sin(0.5 * ti)} for ti in t]
    tau = health.correlation_time(rows)
    assert tau is not None and 0.0 < tau < t[-1]
    assert health.correlation_time(rows[:4]) is None  # too few samples


def test_underresolved_flag_thresholds():
    assert health.underresolved_from_scalars({}) is None
    assert (
        health.underresolved_from_scalars({"spectral_tail_max": 1e-6, "cfl_total": 0.1})
        is False
    )
    assert health.underresolved_from_scalars({"spectral_tail_max": 5e-3}) is True
    assert health.underresolved_from_scalars({"cfl_total": 2.0}) is True
    assert health.underresolved_from_scalars({"mode_occupancy": 0.999}) is True


def test_plateau_requires_stationarity_and_independent_samples():
    rows = [
        {
            "t": float(i),
            "total_energy": 2.0,
            "total_stress": 1.0 if i % 2 else 2.0,
        }
        for i in range(8)
    ]
    stats = health.plateau_window_stats(rows, checkpoint_time=7.0)
    assert stats["plateau_qualified"] is True
    assert stats["effective_independent_samples"] >= 5.0

    transient = [dict(row, total_energy=math.exp(0.5 * row["t"])) for row in rows]
    stats = health.plateau_window_stats(transient, checkpoint_time=7.0)
    assert stats["plateau_qualified"] is False
    assert any("not stationary" in reason for reason in stats["qualification_reasons"])


def test_curl_run_emits_health_scalars_and_budget_closes():
    """The workhorse emits the full health block and the shearing-box energy
    budget closes between cadence rows (empirically ~5e-5 on this run)."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=20, diagnostics_every=2)
    sc = out["scalars"]
    for key in (
        "cfl_advective_x",
        "cfl_advective_y",
        "cfl_advective_z",
        "cfl_alfven_x",
        "cfl_alfven_y",
        "cfl_alfven_z",
        "cfl_rotation",
        "cfl_shear_source",
        "cfl_diffusive",
        "cfl_total",
        "spectral_tail_x",
        "spectral_tail_y",
        "spectral_tail_z",
        "spectral_tail_max",
        "mode_occupancy",
    ):
        assert key in sc and math.isfinite(sc[key]), key
    assert 0.0 <= sc["spectral_tail_max"] <= 1.0
    assert 0.0 < sc["mode_occupancy"] <= 1.0
    assert sc["cfl_total"] < 1.0  # this smoke run is comfortably stable
    assert sc["cfl_rotation"] == pytest.approx(2.0 * (2.0 / 3.0) * 1e-3)
    assert sc["cfl_shear_source"] == pytest.approx(1e-3)
    assert sc["energy_budget_residual"] < 1e-2
    # dissipation streams in the cadence rows for the budget
    row = next(r for r in out["time_series"] if "dissipation_kinetic" in r)
    assert row["dissipation_kinetic"] > 0.0


def test_underresolved_run_is_quarantined_from_classification():
    from production.run_problem import _classification_metadata

    diagnostics = {
        "scalars": {"spectral_tail_max": 0.5, "cfl_total": 0.1},
        "time_series": [
            {"t": 0.0, "total_energy": 1.0, "total_stress": 0.1},
            {"t": 1.0, "total_energy": 2.0, "total_stress": 0.1},
            {"t": 2.0, "total_energy": 4.0, "total_stress": 0.1},
        ],
    }
    result = _classification_metadata(diagnostics)
    assert result["underresolved"] is True
    assert result["scientific_class"] == "inconclusive"
    assert "under-resolved" in result["reason"]


def test_runtime_health_guard_aborts_obvious_underresolution():
    from production.oracles import _raise_on_resolution_health

    with pytest.raises(RuntimeError, match="spectral_tail_max"):
        _raise_on_resolution_health(
            {"spectral_tail_max": 0.5, "cfl_total": 0.1, "mode_occupancy": 0.5},
            t=1.0,
            tstep=50,
        )

    with pytest.raises(RuntimeError, match="cfl_total"):
        _raise_on_resolution_health({"cfl_total": 1.1}, t=1.0, tstep=50)

    # Adaptive stepping owns CFL recovery between blocks, while this callback
    # continues to enforce spectral tails and mode occupancy immediately.
    _raise_on_resolution_health({"cfl_total": 1.1}, t=1.0, tstep=50, enforce_cfl=False)
