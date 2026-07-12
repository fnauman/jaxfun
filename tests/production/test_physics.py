"""FJ-00: single resolved-physics contract tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from production.physics import resolve_physics
from production.problem_spec import ProblemSpecError

ROOT = Path(__file__).resolve().parents[2]


def _mri_spec(**group_overrides):
    spec = {
        "geometry": "pcf",
        "physics": "mri",
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 0.001,
            "eta_mag": 0.001,
            "Re": 1000.0,
            "Rm": 1000.0,
            "Pm": 1.0,
            "B0": 0.025,
        },
        "boundary_conditions": {
            "velocity": {"type": "no_slip"},
            "magnetic": {"type": "conducting"},
        },
        "forcing": {"B0": 0.025},
    }
    spec["nondimensional_groups"].update(group_overrides)
    return spec


def test_resolves_consistent_mri_spec():
    rp = resolve_physics(_mri_spec())
    assert rp.h == 1.0
    assert rp.nu == pytest.approx(0.001)
    assert rp.eta == pytest.approx(0.001)
    assert rp.Re_h == pytest.approx(1000.0)  # |S| h^2 / nu
    assert rp.Rm_h == pytest.approx(1000.0)
    assert rp.Pm == pytest.approx(1.0)
    assert pytest.approx(0.025) == rp.B0
    assert rp.velocity_scale == "shear"


def test_changing_Re_changes_nu():
    """FJ-00 acceptance: changing only Re_h changes the resolved nu."""
    groups = _mri_spec()["nondimensional_groups"]
    # drop nu so Re drives it
    del groups["nu"]
    del groups["Pm"]
    spec = _mri_spec()
    spec["nondimensional_groups"] = groups
    rp1 = resolve_physics(spec)
    spec2 = copy.deepcopy(spec)
    spec2["nondimensional_groups"]["Re"] = 2000.0
    rp2 = resolve_physics(spec2)
    assert rp2.nu == pytest.approx(rp1.nu / 2.0)  # nu = |S| h^2 / Re
    assert rp2.nu != rp1.nu


def test_inconsistent_Re_and_nu_rejected():
    """FJ-00 acceptance: inconsistent {Re, nu} fails before compilation."""
    spec = _mri_spec(Re=500.0)  # nu=0.001 implies Re=1000, not 500
    with pytest.raises(ProblemSpecError):
        resolve_physics(spec)


def test_inconsistent_Rm_and_eta_rejected():
    spec = _mri_spec(Rm=500.0)  # eta=0.001 implies Rm=1000
    with pytest.raises(ProblemSpecError):
        resolve_physics(spec)


def test_inconsistent_Pm_rejected():
    spec = _mri_spec(Pm=2.0)  # nu/eta = 1
    with pytest.raises(ProblemSpecError):
        resolve_physics(spec)


def test_b0_divergence_between_group_and_forcing_rejected():
    spec = _mri_spec()
    spec["forcing"]["B0"] = 0.9  # != groups.B0 = 0.025
    with pytest.raises(ProblemSpecError):
        resolve_physics(spec)


def test_scalar_b0_amplitude_rejects_negative_sign():
    spec = _mri_spec(B0=-0.025)
    spec["forcing"]["B0"] = -0.025
    with pytest.raises(ProblemSpecError, match="must be nonnegative"):
        resolve_physics(spec)


def test_explicit_b0_component_vector_preserves_signed_direction():
    spec = _mri_spec()
    spec["nondimensional_groups"].pop("B0")
    spec["forcing"]["B0"] = [0.0, 0.0, -0.025]
    assert pytest.approx(0.025) == resolve_physics(spec).B0


def test_plain_pcf_mhd_wall_convention():
    spec = {
        "geometry": "pcf",
        "physics": "mhd",
        "domain": {"x": [-1.0, 1.0], "y_period": 12.566, "z_period": 6.283},
        "nondimensional_groups": {
            "Re": 400.0,
            "Rm": 400.0,
            "Pm": 1.0,
            "nu": 0.0025,
            "eta_mag": 0.0025,
            "B0": 0.05,
        },
        "boundary_conditions": {
            "velocity": {"type": "no_slip"},
            "magnetic": {"type": "conducting"},
        },
        "forcing": {"B0": 0.05},
    }
    rp = resolve_physics(spec)
    assert rp.velocity_scale == "wall"
    assert pytest.approx(1.0) == rp.U0  # Re*nu/h = 400*0.0025/1
    assert rp.Re_h == pytest.approx(400.0)
    assert rp.Pm == pytest.approx(1.0)


def test_metadata_roundtrip():
    rp = resolve_physics(_mri_spec())
    meta = rp.to_metadata()
    assert meta["nu"] == pytest.approx(0.001)
    assert meta["Re_h"] == pytest.approx(1000.0)
    assert meta["reynolds_convention"].startswith("Re")


def _tc_mri_spec():
    return json.loads(
        (ROOT / "production" / "runs" / "exp_tc_mri_vector_potential.json").read_text()
    )


def test_tc_resolves_native_and_midpoint_local_controls_from_one_coefficient():
    rp = resolve_physics(_tc_mri_spec())
    expected_b = (rp.Omega1 - rp.Omega2) * rp.R1**2 * rp.R2**2 / (rp.R2**2 - rp.R1**2)
    expected_s_mid = 2.0 * expected_b / rp.r_mid**2
    expected_local_scale = abs(expected_s_mid) * rp.h**2

    assert rp.S_mid == pytest.approx(expected_s_mid)
    assert rp.Re_h == pytest.approx(expected_local_scale / rp.nu)
    assert rp.Rm_h == pytest.approx(expected_local_scale / rp.eta)
    assert rp.Re_TC == pytest.approx(1000.0)
    assert rp.Rm_TC == pytest.approx(1000.0)
    assert rp.curvature == pytest.approx(1.0 / 3.0)
    assert rp.q_mid == pytest.approx(rp.S_mid / rp.Omega_mid)
    assert rp.Ly == pytest.approx(rp.theta_period * rp.r_mid)


def test_tc_local_reynolds_input_derives_the_consumed_nu():
    spec = _tc_mri_spec()
    groups = spec["nondimensional_groups"]
    for key in ("nu", "Re", "Re_TC", "Pm"):
        groups.pop(key, None)
    groups["Re_h"] = 400.0

    rp = resolve_physics(spec)
    assert rp.nu == pytest.approx(abs(rp.S_mid) * rp.h**2 / 400.0)
    assert rp.Re_h == pytest.approx(400.0)
    assert rp.Re_TC == pytest.approx(abs(rp.Omega1) * rp.R1 * rp.gap / rp.nu)


def test_tc_rejects_disagreement_between_native_and_local_controls():
    spec = _tc_mri_spec()
    spec["nondimensional_groups"]["Re_h"] = 400.0
    with pytest.raises(ProblemSpecError, match="Re_h"):
        resolve_physics(spec)
