"""FJ-00: single resolved-physics contract tests."""

from __future__ import annotations

import copy

import pytest

from production.physics import resolve_physics
from production.problem_spec import ProblemSpecError


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
    assert rp.B0 == pytest.approx(0.025)
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
    assert rp.U0 == pytest.approx(1.0)  # Re*nu/h = 400*0.0025/1
    assert rp.Re_h == pytest.approx(400.0)
    assert rp.Pm == pytest.approx(1.0)


def test_metadata_roundtrip():
    rp = resolve_physics(_mri_spec())
    meta = rp.to_metadata()
    assert meta["nu"] == pytest.approx(0.001)
    assert meta["Re_h"] == pytest.approx(1000.0)
    assert meta["reynolds_convention"].startswith("Re")
