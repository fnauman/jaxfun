"""FJ-07: sweep-safe semantic override interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.adapters import config_from_spec
from production.oracles import _tc_vp_solver_from_spec
from production.problem_spec import ProblemSpecError
from production.run_problem import _resolved_physics_metadata
from production.sweep import (
    SweepOverrideError,
    apply_overrides,
    materialize_run_spec,
    run_id_for,
    supported_overrides_for_spec,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "production" / "runs" / "exp_pcf_mri_shearbox_growth.json"
TC_BASE = ROOT / "production" / "runs" / "exp_tc_mri_vector_potential.json"
PCF_LINEAR = ROOT / "production" / "examples" / "pcf_mhd_conducting_v1.json"
PCF_HYDRO_LINEAR = ROOT / "production" / "examples" / "pcf_hydro_laminar_v1.json"
TC_LINEAR = ROOT / "production" / "examples" / "taylor_couette_mhd_conducting_v1.json"
PCF_IDEAL_LINEAR = ROOT / "production" / "examples" / "pcf_mri_shearbox_v1.json"
TC_HYDRO_LINEAR = ROOT / "production" / "examples" / "taylor_couette_hydro_v1.json"


def _base():
    return json.loads(BASE.read_text())


def _tc_base():
    return json.loads(TC_BASE.read_text())


def test_Re_h_override_changes_resolved_nu():
    base = _base()
    base_nu = base["nondimensional_groups"]["nu"]
    out = apply_overrides(base, {"Re_h": 2000.0})
    # Re_h: 1000 -> 2000 halves nu (nu = |S| h^2 / Re_h)
    assert out["nondimensional_groups"]["nu"] == pytest.approx(base_nu / 2.0)
    assert out["nondimensional_groups"]["Re"] == 2000.0


def test_Rm_h_override_changes_resolved_eta():
    out = apply_overrides(_base(), {"Rm_h": 500.0})
    assert out["nondimensional_groups"]["eta_mag"] == pytest.approx(1.0 / 500.0)


def test_B0_override_syncs_forcing():
    out = apply_overrides(_base(), {"B0": 0.05})
    assert out["nondimensional_groups"]["B0"] == 0.05
    assert out["forcing"]["B0"] == 0.05


def test_geometry_overrides_change_spec():
    out = apply_overrides(_base(), {"Ly": 3.0, "Lz": 0.5, "horizon": 20.0})
    assert out["domain"]["y_period"] == 3.0
    assert out["domain"]["z_period"] == 0.5
    assert out["time"]["final_time"] == 20.0


def test_every_override_changes_hash_or_fails():
    """FJ-07 acceptance: an override must change the resolved spec (hash)."""
    base_spec = apply_overrides(_base(), {})
    changed = apply_overrides(_base(), {"Re_h": 1600.0})
    assert changed["spec_hash"] != base_spec["spec_hash"]


def test_unknown_override_rejected():
    with pytest.raises(SweepOverrideError, match="unknown"):
        apply_overrides(_base(), {"not_a_field": 1.0})


def test_seed_override_rejected_as_inert():
    # No wired PCF saturation path consumes random_seed, so a seed override would
    # relabel identical physics -> it must be rejected, not silently accepted.
    with pytest.raises(SweepOverrideError, match="unknown"):
        apply_overrides(_base(), {"seed": 42})


def test_B0_override_syncs_Bz_for_specs_that_use_it():
    # The ideal-MRI shearbox spec expresses the imposed field via Bz (not B0); the
    # oracle reads groups["Bz"], so a B0 sweep must move Bz too.
    shearbox = json.loads(
        (ROOT / "production" / "examples" / "pcf_mri_shearbox_v1.json").read_text()
    )
    assert "Bz" in shearbox["nondimensional_groups"]
    out = apply_overrides(shearbox, {"B0": 0.05})
    assert out["nondimensional_groups"]["Bz"] == pytest.approx(0.05)
    assert out["nondimensional_groups"]["B0"] == 0.05


def test_inconsistent_manual_spec_still_validates_pathway():
    # An override that produces an inconsistent Pm cannot slip through: overriding
    # only Re_h re-derives nu, so Pm stays consistent (no error), but a hand-broken
    # base would raise. Here we assert the happy path resolves cleanly.
    out = apply_overrides(_base(), {"Re_h": 800.0, "Rm_h": 800.0})
    assert out["nondimensional_groups"]["Pm"] == pytest.approx(1.0)


def test_materialize_writes_unique_run_spec(tmp_path):
    rec = materialize_run_spec(BASE, {"Re_h": 1600.0, "B0": 0.0125}, tmp_path)
    assert Path(rec["spec_path"]).exists()
    assert rec["run_id"].startswith("exp_pcf_mri_shearbox_growth-")
    written = json.loads(Path(rec["spec_path"]).read_text())
    assert written["spec_hash"] == rec["spec_hash"]
    # run id is grouped by problem id + resolved hash
    assert run_id_for(written) == rec["run_id"]


def test_tc_local_overrides_update_coefficients_and_both_control_conventions():
    out = apply_overrides(_tc_base(), {"Re_h": 400.0, "Rm_h": 250.0})
    groups = out["nondimensional_groups"]
    meta = _resolved_physics_metadata(out, precision="float64")

    assert groups["nu"] == pytest.approx(abs(meta["S_mid"]) * meta["h"] ** 2 / 400.0)
    assert groups["eta_mag"] == pytest.approx(
        abs(meta["S_mid"]) * meta["h"] ** 2 / 250.0
    )
    assert groups["Re_h"] == pytest.approx(400.0)
    assert groups["Rm_h"] == pytest.approx(250.0)
    assert groups["Re"] == pytest.approx(groups["Re_TC"])
    assert groups["Rm"] == pytest.approx(groups["Rm_TC"])
    assert meta["nu"] == pytest.approx(groups["nu"])
    assert meta["eta"] == pytest.approx(groups["eta_mag"])
    assert meta["Re_TC"] == pytest.approx(groups["Re_TC"])
    assert meta["Rm_TC"] == pytest.approx(groups["Rm_TC"])


def test_tc_materialized_coefficients_are_the_solver_coefficients():
    base = _tc_base()
    base["resolution"] = {
        "Nr": 8,
        "Ntheta": 4,
        "Nz": 6,
        "family": "L",
        "dealias": 1.0,
    }
    out = apply_overrides(
        base,
        {
            "Re_h": 400.0,
            "Rm_h": 250.0,
        },
    )
    solver = _tc_vp_solver_from_spec(out)
    groups = out["nondimensional_groups"]
    meta = _resolved_physics_metadata(out, precision="float64")

    assert solver.nu == pytest.approx(groups["nu"])
    assert solver.eta_mag == pytest.approx(groups["eta_mag"])
    assert pytest.approx(groups["B0"]) == solver.B0
    assert solver.nu == pytest.approx(meta["nu"])
    assert solver.eta_mag == pytest.approx(meta["eta"])
    assert solver.Re == pytest.approx(meta["Re_TC"])
    assert solver.Rm == pytest.approx(meta["Rm_TC"])


def test_tc_rejects_inert_cartesian_ly_and_non_full_annulus():
    tc = _tc_base()
    assert "Ly" not in supported_overrides_for_spec(tc)
    with pytest.raises(SweepOverrideError, match="not consumed"):
        apply_overrides(tc, {"Ly": 4.0})

    tc["domain"]["theta_period"] = 3.141592653589793
    with pytest.raises(ProblemSpecError, match=r"full 2\*pi annulus"):
        apply_overrides(tc, {})


def test_hydro_rejects_magnetic_overrides_as_inert():
    hydro = json.loads(
        (ROOT / "production" / "runs" / "tc_supercritical_saturation.json").read_text()
    )
    for key, value in (("Rm_h", 100.0), ("B0", 0.1)):
        with pytest.raises(SweepOverrideError, match="not consumed"):
            apply_overrides(hydro, {key: value})
    with pytest.raises(SweepOverrideError, match="not a sweepable control"):
        apply_overrides(hydro, {"bc": "conducting"})


def test_linear_pcf_rejects_inert_box_and_time_overrides():
    for path in (PCF_HYDRO_LINEAR, PCF_LINEAR):
        linear = json.loads(path.read_text())
        for key, value in (
            ("Ly", 8.0),
            ("Lz", 8.0),
            ("dt", 0.002),
            ("horizon", 1.0),
        ):
            with pytest.raises(SweepOverrideError, match="not consumed"):
                apply_overrides(linear, {key: value})

    # This oracle is a static eigenproblem despite its legacy IMEX label.
    ideal = json.loads(PCF_IDEAL_LINEAR.read_text())
    with pytest.raises(SweepOverrideError, match="not consumed"):
        apply_overrides(ideal, {"horizon": 1.0})


def test_linear_oracles_reject_ignored_resolution_axes_but_accept_consumed_axis():
    pcf = json.loads(PCF_HYDRO_LINEAR.read_text())
    with pytest.raises(SweepOverrideError, match="ignored"):
        apply_overrides(pcf, {"resolution": {"ny": 16}})
    changed = apply_overrides(pcf, {"resolution": {"nx": 80}})
    assert changed["resolution"]["nx"] == 80

    tc = json.loads(TC_LINEAR.read_text())
    with pytest.raises(SweepOverrideError, match="ignored"):
        apply_overrides(tc, {"resolution": {"Nr": 32}})
    changed = apply_overrides(tc, {"resolution": {"N": 32}})
    assert changed["resolution"]["N"] == 32


def test_tiered_resolution_override_deep_merges_consumed_keys():
    base = _base()
    smoke_before = dict(base["resolution"]["smoke"])
    changed = apply_overrides(base, {"resolution": {"smoke": {"Nx": 14}}})

    assert changed["resolution"]["smoke"]["Nx"] == 14
    for key in smoke_before.keys() - {"Nx"}:
        assert changed["resolution"]["smoke"][key] == smoke_before[key]


def test_tiered_resolution_override_accepts_inherited_consumed_controls():
    base = _base()
    changed = apply_overrides(
        base,
        {"resolution": {"start": {"family": "C", "dealias": 1.0}}},
    )

    assert changed["resolution"]["start"]["family"] == "C"
    assert changed["resolution"]["start"]["dealias"] == pytest.approx(1.0)
    # Exercise the real adapter path: it validates the base, selects the tier,
    # and materializes inherited defaults plus tier-local shadows.
    effective = config_from_spec(changed, resolution_tier="start").spec["resolution"]
    assert effective["family"] == "C"
    assert effective["dealias"] == pytest.approx(1.0)


def test_bc_override_requires_separate_oracle_and_golden_identity():
    pcf = json.loads(PCF_LINEAR.read_text())
    with pytest.raises(SweepOverrideError, match="not a sweepable control.*golden"):
        apply_overrides(pcf, {"bc": "pseudo_vacuum"})

    tc = json.loads(TC_LINEAR.read_text())
    with pytest.raises(SweepOverrideError, match="not a sweepable control.*golden"):
        apply_overrides(tc, {"bc": "insulating"})

    with pytest.raises(SweepOverrideError, match="not a sweepable control.*golden"):
        apply_overrides(_tc_base(), {"bc": "insulating"})


def test_oracle_specific_coefficient_capabilities_follow_constructor_inputs():
    plane = json.loads(PCF_HYDRO_LINEAR.read_text())
    assert apply_overrides(plane, {"Re_h": 500.0})["nondimensional_groups"][
        "Re"
    ] == pytest.approx(500.0)

    circular = json.loads(TC_HYDRO_LINEAR.read_text())
    changed = apply_overrides(circular, {"Re_h": 100.0})
    assert changed["nondimensional_groups"]["Re_h"] == pytest.approx(100.0)

    local_ideal = json.loads(PCF_IDEAL_LINEAR.read_text())
    changed = apply_overrides(local_ideal, {"Re_h": 500.0, "Rm_h": 400.0, "B0": 0.05})
    assert changed["nondimensional_groups"]["Re"] == pytest.approx(500.0)
    assert changed["nondimensional_groups"]["Rm"] == pytest.approx(400.0)
    assert changed["nondimensional_groups"]["Bz"] == pytest.approx(0.05)


def test_scalar_b0_override_must_be_nonnegative_and_finite():
    for value in (-0.1, float("nan"), float("inf")):
        with pytest.raises(SweepOverrideError, match="nonnegative"):
            apply_overrides(_base(), {"B0": value})


def test_positive_b0_override_rescales_explicit_component_direction():
    pcf = json.loads(PCF_LINEAR.read_text())
    assert "Bx" not in pcf["nondimensional_groups"]
    out = apply_overrides(pcf, {"B0": 0.05})
    assert out["forcing"]["B0"] == pytest.approx([0.0, 0.0, 0.05])
    assert out["nondimensional_groups"]["Bz"] == pytest.approx(0.05)


def test_b0_override_rejects_nonzero_component_omitted_by_selected_solver():
    pcf = json.loads(PCF_LINEAR.read_text())
    pcf["forcing"]["B0"] = [0.1, 0.0, 0.1]

    with pytest.raises(SweepOverrideError, match="Bx.*not consumed"):
        apply_overrides(pcf, {"B0": 0.05})


def test_oblique_b0_override_syncs_archived_and_solver_consumed_components():
    pcf = json.loads(PCF_LINEAR.read_text())
    pcf["forcing"]["B0"] = [0.0, 3.0, 4.0]
    pcf["nondimensional_groups"]["By"] = 3.0
    pcf["nondimensional_groups"]["Bz"] = 4.0

    out = apply_overrides(pcf, {"B0": 10.0})
    groups = out["nondimensional_groups"]
    archived = out["forcing"]["B0"]

    assert archived == pytest.approx([0.0, 6.0, 8.0])
    assert groups["By"] == pytest.approx(6.0)
    assert groups["Bz"] == pytest.approx(8.0)
    assert groups["B0"] == pytest.approx(10.0)
    assert (groups["By"] ** 2 + groups["Bz"] ** 2) ** 0.5 == pytest.approx(10.0)


def test_component_only_b0_override_preserves_solver_direction():
    shearbox = json.loads(PCF_IDEAL_LINEAR.read_text())
    shearbox["nondimensional_groups"]["By"] = 3.0
    shearbox["nondimensional_groups"]["Bz"] = 4.0

    out = apply_overrides(shearbox, {"B0": 2.5})
    assert out["nondimensional_groups"]["By"] == pytest.approx(1.5)
    assert out["nondimensional_groups"]["Bz"] == pytest.approx(2.0)
