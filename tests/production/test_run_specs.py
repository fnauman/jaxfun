import json
from pathlib import Path

from production.problem_spec import load_spec

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "production" / "runs"
GOLDENS = ROOT / "production" / "goldens"

EXECUTABLE_RUNS = {
    "pcf_fluct_re400": [3],
    "pcf_mhd_divfree": [3],
    "exp_pcf_mri_shearbox_growth": [1, 2, 3],
    "exp_pcf_mri_pseudo_vacuum": [1, 2, 3],
    "exp_pcf_mri_vector_potential": [1, 2, 3],
    "exp_pcf_mri_vp_insulating": [1, 2, 3],
    "exp_tc_mri_vector_potential": [1, 2, 3],
    "exp_tc_mri_vp_insulating": [1, 2, 3],
    "tc_supercritical_saturation": [2, 3],
    "tc_mri_nonlinear_saturation": [1, 2, 3],
}


def test_phase_j5_executable_inventory_specs_exist_and_validate():
    got = {path.stem for path in RUNS.glob("*.json")}
    assert got == set(EXECUTABLE_RUNS)
    for problem_id in sorted(EXECUTABLE_RUNS):
        spec = load_spec(RUNS / f"{problem_id}.json")
        assert spec["problem_id"] == problem_id
        assert spec["support_state"] == "experimental"
        assert spec["golden"]["artifact_id"] == problem_id
        assert spec["expected_oracle"]["fallback_rungs"] == EXECUTABLE_RUNS[problem_id]


def test_stab_pcf_mri_stability_is_not_an_executable_run_spec():
    assert not (RUNS / "stab_PCF_MRI_stability.json").exists()


def test_run_specs_carry_smoke_start_and_production_resolution():
    for path in RUNS.glob("*.json"):
        raw = json.loads(path.read_text())
        assert "smoke" in raw["resolution"]
        assert "start" in raw["resolution"]
        assert "production" in raw["resolution"]
        assert raw["time"]["final_time"] > 0.0


def test_pcf_mhd_production_resolution_matches_phase_j5_inventory():
    raw = json.loads((RUNS / "pcf_mhd_divfree.json").read_text())

    assert raw["resolution"]["production"] == {"Nx": 32, "Ny": 64, "Nz": 32}
    assert raw["resolution"]["start"] == {"Nx": 16, "Ny": 32, "Nz": 16}


def test_retained_saturation_goldens_track_run_spec_hashes():
    # pcf_mhd_divfree and exp_pcf_mri_shearbox_growth were migrated to the FJ-01
    # numerics contract (v2) and their pre-fix goldens are quarantined, so their
    # run-spec hash intentionally no longer matches the (stale) golden.
    for problem_id in [
        "pcf_fluct_re400",
        "tc_supercritical_saturation",
        "tc_mri_nonlinear_saturation",
    ]:
        spec = load_spec(RUNS / f"{problem_id}.json")
        golden = json.loads(
            (GOLDENS / problem_id / "golden" / "golden.json").read_text()
        )

        assert golden["problem_id"] == spec["problem_id"]
        assert golden["spec_hash"] == spec["spec_hash"]


def test_migrated_primitive_goldens_are_quarantined_pending_regeneration():
    """FJ-01/FJ-03: pre-fix primitive goldens must be quarantined and not tracked."""
    for problem_id in ["pcf_mhd_divfree", "exp_pcf_mri_shearbox_growth"]:
        spec = load_spec(RUNS / f"{problem_id}.json")
        assert spec["numerics_contract_version"] == 2
        golden = json.loads(
            (GOLDENS / problem_id / "golden" / "golden.json").read_text()
        )
        quarantine = golden.get("quarantined")
        assert quarantine and quarantine["forbidden_from_seeding_production"] is True
        # The stale golden must NOT accidentally match the migrated run spec.
        assert golden["spec_hash"] != spec["spec_hash"]


def test_run_specs_smoke_resolution_is_smaller_than_start():
    for path in RUNS.glob("*.json"):
        raw = json.loads(path.read_text())
        smoke = raw["resolution"]["smoke"]
        start = raw["resolution"]["start"]
        for key, smoke_value in smoke.items():
            if key in start and isinstance(smoke_value, int):
                assert smoke_value <= start[key], (path.name, key)
