import json
from pathlib import Path

from production.problem_spec import load_spec


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "production" / "runs"

EXECUTABLE_RUNS = {
    "pcf_fluct_re400": [3],
    "pcf_mhd_divfree": [3],
    "exp_pcf_mri_shearbox_growth": [1, 2, 3],
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


def test_run_specs_carry_start_and_production_resolution():
    for path in RUNS.glob("*.json"):
        raw = json.loads(path.read_text())
        assert "start" in raw["resolution"]
        assert "production" in raw["resolution"]
        assert raw["time"]["final_time"] > 0.0
