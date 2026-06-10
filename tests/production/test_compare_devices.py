import json
from pathlib import Path

from production.compare_devices import compare_final_diagnostics, main

ROOT = Path(__file__).resolve().parents[2]


def test_compare_final_diagnostics_reports_numeric_differences(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "diagnostics.jsonl").write_text(
        json.dumps({"t": 0.0, "energy": 1.0, "label": "left"}) + "\n",
        encoding="utf-8",
    )
    (right / "diagnostics.jsonl").write_text(
        json.dumps({"t": 0.0, "energy": 1.01, "label": "right"}) + "\n",
        encoding="utf-8",
    )

    comparisons = compare_final_diagnostics(left, right, atol=1.0e-4, rtol=1.0e-4)

    assert len(comparisons) == 1
    assert comparisons[0].key == "energy"
    assert comparisons[0].passed is False
    assert "tolerance" in comparisons[0].message


def test_compare_devices_cli_runs_cpu_cpu_channel_smoke(tmp_path):
    out = tmp_path / "compare"
    rc = main(
        [
            "--config",
            str(ROOT / "production" / "examples" / "channel_poiseuille_hydro_v1.json"),
            "--out",
            str(out),
            "--device-a",
            "cpu",
            "--device-b",
            "cpu",
            "--timeout-seconds",
            "1800",
        ]
    )

    assert rc == 0
    report = json.loads((out / "device_comparison.json").read_text(encoding="utf-8"))
    assert report["summary"]["failed"] == 0
    assert report["runs"]["left"]["returncode"] == 0
    assert report["runs"]["right"]["returncode"] == 0
    assert {item["key"] for item in report["comparisons"]} >= {
        "flow_rate",
        "kinetic_energy",
        "pressure_gradient",
    }


def test_compare_devices_cli_accepts_production_resolution_tier(tmp_path):
    out = tmp_path / "compare"
    rc = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "pcf_mhd_divfree.json"),
            "--out",
            str(out),
            "--device-a",
            "cpu",
            "--device-b",
            "cpu",
            "--resolution-tier",
            "smoke",
            "--steps",
            "2",
            "--timeout-seconds",
            "1800",
            "--atol",
            "1e-6",
            "--rtol",
            "1e-5",
        ]
    )

    assert rc == 0
    report = json.loads((out / "device_comparison.json").read_text(encoding="utf-8"))
    assert report["run_options"] == {"steps": 2, "resolution_tier": "smoke"}
    for side in ("left", "right"):
        run_dir = Path(report["runs"][side]["out_dir"])
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["run_options"]["resolution_tier"] == "smoke"
        assert (
            metadata["validation_scope"]["kind"]
            == "cpu_smoke_finiteness_divergence_only"
        )
        assert metadata["adapter"]["effective_resolution"] == {
            "Nx": 8,
            "Ny": 4,
            "Nz": 4,
            "dealias": [1.0, 1.0, 1.0],
            "family": "C",
        }
    assert {item["key"] for item in report["comparisons"]} >= {
        "divergence_u_l2",
        "divergence_b_l2",
        "magnetic_energy",
    }
