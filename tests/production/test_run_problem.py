import json
import subprocess
import sys
from pathlib import Path

import pytest

from production.compare_goldens import validate_golden
from production.oracles import _channel_poiseuille_kmm_state
from production.problem_spec import UnsupportedSpecError
from production.run_problem import (
    SolverExecutionNotImplementedError,
    _assert_required_saturation_checks,
    _saturation_check_metadata,
    main,
    run_problem,
)

ROOT = Path(__file__).resolve().parents[2]


def test_validate_only_writes_metadata_without_claiming_solver_execution(tmp_path):
    out = tmp_path / "run"
    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
        out=out,
        validate_only=True,
        capture_device=False,
    )
    written = json.loads((out / "metadata.json").read_text())
    assert written["problem_id"] == "tc_supercritical_saturation"
    assert written["execution"] == {
        "solver_execution_wired": False,
        "status": "validated",
    }
    assert metadata["adapter"]["axis_conventions"]["axis_0"] == "r radial"


def test_resolution_tier_validate_only_materializes_effective_spec(tmp_path):
    out = tmp_path / "run"
    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "tc_mri_nonlinear_saturation.json",
        out=out,
        resolution_tier="start",
        validate_only=True,
        capture_device=False,
    )

    assert metadata["run_options"]["resolution_tier"] == "start"
    assert metadata["adapter"]["resolution_tier"] == "start"
    assert metadata["adapter"]["effective_resolution"] == {
        "Nr": 40,
        "Nz": 24,
        "dealias": 1.5,
        "family": "C",
    }
    assert metadata["adapter"]["solver_args"]["resolution"]["Nr"] == 40
    assert metadata["base_spec_hash"] != metadata["spec_hash"]


def test_smoke_resolution_tier_materializes_lightweight_pcf_spec(tmp_path):
    out = tmp_path / "run"
    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "pcf_mhd_divfree.json",
        out=out,
        resolution_tier="smoke",
        validate_only=True,
        capture_device=False,
    )

    assert metadata["run_options"]["resolution_tier"] == "smoke"
    assert metadata["adapter"]["resolution_tier"] == "smoke"
    assert metadata["adapter"]["effective_resolution"] == {
        "Nx": 8,
        "Ny": 4,
        "Nz": 4,
        "dealias": [1.0, 1.0, 1.0],
        "family": "C",
    }


def test_cli_accepts_smoke_resolution_tier_for_validate_only(tmp_path):
    out = tmp_path / "cli-smoke"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "production.run_problem",
            "--config",
            str(ROOT / "production" / "runs" / "pcf_mhd_divfree.json"),
            "--out",
            str(out),
            "--resolution-tier",
            "smoke",
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["run_options"]["resolution_tier"] == "smoke"
    assert metadata["adapter"]["effective_resolution"]["Nx"] == 8


def test_full_saturation_check_failure_marks_metadata_failed(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    metadata = {
        "out_dir": str(out),
        "execution": {
            "status": "completed",
            "solver_execution_wired": True,
            "execution_kind": "dns-saturation",
        },
        "saturation_checks": {
            "required": True,
            "passed": False,
            "magnetic_energy_growth_factor": 1.25,
        },
    }

    with pytest.raises(RuntimeError, match="full saturation check failed"):
        _assert_required_saturation_checks(metadata)

    written = json.loads((out / "metadata.json").read_text())
    assert written["execution"]["status"] == "failed"
    assert (
        "magnetic_energy_growth_factor=1.25" in written["execution"]["failure_reason"]
    )


def test_bounded_smoke_saturation_check_is_not_required():
    metadata = _saturation_check_metadata(
        {"scalars": {"saturation_check_passed": False}},
        validation_scope={"kind": "bounded_saturation_smoke"},
    )

    assert metadata == {
        "required": False,
        "passed": False,
        "energy_growth_factor": None,
        "magnetic_energy_growth_factor": None,
    }


def test_pipe_spec_rejected_before_output_directory_is_created(tmp_path):
    out = tmp_path / "pipe"
    with pytest.raises(UnsupportedSpecError):
        run_problem(
            config_path=ROOT
            / "production"
            / "examples"
            / "pipe_hagen_poiseuille_v1.json",
            out=out,
            validate_only=True,
            capture_device=False,
        )
    assert not out.exists()


def test_solver_exception_marks_metadata_failed(tmp_path, monkeypatch):
    def fail_solver(*args, **kwargs):
        raise RuntimeError("synthetic solver crash")

    monkeypatch.setattr("production.run_problem.run_supported_spec", fail_solver)
    out = tmp_path / "run"

    with pytest.raises(RuntimeError, match="synthetic solver crash"):
        run_problem(
            config_path=ROOT
            / "production"
            / "examples"
            / "channel_poiseuille_hydro_v1.json",
            out=out,
            capture_device=False,
        )

    written = json.loads((out / "metadata.json").read_text())
    assert written["execution"] == {
        "status": "failed",
        "solver_execution_wired": True,
        "execution_kind": "analytic-oracle",
        "failure_reason": "RuntimeError: synthetic solver crash",
    }
    assert written["timing"]["solver_wall_time_seconds"] >= 0.0


def test_non_validate_run_fails_explicitly_for_unwired_oracle(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    spec["problem_id"] = "pcf_mhd_unwired_smoke"
    spec["expected_oracle"] = {
        **spec["expected_oracle"],
        "type": "unwired_test_oracle",
    }
    spec["golden"] = {**spec["golden"], "artifact_id": spec["problem_id"]}
    spec_path = tmp_path / "pcf_mhd_unwired_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(
        SolverExecutionNotImplementedError, match="solver execution is not wired"
    ):
        run_problem(
            config_path=spec_path,
            out=tmp_path / "run",
            capture_device=False,
        )


def test_pcf_fluct_re400_smoke_runs_from_phase_j5_spec(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_fluct_re400.json").read_text()
    )
    spec["resolution"] = {
        **spec["resolution"],
        "start": {"Nx": 9, "Ny": 4, "Nz": 4},
        "dealias": [1.0, 1.0, 1.0],
    }
    spec["time"] = {**spec["time"], "dt": 0.001, "final_time": 0.002}
    spec_path = tmp_path / "pcf_fluct_re400_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "pcf_fluct_re400"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        resolution_tier="start",
        capture_device=False,
    )

    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-saturation",
    }
    assert metadata["run_options"]["resolution_tier"] == "start"
    assert metadata["adapter"]["effective_resolution"]["Nx"] == 9
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["kinetic_energy"] > 0.0
    assert final["total_kinetic_energy"] > final["kinetic_energy"]
    assert final["divergence_l2"] < 1.0e-4
    assert "wall_shear_lower" in final
    assert "wall_shear_upper" in final
    assert "streak_rms" in final
    assert "roll_rms" in final
    assert metadata["diagnostics_path"] == str(out / "diagnostics.jsonl")


def test_pcf_mhd_divfree_smoke_runs_from_phase_j5_spec(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    spec["resolution"] = {
        **spec["resolution"],
        "start": {"Nx": 8, "Ny": 4, "Nz": 4},
        "dealias": [1.0, 1.0, 1.0],
        "family": "C",
    }
    spec["time"] = {**spec["time"], "dt": 0.001, "final_time": 0.002}
    spec_path = tmp_path / "pcf_mhd_divfree_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "pcf_mhd_divfree"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        resolution_tier="start",
        capture_device=False,
    )

    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-saturation",
    }
    assert metadata["run_options"]["resolution_tier"] == "start"
    assert metadata["adapter"]["effective_resolution"]["Nx"] == 8
    assert metadata["validation_scope"]["kind"] == "bounded_saturation_smoke"
    assert metadata["validation_scope"]["bounded_smoke"] is True
    assert metadata["validation_scope"]["steps_override"] == 2
    assert metadata["validation_scope"]["resolution_tier"] == "start"
    assert (
        "not a full production saturation golden"
        in metadata["validation_scope"]["reason"]
    )
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["kinetic_energy"] > 0.0
    assert final["magnetic_energy"] > 0.0
    assert final["total_energy"] > final["magnetic_energy"]
    assert final["divergence_u_l2"] < 1.0e-4
    assert final["divergence_b_l2"] < 1.0e-4
    assert "maxwell_stress_xy" in final
    assert "reynolds_stress" in final
    assert metadata["diagnostics_path"] == str(out / "diagnostics.jsonl")


def test_exp_pcf_mri_shearbox_growth_smoke_runs_from_phase_j5_spec(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "exp_pcf_mri_shearbox_growth.json").read_text()
    )
    spec["resolution"] = {
        **spec["resolution"],
        "start": {"Nx": 8, "Ny": 4, "Nz": 6},
        "dealias": [1.0, 1.0, 1.0],
        "family": "L",
    }
    spec["initial_condition"] = {
        **spec["initial_condition"],
        "seeded_modes": {"ky": 0, "kz": [1, 2]},
    }
    spec["time"] = {**spec["time"], "dt": 0.001, "final_time": 0.002}
    spec_path = tmp_path / "exp_pcf_mri_shearbox_growth_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "exp_pcf_mri_shearbox_growth"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        resolution_tier="start",
        capture_device=False,
    )

    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-saturation",
    }
    assert metadata["run_options"]["resolution_tier"] == "start"
    assert metadata["adapter"]["effective_resolution"]["Nx"] == 8
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["magnetic_energy"] > 0.0
    assert final["divergence_b_l2"] < 1.0e-6
    assert "maxwell_stress_xy" in final
    assert "transport_alpha" in final
    assert "butterfly_by_mean" in final
    assert metadata["diagnostics_path"] == str(out / "diagnostics.jsonl")


def test_channel_driven_kmm_state_recovers_poiseuille_profile():
    spec = json.loads(
        (
            ROOT / "production" / "examples" / "channel_poiseuille_hydro_v1.json"
        ).read_text()
    )

    record = _channel_poiseuille_kmm_state(spec)

    assert record["solver"].N == (64, 8, 8)
    assert record["solver"].dpdy == pytest.approx(-0.002)
    assert record["pressure_gradient"] == pytest.approx(-0.002)
    assert record["profile_linf"] < 2.0e-5
    assert record["divergence_l2"] < 2.0e-5


def test_channel_driven_kmm_run_writes_diagnostics_and_compares_golden(tmp_path):
    out = tmp_path / "channel"
    metadata = run_problem(
        config_path=ROOT
        / "production"
        / "examples"
        / "channel_poiseuille_hydro_v1.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"]["status"] == "completed"
    assert metadata["execution"]["solver_execution_wired"] is True
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == [
        "divergence_l2",
        "flow_rate",
        "kinetic_energy",
        "pressure_gradient",
    ]
    assert metadata["timing"]["solver_wall_time_seconds"] >= 0.0
    assert metadata["timing"]["solver_started_at_utc"]
    assert metadata["timing"]["solver_finished_at_utc"]
    assert (out / "spec.json").exists()
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["pressure_gradient"] == pytest.approx(-0.002)


def test_pcf_hydro_laminar_run_writes_diagnostics_and_compares_golden(tmp_path):
    out = tmp_path / "pcf"
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / "pcf_hydro_laminar_v1.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"]["status"] == "completed"
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == [
        "divergence_l2",
        "growth_rate",
        "kinetic_energy",
    ]
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["growth_rate"] == pytest.approx(-0.0034674010999505545)
    assert line["wall_shear_lower"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("problem_id", "expected_keys"),
    [
        (
            "pcf_mhd_conducting_v1",
            ["divergence_b_l2", "growth_rate", "magnetic_energy"],
        ),
        ("pcf_mri_shearbox_v1", ["divergence_b_l2", "growth_rate", "local_mri_growth"]),
    ],
)
def test_pcf_mhd_and_mri_linear_runs_compare_goldens(
    tmp_path, problem_id, expected_keys
):
    out = tmp_path / problem_id
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / f"{problem_id}.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"]["status"] == "completed"
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == expected_keys
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["magnetic_bc"] == "conducting"
    assert line["divergence_b_l2"] == 0.0


@pytest.mark.parametrize(
    "problem_id",
    [
        "taylor_couette_mhd_conducting_v1",
        "taylor_couette_mhd_insulating_v1",
    ],
)
def test_tc_mhd_linear_runs_compare_goldens(tmp_path, problem_id):
    out = tmp_path / problem_id
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / f"{problem_id}.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"]["status"] == "completed"
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == [
        "divergence_b_l2",
        "growth_rate",
        "magnetic_energy",
    ]
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["magnetic_bc"] in {"conducting", "insulating"}
    assert line["divergence_b_l2"] == 0.0


def test_tc_hydro_linear_run_writes_diagnostics_and_compares_golden(tmp_path):
    out = tmp_path / "tc"
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / "taylor_couette_hydro_v1.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"]["status"] == "completed"
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == [
        "divergence_l2",
        "growth_rate",
        "kinetic_energy",
    ]
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["growth_rate"] == pytest.approx(0.371383777641364)
    assert line["rayleigh_stable"] is False


def test_tc_supercritical_saturation_smoke_runs_from_phase_j5_spec(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "tc_supercritical_saturation.json").read_text()
    )
    spec["resolution"] = {
        **spec["resolution"],
        "production": {"Nr": 10, "Nz": 6},
        "dealias": 1.0,
    }
    spec["time"] = {**spec["time"], "final_time": 0.008}
    spec_path = tmp_path / "tc_supercritical_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "tc_supercritical"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        resolution_tier="production",
        capture_device=False,
    )

    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-saturation",
    }
    assert metadata["run_options"]["resolution_tier"] == "production"
    assert metadata["adapter"]["effective_resolution"]["Nr"] == 10
    assert metadata["base_spec_hash"] != metadata["spec_hash"]
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["kinetic_energy"] > 0.0
    assert final["radial_velocity_linf"] > 0.0
    assert final["torque"] > 0.0
    assert final["divergence_l2"] >= 0.0
    assert metadata["diagnostics_path"] == str(out / "diagnostics.jsonl")


def test_tc_mri_nonlinear_saturation_smoke_runs_from_phase_j5_spec(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "tc_mri_nonlinear_saturation.json").read_text()
    )
    spec["resolution"] = {
        **spec["resolution"],
        "start": {"Nr": 10, "Nz": 6},
        "dealias": 1.0,
    }
    spec["time"] = {**spec["time"], "final_time": 0.004}
    spec_path = tmp_path / "tc_mri_saturation_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "tc_mri_saturation"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        resolution_tier="start",
        capture_device=False,
    )

    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-saturation",
    }
    assert metadata["run_options"]["resolution_tier"] == "start"
    assert metadata["adapter"]["effective_resolution"]["Nr"] == 10
    assert metadata["base_spec_hash"] != metadata["spec_hash"]
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["kinetic_energy"] > 0.0
    assert final["magnetic_energy"] > 0.0
    assert final["divergence_b_l2"] >= 0.0
    assert "maxwell_stress_xy" in final
    assert "reynolds_stress" in final
    assert metadata["diagnostics_path"] == str(out / "diagnostics.jsonl")


@pytest.mark.parametrize(
    ("problem_id", "expected_keys"),
    [
        (
            "pcf_hydro_primitive_dns_v1",
            [
                "divergence_u",
                "growth_rate",
                "growth_rate_linear",
                "kinetic_energy",
            ],
        ),
        (
            "pcf_mri_primitive_dns_v1",
            [
                "divergence_b",
                "divergence_u",
                "growth_rate",
                "growth_rate_linear",
                "kinetic_energy",
                "magnetic_energy",
            ],
        ),
    ],
)
def test_pcf_primitive_dns_runs_compare_dns_goldens(
    tmp_path, problem_id, expected_keys
):
    out = tmp_path / problem_id
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / f"{problem_id}.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-linear-window",
    }
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == expected_keys
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[-1]["growth_rate"] == pytest.approx(
        rows[0]["growth_rate_linear"], abs=5.0e-7
    )


@pytest.mark.parametrize(
    ("problem_id", "expected_keys"),
    [
        (
            "taylor_couette_hydro_dns_v1",
            [
                "divergence_linf",
                "growth_rate",
                "growth_rate_linear",
                "kinetic_energy",
            ],
        ),
        (
            "taylor_couette_mhd_dns_v1",
            [
                "divergence_b",
                "divergence_u",
                "growth_rate",
                "growth_rate_linear",
                "kinetic_energy",
                "magnetic_energy",
            ],
        ),
    ],
)
def test_tc_dns_runs_compare_dns_goldens(tmp_path, problem_id, expected_keys):
    out = tmp_path / problem_id
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / f"{problem_id}.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )
    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "dns-linear-window",
    }
    assert metadata["comparison_passed"] is True
    assert metadata["observables_compared"] == expected_keys
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["t"] == 0.0
    assert rows[-1]["growth_rate"] == pytest.approx(
        rows[0]["growth_rate_linear"], abs=2.0e-7
    )


def test_tc_dns_runner_checkpoint_restart_continues(tmp_path):
    pytest.importorskip("h5py")
    import math

    import jax.numpy as jnp

    from examples.taylor_couette_dns_jax import (
        AxisymmetricTCDNSJax,
        AxisymmetricTCState,
        CircularCouette,
    )
    from jaxfun.io import read_checkpoint

    spec = json.loads(
        (
            ROOT / "production" / "examples" / "taylor_couette_hydro_dns_v1.json"
        ).read_text()
    )
    spec["resolution"] = {**spec["resolution"], "Nr": 10, "Nz": 6}
    spec["time"] = {**spec["time"], "final_time": 0.004}
    spec_path = tmp_path / "tc_dns_checkpoint.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "run"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=4,
        checkpoint_every=2,
        capture_device=False,
    )
    checkpoint_path = out / "checkpoints" / "checkpoints.h5"
    assert metadata["checkpoint_path"] == str(checkpoint_path)

    record = read_checkpoint(checkpoint_path)
    assert record.tstep == 4
    assert record.attrs["problem_id"] == "taylor_couette_hydro_dns_v1"
    assert record.attrs["artifact_id"] == "taylor_couette_hydro_dns_v1"
    assert record.attrs["spec_hash"] == metadata["spec_hash"]
    assert record.attrs["checkpoint_schema_version"] == 1
    assert record.attrs["solver_schema_version"] == 1
    assert record.attrs["diagnostics_path"].endswith("diagnostics.jsonl")
    dtype_metadata = json.loads(record.attrs["dtype_metadata_json"])
    assert dtype_metadata["production_run_dtype"] == "float32"
    assert dtype_metadata["field_dtypes"]
    assert dtype_metadata["field_shapes"]
    device_metadata = json.loads(record.attrs["device_metadata_json"])
    assert device_metadata == {"capture_skipped": True}
    assert record.attrs["prng_state_json"] == ""
    payload = record.fields["state"]
    restarted = AxisymmetricTCState(
        u=tuple(payload["u"]),
        p=payload["p"],
        nonlinear_old=tuple(payload["nonlinear_old"]),
        have_old=payload["have_old"],
    )

    groups = spec["nondimensional_groups"]
    solver = AxisymmetricTCDNSJax(
        CircularCouette(groups["R1"], groups["R2"], groups["Omega1"], groups["Omega2"]),
        nu=groups["nu"],
        Nr=spec["resolution"]["Nr"],
        Nz=spec["resolution"]["Nz"],
        Lz=spec["domain"]["z_period"],
        dt=spec["time"]["dt"],
        family=spec["resolution"]["family"],
        dealias=1.0,
    )
    kz_mode = round(spec["mode"]["axial_wavenumber"] * solver.Lz / (2.0 * math.pi))
    state0, _ = solver.seed_linear_eigenmode(
        kz_mode=kz_mode, amp=spec["initial_condition"]["amplitude"]
    )
    direct = solver.solve(state0, 6)
    continued = solver.solve(restarted, 2)

    for got, expected in zip(continued.u, direct.u, strict=True):
        assert jnp.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(continued.p, direct.p, rtol=1.0e-12, atol=1.0e-12)


def test_channel_analytic_run_can_write_schema_v1_golden(tmp_path):
    out = tmp_path / "channel"
    run_problem(
        config_path=ROOT
        / "production"
        / "examples"
        / "channel_poiseuille_hydro_v1.json",
        out=out,
        write_golden=True,
        capture_device=False,
    )
    golden = validate_golden(out / "golden" / "golden.json")
    assert golden["schema_version"] == 1
    assert golden["problem_id"] == "channel_poiseuille_hydro_v1"


def test_cli_validate_only_returns_success(tmp_path):
    code = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "tc_supercritical_saturation.json"),
            "--out",
            str(tmp_path / "run"),
            "--validate-only",
        ]
    )
    assert code == 0


def test_cli_non_validate_returns_not_implemented_status(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    spec["problem_id"] = "pcf_mhd_cli_unwired_smoke"
    spec["expected_oracle"] = {
        **spec["expected_oracle"],
        "type": "unwired_test_oracle",
    }
    spec["golden"] = {**spec["golden"], "artifact_id": spec["problem_id"]}
    spec_path = tmp_path / "pcf_mhd_cli_unwired_smoke.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    code = main(
        [
            "--config",
            str(spec_path),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 2
