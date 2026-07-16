import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from production.adapters import load_config
from production.compare_goldens import validate_golden
from production.oracles import (
    _channel_poiseuille_kmm_state,
    _saturation_passed,
    run_supported_spec,
)
from production.problem_spec import UnsupportedSpecError
from production.quench import QuenchError, resolve_fixed_quench_duration
from production.run_problem import (
    _assert_required_saturation_checks,
    _assert_validation_floor_checks,
    _classification_metadata,
    _effective_diagnostics_every,
    _resolve_resume_or_quench,
    _saturation_check_metadata,
    _validation_floor_metadata,
    _validation_scope_metadata,
    _write_golden,
    main,
    run_problem,
)

ROOT = Path(__file__).resolve().parents[2]


def _install_fake_quench_solver(monkeypatch, solve):
    import production.run_problem as runner

    config_path = ROOT / "production/runs/exp_pcf_mri_vector_potential.json"
    config = load_config(config_path)
    dt = float(config.spec["time"]["dt"])
    duration = resolve_fixed_quench_duration(
        additional_time=None,
        additional_steps=4,
        child_dt=dt,
        parent_time=10 * dt,
        parent_step=10,
    )
    record = SimpleNamespace(t=duration.parent_time, tstep=duration.parent_step)
    quench_metadata = {"mode": "quench", "duration": duration.to_metadata()}
    monkeypatch.setattr(
        runner,
        "_resolve_resume_or_quench",
        lambda *_args, **_kwargs: (record, quench_metadata),
    )
    monkeypatch.setattr(
        runner, "validate_resume_checkpoint", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(runner, "run_supported_spec", solve)
    return config_path, duration


def _run_fake_quench(config_path, out):
    return run_problem(
        config_path=config_path,
        out=out,
        quench_from="unused-parent",
        additional_steps=4,
        capture_device=False,
    )


def test_final_time_only_quench_is_rejected_before_config_or_output(tmp_path):
    out = tmp_path / "must-not-exist"
    with pytest.raises(QuenchError, match="final_time alone"):
        run_problem(
            config_path=tmp_path / "missing-child.json",
            out=out,
            quench_from=tmp_path / "missing-parent",
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "options",
    [
        {"additional_steps": 2},
        {"quench_from": "parent", "steps": 4, "additional_steps": 2},
        {
            "quench_from": "parent",
            "additional_time": 0.1,
            "additional_steps": 2,
        },
    ],
)
def test_run_problem_rejects_ambiguous_duration_api_before_output(tmp_path, options):
    out = tmp_path / "must-not-exist"
    with pytest.raises(QuenchError):
        run_problem(
            config_path=tmp_path / "missing-child.json",
            out=out,
            **options,
        )
    assert not out.exists()


def test_cli_rejects_final_time_only_quench_before_output(tmp_path, capsys):
    out = tmp_path / "must-not-exist"
    rc = main(
        [
            "--config",
            str(tmp_path / "missing-child.json"),
            "--out",
            str(out),
            "--quench",
            str(tmp_path / "missing-parent"),
        ]
    )
    assert rc == 1
    assert "final_time alone" in capsys.readouterr().err
    assert not out.exists()


def test_adaptive_quench_reaches_parent_bank_validation(tmp_path):
    data = json.loads(
        (ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json").read_text(
            encoding="utf-8"
        )
    )
    data["time"]["adaptive_cfl"] = {}
    child = tmp_path / "adaptive-child.json"
    child.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "must-not-exist"

    with pytest.raises(QuenchError, match="checkpoint bank"):
        run_problem(
            config_path=child,
            out=out,
            quench_from=tmp_path / "missing-parent",
            additional_steps=2,
        )
    assert not out.exists()


def test_primitive_adaptive_quench_is_rejected_before_parent_or_output(tmp_path):
    data = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    data["time"]["adaptive_cfl"] = {}
    child = tmp_path / "primitive-adaptive-child.json"
    child.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "must-not-exist"

    with pytest.raises(QuenchError, match="only for vector-potential"):
        run_problem(
            config_path=child,
            out=out,
            quench_from=tmp_path / "missing-parent",
            additional_steps=2,
        )
    assert not out.exists()


def test_adaptive_burn_in_bound_is_rejected_during_quench_resolution(
    tmp_path, monkeypatch
):
    import production.run_problem as runner

    data = json.loads(
        (ROOT / "production" / "runs" / "exp_pcf_mri_vector_potential.json").read_text()
    )
    data["time"]["adaptive_cfl"] = {
        "dt_min": 1.0e-6,
        "dt_max": data["time"]["dt"],
    }
    child = tmp_path / "adaptive-child.json"
    child.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(child)
    parent_spec = dict(config.spec)
    record = SimpleNamespace(t=0.0, tstep=10)
    entry = {
        "tstep": 10,
        "plateau_window_stats": {},
        "file_sha256": "unused",
    }
    monkeypatch.setattr(
        runner, "select_qualified_parent_checkpoint", lambda *_args, **_kwargs: entry
    )
    monkeypatch.setattr(
        runner, "load_resume_checkpoint", lambda *_args, **_kwargs: record
    )
    monkeypatch.setattr(runner, "load_spec", lambda *_args, **_kwargs: parent_spec)
    monkeypatch.setattr(
        runner, "validate_bank_checkpoint_record", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner, "validate_quench", lambda *_args, **_kwargs: {"changed": {}}
    )

    dt = float(config.spec["time"]["dt"])
    with pytest.raises(QuenchError, match="strictly less"):
        _resolve_resume_or_quench(
            config,
            resume=None,
            quench_from=tmp_path / "parent",
            burn_in_steps=2,
            additional_time=2 * dt,
        )


def test_direct_quench_rejects_unwired_runner_before_output_or_parent_load(tmp_path):
    spec = load_config(ROOT / "production/runs/pcf_fluct_re400.json").spec
    out = tmp_path / "must-not-exist"
    with pytest.raises(QuenchError, match="quench runner is not implemented"):
        run_supported_spec(
            spec,
            additional_steps=2,
            quench=True,
            out_dir=out,
        )
    assert not out.exists()


def test_validate_only_api_rejects_unwired_quench_runner_before_output(tmp_path):
    out = tmp_path / "must-not-exist"
    with pytest.raises(QuenchError, match="quench runner is not implemented"):
        run_problem(
            config_path=ROOT / "production/runs/pcf_fluct_re400.json",
            out=out,
            quench_from=tmp_path / "missing-parent",
            additional_steps=2,
            validate_only=True,
        )
    assert not out.exists()


@pytest.mark.parametrize("option", ["compare_golden", "write_golden"])
def test_quench_rejects_golden_workflows_before_config_or_output(tmp_path, option):
    out = tmp_path / "must-not-exist"
    options = {option: True}
    with pytest.raises(QuenchError, match=option):
        run_problem(
            config_path=tmp_path / "missing-child.json",
            out=out,
            quench_from=tmp_path / "missing-parent",
            additional_steps=2,
            **options,
        )
    assert not out.exists()


@pytest.mark.parametrize("burn_in_steps", [True, 1.5, 1])
def test_invalid_nonquench_burn_in_is_rejected_before_config_or_output(
    tmp_path, burn_in_steps
):
    out = tmp_path / "must-not-exist"
    with pytest.raises(QuenchError, match="burn_in_steps"):
        run_problem(
            config_path=tmp_path / "missing-child.json",
            out=out,
            burn_in_steps=burn_in_steps,
        )
    assert not out.exists()


def test_direct_quench_rejects_burn_in_at_resolved_horizon_before_output(tmp_path):
    spec = load_config(ROOT / "production/runs/exp_pcf_mri_vector_potential.json").spec
    out = tmp_path / "must-not-exist"
    record = SimpleNamespace(t=0.0, tstep=0)
    with pytest.raises(QuenchError, match="strictly less"):
        run_supported_spec(
            spec,
            additional_steps=2,
            resume_checkpoint=record,
            quench=True,
            burn_in_steps=2,
            out_dir=out,
        )
    assert not out.exists()


def test_quench_failure_before_first_row_persists_parent_lower_bound(
    tmp_path, monkeypatch
):
    def fail_before_row(*_args, **_kwargs):
        raise RuntimeError("failed before cadence")

    config_path, duration = _install_fake_quench_solver(monkeypatch, fail_before_row)
    out = tmp_path / "failed-before-row"
    with pytest.raises(RuntimeError, match="before cadence"):
        _run_fake_quench(config_path, out)

    recorded = json.loads((out / "metadata.json").read_text())["quench"]["duration"]
    assert all(value is None for value in recorded["attained"].values())
    assert recorded["last_observed"] == {
        "absolute_time": duration.parent_time,
        "absolute_step": duration.parent_step,
        "additional_time": 0.0,
        "additional_steps": 0,
        "source": "parent_checkpoint",
        "is_lower_bound": True,
    }


def test_quench_keyboard_interrupt_after_row_persists_cadence_lower_bound(
    tmp_path, monkeypatch
):
    observed = {}

    def interrupt_after_row(_spec, *, on_row, **_kwargs):
        on_row(observed)
        raise KeyboardInterrupt

    config_path, duration = _install_fake_quench_solver(
        monkeypatch, interrupt_after_row
    )
    observed.update(
        {
            "t": duration.parent_time + 2 * duration.child_dt,
            "tstep": duration.parent_step + 2,
        }
    )
    out = tmp_path / "interrupted-after-row"
    with pytest.raises(KeyboardInterrupt):
        _run_fake_quench(config_path, out)

    recorded = json.loads((out / "metadata.json").read_text())["quench"]["duration"]
    assert all(value is None for value in recorded["attained"].values())
    assert recorded["last_observed"] == {
        "absolute_time": pytest.approx(observed["t"]),
        "absolute_step": observed["tstep"],
        "additional_time": pytest.approx(2 * duration.child_dt),
        "additional_steps": 2,
        "source": "cadence",
        "is_lower_bound": True,
    }
    archived_spec = json.loads((out / "spec.json").read_text())
    assert archived_spec == load_config(config_path).spec
    partial_rows = [
        json.loads(line)
        for line in (out / "diagnostics.partial.jsonl").read_text().splitlines()
    ]
    assert partial_rows == [{**observed, "energy_convention": "integral_abs2"}]


@pytest.mark.parametrize("endpoint_case", ["missing", "mismatched"])
def test_quench_invalid_reported_endpoint_is_archived_without_attainment(
    tmp_path, monkeypatch, endpoint_case
):
    result = {}

    def return_invalid_endpoint(_spec, **_kwargs):
        return result

    config_path, duration = _install_fake_quench_solver(
        monkeypatch, return_invalid_endpoint
    )
    result.update(
        {
            "scalars": {},
            "time_series": [{"t": duration.target_time}],
        }
    )
    if endpoint_case == "mismatched":
        result["run_horizon"] = {
            "final_time": duration.target_time + duration.child_dt,
            "final_step": duration.target_step,
        }
    out = tmp_path / endpoint_case
    with pytest.raises(QuenchError):
        _run_fake_quench(config_path, out)

    recorded = json.loads((out / "metadata.json").read_text())["quench"]["duration"]
    assert all(value is None for value in recorded["attained"].values())
    assert recorded["endpoint_validated"] is False
    assert recorded["series_final_time"] == pytest.approx(duration.target_time)
    if endpoint_case == "missing":
        assert recorded["reported_endpoint"] == {
            "final_time": None,
            "final_step": None,
        }
    else:
        assert recorded["reported_endpoint"] == {
            "final_time": pytest.approx(duration.target_time + duration.child_dt),
            "final_step": duration.target_step,
        }


def test_quench_success_certifies_consistent_endpoint_only_after_validation(
    tmp_path, monkeypatch
):
    result = {}

    def return_success(_spec, *, on_row, **_kwargs):
        on_row(
            {
                "t": result["run_horizon"]["final_time"],
                "tstep": result["run_horizon"]["final_step"],
                "divergence_b_l2": 0.0,
            }
        )
        return result

    config_path, duration = _install_fake_quench_solver(monkeypatch, return_success)
    result.update(
        {
            "scalars": {"magnetic_energy": 1.0, "divergence_b_l2": 0.0},
            "time_series": [
                {
                    "t": duration.target_time,
                    "magnetic_energy": 1.0,
                    "divergence_b_l2": 0.0,
                }
            ],
            "run_horizon": {
                "final_time": duration.target_time,
                "final_step": duration.target_step,
            },
        }
    )
    metadata = _run_fake_quench(config_path, tmp_path / "success")
    recorded = metadata["quench"]["duration"]
    assert recorded["endpoint_validated"] is True
    assert recorded["reported_endpoint"] == result["run_horizon"]
    assert recorded["series_final_time"] == pytest.approx(duration.target_time)
    assert recorded["attained"]["target_reached"] is True
    assert "last_observed" not in recorded


def test_quench_diagnostics_cadence_uses_additional_child_steps():
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    assert (
        _effective_diagnostics_every(
            spec,
            diagnostics_every=None,
            steps=None,
            resolution_tier=None,
            quench_steps=32,
        )
        == 2
    )
    scope = _validation_scope_metadata(
        spec,
        {"scalars": {"saturation_check_passed": False, "divergence_b_l2": 0.0}},
        device_record={"mode": "cpu_smoke"},
        compare_golden=False,
        steps=None,
        resolution_tier=None,
        quench_steps=32,
    )
    assert scope["kind"] == "quench_continuation"
    assert scope["quench_additional_steps"] == 32
    assert (
        _saturation_check_metadata(
            {"scalars": {"saturation_check_passed": False}},
            validation_scope=scope,
        )["required"]
        is False
    )
    assert (
        _validation_floor_metadata(
            {"scalars": {"divergence_b_l2": 0.0}},
            validation_scope=scope,
        )["required"]
        is True
    )


def test_saturation_predicate_rejects_nonfinite_growth_and_energy():
    assert not _saturation_passed(float("inf"), threshold=2.0, final_energies=(1.0,))
    assert not _saturation_passed(3.0, threshold=2.0, final_energies=(float("inf"),))
    assert not _saturation_passed(1.5, threshold=2.0, final_energies=(1.0,))
    assert _saturation_passed(3.0, threshold=2.0, final_energies=(1.0, 0.0))


def test_cpu_full_saturation_scope_uses_generated_gate():
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    scope = _validation_scope_metadata(
        spec,
        {"scalars": {"saturation_check_passed": False}},
        device_record={"mode": "cpu_smoke"},
        compare_golden=False,
        steps=None,
        resolution_tier="production",
    )

    assert scope["kind"] == "generated_saturated_golden"
    assert scope["bounded_smoke"] is False
    checks = _saturation_check_metadata(
        {
            "scalars": {
                "saturation_check_passed": False,
                "stationarity_check_passed": False,
            }
        },
        validation_scope=scope,
    )
    assert checks["required"] is True


def test_decaying_quench_is_an_outcome_but_health_floor_stays_strict(tmp_path):
    spec = json.loads(
        (ROOT / "production" / "runs" / "pcf_mhd_divfree.json").read_text()
    )
    diagnostics = {
        "scalars": {
            "growth_rate": -0.25,
            "saturation_check_passed": False,
            "stationarity_check_passed": False,
            "divergence_b_l2": 1.0e-12,
        },
        "time_series": [
            {"t": 1.0, "magnetic_energy": 1.0, "divergence_b_l2": 1.0e-12},
            {"t": 2.0, "magnetic_energy": 0.5, "divergence_b_l2": 1.0e-12},
        ],
    }
    scope = _validation_scope_metadata(
        spec,
        diagnostics,
        device_record={"mode": "cpu_smoke"},
        compare_golden=False,
        steps=None,
        resolution_tier=None,
        quench_steps=32,
    )
    saturation = _saturation_check_metadata(diagnostics, validation_scope=scope)
    floor = _validation_floor_metadata(diagnostics, validation_scope=scope)
    assert scope["kind"] == "quench_continuation"
    assert saturation["required"] is False
    assert floor["required"] is True
    assert floor["passed"] is True

    metadata = {
        "out_dir": str(tmp_path),
        "execution": {"status": "completed"},
        "saturation_checks": saturation,
        "validation_floor": floor,
    }
    _assert_required_saturation_checks(metadata)
    _assert_validation_floor_checks(metadata)

    bad_floor = _validation_floor_metadata(
        {
            "scalars": {
                "magnetic_energy": float("inf"),
                "divergence_b_l2": 2.0e-2,
            }
        },
        validation_scope=scope,
    )
    assert bad_floor["passed"] is False
    assert bad_floor["nonfinite_diagnostics"] == ["scalars.magnetic_energy"]
    assert bad_floor["max_divergence"] == pytest.approx(2.0e-2)


def test_classification_noise_floor_and_fit_exclude_pre_burn_rows():
    diagnostics = {
        "scalars": {"analysis_t_start": 10.0},
        "time_series": [
            {"t": 0.0, "mag_energy_fluct": 1.0e20},
            *[
                {
                    "t": float(t),
                    "mag_energy_fluct": 2.0 ** (t - 10),
                }
                for t in range(10, 18)
            ],
        ],
    }

    classification = _classification_metadata(diagnostics)

    assert classification["scientific_class"] == "growing"
    assert classification["fit"]["samples"] == 4
    assert classification["fit"]["window_start_t"] == pytest.approx(14.0)
    assert classification["fit"]["noise_floor"] == pytest.approx(1.0e-10 * 2.0**7)


def test_validation_floor_rejects_nonfinite_smoke_diagnostics(tmp_path):
    diagnostics = {
        "scalars": {"kinetic_energy": float("inf"), "divergence_l2": 0.0},
        "time_series": [{"t": 0.0, "kinetic_energy": 1.0}],
    }
    checks = _validation_floor_metadata(
        diagnostics, validation_scope={"kind": "bounded_saturation_smoke"}
    )
    assert checks["required"] is True
    assert checks["passed"] is False
    assert "scalars.kinetic_energy" in checks["nonfinite_diagnostics"]

    out = tmp_path / "run"
    out.mkdir()
    metadata = {
        "out_dir": str(out),
        "execution": {"status": "completed", "solver_execution_wired": True},
        "validation_floor": checks,
    }
    with pytest.raises(RuntimeError, match="validation floor failed"):
        _assert_validation_floor_checks(metadata)
    written = json.loads((out / "metadata.json").read_text())
    assert written["execution"]["status"] == "failed"


def test_validation_floor_rejects_missing_divergence_smoke_diagnostics():
    checks = _validation_floor_metadata(
        {"scalars": {"kinetic_energy": 1.0}},
        validation_scope={"kind": "cpu_smoke_finiteness_divergence_only"},
    )

    assert checks["required"] is True
    assert checks["passed"] is False
    assert checks["divergence_present"] is False


def test_write_golden_hashes_sanitized_scalars(tmp_path):
    config = load_config(
        ROOT / "production" / "examples" / "channel_poiseuille_hydro_v1.json"
    )
    diagnostics = {
        "scalars": {
            "flow_rate": float("inf"),
            "kinetic_energy": 1.0,
            "pressure_gradient": -0.002,
            "divergence_l2": 0.0,
        },
        "time_series": [],
    }

    golden_path = _write_golden(
        tmp_path / "golden.json",
        config,
        diagnostics,
        {"capture_skipped": True},
    )
    golden = validate_golden(golden_path)

    assert golden["diagnostics"]["scalars"]["flow_rate"] is None


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
    assert metadata["compilation_cache"]["requested"] is True
    assert Path(metadata["compilation_cache"]["path"]).exists()


def test_run_problem_restores_compilation_cache_config(tmp_path):
    import jax

    previous_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
    previous_min_time = getattr(
        jax.config, "jax_persistent_cache_min_compile_time_secs", None
    )

    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "tc_supercritical_saturation.json",
        out=tmp_path / "run",
        validate_only=True,
        capture_device=False,
    )

    assert metadata["compilation_cache"]["enabled"] is True
    assert metadata["compilation_cache"]["restores_process_config"] is True
    assert getattr(jax.config, "jax_compilation_cache_dir", None) == previous_dir
    assert (
        getattr(jax.config, "jax_persistent_cache_min_compile_time_secs", None)
        == previous_min_time
    )


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
        "dealias": 1.0,
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


def test_required_saturation_check_missing_key_fails(tmp_path):
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
            "present": False,
            "passed": None,
            "energy_growth_factor": 3.0,
        },
    }

    with pytest.raises(RuntimeError, match="saturation_check_passed is missing"):
        _assert_required_saturation_checks(metadata)

    written = json.loads((out / "metadata.json").read_text())
    assert written["execution"]["status"] == "failed"


def test_bounded_smoke_saturation_check_is_not_required():
    metadata = _saturation_check_metadata(
        {"scalars": {"saturation_check_passed": False}},
        validation_scope={"kind": "bounded_saturation_smoke"},
    )

    assert metadata == {
        "required": False,
        "present": True,
        "type_valid": True,
        "passed": False,
        "energy_growth_factor": None,
        "magnetic_energy_growth_factor": None,
        "stationarity_check_passed": None,
        "stationarity_relative_change": None,
    }


def test_full_generated_scope_validates_all_numeric_diagnostics(tmp_path):
    diagnostics = {
        "scalars": {"kinetic_energy": 1.0, "divergence_l2": 0.0},
        "time_series": [
            {"t": 0.0, "kinetic_energy": 1.0, "divergence_l2": 0.0},
            {"t": 1.0, "kinetic_energy": float("nan"), "divergence_l2": 0.0},
        ],
    }
    checks = _validation_floor_metadata(
        diagnostics, validation_scope={"kind": "generated_saturated_golden"}
    )

    assert checks["required"] is True
    assert checks["passed"] is False
    assert "time_series.1.kinetic_energy" in checks["nonfinite_diagnostics"]


def test_full_saturation_check_rejects_nonboolean_pass_flag(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    checks = _saturation_check_metadata(
        {
            "scalars": {
                "saturation_check_passed": 1.0,
                "energy_growth_factor": 3.0,
                "stationarity_check_passed": True,
            }
        },
        validation_scope={"kind": "generated_saturated_golden"},
    )
    metadata = {"out_dir": str(out), "execution": {}, "saturation_checks": checks}

    assert checks["type_valid"] is False
    with pytest.raises(RuntimeError, match="non-boolean"):
        _assert_required_saturation_checks(metadata)


def test_full_saturation_check_requires_stationarity(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    metadata = {
        "out_dir": str(out),
        "execution": {},
        "saturation_checks": {
            "required": True,
            "present": True,
            "type_valid": True,
            "passed": True,
            "stationarity_check_passed": False,
            "stationarity_relative_change": 0.25,
        },
    }

    with pytest.raises(RuntimeError, match="stationarity_check_passed=False"):
        _assert_required_saturation_checks(metadata)


def test_full_saturation_oracle_collects_stationarity_rows_and_writes_golden(
    tmp_path, monkeypatch
):
    import jax.numpy as jnp

    import examples.taylor_couette_dns_jax as tc_dns
    import production.oracles as oracles

    class FakeCircularCouette:
        R1 = 1.0
        b = 0.0

        def __init__(self, *_args):
            pass

    class FakeState:
        def __init__(self, step: int):
            self.step = int(step)

    class FakeTCSaturationSolver:
        Lz = 1.0
        dt = 0.01
        nu = 0.01

        def __init__(self, *_args, **kwargs):
            self.Lz = float(kwargs.get("Lz", self.Lz))
            self.dt = float(kwargs.get("dt", self.dt))
            self.nu = float(kwargs.get("nu", self.nu))

        def seed_linear_eigenmode(self, *_args, **_kwargs):
            return FakeState(0), complex(0.1, 0.0)

        def diagnostics(self, state):
            energy = 1.0e-6 if state.step == 0 else 2.0e-3
            return {"E": jnp.asarray(energy), "continuity_l2": 0.0, "div_linf": 0.0}

        def solve_with_cadence(
            self,
            state,
            steps,
            cadence,
            *,
            block_size=1,
            on_diagnostics=None,
            on_snapshot=None,
            on_checkpoint=None,
            should_stop=None,
            t0=0.0,
            tstep0=0,
        ):
            out = state
            for local_step in range(1, int(steps) + 1):
                tstep = int(tstep0) + local_step
                t = float(t0) + local_step * self.dt
                out = FakeState(tstep)
                if (
                    on_diagnostics is not None
                    and tstep % cadence.diagnostics_every == 0
                ):
                    on_diagnostics(t, tstep, self.diagnostics(out))
                if should_stop is not None:
                    should_stop(t, tstep, out)
            return out

        def solve(self, state, steps):
            return FakeState(state.step + int(steps))

    monkeypatch.setattr(tc_dns, "CircularCouette", FakeCircularCouette)
    monkeypatch.setattr(tc_dns, "AxisymmetricTCDNSJax", FakeTCSaturationSolver)
    monkeypatch.setattr(oracles, "_radial_velocity_linf", lambda _solver, _state: 0.0)
    monkeypatch.setattr(oracles, "_tc_inner_torque", lambda _solver, _state: 1.0)

    spec = json.loads(
        (ROOT / "production" / "runs" / "tc_supercritical_saturation.json").read_text()
    )
    spec["time"] = {**spec["time"], "dt": 0.01, "final_time": 0.04}
    spec["resolution"] = {**spec["resolution"], "production": {"Nr": 4, "Nz": 4}}
    spec_path = tmp_path / "tc_fake_full_saturation.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "full-saturation"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        write_golden=True,
        capture_device=False,
    )

    assert metadata["validation_scope"]["kind"] == "generated_saturated_golden"
    assert metadata["saturation_checks"]["passed"] is True
    assert metadata["saturation_checks"]["stationarity_check_passed"] is True
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) >= 4
    assert rows[-1]["stationarity_check_passed"] is True

    golden = validate_golden(out / "golden" / "golden.json")
    tolerances = golden["tolerance_model"]["scalars"]
    assert tolerances["stationarity_previous_mean"] > 0.0
    assert tolerances["stationarity_current_mean"] > 0.0
    assert tolerances["stationarity_window_samples"] == 1.0


def test_monitor_reuses_cadenced_diagnostics_for_divergence_guard():
    import production.oracles as oracles
    from jaxfun.io import run_with_cadence

    class CountingSolver:
        dt = 1.0

        def __init__(self):
            self.diagnostic_calls = 0

        def solve(self, state, steps):
            return int(state) + int(steps)

        def solve_with_cadence(
            self,
            state,
            steps,
            cadence,
            *,
            block_size=1,
            on_diagnostics=None,
            on_snapshot=None,
            on_checkpoint=None,
            should_stop=None,
            t0=0.0,
            tstep0=0,
        ):
            return run_with_cadence(
                self.solve,
                state,
                steps=steps,
                dt=self.dt,
                cadence=cadence,
                block_size=block_size,
                diagnostics=self.diagnostics,
                on_diagnostics=on_diagnostics,
                on_snapshot=on_snapshot,
                on_checkpoint=on_checkpoint,
                should_stop=should_stop,
                t0=t0,
                tstep0=tstep0,
            )

        def diagnostics(self, _state):
            self.diagnostic_calls += 1
            return {"divergence_l2": 0.0}

    solver = CountingSolver()
    rows = []

    out = oracles._solve_with_optional_checkpoints(
        solver,
        0,
        3,
        spec={
            "problem_id": "unit",
            "spec_hash": "hash",
            "golden": {"artifact_id": "unit"},
        },
        out_dir=None,
        checkpoint_every=None,
        snapshot_every=None,
        diagnostics_every=1,
        state_kind="unit",
        on_diagnostics_row=lambda t, tstep, diag: rows.append((t, tstep, diag)),
    )

    assert out == 3
    assert len(rows) == 3
    assert solver.diagnostic_calls == 3


def test_divergence_key_matching_is_explicit():
    import production.oracles as oracles
    import production.run_problem as runner

    positives = [
        "divL2",
        "div_linf",
        "divu",
        "divb_l2",
        "divergence_b",
        "continuity_l2",
        "scalars.divergence_l2",
    ]
    negatives = ["dividend_yield", "divide_error", "divergent_energy"]

    for key in positives:
        assert oracles._is_divergence_key(key)
        assert runner._is_divergence_diagnostic(key)
    for key in negatives:
        assert not oracles._is_divergence_key(key)
        assert not runner._is_divergence_diagnostic(key)


@pytest.mark.parametrize(
    "problem_id",
    ["pipe_hagen_poiseuille_v1", "pipe_womersley_v1"],
)
def test_pipe_hydro_golden_comparisons_run_without_shenfun(tmp_path, problem_id):
    out = tmp_path / problem_id
    metadata = run_problem(
        config_path=ROOT / "production" / "examples" / f"{problem_id}.json",
        out=out,
        compare_golden=True,
        capture_device=False,
    )

    assert metadata["comparison_passed"] is True
    assert metadata["execution"] == {
        "status": "completed",
        "solver_execution_wired": True,
        "execution_kind": "analytic-oracle",
    }
    assert metadata["adapter"]["solver_source_files"] == [
        "examples/pipe_flow_dns_jax.py"
    ]
    assert (out / "diagnostics.jsonl").exists()


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
        # Review round 3: a crash records where the streamed cadence rows live.
        "partial_diagnostics_path": str(out / "diagnostics.partial.jsonl"),
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
        UnsupportedSpecError, match="not in the jaxfun implementation allowlist"
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
    assert metadata["saturation_checks"]["present"] is True
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    final = rows[-1]
    assert final["kinetic_energy"] > 0.0
    assert final["total_kinetic_energy"] > final["kinetic_energy"]
    assert final["divergence_l2"] < 1.0e-4
    assert metadata["saturation_checks"]["passed"] == (
        final["energy_growth_factor"] > 2.0
    )
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
        "dealias": 1.0,
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
    assert metadata["saturation_checks"]["present"] is True
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
    assert metadata["saturation_checks"]["passed"] == (
        final["magnetic_energy_growth_factor"] > 2.0
    )
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
        "dealias": 1.0,
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
    # FJ-04: mislabelled "butterfly_by_mean" replaced by the mean-flux components.
    assert "mean_by" in final
    assert "nonaxisymmetric_fraction" in final
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
    assert "solver_steps" not in metadata["timing"]
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
        "eigenvalue_imag",
        "eigenvalue_real",
        "growth_rate",
        "kinetic_energy",
        "wall_shear_lower",
        "wall_shear_upper",
    ]
    line = json.loads((out / "diagnostics.jsonl").read_text().splitlines()[0])
    assert line["growth_rate"] == pytest.approx(-0.0034674010999505545)
    assert line["wall_shear_lower"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("problem_id", "expected_keys"),
    [
        (
            "pcf_mhd_conducting_v1",
            [
                "divergence_b_l2",
                "divergence_u_l2",
                "eigenvalue_imag",
                "eigenvalue_real",
                "growth_rate",
                "kinetic_energy",
                "magnetic_bc",
                "magnetic_energy",
                "maxwell_stress_xy",
                "total_energy",
            ],
        ),
        (
            "pcf_mri_shearbox_v1",
            [
                "divergence_b_l2",
                "divergence_u_l2",
                "eigenvalue_imag",
                "eigenvalue_real",
                "growth_rate",
                "kinetic_energy",
                "local_mri_growth",
                "local_mri_smax_over_omega",
                "magnetic_bc",
                "magnetic_energy",
                "maxwell_stress_xy",
                "q_shear",
                "total_energy",
            ],
        ),
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
        "eigenvalue_imag",
        "eigenvalue_real",
        "growth_rate",
        "kinetic_energy",
        "magnetic_bc",
        "magnetic_energy",
        "total_energy",
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
        "eigenvalue_imag",
        "eigenvalue_real",
        "growth_rate",
        "kinetic_energy",
        "rayleigh_stable",
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
    assert final["divergence_l2"] < 1.0e-4
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
    assert final["divergence_b_l2"] < 1.0e-3
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
                "magnetic_energy",
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
                "magnetic_bc",
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
                "rayleigh_stable",
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
                "magnetic_bc",
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
    spec["time"] = {**spec["time"], "final_time": 0.006}
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
    assert metadata["timing"]["solver_steps"] == 4
    assert metadata["timing"]["ms_per_step"] >= 0.0
    assert metadata["timing"]["steps_per_second"] > 0.0

    record = read_checkpoint(checkpoint_path)
    assert record.tstep == 4
    assert record.attrs["problem_id"] == "taylor_couette_hydro_dns_v1"
    assert record.attrs["artifact_id"] == "taylor_couette_hydro_dns_v1"
    assert record.attrs["spec_hash"] == metadata["spec_hash"]
    assert record.attrs["checkpoint_schema_version"] == 1
    assert record.attrs["solver_schema_version"] == 1
    assert record.attrs["diagnostics_path"].endswith("diagnostics.jsonl")
    dtype_metadata = json.loads(record.attrs["dtype_metadata_json"])
    assert dtype_metadata["production_run_dtype"] == "float64"
    assert dtype_metadata["field_dtypes"]
    assert dtype_metadata["field_shapes"]
    device_metadata = json.loads(record.attrs["device_metadata_json"])
    assert device_metadata == {"capture_skipped": True}
    assert record.attrs["prng_state_json"] == ""
    h5py = pytest.importorskip("h5py")
    with h5py.File(checkpoint_path, "r") as h5:
        assert sorted(h5["checkpoints"].keys()) == ["4"]

    resumed_metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=6,
        checkpoint_every=2,
        resume=out,
        capture_device=False,
    )
    assert resumed_metadata["run_options"]["resume"] == str(out)
    assert resumed_metadata["timing"]["solver_steps"] == 2
    record = read_checkpoint(checkpoint_path)
    assert record.tstep == 6
    with h5py.File(checkpoint_path, "r") as h5:
        assert sorted(h5["checkpoints"].keys()) == ["6"]
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

    for got, expected in zip(restarted.u, direct.u, strict=True):
        assert jnp.allclose(got, expected, rtol=1.0e-12, atol=1.0e-12)
    assert jnp.allclose(restarted.p, direct.p, rtol=1.0e-12, atol=1.0e-12)
    rows = [
        json.loads(line)
        for line in (out / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert [row["t"] for row in rows] == sorted({row["t"] for row in rows})


def test_tc_dns_runner_writes_uniform_snapshots(tmp_path):
    h5py = pytest.importorskip("h5py")
    spec = json.loads(
        (
            ROOT / "production" / "examples" / "taylor_couette_hydro_dns_v1.json"
        ).read_text()
    )
    spec["resolution"] = {**spec["resolution"], "Nr": 8, "Nz": 4}
    spec["time"] = {**spec["time"], "final_time": 0.001}
    spec_path = tmp_path / "tc_dns_snapshot.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "snapshot-run"
    metadata = run_problem(
        config_path=spec_path,
        out=out,
        steps=2,
        snapshot_every=1,
        capture_device=False,
    )

    assert metadata["snapshot_path"] == str(out / "snapshots" / "snapshots.h5")
    assert metadata["snapshot_xdmf_path"] == str(out / "snapshots" / "snapshots.xdmf")
    manifest = json.loads((out / "snapshots" / "manifest.json").read_text())
    assert manifest["problem_id"] == "taylor_couette_hydro_dns_v1"
    assert manifest["snapshot_every"] == 1
    snapshot_path = out / "snapshots" / "snapshots.h5"
    step_path_1 = out / "snapshots" / "steps" / "snapshot_00000001.h5"
    step_path_2 = out / "snapshots" / "steps" / "snapshot_00000002.h5"
    assert step_path_1.exists()
    assert step_path_2.exists()
    with h5py.File(snapshot_path, "r") as h5:
        root = h5["snapshots"]
        assert int(root.attrs["latest_step"]) == 2
        assert isinstance(root.get("1", getlink=True), h5py.ExternalLink)
        assert isinstance(root.get("2", getlink=True), h5py.ExternalLink)
        assert "snapshots/1/mesh/u_x/x0" in h5
        assert "snapshots/1/mesh/u_x/x1" in h5
        assert "snapshots/2/mesh/u_x/x0" in h5
        assert "snapshots/2/mesh/u_x/x1" in h5


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
    assert code == 1
