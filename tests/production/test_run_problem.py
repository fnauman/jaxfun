import json
from pathlib import Path

import pytest

from production.compare_goldens import validate_golden
from production.problem_spec import UnsupportedSpecError
from production.run_problem import SolverExecutionNotImplementedError, main, run_problem

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


def test_non_validate_run_fails_explicitly_until_solver_is_wired(tmp_path):
    with pytest.raises(
        SolverExecutionNotImplementedError, match="solver execution is not wired"
    ):
        run_problem(
            config_path=ROOT
            / "production"
            / "runs"
            / "tc_supercritical_saturation.json",
            out=tmp_path / "run",
            capture_device=False,
        )


def test_channel_analytic_run_writes_diagnostics_and_compares_golden(tmp_path):
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
    code = main(
        [
            "--config",
            str(ROOT / "production" / "runs" / "tc_supercritical_saturation.json"),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 2
