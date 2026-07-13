"""FJ-09 review blocker 5: pseudo-vacuum is a production-wired primitive-family BC."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import pytest

from production.oracles import load_resume_checkpoint, run_supported_spec
from production.problem_spec import UnsupportedSpecError, load_spec

ROOT = Path(__file__).resolve().parents[2]
PV_SPEC = ROOT / "production" / "runs" / "exp_pcf_mri_pseudo_vacuum.json"


def test_pseudo_vacuum_spec_validates_and_hashes():
    spec = load_spec(PV_SPEC)
    assert spec["boundary_conditions"]["magnetic"]["type"] == "pseudo_vacuum"
    assert spec["support_state"] == "experimental"
    assert spec["spec_hash"]


def test_pseudo_vacuum_rejected_for_vector_potential_representation(tmp_path):
    data = json.loads(PV_SPEC.read_text(encoding="utf-8"))
    data["representation"] = "vector_potential"
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(UnsupportedSpecError, match="primitive-b family only"):
        load_spec(path)


def test_pseudo_vacuum_saturation_smoke_propagates_bc(tmp_path):
    """The oracle must construct the solver with pseudo-vacuum walls, seed from
    the pseudo-vacuum linear operator, and record the BC in the scalars."""
    jax.config.update("jax_enable_x64", True)
    data = json.loads(PV_SPEC.read_text(encoding="utf-8"))
    # materialize the smoke tier resolution directly for a fast oracle-level run
    data["resolution"] = {
        **data["resolution"]["smoke"],
        "family": data["resolution"]["family"],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    spec = load_spec(path)

    out = run_supported_spec(spec, steps=3)
    sc = out["scalars"]
    assert sc["magnetic_bc"] == "pseudo_vacuum"
    assert sc["energy_convention"] == "half_integral_abs2"
    for key in ("kinetic_energy", "magnetic_energy", "divergence_b_l2"):
        assert math.isfinite(sc[key])
    assert sc["divergence_b_l2"] < 1e-2


def test_primitive_pcf_quench_runs_exact_additional_steps(tmp_path):
    jax.config.update("jax_enable_x64", True)
    data = json.loads(PV_SPEC.read_text(encoding="utf-8"))
    data["resolution"] = {
        **data["resolution"]["smoke"],
        "family": data["resolution"]["family"],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    parent_spec = load_spec(path)
    parent_dir = tmp_path / "parent"
    run_supported_spec(
        parent_spec,
        steps=1,
        out_dir=parent_dir,
        checkpoint_every=1,
    )
    record = load_resume_checkpoint(parent_dir)

    child = json.loads(json.dumps(parent_spec))
    child["nondimensional_groups"].update(
        {"Rm": 800.0, "eta_mag": 1.0 / 800.0, "Pm": 0.8}
    )
    child["spec_hash"] = "primitive-pcf-quench-child"
    streamed = []
    out = run_supported_spec(
        child,
        additional_steps=2,
        resume_checkpoint=record,
        quench=True,
        diagnostics_every=1,
        on_row=streamed.append,
    )

    assert out["time_series"][0]["t"] == pytest.approx(record.t)
    assert out["time_series"][-1]["t"] == pytest.approx(
        record.t + 2 * child["time"]["dt"]
    )
    assert out["run_horizon"] == {
        "final_time": pytest.approx(record.t + 2 * child["time"]["dt"]),
        "final_step": record.tstep + 2,
    }
    assert streamed
    assert all(isinstance(row["tstep"], int) for row in streamed)


def test_conducting_and_pseudo_vacuum_linear_operators_differ():
    """FJ-09's point: the wall BC changes the linear problem. At a converged
    resolution the pseudo-vacuum leading growth rate must differ measurably
    from conducting for the same shearbox parameters."""
    import math as m

    from examples.pcf_linear_jax import PlaneCouetteLinear

    evs = {}
    for bc in ("conducting", "pseudo_vacuum"):
        lin = PlaneCouetteLinear.shearpy(
            nx=48,
            Re=1000.0,
            Rm=1000.0,
            shear_rate=1.0,
            omega=2.0 / 3.0,
            by=0.0,
            bz=0.025,
            velocity_scale=1.0,
            magnetic_bc=bc,
        )
        w, _ = lin.eigs(2.0 * m.pi / 4.0, 2.0 * m.pi / 1.0, n_return=1)
        evs[bc] = w[0]
    assert abs(evs["pseudo_vacuum"].real - evs["conducting"].real) > 1e-2


def test_primitive_total_field_means_include_imposed_background():
    """Review round 3, blocker 1: mean_bz must report the physical mean field
    (imposed B0 included) so net-flux runs are cross-code comparable, and the
    split identity anchors to the total-field energy."""
    jax.config.update("jax_enable_x64", True)
    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    solver = PCFMRIDNSJax(Nx=8, Ny=4, Nz=4, B0=0.1, dt=1e-3, family="L")
    state = solver.zero_state()
    mbx, mby, mbz = solver.magnetic_component_means(state)
    assert float(mbx) == pytest.approx(0.0, abs=1e-14)
    assert float(mby) == pytest.approx(0.0, abs=1e-14)
    assert float(mbz) == pytest.approx(0.1, rel=1e-12)

    mag_total, mag_mean, mag_fluct = solver.magnetic_energy_split(state)
    # Zero perturbation: the total field IS the uniform imposed field.
    assert float(mag_total) == pytest.approx(float(mag_mean), rel=1e-12)
    assert float(mag_fluct) == pytest.approx(0.0, abs=1e-12)
    # Physical 0.5 * V * B0^2 in the family convention.
    assert float(mag_total) == pytest.approx(0.5 * solver._volume * 0.1**2, rel=1e-9)
