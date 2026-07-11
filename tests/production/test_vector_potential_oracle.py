"""FJ-03: curl / vector-potential PCF-MRI production oracle."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from production.oracles import load_resume_checkpoint, run_supported_spec

# Final-state scalars that must agree between a straight run and a
# checkpoint+resume continuation of the same spec (growth-family scalars are
# excluded: they are measured from the evolved baseline, which differs by
# construction on a resume).
_CONTINUATION_KEYS = (
    "kinetic_energy",
    "magnetic_energy",
    "total_energy",
    "divergence_u_l2",
    "divergence_b_l2",
    "reynolds_stress",
    "maxwell_stress_xy",
    "total_stress",
    "mean_bx",
    "mean_by",
    "mean_bz",
    "mag_energy_mean",
    "mag_energy_fluct",
)


def _vp_spec(**groups):
    spec = {
        "problem_id": "pcf_mri_vector_potential_smoke",
        "spec_hash": "vp-smoke-spec-hash",
        "numerics_contract_version": 2,
        "geometry": "pcf",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "boundary_conditions": {
            "velocity": {"type": "no_slip_shearbox_walls"},
            "magnetic": {"type": "conducting"},
        },
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0, "Omega": 2.0 / 3.0, "nu": 2e-2, "eta_mag": 2e-2,
            "Re": 50.0, "Rm": 50.0, "Pm": 1.0, "B0": 0.05,
        },
        "time": {"integrator": "IMEXRK222", "dt": 1e-3, "final_time": 0.01},
        "resolution": {"Nx": 17, "Ny": 8, "Nz": 16, "family": "L"},
        "initial_condition": {"velocity_amplitude": 0.05, "magnetic_amplitude": 1e-3},
        "forcing": {"B0": 0.05},
        "golden": {
            "artifact_id": "pcf_mri_vector_potential_smoke",
            "regeneration_command": "test-only spec; no committed golden",
        },
    }
    spec["nondimensional_groups"].update(groups)
    spec["forcing"]["B0"] = spec["nondimensional_groups"]["B0"]  # keep sources consistent
    return spec


def test_vector_potential_oracle_is_solenoidal_by_construction():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    # B = curl A -> div B = 0 to roundoff (the invariant the primitive path lacks)
    assert sc["divergence_b_l2"] < 1e-8
    for key in ("kinetic_energy", "magnetic_energy", "total_stress", "alpha_Sh", "growth_rate"):
        assert key in sc
    assert len(out["time_series"]) >= 2


def test_vector_potential_oracle_is_znf_safe():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(B0=0.0), steps=3)
    sc = out["scalars"]
    # ZNF: no net-flux alpha, but the shear-normalized alpha is present and finite
    assert "transport_alpha" not in sc
    assert "alpha_Sh" in sc

    assert math.isfinite(sc["alpha_Sh"])


def test_vector_potential_emits_flux_diagnostics():
    """FJ-04: the ZNF curl workhorse must expose mean flux + mean/fluct split."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=3)
    sc = out["scalars"]
    for key in (
        "mean_bx", "mean_by", "mean_bz",
        "mag_energy_mean", "mag_energy_fluct",
        "flux_drift_bx", "flux_drift_by", "flux_drift_bz",
    ):
        assert key in sc


def test_vector_potential_energy_split_identity_and_total_field_means():
    """Total-field magnetic semantics (review round 3, blocker 1).

    The split and the component means are TOTAL-field quantities: a net-flux run
    reports mean_bz == B0 (matching shearpy), and
    mag_energy_mean + mag_energy_fluct == magnetic_energy_total exactly, in the
    family's integral_abs2 convention.
    """
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(), steps=2)
    sc = out["scalars"]
    assert sc["energy_convention"] == "integral_abs2"
    assert sc["box_volume"] == pytest.approx(2.0 * 4.0 * 1.0)
    # Net-flux run: the physical mean field includes the imposed B0.
    assert sc["mean_bz"] == pytest.approx(0.05, rel=1e-6)
    split = sc["mag_energy_mean"] + sc["mag_energy_fluct"]
    assert sc["magnetic_energy_total"] == pytest.approx(split, rel=1e-9)


def test_vector_potential_znf_split_reduces_to_perturbation_energy():
    """At B0=0 the total field is the perturbation, so the split identity
    coincides with `magnetic_energy` (the original factor-2 regression)."""
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_vp_spec(B0=0.0), steps=2)
    sc = out["scalars"]
    split = sc["mag_energy_mean"] + sc["mag_energy_fluct"]
    assert sc["magnetic_energy"] == pytest.approx(split, rel=1e-9)
    assert abs(sc["mean_bz"]) < 1e-12


def test_vector_potential_checkpoint_resume_matches_straight_run(tmp_path):
    """FJ-03/FJ-05: MHDState checkpoint + resume reproduces the straight run."""
    jax.config.update("jax_enable_x64", True)
    spec = _vp_spec()

    straight_dir = tmp_path / "straight"
    straight = run_supported_spec(
        spec, steps=4, out_dir=straight_dir, checkpoint_every=2
    )

    parent_dir = tmp_path / "parent"
    run_supported_spec(spec, steps=2, out_dir=parent_dir, checkpoint_every=2)
    record = load_resume_checkpoint(parent_dir)
    assert record.tstep == 2
    resumed = run_supported_spec(spec, steps=4, resume_checkpoint=record)

    for key in _CONTINUATION_KEYS:
        assert np.isclose(
            resumed["scalars"][key],
            straight["scalars"][key],
            rtol=1e-10,
            atol=1e-14,
        ), key
    # The resumed series starts from the checkpointed time, not t=0.
    assert resumed["time_series"][0]["t"] == pytest.approx(2 * spec["time"]["dt"])
    # Both checkpoint files carry the curl state_kind.
    assert str(record.attrs["state_kind"]) == "pcf_vector_potential_mhd_saturation"
    assert (straight_dir / "checkpoints" / "checkpoints.h5").exists()


def test_vector_potential_resume_rejects_spec_hash_mismatch(tmp_path):
    jax.config.update("jax_enable_x64", True)
    parent_dir = tmp_path / "parent"
    run_supported_spec(_vp_spec(), steps=2, out_dir=parent_dir, checkpoint_every=2)
    record = load_resume_checkpoint(parent_dir)

    other = _vp_spec()
    other["spec_hash"] = "a-different-spec-hash"
    with pytest.raises(ValueError, match="spec_hash"):
        run_supported_spec(other, steps=4, resume_checkpoint=record)


def test_vector_potential_quench_continues_with_new_physics(tmp_path):
    """FJ-05: a quench (changed eta/Rm) continues from the parent MHDState."""
    jax.config.update("jax_enable_x64", True)
    parent_dir = tmp_path / "parent"
    parent = run_supported_spec(
        _vp_spec(), steps=2, out_dir=parent_dir, checkpoint_every=2
    )
    record = load_resume_checkpoint(parent_dir)

    child = _vp_spec(Rm=40.0, eta_mag=2.5e-2, Pm=0.8)
    child["spec_hash"] = "vp-smoke-quench-child-hash"
    out = run_supported_spec(child, steps=4, resume_checkpoint=record, quench=True)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    assert math.isfinite(sc["magnetic_energy"]) and sc["magnetic_energy"] > 0.0
    assert sc["divergence_b_l2"] < 1e-8
    # FJ-05 baseline: the first series row is the loaded parent state at the
    # checkpoint time, not a fresh seed at t=0.
    first = out["time_series"][0]
    assert first["t"] == pytest.approx(2 * child["time"]["dt"])
    assert first["magnetic_energy"] == pytest.approx(
        parent["scalars"]["magnetic_energy"], rel=1e-10
    )


def test_vector_potential_snapshot_writes_fields(tmp_path):
    jax.config.update("jax_enable_x64", True)
    out_dir = tmp_path / "run"
    run_supported_spec(_vp_spec(), steps=2, out_dir=out_dir, snapshot_every=2)
    snapshot_index = out_dir / "snapshots" / "snapshots.h5"
    assert snapshot_index.exists()
    assert (out_dir / "snapshots" / "steps" / "snapshot_00000002.h5").exists()
    assert (out_dir / "snapshots" / "manifest.json").exists()

    import h5py

    with h5py.File(out_dir / "snapshots" / "steps" / "snapshot_00000002.h5", "r") as h5:
        names = set()
        h5.visit(names.add)
        joined = " ".join(names)
        for field in ("u_x", "u_y", "u_z", "b_x", "b_y", "b_z"):
            assert field in joined
