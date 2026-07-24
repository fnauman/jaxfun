"""Regression tests for the review-fix pass (F3, F4, F6, F7, F8)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import numpy as np
import pytest

from production.autograd_cost_benchmark import _build_production_case
from production.benchmark import _solver_and_seed_builders
from production.oracles import (
    ProductionOracleNotImplementedError,
    _kz_mode_from_spec,
    _pcf_primitive_time_integrator,
    _run_taylor_couette_hydro_dns,
)
from production.problem_spec import ProblemSpecError
from production.run_problem import (
    _assert_precision_matches_spec,
    _integrator_provenance,
    _resolved_physics_metadata,
)

ROOT = Path(__file__).resolve().parents[2]


def test_integrator_provenance_records_imexrk222_for_vector_potential():
    """F8: the curl path runs IMEXRK222, not the primitive CNAB2."""
    spec = {
        "representation": "vector_potential",
        "physics": "mri",
        "expected_oracle": {"type": "gpu_generated_saturated_dns"},
        "time": {"integrator": "IMEXRK222", "dt": 0.005},
    }
    prov = _integrator_provenance(spec)
    assert prov["actual"] == "IMEXRK222"
    assert prov["formal_order"] == 2


def test_integrator_provenance_records_cnab2_for_primitive():
    spec = {
        "representation": "primitive",
        "physics": "mri",
        "expected_oracle": {"type": "mri_saturation_ladder"},
        "time": {"integrator": "IMEXRK222", "dt": 0.005},
    }
    prov = _integrator_provenance(spec)
    assert prov["actual"] == "CNAB2"  # hard-coded in the primitive solver
    assert prov["formal_order"] == 2


@pytest.mark.parametrize("integrator", ["IMEXRK3", "SBDF3"])
def test_primitive_pcf_rejects_unimplemented_third_order_integrators(
    integrator,
) -> None:
    spec = {"time": {"integrator": integrator}}

    with pytest.raises(
        ProductionOracleNotImplementedError,
        match="PCF primitive MHD/MRI requires an implemented time-stepping integrator",
    ):
        _pcf_primitive_time_integrator(spec)


def test_resolved_physics_metadata_records_actual_precision():
    """F4: resolved physics precision must match the active run dtype."""
    spec = {
        "geometry": "pcf",
        "physics": "mri",
        "domain": {"x": [-1.0, 1.0], "y_period": 4.0, "z_period": 1.0},
        "nondimensional_groups": {
            "S": 1.0,
            "Omega": 2.0 / 3.0,
            "nu": 1e-3,
            "eta_mag": 1e-3,
            "Re": 1000.0,
            "Rm": 1000.0,
            "Pm": 1.0,
            "B0": 0.025,
        },
        "boundary_conditions": {"velocity": {}, "magnetic": {"type": "conducting"}},
        "forcing": {"B0": 0.025},
    }
    meta = _resolved_physics_metadata(spec, precision="float32")
    assert meta["precision"] == "float32"
    # default (no precision passed) still resolves
    assert _resolved_physics_metadata(spec)["precision"] == "float64"


def test_precision_mismatch_is_rejected():
    """F3: a spec precision that disagrees with the active dtype must fail loudly."""
    spec = {"precision": "float64"}
    with pytest.raises(ProblemSpecError, match="precision"):
        _assert_precision_matches_spec(spec, {"production_run_dtype": "float32"})
    # matching precision is fine
    _assert_precision_matches_spec(
        {"precision": "float32"}, {"production_run_dtype": "float32"}
    )
    # no declared precision -> no constraint
    _assert_precision_matches_spec({}, {"production_run_dtype": "float32"})


def test_release_ref_and_untracked_archive(tmp_path):
    """F6/F7: the strict gate reports the release-ref status and archives untracked."""
    from production.provenance import _head_release_ref, assert_release_clean

    ref = _head_release_ref()
    assert set(ref) == {
        "exact_tag",
        "tag_commit",
        "remote",
        "remote_tag_commit",
        "remote_verified",
        "is_immutable_ref",
    }
    assert ref["is_immutable_ref"] is ref["remote_verified"]

    # The working tree during development is dirty/unpinned, so allow_dirty archives.
    prov = assert_release_clean(tmp_path, allow_dirty=True)
    gate = prov["release_gate"]
    if gate["discovery_only"]:
        assert "diff_sha256" in gate
        assert "untracked_archived" in gate  # F7: untracked list is captured


def test_nonaxisymmetric_tc_mode_requires_azimuthal_resolution():
    spec = {
        "resolution": {"Nr": 8, "Nz": 4},
        "mode": {
            "axial_wavenumber": math.pi,
            "azimuthal_wavenumber": 1,
        },
    }

    with pytest.raises(
        ProductionOracleNotImplementedError,
        match=r"require resolution\.Ntheta",
    ):
        _kz_mode_from_spec(spec, 2.0)

    spec["resolution"]["Ntheta"] = 4
    assert _kz_mode_from_spec(spec, 2.0) == 1


def test_tc_hydro_oracle_defaults_missing_integrator_to_cnab2():
    jax.config.update("jax_enable_x64", True)
    spec = json.loads(
        (ROOT / "production/examples/taylor_couette_hydro_dns_v1.json").read_text(
            encoding="utf-8"
        )
    )
    spec["resolution"].update({"Nr": 8, "Nz": 4})
    spec["time"].pop("integrator")

    result = _run_taylor_couette_hydro_dns(spec, steps=1)

    assert math.isfinite(result["scalars"]["growth_rate"])
    assert result["scalars"]["divergence_linf"] < 1.0e-6


def test_tc_hydro_benchmark_seed_supports_legacy_modeless_specs():
    jax.config.update("jax_enable_x64", True)
    spec = json.loads(
        (ROOT / "production/examples/taylor_couette_hydro_dns_v1.json").read_text(
            encoding="utf-8"
        )
    )
    spec["resolution"].update({"Nr": 8, "Nz": 4})
    spec.pop("mode")

    build_solver, seed_state = _solver_and_seed_builders(spec)
    solver = build_solver()
    state = seed_state(solver)

    assert all(
        np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(state)
    )


@pytest.mark.parametrize("integrator", ["IMEXRK3", "SBDF3"])
def test_autograd_primitive_override_rejects_unsupported_integrator(integrator):
    with pytest.raises(
        ProductionOracleNotImplementedError,
        match="PCF primitive MHD/MRI requires an implemented time-stepping integrator",
    ):
        _build_production_case("primitive_pcf", (8, 4), integrator)


def test_autograd_primitive_metadata_reports_selected_cnab2():
    _build, _seed, metadata = _build_production_case("primitive_pcf", (8, 4), "CNAB2")
    assert metadata["integrator"] == "CNAB2"
    assert metadata["integrator_order"] == 2
    assert metadata["resolution"] == [8, 4]


@pytest.mark.parametrize(
    "filename",
    ["taylor_couette_hydro_3d_v1.json", "taylor_couette_mhd_3d_v1.json"],
)
def test_3d_tc_growth_tolerance_matches_deterministic_2d_goldens(filename):
    spec = json.loads(
        (ROOT / "production/examples" / filename).read_text(encoding="utf-8")
    )
    assert spec["tolerance_model"]["scalars"]["growth_rate"] == 1.0e-6


@pytest.mark.parametrize(
    "filename",
    ["taylor_couette_hydro_3d_v1.json", "taylor_couette_mhd_3d_v1.json"],
)
def test_3d_tc_native_axes_match_solver_storage_order(filename):
    spec = json.loads(
        (ROOT / "production/examples" / filename).read_text(encoding="utf-8")
    )
    assert spec["native_axes"] == {
        "axis_0": "theta azimuthal",
        "axis_1": "z axial",
        "axis_2": "r radial",
    }
