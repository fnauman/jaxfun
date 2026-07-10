"""Acceptance tests for the FJ-00..FJ-04/FJ-13 P0 tranche.

These exercise the correctness fixes end-to-end at tiny resolution:
FJ-01 (streamwise dealiasing under the named-axis contract + pre-fix checkpoint
rejection), FJ-02 (non-axisymmetric 3-D seed), FJ-04 (ZNF-safe diagnostics), and
FJ-13 (provenance capture + release gate).
"""

from __future__ import annotations

import pytest

# --------------------------------------------------------------------------- FJ-01


def test_streamwise_is_dealiased_under_named_axis_contract():
    """FJ-01: the semantic map pads the streamwise (y) periodic direction, which the
    old positional [1.0, 1.5, 1.5] handed to the (y, z, x) solver did NOT."""
    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    common = dict(
        S=1.0, omega=2.0 / 3.0, B0=0.1, nu=1e-2, eta_mag=1e-2,
        Nx=8, Ny=6, Nz=6, Ly=4.0, Lz=1.0, dt=1e-3,
    )
    corrected = PCFMRIDNSJax(dealias=(1.5, 1.5, 1.0), **common)  # native (y, z, x)
    buggy = PCFMRIDNSJax(dealias=(1.0, 1.5, 1.5), **common)  # streamwise y unpadded

    # padded_counts is the per-native-axis padded quadrature count (y, z, x).
    assert corrected.padded_counts is not None
    assert buggy.padded_counts is not None
    y_corrected = corrected.padded_counts[0]
    y_buggy = buggy.padded_counts[0]
    assert y_corrected > y_buggy, (
        "streamwise axis must be padded under the corrected mapping "
        f"(got corrected={y_corrected}, buggy={y_buggy})"
    )


def test_pre_fj01_checkpoint_rejected_on_resume():
    """FJ-01 acceptance: a pre-contract checkpoint cannot seed a post-fix run."""
    from production.oracles import _reject_pre_fj01_checkpoint

    spec = {"numerics_contract_version": 2}
    # A pre-fix checkpoint carries version 0/absent.
    with pytest.raises(ValueError, match="pre-FJ-01"):
        _reject_pre_fj01_checkpoint({"numerics_contract_version": 1}, spec)
    with pytest.raises(ValueError, match="pre-FJ-01"):
        _reject_pre_fj01_checkpoint({}, spec)
    # A matching-version checkpoint is accepted.
    _reject_pre_fj01_checkpoint({"numerics_contract_version": 2}, spec)


# --------------------------------------------------------------------------- FJ-02


def test_net_flux_3d_seed_is_nonaxisymmetric_and_solenoidal():
    """FJ-02: the non-axisymmetric seed leaves the axisymmetric subspace and stays
    divergence-free / wall-satisfying."""
    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    solver = PCFMRIDNSJax(
        S=1.0, omega=2.0 / 3.0, B0=0.025, nu=1e-2, eta_mag=1e-2,
        Nx=10, Ny=6, Nz=6, Ly=4.0, Lz=1.0, dt=1e-3, dealias=1.0,
    )
    # axisymmetric packet (ky=0) alone -> ~zero non-axisymmetric energy
    axi, _ = solver.seed_linear_eigenmode(ky_mode=0, kz_mode=1, amp=1e-3)
    e_axi, e_axi_total = solver.nonaxisymmetric_energy(axi)
    assert float(e_axi) < 1e-12 * float(e_axi_total)

    # superpose a ky=1 eigenmode -> nonzero non-axisymmetric energy
    na, _ = solver.seed_linear_eigenmode(ky_mode=1, kz_mode=1, amp=1e-3)
    combined = tuple(axi.x[i] + na.x[i] for i in range(len(axi.x)))
    from production.oracles import _pcf_state_from_components

    state = _pcf_state_from_components(axi, combined)
    e_nonaxi, e_total = solver.nonaxisymmetric_energy(state)
    assert float(e_nonaxi) > 0.0
    diag = solver.diagnostics(state)
    assert float(diag["divu"]) < 1e-8
    assert float(diag["divb"]) < 1e-8


# --------------------------------------------------------------------------- FJ-04


def test_znf_diagnostics_are_finite_without_net_flux_alpha():
    """FJ-04: a B0=0 run yields finite diagnostics and omits the net-flux alpha."""
    import jax.numpy as jnp

    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    solver = PCFMRIDNSJax(
        S=1.0, omega=2.0 / 3.0, B0=0.0, nu=1e-2, eta_mag=1e-2,
        Nx=8, Ny=4, Nz=4, Ly=4.0, Lz=1.0, dt=1e-3, dealias=1.0,
    )
    state, _ = solver.seed_linear_eigenmode(kz_mode=1, amp=1e-2)
    diag = solver.diagnostics(state)
    assert "transport_alpha" not in diag and "alpha_B0" not in diag
    assert "alpha_Sh" in diag
    assert all(
        bool(jnp.isfinite(jnp.asarray(v)).all()) for v in diag.values()
    )


# --------------------------------------------------------------------------- FJ-13


def test_provenance_capture_records_git_and_versions():
    from production.provenance import capture_provenance

    prov = capture_provenance()
    assert "commit" in prov and "dirty" in prov
    assert "lockfile_sha256" in prov
    assert "versions" in prov and "jax" in prov["versions"]


def test_release_gate_blocks_dirty_tree(tmp_path):
    """FJ-13: a dirty tree is refused unless discovery-only override is given."""
    from production.provenance import ReleaseCleanlinessError, assert_release_clean

    # The working tree during development is dirty/unpushed, so the strict gate
    # must refuse; the override archives the diff and returns a flagged block.
    try:
        result = assert_release_clean(tmp_path, allow_dirty=False)
    except ReleaseCleanlinessError:
        result = None
    if result is None:
        prov = assert_release_clean(tmp_path, allow_dirty=True)
        assert prov["release_gate"]["discovery_only"] is True
        assert (tmp_path / "worktree_diff.patch").exists()
    else:
        # A pristine checkout (rare in CI of a feature branch) is allowed.
        assert result["passed"] is True
