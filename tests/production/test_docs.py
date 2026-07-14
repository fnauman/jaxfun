from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_validation_scope_docs_cover_bounded_smoke_outputs():
    readme = (ROOT / "production" / "README.md").read_text()
    commands = (ROOT / "production" / "commands.md").read_text()
    validate_gpu = (ROOT / "production" / "validate_gpu.sh").read_text()

    assert "## Validation scopes" in readme
    assert "bounded_saturation_smoke" in readme
    assert "cpu_smoke_finiteness_divergence_only" in readme
    assert "generated_saturated_golden" in readme
    assert "not full production saturation goldens" in commands
    assert "SMOKE ONLY" in validate_gpu
    assert "parity-saturation" in readme
    assert "parity-saturation" in commands
    assert "--resume" in readme
    assert "--resume" in commands
    assert "--snapshot-every" in readme
    assert "--snapshot-every" in commands
    assert "--diagnostics-every" in readme
    assert "--diagnostics-every" in commands
    assert "compilation-cache" in commands
    assert "ms_per_step" in readme
    assert "ms_per_step" in commands
    assert "--allow-same-backend" in readme
    assert "--allow-same-backend" in commands
    assert "validation_scope=bounded_saturation_smoke" in commands
    assert "failed row" in commands
    assert "comparison details before exiting nonzero" in commands
    assert "failed comparison details" in readme
    assert "--resolution-tier smoke" in commands
    assert "bounded CPU/GPU agreement evidence" in commands
    assert "--resolution-tier smoke|start|production" in readme
    assert "missing or false" in readme
    assert "saturation_check_passed`" in commands
    assert "diagnostic is missing or false" in commands
    assert "N=(32,64,32)" in readme
    assert "N=(32,64,32)" in commands
    assert "pcf_fluct_re400" in readme
    assert "pcf_mhd_divfree" in readme
    assert "exp_pcf_mri_shearbox_growth" in readme
    assert "tc_supercritical_saturation" in readme
    assert "tc_mri_nonlinear_saturation" in readme
    assert "production_ready_limited_scope" in readme
    assert "qualified_candidate" in readme
    assert "selected_workhorse_pending_full_run" in readme
    assert "finite_divergence_only" in readme
    assert "exp_pcf_mri_vector_potential" in readme
    # The magnetic-divergence evidence must name the representation that preserves
    # div B and distinguish it from the primitive-`b` paths that do not.
    # (Deliberately not asserting exact float literals: those are golden-derived
    # and reformat without changing correctness -- see the golden metadata for the
    # authoritative values.)
    assert "vector potential" in readme.lower()
    assert "B=B0+curl(A)" in readme
    assert "not roundoff-solenoidal" in readme
    assert "quarantined" in readme
    assert "retained failed generated-saturation" in commands
    assert "per-side wall times" in commands
    assert "64 MB HDF5 checkpoint" in commands
    assert "385 MB HDF5 checkpoint is intentionally not" in commands
    assert "25 MB HDF5 checkpoint is intentionally not" in commands
    assert "52 MB HDF5 checkpoint is intentionally not" in commands
    assert "807 MB HDF5 payload" in commands
    assert "spectraldns.cross_repository_comparison.v1" in readme
    assert "shearbox_to_pcf" in readme
    assert "local_pcf_to_taylor_couette" in readme
    assert "comparison_id" in commands
    assert "pair_id" in commands
    assert "byte-for-byte deterministic" in commands


def test_vector_potential_bc_menu_docs_are_current():
    """The docs must state which scripts/wall types preserve div B = 0, the
    measured evidence, and the honest limits of the new configurations."""
    readme = (ROOT / "production" / "README.md").read_text()
    commands = (ROOT / "production" / "commands.md").read_text()
    known = (ROOT / "production" / "KNOWN_ISSUES.md").read_text()

    # All four solenoidal-preserving configurations are documented.
    for pid in (
        "exp_pcf_mri_vector_potential",
        "exp_pcf_mri_vp_insulating",
        "exp_tc_mri_vector_potential",
        "exp_tc_mri_vp_insulating",
    ):
        assert pid in readme, pid
        assert pid in commands, pid
    assert "examples/taylor_couette_vp_jax.py" in readme
    # Wall-condition menu: what exists and what deliberately does not.
    assert "Magnetic wall-condition menu" in readme
    assert "pseudo_vacuum" in readme
    assert "Stress-free velocity walls: not implemented" in readme
    # Adaptive CFL stepping is documented with its recorded scalars.
    assert "Adaptive CFL stepping" in readme
    assert "adaptive_cfl" in readme
    assert "n_dt_changes" in readme
    assert "adaptive_cfl" in commands
    # Honest nuance: the TC witness is a resolution floor, not a fixed epsilon.
    assert "forward-projected coefficient representation" in readme
    assert "insulating_bc_residual" in readme
    # The ledger records the CPU-only status and stated conventions.
    assert "KI-9" in known
    assert "trapped-flux Faraday row" in known


def test_pipe_hydro_docs_describe_wired_cheap_parity():
    readme = (ROOT / "production" / "README.md").read_text()
    commands = (ROOT / "production" / "commands.md").read_text()

    assert "pipe_hagen_poiseuille_v1" in readme
    assert "pipe_womersley_v1" in readme
    assert "golden comparisons wired for both pipe hydro goldens" in readme
    assert "Nine-run cheap parity batch" in commands
    assert "including `pipe_hagen_poiseuille_v1`" in commands
