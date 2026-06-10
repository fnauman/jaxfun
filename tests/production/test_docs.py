from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_validation_scope_docs_cover_bounded_smoke_outputs():
    readme = (ROOT / "production" / "README.md").read_text()
    commands = (ROOT / "production" / "commands.md").read_text()

    assert "## Validation scopes" in readme
    assert "bounded_saturation_smoke" in readme
    assert "cpu_smoke_finiteness_divergence_only" in readme
    assert "generated_saturated_golden" in readme
    assert "not full production saturation goldens" in commands
    assert "validation_scope=bounded_saturation_smoke" in commands
    assert "failed row" in commands
    assert "comparison details before exiting nonzero" in commands
    assert "failed comparison details" in readme
    assert "--resolution-tier smoke" in commands
    assert "bounded CPU/GPU agreement evidence" in commands
    assert "--resolution-tier smoke|start|production" in readme
    assert "saturation_check_passed=false" in readme
    assert "saturation_check_passed`" in commands
    assert "diagnostic is false" in commands
    assert "N=(32,64,32)" in readme
    assert "N=(32,64,32)" in commands
    assert "pcf_fluct_re400" in readme
    assert "pcf_mhd_divfree" in readme
    assert "exp_pcf_mri_shearbox_growth" in readme
    assert "tc_supercritical_saturation" in readme
    assert "tc_mri_nonlinear_saturation" in readme
    assert "promoted generated saturated golden" in readme
    assert "64 MB HDF5 checkpoint" in commands
    assert "385 MB HDF5 checkpoint is intentionally not" in commands
    assert "25 MB HDF5 checkpoint is intentionally not" in commands
    assert "52 MB HDF5 checkpoint is intentionally not" in commands
    assert "807 MB HDF5 payload" in commands


def test_pipe_hydro_docs_describe_wired_cheap_parity():
    readme = (ROOT / "production" / "README.md").read_text()
    commands = (ROOT / "production" / "commands.md").read_text()

    assert "pipe_hagen_poiseuille_v1" in readme
    assert "pipe_womersley_v1" in readme
    assert "golden comparisons wired for both pipe hydro goldens" in readme
    assert "Nine-run cheap parity batch" in commands
    assert "including `pipe_hagen_poiseuille_v1`" in commands
