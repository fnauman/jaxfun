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
    assert "exp_pcf_mri_shearbox_growth" in readme
    assert "promoted generated saturated golden" in readme
    assert "385 MB HDF5 checkpoint is intentionally not" in commands
