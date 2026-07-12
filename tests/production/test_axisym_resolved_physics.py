"""FJ-00/FJ-01 on the axisymmetric primitive DNS path (review blocker 4).

The axisymmetric runner consumed ``groups["nu"]`` directly and hardcoded
``dealias=1.0``: a valid Re/Rm-only spec passed schema validation and then died
at execution with ``KeyError: 'nu'``, and the spec's dealias contract was
silently ignored. Both matter for the Phase-2 axisymmetric confirmation runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import pytest

from production.oracles import ProductionOracleNotImplementedError, run_supported_spec
from production.problem_spec import load_spec

ROOT = Path(__file__).resolve().parents[2]


def _mri_axisym_spec(tmp_path, mutate):
    data = json.loads(
        (ROOT / "production" / "examples" / "pcf_mri_primitive_dns_v1.json").read_text(
            encoding="utf-8"
        )
    )
    data["resolution"] = {**data["resolution"], "Nx": 24, "Nz": 8}
    mutate(data)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_spec(path)


def test_re_rm_only_axisym_spec_executes(tmp_path):
    """A spec stating only Re/Rm (no nu/eta_mag) must resolve and run (FJ-00)."""
    jax.config.update("jax_enable_x64", True)

    def strip_coefficients(data):
        del data["nondimensional_groups"]["nu"]
        del data["nondimensional_groups"]["eta_mag"]

    spec = _mri_axisym_spec(tmp_path, strip_coefficients)
    out = run_supported_spec(spec, steps=2)
    sc = out["scalars"]
    assert math.isfinite(sc["kinetic_energy"])
    assert math.isfinite(sc["magnetic_energy"])
    assert sc["energy_convention"] == "half_integral_abs2"


def test_axisym_spec_dealias_is_honored_not_hardcoded(tmp_path):
    """FJ-01: the spec's dealias reaches the 2-D solver; anisotropy fails loudly."""
    jax.config.update("jax_enable_x64", True)

    def anisotropic(data):
        data["resolution"]["dealias"] = {"x": 1.0, "y": 1.5, "z": 1.5}

    spec = _mri_axisym_spec(tmp_path, anisotropic)
    with pytest.raises(ProductionOracleNotImplementedError, match="uniform dealias"):
        run_supported_spec(spec, steps=1)

    def uniform(data):
        data["resolution"]["dealias"] = 1.5

    spec = _mri_axisym_spec(tmp_path, uniform)
    out = run_supported_spec(spec, steps=1)
    assert math.isfinite(out["scalars"]["kinetic_energy"])


def test_axisym_runner_rejects_pseudo_vacuum_loudly(tmp_path):
    """The 2-D solver is conducting-only; a pseudo-vacuum spec must not silently
    run with the wrong wall (FJ-09)."""
    jax.config.update("jax_enable_x64", True)

    def pseudo_vacuum(data):
        data["boundary_conditions"]["magnetic"] = {"type": "pseudo_vacuum"}

    spec = _mri_axisym_spec(tmp_path, pseudo_vacuum)
    with pytest.raises(ProductionOracleNotImplementedError, match="conducting"):
        run_supported_spec(spec, steps=1)
