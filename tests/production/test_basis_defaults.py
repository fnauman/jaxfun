"""Chebyshev is the default basis for every implemented wall-bounded flow."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from examples.channelflow_kmm import KMM
from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax
from examples.taylor_couette_dns_jax import (
    AxisymmetricMRIDNSJax,
    AxisymmetricTCDNSJax,
    TaylorCouetteDNSJax,
    TaylorCouetteMRIDNSJax,
)
from examples.taylor_couette_linear_jax import TaylorCouetteLinearJax
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax
from production.problem_spec import DEFAULT_FLOW_BASIS_FAMILY, validate_spec

ROOT = Path(__file__).resolve().parents[2]


def test_omitted_production_family_materializes_chebyshev_but_explicit_legendre_stays():
    raw = json.loads(
        (ROOT / "production/examples/channel_poiseuille_hydro_v1.json").read_text()
    )
    raw["resolution"].pop("family")
    assert validate_spec(raw)["resolution"]["family"] == "C"

    raw["resolution"]["family"] = "L"
    assert validate_spec(raw)["resolution"]["family"] == "L"


def test_invalid_production_basis_family_is_rejected():
    raw = json.loads(
        (ROOT / "production/examples/channel_poiseuille_hydro_v1.json").read_text()
    )
    raw["resolution"]["family"] = "unsupported"
    try:
        validate_spec(raw)
    except ValueError as exc:
        assert "Chebyshev" in str(exc) and "Legendre" in str(exc)
    else:  # pragma: no cover - must reject before solver construction
        raise AssertionError("unsupported basis family was accepted")


def test_shipped_production_runs_select_chebyshev():
    assert DEFAULT_FLOW_BASIS_FAMILY == "C"
    for path in sorted((ROOT / "production/runs").glob("*.json")):
        raw = json.loads(path.read_text())
        assert raw["resolution"].get("family", DEFAULT_FLOW_BASIS_FAMILY) == "C", path


def test_primary_jax_flow_constructors_default_to_chebyshev():
    classes = (
        KMM,
        PlaneCouetteFluctuationJax,
        PlaneCouetteMHDJax,
        PlaneCouetteMRIShearpyJax,
        TaylorCouetteLinearJax,
        AxisymmetricTCDNSJax,
        TaylorCouetteDNSJax,
        AxisymmetricMRIDNSJax,
        TaylorCouetteMRIDNSJax,
        TaylorCouetteMRIJax,
        TaylorCouetteVPMRIDNSJax,
    )
    for cls in classes:
        assert inspect.signature(cls).parameters["family"].default == "C", cls


def test_flow_sources_have_no_implicit_legendre_default_or_cli_default():
    sources = (
        "examples/channelflow_kmm.py",
        "examples/pcf_fluctuations_jax.py",
        "examples/pcf_fluctuations_divv_jax.py",
        "examples/pcf_mhd_jax.py",
        "examples/pcf_mhd_mri_shearpy_jax.py",
        "examples/taylor_couette_dns_jax.py",
        "examples/taylor_couette_linear_jax.py",
        "examples/taylor_couette_mri_jax.py",
        "examples/taylor_couette_vp_jax.py",
        "couette/taylor_couette_dns.py",
        "couette/taylor_couette_linear.py",
        "couette/taylor_couette_mri.py",
    )
    forbidden = ('family="L"', 'family: str = "L"', 'default="L"')
    for relative in sources:
        text = (ROOT / relative).read_text()
        assert not any(token in text for token in forbidden), relative
