"""FJ-07: W&B sink -- strict when requested, degrades cleanly when not."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from production.wandb_sink import WandbSink, WandbUnavailableError

ROOT = Path(__file__).resolve().parents[2]


def test_disabled_sink_is_inert():
    sink = WandbSink(enabled=False, project="p")
    assert sink.active is False
    # All operations are safe no-ops.
    sink.log_cadence({"t": 0.0, "kinetic_energy": 1.0})
    sink.log_summary({"scientific_class": "growing"})
    sink.finish()


def test_context_manager_is_inert_when_disabled():
    with WandbSink(enabled=False) as sink:
        sink.log_cadence({"t": 1.0, "total_stress": 0.1})
    assert sink.active is False


def _block_wandb_import(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_enabled_without_wandb_installed_degrades(monkeypatch):
    _block_wandb_import(monkeypatch)
    sink = WandbSink(enabled=True, project="p")
    assert sink.enabled is False and sink.active is False
    sink.log_cadence({"t": 0.0})  # still safe
    sink.finish()


def test_strict_without_wandb_installed_raises(monkeypatch):
    """Review fix: an explicit --wandb must not silently disable tracking."""
    _block_wandb_import(monkeypatch)
    with pytest.raises(WandbUnavailableError, match="wandb"):
        WandbSink(enabled=True, strict=True, project="p")


class _FakeRun:
    def __init__(self):
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.finished: list[int] = []

    def log(self, payload):
        self.logged.append(dict(payload))

    def finish(self, exit_code=0):
        self.finished.append(int(exit_code))


def _install_fake_wandb(monkeypatch):
    run = _FakeRun()
    fake = types.ModuleType("wandb")
    fake.init = lambda **kwargs: run
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return run


def test_strict_init_failure_raises(monkeypatch):
    fake = types.ModuleType("wandb")

    def broken_init(**kwargs):
        raise RuntimeError("no API key")

    fake.init = broken_init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(WandbUnavailableError, match="wandb.init failed"):
        WandbSink(enabled=True, strict=True, project="p")
    # Non-strict still degrades.
    sink = WandbSink(enabled=True, strict=False, project="p")
    assert sink.active is False


def test_runner_streams_cadence_rows_live_and_finishes_once(monkeypatch, tmp_path):
    """Review fix: --wandb streams rows during the solve, not as a post-run replay.

    The sink is constructed before the solve and receives each canonical cadence
    row via the runner's on_row callback; the summary + finish happen exactly
    once in the runner's finally.
    """
    run = _install_fake_wandb(monkeypatch)

    from production.run_problem import run_problem

    metadata = run_problem(
        config_path=ROOT / "production" / "runs" / "pcf_mhd_divfree.json",
        out=tmp_path / "run",
        steps=2,
        diagnostics_every=1,
        resolution_tier="smoke",
        wandb=True,
    )

    assert metadata["execution"]["status"] == "completed"
    # Cadence rows were streamed (2 steps at diagnostics_every=1).
    assert len(run.logged) >= 2
    assert any("magnetic_energy" in row for row in run.logged)
    # Summary carries the operational status and final scalars; finish exactly once.
    assert run.summary.get("operational_status") == "completed"
    assert run.finished == [0]


def test_runner_records_failure_summary(monkeypatch, tmp_path):
    """A crashing solve still closes the run with exit_code 1 and a failure reason."""
    run = _install_fake_wandb(monkeypatch)

    import production.run_problem as rp

    def boom(*args, **kwargs):
        raise FloatingPointError("nonfinite solver state at tstep=1 t=0.005")

    monkeypatch.setattr(rp, "run_supported_spec", boom)

    with pytest.raises(FloatingPointError):
        rp.run_problem(
            config_path=ROOT / "production" / "runs" / "pcf_mhd_divfree.json",
            out=tmp_path / "run",
            steps=2,
            resolution_tier="smoke",
            wandb=True,
        )

    assert run.finished == [1]
    assert run.summary.get("operational_status") == "nan_inf"
    assert "nonfinite" in str(run.summary.get("failure_reason", ""))
