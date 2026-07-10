"""FJ-07: optional W&B sink degrades cleanly when disabled/uninstalled."""

from __future__ import annotations

from production.wandb_sink import WandbSink


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


def test_enabled_without_wandb_installed_degrades(monkeypatch):
    # Simulate wandb not being importable.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sink = WandbSink(enabled=True, project="p")
    assert sink.enabled is False and sink.active is False
    sink.log_cadence({"t": 0.0})  # still safe
    sink.finish()
