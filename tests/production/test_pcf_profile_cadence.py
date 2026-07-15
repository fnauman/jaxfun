"""Focused cadence regression for PCF multiplane output."""

from __future__ import annotations

from pathlib import Path

import production.oracles as oracles
import production.profiles as profiles_module
from jaxfun.io import run_with_cadence


class _CountingSolver:
    dt = 1.0

    @staticmethod
    def solve(state, steps):
        return int(state) + int(steps)

    def solve_with_cadence(
        self,
        state,
        steps,
        cadence,
        *,
        block_size=1,
        on_diagnostics=None,
        on_snapshot=None,
        on_checkpoint=None,
        should_stop=None,
        t0=0.0,
        tstep0=0,
    ):
        return run_with_cadence(
            self.solve,
            state,
            steps=steps,
            dt=self.dt,
            cadence=cadence,
            block_size=block_size,
            on_diagnostics=on_diagnostics,
            on_snapshot=on_snapshot,
            on_checkpoint=on_checkpoint,
            should_stop=should_stop,
            t0=t0,
            tstep0=tstep0,
        )


def test_empty_gcd_output_tick_does_not_repeat_divergence_guard(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[int] = []

    def record_guard(_solver, _state, *, tstep, **_kwargs):
        calls.append(int(tstep))

    monkeypatch.setattr(oracles, "_raise_on_divergence_drift", record_guard)
    monkeypatch.setattr(oracles, "_snapshot_payload", lambda *_args: ({}, {}))
    monkeypatch.setattr(
        oracles, "_write_atomic_uniform_snapshot", lambda *_a, **_k: None
    )
    monkeypatch.setattr(profiles_module, "pcf_multiplane_profiles", lambda *_a: {})
    monkeypatch.setattr(
        profiles_module, "write_pcf_multiplane_h5", lambda *_a, **_k: None
    )

    out = oracles._solve_with_optional_checkpoints(
        _CountingSolver(),
        0,
        8,
        spec={"problem_id": "unit", "spec_hash": "hash"},
        out_dir=tmp_path,
        checkpoint_every=None,
        snapshot_every=4,
        profiles_every=6,
        diagnostics_every=None,
        state_kind="unit",
    )

    assert out == 8
    assert calls.count(2) == 1
    assert calls.count(4) == 2
    assert calls.count(6) == 2
