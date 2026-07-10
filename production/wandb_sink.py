"""Optional Weights & Biases sink (FJ-07).

The local ``diagnostics.jsonl`` / ``metadata.json`` / summaries / checkpoints remain
the source of truth. W&B is an *optional* mirror: it is import-tolerant (a run with
``wandb`` uninstalled or logging disabled behaves identically minus the mirror), it
is only ever called from the host-side cadence callback (never inside JAX tracing),
and it logs the complete canonical cadence dictionary plus a run summary.

Enable by constructing :class:`WandbSink` with ``enabled=True`` (typically gated on a
CLI flag). Offline runs use ``WANDB_MODE=offline`` and a later ``wandb sync``.
"""

from __future__ import annotations

from typing import Any


class WandbSink:
    """A no-op-by-default mirror of the cadence stream and run summary to W&B."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        run_id: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None
        self._wandb = None
        self._closed = False
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            # Uninstalled / broken install -> silently degrade to a local-only run.
            self.enabled = False
            return
        self._wandb = wandb
        init_kwargs: dict[str, Any] = {"project": project, "config": config or {}}
        if entity is not None:
            init_kwargs["entity"] = entity
        if group is not None:
            init_kwargs["group"] = group
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        if mode is not None:
            init_kwargs["mode"] = mode
        try:
            self._run = wandb.init(**init_kwargs)
        except Exception:
            self.enabled = False
            self._run = None

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._run is not None)

    def log_cadence(self, row: dict[str, Any]) -> None:
        """Log one cadence row (the complete canonical dictionary), keyed by step time.

        Must be called only from the host-side callback, never inside JAX tracing.
        """

        if not self.active:
            return
        payload = {k: v for k, v in row.items() if _is_scalarish(v)}
        step = row.get("t")
        try:
            self._run.log(payload)  # type: ignore[union-attr]
        except Exception:
            pass
        _ = step  # step is included in payload as "t"

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Populate the run summary (status, class, growth, stresses, cost, ...)."""

        if not self.active:
            return
        try:
            for key, value in summary.items():
                if _is_scalarish(value) or isinstance(value, (str, bool)):
                    self._run.summary[key] = value  # type: ignore[union-attr]
        except Exception:
            pass

    def finish(self, exit_code: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._run is not None:
            try:
                self._run.finish(exit_code=exit_code)
            except Exception:
                pass

    def __enter__(self) -> WandbSink:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(exit_code=0 if exc_type is None else 1)


def _is_scalarish(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
