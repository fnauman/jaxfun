"""Optional Weights & Biases sink (FJ-07).

The local ``diagnostics.jsonl`` / ``metadata.json`` / summaries / checkpoints remain
the source of truth. W&B is an *optional* mirror: it is only ever called from the
host-side cadence callback (never inside JAX tracing), it streams each canonical
cadence row as it is produced (live telemetry for long remote runs), and it logs a
run summary exactly once when the run finishes or fails.

Install it with the declared extra (``pip install .[wandb]``). When tracking is
explicitly requested (``strict=True``, the runner's ``--wandb`` flag) an
uninstalled/broken ``wandb`` raises :class:`WandbUnavailableError` instead of
silently disabling tracking; a non-strict sink degrades to a local-only no-op.
Offline runs use ``WANDB_MODE=offline`` and a later ``wandb sync``.
"""

from __future__ import annotations

from typing import Any


class WandbUnavailableError(RuntimeError):
    """Raised when tracking was explicitly requested but cannot be initialized."""


class WandbSink:
    """A mirror of the cadence stream and run summary to W&B."""

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
        strict: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None
        self._wandb = None
        self._closed = False
        if not self.enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception as exc:
            if strict:
                raise WandbUnavailableError(
                    "W&B tracking was requested (--wandb) but the `wandb` package "
                    "cannot be imported; install the optional dependency "
                    "(`pip install jaxfun[wandb]`) or drop --wandb"
                ) from exc
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
        except Exception as exc:
            if strict:
                raise WandbUnavailableError(
                    f"W&B tracking was requested (--wandb) but wandb.init failed: {exc}"
                ) from exc
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
