"""Production checkpoint wrapper over :mod:`jaxfun.io`.

The low-level IO layer stores coefficient pytrees exactly. This module adds the
metadata envelope required by the production runner so restarts carry the same
contract, dtype, device, and diagnostics context as ``metadata.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_SCHEMA_VERSION = 1
SOLVER_SCHEMA_VERSION = 1


def write_production_checkpoint(
    path: str | Path,
    fields: Mapping[str, Any],
    *,
    t: float,
    tstep: int,
    spec: dict[str, Any],
    state_kind: str,
    device_record: Mapping[str, Any] | None = None,
    diagnostics_path: str | Path | None = None,
    prng_state: Any | None = None,
    mode: str = "a",
) -> None:
    """Atomically write the latest production restart checkpoint.

    Production resume only consumes the latest step, so the wrapper deliberately
    rewrites a compact single-step HDF5 file instead of copy-appending an
    ever-growing checkpoint history at each interval.
    """

    from jaxfun.io import write_checkpoint

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    attrs = production_checkpoint_attrs(
        spec,
        state_kind=state_kind,
        fields=fields,
        device_record=device_record,
        diagnostics_path=diagnostics_path,
        prng_state=prng_state,
    )
    try:
        write_mode = "w" if mode == "a" else mode
        write_checkpoint(
            tmp,
            fields,
            t=t,
            tstep=tstep,
            attrs=attrs,
            mode=write_mode,
        )
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def production_checkpoint_attrs(
    spec: dict[str, Any],
    *,
    state_kind: str,
    fields: Mapping[str, Any],
    device_record: Mapping[str, Any] | None = None,
    diagnostics_path: str | Path | None = None,
    prng_state: Any | None = None,
) -> dict[str, Any]:
    """Return flat HDF5-safe attrs for a production checkpoint."""

    production_dtype = str(
        _device_value(device_record, "production_run_dtype")
        or _production_dtype_from_fields(fields)
        or os.environ.get("JAXFUN_PRODUCTION_DTYPE", "float32")
    )
    jax_enable_x64 = _device_value(device_record, "jax_enable_x64")
    if jax_enable_x64 is None:
        jax_enable_x64 = production_dtype == "float64"
    dtype_metadata = {
        "field_dtypes": _tree_dtypes(fields),
        "field_shapes": _tree_shapes(fields),
        "production_run_dtype": production_dtype,
        "jax_enable_x64": bool(jax_enable_x64),
    }
    return {
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "solver_schema_version": SOLVER_SCHEMA_VERSION,
        "problem_id": spec["problem_id"],
        "artifact_id": spec["golden"]["artifact_id"],
        "spec_hash": spec["spec_hash"],
        # FJ-01: stamp the numerics contract so a pre-fix checkpoint cannot seed a
        # post-fix production continuation (0 == pre-contract / unstamped).
        "numerics_contract_version": int(spec.get("numerics_contract_version", 0)),
        "state_kind": state_kind,
        "diagnostics_path": "" if diagnostics_path is None else str(diagnostics_path),
        "dtype_metadata_json": _json_attr(dtype_metadata),
        "device_metadata_json": _json_attr(dict(device_record or {})),
        "prng_state_json": "" if prng_state is None else _json_attr(prng_state),
    }


def _production_dtype_from_fields(fields: Mapping[str, Any]) -> str | None:
    dtypes = _tree_dtypes(fields)
    if any(dtype in {"float64", "complex128"} for dtype in dtypes):
        return "float64"
    if any(dtype in {"float32", "complex64"} for dtype in dtypes):
        return "float32"
    return None


def _tree_dtypes(tree: Any) -> list[str]:
    return sorted({str(np.asarray(leaf).dtype) for leaf in _tree_leaves(tree)})


def _tree_shapes(tree: Any) -> list[list[int]]:
    shapes = {
        tuple(int(size) for size in np.asarray(leaf).shape)
        for leaf in _tree_leaves(tree)
    }
    return [list(shape) for shape in sorted(shapes)]


def _tree_leaves(tree: Any) -> list[Any]:
    import jax

    return list(jax.tree_util.tree_leaves(tree))


def _device_value(device_record: Mapping[str, Any] | None, key: str) -> Any:
    if device_record is None:
        return None
    return device_record.get(key)


def _json_attr(value: Any) -> str:
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "SOLVER_SCHEMA_VERSION",
    "production_checkpoint_attrs",
    "write_production_checkpoint",
]
