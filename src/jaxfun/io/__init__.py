"""Optional HDF5 checkpoint and snapshot helpers.

The routines in this module keep IO on the host and outside jitted code.  They
mirror the parts of shenfun's checkpoint/file helpers used by the Couette
examples: exact coefficient checkpoints, physical snapshots on output meshes,
and a compact XDMF sidecar for visualization tools.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CheckpointRecord:
    """Checkpoint payload read from an HDF5 file."""

    fields: dict[str, Any]
    t: float
    tstep: int
    attrs: dict[str, Any]


def _h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when extra absent
        raise ModuleNotFoundError(
            "jaxfun.io HDF5 support requires h5py; install with `jaxfun[io]`."
        ) from exc
    return h5py


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _replace_child(group: Any, name: str) -> None:
    if name in group:
        del group[name]


def _write_dataset(group: Any, name: str, value: Any) -> None:
    _replace_child(group, name)
    group.create_dataset(name, data=_host_array(value))


def _write_tree(group: Any, name: str, value: Any) -> None:
    if isinstance(value, Mapping):
        _replace_child(group, name)
        child = group.create_group(name)
        child.attrs["__kind__"] = "dict"
        for key, item in value.items():
            _write_tree(child, str(key), item)
        return
    if isinstance(value, tuple):
        _replace_child(group, name)
        child = group.create_group(name)
        child.attrs["__kind__"] = "tuple"
        child.attrs["__length__"] = len(value)
        for idx, item in enumerate(value):
            _write_tree(child, str(idx), item)
        return
    if isinstance(value, list):
        _replace_child(group, name)
        child = group.create_group(name)
        child.attrs["__kind__"] = "list"
        child.attrs["__length__"] = len(value)
        for idx, item in enumerate(value):
            _write_tree(child, str(idx), item)
        return
    _write_dataset(group, name, value)


def _read_tree(node: Any) -> Any:
    if hasattr(node, "keys"):
        kind = node.attrs.get("__kind__", "dict")
        if isinstance(kind, bytes):
            kind = kind.decode()
        if kind == "tuple":
            length = int(node.attrs["__length__"])
            return tuple(_read_tree(node[str(i)]) for i in range(length))
        if kind == "list":
            length = int(node.attrs["__length__"])
            return [_read_tree(node[str(i)]) for i in range(length)]
        return {key: _read_tree(node[key]) for key in node}
    return jnp.asarray(node[()])


@dataclass(frozen=True)
class Cadence:
    """Host-side cadence for diagnostics and IO callbacks."""

    diagnostics_every: int | None = None
    snapshot_every: int | None = None
    checkpoint_every: int | None = None

    def due_steps(self) -> tuple[int, ...]:
        return tuple(
            int(every)
            for every in (
                self.diagnostics_every,
                self.snapshot_every,
                self.checkpoint_every,
            )
            if every is not None and every > 0
        )


def cadence_due(tstep: int, every: int | None) -> bool:
    """Return whether a positive cadence divides ``tstep``."""
    return every is not None and every > 0 and tstep % every == 0


def _steps_until_next_due(tstep: int, final_tstep: int, cadence: Cadence) -> int:
    candidates = [final_tstep - tstep]
    for every in cadence.due_steps():
        next_due = ((tstep // every) + 1) * every
        candidates.append(max(1, next_due - tstep))
    return min(candidates)


def _block_until_ready(value: Any) -> Any:
    def ready(leaf: Any) -> Any:
        if hasattr(leaf, "block_until_ready"):
            return leaf.block_until_ready()
        return leaf

    return jax.tree.map(ready, value)


def run_with_cadence(
    advance: Any,
    state: Any,
    *,
    steps: int,
    dt: float,
    cadence: Cadence,
    block_size: int = 1,
    t0: float = 0.0,
    tstep0: int = 0,
    diagnostics: Any | None = None,
    on_diagnostics: Any | None = None,
    on_snapshot: Any | None = None,
    on_checkpoint: Any | None = None,
    should_stop: Any | None = None,
) -> Any:
    """Advance in compiled blocks and run host callbacks at cadence boundaries.

    ``advance(state, nsteps)`` should contain the jitted stepping work.  This
    function deliberately stays on the host: after each block it waits for the
    device result, computes optional diagnostics, and invokes IO callbacks.
    If ``should_stop`` is provided, it is called as
    ``should_stop(t, tstep, state)`` after due callbacks and may return true to
    stop before another compiled block is launched.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    out = state
    tstep = int(tstep0)
    final_tstep = tstep + int(steps)
    while tstep < final_tstep:
        nsteps = min(block_size, _steps_until_next_due(tstep, final_tstep, cadence))
        out = _block_until_ready(advance(out, nsteps))
        tstep += nsteps
        t = float(t0) + (tstep - int(tstep0)) * float(dt)
        if cadence_due(tstep, cadence.diagnostics_every) and on_diagnostics:
            diag = diagnostics(out) if diagnostics is not None else None
            on_diagnostics(t, tstep, diag)
        if cadence_due(tstep, cadence.snapshot_every) and on_snapshot:
            on_snapshot(t, tstep, out)
        if cadence_due(tstep, cadence.checkpoint_every) and on_checkpoint:
            on_checkpoint(t, tstep, out)
        if should_stop is not None and bool(should_stop(t, tstep, out)):
            break
    return out


def write_checkpoint(
    filename: str | Path,
    fields: Mapping[str, Any],
    *,
    t: float,
    tstep: int,
    attrs: Mapping[str, Any] | None = None,
    mode: str = "a",
) -> None:
    """Write an exact coefficient-space checkpoint.

    Fields may be arrays or nested dict/list/tuple pytrees.  The latest step is
    tracked in root attributes so :func:`read_checkpoint` can restart from the
    newest checkpoint by default.
    """
    h5py = _h5py()
    with h5py.File(filename, mode) as h5:
        root = h5.require_group("checkpoints")
        step_name = str(int(tstep))
        _replace_child(root, step_name)
        step_group = root.create_group(step_name)
        step_group.attrs["t"] = float(t)
        step_group.attrs["tstep"] = int(tstep)
        if attrs:
            for key, value in attrs.items():
                step_group.attrs[str(key)] = value
        fields_group = step_group.create_group("fields")
        for name, value in fields.items():
            _write_tree(fields_group, str(name), value)
        root.attrs["latest_step"] = int(tstep)
        h5.attrs["t"] = float(t)
        h5.attrs["tstep"] = int(tstep)


def read_checkpoint(
    filename: str | Path, *, step: int | None = None
) -> CheckpointRecord:
    """Read a coefficient-space checkpoint from HDF5."""
    h5py = _h5py()
    with h5py.File(filename, "r") as h5:
        root = h5["checkpoints"]
        if step is None:
            step = int(root.attrs.get("latest_step", max(int(k) for k in root)))
        step_group = root[str(int(step))]
        fields = {
            name: _read_tree(step_group["fields"][name])
            for name in step_group["fields"]
        }
        attrs = {key: step_group.attrs[key] for key in step_group.attrs}
        return CheckpointRecord(
            fields=fields,
            t=float(step_group.attrs["t"]),
            tstep=int(step_group.attrs["tstep"]),
            attrs=attrs,
        )


def _field_uniform_values(value: Any, space: Any = None, N: Any = None) -> Any:
    if hasattr(value, "functionspace") and hasattr(value, "array"):
        functionspace = value.functionspace
        if hasattr(value, "evaluate_mesh"):
            return value.evaluate_mesh(kind="uniform", N=N)
        return functionspace.backward(value.array, N=N, kind="uniform")
    if hasattr(value, "functionspace") and hasattr(value, "values"):
        return value.backward()
    if space is not None:
        if hasattr(space, "evaluate_mesh"):
            return space.evaluate_mesh(value, kind="uniform", N=N)
        return space.backward(value, N=N)
    return value


def write_uniform_snapshot(
    filename: str | Path,
    fields: Mapping[str, Any],
    *,
    t: float,
    tstep: int,
    spaces: Mapping[str, Any] | None = None,
    N: Any = None,
    attrs: Mapping[str, Any] | None = None,
    mode: str = "a",
) -> None:
    """Write physical-field snapshots on uniform output meshes.

    A field may be a raw physical array, a :class:`jaxfun.galerkin.JAXFunction`,
    a physical ``Array`` wrapper, or coefficient data with a matching entry in
    ``spaces``.  Nested vector fields are stored as HDF5 groups.
    """
    h5py = _h5py()
    with h5py.File(filename, mode) as h5:
        root = h5.require_group("snapshots")
        step_name = str(int(tstep))
        _replace_child(root, step_name)
        step_group = root.create_group(step_name)
        step_group.attrs["t"] = float(t)
        step_group.attrs["tstep"] = int(tstep)
        if attrs:
            for key, value in attrs.items():
                step_group.attrs[str(key)] = value
        fields_group = step_group.create_group("fields")
        spaces = {} if spaces is None else spaces
        for name, value in fields.items():
            counts = N.get(name) if isinstance(N, Mapping) else N
            physical = _field_uniform_values(value, spaces.get(name), counts)
            _write_tree(fields_group, str(name), physical)
        root.attrs["latest_step"] = int(tstep)


def _iter_datasets(group: Any, prefix: str = ""):
    for key in group:
        node = group[key]
        name = f"{prefix}/{key}" if prefix else key
        if hasattr(node, "keys"):
            yield from _iter_datasets(node, name)
        else:
            yield name, node


def _xdmf_number_type(dtype: np.dtype) -> tuple[str, int]:
    if np.issubdtype(dtype, np.integer):
        return "Int", int(dtype.itemsize)
    return "Float", int(dtype.itemsize)


def _grid_xml(shape: tuple[int, ...]) -> list[str]:
    if len(shape) == 1:
        coords = " ".join(str(i) for i in range(shape[0]))
        return [
            f'<Topology TopologyType="Polyvertex" NumberOfElements="{shape[0]}"/>',
            '<Geometry GeometryType="X">',
            (
                f'<DataItem Dimensions="{shape[0]}" NumberType="Float" '
                f'Format="XML">{coords}</DataItem>'
            ),
            '</Geometry>',
        ]
    if len(shape) == 2:
        dims = f"{shape[0]} {shape[1]}"
        topology = "2DCoRectMesh"
        geom_dims = 2
    elif len(shape) == 3:
        dims = f"{shape[0]} {shape[1]} {shape[2]}"
        topology = "3DCoRectMesh"
        geom_dims = 3
    else:
        n = int(np.prod(shape))
        return [
            f'<Topology TopologyType="Polyvertex" NumberOfElements="{n}"/>',
            '<Geometry GeometryType="X">',
            f'<DataItem Dimensions="{n}" NumberType="Float" Format="XML">0</DataItem>',
            '</Geometry>',
        ]
    zeros = " ".join("0" for _ in range(geom_dims))
    ones = " ".join("1" for _ in range(geom_dims))
    return [
        f'<Topology TopologyType="{topology}" Dimensions="{dims}"/>',
        '<Geometry GeometryType="ORIGIN_DXDYDZ">',
        f'<DataItem Dimensions="{geom_dims}" Format="XML">{zeros}</DataItem>',
        f'<DataItem Dimensions="{geom_dims}" Format="XML">{ones}</DataItem>',
        '</Geometry>',
    ]


def generate_xdmf(
    h5_filename: str | Path,
    xdmf_filename: str | Path | None = None,
    *,
    snapshot_group: str = "snapshots",
) -> Path:
    """Generate an XDMF sidecar for uniform HDF5 snapshots."""
    h5py = _h5py()
    h5_path = Path(h5_filename)
    xdmf_path = (
        h5_path.with_suffix(".xdmf")
        if xdmf_filename is None
        else Path(xdmf_filename)
    )
    lines = [
        '<?xml version="1.0" ?>',
        '<Xdmf Version="3.0">',
        '<Domain>',
        (
            '<Grid Name="jaxfun_snapshots" GridType="Collection" '
            'CollectionType="Temporal">'
        ),
    ]
    with h5py.File(h5_path, "r") as h5:
        root = h5[snapshot_group]
        for step_name in sorted(root.keys(), key=lambda key: int(key)):
            step = root[step_name]
            lines.append(f'<Grid Name="step_{step_name}" GridType="Uniform">')
            lines.append(f'<Time Value="{float(step.attrs["t"]):.17g}"/>')
            datasets = list(_iter_datasets(step["fields"]))
            if not datasets:
                lines.append('</Grid>')
                continue
            first = np.asarray(datasets[0][1])
            lines.extend(_grid_xml(tuple(first.shape)))
            for name, dataset in datasets:
                arr = np.asarray(dataset)
                if np.iscomplexobj(arr):
                    continue
                number_type, precision = _xdmf_number_type(arr.dtype)
                dims = " ".join(str(n) for n in arr.shape)
                dataset_path = f"/{snapshot_group}/{step_name}/fields/{name}"
                lines.extend(
                    [
                        (
                            f'<Attribute Name="{name}" AttributeType="Scalar" '
                            f'Center="Node">'
                        ),
                        (
                            f'<DataItem Dimensions="{dims}" NumberType="{number_type}" '
                            f'Precision="{precision}" Format="HDF">'
                            f'{h5_path.name}:{dataset_path}</DataItem>'
                        ),
                        '</Attribute>',
                    ]
                )
            lines.append('</Grid>')
    lines.extend(['</Grid>', '</Domain>', '</Xdmf>', ''])
    xdmf_path.write_text("\n".join(lines))
    return xdmf_path
