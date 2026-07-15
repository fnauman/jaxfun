"""Wall-bounded PCF counterpart of shearpy's ``multiplane_v2`` output.

The channel names and plane layout intentionally match shearpy so analysis code
can consume both files. The coordinate metadata remains honest about the
physical difference: PCF is already in a fixed wall-bounded Cartesian frame,
and means over the wall-normal direction use the Galerkin quadrature weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxfun.integrators.nonlinear import physical_cross

MULTIPLANE_PROFILE_CHANNELS = (
    "u_x",
    "u_y",
    "u_z",
    "b_x",
    "b_y",
    "b_z",
    "omega_x",
    "omega_y",
    "omega_z",
    "j_x",
    "j_y",
    "j_z",
    "emf_x",
    "emf_y",
    "emf_z",
    "ke_x",
    "ke_y",
    "ke_z",
    "me_x",
    "me_y",
    "me_z",
    "reynolds_xx",
    "reynolds_xy",
    "reynolds_xz",
    "reynolds_yy",
    "reynolds_yz",
    "reynolds_zz",
    "maxwell_xx",
    "maxwell_xy",
    "maxwell_xz",
    "maxwell_yy",
    "maxwell_yz",
    "maxwell_zz",
)

_STRESS_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_PROFILE_RECORDS = ("z_profile", "xy", "xz", "yz")
_ROOT_RECORDS = ("time", "tstep")


def pcf_multiplane_profiles(solver: Any, state: Any) -> dict[str, Any]:
    """Compute the 33-channel multiplane bundle for a PCF curl state.

    Velocity and Reynolds channels are perturbations about the Couette base
    flow. Magnetic channels contain the represented field plus any configured
    uniform background. EMF uses the same total-velocity/total-field cross
    product as the induction equation, so profile budgets retain base-shear
    contributions.
    """

    u = tuple(solver._backward_velocity(state.flow.u))
    u_total = tuple(solver.total_velocity_physical(state.flow))
    b_coeff = tuple(solver.update_B_from_A(state.A))
    b_fluctuation = tuple(solver._backward_B(b_coeff, padded=False))
    background = tuple(getattr(solver, "background_b", (0.0, 0.0, 0.0)))
    b = tuple(
        component + background[index] for index, component in enumerate(b_fluctuation)
    )
    omega = tuple(solver.velocity_vorticity_physical(state.flow.u))
    current_coeff = tuple(solver.update_J_from_B(b_coeff))
    current = tuple(solver._backward_J(current_coeff, padded=False))
    emf = tuple(physical_cross(u_total, b))
    kinetic = tuple(0.5 * component * component for component in u)
    magnetic = tuple(0.5 * component * component for component in b)
    reynolds = tuple(u[i] * u[j] for i, j in _STRESS_PAIRS)
    maxwell = tuple(-b[i] * b[j] for i, j in _STRESS_PAIRS)
    channels = jnp.stack(
        (*u, *b, *omega, *current, *emf, *kinetic, *magnetic, *reynolds, *maxwell)
    )

    x, y, z = solver.X
    x_axis = jnp.real(x[:, 0, 0])
    y_axis = jnp.real(y[0, :, 0])
    z_axis = jnp.real(z[0, 0, :])
    wx = jnp.asarray(solver.TD.basespaces[0].integration_weights())
    wx = wx / jnp.sum(wx)
    mean_x = jnp.einsum("x,cxyz->cyz", wx, channels)

    return {
        "schema": "multiplane_v2",
        "channels": MULTIPLANE_PROFILE_CHANNELS,
        "x": np.asarray(x_axis),
        "y": np.asarray(y_axis),
        "z": np.asarray(z_axis),
        "z_profile": np.asarray(jnp.mean(mean_x, axis=1)),
        "xy": np.asarray(jnp.mean(channels, axis=3)),
        "xz": np.asarray(jnp.mean(channels, axis=2)),
        "yz": np.asarray(mean_x),
    }


def _append_row(group: Any, name: str, value: Any) -> None:
    data = np.asarray(value)
    if name not in group:
        group.create_dataset(
            name,
            shape=(0, *data.shape),
            maxshape=(None, *data.shape),
            chunks=(1, *data.shape),
            dtype=data.dtype,
        )
    dataset = group[name]
    if tuple(dataset.shape[1:]) != tuple(data.shape):
        raise ValueError(
            f"profile dataset {name} changed shape from {dataset.shape[1:]} "
            f"to {data.shape}"
        )
    dataset.resize((dataset.shape[0] + 1, *dataset.shape[1:]))
    dataset[-1] = data


def _axis(group: Any, name: str, value: Any) -> None:
    data = np.asarray(value)
    if name in group:
        if group[name].shape != data.shape or not np.array_equal(
            group[name][...], data
        ):
            raise ValueError(f"profile axis {name} changed")
        return
    group.create_dataset(name, data=data)


def _validate_record_layout(handle: Any) -> int:
    group = handle.get("multiplane_profiles")
    present = {name for name in _ROOT_RECORDS if name in handle}
    if group is not None:
        present.update(name for name in _PROFILE_RECORDS if name in group)
    expected = {*_ROOT_RECORDS, *_PROFILE_RECORDS}
    if not present:
        return 0
    if present != expected:
        missing = sorted(expected - present)
        raise ValueError(f"incomplete multiplane record layout; missing {missing}")
    lengths = [int(handle[name].shape[0]) for name in _ROOT_RECORDS]
    lengths.extend(int(group[name].shape[0]) for name in _PROFILE_RECORDS)
    if len(set(lengths)) != 1:
        raise ValueError(
            f"multiplane record datasets have inconsistent lengths {lengths}"
        )
    steps = np.asarray(handle["tstep"][...], dtype=np.int64)
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("multiplane tsteps must be strictly increasing")
    return lengths[0]


def _resize_records(handle: Any, length: int) -> None:
    group = handle["multiplane_profiles"]
    for name in _ROOT_RECORDS:
        dataset = handle[name]
        dataset.resize((length, *dataset.shape[1:]))
    for name in _PROFILE_RECORDS:
        dataset = group[name]
        dataset.resize((length, *dataset.shape[1:]))


def truncate_pcf_multiplane_h5(path: str | Path, *, after_tstep: int) -> Path:
    """Discard profile records newer than a resume checkpoint."""

    import h5py

    path = Path(path)
    if not path.exists():
        return path
    with h5py.File(path, "r+") as handle:
        count = _validate_record_layout(handle)
        if count == 0:
            return path
        steps = np.asarray(handle["tstep"][...], dtype=np.int64)
        keep = int(np.searchsorted(steps, int(after_tstep), side="right"))
        if keep < count:
            _resize_records(handle, keep)
    return path


def write_pcf_multiplane_h5(
    path: str | Path, *, profiles: dict[str, Any], t: float, tstep: int
) -> Path:
    """Append or replace one profile record while keeping tsteps monotonic."""

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        count = _validate_record_layout(handle)
        meta = handle.require_group("meta")
        group = handle.require_group("multiplane_profiles")
        channels = tuple(profiles["channels"])
        if channels != MULTIPLANE_PROFILE_CHANNELS:
            raise ValueError("multiplane profile channels do not match the v2 schema")
        encoded = json.dumps(channels)
        previous = group.attrs.get("channels_json")
        if previous is not None and previous != encoded:
            raise ValueError("multiplane profile channel metadata changed")
        group.attrs["channels_json"] = encoded
        group.attrs["xy_convention"] = "arithmetic_mean_over_periodic_z"
        group.attrs["xz_convention"] = "arithmetic_mean_over_periodic_y"
        group.attrs["yz_convention"] = "galerkin_quadrature_mean_over_wall_normal_x"
        group.attrs["z_profile_convention"] = (
            "galerkin_quadrature_mean_over_x_then_arithmetic_mean_over_y"
        )
        meta.attrs["format_version"] = 2
        meta.attrs["coordinate_frame"] = "wall_bounded_cartesian"
        meta.attrs["velocity_convention"] = "perturbation_about_Uy=-S*x"
        meta.attrs["magnetic_convention"] = "total_represented_field"
        meta.attrs["emf_convention"] = "total_velocity_cross_total_represented_field"

        for axis in ("x", "y", "z"):
            _axis(group, axis, profiles[axis])

        if count:
            steps = np.asarray(handle["tstep"][...], dtype=np.int64)
            insertion = int(np.searchsorted(steps, int(tstep), side="left"))
            if insertion < count:
                _resize_records(handle, insertion)

        for plane in _PROFILE_RECORDS:
            _append_row(group, plane, profiles[plane])
        _append_row(handle, "time", np.asarray(float(t)))
        _append_row(handle, "tstep", np.asarray(int(tstep), dtype=np.int64))
    return path
