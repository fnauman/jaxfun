"""Wall-bounded PCF counterpart of shearpy's ``multiplane_v2`` output.

The channel names and plane layout intentionally match shearpy so analysis code
can consume both files.  The coordinate metadata remains honest about the
physical difference: PCF is already in a fixed wall-bounded Cartesian frame,
and means over the wall-normal direction use the Galerkin quadrature weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

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


def _curl_from_coefficients(
    coefficients: tuple[Any, Any, Any],
    spaces: tuple[Any, Any, Any],
    counts: tuple[int, ...],
) -> tuple[Any, Any, Any]:
    """Evaluate a spectral curl on the standard PCF output mesh."""

    d = lambda component, derivative: spaces[component].backward_primitive(  # noqa: E731
        coefficients[component], derivative, N=counts
    )
    return (
        d(2, (0, 1, 0)) - d(1, (0, 0, 1)),
        d(0, (0, 0, 1)) - d(2, (1, 0, 0)),
        d(1, (1, 0, 0)) - d(0, (0, 1, 0)),
    )


def pcf_multiplane_profiles(solver: Any, state: Any) -> dict[str, Any]:
    """Compute the 33-channel multiplane bundle for a PCF curl state.

    Velocity is the perturbation about ``U_y=-S*x`` and magnetic field is the
    represented total field (the configured uniform background is included).
    This matches shearpy's configured-fluctuation EMF convention while keeping
    mean-flux diagnostics physically meaningful for net-flux runs.
    """

    counts = tuple(int(value) for value in solver.TD.num_quad_points)
    u = tuple(solver._backward_velocity(state.flow.u))
    b_coeff = tuple(solver.update_B_from_A(state.A))
    b = tuple(solver._total_B_physical(b_coeff, padded=False))
    omega = _curl_from_coefficients(
        tuple(state.flow.u), (solver.TB, solver.TD, solver.TD), counts
    )
    current = _curl_from_coefficients(b_coeff, tuple(solver.b_coeff_spaces), counts)
    emf = (
        u[1] * b[2] - u[2] * b[1],
        u[2] * b[0] - u[0] * b[2],
        u[0] * b[1] - u[1] * b[0],
    )
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


def write_pcf_multiplane_h5(
    path: str | Path, *, profiles: dict[str, Any], t: float, tstep: int
) -> Path:
    """Append one PCF multiplane record using the shearpy-compatible layout."""

    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
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

        for axis in ("x", "y", "z"):
            _axis(group, axis, profiles[axis])
        for plane in ("z_profile", "xy", "xz", "yz"):
            _append_row(group, plane, profiles[plane])
        _append_row(handle, "time", np.asarray(float(t)))
        _append_row(handle, "tstep", np.asarray(int(tstep), dtype=np.int64))
    return path
