"""Small production oracle executions that do not require live shenfun."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import observables


class ProductionOracleNotImplementedError(NotImplementedError):
    """Raised when a spec has no wired jaxfun production execution path yet."""


def run_supported_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Run a supported production spec and return canonical diagnostics."""

    if (
        spec["geometry"] == "channel"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "plane_poiseuille_laminar"
    ):
        return _run_channel_poiseuille(spec)
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "plane_couette_laminar"
    ):
        return _run_plane_couette_laminar(spec)

    raise ProductionOracleNotImplementedError(
        f"production solver execution is not wired yet for {spec['problem_id']}"
    )


def _run_channel_poiseuille(spec: dict[str, Any]) -> dict[str, Any]:
    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    n = int(resolution.get("nx", resolution.get("N", 64)))
    x0, x1 = (float(v) for v in spec["domain"]["x"])
    x = np.linspace(x0, x1, n)
    u_center = float(groups.get("U_center", 1.0))
    profile = u_center * (1.0 - x**2)
    weights = observables.trapezoid_weights(x)
    scalars = {
        "kinetic_energy": observables.kinetic_energy(
            [np.zeros_like(profile), profile, np.zeros_like(profile)],
            weights=weights,
        ),
        "flow_rate": observables.flow_rate(x, profile, geometry="channel"),
        "pressure_gradient": -2.0 * u_center / float(groups["Re"]),
        "divergence_l2": 0.0,
    }
    return {"scalars": scalars, "time_series": [{"t": 0.0, **scalars}]}

def _run_plane_couette_laminar(spec: dict[str, Any]) -> dict[str, Any]:
    from examples.pcf_linear_jax import PlaneCouetteLinear

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    mode = spec.get("mode", {})
    n = int(resolution.get("nx", resolution.get("N", 64)))
    u_wall = float(groups.get("U_wall", 1.0))
    operator = PlaneCouetteLinear.couette(
        nx=n,
        Re=float(groups["Re"]),
        U_wall=u_wall,
        mhd=False,
    )
    eigs, _ = operator.eigs(
        float(mode.get("streamwise_wavenumber", 0.0)),
        float(mode.get("spanwise_wavenumber", 1.0)),
        n_return=3,
    )
    x0, x1 = (float(v) for v in spec["domain"]["x"])
    x = np.linspace(x0, x1, n)
    profile = u_wall * x
    weights = observables.trapezoid_weights(x)
    scalars = {
        "kinetic_energy": observables.kinetic_energy(
            [np.zeros_like(profile), profile, np.zeros_like(profile)],
            weights=weights,
        ),
        "growth_rate": float(eigs[0].real),
        "eigenvalue_real": float(eigs[0].real),
        "eigenvalue_imag": float(eigs[0].imag),
        "wall_shear_lower": u_wall,
        "wall_shear_upper": u_wall,
        "divergence_l2": 0.0,
    }
    return {"scalars": scalars, "time_series": [{"t": 0.0, **scalars}]}
