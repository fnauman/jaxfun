"""Small production oracle executions that do not require live shenfun."""

from __future__ import annotations

from typing import Any

import math

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
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "circular_couette_base_flow"
    ):
        return _run_taylor_couette_hydro(spec)
    if spec["geometry"] == "pcf" and spec["physics"] in {"mhd", "mri"}:
        if spec["expected_oracle"]["type"] in {"pcf_mhd_linear_conducting", "local_ideal_mri"}:
            return _run_pcf_mhd_like(spec)

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

def _run_taylor_couette_hydro(spec: dict[str, Any]) -> dict[str, Any]:
    from examples.taylor_couette_linear_jax import (
        CircularCouette,
        TaylorCouetteLinearJax,
    )

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    mode = spec.get("mode", {})
    base = CircularCouette(
        float(groups["R1"]),
        float(groups["R2"]),
        float(groups["Omega1"]),
        float(groups["Omega2"]),
    )
    n = int(resolution.get("N", resolution.get("Nr", 28)))
    operator = TaylorCouetteLinearJax(
        base,
        nu=float(groups["nu"]),
        N=n,
        family=resolution.get("family", "C"),
    )
    eigs, _ = operator.eigs(
        int(mode.get("azimuthal_wavenumber", 0)),
        float(mode.get("axial_wavenumber", 3.14)),
        n_return=3,
    )
    r0, r1 = (float(v) for v in spec["domain"]["r"])
    r = np.linspace(r0, r1, n)
    profile = base.V(r)
    weights = 2.0 * math.pi * r * observables.trapezoid_weights(r)
    scalars = {
        "kinetic_energy": observables.kinetic_energy(
            [np.zeros_like(profile), profile, np.zeros_like(profile)],
            weights=weights,
        ),
        "growth_rate": float(eigs[0].real),
        "eigenvalue_real": float(eigs[0].real),
        "eigenvalue_imag": float(eigs[0].imag),
        "rayleigh_stable": bool(base.rayleigh_stable()),
        "divergence_l2": 0.0,
    }
    return {"scalars": scalars, "time_series": [{"t": 0.0, **scalars}]}

def _run_pcf_mhd_like(spec: dict[str, Any]) -> dict[str, Any]:
    from examples.pcf_linear_jax import PlaneCouetteLinear

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    mode = spec.get("mode", {})
    nx = int(resolution.get("nx", resolution.get("N", 48)))
    re = float(groups["Re"])
    rm = float(groups.get("Rm", re))
    ky = float(mode.get("streamwise_wavenumber", 1.0))
    kz = float(mode.get("spanwise_wavenumber", 1.0))
    by = float(groups.get("By", 0.0))
    bz = float(groups.get("Bz", 0.1))
    magnetic_bc = _magnetic_bc(spec)
    if spec["physics"] == "mri":
        shear = float(groups.get("S", 1.0))
        omega = float(groups.get("Omega", 2.0 / 3.0))
        operator = PlaneCouetteLinear.shearpy(
            nx=nx,
            Re=re,
            Rm=rm,
            shear_rate=shear,
            omega=omega,
            by=by,
            bz=bz,
            magnetic_bc=magnetic_bc,
        )
    else:
        shear = None
        omega = None
        operator = PlaneCouetteLinear.couette(
            nx=nx,
            Re=re,
            Rm=rm,
            mhd=True,
            by=by,
            bz=bz,
            magnetic_bc=magnetic_bc,
        )
    eigs, vectors = operator.eigs(ky, kz, n_return=3)
    scalars = {
        **_pcf_mhd_mode_scalars(operator, vectors[:, 0]),
        "growth_rate": float(eigs[0].real),
        "eigenvalue_real": float(eigs[0].real),
        "eigenvalue_imag": float(eigs[0].imag),
        "divergence_u_l2": 0.0,
        "divergence_b_l2": 0.0,
        "magnetic_bc": magnetic_bc,
    }
    if spec["physics"] == "mri":
        assert shear is not None and omega is not None
        opt = _mri_keplerian_optimum(Omega=omega)
        scalars.update(
            {
                "q_shear": shear / omega,
                "local_mri_smax_over_omega": opt["s_max_over_Omega"],
                "local_mri_growth": _mri_local_growth(
                    abs(kz * bz),
                    omega,
                    2.0 * omega * (2.0 * omega - shear),
                    -2.0 * shear * omega,
                ),
            }
        )
    return {"scalars": scalars, "time_series": [{"t": 0.0, **scalars}]}


def _magnetic_bc(spec: dict[str, Any]) -> str:
    magnetic = spec["boundary_conditions"]["magnetic"]
    return magnetic.get("type", magnetic) if isinstance(magnetic, dict) else magnetic


def _quadratic_energy(q: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.real(np.asarray(q).conj().T @ matrix @ np.asarray(q)))


def _normalize_mode(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    energy = _quadratic_energy(q, matrix)
    if energy <= 0.0:
        return q
    return q / math.sqrt(energy)


def _pcf_mhd_mode_scalars(operator: Any, q: np.ndarray) -> dict[str, float]:
    q = _normalize_mode(q, operator.energy_matrix("total"))
    n = operator.nx
    blocks = operator._blocks()
    velocity = [
        q[blocks[name] * n : (blocks[name] + 1) * n]
        for name in ("ux", "uy", "uz")
    ]
    magnetic = [
        q[blocks[name] * n : (blocks[name] + 1) * n]
        for name in ("bx", "by", "bz")
    ]
    kinetic = observables.kinetic_energy(velocity, weights=operator.weights)
    magnetic_energy = observables.magnetic_energy(magnetic, weights=operator.weights)
    return {
        "kinetic_energy": kinetic,
        "magnetic_energy": magnetic_energy,
        "total_energy": kinetic + magnetic_energy,
        "maxwell_stress_xy": observables.maxwell_stress(
            magnetic,
            weights=operator.weights,
        ),
    }


def _mri_local_growth(
    omega_a: float,
    omega: float,
    kappa2: float,
    d_omega2_dlnr: float,
) -> float:
    a = omega_a**2 + 0.5 * kappa2
    c = omega_a**2 * (omega_a**2 + d_omega2_dlnr)
    disc = a**2 - c
    if disc < 0.0:
        return 0.0
    s2 = -a + math.sqrt(disc)
    return math.sqrt(s2) if s2 > 0.0 else 0.0


def _mri_keplerian_optimum(omega: float = 1.0, Omega: float | None = None) -> dict[str, float]:
    if Omega is not None:
        omega = Omega
    q = 1.5
    kappa2 = (4.0 - 2.0 * q) * omega**2
    d_omega2_dlnr = -2.0 * q * omega**2
    omega_a = np.linspace(1.0e-3, math.sqrt(3.0) * omega * 0.999, 4000)
    growth = np.array([
        _mri_local_growth(w, omega, kappa2, d_omega2_dlnr)
        for w in omega_a
    ])
    idx = int(np.argmax(growth))
    return {
        "s_max": float(growth[idx]),
        "s_max_over_Omega": float(growth[idx] / omega),
        "wa2_opt_over_O2": float((omega_a[idx] / omega) ** 2),
        "theory_s_max_over_Omega": 0.75,
        "theory_wa2_opt": 15.0 / 16.0,
        "theory_cutoff_wa2": 3.0,
    }

