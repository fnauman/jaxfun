"""Small production oracle executions that do not require live shenfun."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import observables


class ProductionOracleNotImplementedError(NotImplementedError):
    """Raised when a spec has no wired jaxfun production execution path yet."""


def run_supported_spec(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    """Run a supported production spec and return canonical diagnostics."""

    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "circular_couette_dns_growth"
    ):
        return _run_taylor_couette_hydro_dns(
            spec, steps=steps, out_dir=out_dir, checkpoint_every=checkpoint_every
        )
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] in {"mhd", "mri"}
        and spec["expected_oracle"]["type"] == "tc_mri_dns_growth"
    ):
        return _run_taylor_couette_mhd_dns(
            spec, steps=steps, out_dir=out_dir, checkpoint_every=checkpoint_every
        )
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "pcf_hydro_dns_decay"
    ):
        return _run_pcf_primitive_dns(
            spec, steps=steps, out_dir=out_dir, checkpoint_every=checkpoint_every
        )
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] == "mri"
        and spec["expected_oracle"]["type"] == "pcf_mri_dns_growth"
    ):
        return _run_pcf_primitive_dns(
            spec, steps=steps, out_dir=out_dir, checkpoint_every=checkpoint_every
        )
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
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] in {"mhd", "mri"}
        and spec["expected_oracle"]["type"]
        in {"pcf_mhd_linear_conducting", "local_ideal_mri"}
    ):
        return _run_pcf_mhd_like(spec)
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] in {"mhd", "mri"}
        and spec["expected_oracle"]["type"]
        in {"tc_mhd_linear_conducting", "tc_mhd_linear_insulating"}
    ):
        return _run_taylor_couette_mhd(spec)

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


def _run_taylor_couette_hydro_dns(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    from examples.taylor_couette_dns_jax import AxisymmetricTCDNSJax, CircularCouette

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    base = CircularCouette(
        float(groups["R1"]),
        float(groups["R2"]),
        float(groups["Omega1"]),
        float(groups["Omega2"]),
    )
    solver = AxisymmetricTCDNSJax(
        base,
        nu=float(groups["nu"]),
        Nr=int(resolution.get("Nr", resolution.get("N", 40))),
        Nz=int(resolution.get("Nz", 8)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "C"),
        dealias=1.0,
    )
    state, eigenvalue = solver.seed_linear_eigenmode(
        kz_mode=_kz_mode_from_spec(spec, solver.Lz),
        amp=float(spec["initial_condition"].get("amplitude", 1.0e-6)),
    )
    initial = solver.diagnostics(state)
    n_steps = _steps_from_spec(spec, steps=steps)
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        state_kind="axisymmetric_tc_hydro",
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(initial["E"], final["E"], n_steps, solver.dt)
    elapsed = n_steps * float(spec["time"]["dt"])
    scalars = {
        "kinetic_energy": float(final["E"]),
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_linf": float(final["div_linf"]),
        "rayleigh_stable": bool(base.rayleigh_stable()),
    }
    return {
        "scalars": scalars,
        "time_series": [
            {
                "t": 0.0,
                "kinetic_energy": float(initial["E"]),
                "growth_rate_linear": float(eigenvalue.real),
            },
            {
                "t": elapsed,
                "kinetic_energy": float(final["E"]),
                "growth_rate": float(growth_rate),
            },
        ],
    }


def _run_pcf_primitive_dns(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    from examples.pcf_mri_primitive_jax import AxisymmetricPCFMRIDNSJax

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    is_hydro = spec["physics"] == "hydro"
    solver = AxisymmetricPCFMRIDNSJax(
        S=float(groups.get("S", 1.0)),
        omega=float(groups.get("Omega", 0.0)),
        B0=0.0
        if is_hydro
        else float(groups.get("B0", spec.get("forcing", {}).get("B0", 0.1))),
        nu=float(groups["nu"]),
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
        Nx=int(resolution.get("Nx", resolution.get("N", 40))),
        Nz=int(resolution.get("Nz", 16)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "C"),
        dealias=1.0,
    )
    seed = solver.seed_hydro_eigenmode if is_hydro else solver.seed_linear_eigenmode
    state, eigenvalue = seed(
        kz_mode=int(spec.get("mode", {}).get("axial_mode", 1)),
        amp=float(spec["initial_condition"].get("amplitude", 1.0e-7)),
    )
    initial = solver.diagnostics(state)
    n_steps = _steps_from_spec(spec, steps=steps)
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        state_kind="axisymmetric_pcf_primitive",
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(initial["E"], final["E"], n_steps, solver.dt)
    elapsed = n_steps * float(spec["time"]["dt"])
    scalars = {
        "kinetic_energy": float(final["Ekin"]),
        "magnetic_energy": float(final["Emag"]),
        "growth_rate": growth_rate,
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_u": float(final["divu"]),
    }
    first = {
        "t": 0.0,
        "kinetic_energy": float(initial["Ekin"]),
        "growth_rate_linear": float(eigenvalue.real),
    }
    last = {
        "t": elapsed,
        "kinetic_energy": float(final["Ekin"]),
        "growth_rate": growth_rate,
    }
    if not is_hydro:
        magnetic_bc = _magnetic_bc(spec)
        scalars.update(
            {
                "divergence_b": float(final["divb"]),
                "magnetic_bc": magnetic_bc,
            }
        )
        first["magnetic_energy"] = float(initial["Emag"])
        last["magnetic_energy"] = float(final["Emag"])
    return {"scalars": scalars, "time_series": [first, last]}


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
        q[blocks[name] * n : (blocks[name] + 1) * n] for name in ("ux", "uy", "uz")
    ]
    magnetic = [
        q[blocks[name] * n : (blocks[name] + 1) * n] for name in ("bx", "by", "bz")
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


def _mri_keplerian_optimum(
    omega: float = 1.0, Omega: float | None = None
) -> dict[str, float]:
    if Omega is not None:
        omega = Omega
    q = 1.5
    kappa2 = (4.0 - 2.0 * q) * omega**2
    d_omega2_dlnr = -2.0 * q * omega**2
    omega_a = np.linspace(1.0e-3, math.sqrt(3.0) * omega * 0.999, 4000)
    growth = np.array(
        [_mri_local_growth(w, omega, kappa2, d_omega2_dlnr) for w in omega_a]
    )
    idx = int(np.argmax(growth))
    return {
        "s_max": float(growth[idx]),
        "s_max_over_Omega": float(growth[idx] / omega),
        "wa2_opt_over_O2": float((omega_a[idx] / omega) ** 2),
        "theory_s_max_over_Omega": 0.75,
        "theory_wa2_opt": 15.0 / 16.0,
        "theory_cutoff_wa2": 3.0,
    }


def _run_taylor_couette_mhd(spec: dict[str, Any]) -> dict[str, Any]:
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax

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
    m = int(mode.get("azimuthal_wavenumber", 0))
    kz = float(mode.get("axial_wavenumber", 3.0))
    magnetic_bc = _magnetic_bc(spec)
    operator = TaylorCouetteMRIJax(
        base,
        B0=float(groups.get("B0", 0.1)),
        nu=float(groups["nu"]),
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
        N=n,
        family=resolution.get("family", "C"),
        magnetic_bc=magnetic_bc,
    )
    eigs, vectors = operator.eigs(m, kz, n_return=3)
    scalars = {
        **_tc_mhd_mode_scalars(operator, m, kz, vectors[:, 0]),
        "growth_rate": float(eigs[0].real),
        "eigenvalue_real": float(eigs[0].real),
        "eigenvalue_imag": float(eigs[0].imag),
        "divergence_b_l2": 0.0,
        "magnetic_bc": magnetic_bc,
    }
    return {"scalars": scalars, "time_series": [{"t": 0.0, **scalars}]}


def _run_taylor_couette_mhd_dns(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    from examples.taylor_couette_dns_jax import AxisymmetricMRIDNSJax, CircularCouette

    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc != "conducting":
        raise ProductionOracleNotImplementedError(
            "TC MHD DNS golden parity is wired only for conducting walls, "
            f"got {magnetic_bc!r}"
        )
    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    solver = AxisymmetricMRIDNSJax(
        CircularCouette(
            float(groups["R1"]),
            float(groups["R2"]),
            float(groups["Omega1"]),
            float(groups["Omega2"]),
        ),
        B0=float(groups.get("B0", spec.get("forcing", {}).get("B0", 0.1))),
        nu=float(groups["nu"]),
        eta_mag=float(groups["eta_mag"]),
        Nr=int(resolution.get("Nr", resolution.get("N", 40))),
        Nz=int(resolution.get("Nz", 8)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "C"),
        dealias=1.0,
    )
    state, eigenvalue = solver.seed_linear_eigenmode(
        kz_mode=_kz_mode_from_spec(spec, solver.Lz),
        amp=float(spec["initial_condition"].get("amplitude", 1.0e-7)),
    )
    initial = solver.diagnostics(state)
    n_steps = _steps_from_spec(spec, steps=steps)
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        state_kind="axisymmetric_tc_mhd",
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(initial["E"], final["E"], n_steps, solver.dt)
    elapsed = n_steps * float(spec["time"]["dt"])
    scalars = {
        "kinetic_energy": float(final["Ekin"]),
        "magnetic_energy": float(final["Emag"]),
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_u": float(final["divu"]),
        "divergence_b": float(final["divb"]),
        "magnetic_bc": magnetic_bc,
    }
    return {
        "scalars": scalars,
        "time_series": [
            {
                "t": 0.0,
                "kinetic_energy": float(initial["Ekin"]),
                "magnetic_energy": float(initial["Emag"]),
                "growth_rate_linear": float(eigenvalue.real),
            },
            {
                "t": elapsed,
                "kinetic_energy": float(final["Ekin"]),
                "magnetic_energy": float(final["Emag"]),
                "growth_rate": float(growth_rate),
            },
        ],
    }


def _tc_mhd_mode_scalars(
    operator: Any,
    m: int,
    kz: float,
    q: np.ndarray,
) -> dict[str, float]:
    q = _normalize_mode(q, operator.energy_matrix(m, kz, "total"))
    kinetic = _quadratic_energy(q, operator.energy_matrix(m, kz, "kinetic"))
    magnetic_energy = _quadratic_energy(q, operator.energy_matrix(m, kz, "magnetic"))
    return {
        "kinetic_energy": kinetic,
        "magnetic_energy": magnetic_energy,
        "total_energy": kinetic + magnetic_energy,
    }


def _solve_with_optional_checkpoints(
    solver: Any,
    state: Any,
    steps: int,
    *,
    spec: dict[str, Any],
    out_dir: str | Path | None,
    checkpoint_every: int | None,
    state_kind: str,
) -> Any:
    if checkpoint_every is None:
        return solver.solve(state, steps)
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if out_dir is None:
        raise ValueError("out_dir is required when checkpoint_every is set")

    from jaxfun.io import Cadence, write_checkpoint

    checkpoint_path = Path(out_dir) / "checkpoints" / "checkpoints.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def on_checkpoint(t: float, tstep: int, checkpoint_state: Any) -> None:
        write_checkpoint(
            checkpoint_path,
            {"state": _checkpoint_payload(checkpoint_state)},
            t=t,
            tstep=tstep,
            attrs={
                "problem_id": spec["problem_id"],
                "state_kind": state_kind,
                "schema_version": 1,
            },
        )

    out = solver.solve_with_cadence(
        state,
        steps,
        Cadence(checkpoint_every=checkpoint_every),
        block_size=max(1, int(checkpoint_every)),
        on_checkpoint=on_checkpoint,
    )
    if steps == 0 or steps % int(checkpoint_every) != 0:
        on_checkpoint(float(steps) * float(solver.dt), int(steps), out)
    return out


def _checkpoint_payload(state: Any) -> dict[str, Any]:
    if hasattr(state, "u"):
        return {
            "u": state.u,
            "p": state.p,
            "nonlinear_old": state.nonlinear_old,
            "have_old": state.have_old,
        }
    if hasattr(state, "x"):
        return {
            "x": state.x,
            "p": state.p,
            "nonlinear_old": state.nonlinear_old,
            "have_old": state.have_old,
        }
    raise TypeError(f"unsupported checkpoint state type {type(state).__name__}")


def _growth_rate_from_energy(e0: Any, e1: Any, steps: int, dt: float) -> float:
    elapsed = int(steps) * float(dt)
    if elapsed <= 0.0:
        raise ValueError("growth-rate diagnostics require at least one DNS step")
    return 0.5 * math.log(float(e1) / float(e0)) / elapsed


def _steps_from_spec(spec: dict[str, Any], *, steps: int | None = None) -> int:
    if steps is not None:
        if steps < 0:
            raise ValueError("steps override must be non-negative")
        return int(steps)
    dt = float(spec["time"]["dt"])
    final_time = float(spec["time"]["final_time"])
    n_steps = int(round(final_time / dt))
    if not math.isclose(n_steps * dt, final_time, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "final_time must be an integer multiple of dt for DNS parity runs"
        )
    return n_steps


def _kz_mode_from_spec(spec: dict[str, Any], Lz: float) -> int:
    mode = spec.get("mode", {})
    if int(mode.get("azimuthal_wavenumber", 0)) != 0:
        raise ProductionOracleNotImplementedError(
            "Taylor-Couette DNS golden parity is wired only for axisymmetric m=0 specs"
        )
    kz = float(mode["axial_wavenumber"])
    kz_mode = int(round(kz * float(Lz) / (2.0 * math.pi)))
    if kz_mode < 1:
        raise ValueError("axial_wavenumber does not map to a positive Fourier mode")
    resolved = 2.0 * math.pi * kz_mode / float(Lz)
    if not math.isclose(resolved, kz, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            f"axial_wavenumber={kz!r} does not map to an integer Fourier "
            f"mode for Lz={Lz!r}"
        )
    return kz_mode
