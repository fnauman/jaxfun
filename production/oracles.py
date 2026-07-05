"""Small production oracle executions that do not require live shenfun."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import observables
from .checkpoint import write_production_checkpoint


class ProductionOracleNotImplementedError(NotImplementedError):
    """Raised when a spec has no wired jaxfun production execution path yet."""


def load_resume_checkpoint(run_dir: str | Path):
    """Read the latest production checkpoint from a run directory or HDF5 file."""
    from jaxfun.io import read_checkpoint

    path = Path(run_dir)
    checkpoint_path = path if path.suffix == ".h5" else path / "checkpoints" / "checkpoints.h5"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found at {checkpoint_path}")
    return read_checkpoint(checkpoint_path)


def validate_resume_checkpoint(
    record: Any, spec: dict[str, Any], device_record: dict[str, Any] | None = None
) -> None:
    """Validate checkpoint metadata against the run being resumed."""
    attrs = record.attrs
    if str(attrs.get("spec_hash")) != str(spec["spec_hash"]):
        raise ValueError("resume checkpoint spec_hash does not match requested spec")
    dtype_json = attrs.get("dtype_metadata_json", "{}")
    if isinstance(dtype_json, bytes):
        dtype_json = dtype_json.decode()
    dtype_metadata = json.loads(dtype_json or "{}")
    expected_dtype = dtype_metadata.get("production_run_dtype")
    active_dtype = None if device_record is None else device_record.get("production_run_dtype")
    if active_dtype is not None and expected_dtype is not None and active_dtype != expected_dtype:
        raise ValueError(
            "resume checkpoint production dtype "
            f"{expected_dtype!r} does not match active dtype {active_dtype!r}"
        )


def _resume_or_initial_state(
    resume_checkpoint: Any | None,
    initial_state: Any,
    *,
    spec: dict[str, Any],
    state_kind: str,
) -> tuple[Any, int, float]:
    if resume_checkpoint is None:
        return initial_state, 0, 0.0
    validate_resume_checkpoint(resume_checkpoint, spec)
    checkpoint_kind = str(resume_checkpoint.attrs.get("state_kind"))
    if checkpoint_kind != state_kind:
        raise ValueError(
            f"resume checkpoint state_kind {checkpoint_kind!r} does not match "
            f"expected {state_kind!r}"
        )
    payload = resume_checkpoint.fields.get("state")
    if not isinstance(payload, dict):
        raise ValueError("resume checkpoint is missing state payload")
    return (
        _state_from_checkpoint_payload(payload, state_kind=state_kind),
        int(resume_checkpoint.tstep),
        float(resume_checkpoint.t),
    )


def _state_from_checkpoint_payload(payload: dict[str, Any], *, state_kind: str) -> Any:
    if state_kind == "pcf_fluctuation_saturation":
        from examples.channelflow_kmm import KMMState

        return KMMState(u=tuple(payload["u"]), g=payload["g"])
    if state_kind in {"axisymmetric_tc_hydro", "axisymmetric_tc_hydro_saturation"}:
        from examples.taylor_couette_dns_jax import AxisymmetricTCState

        return AxisymmetricTCState(
            u=tuple(payload["u"]),
            p=payload["p"],
            nonlinear_old=tuple(payload["nonlinear_old"]),
            have_old=payload["have_old"],
        )
    if state_kind in {"axisymmetric_tc_mhd", "axisymmetric_tc_mhd_saturation"}:
        from examples.taylor_couette_dns_jax import AxisymmetricMRIState

        return AxisymmetricMRIState(
            x=tuple(payload["x"]),
            p=payload["p"],
            nonlinear_old=tuple(payload["nonlinear_old"]),
            have_old=payload["have_old"],
        )
    if state_kind in {"axisymmetric_pcf_primitive", "pcf_primitive_mhd_saturation"}:
        from examples.pcf_mri_primitive_jax import AxisymmetricPCFState

        return AxisymmetricPCFState(
            x=tuple(payload["x"]),
            p=payload["p"],
            nonlinear_old=tuple(payload["nonlinear_old"]),
            have_old=payload["have_old"],
        )
    raise ValueError(f"unsupported resume state_kind {state_kind!r}")


def _remaining_steps_from_resume(
    spec: dict[str, Any], *, steps: int | None, tstep0: int
) -> tuple[int, int]:
    target = _steps_from_spec(spec, steps=steps)
    if int(tstep0) > target:
        raise ValueError(
            f"resume checkpoint step {int(tstep0)} is beyond target step {target}"
        )
    return target, target - int(tstep0)


def _saturation_passed(
    growth: Any, *, threshold: float, final_energies: tuple[Any, ...]
) -> bool:
    try:
        growth_value = float(growth)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(growth_value) or growth_value <= threshold:
        return False
    for energy in final_energies:
        try:
            energy_value = float(energy)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(energy_value) or energy_value < 0.0:
            return False
    return True


def run_supported_spec(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
) -> dict[str, Any]:
    """Run a supported production spec and return canonical diagnostics."""

    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "tc_hydro_saturation_ladder"
    ):
        return _run_taylor_couette_hydro_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "circular_couette_dns_growth"
    ):
        return _run_taylor_couette_hydro_dns(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] in {"mhd", "mri"}
        and spec["expected_oracle"]["type"] == "tc_mri_saturation_ladder"
    ):
        return _run_taylor_couette_mhd_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "taylor_couette"
        and spec["physics"] in {"mhd", "mri"}
        and spec["expected_oracle"]["type"] == "tc_mri_dns_growth"
    ):
        return _run_taylor_couette_mhd_dns(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "pcf_hydro_dns_decay"
    ):
        return _run_pcf_primitive_dns(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] == "mri"
        and spec["expected_oracle"]["type"] == "pcf_mri_dns_growth"
    ):
        return _run_pcf_primitive_dns(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["problem_id"] == "pcf_fluct_re400"
        and spec["geometry"] == "pcf"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] == "gpu_generated_saturated_dns"
    ):
        return _run_pcf_fluctuation_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["problem_id"] == "pcf_mhd_divfree"
        and spec["geometry"] == "pcf"
        and spec["physics"] == "mhd"
        and spec["expected_oracle"]["type"] == "gpu_generated_saturated_dns"
    ):
        return _run_pcf_primitive_mhd_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["problem_id"] == "exp_pcf_mri_shearbox_growth"
        and spec["geometry"] == "pcf"
        and spec["physics"] == "mri"
        and spec["expected_oracle"]["type"] == "mri_saturation_ladder"
    ):
        return _run_pcf_primitive_mhd_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
        )
    if (
        spec["geometry"] == "pipe"
        and spec["physics"] == "hydro"
        and spec["expected_oracle"]["type"] in {"hagen_poiseuille", "pipe_womersley"}
    ):
        return _run_pipe_hydro(spec, steps=steps)
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


def _run_pipe_hydro(
    spec: dict[str, Any], *, steps: int | None = None
) -> dict[str, Any]:
    from examples.pipe_flow_dns_jax import (
        hagen_poiseuille_diagnostics,
        womersley_cn_diagnostics,
    )

    groups = spec["nondimensional_groups"]
    time = spec["time"]
    domain = spec["domain"]
    r0, r1 = (float(value) for value in domain["r"])
    radius = float(groups.get("R", r1))
    if not (np.isclose(r0, 0.0) and np.isclose(r1, radius)):
        raise ProductionOracleNotImplementedError(
            "pipe hydro oracle is wired for a regular-axis domain [0, R]"
        )
    if time.get("integrator") != "CNAB2":
        raise ProductionOracleNotImplementedError(
            "pipe hydro oracle is wired for the CNAB2 production specs"
        )

    nu = float(groups["nu"])
    length = float(domain.get("z_period", 2.0 * math.pi))
    dt = float(time["dt"])
    elapsed = _steps_from_spec(spec, steps=steps) * dt
    oracle = spec["expected_oracle"]["type"]

    if oracle == "pipe_womersley":
        amplitude = float(spec["forcing"].get("amplitude", 1.0))
        omega = float(spec["forcing"].get("omega", groups.get("omega", 1.0)))
        initial, final = womersley_cn_diagnostics(
            amplitude=amplitude,
            omega=omega,
            nu=nu,
            radius=radius,
            length=length,
            dt=dt,
            final_time=elapsed,
        )
    else:
        fz = float(spec["forcing"].get("fz", groups.get("fz", 4.0 * nu)))
        initial = final = hagen_poiseuille_diagnostics(
            fz=fz,
            nu=nu,
            radius=radius,
            length=length,
        )

    return {
        "scalars": final.scalars(),
        "time_series": [
            {"t": 0.0, **initial.scalars()},
            {"t": elapsed, **final.scalars()},
        ],
    }


def _run_channel_poiseuille(spec: dict[str, Any]) -> dict[str, Any]:
    _channel_poiseuille_kmm_state(spec)

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


def _channel_poiseuille_kmm_state(spec: dict[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp

    from examples.channelflow_kmm import KMM, KMMState

    resolution = spec["resolution"]
    groups = spec["nondimensional_groups"]
    domain = spec["domain"]
    x0, x1 = (float(v) for v in domain["x"])
    if not (np.isclose(x0, -1.0) and np.isclose(x1, 1.0)):
        raise ProductionOracleNotImplementedError(
            "channel Poiseuille KMM oracle is wired for the half-gap domain [-1, 1]"
        )

    re = float(groups["Re"])
    u_center = float(groups.get("U_center", 1.0))
    pressure_gradient = -2.0 * u_center / re
    solver = KMM(
        N=(
            int(resolution.get("nx", resolution.get("N", 64))),
            int(resolution.get("ny", 8)),
            int(resolution.get("nz", 8)),
        ),
        domain=(
            (x0, x1),
            (0.0, float(domain.get("y_period", 4.0))),
            (0.0, float(domain.get("z_period", 4.0))),
        ),
        nu=1.0 / re,
        dt=max(float(spec["time"].get("dt", 0.0)), 1.0e-3),
        family=resolution.get("family", "C"),
        padding_factor=(1.0, 1.0, 1.0),
        dpdy=pressure_gradient,
    )
    v00 = solver.L00.solve(-solver.dpdy_rhs)
    g = jnp.zeros(solver.TD.num_dofs, dtype=complex)
    state = KMMState(
        u=solver._reconstruct_velocity(
            jnp.zeros(solver.TB.num_dofs, dtype=complex),
            g,
            v00,
            jnp.zeros_like(v00),
        ),
        g=g,
    )
    x = solver.D00.mesh()
    profile = solver.D00.backward(v00)
    expected = u_center * (1.0 - x**2)
    profile_linf = float(jnp.max(jnp.abs(profile - expected)))
    divergence_l2 = float(solver.divergence_l2(state))
    dtype_eps = np.finfo(np.asarray(profile).dtype).eps
    tolerance = max(1.0e-10, 100.0 * float(dtype_eps))
    if profile_linf > tolerance or divergence_l2 > tolerance:
        raise RuntimeError(
            "driven channel KMM steady state did not recover Poiseuille "
            f"(profile_linf={profile_linf}, divergence_l2={divergence_l2})"
        )
    return {
        "solver": solver,
        "state": state,
        "profile_linf": profile_linf,
        "divergence_l2": divergence_l2,
        "pressure_gradient": pressure_gradient,
    }


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


def _run_taylor_couette_hydro_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
) -> dict[str, Any]:
    from examples.taylor_couette_dns_jax import AxisymmetricTCDNSJax, CircularCouette

    resolution = _selected_resolution(spec)
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
        Nz=int(resolution.get("Nz", 16)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=spec["resolution"].get("family", resolution.get("family", "C")),
        dealias=float(spec["resolution"].get("dealias", 1.0)),
    )
    state, eigenvalue = solver.seed_linear_eigenmode(
        kz_mode=_kz_mode_from_spec(spec, solver.Lz, strict=False),
        amp=float(spec["initial_condition"].get("amplitude", 1.0e-4)),
    )
    initial = solver.diagnostics(state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        diagnostic_rows.append(
            {
                "t": float(t),
                "kinetic_energy": float(diag["E"]),
                "divergence_l2": float(diag.get("continuity_l2", diag["div_linf"])),
                "divergence_linf": float(diag["div_linf"]),
            }
        )

    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="axisymmetric_tc_hydro_saturation",
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="axisymmetric_tc_hydro_saturation",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(
        initial["E"], final["E"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
    energy_growth = (
        float(final["E"] / initial["E"]) if float(initial["E"]) > 0.0 else 0.0
    )
    radial_velocity_linf = _radial_velocity_linf(solver, out)
    torque = _tc_inner_torque(solver, out)
    scalars = {
        "kinetic_energy": float(final["E"]),
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_l2": float(final["continuity_l2"]),
        "divergence_linf": float(final["div_linf"]),
        "torque": float(torque),
        "radial_velocity_linf": float(radial_velocity_linf),
        "energy_growth_factor": float(energy_growth),
        "saturation_check_passed": _saturation_passed(
            energy_growth,
            threshold=1.0e3,
            final_energies=(
                final["E"],
                final["continuity_l2"],
                final["div_linf"],
                radial_velocity_linf,
            ),
        ),
    }
    first = {
        "t": 0.0,
        "kinetic_energy": float(initial["E"]),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_l2": float(initial["continuity_l2"]),
        "radial_velocity_linf": _radial_velocity_linf(solver, state),
    }
    last = {
        "t": elapsed,
        "kinetic_energy": float(final["E"]),
        "growth_rate": float(growth_rate),
        "divergence_l2": float(final["continuity_l2"]),
        "divergence_linf": float(final["div_linf"]),
        "radial_velocity_linf": float(radial_velocity_linf),
        "torque": float(torque),
    }
    series = _dedupe_time_rows([first, *diagnostic_rows, last])
    scalars.update(_stationarity_scalars(series, key="kinetic_energy"))
    return {"scalars": scalars, "time_series": series}


def _run_taylor_couette_hydro_dns(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
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
    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint, state, spec=spec, state_kind="axisymmetric_tc_hydro"
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="axisymmetric_tc_hydro",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(
        initial["E"], final["E"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
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
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
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
    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="axisymmetric_pcf_primitive",
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="axisymmetric_pcf_primitive",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(
        initial["E"], final["E"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
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


def _run_pcf_fluctuation_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
) -> dict[str, Any]:
    from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax

    resolution = _selected_resolution(spec)
    domain = (
        tuple(float(value) for value in spec["domain"]["x"]),
        (0.0, float(spec["domain"]["y_period"])),
        (0.0, float(spec["domain"]["z_period"])),
    )
    groups = spec["nondimensional_groups"]
    solver = PlaneCouetteFluctuationJax(
        N=(
            int(resolution.get("Nx", resolution.get("N", 32))),
            int(resolution.get("Ny", 64)),
            int(resolution.get("Nz", 32)),
        ),
        domain=domain,
        Re=float(groups["Re"]),
        U_wall=float(groups.get("U_wall", 1.0)),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "L"),
        padding_factor=_padding_factor(resolution, dimensions=3),
        perturbation_amplitude=float(spec["initial_condition"].get("amplitude", 0.1)),
    )
    state = solver.initial_state()
    initial = _pcf_fluctuation_scalars(solver, state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        diagnostic_rows.append(
            {
                "t": float(t),
                "kinetic_energy": float(diag["Epert"]),
                "total_kinetic_energy": float(diag["Etot"]),
                "divergence_l2": float(diag["divL2"]),
                "mean_shear": float(diag["mean_shear"]),
            }
        )

    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="pcf_fluctuation_saturation",
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="pcf_fluctuation_saturation",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = _pcf_fluctuation_scalars(solver, out)
    growth_rate = _growth_rate_from_energy(
        initial["kinetic_energy"],
        final["kinetic_energy"],
        target_steps,
        solver.dt,
    )
    elapsed = target_steps * float(spec["time"]["dt"])
    energy_growth = (
        final["kinetic_energy"] / initial["kinetic_energy"]
        if initial["kinetic_energy"] > 0.0
        else 0.0
    )
    scalars = {
        **final,
        "growth_rate": float(growth_rate),
        "energy_growth_factor": float(energy_growth),
        "saturation_check_passed": _saturation_passed(
            energy_growth,
            threshold=2.0,
            final_energies=(
                final["kinetic_energy"],
                final["total_kinetic_energy"],
                final["divergence_l2"],
                final["streak_rms"],
                final["roll_rms"],
            ),
        ),
    }
    first = {
        "t": 0.0,
        "kinetic_energy": initial["kinetic_energy"],
        "total_kinetic_energy": initial["total_kinetic_energy"],
        "divergence_l2": initial["divergence_l2"],
        "mean_shear": initial["mean_shear"],
    }
    last = {
        "t": elapsed,
        "kinetic_energy": final["kinetic_energy"],
        "total_kinetic_energy": final["total_kinetic_energy"],
        "divergence_l2": final["divergence_l2"],
        "mean_shear": final["mean_shear"],
        "growth_rate": float(growth_rate),
        "energy_growth_factor": float(energy_growth),
        "saturation_check_passed": scalars["saturation_check_passed"],
        "wall_shear_lower": final["wall_shear_lower"],
        "wall_shear_upper": final["wall_shear_upper"],
        "streak_rms": final["streak_rms"],
        "roll_rms": final["roll_rms"],
    }
    series = _dedupe_time_rows([first, *diagnostic_rows, last])
    scalars.update(_stationarity_scalars(series, key="kinetic_energy"))
    return {"scalars": scalars, "time_series": series}


def _run_pcf_primitive_mhd_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
) -> dict[str, Any]:
    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc != "conducting":
        raise ProductionOracleNotImplementedError(
            "PCF primitive-b saturation runner is wired only for conducting walls, "
            f"got {magnetic_bc!r}"
        )
    _assert_pcf_half_gap_domain(spec)
    resolution = _selected_resolution(spec)
    groups = spec["nondimensional_groups"]
    solver = PCFMRIDNSJax(
        S=float(groups.get("S", 1.0)),
        omega=float(groups.get("Omega", 0.0)),
        B0=float(groups.get("B0", spec.get("forcing", {}).get("B0", 0.1))),
        nu=float(groups["nu"]),
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
        Nx=int(resolution.get("Nx", resolution.get("N", 32))),
        Ny=int(resolution.get("Ny", 8)),
        Nz=int(resolution.get("Nz", 16)),
        Ly=float(spec["domain"]["y_period"]),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "L"),
        dealias=_padding_factor(resolution, dimensions=3),
    )
    if spec["physics"] == "mri":
        state, eigenvalue = _pcf_mri_packet_state(solver, spec)
    else:
        state, eigenvalue = _pcf_mhd_perturbation_state(solver, spec)

    initial = _pcf_primitive_3d_scalars(solver, state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        diagnostic_rows.append(
            {
                "t": float(t),
                "kinetic_energy": float(diag["Ekin"]),
                "magnetic_energy": float(diag["Emag"]),
                "total_energy": float(diag["E"]),
                "divergence_u_l2": float(diag["divu"]),
                "divergence_b_l2": float(diag["divb"]),
            }
        )

    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="pcf_primitive_mhd_saturation",
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="pcf_primitive_mhd_saturation",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = _pcf_primitive_3d_scalars(solver, out)
    growth_rate = _growth_rate_from_energy(
        initial["total_energy"], final["total_energy"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
    magnetic_growth = (
        final["magnetic_energy"] / initial["magnetic_energy"]
        if initial["magnetic_energy"] > 0.0
        else 0.0
    )
    saturation_passed = _saturation_passed(
        magnetic_growth,
        threshold=2.0,
        final_energies=(
            final["kinetic_energy"],
            final["magnetic_energy"],
            final["total_energy"],
            final["divergence_u_l2"],
            final["divergence_b_l2"],
        ),
    )
    scalars = {
        **final,
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
        "magnetic_bc": magnetic_bc,
    }
    first = {
        "t": 0.0,
        "kinetic_energy": initial["kinetic_energy"],
        "magnetic_energy": initial["magnetic_energy"],
        "total_energy": initial["total_energy"],
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_u_l2": initial["divergence_u_l2"],
        "divergence_b_l2": initial["divergence_b_l2"],
        "maxwell_stress_xy": initial["maxwell_stress_xy"],
        "reynolds_stress": initial["reynolds_stress"],
        "transport_alpha": initial["transport_alpha"],
        "butterfly_by_mean": initial["butterfly_by_mean"],
    }
    last = {
        "t": elapsed,
        "kinetic_energy": final["kinetic_energy"],
        "magnetic_energy": final["magnetic_energy"],
        "total_energy": final["total_energy"],
        "growth_rate": float(growth_rate),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
        "divergence_u_l2": final["divergence_u_l2"],
        "divergence_b_l2": final["divergence_b_l2"],
        "maxwell_stress_xy": final["maxwell_stress_xy"],
        "reynolds_stress": final["reynolds_stress"],
        "transport_alpha": final["transport_alpha"],
        "butterfly_by_mean": final["butterfly_by_mean"],
    }
    series = _dedupe_time_rows([first, *diagnostic_rows, last])
    scalars.update(_stationarity_scalars(series, key="total_energy"))
    return {"scalars": scalars, "time_series": series}


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
    bz = float(groups.get("Bz", 0.025))
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
        if shear <= 0.0 or omega <= 0.0:
            raise ValueError("PCF MRI linear oracle requires positive S and Omega")
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


def _run_taylor_couette_mhd_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
) -> dict[str, Any]:
    from examples.taylor_couette_dns_jax import AxisymmetricMRIDNSJax, CircularCouette

    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc != "conducting":
        raise ProductionOracleNotImplementedError(
            "TC MRI saturation runner is wired only for conducting walls, "
            f"got {magnetic_bc!r}"
        )
    resolution = _selected_resolution(spec)
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
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
        Nr=int(resolution.get("Nr", resolution.get("N", 40))),
        Nz=int(resolution.get("Nz", 24)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=spec["resolution"].get("family", resolution.get("family", "C")),
        dealias=float(spec["resolution"].get("dealias", 1.0)),
    )
    state, eigenvalue = solver.seed_linear_eigenmode(
        kz_mode=_kz_mode_from_spec(spec, solver.Lz, strict=False),
        amp=float(spec["initial_condition"].get("amplitude", 1.0e-4)),
    )
    initial = solver.diagnostics(state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        diagnostic_rows.append(
            {
                "t": float(t),
                "kinetic_energy": float(diag["Ekin"]),
                "magnetic_energy": float(diag["Emag"]),
                "divergence_u": float(diag["divu"]),
                "divergence_b": float(diag["divb"]),
                "divergence_b_l2": float(diag["divb"]),
            }
        )

    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="axisymmetric_tc_mhd_saturation",
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="axisymmetric_tc_mhd_saturation",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(
        initial["E"], final["E"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
    magnetic_growth = (
        float(final["Emag"] / initial["Emag"]) if float(initial["Emag"]) > 0.0 else 0.0
    )
    reynolds_stress, maxwell_stress = _tc_mhd_stresses(solver, out)
    scalars = {
        "kinetic_energy": float(final["Ekin"]),
        "magnetic_energy": float(final["Emag"]),
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_u": float(final["divu"]),
        "divergence_b": float(final["divb"]),
        "divergence_b_l2": float(final["divb"]),
        "maxwell_stress_xy": float(maxwell_stress),
        "reynolds_stress": float(reynolds_stress),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": _saturation_passed(
            magnetic_growth,
            threshold=2.0,
            final_energies=(
                final["Ekin"],
                final["Emag"],
                final["divu"],
                final["divb"],
            ),
        ),
        "magnetic_bc": magnetic_bc,
    }
    first = {
        "t": 0.0,
        "kinetic_energy": float(initial["Ekin"]),
        "magnetic_energy": float(initial["Emag"]),
        "growth_rate_linear": float(eigenvalue.real),
        "divergence_b_l2": float(initial["divb"]),
    }
    last = {
        "t": elapsed,
        "kinetic_energy": float(final["Ekin"]),
        "magnetic_energy": float(final["Emag"]),
        "growth_rate": float(growth_rate),
        "divergence_b_l2": float(final["divb"]),
        "maxwell_stress_xy": float(maxwell_stress),
        "reynolds_stress": float(reynolds_stress),
    }
    series = _dedupe_time_rows([first, *diagnostic_rows, last])
    scalars.update(_stationarity_scalars(series, key="magnetic_energy"))
    return {"scalars": scalars, "time_series": series}


def _run_taylor_couette_mhd_dns(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
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
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
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
    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint, state, spec=spec, state_kind="axisymmetric_tc_mhd"
    )
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    out = _solve_with_optional_checkpoints(
        solver,
        state,
        n_steps,
        spec=spec,
        out_dir=out_dir,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
        state_kind="axisymmetric_tc_mhd",
        device_record=device_record,
        t0=t0,
        tstep0=tstep0,
    )
    final = solver.diagnostics(out)
    growth_rate = _growth_rate_from_energy(
        initial["E"], final["E"], target_steps, solver.dt
    )
    elapsed = target_steps * float(spec["time"]["dt"])
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
    snapshot_every: int | None,
    diagnostics_every: int | None,
    state_kind: str,
    device_record: dict[str, Any] | None = None,
    on_diagnostics_row: Any | None = None,
    t0: float = 0.0,
    tstep0: int = 0,
) -> Any:
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if snapshot_every is not None and snapshot_every <= 0:
        raise ValueError("snapshot_every must be positive")
    if diagnostics_every is not None and diagnostics_every <= 0:
        raise ValueError("diagnostics_every must be positive")
    if (checkpoint_every is not None or snapshot_every is not None) and out_dir is None:
        raise ValueError("out_dir is required when checkpoint or snapshot output is set")

    monitor_every = _monitor_every(
        steps,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
    )
    if monitor_every is None and not (
        checkpoint_every or snapshot_every or diagnostics_every
    ):
        return solver.solve(state, steps)

    from jaxfun.io import Cadence, generate_xdmf, write_uniform_snapshot

    out_path = None if out_dir is None else Path(out_dir)
    checkpoint_path = None
    diagnostics_path = None
    snapshot_path = None
    if out_path is not None:
        checkpoint_path = out_path / "checkpoints" / "checkpoints.h5"
        diagnostics_path = out_path / "diagnostics.jsonl"
        snapshot_path = out_path / "snapshots" / "snapshots.h5"

    def on_checkpoint(t: float, tstep: int, checkpoint_state: Any) -> None:
        assert checkpoint_path is not None
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        write_production_checkpoint(
            checkpoint_path,
            {"state": _checkpoint_payload(checkpoint_state)},
            t=t,
            tstep=tstep,
            spec=spec,
            state_kind=state_kind,
            device_record=device_record,
            diagnostics_path=diagnostics_path,
        )

    def on_snapshot(t: float, tstep: int, snapshot_state: Any) -> None:
        assert snapshot_path is not None
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        write_uniform_snapshot(
            snapshot_path,
            _snapshot_payload(solver, snapshot_state),
            t=t,
            tstep=tstep,
            attrs={
                "problem_id": spec["problem_id"],
                "spec_hash": spec["spec_hash"],
                "state_kind": state_kind,
            },
        )

    def on_diagnostics(t: float, tstep: int, diag: Any) -> None:
        if on_diagnostics_row is not None:
            on_diagnostics_row(t, tstep, diag)

    def should_stop(t: float, tstep: int, candidate_state: Any) -> bool:
        if not _tree_all_finite(candidate_state):
            raise FloatingPointError(
                f"nonfinite solver state at tstep={int(tstep)} t={float(t):g}"
            )
        _raise_on_divergence_drift(solver, candidate_state, t=t, tstep=tstep)
        return False

    cadence = Cadence(
        diagnostics_every=diagnostics_every,
        snapshot_every=snapshot_every,
        checkpoint_every=checkpoint_every,
    )
    block_size = monitor_every or max(1, int(steps))
    out = solver.solve_with_cadence(
        state,
        steps,
        cadence,
        block_size=block_size,
        on_diagnostics=on_diagnostics
        if diagnostics_every is not None and on_diagnostics_row is not None
        else None,
        on_snapshot=on_snapshot if snapshot_every is not None else None,
        on_checkpoint=on_checkpoint if checkpoint_every is not None else None,
        should_stop=should_stop,
        t0=float(t0),
        tstep0=int(tstep0),
    )
    final_tstep = int(tstep0) + int(steps)
    final_t = float(t0) + int(steps) * float(solver.dt)
    if checkpoint_every is not None and (
        steps == 0 or final_tstep % int(checkpoint_every) != 0
    ):
        on_checkpoint(final_t, final_tstep, out)
    if snapshot_every is not None and (
        steps == 0 or final_tstep % int(snapshot_every) != 0
    ):
        on_snapshot(final_t, final_tstep, out)
    if snapshot_path is not None and snapshot_path.exists():
        xdmf_path = generate_xdmf(snapshot_path)
        _write_snapshot_manifest(
            snapshot_path.with_name("manifest.json"),
            snapshot_path=snapshot_path,
            xdmf_path=xdmf_path,
            spec=spec,
            state_kind=state_kind,
            snapshot_every=snapshot_every,
            device_record=device_record,
        )
    return out


def _stationarity_scalars(
    rows: list[dict[str, Any]],
    *,
    key: str,
    tolerance: float = 5.0e-2,
) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    samples = len(values)
    if samples < 4:
        return {
            "stationarity_key": key,
            "stationarity_window_samples": samples,
            "stationarity_relative_tolerance": tolerance,
            "stationarity_relative_change": None,
            "stationarity_check_passed": None,
        }
    quarter = max(1, samples // 4)
    previous = values[-2 * quarter : -quarter]
    current = values[-quarter:]
    previous_mean = float(np.mean(previous))
    current_mean = float(np.mean(current))
    denom = max(abs(previous_mean), abs(current_mean), 1.0e-300)
    relative_change = abs(current_mean - previous_mean) / denom
    return {
        "stationarity_key": key,
        "stationarity_window_samples": samples,
        "stationarity_relative_tolerance": tolerance,
        "stationarity_previous_mean": previous_mean,
        "stationarity_current_mean": current_mean,
        "stationarity_relative_change": float(relative_change),
        "stationarity_check_passed": bool(relative_change <= tolerance),
    }


def _dedupe_time_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[float] = set()
    for row in rows:
        t = float(row.get("t", len(out)))
        if t in seen:
            out[-1] = row
            continue
        seen.add(t)
        out.append(row)
    return out


def _monitor_every(
    steps: int,
    *,
    checkpoint_every: int | None,
    snapshot_every: int | None,
    diagnostics_every: int | None,
) -> int | None:
    candidates = [
        value
        for value in (checkpoint_every, snapshot_every, diagnostics_every)
        if value is not None and value > 0
    ]
    if candidates:
        return min(int(value) for value in candidates)
    if steps <= 0:
        return 1
    return min(100, max(1, int(steps)))


_DIVERGENCE_GUARD_LIMIT = 1.0e-2


def _raise_on_divergence_drift(
    solver: Any, state: Any, *, t: float, tstep: int
) -> None:
    diagnostics = getattr(solver, "diagnostics", None)
    if diagnostics is None:
        return
    diag = diagnostics(state)
    offenders = []
    for key, value in diag.items():
        if not _is_divergence_key(str(key)):
            continue
        try:
            magnitude = abs(float(value))
        except (TypeError, ValueError):
            offenders.append(f"{key}=non-numeric")
            continue
        if not math.isfinite(magnitude) or magnitude > _DIVERGENCE_GUARD_LIMIT:
            offenders.append(f"{key}={magnitude:g}")
    if offenders:
        details = ", ".join(offenders)
        raise FloatingPointError(
            f"divergence guard failed at tstep={int(tstep)} t={float(t):g}: "
            f"{details} > {_DIVERGENCE_GUARD_LIMIT:g}"
        )


def _is_divergence_key(key: str) -> bool:
    name = key.rsplit(".", 1)[-1].lower()
    return (
        name.startswith("divergence")
        or name.startswith("divu")
        or name.startswith("divb")
        or name.startswith("div")
        or name.startswith("continuity")
    )


def _tree_all_finite(tree: Any) -> bool:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves if hasattr(leaf, "dtype")]
    if not checks:
        return True
    return bool(jax.device_get(jnp.all(jnp.asarray(checks))))


def _snapshot_payload(solver: Any, state: Any) -> dict[str, Any]:
    def real_fields(names: tuple[str, ...], values: tuple[Any, ...]) -> dict[str, Any]:
        return {name: jnp.real(value) for name, value in zip(names, values, strict=True)}

    if hasattr(solver, "fields_physical"):
        fields = tuple(solver.fields_physical(state))
        if len(fields) >= 6:
            return real_fields(
                ("u_x", "u_y", "u_z", "b_x", "b_y", "b_z"), fields[:6]
            )
        if len(fields) >= 3:
            return real_fields(("u_x", "u_y", "u_z"), fields[:3])
    if hasattr(solver, "velocity_physical"):
        u = tuple(solver.velocity_physical(state))
        return real_fields(("u_x", "u_y", "u_z"), u[:3])
    if hasattr(solver, "total_velocity_physical"):
        u = tuple(solver.total_velocity_physical(state))
        return real_fields(("u_x", "u_y", "u_z"), u[:3])
    if hasattr(solver, "_backward_velocity") and hasattr(state, "u"):
        u = tuple(solver._backward_velocity(state.u))
        return real_fields(("u_x", "u_y", "u_z"), u[:3])
    raise TypeError(f"solver {type(solver).__name__} does not expose snapshot fields")


def _write_snapshot_manifest(
    path: Path,
    *,
    snapshot_path: Path,
    xdmf_path: Path,
    spec: dict[str, Any],
    state_kind: str,
    snapshot_every: int | None,
    device_record: dict[str, Any] | None,
) -> None:
    data = {
        "schema_version": 1,
        "problem_id": spec["problem_id"],
        "spec_hash": spec["spec_hash"],
        "state_kind": state_kind,
        "snapshot_every": snapshot_every,
        "snapshot_path": str(snapshot_path),
        "xdmf_path": str(xdmf_path),
        "device": device_record or {},
    }
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _checkpoint_payload(state: Any) -> dict[str, Any]:
    if hasattr(state, "u") and hasattr(state, "g"):
        return {"u": state.u, "g": state.g}
    if hasattr(state, "u"):
        return {
            "u": state.u,
            "p": state.p,
            "nonlinear_old": state.nonlinear_old,
            "have_old": jnp.asarray(state.have_old, dtype=jnp.float32),
        }
    if hasattr(state, "x"):
        return {
            "x": state.x,
            "p": state.p,
            "nonlinear_old": state.nonlinear_old,
            "have_old": jnp.asarray(state.have_old, dtype=jnp.float32),
        }
    raise TypeError(f"unsupported checkpoint state type {type(state).__name__}")


def _growth_rate_from_energy(e0: Any, e1: Any, steps: int, dt: float) -> float:
    elapsed = int(steps) * float(dt)
    if elapsed <= 0.0:
        raise ValueError("growth-rate diagnostics require at least one DNS step")
    e0f = float(e0)
    e1f = float(e1)
    if not (math.isfinite(e0f) and math.isfinite(e1f)) or e0f <= 0.0 or e1f <= 0.0:
        if math.isfinite(e0f) and e0f > 0.0 and e1f == 0.0:
            return -math.inf
        return math.nan
    return 0.5 * math.log(e1f / e0f) / elapsed


def _selected_resolution(spec: dict[str, Any]) -> dict[str, Any]:
    resolution = spec["resolution"]
    if any(key in resolution for key in ("N", "Nr", "Nx", "Nz")):
        return resolution
    selected = resolution.get("production", resolution)
    return {**resolution, **selected}


def _padding_factor(
    resolution: dict[str, Any], *, dimensions: int
) -> tuple[float, ...]:
    dealias = resolution.get("dealias", 1.0)
    if isinstance(dealias, (list, tuple)):
        values = tuple(float(value) for value in dealias)
        if len(values) != dimensions:
            raise ValueError(f"expected {dimensions} dealias values, got {len(values)}")
        return values
    return tuple(float(dealias) for _ in range(dimensions))


def _assert_pcf_half_gap_domain(spec: dict[str, Any]) -> None:
    x0, x1 = (float(value) for value in spec["domain"]["x"])
    if not (
        math.isclose(x0, -1.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(x1, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ProductionOracleNotImplementedError(
            "PCF primitive-b DNS is wired for the shenfun half-gap domain "
            f"[-1, 1], got [{x0:g}, {x1:g}]"
        )


def _pcf_state_from_components(template: Any, x: tuple[Any, ...]) -> Any:
    import jax.numpy as jnp

    return type(template)(
        x=x,
        p=jnp.zeros_like(template.p),
        nonlinear_old=tuple(jnp.zeros_like(component) for component in x),
        have_old=False,
    )


def _pcf_mhd_perturbation_state(
    solver: Any, spec: dict[str, Any]
) -> tuple[Any, complex]:
    initial = spec["initial_condition"]
    state, eigenvalue = solver.seed_linear_eigenmode(
        ky_mode=int(initial.get("ky_mode", 1)),
        kz_mode=int(initial.get("kz_mode", 1)),
        amp=1.0,
    )
    velocity_amplitude = float(initial.get("velocity_amplitude", 0.1))
    magnetic_amplitude = float(initial.get("magnetic_amplitude", velocity_amplitude))
    x = tuple(
        (velocity_amplitude if i < 3 else magnetic_amplitude) * component
        for i, component in enumerate(state.x)
    )
    return _pcf_state_from_components(state, x), eigenvalue


def _pcf_mri_packet_state(solver: Any, spec: dict[str, Any]) -> tuple[Any, complex]:
    initial = spec["initial_condition"]
    seeded_modes = initial.get("seeded_modes", {})
    ky_mode = int(seeded_modes.get("ky", 0))
    kz_modes = seeded_modes.get("kz", [1])
    if not kz_modes:
        raise ValueError("mri_eigenmode_packet requires at least one kz mode")
    amplitude = float(initial.get("amplitude", 1.0e-3)) / math.sqrt(len(kz_modes))
    states = []
    eigenvalues = []
    for kz_mode in kz_modes:
        state, eigenvalue = solver.seed_linear_eigenmode(
            ky_mode=ky_mode, kz_mode=int(kz_mode), amp=amplitude
        )
        states.append(state)
        eigenvalues.append(eigenvalue)
    x = tuple(
        sum((state.x[i] for state in states[1:]), states[0].x[i])
        for i in range(len(states[0].x))
    )
    eigenvalue = max(eigenvalues, key=lambda value: value.real)
    return _pcf_state_from_components(states[0], x), eigenvalue


def _pcf_primitive_3d_scalars(solver: Any, state: Any) -> dict[str, float]:
    import jax.numpy as jnp

    diag = solver.diagnostics(state)
    fields = solver.fields_physical(state)
    butterfly_by_mean = jnp.mean(jnp.real(fields[4]))
    return {
        "kinetic_energy": float(diag["Ekin"]),
        "magnetic_energy": float(diag["Emag"]),
        "total_energy": float(diag["E"]),
        "divergence_u_l2": float(diag["divu"]),
        "divergence_b_l2": float(diag["divb"]),
        "maxwell_stress_xy": float(diag["maxwell_stress"]),
        "reynolds_stress": float(diag["reynolds_stress"]),
        "transport_alpha": float(diag["transport_alpha"]),
        "butterfly_by_mean": float(butterfly_by_mean),
    }


def _pcf_fluctuation_scalars(solver: Any, state: Any) -> dict[str, float]:
    import jax.numpy as jnp

    diag = solver.diagnostics(state)
    up = solver._backward_velocity(state.u)
    total_dv_dx = solver.TD.backward_primitive(state.u[1], (1, 0, 0)) + solver.dUb_dx
    streak = jnp.sqrt(jnp.mean(jnp.real(up[1] * jnp.conj(up[1]))))
    roll = jnp.sqrt(
        jnp.mean(jnp.real(up[0] * jnp.conj(up[0]) + up[2] * jnp.conj(up[2])))
    )
    return {
        "kinetic_energy": float(diag["Epert"]),
        "total_kinetic_energy": float(diag["Etot"]),
        "divergence_l2": float(diag["divL2"]),
        "wall_shear_lower": float(jnp.mean(jnp.real(total_dv_dx[0, :, :]))),
        "wall_shear_upper": float(jnp.mean(jnp.real(total_dv_dx[-1, :, :]))),
        "wall_velocity_lower": float(diag["u_bot"]),
        "wall_velocity_upper": float(diag["u_top"]),
        "mean_shear": float(diag["mean_shear"]),
        "streak_rms": float(streak),
        "roll_rms": float(roll),
    }


def _radial_velocity_linf(solver: Any, state: Any) -> float:
    import jax.numpy as jnp

    velocity = solver.velocity_physical(state)
    return float(jnp.max(jnp.abs(velocity[0])))


def _tc_mhd_stresses(solver: Any, state: Any) -> tuple[float, float]:
    import jax.numpy as jnp

    from jaxfun.galerkin.inner import integrate

    fields = solver.fields_physical(state)
    ur, ut = fields[0], fields[1]
    br, bt = fields[3], fields[4]
    reynolds = jnp.real(integrate(jnp.real(ur * jnp.conj(ut)) * solver.R, solver.T0))
    maxwell = -jnp.real(integrate(jnp.real(br * jnp.conj(bt)) * solver.R, solver.T0))
    return float(reynolds), float(maxwell)


def _tc_inner_torque(solver: Any, state: Any) -> float:
    import jax.numpy as jnp

    r_inner = float(solver.base.R1)
    b = float(solver.base.b)
    nu = float(solver.nu)
    dut_dr = solver.TD.backward_primitive(state.u[1], (0, 1))
    ut = solver.TD.backward(state.u[1])
    perturbation_shear = jnp.mean(dut_dr[:, 0] - ut[:, 0] / r_inner)
    base_shear = -2.0 * b / (r_inner**2)
    return float(2.0 * math.pi * nu * r_inner**2 * abs(base_shear + perturbation_shear))


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


def _kz_mode_from_spec(spec: dict[str, Any], Lz: float, *, strict: bool = True) -> int:
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
    if strict and not math.isclose(resolved, kz, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            f"axial_wavenumber={kz!r} does not map to an integer Fourier "
            f"mode for Lz={Lz!r}"
        )
    return kz_mode
