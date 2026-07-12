"""Small production oracle executions that do not require live shenfun."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import health, observables
from .adaptive import adaptive_cfl_from_spec, run_adaptive_cfl
from .checkpoint import write_production_checkpoint


class ProductionOracleNotImplementedError(NotImplementedError):
    """Raised when a spec has no wired jaxfun production execution path yet."""


def load_resume_checkpoint(run_dir: str | Path, *, step: int | None = None):
    """Read a production checkpoint from a run directory or HDF5 file.

    ``step=None`` reads the latest checkpoint. With ``step``, a banked entry
    (``checkpoints/bank/checkpoint_<step>.h5``, written under
    ``--checkpoint-bank``) is preferred so a quench can select any retained
    plateau time, falling back to the latest file when it holds that step.
    """
    from jaxfun.io import read_checkpoint

    path = Path(run_dir)
    if path.suffix == ".h5":
        checkpoint_path = path
    else:
        checkpoint_path = path / "checkpoints" / "checkpoints.h5"
        if step is not None:
            banked = _bank_checkpoint_path(path, step)
            if banked.exists():
                checkpoint_path = banked
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found at {checkpoint_path}")
    return read_checkpoint(checkpoint_path, step=step)


def _bank_checkpoint_path(run_dir: Path, tstep: int) -> Path:
    return Path(run_dir) / "checkpoints" / "bank" / f"checkpoint_{int(tstep):08d}.h5"


def load_checkpoint_bank_index(run_dir: str | Path) -> list[dict[str, Any]]:
    """Return the parent-bank manifest entries for a run directory (FJ-05)."""

    index_path = Path(run_dir) / "checkpoints" / "bank" / "index.json"
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def _resolve_bank_entry_path(run_dir: Path, entry: dict[str, Any]) -> Path:
    """Resolve portable and legacy bank-manifest checkpoint paths."""

    tstep = int(entry.get("tstep", -1))
    expected = _bank_checkpoint_path(run_dir, tstep)
    recorded = Path(str(entry.get("checkpoint_path", "")))
    if recorded.is_absolute():
        return recorded

    # New manifests store ``checkpoints/bank/<file>`` relative to the parent
    # run.  Legacy manifests may contain an out-dir-prefixed relative path such
    # as ``runs/foo/checkpoints/bank/<file>``; its canonical three-part suffix
    # still identifies the same immutable bank slot.
    canonical_suffix = Path("checkpoints") / "bank" / expected.name
    if len(recorded.parts) >= 3 and Path(*recorded.parts[-3:]) == canonical_suffix:
        return run_dir / canonical_suffix
    return run_dir / recorded


def select_qualified_parent_checkpoint(
    run_dir: str | Path, *, step: int | None = None
) -> dict[str, Any]:
    """Select a stationary, independently sampled parent-bank entry.

    Cadence checkpoints are retained for audit and exact resume, but only an
    explicitly plateau-qualified entry may seed a quench.  With no requested
    step, the newest qualified parent is selected.
    """

    from .quench import QuenchError, file_sha256

    entries = load_checkpoint_bank_index(run_dir)
    if not entries:
        raise QuenchError(
            "quench requires a checkpoint bank with plateau qualification; "
            "run the parent with --checkpoint-bank and diagnostics cadence"
        )
    candidates = entries
    if step is not None:
        candidates = [
            entry for entry in entries if int(entry.get("tstep", -1)) == int(step)
        ]
        if not candidates:
            raise QuenchError(
                f"checkpoint bank has no entry for requested tstep={int(step)}"
            )
    rejection_reasons: dict[str, list[str]] = {}
    qualified = []
    for entry in candidates:
        stats = entry.get("plateau_window_stats", {})
        reasons = []
        if entry.get("plateau_qualified") is not True:
            reasons.append("entry is quarantined")
        if stats.get("plateau_qualified") is not True:
            reasons.extend(stats.get("qualification_reasons", []))
            if not stats.get("qualification_reasons"):
                reasons.append("missing plateau qualification")
        if stats.get("stationary") is not True:
            reasons.append("stationarity gate did not pass")
        if stats.get("diagnostics_current") is not True:
            reasons.append("diagnostics do not reach the checkpoint time")
        if stats.get("persistent_stress") is not True:
            reasons.append("persistent-stress gate did not pass")
        if stats.get("checkpoint_health_underresolved") is not False:
            reasons.append("checkpoint resolution-health gate did not pass")
        tau = stats.get("correlation_time_total_stress")
        if not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau <= 0.0:
            reasons.append("finite positive stress correlation time is missing")
        independent = stats.get("effective_independent_samples")
        required = stats.get("required_independent_samples")
        if (
            not isinstance(independent, (int, float))
            or not isinstance(required, (int, float))
            or independent < required
        ):
            reasons.append("independent-sample gate did not pass")
        expected_path = _bank_checkpoint_path(
            Path(run_dir), int(entry.get("tstep", -1))
        )
        actual_path = _resolve_bank_entry_path(Path(run_dir), entry)
        if actual_path.resolve() != expected_path.resolve() or not actual_path.exists():
            reasons.append(
                "checkpoint path is missing or outside the selected bank slot"
            )
        elif entry.get("file_sha256") != file_sha256(str(actual_path)):
            reasons.append("checkpoint SHA256 does not match the bank manifest")
        if reasons:
            rejection_reasons[str(entry.get("tstep"))] = reasons
        else:
            qualified.append(entry)
    if not qualified:
        requested = "latest" if step is None else str(int(step))
        raise QuenchError(
            f"checkpoint-bank parent {requested} is not plateau-qualified: "
            f"{rejection_reasons}"
        )
    return max(qualified, key=lambda entry: int(entry["tstep"]))


def validate_resume_checkpoint(
    record: Any,
    spec: dict[str, Any],
    device_record: dict[str, Any] | None = None,
    *,
    quench: bool = False,
) -> None:
    """Validate checkpoint metadata against the run being resumed.

    ``quench=True`` (FJ-05 continue-from) skips the strict spec_hash equality -- the
    caller has already validated via :func:`production.quench.validate_quench` that the
    child differs from the parent only in the mutable coefficient allowlist -- but the
    numerics-contract, dtype, and state-layout checks still apply.
    """
    attrs = record.attrs
    if not quench and str(attrs.get("spec_hash")) != str(spec["spec_hash"]):
        raise ValueError("resume checkpoint spec_hash does not match requested spec")
    _reject_pre_fj01_checkpoint(attrs, spec)
    dtype_json = attrs.get("dtype_metadata_json", "{}")
    if isinstance(dtype_json, bytes):
        dtype_json = dtype_json.decode()
    dtype_metadata = json.loads(dtype_json or "{}")
    expected_dtype = dtype_metadata.get("production_run_dtype")
    active_dtype = (
        None if device_record is None else device_record.get("production_run_dtype")
    )
    if (
        active_dtype is not None
        and expected_dtype is not None
        and active_dtype != expected_dtype
    ):
        raise ValueError(
            "resume checkpoint production dtype "
            f"{expected_dtype!r} does not match active dtype {active_dtype!r}"
        )


def _reject_pre_fj01_checkpoint(attrs: Any, spec: dict[str, Any]) -> None:
    """FJ-01: forbid a pre-numerics-contract checkpoint from seeding a post-fix run."""

    from production.problem_spec import NUMERICS_CONTRACT_VERSION

    spec_version = spec.get("numerics_contract_version")
    if spec_version is None:
        return  # the spec itself is pre-contract; legacy resume stays permitted.
    ckpt_version = attrs.get("numerics_contract_version")
    try:
        ckpt_version = int(ckpt_version) if ckpt_version is not None else 0
    except (TypeError, ValueError):
        ckpt_version = 0
    if ckpt_version < int(spec_version):
        raise ValueError(
            "resume checkpoint is pre-FJ-01 (numerics_contract_version "
            f"{ckpt_version} < {NUMERICS_CONTRACT_VERSION}); it must not seed a "
            "post-fix production continuation. Regenerate under the current contract."
        )


def _resume_or_initial_state(
    resume_checkpoint: Any | None,
    initial_state: Any,
    *,
    spec: dict[str, Any],
    state_kind: str,
    quench: bool = False,
) -> tuple[Any, int, float]:
    if resume_checkpoint is None:
        return initial_state, 0, 0.0
    validate_resume_checkpoint(resume_checkpoint, spec, quench=quench)
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
    if state_kind == "pcf_vector_potential_mhd_saturation":
        from examples.channelflow_kmm import KMMState
        from examples.pcf_mhd_jax import MHDState

        return MHDState(
            flow=KMMState(u=tuple(payload["flow_u"]), g=payload["flow_g"]),
            A=tuple(payload["A"]),
        )
    if state_kind == "tc_vector_potential_mhd_saturation":
        from examples.taylor_couette_vp_jax import TCVPState

        return TCVPState(
            u=tuple(payload["u"]),
            p=payload["p"],
            A=tuple(payload["A"]),
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
    quench: bool = False,
    on_row: Any | None = None,
    checkpoint_bank: bool = False,
    burn_in_steps: int = 0,
) -> dict[str, Any]:
    """Run a supported production spec and return canonical diagnostics.

    ``on_row`` is an optional host-side callback invoked with each canonical
    cadence row as it is produced (FJ-07 live telemetry); the materialized
    ``time_series`` stays the source of truth. ``checkpoint_bank`` retains an
    immutable per-interval checkpoint bank (FJ-05 parent-bank workflow), and
    ``burn_in_steps`` excludes the first steps after a resume/quench from the
    stationarity/classification analysis window.
    """

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
            on_row=on_row,
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
        and spec.get("representation") == "vector_potential"
    ):
        return _run_taylor_couette_vp_mhd_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
            on_row=on_row,
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
            on_row=on_row,
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
            on_row=on_row,
        )
    if (
        spec["geometry"] == "pcf"
        and spec["physics"] in {"mhd", "mri"}
        and spec.get("representation") == "vector_potential"
    ):
        return _run_pcf_vector_potential_mhd_saturation(
            spec,
            steps=steps,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            device_record=device_record,
            resume_checkpoint=resume_checkpoint,
            quench=quench,
            on_row=on_row,
            checkpoint_bank=checkpoint_bank,
            burn_in_steps=burn_in_steps,
        )
    if (
        spec["geometry"] == "pcf"
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
            quench=quench,
            on_row=on_row,
            checkpoint_bank=checkpoint_bank,
            burn_in_steps=burn_in_steps,
        )
    if (
        spec["geometry"] == "pcf"
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
            quench=quench,
            on_row=on_row,
            checkpoint_bank=checkpoint_bank,
            burn_in_steps=burn_in_steps,
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
    on_row: Any | None = None,
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
    initial_state = state
    initial = solver.diagnostics(initial_state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["E"]),
            "divergence_l2": float(diag.get("continuity_l2", diag["div_linf"])),
            "divergence_linf": float(diag["div_linf"]),
        }
        diagnostic_rows.append(row)
        if on_row is not None:
            on_row(row)

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
        on_diagnostics_row=collect_diagnostics,
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
        "radial_velocity_linf": _radial_velocity_linf(solver, initial_state),
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

    resolution = _selected_resolution(spec)
    is_hydro = spec["physics"] == "hydro"
    # FJ-00: coefficients come from the single resolved-physics object, so a
    # Re/Rm-only spec runs identically to a nu/eta one instead of KeyError'ing.
    physics = _resolved_physics(spec)
    if not is_hydro:
        magnetic_bc = _magnetic_bc(spec)
        if magnetic_bc != "conducting":
            raise ProductionOracleNotImplementedError(
                "the axisymmetric primitive DNS runner is wired for conducting "
                f"magnetic walls only, got {magnetic_bc!r}; pseudo-vacuum runs "
                "use the 3-D primitive saturation family (FJ-09)"
            )
    solver = AxisymmetricPCFMRIDNSJax(
        S=physics.S,
        omega=physics.Omega,
        B0=0.0 if is_hydro else physics.B0,
        nu=physics.nu,
        eta_mag=physics.nu if physics.eta is None else physics.eta,
        Nx=int(resolution.get("Nx", resolution.get("N", 40))),
        Nz=int(resolution.get("Nz", 16)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "C"),
        # FJ-01: honor the spec's dealias contract on the 2-D (z, x) layout.
        dealias=_axisymmetric_dealias(resolution),
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
        "energy_convention": "half_integral_abs2",
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
    on_row: Any | None = None,
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
        padding_factor=_padding_factor(resolution, solver_family="pcf_kmm"),
        perturbation_amplitude=float(spec["initial_condition"].get("amplitude", 0.1)),
    )
    state = solver.initial_state()
    initial = _pcf_fluctuation_scalars(solver, state)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["Epert"]),
            "total_kinetic_energy": float(diag["Etot"]),
            "divergence_l2": float(diag["divL2"]),
            "mean_shear": float(diag["mean_shear"]),
        }
        diagnostic_rows.append(row)
        if on_row is not None:
            on_row(row)

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
        on_diagnostics_row=collect_diagnostics,
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
    quench: bool = False,
    on_row: Any | None = None,
    checkpoint_bank: bool = False,
    burn_in_steps: int = 0,
) -> dict[str, Any]:
    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc not in {"conducting", "pseudo_vacuum"}:
        raise ProductionOracleNotImplementedError(
            "PCF primitive-b saturation runner is wired for conducting or "
            f"pseudo-vacuum walls, got {magnetic_bc!r}"
        )
    _assert_pcf_half_gap_domain(spec)
    solver = _primitive_solver_from_spec(spec)
    if spec["physics"] == "mri":
        state, eigenvalue = _pcf_mri_packet_state(solver, spec)
    else:
        state, eigenvalue = _pcf_mhd_perturbation_state(solver, spec)

    _assert_nonaxisymmetric_seed(solver, state, spec)
    diagnostic_rows: list[dict[str, Any]] = []

    def collect_diagnostics(t: float, _tstep: int, diag: dict[str, Any]) -> None:
        # FJ-06: the cadence row keeps the stresses/mean-flux/non-axisymmetric
        # signals the solver already computes, not just energies + divergence.
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["Ekin"]),
            "magnetic_energy": float(diag["Emag"]),
            "total_energy": float(diag["E"]),
            "divergence_u_l2": float(diag["divu"]),
            "divergence_b_l2": float(diag["divb"]),
            "reynolds_stress": float(diag["reynolds_stress"]),
            "maxwell_stress_xy": float(diag["maxwell_stress"]),
            "total_stress": float(diag["total_stress"]),
            "alpha_Sh": float(diag["alpha_Sh"]),
            "mag_energy_mean": float(diag["mag_energy_mean"]),
            "mag_energy_fluct": float(diag["mag_energy_fluct"]),
            "mean_bx": float(diag["mean_bx"]),
            "mean_by": float(diag["mean_by"]),
            "mean_bz": float(diag["mean_bz"]),
            "nonaxisymmetric_fraction": float(diag["nonaxisymmetric_fraction"]),
            "energy_convention": "half_integral_abs2",
        }
        if "transport_alpha" in diag:
            row["transport_alpha"] = float(diag["transport_alpha"])
        diagnostic_rows.append(row)
        if on_row is not None:
            on_row(row)

    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="pcf_primitive_mhd_saturation",
        quench=quench,
    )
    # FJ-05: baseline scalars must come from the state that is actually evolved (the
    # loaded parent checkpoint on a quench), not the fresh configured seed, so growth
    # rate / magnetic-growth / flux drift reference the correct starting point.
    initial = _pcf_primitive_3d_scalars(solver, state)
    target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )
    health_observations: list[dict[str, float]] = []
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
        on_diagnostics_row=collect_diagnostics,
        t0=t0,
        tstep0=tstep0,
        health_block=_PRODUCTION_HEALTH_BLOCK,
        health_scalars_fn=health.primitive_health_scalars,
        health_observations=health_observations,
        checkpoint_bank=checkpoint_bank,
        plateau_rows=diagnostic_rows,
    )
    final = _pcf_primitive_3d_scalars(solver, out)
    # Growth is measured from the evolved baseline over the steps actually taken
    # (n_steps == target_steps on a fresh run; smaller on a resume/quench).
    growth_rate = _growth_rate_from_energy(
        initial["total_energy"], final["total_energy"], n_steps, solver.dt
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
    # FJ-04: mean magnetic-flux drift from the initial state (net-flux monitor).
    flux_drift = {
        f"flux_drift_b{axis}": final[f"mean_b{axis}"] - initial[f"mean_b{axis}"]
        for axis in ("x", "y", "z")
    }
    scalars = {
        **final,
        **flux_drift,
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
        "magnetic_bc": magnetic_bc,
        # E* scalars here are the physical 0.5 * volume integral of |field|^2
        # (couette/pcf_mri_primitive.py convention); the curl family reports 2x.
        "energy_convention": "half_integral_abs2",
    }
    first = _pcf_primitive_summary_row(
        initial, t=float(t0), extra={"growth_rate_linear": float(eigenvalue.real)}
    )
    last = _pcf_primitive_summary_row(
        final,
        t=elapsed,
        extra={
            "growth_rate": float(growth_rate),
            "magnetic_energy_growth_factor": float(magnetic_growth),
            "saturation_check_passed": bool(saturation_passed),
        },
    )
    series = _dedupe_time_rows([first, *diagnostic_rows, last])
    # FJ-05: exclude the post-quench burn-in window from the fitted history.
    analysis_rows, analysis_start = _analysis_window(
        diagnostic_rows,
        t0=t0,
        dt=float(spec["time"]["dt"]),
        burn_in_steps=burn_in_steps,
    )
    fit_series = (
        series if analysis_start is None else _dedupe_time_rows([*analysis_rows, last])
    )
    scalars.update(_stationarity_scalars(fit_series, key="total_energy"))
    if analysis_start is not None:
        scalars["analysis_burn_in_steps"] = int(burn_in_steps)
        scalars["analysis_t_start"] = float(analysis_start)
    # Resolution health is enforced every compiled block; report the maximum
    # observed load, not merely a potentially benign final value.
    scalars.update(_max_health_observations(health_observations))
    tau = health.correlation_time(analysis_rows)
    if tau is not None:
        scalars["correlation_time_total_stress"] = float(tau)
        independent = health.effective_independent_samples(
            analysis_rows, correlation=tau
        )
        if independent is not None:
            scalars["effective_independent_samples_total_stress"] = float(independent)
    return {"scalars": scalars, "time_series": series}


def _pcf_primitive_summary_row(
    scalars: dict[str, float], *, t: float, extra: dict[str, Any]
) -> dict[str, Any]:
    """Build a first/last time-series row from primitive-DNS scalars (FJ-04-safe)."""

    keys = (
        "kinetic_energy",
        "magnetic_energy",
        "total_energy",
        "divergence_u_l2",
        "divergence_b_l2",
        "maxwell_stress_xy",
        "reynolds_stress",
        "total_stress",
        "alpha_Sh",
        "mag_energy_mean",
        "mag_energy_fluct",
        "mean_bx",
        "mean_by",
        "mean_bz",
        "nonaxisymmetric_fraction",
    )
    row: dict[str, Any] = {"t": float(t)}
    for key in keys:
        if key in scalars:
            row[key] = scalars[key]
    if "transport_alpha" in scalars:
        row["transport_alpha"] = scalars["transport_alpha"]
    row.update(extra)
    return row


def _primitive_solver_from_spec(spec: dict[str, Any]):
    """Construct the primitive-b 3-D solver exactly as the saturation oracle does.

    Shared with the FJ-12 benchmark CLI so measured costs are the production
    solver's, not a stand-in's.
    """

    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

    resolution = _selected_resolution(spec)
    physics = _resolved_physics(spec)
    return PCFMRIDNSJax(
        S=physics.S,
        omega=physics.Omega,
        B0=physics.B0,
        nu=physics.nu,
        eta_mag=physics.eta,
        Nx=int(resolution.get("Nx", resolution.get("N", 32))),
        Ny=int(resolution.get("Ny", 8)),
        Nz=int(resolution.get("Nz", 16)),
        Ly=float(spec["domain"]["y_period"]),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "L"),
        dealias=_padding_factor(resolution, solver_family="pcf_primitive"),
        magnetic_bc=_magnetic_bc(spec),
    )


def _curl_solver_from_spec(spec: dict[str, Any]):
    """Construct the vector-potential workhorse exactly as the oracle does."""

    from examples.pcf_mhd_mri_shearpy_jax import (
        PlaneCouetteMRIShearpyInsulatingJax,
        PlaneCouetteMRIShearpyJax,
    )

    magnetic_bc = _magnetic_bc(spec)
    solver_cls = {
        "conducting": PlaneCouetteMRIShearpyJax,
        "insulating": PlaneCouetteMRIShearpyInsulatingJax,
    }[magnetic_bc]
    resolution = _selected_resolution(spec)
    physics = _resolved_physics(spec)
    initial = spec["initial_condition"]
    x0, x1 = (float(v) for v in spec["domain"]["x"])
    return solver_cls(
        N=(
            int(resolution.get("Nx", resolution.get("N", 17))),
            int(resolution.get("Ny", 16)),
            int(resolution.get("Nz", 16)),
        ),
        domain=(
            (x0, x1),
            (0.0, float(spec["domain"]["y_period"])),
            (0.0, float(spec["domain"]["z_period"])),
        ),
        Re=physics.Re_h,
        Rm=physics.Rm_h if physics.Rm_h is not None else physics.Re_h,
        omega=physics.Omega,
        shear_rate=physics.S,
        background_b=(0.0, 0.0, physics.B0),
        dt=float(spec["time"]["dt"]),
        family=resolution.get("family", "L"),
        padding_factor=_padding_factor(
            resolution, solver_family="pcf_vector_potential"
        ),
        perturbation_amplitude=float(initial.get("velocity_amplitude", 0.05)),
        magnetic_amplitude=float(initial.get("magnetic_amplitude", 0.0)),
    )


def _box_volume(solver: Any) -> float:
    """Box volume for converting family-convention energies to physical means."""

    if hasattr(solver, "_volume"):
        return float(solver._volume)
    domain = solver.domain
    return float(
        (domain[0][1] - domain[0][0])
        * (domain[1][1] - domain[1][0])
        * (domain[2][1] - domain[2][0])
    )


def _pcf_curl_scalars(solver: Any, state: Any) -> dict[str, float]:
    """Canonical scalars from the curl/vector-potential MRI solver (FJ-03)."""

    diag = solver.diagnostics(state)
    scalars = {
        "box_volume": _box_volume(solver),
        "kinetic_energy": float(diag["Epert"]),
        "magnetic_energy": float(diag["Emag"]),
        "magnetic_energy_total": float(diag["Emag_total"]),
        "total_energy": float(diag["Epert"]) + float(diag["Emag"]),
        "divergence_u_l2": float(diag["divL2"]),
        "divergence_b_l2": float(diag["divB_L2"]),
        "maxwell_stress_xy": float(diag["maxwell_stress"]),
        "reynolds_stress": float(diag["reynolds_stress"]),
        "total_stress": float(diag["total_stress"]),
        "alpha_Sh": float(diag["alpha_Sh"]),
        # FJ-04 mean-flux / mean-fluctuating split (mean-flux contamination monitor)
        "mean_bx": float(diag["mean_bx"]),
        "mean_by": float(diag["mean_by"]),
        "mean_bz": float(diag["mean_bz"]),
        "mag_energy_mean": float(diag["mag_energy_mean"]),
        "mag_energy_fluct": float(diag["mag_energy_fluct"]),
    }
    if "alpha" in diag:  # net-flux alpha only when B0 != 0 (ZNF-safe)
        scalars["transport_alpha"] = float(diag["alpha"])
        scalars["alpha_B0"] = float(diag["alpha_B0"])
    if "insulating_bc_residual" in diag:
        scalars["insulating_bc_residual"] = float(diag["insulating_bc_residual"])
    return scalars


def _run_pcf_vector_potential_mhd_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
    quench: bool = False,
    on_row: Any | None = None,
    checkpoint_bank: bool = False,
    burn_in_steps: int = 0,
) -> dict[str, Any]:
    """FJ-03: curl/vector-potential PCF-MRI DNS.

    The magnetic field is ``B = curl A`` and is solenoidal by construction.

    The zero-net-flux workhorse. MHDState (KMM flow block + A coefficients)
    checkpoint, snapshot, resume-exact, and quench continuation run through the
    shared ``_solve_with_optional_checkpoints`` machinery, and the per-block health
    guards always run (``health_block``) with or without a diagnostics cadence.
    """

    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc not in {"conducting", "insulating"}:
        raise ProductionOracleNotImplementedError(
            "the vector-potential (curl) PCF-MRI family is wired for conducting "
            f"or insulating walls, got {magnetic_bc!r}; pseudo-vacuum runs use "
            "the primitive-b family (FJ-09)"
        )
    _assert_pcf_half_gap_domain(spec)
    solver = _curl_solver_from_spec(spec)
    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        solver.initial_state(),
        spec=spec,
        state_kind="pcf_vector_potential_mhd_saturation",
        quench=quench,
    )
    # FJ-05: baseline scalars come from the state actually evolved (the loaded
    # parent checkpoint on a resume/quench), not the fresh configured seed.
    first_scalars = _pcf_curl_scalars(solver, state)
    _target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )

    rows: list[dict[str, Any]] = []

    def collect(t: float, tstep: int, diag: dict[str, Any]) -> None:
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["Epert"]),
            "magnetic_energy": float(diag["Emag"]),
            "total_energy": float(diag["Epert"]) + float(diag["Emag"]),
            "divergence_u_l2": float(diag["divL2"]),
            "divergence_b_l2": float(diag["divB_L2"]),
            "reynolds_stress": float(diag["reynolds_stress"]),
            "maxwell_stress_xy": float(diag["maxwell_stress"]),
            "total_stress": float(diag["total_stress"]),
            "alpha_Sh": float(diag["alpha_Sh"]),
            "mean_bx": float(diag["mean_bx"]),
            "mean_by": float(diag["mean_by"]),
            "mean_bz": float(diag["mean_bz"]),
            "mag_energy_mean": float(diag["mag_energy_mean"]),
            "mag_energy_fluct": float(diag["mag_energy_fluct"]),
            "dissipation_kinetic": float(diag["dissipation_kinetic"]),
            "dissipation_magnetic": float(diag["dissipation_magnetic"]),
            "energy_convention": "integral_abs2",
            "dt": float(solver.dt),
            **({"transport_alpha": float(diag["alpha"])} if "alpha" in diag else {}),
            **(
                {"insulating_bc_residual": float(diag["insulating_bc_residual"])}
                if "insulating_bc_residual" in diag
                else {}
            ),
            **({"cfl_total": float(diag["cfl_total"])} if "cfl_total" in diag else {}),
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)

    health_observations: list[dict[str, float]] = []
    adaptive = adaptive_cfl_from_spec(spec)
    adaptive_record: dict[str, Any] | None = None
    if adaptive is not None:
        # Experimental adaptive-CFL path (chunked dt adaptation with exact
        # time accounting).  Fresh starts only; cadence IO reduces to
        # per-check diagnostics rows plus a final checkpoint.
        if resume_checkpoint is not None or quench or checkpoint_bank:
            raise ProductionOracleNotImplementedError(
                "adaptive_cfl runs are wired for fresh starts without the "
                "checkpoint-bank/quench machinery (experimental)"
            )
        out, adaptive_record = _run_vp_adaptive_blocks(
            solver,
            state,
            steps=n_steps,
            config=adaptive,
            spec=spec,
            health_scalars_fn=health.curl_health_scalars,
            diagnostics_row_fn=collect,
            health_observations=health_observations,
            t0=t0,
            tstep0=tstep0,
            out_dir=out_dir,
            state_kind="pcf_vector_potential_mhd_saturation",
            device_record=device_record,
        )
    else:
        out = _solve_with_optional_checkpoints(
            solver,
            state,
            n_steps,
            spec=spec,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            state_kind="pcf_vector_potential_mhd_saturation",
            device_record=device_record,
            on_diagnostics_row=collect,
            t0=t0,
            tstep0=tstep0,
            health_block=_PRODUCTION_HEALTH_BLOCK,
            health_scalars_fn=health.curl_health_scalars,
            health_observations=health_observations,
            checkpoint_bank=checkpoint_bank,
            plateau_rows=rows,
        )
    final_scalars = _pcf_curl_scalars(solver, out)
    if adaptive_record is not None:
        elapsed = float(adaptive_record["elapsed_time"])
        growth_rate = _growth_rate_from_elapsed(
            first_scalars["total_energy"], final_scalars["total_energy"], elapsed
        )
    else:
        elapsed = n_steps * float(spec["time"]["dt"])
        # Growth is measured from the evolved baseline over the steps actually
        # taken (n_steps == target_steps on a fresh run; smaller on a resume).
        growth_rate = _growth_rate_from_energy(
            first_scalars["total_energy"],
            final_scalars["total_energy"],
            n_steps,
            solver.dt,
        )
    magnetic_growth = (
        final_scalars["magnetic_energy"] / first_scalars["magnetic_energy"]
        if first_scalars["magnetic_energy"] > 0.0
        else 0.0
    )
    saturation_passed = _saturation_passed(
        magnetic_growth,
        threshold=2.0,
        final_energies=(
            final_scalars["kinetic_energy"],
            final_scalars["magnetic_energy"],
            final_scalars["total_energy"],
            final_scalars["divergence_u_l2"],
            final_scalars["divergence_b_l2"],
        ),
    )
    flux_drift = {
        f"flux_drift_b{axis}": final_scalars[f"mean_b{axis}"]
        - first_scalars[f"mean_b{axis}"]
        for axis in ("x", "y", "z")
    }
    scalars = {
        **final_scalars,
        **flux_drift,
        "growth_rate": float(growth_rate),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
        "magnetic_bc": magnetic_bc,
        "representation": "vector_potential",
        # E* scalars here are plain volume integrals of |field|^2 (the curl-family
        # reference convention, couette/pcf_mhd_mri_shearpy.py); the primitive
        # family reports the physical 0.5x per its own reference.
        "energy_convention": "integral_abs2",
    }
    if adaptive_record is not None:
        scalars.update(_adaptive_scalars(adaptive_record))
    first = {
        "t": float(t0),
        **{k: v for k, v in first_scalars.items() if _is_number_scalar(v)},
    }
    last = {
        "t": float(t0) + elapsed,
        "dt": float(solver.dt),
        **{k: v for k, v in final_scalars.items() if _is_number_scalar(v)},
        "growth_rate": float(growth_rate),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
    }
    series = _dedupe_time_rows([first, *rows, last])
    # FJ-05: on a quench, the burn-in window is excluded from the fitted
    # history (stationarity/correlation/budget), not merely recorded.
    analysis_rows, analysis_start = _analysis_window(
        rows, t0=t0, dt=float(spec["time"]["dt"]), burn_in_steps=burn_in_steps
    )
    fit_series = (
        series if analysis_start is None else _dedupe_time_rows([*analysis_rows, last])
    )
    scalars.update(_stationarity_scalars(fit_series, key="total_energy"))
    if analysis_start is not None:
        scalars["analysis_burn_in_steps"] = int(burn_in_steps)
        scalars["analysis_t_start"] = float(analysis_start)
    # Resolution health is enforced every compiled block; report maxima over
    # the trajectory so a transient violation cannot disappear at final time.
    scalars.update(_max_health_observations(health_observations))
    tau = health.correlation_time(analysis_rows)
    if tau is not None:
        scalars["correlation_time_total_stress"] = float(tau)
        independent = health.effective_independent_samples(
            analysis_rows, correlation=tau
        )
        if independent is not None:
            scalars["effective_independent_samples_total_stress"] = float(independent)
    budget = health.energy_budget_residual(
        analysis_rows,
        shear_rate=float(solver.shear_rate),
        volume=scalars["box_volume"],
    )
    if budget is not None:
        scalars["energy_budget_residual"] = float(budget)
    return {"scalars": scalars, "time_series": series}


def _is_number_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    magnetic = spec.get("boundary_conditions", {}).get("magnetic", "conducting")
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
    on_row: Any | None = None,
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
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["Ekin"]),
            "magnetic_energy": float(diag["Emag"]),
            "divergence_u": float(diag["divu"]),
            "divergence_b": float(diag["divb"]),
            "divergence_b_l2": float(diag["divb"]),
        }
        diagnostic_rows.append(row)
        if on_row is not None:
            on_row(row)

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
        on_diagnostics_row=collect_diagnostics,
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


def _tc_vp_solver_from_spec(spec: dict[str, Any]):
    """Construct the vector-potential TC solver exactly as the oracle does."""

    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    resolution = _selected_resolution(spec)
    groups = spec["nondimensional_groups"]
    return TaylorCouetteVPMRIDNSJax(
        CircularCouette(
            float(groups["R1"]),
            float(groups["R2"]),
            float(groups["Omega1"]),
            float(groups["Omega2"]),
        ),
        B0=float(groups.get("B0", spec.get("forcing", {}).get("B0", 0.1))),
        nu=float(groups["nu"]),
        eta_mag=float(groups.get("eta_mag", groups["nu"])),
        Nr=int(resolution.get("Nr", resolution.get("N", 24))),
        Ntheta=int(resolution.get("Ntheta", 8)),
        Nz=int(resolution.get("Nz", 16)),
        Lz=float(spec["domain"]["z_period"]),
        dt=float(spec["time"]["dt"]),
        family=spec["resolution"].get("family", resolution.get("family", "L")),
        dealias=float(spec["resolution"].get("dealias", 1.5)),
        magnetic_bc=_magnetic_bc(spec),
    )


def _tc_vp_scalars(solver: Any, state: Any) -> dict[str, float]:
    """Canonical scalars of the vector-potential TC family.

    Energies are the physical 0.5 * integral |field|^2 r dr dtheta dz over the
    full annulus (the 2*pi azimuthal integral included -- the axisymmetric
    primitive family omits it, so cross-family comparisons must convert).
    """

    diag = solver.diagnostics(state)
    scalars = {
        "box_volume": float(solver._volume()),
        "kinetic_energy": float(diag["Ekin"]),
        "magnetic_energy": float(diag["Emag"]),
        "total_energy": float(diag["E"]),
        "divergence_u": float(diag["divu"]),
        "divergence_b": float(diag["divb"]),
        "divergence_u_l2": float(diag["divu_l2"]),
        "divergence_b_l2": float(diag["divb_l2"]),
        "reynolds_stress": float(diag["reynolds_rt"]),
        "maxwell_stress_rt": float(diag["maxwell_rt"]),
        "total_stress": float(diag["total_stress"]),
        "mean_bz": float(diag["mean_bz"]),
    }
    if "insulating_bc_residual" in diag:
        scalars["insulating_bc_residual"] = float(diag["insulating_bc_residual"])
    return scalars


def _run_taylor_couette_vp_mhd_saturation(
    spec: dict[str, Any],
    *,
    steps: int | None = None,
    out_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    snapshot_every: int | None = None,
    diagnostics_every: int | None = None,
    device_record: dict[str, Any] | None = None,
    resume_checkpoint: Any | None = None,
    on_row: Any | None = None,
) -> dict[str, Any]:
    """Vector-potential (curl) Taylor-Couette MHD/MRI DNS runner.

    ``B = B0 e_z + curl(A)`` in full 3D ``(theta, z, r)``; conducting or
    insulating cylinders; seeded from the matching linear eigensolver.  The
    solenoidal witness (``divergence_b_l2``) is the divergence of the
    projected coefficient representation of ``b`` and must stay at its
    resolution floor for the whole horizon -- the invariant the primitive-b
    family cannot hold at finite amplitude.
    """

    magnetic_bc = _magnetic_bc(spec)
    if magnetic_bc not in {"conducting", "insulating"}:
        raise ProductionOracleNotImplementedError(
            "the vector-potential TC family is wired for conducting or "
            f"insulating cylinders, got {magnetic_bc!r}"
        )
    solver = _tc_vp_solver_from_spec(spec)
    initial = spec["initial_condition"]
    m_seed = int(initial.get("azimuthal_mode", 0))
    if magnetic_bc == "insulating" and m_seed != 0:
        raise ProductionOracleNotImplementedError(
            "insulating TC *eigenmode seeding* is anchored to the m=0 flux "
            "eigensolver; use initial_condition.symmetry_break_amplitude to "
            "populate non-axisymmetric modes on top of the m=0 seed"
        )
    if "seeded_kz_mode" in initial:
        kz_mode = int(initial["seeded_kz_mode"])
    elif "mode" in spec:
        kz_mode = _kz_mode_from_spec(spec, solver.Lz, strict=False)
    else:
        kz_mode = 1
    state, eigenvalue = solver.seed_linear_eigenmode(
        m=m_seed,
        kz_mode=kz_mode,
        amp=float(initial.get("amplitude", 1.0e-4)),
    )
    # An axisymmetric eigenmode seed stays axisymmetric under nonlinear
    # evolution, so production runs superpose a small non-axisymmetric
    # solenoidal perturbation by default spec so the full 3D dynamics (and,
    # for insulating walls, the non-axisymmetric vacuum-matching rows) are
    # actually exercised.
    sb_amp = float(initial.get("symmetry_break_amplitude", 0.0))
    if sb_amp > 0.0:
        state = solver.add_symmetry_breaking_perturbation(
            state,
            sb_amp,
            m=int(initial.get("symmetry_break_m", 1)),
            kz_mode=kz_mode,
        )
    state, tstep0, t0 = _resume_or_initial_state(
        resume_checkpoint,
        state,
        spec=spec,
        state_kind="tc_vector_potential_mhd_saturation",
    )
    first_scalars = _tc_vp_scalars(solver, state)
    _target_steps, n_steps = _remaining_steps_from_resume(
        spec, steps=steps, tstep0=tstep0
    )

    rows: list[dict[str, Any]] = []

    def collect(t: float, tstep: int, diag: dict[str, Any]) -> None:
        row = {
            "t": float(t),
            "kinetic_energy": float(diag["Ekin"]),
            "magnetic_energy": float(diag["Emag"]),
            "total_energy": float(diag["E"]),
            "divergence_u_l2": float(diag["divu_l2"]),
            "divergence_b_l2": float(diag["divb_l2"]),
            "reynolds_stress": float(diag["reynolds_rt"]),
            "maxwell_stress_rt": float(diag["maxwell_rt"]),
            "total_stress": float(diag["total_stress"]),
            "mean_bz": float(diag["mean_bz"]),
            "dt": float(solver.dt),
            **(
                {"insulating_bc_residual": float(diag["insulating_bc_residual"])}
                if "insulating_bc_residual" in diag
                else {}
            ),
            **({"cfl_total": float(diag["cfl_total"])} if "cfl_total" in diag else {}),
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)

    health_observations: list[dict[str, float]] = []
    adaptive = adaptive_cfl_from_spec(spec)
    adaptive_record: dict[str, Any] | None = None
    if adaptive is not None:
        if resume_checkpoint is not None:
            raise ProductionOracleNotImplementedError(
                "adaptive_cfl runs are wired for fresh starts (experimental)"
            )
        out, adaptive_record = _run_vp_adaptive_blocks(
            solver,
            state,
            steps=n_steps,
            config=adaptive,
            spec=spec,
            health_scalars_fn=health.tc_curl_health_scalars,
            diagnostics_row_fn=collect,
            health_observations=health_observations,
            t0=t0,
            tstep0=tstep0,
            out_dir=out_dir,
            state_kind="tc_vector_potential_mhd_saturation",
            device_record=device_record,
        )
    else:
        out = _solve_with_optional_checkpoints(
            solver,
            state,
            n_steps,
            spec=spec,
            out_dir=out_dir,
            checkpoint_every=checkpoint_every,
            snapshot_every=snapshot_every,
            diagnostics_every=diagnostics_every,
            state_kind="tc_vector_potential_mhd_saturation",
            device_record=device_record,
            on_diagnostics_row=collect,
            t0=t0,
            tstep0=tstep0,
            health_block=_PRODUCTION_HEALTH_BLOCK,
            health_scalars_fn=health.tc_curl_health_scalars,
            health_observations=health_observations,
        )
    final_scalars = _tc_vp_scalars(solver, out)
    if adaptive_record is not None:
        elapsed = float(adaptive_record["elapsed_time"])
        growth_rate = _growth_rate_from_elapsed(
            first_scalars["total_energy"], final_scalars["total_energy"], elapsed
        )
    else:
        elapsed = n_steps * float(spec["time"]["dt"])
        growth_rate = _growth_rate_from_energy(
            first_scalars["total_energy"],
            final_scalars["total_energy"],
            n_steps,
            solver.dt,
        )
    magnetic_growth = (
        final_scalars["magnetic_energy"] / first_scalars["magnetic_energy"]
        if first_scalars["magnetic_energy"] > 0.0
        else 0.0
    )
    saturation_passed = _saturation_passed(
        magnetic_growth,
        threshold=2.0,
        final_energies=(
            final_scalars["kinetic_energy"],
            final_scalars["magnetic_energy"],
            final_scalars["total_energy"],
            final_scalars["divergence_u_l2"],
            final_scalars["divergence_b_l2"],
        ),
    )
    scalars = {
        **final_scalars,
        "growth_rate": float(growth_rate),
        "growth_rate_linear": float(eigenvalue.real),
        "magnetic_energy_growth_factor": float(magnetic_growth),
        "saturation_check_passed": bool(saturation_passed),
        "magnetic_bc": magnetic_bc,
        "representation": "vector_potential",
        # 0.5 * integral |field|^2 r dr dtheta dz over the full annulus; the
        # axisymmetric primitive family omits the 2*pi azimuthal factor.
        "energy_convention": "half_integral_abs2_annulus",
    }
    if adaptive_record is not None:
        scalars.update(_adaptive_scalars(adaptive_record))
    first = {
        "t": float(t0),
        **{k: v for k, v in first_scalars.items() if _is_number_scalar(v)},
        "growth_rate_linear": float(eigenvalue.real),
    }
    last = {
        "t": float(t0) + elapsed,
        "dt": float(solver.dt),
        **{k: v for k, v in final_scalars.items() if _is_number_scalar(v)},
        "growth_rate": float(growth_rate),
        "magnetic_energy_growth_factor": float(magnetic_growth),
    }
    series = _dedupe_time_rows([first, *rows, last])
    scalars.update(_stationarity_scalars(series, key="magnetic_energy"))
    scalars.update(_max_health_observations(health_observations))
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
    health_block: int | None = None,
    health_scalars_fn: Any | None = None,
    health_observations: list[dict[str, float]] | None = None,
    checkpoint_bank: bool = False,
    plateau_rows: list[dict[str, Any]] | None = None,
) -> Any:
    if checkpoint_every is not None and checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if snapshot_every is not None and snapshot_every <= 0:
        raise ValueError("snapshot_every must be positive")
    if diagnostics_every is not None and diagnostics_every <= 0:
        raise ValueError("diagnostics_every must be positive")
    if (checkpoint_every is not None or snapshot_every is not None) and out_dir is None:
        raise ValueError(
            "out_dir is required when checkpoint or snapshot output is set"
        )

    monitor_every = _monitor_every(
        steps,
        checkpoint_every=checkpoint_every,
        snapshot_every=snapshot_every,
        diagnostics_every=diagnostics_every,
    )
    if (
        monitor_every is None
        and health_block is None
        and not (checkpoint_every or snapshot_every or diagnostics_every)
    ):
        return solver.solve(state, steps)

    from jaxfun.io import Cadence, generate_xdmf

    out_path = None if out_dir is None else Path(out_dir)
    checkpoint_path = None
    diagnostics_path = None
    snapshot_path = None
    if out_path is not None:
        checkpoint_path = out_path / "checkpoints" / "checkpoints.h5"
        diagnostics_path = out_path / "diagnostics.jsonl"
        snapshot_path = out_path / "snapshots" / "snapshots.h5"

    health_cache: dict[int, dict[str, float]] = {}

    def evaluate_health(t: float, tstep: int, candidate_state: Any) -> dict[str, float]:
        cached = health_cache.get(int(tstep))
        if cached is not None:
            return cached
        if health_scalars_fn is None:
            return {}
        values = {
            str(key): float(value)
            for key, value in health_scalars_fn(solver, candidate_state).items()
        }
        health_cache[int(tstep)] = values
        if health_observations is not None:
            health_observations.append(values)
        _raise_on_resolution_health(
            values,
            t=t,
            tstep=tstep,
            enforce_spectral=(int(tstep) - int(tstep0))
            >= health.EARLY_ABORT_STARTUP_STEPS,
        )
        return values

    def on_checkpoint(t: float, tstep: int, checkpoint_state: Any) -> None:
        assert checkpoint_path is not None
        # Callback ordering writes checkpoints before should_stop.  Enforce the
        # resolution guard here as well so a rejected state is never promoted
        # into a selectable parent bank.
        checkpoint_health = evaluate_health(t, tstep, checkpoint_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": _checkpoint_payload(checkpoint_state)}
        write_production_checkpoint(
            checkpoint_path,
            payload,
            t=t,
            tstep=tstep,
            spec=spec,
            state_kind=state_kind,
            device_record=device_record,
            diagnostics_path=diagnostics_path,
        )
        if checkpoint_bank:
            _write_bank_checkpoint(
                checkpoint_path.parent,
                payload,
                t=t,
                tstep=tstep,
                spec=spec,
                state_kind=state_kind,
                device_record=device_record,
                diagnostics_path=diagnostics_path,
                plateau_stats=_checkpoint_plateau_stats(
                    plateau_rows or [],
                    checkpoint_time=float(t),
                    health_scalars=checkpoint_health,
                ),
            )

    def on_snapshot(t: float, tstep: int, snapshot_state: Any) -> None:
        assert snapshot_path is not None
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload, snapshot_spaces = _snapshot_payload(solver, snapshot_state)
        _write_atomic_uniform_snapshot(
            snapshot_path,
            snapshot_payload,
            t=t,
            tstep=tstep,
            spaces=snapshot_spaces,
            attrs={
                "problem_id": spec["problem_id"],
                "spec_hash": spec["spec_hash"],
                "state_kind": state_kind,
            },
        )

    diagnostics_cache: dict[int, Any] = {}

    def on_diagnostics(t: float, tstep: int, diag: Any) -> None:
        diagnostics_cache[int(tstep)] = diag
        if on_diagnostics_row is not None:
            on_diagnostics_row(t, tstep, diag)

    def should_stop(t: float, tstep: int, candidate_state: Any) -> bool:
        if not _tree_all_finite(candidate_state):
            raise FloatingPointError(
                f"nonfinite solver state at tstep={int(tstep)} t={float(t):g}"
            )
        diag = diagnostics_cache.pop(int(tstep), None)
        if diag is None and health_block is not None:
            # FJ-06: an always-on health cadence computes real diagnostics per
            # block so the energy/divergence guards run even without a
            # diagnostics cadence.
            diagnostics_fn = getattr(solver, "diagnostics", None)
            if diagnostics_fn is not None:
                diag = diagnostics_fn(candidate_state)
        # FJ-06: catch a finite-but-runaway energy in addition to non-finite state.
        _raise_on_energy_runaway(
            solver, candidate_state, t=t, tstep=tstep, diagnostics=diag
        )
        _raise_on_divergence_drift(
            solver, candidate_state, t=t, tstep=tstep, diagnostics=diag
        )
        evaluate_health(t, tstep, candidate_state)
        return False

    cadence = Cadence(
        diagnostics_every=diagnostics_every,
        snapshot_every=snapshot_every,
        checkpoint_every=checkpoint_every,
    )
    block_size = monitor_every or max(1, int(steps))
    if health_block is not None:
        block_size = max(1, min(block_size, int(health_block)))
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
    values = [
        float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))
    ]
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


def _analysis_window(
    rows: list[dict[str, Any]], *, t0: float, dt: float, burn_in_steps: int
) -> tuple[list[dict[str, Any]], float | None]:
    """FJ-05: cadence rows inside the analysis window (post burn-in).

    Returns ``(rows, None)`` when no burn-in applies; otherwise the rows at
    ``t >= t0 + burn_in_steps * dt`` and that start time.
    """

    if int(burn_in_steps) <= 0:
        return list(rows), None
    start = float(t0) + int(burn_in_steps) * float(dt)
    kept = [row for row in rows if float(row.get("t", -math.inf)) >= start - 1.0e-12]
    return kept, start


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
    return None


_DIVERGENCE_GUARD_LIMIT = 1.0e-2


_ENERGY_RUNAWAY_LIMIT = 1.0e30

# Cap on how many steps either production PCF family advances between full
# CFL/spectral/occupancy checks.
_PRODUCTION_HEALTH_BLOCK = 50


def _raise_on_resolution_health(
    scalars: dict[str, float],
    *,
    t: float,
    tstep: int,
    enforce_spectral: bool = True,
    enforce_cfl: bool = True,
) -> None:
    """Abort as soon as CFL, spectral tails, or occupancy violate policy."""

    offenders = []
    if (
        enforce_spectral
        and scalars.get("spectral_tail_max", 0.0)
        > health.EARLY_ABORT_SPECTRAL_TAIL_LIMIT
    ):
        offenders.append(
            f"spectral_tail_max={scalars['spectral_tail_max']:.6g} > "
            f"{health.EARLY_ABORT_SPECTRAL_TAIL_LIMIT:g}"
        )
    if enforce_cfl and scalars.get("cfl_total", 0.0) > health.CFL_LIMIT:
        offenders.append(f"cfl_total={scalars['cfl_total']:.6g} > {health.CFL_LIMIT:g}")
    if (
        enforce_spectral
        and scalars.get("mode_occupancy", 0.0) > health.EARLY_ABORT_MODE_OCCUPANCY_LIMIT
    ):
        offenders.append(
            f"mode_occupancy={scalars['mode_occupancy']:.6g} > "
            f"{health.EARLY_ABORT_MODE_OCCUPANCY_LIMIT:g}"
        )
    if not offenders:
        return
    raise RuntimeError(
        f"underresolved health guard failed at tstep={int(tstep)} t={float(t):g}: "
        + ", ".join(offenders)
    )


def _max_health_observations(
    observations: list[dict[str, float]],
) -> dict[str, float]:
    """Maximum finite value of each health component over checked blocks."""

    keys = {key for row in observations for key in row}
    return {
        key: max(float(row[key]) for row in observations if key in row)
        for key in sorted(keys)
    }


def _checkpoint_plateau_stats(
    rows: list[dict[str, Any]],
    *,
    checkpoint_time: float,
    health_scalars: dict[str, float],
) -> dict[str, Any]:
    """Combine stationarity/sampling and strict resolution health for banking."""

    stats = health.plateau_window_stats(rows, checkpoint_time=checkpoint_time)
    underresolved = health.underresolved_from_scalars(health_scalars)
    stats["checkpoint_health"] = dict(health_scalars)
    stats["checkpoint_health_underresolved"] = underresolved
    if underresolved is not False:
        stats["plateau_qualified"] = False
        stats["qualification_reasons"].append(
            "checkpoint failed strict CFL/spectral-tail/occupancy health"
        )
    return stats


def _raise_on_energy_runaway(
    solver: Any, state: Any, *, t: float, tstep: int, diagnostics: Any
) -> None:
    """FJ-06: stop a run whose energy blows up while still finite."""

    energy: float | None = None
    if isinstance(diagnostics, dict):
        for key in ("E", "total_energy", "Etot", "Ekin"):
            value = diagnostics.get(key)
            if value is not None:
                try:
                    energy = float(value)
                except (TypeError, ValueError):
                    energy = None
                if energy is not None:
                    break
    if energy is None:
        energy_fn = getattr(solver, "energy", None)
        if energy_fn is None:
            return
        try:
            energy = float(energy_fn(state))
        except Exception:  # pragma: no cover - energy not computable for this solver
            return
    if not math.isfinite(energy) or abs(energy) > _ENERGY_RUNAWAY_LIMIT:
        raise FloatingPointError(
            f"energy runaway ceiling exceeded at tstep={int(tstep)} "
            f"t={float(t):g}: E={energy:g} > {_ENERGY_RUNAWAY_LIMIT:g}"
        )


def _raise_on_divergence_drift(
    solver: Any,
    state: Any,
    *,
    t: float,
    tstep: int,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    if diagnostics is None:
        diagnostics_fn = getattr(solver, "diagnostics", None)
        if diagnostics_fn is None:
            return
        diagnostics = diagnostics_fn(state)
    diag = diagnostics
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


_DIVERGENCE_KEY_RE = re.compile(
    r"^(?:"
    r"divergence(?:[_-](?:u|b))?(?:[_-]?(?:l2|linf|norm|rms|max))?"
    r"|div(?:u|b)(?:[_-]?(?:l2|linf|norm|rms|max))?"
    r"|div(?:[_-]?(?:l2|linf|norm|rms|max))"
    r"|continuity(?:[_-]?(?:residual|l2|linf|norm|rms|max))?"
    r")$"
)


def _is_divergence_key(key: str) -> bool:
    name = key.rsplit(".", 1)[-1].lower()
    return _DIVERGENCE_KEY_RE.match(name) is not None


def _tree_all_finite(tree: Any) -> bool:
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return True
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves if hasattr(leaf, "dtype")]
    if not checks:
        return True
    return bool(jax.device_get(jnp.all(jnp.asarray(checks))))


def _write_atomic_uniform_snapshot(
    snapshot_path: Path,
    fields: dict[str, Any],
    *,
    t: float,
    tstep: int,
    spaces: dict[str, Any] | None,
    attrs: dict[str, Any],
) -> None:
    from jaxfun.io import write_uniform_snapshot

    step_path = _snapshot_step_path(snapshot_path, tstep)
    step_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = step_path.with_name(f".{step_path.name}.tmp")
    try:
        write_uniform_snapshot(
            tmp,
            fields,
            t=t,
            tstep=tstep,
            spaces=spaces,
            attrs=attrs,
            mode="w",
        )
        tmp.replace(step_path)
        _update_snapshot_index(snapshot_path, step_path=step_path, tstep=tstep)
    finally:
        if tmp.exists():
            tmp.unlink()


def _snapshot_step_path(snapshot_path: Path, tstep: int) -> Path:
    return snapshot_path.parent / "steps" / f"snapshot_{int(tstep):08d}.h5"


def _update_snapshot_index(snapshot_path: Path, *, step_path: Path, tstep: int) -> None:
    import h5py

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
    entries: dict[str, tuple[str, str]] = {}
    latest_step: int | None = None
    if snapshot_path.exists():
        with h5py.File(snapshot_path, "r") as existing:
            if "snapshots" in existing:
                root = existing["snapshots"]
                for step_name in root:
                    link = root.get(step_name, getlink=True)
                    if isinstance(link, h5py.ExternalLink):
                        entries[str(step_name)] = (link.filename, link.path)
                        step = int(step_name)
                        latest_step = (
                            step if latest_step is None else max(step, latest_step)
                        )
    rel = os.path.relpath(step_path, snapshot_path.parent)
    step_name = str(int(tstep))
    entries[step_name] = (rel, f"/snapshots/{step_name}")
    latest_step = int(tstep) if latest_step is None else max(int(tstep), latest_step)
    try:
        with h5py.File(tmp, "w") as h5:
            root = h5.create_group("snapshots")
            for name in sorted(entries, key=lambda item: int(item)):
                filename, link_path = entries[name]
                root[name] = h5py.ExternalLink(filename, link_path)
            root.attrs["latest_step"] = latest_step
        tmp.replace(snapshot_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _snapshot_payload(solver: Any, state: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    def real_fields(names: tuple[str, ...], values: tuple[Any, ...]) -> dict[str, Any]:
        return {
            name: jnp.real(value) for name, value in zip(names, values, strict=True)
        }

    def spaces_for(names: tuple[str, ...]) -> dict[str, Any]:
        space = _snapshot_space(solver)
        return {} if space is None else {name: space for name in names}

    if hasattr(solver, "fields_physical"):
        fields = tuple(solver.fields_physical(state))
        if len(fields) >= 6:
            names = ("u_x", "u_y", "u_z", "b_x", "b_y", "b_z")
            return real_fields(names, fields[:6]), spaces_for(names)
        if len(fields) >= 3:
            names = ("u_x", "u_y", "u_z")
            return real_fields(names, fields[:3]), spaces_for(names)
    if hasattr(solver, "velocity_physical"):
        names = ("u_x", "u_y", "u_z")
        u = tuple(solver.velocity_physical(state))
        return real_fields(names, u[:3]), spaces_for(names)
    if hasattr(solver, "total_velocity_physical"):
        names = ("u_x", "u_y", "u_z")
        u = tuple(solver.total_velocity_physical(state))
        return real_fields(names, u[:3]), spaces_for(names)
    if hasattr(solver, "_backward_velocity") and hasattr(state, "u"):
        names = ("u_x", "u_y", "u_z")
        u = tuple(solver._backward_velocity(state.u))
        return real_fields(names, u[:3]), spaces_for(names)
    raise TypeError(f"solver {type(solver).__name__} does not expose snapshot fields")


def _snapshot_space(solver: Any) -> Any | None:
    for name in ("T0", "TD", "TC", "TP"):
        space = getattr(solver, name, None)
        if space is not None and hasattr(space, "mesh"):
            return space
    return None


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


def _write_bank_checkpoint(
    checkpoint_dir: Path,
    payload: dict[str, Any],
    *,
    t: float,
    tstep: int,
    spec: dict[str, Any],
    state_kind: str,
    device_record: dict[str, Any] | None,
    diagnostics_path: Path | None,
    plateau_stats: dict[str, Any],
) -> None:
    """FJ-05: retain an immutable per-step bank entry + manifest.

    The latest ``checkpoints.h5`` is rewritten in O(1) at each interval, so
    multiple plateau times only survive when banked; a quench then selects any
    entry via ``--quench-step``.
    """

    from .quench import checkpoint_bank_entry, file_sha256, stable_manifest_json

    bank_dir = checkpoint_dir / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    target = bank_dir / f"checkpoint_{int(tstep):08d}.h5"
    write_production_checkpoint(
        target,
        payload,
        t=t,
        tstep=tstep,
        spec=spec,
        state_kind=state_kind,
        device_record=device_record,
        diagnostics_path=diagnostics_path,
    )
    entry = checkpoint_bank_entry(
        parent_run_id=str(spec["problem_id"]),
        child_run_id=None,
        t=float(t),
        tstep=int(tstep),
        spec_hash=str(spec["spec_hash"]),
        representation=str(spec.get("representation", "primitive")),
        numerics_contract_version=int(spec.get("numerics_contract_version", 0)),
        checkpoint_path=str(target.relative_to(checkpoint_dir.parent)),
        file_sha256=file_sha256(str(target)),
        plateau_stats=plateau_stats,
    )
    index_path = bank_dir / "index.json"
    entries: list[dict[str, Any]] = []
    if index_path.exists():
        entries = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [e for e in entries if int(e.get("tstep", -1)) != int(tstep)]
    entries.append(entry)
    entries.sort(key=lambda item: int(item["tstep"]))
    tmp = index_path.with_name(f".{index_path.name}.tmp")
    tmp.write_text(stable_manifest_json(entries) + "\n", encoding="utf-8")
    tmp.replace(index_path)


def _checkpoint_payload(state: Any) -> dict[str, Any]:
    if hasattr(state, "flow") and hasattr(state, "A"):
        # MHDState (curl / vector-potential family): KMM flow block + A coefficients.
        return {"flow_u": state.flow.u, "flow_g": state.flow.g, "A": state.A}
    if hasattr(state, "u") and hasattr(state, "A"):
        # TCVPState (vector-potential Taylor-Couette family).
        return {
            "u": state.u,
            "p": state.p,
            "A": state.A,
            "nonlinear_old": state.nonlinear_old,
            "have_old": jnp.asarray(state.have_old, dtype=jnp.float32),
        }
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


def _growth_rate_from_elapsed(e0: Any, e1: Any, elapsed: float) -> float:
    """Growth rate over an exact elapsed time (adaptive-dt runs)."""
    if elapsed <= 0.0:
        raise ValueError("growth-rate diagnostics require positive elapsed time")
    e0f = float(e0)
    e1f = float(e1)
    if not (math.isfinite(e0f) and math.isfinite(e1f)) or e0f <= 0.0 or e1f <= 0.0:
        if math.isfinite(e0f) and e0f > 0.0 and e1f == 0.0:
            return -math.inf
        return math.nan
    return 0.5 * math.log(e1f / e0f) / float(elapsed)


def _adaptive_scalars(record: dict[str, Any]) -> dict[str, Any]:
    """Numeric adaptive-CFL summary scalars (the per-change history lives in
    the time series rows via their ``dt``/``cfl_total`` keys)."""
    return {
        "adaptive_cfl_target": float(record["adaptive_cfl_target"]),
        "dt_final": float(record["dt_final"]),
        "dt_min_used": float(record["dt_min_used"]),
        "dt_max_used": float(record["dt_max_used"]),
        "n_dt_changes": int(record["n_dt_changes"]),
        "adaptive_steps_taken": int(record["steps_taken"]),
        "adaptive_final_step_clipped": bool(record["final_step_clipped"]),
        "cfl_total_max_observed": float(record["cfl_total_max_observed"]),
    }


def _run_vp_adaptive_blocks(
    solver: Any,
    state: Any,
    *,
    steps: int,
    config: Any,
    spec: dict[str, Any],
    health_scalars_fn: Any,
    diagnostics_row_fn: Any,
    health_observations: list[dict[str, float]],
    t0: float,
    tstep0: int,
    out_dir: str | Path | None,
    state_kind: str,
    device_record: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Adaptive-CFL block driver shared by the vector-potential runners.

    The horizon is the *elapsed time* the fixed-dt run would cover
    (``steps * spec dt``): dt adaptation changes the step count and the
    driver clips the final step to land exactly on that time, so growth
    windows and saturation horizons keep the spec contract.  Every compiled
    block produces a diagnostics row (with the exact accumulated time and
    the dt actually used) and runs the full guard set: non-finite state,
    energy runaway, divergence drift, and the resolution health gate.
    A final production checkpoint is written when an output directory is set
    (per-interval checkpoints under adaptive dt are not wired yet).
    """

    def on_block(
        t: float, done: int, block_state: Any, health_values: dict[str, float]
    ) -> None:
        tstep = int(tstep0) + int(done)
        if not _tree_all_finite(block_state):
            raise FloatingPointError(
                f"nonfinite solver state at tstep={tstep} t={float(t):g}"
            )
        diag = dict(solver.diagnostics(block_state))
        if "cfl_total" in health_values:
            diag["cfl_total"] = health_values["cfl_total"]
        _raise_on_energy_runaway(
            solver, block_state, t=t, tstep=tstep, diagnostics=diag
        )
        _raise_on_divergence_drift(
            solver, block_state, t=t, tstep=tstep, diagnostics=diag
        )
        _raise_on_resolution_health(
            health_values,
            t=t,
            tstep=tstep,
            enforce_spectral=int(done) >= health.EARLY_ABORT_STARTUP_STEPS,
            # The adaptive driver owns CFL recovery: this post-block value is
            # the pre-block measurement used to shrink before the next solve.
            # Spectral-tail and occupancy failures remain immediate here.
            enforce_cfl=False,
        )
        health_observations.append(dict(health_values))
        diagnostics_row_fn(float(t), tstep, diag)

    out, record = run_adaptive_cfl(
        solver,
        state,
        elapsed_target=int(steps) * float(spec["time"]["dt"]),
        config=config,
        health_scalars_fn=health_scalars_fn,
        on_block=on_block,
        t0=float(t0),
    )
    if out_dir is not None:
        checkpoint_path = Path(out_dir) / "checkpoints" / "checkpoints.h5"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        write_production_checkpoint(
            checkpoint_path,
            {"state": _checkpoint_payload(out)},
            t=float(t0) + float(record["elapsed_time"]),
            tstep=int(tstep0) + int(record["steps_taken"]),
            spec=spec,
            state_kind=state_kind,
            device_record=device_record,
            diagnostics_path=None,
        )
    return out, record


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
    resolution: dict[str, Any], *, solver_family: str, dimensions: int = 3
) -> tuple[float, ...]:
    """Return the native-order 3/2-rule padding for ``solver_family`` (FJ-01).

    The spec's semantic ``resolution.dealias`` (``{x, y, z}`` map or scalar) is
    remapped to the solver's native array axis order so the same spec dealiases
    correctly whether the solver is ``(y, z, x)`` (primitive) or ``(x, y, z)`` (KMM).
    """

    from production.axes import native_padding_for_solver

    return native_padding_for_solver(
        resolution, solver_family=solver_family, dimensions=dimensions
    )


def _axisymmetric_dealias(resolution: dict[str, Any]) -> float:
    """FJ-01: spec dealias for the 2-D ``(z, x)`` axisymmetric primitive solver.

    The solver applies one uniform padding factor, so a semantic per-axis map must
    agree on the axes it resolves to; a genuinely anisotropic request fails loudly
    instead of being silently truncated to one axis.
    """

    padding = _padding_factor(
        resolution, solver_family="pcf_primitive_axisymmetric", dimensions=2
    )
    if any(
        not math.isclose(p, padding[0], rel_tol=0.0, abs_tol=1.0e-12)
        for p in padding[1:]
    ):
        raise ProductionOracleNotImplementedError(
            "the axisymmetric primitive solver applies a single uniform dealias "
            f"factor; got per-axis padding {padding}"
        )
    return float(padding[0])


def _resolved_physics(spec: dict[str, Any]):
    """Return the single :class:`ResolvedPhysics` object for a spec (FJ-00)."""

    import jax

    from production.physics import resolve_physics

    precision = "float64" if jax.config.read("jax_enable_x64") else "float32"
    return resolve_physics(spec, precision=precision)


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
    """Seed the MRI eigenmode packet (FJ-02: named IC modes + non-axisymmetric seed).

    The base packet superposes axisymmetric (``k_y``-selected) eigenmodes. When the
    spec requests a 3-D nonlinear IC it must also carry a ``nonaxisymmetric_seed``
    block ``{ky_mode>=1, kz_mode, amplitude}``; that divergence-free, wall-satisfying
    ``k_y != 0`` eigenmode is superposed so the run leaves the axisymmetric invariant
    subspace. Recognized ``initial_condition.type`` values:
    ``mri_eigenmode_packet`` (axisymmetric), ``net_flux_3d_perturbed``.
    """

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
    components = [
        sum((state.x[i] for state in states[1:]), states[0].x[i])
        for i in range(len(states[0].x))
    ]
    eigenvalue = max(eigenvalues, key=lambda value: value.real)

    seed = initial.get("nonaxisymmetric_seed")
    if seed is not None:
        na_ky = int(seed.get("ky_mode", 1))
        if na_ky == 0:
            raise ValueError("nonaxisymmetric_seed.ky_mode must be nonzero")
        na_state, _ = solver.seed_linear_eigenmode(
            ky_mode=na_ky,
            kz_mode=int(seed.get("kz_mode", 1)),
            amp=float(seed.get("amplitude", amplitude)),
        )
        components = [components[i] + na_state.x[i] for i in range(len(components))]

    return _pcf_state_from_components(states[0], tuple(components)), eigenvalue


def _assert_nonaxisymmetric_seed(solver: Any, state: Any, spec: dict[str, Any]) -> None:
    """FJ-02: a nonlinear 3-D run must not start in the axisymmetric subspace."""

    if int(getattr(solver, "Ny", 1)) <= 1:
        return  # a genuinely 2-D (k_y=0) run is axisymmetric by construction.
    e_nonaxi, e_total = solver.nonaxisymmetric_energy(state)
    e_nonaxi = float(e_nonaxi)
    e_total = float(e_total)
    if e_total > 0.0 and e_nonaxi <= 1.0e-24 * e_total:
        raise ProductionOracleNotImplementedError(
            "3-D nonlinear PCF spec starts in the exactly axisymmetric (k_y=0) "
            "subspace: initial non-axisymmetric energy is "
            f"{e_nonaxi:.3e} of {e_total:.3e}. Add an initial_condition."
            "nonaxisymmetric_seed {ky_mode>=1, kz_mode, amplitude} (FJ-02)."
        )


def _pcf_primitive_3d_scalars(solver: Any, state: Any) -> dict[str, float]:
    diag = solver.diagnostics(state)
    scalars = {
        "box_volume": _box_volume(solver),
        "kinetic_energy": float(diag["Ekin"]),
        "magnetic_energy": float(diag["Emag"]),
        "total_energy": float(diag["E"]),
        "divergence_u_l2": float(diag["divu"]),
        "divergence_b_l2": float(diag["divb"]),
        "maxwell_stress_xy": float(diag["maxwell_stress"]),
        "reynolds_stress": float(diag["reynolds_stress"]),
        "total_stress": float(diag["total_stress"]),
        "alpha_Sh": float(diag["alpha_Sh"]),
        "mag_energy_total": float(diag["mag_energy_total"]),
        "mag_energy_mean": float(diag["mag_energy_mean"]),
        "mag_energy_fluct": float(diag["mag_energy_fluct"]),
        # mean flux components (FJ-04 replaces the mislabelled "butterfly_by_mean").
        "mean_bx": float(diag["mean_bx"]),
        "mean_by": float(diag["mean_by"]),
        "mean_bz": float(diag["mean_bz"]),
        "E_nonaxisymmetric": float(diag["E_nonaxisymmetric"]),
        "E_total": float(diag["E_total"]),
        "nonaxisymmetric_fraction": float(diag["nonaxisymmetric_fraction"]),
    }
    # Net-flux alpha only when an imposed field exists (ZNF runs never see NaN).
    if "transport_alpha" in diag:
        scalars["transport_alpha"] = float(diag["transport_alpha"])
        scalars["alpha_B0"] = float(diag["alpha_B0"])
    return scalars


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
