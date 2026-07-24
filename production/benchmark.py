"""3-D performance benchmark harness + cost model (FJ-12).

Correctness gates do not make a campaign affordable. This harness measures the cost
of a solver so a campaign's GPU-hours can be predicted and enforced. It deliberately
separates:

* **compile/factorization** time (first traced step / operator build),
* **warm steady-state** time per step (median over a timed window),
* **checkpoint/diagnostic I/O** (measured by the caller and passed in),
* **peak memory** (best-effort from the JAX live-buffer stats).

and reports **cost per simulated shear time** rather than only per step. The workhorse
*decision* remains gated on an authorized GPU run -- this module only produces the
measurements and a fitted cost model; it does not select a solver.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StepTiming:
    label: str
    compile_s: float
    warm_step_s: float
    warm_step_p50_s: float
    warm_step_p90_s: float
    timed_steps: int
    dt: float
    peak_bytes: int | None
    rollout_steps: int = 1
    generated_code_bytes: int | None = None
    temp_bytes: int | None = None
    argument_bytes: int | None = None
    output_bytes: int | None = None
    alias_bytes: int | None = None
    rollout_cache_info: dict[str, Any] | None = None
    compilation_cache_info: dict[str, Any] | None = None
    dt_transition_probe: dict[str, Any] | None = None
    state_checksum: dict[str, float | int | str] | None = None
    final_diagnostics: dict[str, float] | None = None

    @property
    def cost_per_shear_time_s(self) -> float:
        """Wall-seconds to advance one shear time ``S*t = 1`` (dt in shear units)."""
        if self.dt <= 0.0:
            return float("nan")
        return self.warm_step_s / self.dt

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timed_blocks"] = self.timed_steps
        d["total_timed_steps"] = self.timed_steps * self.rollout_steps
        d["cost_per_shear_time_s"] = self.cost_per_shear_time_s
        return d


def _block_until_ready(state: Any) -> None:
    try:
        import jax

        jax.block_until_ready(state)
    except (AttributeError, TypeError):  # pragma: no cover - non-JAX states
        pass


def _peak_bytes() -> int | None:
    try:
        import jax

        for device in jax.devices():
            stats = device.memory_stats() if hasattr(device, "memory_stats") else None
            if stats and "peak_bytes_in_use" in stats:
                return int(stats["peak_bytes_in_use"])
    except Exception:  # pragma: no cover
        return None
    return None


def _advance(solver: Any, state: Any, steps: int) -> Any:
    """Advance through the production rollout when one is available."""

    solve = getattr(solver, "solve", None)
    if callable(solve):
        return solve(state, int(steps))
    for _ in range(int(steps)):
        state = solver.step(state)
    return state


def _rollout_cache_info(solver: Any) -> dict[str, Any] | None:
    info_fn = getattr(solver, "rollout_cache_info", None)
    if not callable(info_fn):
        return None
    try:
        return asdict(info_fn())
    except (AttributeError, TypeError):
        return None


def _persistent_cache_snapshot() -> dict[str, Any]:
    """Return a best-effort snapshot of JAX persistent cache artifacts."""

    try:
        import jax

        configured = getattr(jax.config, "jax_compilation_cache_dir", None)
        if not configured:
            return {"enabled": False, "path": None, "artifact_count": 0, "bytes": 0}
        path = Path(str(configured))
        files = [entry for entry in path.rglob("*") if entry.is_file()]
        return {
            "enabled": True,
            "path": str(path),
            "artifact_count": len(files),
            "bytes": sum(entry.stat().st_size for entry in files),
        }
    except (OSError, TypeError, ValueError):  # pragma: no cover - best effort
        return {"enabled": False, "path": None, "artifact_count": 0, "bytes": 0}


def _persistent_cache_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    return {
        **after,
        "new_artifact_count": max(
            0, int(after["artifact_count"]) - int(before["artifact_count"])
        ),
        "new_bytes": max(0, int(after["bytes"]) - int(before["bytes"])),
    }


def _probe_dt_transitions(
    solver: Any, state: Any, *, rollout_steps: int, transitions: int
) -> tuple[Any, dict[str, Any] | None]:
    """Exercise same-shape factor updates and report rollout-cache reuse.

    Each transition starts from the same immutable input state. This keeps the
    probe valid for fixed-step multistep solvers: their first probe block can
    bootstrap its history at the new ``dt`` instead of consuming history from
    a different timestep. The production state is returned unchanged.
    """

    set_dt = getattr(solver, "set_dt", None)
    before = _rollout_cache_info(solver)
    if transitions <= 0 or not callable(set_dt) or before is None:
        return state, None
    base_dt = float(solver.dt)
    dt_values = [base_dt * (0.80 + 0.05 * index) for index in range(transitions)]
    try:
        for dt_value in dt_values:
            set_dt(dt_value)
            probe_state = _advance(solver, state, rollout_steps)
            _block_until_ready(probe_state)
    except NotImplementedError as exc:
        return state, {
            "transitions": 0,
            "dt_values": [],
            "supported": False,
            "reason": str(exc),
            "rollout_cache_hits_delta": 0,
            "rollout_cache_misses_delta": 0,
            "live_entries_before": int(before["live_entries"]),
            "live_entries_after": int(before["live_entries"]),
            "reused_compiled_variant": None,
        }
    set_dt(base_dt)
    after = _rollout_cache_info(solver)
    assert after is not None
    misses_delta = int(after["misses"]) - int(before["misses"])
    hits_delta = int(after["hits"]) - int(before["hits"])
    return state, {
        "transitions": int(transitions),
        "supported": True,
        "dt_values": dt_values,
        "rollout_cache_hits_delta": hits_delta,
        "rollout_cache_misses_delta": misses_delta,
        "live_entries_before": int(before["live_entries"]),
        "live_entries_after": int(after["live_entries"]),
        "reused_compiled_variant": misses_delta == 0,
    }


def _compiled_memory_analysis(
    solver: Any, state: Any, rollout_steps: int
) -> dict[str, int | None]:
    empty = {
        "generated_code_bytes": None,
        "temp_bytes": None,
        "argument_bytes": None,
        "output_bytes": None,
        "alias_bytes": None,
    }
    cache_hook = getattr(solver, "benchmark_rollout_cache", None)
    cache = (
        cache_hook()
        if callable(cache_hook)
        else getattr(solver, "_rollout_cache", None)
    )
    analyse = getattr(cache, "compiled_memory_analysis", None)
    if not callable(analyse):
        return empty
    try:
        analysis = analyse(state, rollout_steps)
    except (RuntimeError, TypeError, ValueError):
        return empty
    if analysis is None:
        return empty
    return {
        "generated_code_bytes": _optional_int(
            getattr(analysis, "generated_code_size_in_bytes", None)
        ),
        "temp_bytes": _optional_int(getattr(analysis, "temp_size_in_bytes", None)),
        "argument_bytes": _optional_int(
            getattr(analysis, "argument_size_in_bytes", None)
        ),
        "output_bytes": _optional_int(getattr(analysis, "output_size_in_bytes", None)),
        "alias_bytes": _optional_int(getattr(analysis, "alias_size_in_bytes", None)),
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _state_checksum(solver: Any, state: Any) -> dict[str, float | int | str]:
    """Return a checksum over primary fields, excluding integrator history."""

    import jax
    import jax.numpy as jnp

    fields_hook = getattr(solver, "benchmark_state_fields", None)
    fields = fields_hook(state) if callable(fields_hook) else state
    leaves = [jnp.asarray(value) for value in jax.tree.leaves(fields)]
    real_sum = 0.0
    imag_sum = 0.0
    norm2 = 0.0
    element_count = 0
    for value in leaves:
        host_value = np.asarray(jax.device_get(value))
        real_sum += float(np.sum(np.real(host_value), dtype=np.float64))
        imag_sum += float(np.sum(np.imag(host_value), dtype=np.float64))
        norm2 += float(np.sum(np.abs(host_value) ** 2, dtype=np.float64))
        element_count += int(value.size)
    return {
        "scope": "primary_fields" if callable(fields_hook) else "state_tree",
        "leaf_count": len(leaves),
        "element_count": element_count,
        "real_sum": real_sum,
        "imag_sum": imag_sum,
        "l2_norm": float(np.sqrt(norm2)),
    }


def _compact_diagnostics(solver: Any, state: Any) -> dict[str, float] | None:
    diagnostics = getattr(solver, "diagnostics", None)
    if not callable(diagnostics):
        return None
    try:
        import jax
        import jax.numpy as jnp

        values = diagnostics(state)
        jax.block_until_ready(values)
        return {
            str(name): float(jax.device_get(value))
            for name, value in values.items()
            if jnp.asarray(value).ndim == 0
        }
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def benchmark_step(
    build_solver: Callable[[], Any],
    *,
    label: str,
    warmup_steps: int = 2,
    timed_steps: int = 10,
    seed_state: Callable[[Any], Any] | None = None,
    rollout_steps: int = 1,
    dt_transition_probes: int = 3,
) -> StepTiming:
    """Benchmark compile and cached production-rollout cost per timestep.

    ``timed_steps`` is retained as the number of independently timed blocks for
    API compatibility. Each block advances ``rollout_steps`` physical
    timesteps through ``solver.solve`` when available, and reported warm times
    are divided by that block length. This measures the compiled scan used in
    production rather than Python dispatch around ``solver.step``.
    """

    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if dt_transition_probes < 0:
        raise ValueError("dt_transition_probes must be non-negative")

    persistent_cache_before = _persistent_cache_snapshot()
    t0 = time.perf_counter()
    solver = build_solver()
    state = seed_state(solver) if seed_state is not None else solver.zero_state()
    # First block includes tracing/compilation and operator factorization.
    state = _advance(solver, state, rollout_steps)
    _block_until_ready(state)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup_steps)):
        state = _advance(solver, state, rollout_steps)
    _block_until_ready(state)

    per_step: list[float] = []
    for _ in range(max(1, timed_steps)):
        s = time.perf_counter()
        state = _advance(solver, state, rollout_steps)
        _block_until_ready(state)
        per_step.append((time.perf_counter() - s) / rollout_steps)

    arr = np.asarray(per_step)
    state, dt_probe = _probe_dt_transitions(
        solver,
        state,
        rollout_steps=rollout_steps,
        transitions=dt_transition_probes,
    )
    memory = _compiled_memory_analysis(solver, state, rollout_steps)
    persistent_cache_after = _persistent_cache_snapshot()
    return StepTiming(
        label=label,
        compile_s=float(compile_s),
        warm_step_s=float(np.median(arr)),
        warm_step_p50_s=float(np.percentile(arr, 50)),
        warm_step_p90_s=float(np.percentile(arr, 90)),
        timed_steps=int(arr.size),
        dt=float(getattr(solver, "dt", float("nan"))),
        peak_bytes=_peak_bytes(),
        rollout_steps=int(rollout_steps),
        rollout_cache_info=_rollout_cache_info(solver),
        compilation_cache_info=_persistent_cache_delta(
            persistent_cache_before, persistent_cache_after
        ),
        dt_transition_probe=dt_probe,
        state_checksum=_state_checksum(solver, state),
        final_diagnostics=_compact_diagnostics(solver, state),
        **memory,
    )


@dataclass(frozen=True)
class CostModel:
    """Power-law fit ``warm_step_s ~ a * dof^b`` over degrees of freedom."""

    a: float
    b: float
    dofs: list[float]
    times: list[float]

    def predict(self, dof: float) -> float:
        return float(self.a * dof**self.b)

    def relative_error(self, dof: float, observed: float) -> float:
        pred = self.predict(dof)
        return abs(pred - observed) / observed if observed else float("nan")


def fit_cost_model(dofs: list[float], warm_step_times: list[float]) -> CostModel:
    """Fit a log-log power law of per-step time vs total degrees of freedom."""

    d = np.asarray(dofs, dtype=float)
    t = np.asarray(warm_step_times, dtype=float)
    if d.size < 2 or np.any(d <= 0) or np.any(t <= 0):
        raise ValueError("need >=2 positive (dof, time) samples to fit a cost model")
    b, log_a = np.polyfit(np.log(d), np.log(t), 1)
    return CostModel(a=float(np.exp(log_a)), b=float(b), dofs=list(d), times=list(t))


def predicted_gpu_hours(
    warm_step_s: float, *, shear_times: float, dt: float, io_overhead_frac: float = 0.0
) -> float:
    """Predicted GPU-hours to advance ``shear_times`` at step cost ``warm_step_s``."""

    if dt <= 0.0:
        return float("nan")
    steps = shear_times / dt
    seconds = steps * warm_step_s * (1.0 + io_overhead_frac)
    return seconds / 3600.0


# ---------------------------------------------------------------------------
# Real-measurement path (review blocker 6: the harness must measure the actual
# production solvers, not only synthetic stand-ins).
# ---------------------------------------------------------------------------


def _solver_and_seed_builders(spec: dict[str, Any]):
    """Return ``(build_solver, seed_state)`` for a spec's production solver.

    Uses the same construction helpers as the oracles so the measured cost is the
    production solver's, byte-identical configuration included.
    """

    representation = spec.get("representation")
    geometry = spec.get("geometry")
    physics = spec.get("physics")
    if geometry == "pcf" and representation == "vector_potential":
        from production.oracles import _curl_solver_from_spec

        return (
            lambda: _curl_solver_from_spec(spec),
            lambda solver: solver.initial_state(),
        )
    if geometry == "pcf" and physics in {"mhd", "mri"}:
        from production.oracles import (
            _pcf_mhd_perturbation_state,
            _pcf_mri_packet_state,
            _primitive_solver_from_spec,
        )

        def seed(solver: Any) -> Any:
            if physics == "mri":
                return _pcf_mri_packet_state(solver, spec)[0]
            return _pcf_mhd_perturbation_state(solver, spec)[0]

        return (lambda: _primitive_solver_from_spec(spec)), seed
    if geometry == "pcf" and physics == "hydro":
        from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
        from production.oracles import (
            DEFAULT_FLOW_BASIS_FAMILY,
            _padding_factor,
            _pcf_kmm_time_integrator,
            _selected_resolution,
        )

        resolution = _selected_resolution(spec)
        groups = spec["nondimensional_groups"]
        time_integrator = _pcf_kmm_time_integrator(
            spec, solver_family="PCF hydrodynamic KMM benchmark"
        )
        domain = (
            tuple(float(value) for value in spec["domain"]["x"]),
            (0.0, float(spec["domain"]["y_period"])),
            (0.0, float(spec["domain"]["z_period"])),
        )

        def build_pcf_hydro() -> Any:
            return PlaneCouetteFluctuationJax(
                N=(
                    int(resolution.get("Nx", resolution.get("N", 32))),
                    int(resolution.get("Ny", 64)),
                    int(resolution.get("Nz", 32)),
                ),
                domain=domain,
                Re=float(groups["Re"]),
                U_wall=float(groups.get("U_wall", 1.0)),
                dt=float(spec["time"]["dt"]),
                family=resolution.get("family", DEFAULT_FLOW_BASIS_FAMILY),
                padding_factor=_padding_factor(resolution, solver_family="pcf_kmm"),
                perturbation_amplitude=float(
                    spec["initial_condition"].get("amplitude", 0.1)
                ),
                time_integrator=time_integrator,
            )

        return build_pcf_hydro, lambda solver: solver.initial_state()
    if geometry == "taylor_couette":
        from examples.taylor_couette_dns_jax import (
            AxisymmetricMRIDNSJax,
            AxisymmetricTCDNSJax,
            CircularCouette,
            TaylorCouetteDNSJax,
            TaylorCouetteMRIDNSJax,
        )
        from production.oracles import (
            DEFAULT_FLOW_BASIS_FAMILY,
            _kz_mode_from_spec,
            _resolved_physics,
            _selected_resolution,
            _tc_vp_solver_from_spec,
        )

        resolution = _selected_resolution(spec)
        groups = spec["nondimensional_groups"]
        resolved = _resolved_physics(spec)
        base_args = (
            float(groups["R1"]),
            float(groups["R2"]),
            float(groups["Omega1"]),
            float(groups["Omega2"]),
        )
        nr = int(resolution.get("Nr", resolution.get("N", 40)))
        nz = int(resolution.get("Nz", 16))
        ntheta = resolution.get("Ntheta")
        family = spec["resolution"].get(
            "family", resolution.get("family", DEFAULT_FLOW_BASIS_FAMILY)
        )
        dealias = float(spec["resolution"].get("dealias", 1.0))
        lz = float(spec["domain"]["z_period"])
        dt = float(spec["time"]["dt"])
        time_integrator = str(spec["time"].get("integrator", "CNAB2"))
        azimuthal_mode = int(spec.get("mode", {}).get("azimuthal_wavenumber", 0))
        amplitude = float(spec["initial_condition"].get("amplitude", 1.0e-4))

        if representation == "vector_potential":
            initial = spec["initial_condition"]

            def seed_tc_vp(solver: Any) -> Any:
                if "seeded_kz_mode" in initial:
                    kz_mode = int(initial["seeded_kz_mode"])
                elif "mode" in spec:
                    kz_mode = _kz_mode_from_spec(spec, solver.Lz, strict=False)
                else:
                    kz_mode = 1
                state, _eigenvalue = solver.seed_linear_eigenmode(
                    m=int(initial.get("azimuthal_mode", 0)),
                    kz_mode=kz_mode,
                    amp=amplitude,
                )
                symmetry_amp = float(initial.get("symmetry_break_amplitude", 0.0))
                if symmetry_amp > 0.0:
                    state = solver.add_symmetry_breaking_perturbation(
                        state,
                        symmetry_amp,
                        m=int(initial.get("symmetry_break_m", 1)),
                        kz_mode=kz_mode,
                    )
                return state

            return (lambda: _tc_vp_solver_from_spec(spec)), seed_tc_vp

        if physics == "hydro":
            solver_cls = AxisymmetricTCDNSJax if ntheta is None else TaylorCouetteDNSJax

            def build_tc_hydro() -> Any:
                kwargs = dict(
                    base=CircularCouette(*base_args),
                    nu=resolved.nu,
                    Nr=nr,
                    Nz=nz,
                    Lz=lz,
                    dt=dt,
                    family=family,
                    dealias=dealias,
                    time_integrator=time_integrator,
                )
                if ntheta is not None:
                    kwargs["Ntheta"] = int(ntheta)
                return solver_cls(**kwargs)

            def seed_tc_hydro(solver: Any) -> Any:
                mode = spec.get("mode", {})
                if "axial_wavenumber" not in mode:
                    kwargs = {"amp": amplitude}
                    if ntheta is not None:
                        kwargs["m"] = azimuthal_mode
                    return solver.initial_state(**kwargs)
                kwargs = {
                    "kz_mode": _kz_mode_from_spec(spec, solver.Lz, strict=False),
                    "amp": amplitude,
                }
                if ntheta is not None:
                    kwargs["m"] = azimuthal_mode
                return solver.seed_linear_eigenmode(**kwargs)[0]

            return build_tc_hydro, seed_tc_hydro

        solver_cls = AxisymmetricMRIDNSJax if ntheta is None else TaylorCouetteMRIDNSJax

        def build_tc_mhd() -> Any:
            kwargs = dict(
                base=CircularCouette(*base_args),
                B0=resolved.B0,
                nu=resolved.nu,
                eta_mag=resolved.eta if resolved.eta is not None else resolved.nu,
                Nr=nr,
                Nz=nz,
                Lz=lz,
                dt=dt,
                family=family,
                dealias=dealias,
                time_integrator=time_integrator,
            )
            if ntheta is not None:
                kwargs["Ntheta"] = int(ntheta)
            return solver_cls(**kwargs)

        def seed_tc_mhd(solver: Any) -> Any:
            kwargs = {
                "kz_mode": _kz_mode_from_spec(spec, solver.Lz, strict=False),
                "amp": amplitude,
            }
            if ntheta is not None:
                kwargs["m"] = azimuthal_mode
            return solver.seed_linear_eigenmode(**kwargs)[0]

        return build_tc_mhd, seed_tc_mhd
    raise ValueError(
        f"no benchmark solver factory for {spec.get('problem_id')!r} "
        f"(geometry={geometry!r}, physics={physics!r}, "
        f"representation={representation!r})"
    )


def _spec_dof(spec: dict[str, Any]) -> float:
    """Nominal degrees of freedom: grid size x number of evolved fields."""

    from production.oracles import _selected_resolution

    resolution = _selected_resolution(spec)
    if "Nr" in resolution:
        dimension_keys = ("Nr", "Ntheta", "Nz")
    elif "Nx" in resolution:
        dimension_keys = ("Nx", "Ny", "Nz")
    else:
        dimension_keys = ("N",)
    cells = 1.0
    for key in dimension_keys:
        value = resolution.get(key)
        if value is not None:
            cells *= float(int(value))
    fields = max(1, len(spec.get("evolved_variables", [])))
    return cells * fields


def measure_spec(
    config_path: str | Any,
    *,
    tiers: tuple[str, ...] = ("smoke",),
    timed_steps: int = 10,
    warmup_steps: int = 2,
    shear_times: float | None = None,
    holdout_tier: str | None = None,
    rollout_steps: int = 25,
    dt_transition_probes: int = 3,
) -> dict[str, Any]:
    """FJ-12: measure the real production solver at materialized resolution tiers.

    Returns an artifact dict with per-tier :class:`StepTiming` measurements, the
    fitted cost model when >= 2 tiers were measured, and predicted GPU-hours for
    the spec's configured horizon. CPU measurements bound the campaign shape; the
    workhorse *decision* still requires the authorized GPU run of this same CLI.
    """

    from production.adapters import load_config

    measurements: list[dict[str, Any]] = []
    dofs: list[float] = []
    times: list[float] = []
    spec: dict[str, Any] = {}
    for tier in tiers:
        config = load_config(config_path, resolution_tier=tier)
        spec = config.spec
        build_solver, seed_state = _solver_and_seed_builders(spec)
        timing = benchmark_step(
            build_solver,
            label=f"{spec['problem_id']}@{tier}",
            warmup_steps=warmup_steps,
            timed_steps=timed_steps,
            seed_state=seed_state,
            rollout_steps=rollout_steps,
            dt_transition_probes=dt_transition_probes,
        )
        horizon = (
            float(shear_times)
            if shear_times is not None
            else float(spec["time"]["final_time"])
        )
        dof = _spec_dof(spec)
        measurements.append(
            {
                "tier": tier,
                "dof": dof,
                **timing.to_dict(),
                "predicted_hours_full_horizon": predicted_gpu_hours(
                    timing.warm_step_s, shear_times=horizon, dt=timing.dt
                ),
            }
        )
        dofs.append(dof)
        times.append(timing.warm_step_s)

    artifact: dict[str, Any] = {
        "schema_version": 3,
        "problem_id": spec.get("problem_id"),
        "spec_hash": spec.get("spec_hash"),
        "backend": _backend_name(),
        "timed_steps": int(timed_steps),
        "warmup_steps": int(warmup_steps),
        "rollout_steps": int(rollout_steps),
        "dt_transition_probes": int(dt_transition_probes),
        "measurements": measurements,
        "provenance": _provenance_safe(),
    }
    if len(measurements) >= 2:
        model = fit_cost_model(dofs, times)
        artifact["cost_model"] = {
            "a": model.a,
            "b": model.b,
            "dofs": model.dofs,
            "times": model.times,
        }
    else:
        artifact["cost_model"] = None
        artifact["cost_model_note"] = (
            "need >= 2 measured tiers to fit the power-law cost model"
        )
    # Review round 3: held-out validation -- fit on the other tiers, predict the
    # held-out tier, and report the relative error so the cost model is tested
    # on a point it never saw.
    if holdout_tier is not None:
        held = [m for m in measurements if m["tier"] == holdout_tier]
        fit_rows = [m for m in measurements if m["tier"] != holdout_tier]
        if not held:
            raise ValueError(f"holdout tier {holdout_tier!r} was not measured")
        if len(fit_rows) < 2:
            raise ValueError("held-out validation needs >= 2 non-holdout tiers to fit")
        held_row = held[0]
        held_model = fit_cost_model(
            [m["dof"] for m in fit_rows], [m["warm_step_s"] for m in fit_rows]
        )
        predicted = held_model.predict(held_row["dof"])
        artifact["holdout_validation"] = {
            "tier": holdout_tier,
            "dof": held_row["dof"],
            "predicted_warm_step_s": predicted,
            "observed_warm_step_s": held_row["warm_step_s"],
            "relative_error": held_model.relative_error(
                held_row["dof"], held_row["warm_step_s"]
            ),
        }
    return artifact


def _backend_name() -> str:
    try:
        import jax

        return str(jax.default_backend())
    except Exception:  # pragma: no cover
        return "unknown"


def _provenance_safe() -> dict[str, Any]:
    try:
        from production.provenance import capture_provenance

        return capture_provenance()
    except Exception:  # pragma: no cover - provenance must never break a benchmark
        return {}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="FJ-12: benchmark the real production solver per resolution tier."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tiers", default="smoke", help="comma-separated tiers")
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=25,
        help="Physical timesteps per compiled production scan block.",
    )
    parser.add_argument(
        "--dt-transition-probes",
        type=int,
        default=3,
        help="Same-shape set_dt/rollout transitions used to detect recompilation.",
    )
    parser.add_argument("--shear-times", type=float, default=None)
    parser.add_argument(
        "--holdout-tier",
        default=None,
        help="Validate the cost model against this measured-but-unfitted tier.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    artifact = measure_spec(
        args.config,
        tiers=tuple(t.strip() for t in args.tiers.split(",") if t.strip()),
        timed_steps=args.timed_steps,
        warmup_steps=args.warmup_steps,
        shear_times=args.shear_times,
        holdout_tier=args.holdout_tier,
        rollout_steps=args.rollout_steps,
        dt_transition_probes=args.dt_transition_probes,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    for row in artifact["measurements"]:
        print(
            f"{row['label']}: dof={row['dof']:.0f} compile={row['compile_s']:.2f}s "
            f"warm_step={row['warm_step_s'] * 1000.0:.2f}ms "
            f"full_horizon={row['predicted_hours_full_horizon']:.2f}h"
        )
    print(f"wrote benchmark artifact -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
