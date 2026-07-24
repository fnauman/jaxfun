"""Autograd (reverse-mode VJP) vs forward cost benchmark for the production DNS solvers.

For a chosen problem this measures, over an identical ``steps``-long compiled rollout:

* ``forward``  = wall time of ``jax.jit(loss)(state0)`` where
  ``loss(s) = quadratic_energy(solver.solve(s, steps))``;
* ``autograd`` = wall time of ``jax.jit(jax.value_and_grad(loss))(state0)`` --
  i.e. the cost of obtaining the gradient of a scalar objective w.r.t. the whole
  initial-state pytree (the adjoint/DAL sensitivity), value included;
* their ratio ``autograd/forward`` -- the "backward cost multiple".

The rollout wraps each step in ``jax.checkpoint`` inside ``lax.scan`` (see
``ScanRolloutCache``), so the backward pass rematerialises the forward: expect a
multiple of ~3-4x rather than the ~2x of a store-everything VJP, and memory that
stays bounded in ``steps``.

Compile time (first traced call) is measured separately and excluded from warm
timing. Every timed result is ``block_until_ready``-synchronised. Peak device
memory is read from the JAX live-buffer stats (high-water mark).

Env (set by the caller BEFORE importing jax):
    CUDA_VISIBLE_DEVICES=1 JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
    PYTHONPATH=<repo>:<repo>/examples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# --------------------------------------------------------------------------- #
# Problem registry: each entry returns (build_solver, seed_state, meta).
# Settings mirror production/benchmark_pcf_refactor.py and the taylor_couette
# production specs (production/examples/*.json).
# --------------------------------------------------------------------------- #

_PCF_DOMAIN = ((-1.0, 1.0), (0.0, 12.566370614359172), (0.0, 6.283185307179586))


def _build_hydro_pcf(resolution: tuple[int, int, int], integrator: str):
    from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax

    resolution = tuple(resolution) if resolution else (65, 64, 64)

    def build():
        return PlaneCouetteFluctuationJax(
            N=resolution,
            dt=1.0e-3,
            family="C",
            padding_factor=(1.0, 1.5, 1.5),
            time_integrator=integrator,
            coefficient_path="optimized",
            nonlinear_form="rotational",
            solve_batching="batched",
            Re=400.0,
            perturbation_amplitude=0.02,
        )

    meta = {
        "family": "hydrodynamic plane Couette (KMM)",
        "solver": "PlaneCouetteFluctuationJax",
        "physics": "hydro",
        "Re": 400.0,
        "resolution": list(resolution),
        "dt": 1.0e-3,
        "integrator": integrator,
        "integrator_order": 3,
        "domain": _PCF_DOMAIN,
    }
    return build, (lambda s: s.initial_state()), meta


def _build_mhd_mri_pcf(resolution: tuple[int, int, int], integrator: str):
    from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax

    resolution = tuple(resolution) if resolution else (65, 64, 64)

    def build():
        return PlaneCouetteMRIShearpyJax(
            N=resolution,
            dt=1.0e-3,
            family="C",
            padding_factor=(1.0, 1.5, 1.5),
            time_integrator=integrator,
            coefficient_path="optimized",
            nonlinear_form="rotational",
            solve_batching="batched",
            Re=400.0,
            Rm=400.0,
            perturbation_amplitude=0.02,
            magnetic_amplitude=0.005,
        )

    meta = {
        "family": "MHD/MRI plane Couette (vector-potential, div-B preserving)",
        "solver": "PlaneCouetteMRIShearpyJax",
        "physics": "mri",
        "Re": 400.0,
        "Rm": 400.0,
        "resolution": list(resolution),
        "dt": 1.0e-3,
        "integrator": integrator,
        "integrator_order": 3,
        "domain": _PCF_DOMAIN,
    }
    return build, (lambda s: s.initial_state()), meta


def _build_tc_hydro(resolution: tuple[int, int] | None, integrator: str):
    from examples.taylor_couette_dns_jax import AxisymmetricTCDNSJax, CircularCouette

    nr, nz = resolution if resolution is not None else (40, 8)

    def build():
        return AxisymmetricTCDNSJax(
            base=CircularCouette(1.0, 2.0, 1.0, 0.0),
            nu=0.01,
            Nr=nr,
            Nz=nz,
            Lz=2.0074074463832545,
            dt=1.0e-3,
            family="C",
            dealias=1.0,
            time_integrator=integrator,
        )

    meta = {
        "family": "Taylor-Couette (axisymmetric hydro DNS)",
        "solver": "AxisymmetricTCDNSJax",
        "physics": "hydro",
        "Re": 100.0,
        "resolution": [nr, nz],
        "dt": 1.0e-3,
        "integrator": integrator,
        "integrator_order": 2 if integrator == "CNAB2" else 3,
    }
    return build, (lambda s: s.initial_state(amp=1.0e-6)), meta


def _build_tc_mhd(resolution: tuple[int, int] | None, integrator: str):
    from examples.taylor_couette_dns_jax import AxisymmetricMRIDNSJax, CircularCouette

    nr, nz = resolution if resolution is not None else (40, 8)

    def build():
        return AxisymmetricMRIDNSJax(
            base=CircularCouette(1.0, 2.0, 1.0, 0.3535533905932738),
            B0=0.1,
            nu=0.001,
            eta_mag=0.001,
            Nr=nr,
            Nz=nz,
            Lz=1.0471975511965976,
            dt=2.0e-3,
            family="C",
            dealias=1.0,
            time_integrator=integrator,
        )

    meta = {
        "family": "Taylor-Couette MRI (axisymmetric MHD DNS)",
        "solver": "AxisymmetricMRIDNSJax",
        "physics": "mri",
        "Re": 1000.0,
        "Rm": 1000.0,
        "B0": 0.1,
        "resolution": [nr, nz],
        "dt": 2.0e-3,
        "integrator": integrator,
        "integrator_order": 2 if integrator == "CNAB2" else 3,
    }
    return build, (lambda s: s.seed_linear_eigenmode(kz_mode=1, amp=1.0e-7)[0]), meta


def _build_tc_hydro_3d(resolution, integrator: str):
    """Full 3D (theta, z, r) hydro Taylor-Couette DNS."""
    from examples.taylor_couette_dns_jax import CircularCouette, TaylorCouetteDNSJax

    nr, ntheta, nz = resolution if resolution else (32, 16, 16)

    def build():
        return TaylorCouetteDNSJax(
            base=CircularCouette(1.0, 2.0, 1.0, 0.0),
            nu=0.01,
            Nr=nr,
            Ntheta=ntheta,
            Nz=nz,
            Lz=2.0074074463832545,
            dt=1.0e-3,
            family="C",
            dealias=1.5,
            time_integrator=integrator,
        )

    meta = {
        "family": "Taylor-Couette (FULL 3D hydro DNS)",
        "solver": "TaylorCouetteDNSJax",
        "physics": "hydro",
        "Re": 100.0,
        "resolution": [nr, ntheta, nz],
        "dt": 1.0e-3,
        "integrator": integrator,
        "integrator_order": 2 if integrator == "CNAB2" else 3,
    }
    return build, (lambda s: s.initial_state(amp=1.0e-6)), meta


def _build_tc_mhd_3d(resolution, integrator: str):
    """Full 3D (theta, z, r) conducting-wall MHD/MRI Taylor-Couette DNS."""
    from examples.taylor_couette_dns_jax import CircularCouette, TaylorCouetteMRIDNSJax

    nr, ntheta, nz = resolution if resolution else (24, 16, 16)

    def build():
        return TaylorCouetteMRIDNSJax(
            base=CircularCouette(1.0, 2.0, 1.0, 0.3535533905932738),
            B0=0.1,
            nu=0.001,
            eta_mag=0.001,
            Nr=nr,
            Ntheta=ntheta,
            Nz=nz,
            Lz=1.0471975511965976,
            dt=2.0e-3,
            family="C",
            dealias=1.5,
            time_integrator=integrator,
        )

    meta = {
        "family": "Taylor-Couette MRI (FULL 3D MHD DNS, conducting wall)",
        "solver": "TaylorCouetteMRIDNSJax",
        "physics": "mri",
        "Re": 1000.0,
        "Rm": 1000.0,
        "B0": 0.1,
        "resolution": [nr, ntheta, nz],
        "dt": 2.0e-3,
        "integrator": integrator,
        "integrator_order": 2 if integrator == "CNAB2" else 3,
    }
    return build, (lambda s: s.seed_linear_eigenmode(kz_mode=1, amp=1.0e-7)[0]), meta


def _build_primitive_pcf(resolution: tuple[int, int, int], integrator: str):
    """Primitive-variable MRI shearing box (CNAB2-only) for comparison."""
    from production.adapters import load_config
    from production.oracles import _pcf_mri_packet_state, _primitive_solver_from_spec

    cfg = load_config(
        "production/examples/pcf_mri_shearbox_dns_v1.json", resolution_tier="smoke"
    )
    spec = cfg.spec

    def build():
        return _primitive_solver_from_spec(spec)

    def seed(solver):
        return _pcf_mri_packet_state(solver, spec)[0]

    meta = {
        "family": "MHD/MRI plane Couette (primitive-variable shearing box)",
        "solver": "PCFMRIDNSJax",
        "physics": "mri",
        "integrator": integrator,
        "integrator_order": 2,
        "integrator_note": "primitive solver is CNAB2-only",
        "config": "production/examples/pcf_mri_shearbox_dns_v1.json@smoke",
    }
    return build, seed, meta


_REGISTRY: dict[str, Callable[..., Any]] = {
    "hydro_pcf": _build_hydro_pcf,
    "mhd_mri_pcf": _build_mhd_mri_pcf,
    "tc_hydro": _build_tc_hydro,
    "tc_mhd": _build_tc_mhd,
    "tc_hydro_3d": _build_tc_hydro_3d,
    "tc_mhd_3d": _build_tc_mhd_3d,
    "primitive_pcf": _build_primitive_pcf,
}


# --------------------------------------------------------------------------- #
# Loss and measurement
# --------------------------------------------------------------------------- #


def _quadratic_energy(state: Any):
    """Sum |leaf|^2 over inexact (float/complex) leaves of the state pytree.

    A smooth real scalar depending on the entire final state; its reverse-mode
    cost is identical to any physically weighted energy but has no per-solver
    API quirks. Integer/None leaves (history_steps, unset SBDF3 history) are
    skipped automatically by tree_leaves / the dtype check.
    """
    import jax
    import jax.numpy as jnp

    total = jnp.array(0.0, dtype=jnp.float64)
    for leaf in jax.tree_util.tree_leaves(state):
        arr = jnp.asarray(leaf)
        if jnp.issubdtype(arr.dtype, jnp.complexfloating) or jnp.issubdtype(
            arr.dtype, jnp.floating
        ):
            total = total + jnp.sum(jnp.abs(arr) ** 2)
    return total


def _peak_bytes() -> int | None:
    import jax

    try:
        for device in jax.devices():
            stats = device.memory_stats() if hasattr(device, "memory_stats") else None
            if stats and "peak_bytes_in_use" in stats:
                return int(stats["peak_bytes_in_use"])
    except Exception:
        return None
    return None


def _time_calls(fn: Callable[[], Any], *, warmup: int, timed: int) -> dict[str, float]:
    import jax

    # First call: trace + compile (excluded from warm timing).
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    for _ in range(max(0, warmup)):
        out = fn()
    jax.block_until_ready(out)

    per: list[float] = []
    for _ in range(max(1, timed)):
        s = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        per.append(time.perf_counter() - s)
    arr = np.asarray(per)
    return {
        "compile_s": float(compile_s),
        "warm_s": float(np.median(arr)),
        "warm_p50_s": float(np.percentile(arr, 50)),
        "warm_p90_s": float(np.percentile(arr, 90)),
        "warm_min_s": float(np.min(arr)),
        "timed_calls": int(arr.size),
    }


def _make_scan_rollout(solver, steps: int):
    """Explicit lax.scan of jax.checkpoint(solver.step).

    This is exactly what the production ``ScanRolloutCache`` runs internally
    (per-step ``jax.checkpoint`` = remat, wrapped in ``lax.scan``), but exposed
    so BOTH the forward and the value_and_grad get compiled as a single fused
    XLA program with one dispatch. Calling ``solver.solve`` inside an outer jit
    instead defeats that fusion (its cache dispatches per step), which makes a
    forward-vs-autograd comparison overhead-bound and meaningless at small sizes.
    """
    import jax
    from jax import lax

    ckpt_step = jax.checkpoint(lambda st: solver.step(st))

    def rollout(state0):
        final, _ = lax.scan(
            lambda carry, _: (ckpt_step(carry), None),
            state0,
            xs=None,
            length=steps,
        )
        return final

    return rollout


def run(
    problem: str,
    *,
    steps: int,
    resolution,
    warmup: int,
    timed: int,
    integrator: str | None = None,
) -> dict:
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX_ENABLE_X64=1 is required (float64 production path).")

    builder = _REGISTRY[problem]
    selected_integrator = integrator or meta_integrator(problem)
    build_solver, seed_state, meta = builder(resolution, selected_integrator)
    solver = build_solver()
    state0 = solver.solve(seed_state(solver), 0)
    jax.block_until_ready(state0)

    rollout = _make_scan_rollout(solver, steps)

    # Differentiate only w.r.t. the inexact (float/complex) leaves of the state.
    # Non-inexact leaves -- a Python-bool ``have_old`` flag, integer step
    # counters -- are physically not initial conditions and JAX refuses to
    # differentiate a bool leaf; freeze them and reconstruct the full state
    # inside the loss. For states whose leaves are all inexact this is identical
    # to differentiating the whole pytree.
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(state0)

    def _is_diff(x) -> bool:
        return hasattr(x, "dtype") and (
            jnp.issubdtype(x.dtype, jnp.floating)
            or jnp.issubdtype(x.dtype, jnp.complexfloating)
        )

    mask = [_is_diff(leaf) for leaf in leaves]
    frozen = [None if m else leaf for m, leaf in zip(mask, leaves)]
    diff0 = [leaf for m, leaf in zip(mask, leaves) if m]

    def _rebuild(diff_leaves):
        it = iter(diff_leaves)
        full = [next(it) if m else f for m, f in zip(mask, frozen)]
        return jax.tree_util.tree_unflatten(treedef, full)

    def loss(diff_leaves):
        return _quadratic_energy(rollout(_rebuild(diff_leaves)))

    # Three compilations of the SAME rollout:
    #  * fwd_naive  = plain forward jit -- the cost of running solve() forward.
    #  * fwd_ad     = forward extracted from value_and_grad (gradient DCE'd by XLA);
    #                 compiled with the identical AD machinery as the autograd run,
    #                 so it is the fair, compilation-matched forward baseline.
    #  * vjp        = value + gradient (the autograd cost).
    # These differ because a plain forward can be scheduled differently from the
    # primal pass inside reverse-mode AD (e.g. serialized custom-calls), so the
    # honest "autograd overhead" is vjp / fwd_ad while vjp / fwd_naive says how
    # autograd compares to just calling the forward solver.
    fwd_naive = jax.jit(loss)
    fwd_ad = jax.jit(lambda s: jax.value_and_grad(loss)(s)[0])
    vjp = jax.jit(jax.value_and_grad(loss))

    # --- naive forward ---
    peak_before = _peak_bytes()
    fwd_t = _time_calls(lambda: fwd_naive(diff0), warmup=warmup, timed=timed)
    loss_val = float(fwd_naive(diff0))
    peak_after_forward = _peak_bytes()

    # --- compilation-matched forward (AD-scheduled, gradient discarded) ---
    fwd_ad_t = _time_calls(lambda: fwd_ad(diff0), warmup=warmup, timed=timed)

    # --- autograd (value_and_grad) ---
    vjp_t = _time_calls(lambda: vjp(diff0), warmup=warmup, timed=timed)
    val2, grad = vjp(diff0)
    jax.block_until_ready((val2, grad))
    peak_after_autograd = _peak_bytes()

    # gradient sanity
    grad_leaves = [
        np.asarray(jax.device_get(g))
        for g in jax.tree_util.tree_leaves(grad)
        if hasattr(g, "dtype")
        and (
            np.issubdtype(g.dtype, np.floating)
            or np.issubdtype(g.dtype, np.complexfloating)
        )
    ]
    grad_norm = float(np.sqrt(sum(float(np.sum(np.abs(g) ** 2)) for g in grad_leaves)))
    grad_finite = bool(all(np.all(np.isfinite(g)) for g in grad_leaves))

    # --- reference: raw production solve() (concrete-history path) ---
    def raw_solve():
        return solver.solve(state0, steps)

    raw_t = _time_calls(raw_solve, warmup=warmup, timed=timed)

    def _ratio(num, den):
        return num / den if den > 0 else float("nan")

    ratio_naive = _ratio(vjp_t["warm_s"], fwd_t["warm_s"])
    ratio_matched = _ratio(vjp_t["warm_s"], fwd_ad_t["warm_s"])

    return {
        "problem": problem,
        "meta": meta,
        "steps": steps,
        "protocol": {"warmup_calls": warmup, "timed_calls": timed},
        "loss_value": loss_val,
        "value_and_grad_value": float(val2),
        "grad_l2_norm": grad_norm,
        "grad_finite": grad_finite,
        "forward": fwd_t,
        "forward_ad_matched": fwd_ad_t,
        "autograd": vjp_t,
        "raw_production_solve": raw_t,
        "ratio_autograd_over_forward_matched": ratio_matched,
        "ratio_autograd_over_forward_naive": ratio_naive,
        "per_step_forward_naive_ms": 1e3 * fwd_t["warm_s"] / steps,
        "per_step_forward_matched_ms": 1e3 * fwd_ad_t["warm_s"] / steps,
        "per_step_autograd_ms": 1e3 * vjp_t["warm_s"] / steps,
        "memory": {
            "peak_before_bytes": peak_before,
            "peak_after_forward_bytes": peak_after_forward,
            "peak_after_autograd_bytes": peak_after_autograd,
            "autograd_extra_bytes": (
                None
                if peak_after_autograd is None or peak_after_forward is None
                else int(peak_after_autograd - peak_after_forward)
            ),
        },
    }


def meta_integrator(problem: str) -> str:
    # 3rd-order where the solver supports it, else CNAB2.
    return {
        "hydro_pcf": "SBDF3",
        "mhd_mri_pcf": "SBDF3",
        "tc_hydro": "SBDF3",
        "tc_mhd": "SBDF3",
        "tc_hydro_3d": "SBDF3",
        "tc_mhd_3d": "SBDF3",
        "primitive_pcf": "CNAB2",
    }[problem]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=tuple(_REGISTRY))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--integrator",
        choices=("CNAB2", "SBDF3", "IMEXRK3"),
        default=None,
        help="Override the benchmark's default time integrator.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override resolution: 3 ints for PCF/full-3D TC, "
            "2 ints (Nr Nz) for axisymmetric TC."
        ),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=12)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    resolution = tuple(args.resolution) if args.resolution else None
    result = run(
        args.problem,
        steps=args.steps,
        resolution=resolution,
        warmup=args.warmup,
        timed=args.timed,
        integrator=args.integrator,
    )

    import jax

    result["backend"] = jax.default_backend()
    result["device"] = str(jax.devices()[0])
    try:
        from production.provenance import capture_provenance

        result["provenance"] = capture_provenance()
    except Exception:
        result["provenance"] = {}

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    m = result["meta"]
    print(f"\n=== {args.problem}: {m['family']} ===")
    print(
        f"  solver={m['solver']}  integrator={m['integrator']} "
        f"(order {m['integrator_order']})  steps={args.steps}"
    )
    print(f"  resolution={m.get('resolution')}")
    naive_ms = result["forward"]["warm_s"] * 1e3
    matched_ms = result["forward_ad_matched"]["warm_s"] * 1e3
    raw_ms = result["raw_production_solve"]["warm_s"] * 1e3
    print(
        f"  forward (naive jit)   warm={naive_ms:9.2f} ms  "
        f"per-step {result['per_step_forward_naive_ms']:8.3f} ms"
    )
    print(
        f"  forward (AD-matched)  warm={matched_ms:9.2f} ms  "
        f"per-step {result['per_step_forward_matched_ms']:8.3f} ms"
    )
    print(
        f"  autograd (val+grad)   warm={result['autograd']['warm_s'] * 1e3:9.2f} ms  "
        f"per-step {result['per_step_autograd_ms']:8.3f} ms"
    )
    print(f"  raw solver.solve()    warm={raw_ms:9.2f} ms")
    print(
        f"  >>> autograd / forward(AD-matched) = "
        f"{result['ratio_autograd_over_forward_matched']:.2f}x   "
        f"(vs naive forward = {result['ratio_autograd_over_forward_naive']:.2f}x)"
    )
    print(f"  grad L2={result['grad_l2_norm']:.4e}  finite={result['grad_finite']}")
    mem = result["memory"]
    if mem["peak_after_autograd_bytes"]:
        print(
            f"  peak mem: forward {mem['peak_after_forward_bytes'] / 1e9:.3f} GB  "
            f"autograd {mem['peak_after_autograd_bytes'] / 1e9:.3f} GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
