"""Explicit-resolution benchmark driver for the JAXfun PCF refactor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from production.benchmark import benchmark_step
from production.provenance import capture_provenance

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


_VARIANTS = {
    "transform-gradient": ("transform", "gradient", "separate"),
    "coefficient-gradient": ("optimized", "gradient", "separate"),
    "optimized-gradient": ("optimized", "gradient", "batched"),
    "optimized-rotational": ("optimized", "rotational", "batched"),
}


def _build_solver(args: argparse.Namespace):
    common: dict[str, Any] = {
        "N": tuple(args.resolution),
        "dt": args.dt,
        "family": args.family,
        "padding_factor": tuple(args.padding),
        "time_integrator": args.integrator,
        "coefficient_path": _VARIANTS[args.variant][0],
        "nonlinear_form": _VARIANTS[args.variant][1],
        "solve_batching": _VARIANTS[args.variant][2],
    }
    if args.solver == "hydro":
        from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax

        return PlaneCouetteFluctuationJax(
            **common,
            Re=args.re,
            perturbation_amplitude=args.velocity_amplitude,
        )
    if args.solver == "mhd":
        from examples.pcf_mhd_jax import PlaneCouetteMHDJax

        return PlaneCouetteMHDJax(
            **common,
            Re=args.re,
            Rm=args.rm,
            perturbation_amplitude=args.velocity_amplitude,
            magnetic_amplitude=args.magnetic_amplitude,
        )
    from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax

    return PlaneCouetteMRIShearpyJax(
        **common,
        Re=args.re,
        Rm=args.rm,
        perturbation_amplitude=args.velocity_amplitude,
        magnetic_amplitude=args.magnetic_amplitude,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("hydro", "mhd", "mri"), required=True)
    parser.add_argument("--resolution", type=int, nargs=3, required=True)
    parser.add_argument("--variant", choices=tuple(_VARIANTS), required=True)
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--family", choices=("C", "L"), default="C")
    parser.add_argument("--padding", type=float, nargs=3, default=(1.0, 1.5, 1.5))
    parser.add_argument("--re", type=float, default=400.0)
    parser.add_argument("--rm", type=float, default=400.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--velocity-amplitude", type=float, default=0.02)
    parser.add_argument("--magnetic-amplitude", type=float, default=0.005)
    parser.add_argument("--warmup-blocks", type=int, default=2)
    parser.add_argument("--timed-blocks", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=25)
    parser.add_argument("--dt-transition-probes", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if any(value <= 0 for value in args.resolution):
        raise ValueError("resolution entries must be positive")
    if int(args.resolution[0] * args.resolution[1] * args.resolution[2]) > 128**3:
        raise ValueError("resolution exceeds the bounded 128^3 collocation cap")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("PCF refactor benchmarks must use CUDA_VISIBLE_DEVICES=0")
    if args.integrator is None:
        args.integrator = "IMEXRK3" if args.solver == "hydro" else "CNAB2"

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(args.cache_dir.resolve())
    timing = benchmark_step(
        lambda: _build_solver(args),
        label=f"{args.solver}:{args.variant}:{'x'.join(map(str, args.resolution))}",
        warmup_steps=args.warmup_blocks,
        timed_steps=args.timed_blocks,
        seed_state=lambda value: value.initial_state(),
        rollout_steps=args.rollout_steps,
        dt_transition_probes=args.dt_transition_probes,
    )
    artifact = {
        "schema_version": 1,
        "benchmark": "jaxfun_pcf_perf_refactor",
        "solver": args.solver,
        "variant": args.variant,
        "configuration": {
            "basis_family": args.family,
            "domain": (
                (-1.0, 1.0),
                (0.0, 12.566370614359172),
                (0.0, 6.283185307179586),
            ),
            "resolution": args.resolution,
            "padding": args.padding,
            "Re": args.re,
            "Rm": None if args.solver == "hydro" else args.rm,
            "dt": args.dt,
            "integrator": args.integrator,
            "coefficient_path": _VARIANTS[args.variant][0],
            "nonlinear_form": _VARIANTS[args.variant][1],
            "solve_batching": _VARIANTS[args.variant][2],
            "wavenumber_solver": os.environ.get("JAXFUN_WAVENUMBER_SOLVER", "jax"),
            "compilation_cache_dir": str(args.cache_dir.resolve()),
        },
        "protocol": {
            "warmup_blocks": args.warmup_blocks,
            "timed_blocks": args.timed_blocks,
            "physical_steps_per_block": args.rollout_steps,
            "dt_transition_probes": args.dt_transition_probes,
        },
        "timing": timing.to_dict(),
        "provenance": capture_provenance(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
