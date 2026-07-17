"""Device and dtype capture for production runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def configure_production_dtype(
    dtype: str | None = None, *, apply_to_process: bool = False
) -> str:
    """Return the production dtype policy, optionally applying it to this process.

    Importable helpers must not silently downgrade an already-configured x64 test
    or analysis process. CLI entrypoints pass ``apply_to_process=True`` before
    solver imports so subprocess production runs still get the requested dtype.
    """

    requested = (dtype or os.environ.get("JAXFUN_PRODUCTION_DTYPE", "float32")).lower()
    aliases = {
        "float32": "float32",
        "single": "float32",
        "fp32": "float32",
        "float64": "float64",
        "double": "float64",
        "fp64": "float64",
    }
    if requested not in aliases:
        raise ValueError(
            "JAXFUN_PRODUCTION_DTYPE must be one of float32, fp32, float64, or fp64"
        )
    canonical = aliases[requested]
    if apply_to_process:
        os.environ["JAXFUN_PRODUCTION_DTYPE"] = canonical
        x64_enabled = "1" if canonical == "float64" else "0"
        os.environ["JAXFUN_ENABLE_X64"] = x64_enabled
        os.environ["JAX_ENABLE_X64"] = x64_enabled
    return canonical


def _env_truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def capture_device_record(
    requested: str = "auto", *, apply_dtype_to_process: bool = False
) -> dict[str, Any]:
    """Return live JAX/JAXLIB/device metadata.

    Local production smoke runs default to float32 through
    ``JAXFUN_PRODUCTION_DTYPE`` while keeping jaxfun's x64 capability enabled for
    parity tests that explicitly need it.
    """

    requested_production_dtype = configure_production_dtype(
        apply_to_process=apply_dtype_to_process
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if requested == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif requested in {"cuda", "gpu"}:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import jax

    if apply_dtype_to_process:
        jax.config.update(
            "jax_enable_x64",
            _env_truthy(os.environ.get("JAXFUN_ENABLE_X64"), default=True),
        )
    import jax.numpy as jnp  # noqa: I001
    import jaxlib  # noqa: I001
    import jaxfun  # noqa: F401,I001 - import applies jaxfun dtype/prealloc policy

    live_production_dtype = (
        requested_production_dtype
        if apply_dtype_to_process
        else _live_production_dtype(jax)
    )
    # FJ-08/FJ-11: assert the JAX x64 state matches the production dtype after imports,
    # so a mis-set env fails loudly instead of silently downgrading a float64 claim run.
    actual_x64 = bool(jax.config.read("jax_enable_x64"))
    expected_x64 = live_production_dtype == "float64"
    if actual_x64 != expected_x64:
        raise RuntimeError(
            f"JAX x64 state ({actual_x64}) is inconsistent with the production dtype "
            f"{live_production_dtype!r} after imports; set JAXFUN_ENABLE_X64 / "
            "JAX_ENABLE_X64 to match (FJ-08)."
        )
    devices = jax.devices()
    backend = jax.default_backend()
    mode = "gpu" if backend in {"gpu", "cuda"} else "cpu_smoke"
    return {
        "captured_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "interpreter": sys.executable,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "default_backend": backend,
        "devices": [str(device) for device in devices],
        "degraded": mode == "cpu_smoke",
        "mode": mode,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_use_simplified_jaxpr_constants": bool(
            getattr(jax.config, "jax_use_simplified_jaxpr_constants", False)
        ),
        "jaxfun_use_simplified_jaxpr_constants": os.environ.get(
            "JAXFUN_USE_SIMPLIFIED_JAXPR_CONSTANTS"
        ),
        "jaxfun_wavenumber_solver": os.environ.get("JAXFUN_WAVENUMBER_SOLVER", "jax"),
        "jax_use_simplified_jaxpr_constants_env": os.environ.get(
            "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"
        ),
        "jax_default_scalar_dtype": str(jnp.asarray(1.0).dtype),
        "production_run_dtype": live_production_dtype,
        "requested_production_dtype": requested_production_dtype,
        "jaxfun_enable_x64": os.environ.get("JAXFUN_ENABLE_X64"),
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _live_production_dtype(jax_module: Any) -> str:
    return "float64" if bool(jax_module.config.read("jax_enable_x64")) else "float32"


def production_run_env(
    requested: str = "auto", *, apply_dtype_to_process: bool = False
) -> dict[str, Any]:
    from .compare_goldens import resolve_golden, vendored_golden_root

    device = capture_device_record(
        requested, apply_dtype_to_process=apply_dtype_to_process
    )
    run_specs_dir = Path(__file__).resolve().parent / "runs"
    run_specs = (
        sorted(path.stem for path in run_specs_dir.glob("*.json"))
        if run_specs_dir.exists()
        else []
    )
    try:
        resolution = resolve_golden("pcf_hydro_laminar_v1")
        golden_policy = resolution.policy
        golden_root = str(resolution.root)
    except FileNotFoundError:
        golden_policy = "missing"
        golden_root = str(vendored_golden_root())

    return {
        **device,
        "golden_path_policy": golden_policy,
        "golden_root": golden_root,
        "test_commands": [
            ".venv/bin/python -m pytest -q tests/test_x64_default.py",
            ".venv/bin/python -m pytest -q "
            "tests/couette/test_taylor_couette_linear_jax.py",
            ".venv/bin/python -m pytest -q tests/production",
        ],
        "known_gated_tests": ["live_shenfun", "spmd", "gpu", "slow", "integration"],
        "production_run_specs": run_specs,
        "output_locations": {
            "runs": "runs/<problem_id>/<timestamp>",
            "checkpoints": "runs/<problem_id>/<timestamp>/checkpoints",
            "goldens": "runs/<problem_id>/<timestamp>/golden/golden.json",
            "vendored_goldens": "production/goldens/<problem_id>/golden/golden.json",
        },
    }


def write_run_env(
    path: str | Path, requested: str = "auto", *, apply_dtype_to_process: bool = False
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            production_run_env(
                requested, apply_dtype_to_process=apply_dtype_to_process
            ),
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"]
    )
    parser.add_argument("--write")
    args = parser.parse_args(argv)
    data = production_run_env(args.device, apply_dtype_to_process=True)
    payload = json.dumps(data, sort_keys=True, indent=2) + "\n"
    if args.write:
        Path(args.write).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
