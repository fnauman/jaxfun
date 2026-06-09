"""Device and dtype capture for production runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def capture_device_record(requested: str = "auto") -> dict[str, Any]:
    """Return live JAX/JAXLIB/device metadata.

    Local production smoke runs default to float32 through
    ``JAXFUN_PRODUCTION_DTYPE`` while keeping jaxfun's x64 capability enabled for
    parity tests that explicitly need it.
    """

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if requested == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif requested in {"cuda", "gpu"}:
        os.environ.setdefault("JAX_PLATFORMS", "cuda")

    import jax
    import jax.numpy as jnp
    import jaxlib
    import jaxfun  # noqa: F401 - import enables x64 and prealloc policy

    devices = jax.devices()
    backend = jax.default_backend()
    mode = "gpu" if backend in {"gpu", "cuda"} else "cpu_smoke"
    return {
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "interpreter": sys.executable,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "default_backend": backend,
        "devices": [str(device) for device in devices],
        "degraded": mode == "cpu_smoke",
        "mode": mode,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_default_scalar_dtype": str(jnp.asarray(1.0).dtype),
        "production_run_dtype": os.environ.get("JAXFUN_PRODUCTION_DTYPE", "float32"),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def production_run_env(requested: str = "auto") -> dict[str, Any]:
    from .compare_goldens import resolve_golden, vendored_golden_root

    device = capture_device_record(requested)
    run_specs_dir = Path(__file__).resolve().parent / "runs"
    run_specs = sorted(path.stem for path in run_specs_dir.glob("*.json")) if run_specs_dir.exists() else []
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
            ".venv/bin/python -m pytest -q tests/couette/test_taylor_couette_linear_jax.py",
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


def write_run_env(path: str | Path, requested: str = "auto") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(production_run_env(requested), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"])
    parser.add_argument("--write")
    args = parser.parse_args(argv)
    data = production_run_env(args.device)
    payload = json.dumps(data, sort_keys=True, indent=2) + "\n"
    if args.write:
        Path(args.write).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
