import json
import os
import subprocess
import sys


def _run_capture(code: str, env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(proc.stdout)


def test_device_capture_preserves_jaxfun_default_x64_by_default_in_fresh_process():
    env = os.environ.copy()
    for key in (
        "JAXFUN_PRODUCTION_DTYPE",
        "JAXFUN_ENABLE_X64",
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
    ):
        env.pop(key, None)
    code = (
        "import json; "
        "from production.device import capture_device_record; "
        "print(json.dumps(capture_device_record('cpu'), sort_keys=True))"
    )

    record = _run_capture(code, env)

    assert record["production_run_dtype"] == "float64"
    assert record["requested_production_dtype"] == "float32"
    assert record["jax_enable_x64"] is True
    assert record["jax_default_scalar_dtype"] == "float64"
    assert record["jaxfun_enable_x64"] is None
    assert record["jax_enable_x64_env"] is None
    assert record["jax_platforms"] == "cpu"


def test_device_capture_preserves_existing_x64_until_apply_requested():
    env = os.environ.copy()
    env["JAXFUN_PRODUCTION_DTYPE"] = "float32"
    env["JAXFUN_ENABLE_X64"] = "1"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("JAX_PLATFORMS", None)
    code = (
        "import json; "
        "from production.device import capture_device_record; "
        "print(json.dumps(capture_device_record('cpu'), sort_keys=True))"
    )

    record = _run_capture(code, env)

    assert record["production_run_dtype"] == "float64"
    assert record["requested_production_dtype"] == "float32"
    assert record["jax_enable_x64"] is True
    assert record["jax_default_scalar_dtype"] == "float64"
    assert record["jaxfun_enable_x64"] == "1"
    assert record["jax_enable_x64_env"] == "1"


def test_requested_float32_overrides_stale_x64_env_when_applied():
    env = os.environ.copy()
    env["JAXFUN_PRODUCTION_DTYPE"] = "float32"
    env["JAXFUN_ENABLE_X64"] = "1"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("JAX_PLATFORMS", None)
    code = (
        "import json; "
        "from production.device import capture_device_record; "
        "print(json.dumps(capture_device_record("
        "'cpu', apply_dtype_to_process=True), sort_keys=True))"
    )

    record = _run_capture(code, env)

    assert record["production_run_dtype"] == "float32"
    assert record["requested_production_dtype"] == "float32"
    assert record["jax_enable_x64"] is False
    assert record["jax_default_scalar_dtype"] == "float32"
    assert record["jaxfun_enable_x64"] == "0"
    assert record["jax_enable_x64_env"] == "0"
