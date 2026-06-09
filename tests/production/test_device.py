import json
import os
import subprocess
import sys


def test_device_capture_defaults_fresh_process_to_float32_cpu():
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
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    record = json.loads(proc.stdout)
    assert record["production_run_dtype"] == "float32"
    assert record["jax_enable_x64"] is False
    assert record["jax_default_scalar_dtype"] == "float32"
    assert record["jaxfun_enable_x64"] == "0"
    assert record["jax_enable_x64_env"] == "0"
    assert record["jax_platforms"] == "cpu"
