from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"


def run_divfree_case(args, timeout=120):
    code = textwrap.dedent(
        f"""
        import json
        import sys
        sys.path.insert(0, {str(DEMO_DIR)!r})
        import pcf_mhd_divfree
        diag = pcf_mhd_divfree.run({args!r})
        print("JSON_DIAG " + json.dumps(diag, sort_keys=True))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("JSON_DIAG "):
            return json.loads(line.split(" ", 1)[1])
    raise AssertionError(proc.stdout)


def test_tiny_legendre_mhd_keeps_divergence_at_roundoff():
    diag = run_divfree_case([
        "--family", "L",
        "--nx", "8", "--ny", "8", "--nz", "8",
        "--dt", "0.001", "--end-time", "0.003",
        "--moderror", "0",
        "--perturbation-amplitude", "0.05",
        "--magnetic-amplitude", "0.02",
        "--filename", "pytest_PCF_divfree_L",
        "--max-divb-l2", "1e-12",
        "--max-divu-l2", "1e-12",
        "--assert-every-step",
    ])
    assert diag["Emag"] > 0
    assert diag["divb_l2"] < 1e-12
    assert diag["divu_l2"] < 1e-12


def test_chebyshev_mhd_uses_compatible_curl_spaces():
    diag = run_divfree_case([
        "--family", "C",
        "--nx", "16", "--ny", "16", "--nz", "16",
        "--dt", "0.001", "--end-time", "0.001",
        "--moderror", "0",
        "--perturbation-amplitude", "0.05",
        "--magnetic-amplitude", "0.02",
        "--filename", "pytest_PCF_divfree_C",
        "--max-divb-l2", "1e-12",
        "--max-divu-l2", "1e-12",
        "--assert-every-step",
    ])
    assert diag["Emag"] > 0
    assert diag["divb_l2"] < 1e-12
    assert diag["divu_l2"] < 1e-12
