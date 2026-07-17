"""Regression coverage for running the production entry point from checkout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_run_problem_direct_script_help_finds_src_bootstrap() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "production/run_problem.py", "--help"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout


def test_run_problem_module_help_finds_src_bootstrap_from_clean_checkout() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = f"""
import runpy
import sys
from pathlib import Path

root = Path({str(ROOT)!r}).resolve()
src = (root / "src").resolve()
sys.path[:] = [
    entry
    for entry in sys.path
    if not entry or Path(entry).resolve() != src
]
sys.path.insert(0, str(root))
sys.argv = ["production.run_problem", "--help"]
runpy.run_module("production.run_problem", run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
