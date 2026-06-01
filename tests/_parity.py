from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _shenfun_python() -> str:
    explicit = os.environ.get("SHENFUN_PYTHON")
    if explicit:
        return explicit
    env = os.environ.get("SHENFUN_CONDA_ENV", "shenfun")
    candidate = Path.home() / "miniconda3" / "envs" / env / "bin" / "python"
    return str(candidate)


def require_local_shenfun() -> None:
    """Skip when the local shenfun reference runner is unavailable."""
    if not Path(_shenfun_python()).exists():
        pytest.skip("set SHENFUN_PYTHON to a Python executable with shenfun installed")
    if not (REPO_ROOT.parent / "shenfun").exists():
        pytest.skip("sibling ../shenfun checkout is required for live parity tests")


def run_shenfun_json(source: str) -> Any:
    """Run a small script in the local shenfun conda env and parse JSON output."""
    require_local_shenfun()
    prelude = "\n".join(
        [
            "import json",
            "import sys",
            "import types",
            "m = types.ModuleType('_demo_utils')",
            "m.default_thread_cap = lambda: None",
            "sys.modules['_demo_utils'] = m",
            "sys.path.insert(0, 'couette')",
        ]
    )
    proc = subprocess.run(
        [_shenfun_python(), "-c", prelude + "\n" + source],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.skip(
            "local shenfun reference run failed:\n"
            + proc.stdout[-2000:]
            + proc.stderr[-2000:]
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("shenfun reference runner produced no JSON output")
    return json.loads(lines[-1])


def complex_array(rows: Any) -> np.ndarray:
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0] + 1j * arr[:, 1]


def tc_linear_eigenvalues(*, n: int = 12, m: int = 0, kz: float = 3.0) -> np.ndarray:
    rows = run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
            s = TaylorCouetteLinear(CircularCouette(), nu=0.002, N={n}, family='L')
            w, _ = s.eigs({m}, {kz!r}, 6)
            print(json.dumps([[float(z.real), float(z.imag)] for z in w]))
            """
        )
    )
    return complex_array(rows)


def tc_linear_nonmodal(*, n: int = 12, m: int = 0, kz: float = 3.0) -> list[dict]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
            s = TaylorCouetteLinear(CircularCouette(), nu=0.002, N={n}, family='L')
            rows = s.nonmodal_growth({m}, {kz!r}, [0.0, 0.5], n_modes=12)
            print(json.dumps(rows))
            """
        )
    )


def tc_mri_eigenvalues(*, magnetic_bc: str, n: int = 12) -> np.ndarray:
    rows = run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette
            from taylor_couette_mri import TaylorCouetteMRI
            eta = 0.5
            base = CircularCouette(1.0, 2.0, 1.0, eta**1.5)
            s = TaylorCouetteMRI(
                base, B0=0.1, nu=0.001, eta_mag=0.001, N={n},
                family='L', magnetic_bc={magnetic_bc!r}
            )
            w, _ = s.eigs(0, 3.0, 6)
            print(json.dumps([[float(z.real), float(z.imag)] for z in w]))
            """
        )
    )
    return complex_array(rows)


def tc_mri_nonmodal(
    *, magnetic_bc: str, n: int = 10, energy: str = "total"
) -> list[dict]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette
            from taylor_couette_mri import TaylorCouetteMRI
            eta = 0.5
            base = CircularCouette(1.0, 2.0, 1.0, eta**1.5)
            s = TaylorCouetteMRI(
                base, B0=0.1, nu=0.001, eta_mag=0.001, N={n},
                family='L', magnetic_bc={magnetic_bc!r}
            )
            rows = s.nonmodal_growth(
                0, 3.0, [0.0, 0.25], n_modes=10, energy={energy!r}
            )
            print(json.dumps(rows))
            """
        )
    )


def tc_radial_dealias_product(*, n: int = 8) -> np.ndarray:
    """Reference padded radial/Fourier product projected with shenfun."""
    rows = run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from shenfun import (
                Array,
                Function,
                FunctionSpace,
                TensorProductSpace,
                comm,
            )
            F = FunctionSpace({n}, 'F', dtype='D', domain=(0, 2*np.pi))
            S = FunctionSpace({n}, 'L', domain=(1.0, 2.0))
            T = TensorProductSpace(comm, (F, S), dtype='D')
            Tp = T.get_dealiased((1.5, 1.5))
            u = Function(T)
            v = Function(T)
            # Avoid the Fourier Nyquist mode; jaxfun masks it in solver products.
            u[0, 1] = 0.5
            u[1, 2] = 0.75 + 0.25j
            v[0, 3] = -0.4
            v[1, 1] = -0.2 + 0.1j
            hp = Array(
                Tp,
                buffer=np.array(Tp.backward(u)) * np.array(Tp.backward(v)),
            )
            h = Tp.forward(hp)
            print(json.dumps([
                [[float(z.real), float(z.imag)] for z in row]
                for row in np.asarray(h)
            ]))
            """
        )
    )
    arr = np.asarray(rows, dtype=float)
    return arr[..., 0] + 1j * arr[..., 1]
