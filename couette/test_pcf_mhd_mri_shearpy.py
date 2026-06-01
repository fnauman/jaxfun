from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"


def run_mri_case(args, timeout=120):
    code = textwrap.dedent(
        f"""
        import json
        import sys
        sys.path.insert(0, {str(DEMO_DIR)!r})
        import pcf_mhd_mri_shearpy
        diag = pcf_mhd_mri_shearpy.run({args!r})
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


def test_tiny_legendre_shearpy_mri_pcf_analogue_runs():
    diag = run_mri_case([
        "--family", "L",
        "--nx", "8", "--ny", "8", "--nz", "8",
        "--lx", "4.0", "--ly", "4.0", "--lz", "1.0",
        "--Re", "1000", "--Rm", "1000",
        "--shear", "1.0", "--omega", "0.6667",
        "--by", "0.0", "--bz", "0.025",
        "--dt", "0.001", "--end-time", "0.002",
        "--moderror", "0",
        "--perturbation-amplitude", "0.001",
        "--magnetic-amplitude", "0.0",
        "--filename", "pytest_PCF_MRI_shearpy_L",
        "--max-divb-l2", "1e-12",
        "--max-divu-l2", "1e-12",
        "--assert-every-step",
    ])
    assert diag["Epert"] > 0
    assert diag["Emag_total"] > 0
    assert abs(diag["B0z"] - 0.025) < 1e-15
    assert abs(diag["q_shear"] - 1.0 / 0.6667) < 1e-12
    assert abs(diag["mean_shear"] + 1.0) < 1e-10
    assert diag["divb_l2"] < 1e-12
    assert diag["divu_l2"] < 1e-12


def run_mri_growth_history(timeout=180):
    """Integrate the net-flux MRI case and return the magnetic-energy history.

    The solver is driven directly (rather than through ``run``) so that the
    intermediate perturbation magnetic energy can be sampled and checked for
    exponential MRI growth, not merely a non-crashing run.
    """
    code = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(DEMO_DIR)!r})
        from shenfun import comm
        import pcf_mhd_mri_shearpy as M

        s = M.PlaneCouetteMRIShearpy(
            N=(16, 8, 16), domain=((-2, 2), (0, 4.0), (0, 1.0)),
            Re=1000.0, Rm=1000.0, shear_rate=1.0, omega=2.0 / 3.0,
            by=0.0, bz=0.025, dt=0.005, moderror=0,
            perturbation_amplitude=1e-3, magnetic_amplitude=0.0,
            family="L", filename="pytest_PCF_MRI_growth")
        s.initialize()
        s.assemble()
        hist = []
        t, tstep = 0.0, 0
        while t < 3.0 - 1e-9:
            for rk in range(s.PDE.steps()):
                s.prepare_step(rk)
                for eq in s.pdes.values(): eq.compute_rhs(rk)
                for eq in s.pdesA: eq.compute_rhs(rk)
                for eq in s.pdes.values(): eq.solve_step(rk)
                s.compute_vw(rk)
                for eq in s.pdesA: eq.solve_step(rk)
            t += s.dt; tstep += 1
            if tstep % 100 == 0:
                d = s.compute_diagnostics(t, tstep)
                hist.append((d["t"], d["Emag"], d["divb_l2"]))
        if comm.Get_rank() == 0:
            print("JSON_HIST " + json.dumps(hist))
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
        if line.startswith("JSON_HIST "):
            return json.loads(line.split(" ", 1)[1])
    raise AssertionError(proc.stdout)


def test_netflux_mri_magnetic_energy_grows():
    """The axisymmetric channel-mode seed must produce exponential MRI growth."""
    hist = run_mri_growth_history()
    assert len(hist) >= 4
    emag = [row[1] for row in hist]
    divb = [row[2] for row in hist]

    # Perturbation magnetic energy must grow substantially (expected ~7x over
    # t=1..3; require >2x with margin) and monotonically once the channel mode
    # dominates (after the first sample).
    assert emag[-1] > 2.0 * emag[0]
    assert all(b > emag[i] for i, b in enumerate(emag[1:])), emag
    # div(B) stays at machine precision throughout the growth.
    assert max(divb) < 1e-10
