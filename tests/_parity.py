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


def shenfun_uniform_snapshot_reference(
    *, n0: int = 8, n1: int = 6, step: int = 2
) -> dict[str, Any]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import os
            import tempfile
            import h5py
            import numpy as np
            from mpi4py import MPI
            from shenfun import Array, FunctionSpace, ShenfunFile, TensorProductSpace

            K0 = FunctionSpace({n0}, 'F')
            K1 = FunctionSpace({n1}, 'C', bc=(0, 0))
            T = TensorProductSpace(MPI.COMM_WORLD, (K0, K1))
            base = tempfile.NamedTemporaryFile(
                prefix='shenfun_uniform_snapshot_', suffix='', delete=False
            ).name
            os.unlink(base)
            x0, x1 = [np.squeeze(x) for x in T.mesh(kind='uniform')]
            values = np.cos(x0.reshape((-1, 1))) * (1.0 - x1.reshape((1, -1))**2)
            u = Array(T)
            u[:] = values
            hfile = ShenfunFile(base, T, backend='hdf5', mode='w', mesh='uniform')
            hfile.write({step}, {{'u': [u]}}, forward_output=False)
            close = getattr(hfile, 'close', None)
            if close is not None:
                close()

            filename = base + '.h5'
            with h5py.File(filename, 'r') as h5:
                out = {{
                    'u': np.asarray(h5['u/2D/{step}']).tolist(),
                    'mesh': {{
                        'x0': np.asarray(h5['u/mesh/x0']).tolist(),
                        'x1': np.asarray(h5['u/mesh/x1']).tolist(),
                    }},
                }}
            os.remove(filename)
            T.destroy()
            print(json.dumps(out))
            """
        )
    )


def shenfun_basis_stencils(*, n: int = 8) -> dict[str, dict]:
    """Return live shenfun Dirichlet/Biharmonic basis stencil matrices."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            from shenfun import FunctionSpace

            rows = {{}}
            for family in ('L', 'C'):
                for bc in ((0, 0), (0, 0, 0, 0)):
                    space = FunctionSpace({int(n)!r}, family, bc=bc)
                    stencil = space.stencil_matrix().diags('csr').toarray()
                    rows[f'{{family}}_{{len(bc)}}'] = {{
                        'type': type(space).__name__,
                        'dim': int(space.dim()),
                        'shape': list(stencil.shape),
                        'stencil': stencil.tolist(),
                    }}
            print(json.dumps(rows))
            """
        )
    )


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


def tc_linear_operator_parts(
    *,
    n: int = 8,
    m: int = 1,
    kz: float = 2.0,
    nu: float = 0.002,
    family: str = "L",
) -> dict[str, Any]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
            s = TaylorCouetteLinear(
                CircularCouette(), nu={nu!r}, N={n}, family={family!r}
            )
            L0, Lv, M = s.assemble_parts({m}, {kz!r})

            def matrix_rows(arr):
                return [
                    [[float(z.real), float(z.imag)] for z in row]
                    for row in arr
                ]

            print(json.dumps({{
                "L0": matrix_rows(L0),
                "Lv": matrix_rows(Lv),
                "M": matrix_rows(M),
            }}))
            """
        )
    )


def tc_mri_operator_parts(
    *,
    magnetic_bc: str,
    n: int = 8,
    m: int = 0,
    kz: float = 2.0,
    family: str = "L",
) -> dict[str, Any]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            from taylor_couette_linear import CircularCouette
            from taylor_couette_mri import TaylorCouetteMRI
            eta = 0.5
            base = CircularCouette(1.0, 2.0, 1.0, eta**1.5)
            s = TaylorCouetteMRI(
                base, B0=0.1, nu=0.001, eta_mag=0.001, N={n},
                family={family!r}, magnetic_bc={magnetic_bc!r}
            )
            L0, Lnu, Leta, M = s.assemble_parts({m}, {kz!r})

            def matrix_rows(arr):
                return [
                    [[float(z.real), float(z.imag)] for z in row]
                    for row in arr
                ]

            print(json.dumps({{
                "L0": matrix_rows(L0),
                "Lnu": matrix_rows(Lnu),
                "Leta": matrix_rows(Leta),
                "M": matrix_rows(M),
            }}))
            """
        )
    )


def tc_linear_critical_scan(*, n: int = 8, iters: int = 8) -> dict[str, Any]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import numpy as np
            from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
            base = CircularCouette()
            s = TaylorCouetteLinear(base, nu=0.001, N={n}, family='L')
            kzs = np.array([2.0, 3.0, 4.0])
            kz_c, nu_c = s.critical_over_kz(0, kzs, iters={iters})
            print(json.dumps({{
                "kz_c": float(kz_c),
                "nu_c": float(nu_c),
                "Re_c": float(base.Omega1 * base.R1 * base.gap / nu_c),
                "a_c": float(kz_c * base.gap),
            }}))
            """
        )
    )


def tc_mri_critical_scans(
    *, magnetic_bc: str, n: int = 8, iters: int = 8
) -> dict[str, Any]:
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import numpy as np
            from taylor_couette_linear import CircularCouette
            from taylor_couette_mri import TaylorCouetteMRI
            eta = 0.5
            base = CircularCouette(1.0, 2.0, 1.0, eta**1.5)
            s = TaylorCouetteMRI(
                base, B0=0.1, nu=0.001, eta_mag=0.001, N={n},
                family='L', magnetic_bc={magnetic_bc!r}
            )
            kzs = np.array([2.0, 3.0])
            print(json.dumps({{
                "fixed_B0_nu": s.critical_Rm_fixed_B0_nu(
                    0, kzs, iters={iters}
                ),
                "fixed_controls": s.critical_Rm(0, kzs, iters={iters}),
            }}))
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


def pcf_fluctuation_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    n: tuple[int, int, int] = (9, 8, 8),
    dt: float = 1.0e-3,
    re: float = 600.0,
    perturbation_amplitude: float = 0.05,
    family: str = "L",
    include_velocity: bool = False,
    include_coefficients: bool = False,
    include_pressure: bool = False,
) -> list[dict]:
    """Run the live shenfun PCF fluctuation reference and return parity rows."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from pcf_fluctuations_corrected import PlaneCouetteFluctuation
            from shenfun import *

            solver = PlaneCouetteFluctuation(
                N={tuple(n)!r},
                family={family!r},
                dt={dt!r},
                Re={re!r},
                perturbation_amplitude={perturbation_amplitude!r},
                padding_factor=(1, 1.5, 1.5),
                modplot=-1,
                modsave=10**9,
                moderror=10**9,
                modanalysis=10**9,
                modspectra=10**9,
                modssp=10**9,
                checkpoint=10**9,
                enable_live_plots=False,
                save_plots=False,
                save_analysis=False,
                save_spectra=False,
                save_ssp=False,
                timestepper='IMEXRK222',
                filename='/tmp/jaxfun_pcf_parity',
            )
            t, tstep = solver.initialize(False)

            def diagnostics(step):
                ubp = solver.u_.backward(solver.ub)
                ubt = solver.total_velocity_physical_from(ubp)
                divu = solver.divu().backward()
                dvdx = solver.dvdx().backward()
                row = {{
                    'steps': int(step),
                    'Epert': float(
                        inner(1, ubp[0]*ubp[0])
                        + inner(1, ubp[1]*ubp[1])
                        + inner(1, ubp[2]*ubp[2])
                    ),
                    'Etot': float(
                        inner(1, ubt[0]*ubt[0])
                        + inner(1, ubt[1]*ubt[1])
                        + inner(1, ubt[2]*ubt[2])
                    ),
                    'divL2': float(np.sqrt(inner(1, divu*divu))),
                    'u_top': float(np.mean(ubt[1][-1, :, :])),
                    'u_bot': float(np.mean(ubt[1][0, :, :])),
                    'mean_shear': float(np.mean(dvdx + solver.dUb_dx)),
                }}
                if {include_velocity!r}:
                    row['velocity'] = [
                        np.asarray(ubp[i], dtype=float).tolist()
                        for i in range(3)
                    ]
                if {include_pressure!r}:
                    solver.convection()
                    d2proj = Project(solver.nu*Dx(solver.u_[0], 0, 2), solver.TC)
                    d2udx2 = d2proj.output_array
                    N0 = FunctionSpace(
                        solver.N[0],
                        solver.B0.family(),
                        bc={{'left': {{'N': 1.0}}, 'right': {{'N': 1.0}}}},
                    )
                    TN = TensorProductSpace(
                        comm,
                        (N0, solver.F1, solver.F2),
                        collapse_fourier=False,
                        slab=True,
                        modify_spaces_inplace=True,
                    )
                    d2proj()
                    coeff = np.asarray(d2udx2)
                    sign = (-1.0)**np.arange(coeff.shape[0])
                    left = np.tensordot(sign, coeff, axes=(0, 0))
                    right = np.sum(coeff, axis=0)
                    N0.bc.bcs_final[0] = left.copy()
                    N0.bc.bcs_final[1] = right.copy()
                    N0.bc.bcs[0] = left.copy()
                    N0.bc.bcs[1] = right.copy()
                    pressure_solver = (
                        chebyshev.la.Helmholtz
                        if solver.B0.family() == 'chebyshev'
                        else la.SolverGeneric1ND
                    )
                    divH = Inner(TestFunction(TN), -div(solver.H_))
                    solP = pressure_solver(
                        inner(TestFunction(TN), div(grad(TrialFunction(TN))))
                    )
                    pressure = solP(
                        divH(), Function(TN), constraints=((0, 0, 0),)
                    ).backward()
                    row['pressure'] = np.asarray(pressure, dtype=float).tolist()
                if {include_coefficients!r}:
                    def complex_rows(arr):
                        arr = np.asarray(arr)
                        return [
                            [
                                [[float(z.real), float(z.imag)] for z in rowz]
                                for rowz in rowy
                            ]
                            for rowy in arr
                        ]

                    row['coefficients'] = {{
                        'u': [complex_rows(solver.u_[i]) for i in range(3)],
                        'g': complex_rows(solver.g_),
                    }}
                return row

            rows = []
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                solver.solve(t=t, tstep=tstep, end_time=target*solver.dt)
                t = target*solver.dt
                tstep = target
                rows.append(diagnostics(target))
            print(json.dumps(rows))
            """
        )
    )

def pcf_mhd_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    n: tuple[int, int, int] = (9, 8, 8),
    dt: float = 1.0e-3,
    re: float = 400.0,
    rm: float | None = None,
    perturbation_amplitude: float = 0.05,
    magnetic_amplitude: float = 0.05,
    family: str = "L",
    include_coefficients: bool = False,
) -> list[dict]:
    """Run the live shenfun PCF-MHD reference and return parity rows."""
    rm_expr = "None" if rm is None else repr(float(rm))
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from pcf_mhd_divfree import PlaneCouetteMHDDivFree

            solver = PlaneCouetteMHDDivFree(
                N={tuple(n)!r},
                family={family!r},
                dt={dt!r},
                Re={re!r},
                Rm={rm_expr},
                perturbation_amplitude={perturbation_amplitude!r},
                magnetic_amplitude={magnetic_amplitude!r},
                padding_factor=(1, 1.5, 1.5),
                modsave=10**9,
                moderror=10**9,
                checkpoint=10**9,
                filename='/tmp/jaxfun_pcf_mhd_parity',
                prefer_numba=False,
                store_history=False,
                timestepper='IMEXRK222',
            )
            t, tstep = solver.initialize(False)

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [
                        [[float(z.real), float(z.imag)] for z in rowz]
                        for rowz in rowy
                    ]
                    for rowy in arr
                ]

            rows = []
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.solve(t=t, tstep=tstep, end_time=target*solver.dt)
                t = target*solver.dt
                tstep = target
                row = {{
                    key: diag[key]
                    for key in (
                        'Epert',
                        'Etot',
                        'Emag',
                        'divu_l2',
                        'divb_l2',
                        'top_wall_streamwise',
                        'bottom_wall_streamwise',
                        'mean_shear',
                        'bmax',
                    )
                }}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'u': [complex_rows(solver.u_[i]) for i in range(3)],
                        'g': complex_rows(solver.g_),
                        'A': [complex_rows(solver.a_[i]) for i in range(3)],
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )

def pcf_mhd_shearpy_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    n: tuple[int, int, int] = (9, 8, 8),
    dt: float = 1.0e-3,
    re: float = 400.0,
    rm: float | None = None,
    omega: float = 1.0,
    shear_rate: float = 1.0,
    background_b: tuple[float, float, float] = (0.0, 0.0, 0.1),
    perturbation_amplitude: float = 0.05,
    magnetic_amplitude: float = 0.05,
    family: str = "L",
    include_coefficients: bool = False,
) -> list[dict]:
    """Run the live shenfun PCF-MHD shearpy reference and return rows."""
    rm_expr = "None" if rm is None else repr(float(rm))
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from pcf_mhd_mri_shearpy import PlaneCouetteMRIShearpy

            solver = PlaneCouetteMRIShearpy(
                N={tuple(n)!r},
                domain=((-1, 1), (0, 4*np.pi), (0, 2*np.pi)),
                family={family!r},
                dt={dt!r},
                Re={re!r},
                Rm={rm_expr},
                omega={omega!r},
                shear_rate={shear_rate!r},
                by={background_b[1]!r},
                bz={background_b[2]!r},
                perturbation_amplitude={perturbation_amplitude!r},
                magnetic_amplitude={magnetic_amplitude!r},
                padding_factor=(1, 1.5, 1.5),
                modsave=10**9,
                moderror=10**9,
                checkpoint=10**9,
                filename='/tmp/jaxfun_pcf_mhd_shearpy_parity',
                prefer_numba=False,
                store_history=False,
                timestepper='IMEXRK222',
            )
            t, tstep = solver.initialize(False)

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [
                        [[float(z.real), float(z.imag)] for z in rowz]
                        for rowz in rowy
                    ]
                    for rowy in arr
                ]

            rows = []
            keys = (
                'Epert',
                'Etot',
                'Emag',
                'Emag_total',
                'divu_l2',
                'divb_l2',
                'top_wall_streamwise',
                'bottom_wall_streamwise',
                'mean_shear',
                'bmax',
                'bmax_total',
                'reynolds_stress',
                'maxwell_stress',
                'alpha',
                'q_shear',
                'kappa2',
            )
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.solve(t=t, tstep=tstep, end_time=target*solver.dt)
                t = target*solver.dt
                tstep = target
                row = {{key: diag[key] for key in keys}}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'u': [complex_rows(solver.u_[i]) for i in range(3)],
                        'g': complex_rows(solver.g_),
                        'A': [complex_rows(solver.a_[i]) for i in range(3)],
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )


def tc_axisymmetric_dns_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    nr: int = 8,
    nz: int = 6,
    dt: float = 1.0e-3,
    nu: float = 0.002,
    amp: float = 1.0e-4,
    family: str = "L",
    dealias: float = 1.0,
    include_coefficients: bool = False,
) -> list[dict]:
    """Run live shenfun axisymmetric hydrodynamic TC DNS and return rows."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from taylor_couette_linear import CircularCouette
            from taylor_couette_dns import AxisymmetricTCDNS

            solver = AxisymmetricTCDNS(
                CircularCouette(),
                nu={nu!r},
                Nr={int(nr)!r},
                Nz={int(nz)!r},
                dt={dt!r},
                family={family!r},
                dealias={dealias!r},
            )
            solver.set_perturbation(amp={amp!r}, kz_mode=1)

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [[float(z.real), float(z.imag)] for z in row]
                    for row in arr
                ]

            rows = []
            tstep = 0
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.run(end_time=(target - tstep)*solver.dt, moderror=0)
                tstep = target
                row = {{key: diag[key] for key in ('E', 'div_linf', 'wall', 'Eth')}}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'u': [complex_rows(solver.u_hat[i]) for i in range(3)],
                        'p': complex_rows(solver.p_hat),
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )


def tc_3d_dns_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    nr: int = 8,
    ntheta: int = 4,
    nz: int = 6,
    dt: float = 1.0e-3,
    nu: float = 0.002,
    amp: float = 1.0e-4,
    m: int = 1,
    family: str = "L",
    dealias: float = 1.0,
    include_coefficients: bool = False,
) -> list[dict]:
    """Run live shenfun 3D hydrodynamic TC DNS and return rows."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from taylor_couette_linear import CircularCouette
            from taylor_couette_dns import TaylorCouetteDNS

            solver = TaylorCouetteDNS(
                CircularCouette(),
                nu={nu!r},
                Nr={int(nr)!r},
                Ntheta={int(ntheta)!r},
                Nz={int(nz)!r},
                dt={dt!r},
                family={family!r},
                dealias={dealias!r},
            )
            solver.set_perturbation(amp={amp!r}, m={int(m)!r}, kz_mode=1)

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [
                        [[float(z.real), float(z.imag)] for z in radial]
                        for radial in zrows
                    ]
                    for zrows in arr
                ]

            rows = []
            tstep = 0
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.run(end_time=(target - tstep)*solver.dt, moderror=0)
                tstep = target
                row = {{key: diag[key] for key in ('E', 'div_linf')}}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'u': [complex_rows(solver.u_hat[i]) for i in range(3)],
                        'p': complex_rows(solver.p_hat),
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )


def tc_axisymmetric_mri_dns_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    nr: int = 8,
    nz: int = 6,
    dt: float = 1.0e-3,
    b0: float = 0.1,
    nu: float = 0.001,
    eta_mag: float = 0.001,
    amp: float = 1.0e-8,
    family: str = "L",
    dealias: float = 1.0,
    include_coefficients: bool = False,
) -> list[dict]:
    """Run live shenfun axisymmetric Taylor-Couette MHD/MRI DNS and return rows."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from taylor_couette_linear import CircularCouette
            from taylor_couette_dns import AxisymmetricMRIDNS

            eta = 0.5
            solver = AxisymmetricMRIDNS(
                CircularCouette(1.0, 2.0, 1.0, eta**1.5),
                B0={b0!r},
                nu={nu!r},
                eta_mag={eta_mag!r},
                Nr={int(nr)!r},
                Nz={int(nz)!r},
                dt={dt!r},
                family={family!r},
                dealias={dealias!r},
            )
            solver.seed_linear_eigenmode(kz_mode=1, amp={amp!r})

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [[float(z.real), float(z.imag)] for z in row]
                    for row in arr
                ]

            rows = []
            tstep = 0
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.run(end_time=(target - tstep)*solver.dt, moderror=0)
                tstep = target
                row = {{
                    key: diag[key]
                    for key in ('Ekin', 'Emag', 'E', 'divu', 'divb')
                }}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'x': [complex_rows(solver.x[i]) for i in range(6)],
                        'p': complex_rows(solver.p_hat),
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )


def tc_3d_mri_dns_reference(
    *,
    steps: tuple[int, ...] = (1, 5, 50),
    nr: int = 8,
    ntheta: int = 4,
    nz: int = 6,
    dt: float = 1.0e-3,
    b0: float = 0.1,
    nu: float = 0.001,
    eta_mag: float = 0.001,
    amp: float = 1.0e-8,
    m: int = 1,
    family: str = "L",
    dealias: float = 1.0,
    include_coefficients: bool = False,
) -> list[dict]:
    """Run live shenfun 3D Taylor-Couette MHD/MRI DNS and return rows."""
    return run_shenfun_json(
        textwrap.dedent(
            f"""
            import json
            import numpy as np
            from taylor_couette_linear import CircularCouette
            from taylor_couette_dns import TaylorCouetteMRIDNS

            eta = 0.5
            solver = TaylorCouetteMRIDNS(
                CircularCouette(1.0, 2.0, 1.0, eta**1.5),
                B0={b0!r},
                nu={nu!r},
                eta_mag={eta_mag!r},
                Nr={int(nr)!r},
                Ntheta={int(ntheta)!r},
                Nz={int(nz)!r},
                dt={dt!r},
                family={family!r},
                dealias={dealias!r},
            )
            solver.seed_linear_eigenmode(m={int(m)!r}, kz_mode=1, amp={amp!r})

            def complex_rows(arr):
                arr = np.asarray(arr)
                return [
                    [
                        [[float(z.real), float(z.imag)] for z in radial]
                        for radial in zrows
                    ]
                    for zrows in arr
                ]

            rows = []
            tstep = 0
            for target in {tuple(int(step) for step in steps)!r}:
                if target < tstep:
                    raise ValueError('steps must be sorted increasingly')
                diag = solver.run(end_time=(target - tstep)*solver.dt, moderror=0)
                tstep = target
                row = {{
                    key: diag[key]
                    for key in ('Ekin', 'Emag', 'E', 'divu', 'divb')
                }}
                row['steps'] = int(target)
                if {include_coefficients!r}:
                    row['coefficients'] = {{
                        'x': [complex_rows(solver.x[i]) for i in range(6)],
                        'p': complex_rows(solver.p_hat),
                    }}
                rows.append(row)
            print(json.dumps(rows))
            """
        )
    )
