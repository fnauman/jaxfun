import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from examples.pcf_mhd_jax import PlaneCouetteMHDJax
from examples.pcf_mhd_mri_shearpy_jax import PlaneCouetteMRIShearpyJax
from examples.pcf_mri_primitive_jax import (
    AxisymmetricPCFMRIDNSJax,
    AxisymmetricPCFState,
    PCFMRIDNSJax,
)
from examples.taylor_couette_dns_jax import (
    AxisymmetricMRIDNSJax,
    AxisymmetricTCDNSJax,
    TaylorCouetteDNSJax,
    TaylorCouetteMRIDNSJax,
)
from examples.taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
from jaxfun import Domain
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre
from tests._parity import (
    pcf_fluctuation_reference,
    pcf_mhd_reference,
    pcf_mhd_shearpy_reference,
    pcf_primitive_3d_reference,
    pcf_primitive_axisymmetric_reference,
    run_shenfun_json,
    tc_3d_dns_reference,
    tc_3d_mri_dns_reference,
    tc_axisymmetric_dns_reference,
    tc_axisymmetric_mri_dns_reference,
    tc_linear_critical_scan,
    tc_linear_eigenvalues,
    tc_linear_nonmodal,
    tc_linear_operator_parts,
    tc_mri_critical_scans,
    tc_mri_eigenvalues,
    tc_mri_nonmodal,
    tc_mri_operator_parts,
    tc_radial_dealias_product,
)

pytestmark = [pytest.mark.integration, pytest.mark.live_shenfun]

TC_DNS_PARITY_STEPS = (1, 5, 50, 100)
TC_MHD_NONLINEAR_PARITY_STEPS = (50,)
TC_MHD_NONLINEAR_PARITY_AMP = 1.0e-3
TC_AXISYM_MRI_REFERENCE_AMP_SCALE = 0.5
PCF_PRIMITIVE_NONLINEAR_PARITY_STEPS = (10,)
PCF_PRIMITIVE_NONLINEAR_PARITY_AMP = 1.0e-3


def test_live_shenfun_reference_runner_executes():
    assert run_shenfun_json("print(json.dumps({'ok': True}))") == {"ok": True}


def _keplerian_base():
    eta = 0.5
    return CircularCouette(1.0, 2.0, 1.0, eta**1.5)


def _nested_complex(rows):
    arr = np.asarray(rows, dtype=float)
    return arr[..., 0] + 1j * arr[..., 1]


def _assert_eigenvalue_multiset_close(got, expected, *, rtol, atol):
    """Compare spectra without assigning significance to conjugate ordering.

    LAPACK may perturb the nominally identical real parts of a conjugate pair
    in opposite directions, so a raw lexicographic ordering is not stable
    across the independent jaxfun and Shenfun solves.
    """
    got = np.asarray(got, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    assert got.shape == expected.shape
    got_index, expected_index = linear_sum_assignment(
        np.abs(got[:, None] - expected[None, :])
    )
    assert np.allclose(got[got_index], expected[expected_index], rtol=rtol, atol=atol)


def _tc_axisymmetric_reference_layout(rows, *, axial_n: int):
    out = np.array(_nested_complex(rows), copy=True)
    if axial_n % 2 == 0:
        out[axial_n // 2, :] = 0.0
    return out


def _tc_3d_reference_layout(rows, *, axial_n: int, azimuthal_n: int):
    out = np.array(_nested_complex(rows), copy=True)
    if axial_n % 2 == 0:
        out[:, axial_n // 2, :] = 0.0
    if azimuthal_n % 2 == 0:
        out[azimuthal_n // 2, :, :] = 0.0
    return out


def _assert_conjugate_symmetric(coeff, periodic_axes: tuple[int, ...]) -> None:
    coeff_np = np.asarray(coeff)
    axis_shape = tuple(coeff_np.shape[axis] for axis in periodic_axes)
    for mode in np.ndindex(axis_shape):
        src = [slice(None)] * coeff_np.ndim
        dst = [slice(None)] * coeff_np.ndim
        for axis, index in zip(periodic_axes, mode, strict=True):
            src[axis] = index
            dst[axis] = (-index) % coeff_np.shape[axis]
        src_values = coeff_np[tuple(src)]
        dst_values = coeff_np[tuple(dst)]
        if max(np.max(np.abs(src_values)), np.max(np.abs(dst_values))) < 1.0e-8:
            continue
        assert np.allclose(
            dst_values,
            np.conj(src_values),
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def _assert_active_coefficients_close(got, expected, *, floor: float = 1.0e-8) -> None:
    got_np = np.asarray(got)
    expected_np = np.asarray(expected)
    active = (np.abs(got_np) >= floor) | (np.abs(expected_np) >= floor)
    assert np.any(active)
    assert np.allclose(got_np[active], expected_np[active], rtol=1.0e-8, atol=1.0e-10)


def _shenfun_rfft_coeff_layout(coeff, *, radial_n: int, spanwise_n: int):
    coeff_np = np.asarray(coeff)
    _assert_conjugate_symmetric(coeff_np, periodic_axes=(1, 2))
    out = np.zeros(
        (radial_n, coeff_np.shape[1], spanwise_n // 2 + 1),
        dtype=complex,
    )
    out[: coeff_np.shape[0], :, :] = coeff_np[:, :, : spanwise_n // 2 + 1]
    return out


def test_pcf_fluctuation_matches_live_shenfun_diagnostics_velocity_and_coeffs():
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
    )
    state0 = solver.initial_state()
    references = pcf_fluctuation_reference(
        include_velocity=True, include_coefficients=True
    )

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for key in ("Epert", "Etot", "u_top", "u_bot", "mean_shear"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-10, abs=1.0e-12
            )
        assert float(diag["divL2"]) == pytest.approx(
            reference["divL2"], rel=0.0, abs=5.0e-15
        )

        velocity = solver._backward_velocity(state.u)
        for got, expected in zip(velocity, reference["velocity"], strict=True):
            got_np = np.asarray(got)
            assert np.max(np.abs(got_np.imag)) < 1.0e-12
            assert np.allclose(
                got_np.real, np.asarray(expected), rtol=1.0e-8, atol=1.0e-10
            )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.u, ref_coeffs["u"], strict=True):
            got_layout = _shenfun_rfft_coeff_layout(
                got, radial_n=solver.N[0], spanwise_n=solver.N[2]
            )
            assert np.allclose(
                got_layout,
                _nested_complex(expected),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        got_g = _shenfun_rfft_coeff_layout(
            state.g, radial_n=solver.N[0], spanwise_n=solver.N[2]
        )
        assert np.allclose(
            got_g, _nested_complex(ref_coeffs["g"]), rtol=1.0e-8, atol=1.0e-10
        )


def test_kmm_pressure_recovery_matches_live_shenfun():
    solver = PlaneCouetteFluctuationJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
    )
    state = solver.solve(solver.initial_state(), 1)
    got = solver.compute_pressure(state)
    reference = pcf_fluctuation_reference(
        steps=(1,),
        include_pressure=True,
    )[0]

    got_np = np.asarray(got)
    assert np.max(np.abs(got_np.imag)) < 1.0e-12
    assert np.allclose(
        got_np.real, np.asarray(reference["pressure"]), rtol=1.0e-8, atol=1.0e-10
    )


def test_pcf_mhd_matches_live_shenfun_diagnostics_and_coeffs():
    solver = PlaneCouetteMHDJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
    )
    state0 = solver.initial_state()
    references = pcf_mhd_reference(include_coefficients=True)

    diag_key_map = {
        "Epert": "Epert",
        "Etot": "Etot",
        "Emag": "Emag",
        "divu_l2": "divL2",
        "divb_l2": "divB_L2",
        "top_wall_streamwise": "u_top",
        "bottom_wall_streamwise": "u_bot",
        "mean_shear": "mean_shear",
    }

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for ref_key, jax_key in diag_key_map.items():
            atol = 5.0e-15 if "div" in ref_key else 1.0e-12
            assert float(diag[jax_key]) == pytest.approx(
                reference[ref_key], rel=1.0e-10, abs=atol
            )

        B = solver.update_B_from_A(state.A)
        bmax = max(float(np.max(np.abs(np.asarray(b)))) for b in solver._backward_B(B))
        assert bmax == pytest.approx(reference["bmax"], rel=1.0e-10, abs=1.0e-12)

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.flow.u, ref_coeffs["u"], strict=True):
            got_layout = _shenfun_rfft_coeff_layout(
                got, radial_n=solver.N[0], spanwise_n=solver.N[2]
            )
            assert np.allclose(
                got_layout,
                _nested_complex(expected),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        got_g = _shenfun_rfft_coeff_layout(
            state.flow.g, radial_n=solver.N[0], spanwise_n=solver.N[2]
        )
        assert np.allclose(
            got_g, _nested_complex(ref_coeffs["g"]), rtol=1.0e-8, atol=1.0e-10
        )
        for got, expected in zip(state.A, ref_coeffs["A"], strict=True):
            got_layout = _shenfun_rfft_coeff_layout(
                got, radial_n=solver.N[0], spanwise_n=solver.N[2]
            )
            assert np.allclose(
                got_layout,
                _nested_complex(expected),
                rtol=1.0e-8,
                atol=1.0e-10,
            )


def test_pcf_mhd_shearpy_matches_live_shenfun_diagnostics_and_coeffs():
    solver = PlaneCouetteMRIShearpyJax(
        N=(9, 8, 8),
        family="L",
        dt=1.0e-3,
        omega=1.0,
        shear_rate=1.0,
        background_b=(0.0, 0.0, 0.1),
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
    )
    state0 = solver.initial_state()
    references = pcf_mhd_shearpy_reference(
        background_b=(0.0, 0.0, 0.1),
        perturbation_amplitude=0.05,
        magnetic_amplitude=0.05,
        include_coefficients=True,
    )

    diag_key_map = {
        "Epert": "Epert",
        "Etot": "Etot",
        "Emag": "Emag",
        "Emag_total": "Emag_total",
        "divu_l2": "divL2",
        "divb_l2": "divB_L2",
        "top_wall_streamwise": "u_top",
        "bottom_wall_streamwise": "u_bot",
        "mean_shear": "mean_shear",
        "bmax": "bmax",
        "bmax_total": "bmax_total",
        "reynolds_stress": "reynolds_stress",
        "maxwell_stress": "maxwell_stress",
        "alpha": "alpha",
        "q_shear": "q_shear",
        "kappa2": "kappa2",
    }

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for ref_key, jax_key in diag_key_map.items():
            atol = 5.0e-15 if "div" in ref_key else 1.0e-12
            assert float(diag[jax_key]) == pytest.approx(
                reference[ref_key], rel=1.0e-10, abs=atol
            )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.flow.u, ref_coeffs["u"], strict=True):
            got_layout = _shenfun_rfft_coeff_layout(
                got, radial_n=solver.N[0], spanwise_n=solver.N[2]
            )
            assert np.allclose(
                got_layout,
                _nested_complex(expected),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        got_g = _shenfun_rfft_coeff_layout(
            state.flow.g, radial_n=solver.N[0], spanwise_n=solver.N[2]
        )
        assert np.allclose(
            got_g, _nested_complex(ref_coeffs["g"]), rtol=1.0e-8, atol=1.0e-10
        )
        for got, expected in zip(state.A, ref_coeffs["A"], strict=True):
            got_layout = _shenfun_rfft_coeff_layout(
                got, radial_n=solver.N[0], spanwise_n=solver.N[2]
            )
            assert np.allclose(
                got_layout,
                _nested_complex(expected),
                rtol=1.0e-8,
                atol=1.0e-10,
            )


def test_pcf_primitive_axisymmetric_finite_amplitude_matches_live_shenfun():
    solver = AxisymmetricPCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=1.0e-3,
        eta_mag=1.0e-3,
        Nx=8,
        Nz=6,
        Lz=1.0,
        dt=1.0e-3,
        family="C",
        dealias=1.5,
    )
    references = pcf_primitive_axisymmetric_reference(
        steps=(0, *PCF_PRIMITIVE_NONLINEAR_PARITY_STEPS),
        n=(solver.Nx, solver.Nz),
        amp=PCF_PRIMITIVE_NONLINEAR_PARITY_AMP,
        dealias=1.5,
        include_coefficients=True,
    )
    initial_reference, reference = references[0], references[-1]

    state0 = _pcf_axisymmetric_state_from_reference(solver, initial_reference)
    state = solver.solve(state0, reference["steps"])
    diag = solver.diagnostics(state)
    assert float(diag["divu"]) < 1.0e-4
    assert float(diag["divb"]) < 1.0e-4

    ref_coeffs = reference["coefficients"]
    for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
        got_layout = _shenfun_tc_rfft_coeff_layout(
            got, radial_n=solver.Nx, axial_n=solver.Nz, mask_nyquist=True
        )
        _assert_active_coefficients_close(
            got_layout,
            _tc_axisymmetric_reference_layout(expected, axial_n=solver.Nz),
            floor=1.0e-10,
        )


@pytest.mark.parametrize("magnetic_bc", ["conducting", "pseudo_vacuum"])
def test_pcf_primitive_3d_finite_amplitude_matches_live_shenfun(magnetic_bc):
    solver = PCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.1,
        nu=1.0e-3,
        eta_mag=1.0e-3,
        Nx=8,
        Ny=4,
        Nz=6,
        Ly=4.0,
        Lz=1.0,
        dt=1.0e-3,
        family="C",
        dealias=1.0,
        magnetic_bc=magnetic_bc,
    )
    references = pcf_primitive_3d_reference(
        steps=(0, *PCF_PRIMITIVE_NONLINEAR_PARITY_STEPS),
        n=(solver.Nx, solver.Ny, solver.Nz),
        amp=PCF_PRIMITIVE_NONLINEAR_PARITY_AMP,
        magnetic_bc=magnetic_bc,
        include_coefficients=True,
    )
    initial_reference, reference = references[0], references[-1]

    state0 = _pcf_3d_state_from_reference(solver, initial_reference)
    state = solver.solve(state0, reference["steps"])
    diag = solver.diagnostics(state)
    assert float(diag["divu"]) < 1.0e-4
    assert float(diag["divb"]) < 1.0e-4

    ref_coeffs = reference["coefficients"]
    for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
        got_layout = _shenfun_tc3d_rfft_coeff_layout(
            got, radial_n=solver.Nx, axial_n=solver.Nz, mask_nyquist=True
        )
        _assert_active_coefficients_close(
            got_layout,
            _tc_3d_reference_layout(expected, axial_n=solver.Nz, azimuthal_n=solver.Ny),
            floor=1.0e-10,
        )


def _pcf_axisymmetric_coeff_from_reference(rows, template, *, axial_n: int):
    layout = _nested_complex(rows)
    out = np.zeros(np.asarray(template).shape, dtype=np.asarray(template).dtype)
    out[: axial_n // 2 + 1, : layout.shape[1]] = layout[:, : out.shape[1]]
    for k in range(1, axial_n // 2):
        out[-k, : layout.shape[1]] = np.conj(layout[k, : out.shape[1]])
    return jnp.asarray(out)


def _pcf_3d_coeff_from_reference(rows, template, *, azimuthal_n: int, axial_n: int):
    layout = _nested_complex(rows)
    out = np.zeros(np.asarray(template).shape, dtype=np.asarray(template).dtype)
    out[:, : axial_n // 2 + 1, : layout.shape[2]] = layout[:, :, : out.shape[2]]
    for m in range(azimuthal_n):
        mneg = (-m) % azimuthal_n
        for k in range(1, axial_n // 2):
            out[mneg, -k, : layout.shape[2]] = np.conj(layout[m, k, : out.shape[2]])
    return jnp.asarray(out)


def _pcf_axisymmetric_state_from_reference(solver, reference):
    zero = solver.zero_state()
    x = tuple(
        _pcf_axisymmetric_coeff_from_reference(rows, template, axial_n=solver.Nz)
        for rows, template in zip(reference["coefficients"]["x"], zero.x, strict=True)
    )
    return AxisymmetricPCFState(
        x=x,
        p=jnp.zeros_like(zero.p),
        nonlinear_old=tuple(jnp.zeros_like(component) for component in x),
        have_old=False,
    )


def _pcf_3d_state_from_reference(solver, reference):
    zero = solver.zero_state()
    x = tuple(
        _pcf_3d_coeff_from_reference(
            rows, template, azimuthal_n=solver.Ny, axial_n=solver.Nz
        )
        for rows, template in zip(reference["coefficients"]["x"], zero.x, strict=True)
    )
    return AxisymmetricPCFState(
        x=x,
        p=jnp.zeros_like(zero.p),
        nonlinear_old=tuple(jnp.zeros_like(component) for component in x),
        have_old=False,
    )


def _shenfun_tc_rfft_coeff_layout(
    coeff, *, radial_n: int, axial_n: int, mask_nyquist: bool = False
):
    coeff_np = np.asarray(coeff)
    _assert_conjugate_symmetric(coeff_np, periodic_axes=(0,))
    out = np.zeros((axial_n // 2 + 1, radial_n), dtype=complex)
    out[:, : coeff_np.shape[1]] = coeff_np[: axial_n // 2 + 1, :]
    if mask_nyquist and axial_n % 2 == 0:
        out[axial_n // 2, :] = 0.0
    return out


def test_tc_axisymmetric_dns_matches_live_shenfun_diagnostics_and_coeffs():
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), nu=0.002, Nr=8, Nz=6, dt=1.0e-3, dealias=1.0
    )
    state0 = solver.initial_state(amp=1.0e-4)
    references = tc_axisymmetric_dns_reference(
        steps=TC_DNS_PARITY_STEPS, include_coefficients=True
    )

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for key in ("E", "div_linf", "wall", "Eth"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-10, abs=1.0e-12
            )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.u, ref_coeffs["u"], strict=True):
            got_layout = _shenfun_tc_rfft_coeff_layout(
                got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
            )
            expected_layout = _tc_axisymmetric_reference_layout(
                expected, axial_n=solver.Nz
            )
            assert np.allclose(got_layout, expected_layout, rtol=1.0e-8, atol=1.0e-10)
        p_layout = _shenfun_tc_rfft_coeff_layout(
            state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        assert np.allclose(
            p_layout,
            _tc_axisymmetric_reference_layout(ref_coeffs["p"], axial_n=solver.Nz),
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def _shenfun_tc3d_rfft_coeff_layout(
    coeff, *, radial_n: int, axial_n: int, mask_nyquist: bool = False
):
    coeff_np = np.asarray(coeff)
    out = np.zeros((coeff_np.shape[0], axial_n // 2 + 1, radial_n), dtype=complex)
    out[:, :, : coeff_np.shape[2]] = coeff_np[:, : axial_n // 2 + 1, :]
    if mask_nyquist and axial_n % 2 == 0:
        out[:, axial_n // 2, :] = 0.0
    if mask_nyquist and coeff_np.shape[0] % 2 == 0:
        out[coeff_np.shape[0] // 2, :, :] = 0.0
    return out


def test_tc_3d_dns_matches_live_shenfun_diagnostics_and_coeffs():
    solver = TaylorCouetteDNSJax(
        CircularCouette(),
        nu=0.002,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state0 = solver.initial_state(amp=1.0e-4, m=1, kz_mode=1)
    references = tc_3d_dns_reference(
        steps=TC_DNS_PARITY_STEPS, include_coefficients=True
    )

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        assert float(diag["E"]) == pytest.approx(
            reference["E"], rel=1.0e-10, abs=1.0e-12
        )
        assert float(diag["div_linf"]) == pytest.approx(
            reference["div_linf"], rel=2.0e-5, abs=1.0e-10
        )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.u, ref_coeffs["u"], strict=True):
            got_layout = _shenfun_tc3d_rfft_coeff_layout(
                got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
            )
            assert np.allclose(
                got_layout,
                _tc_3d_reference_layout(
                    expected, axial_n=solver.Nz, azimuthal_n=solver.Ntheta
                ),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        p_layout = _shenfun_tc3d_rfft_coeff_layout(
            state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        assert np.allclose(
            p_layout,
            _tc_3d_reference_layout(
                ref_coeffs["p"], axial_n=solver.Nz, azimuthal_n=solver.Ntheta
            ),
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_tc_axisymmetric_mri_dns_matches_live_shenfun_diagnostics_and_coeffs():
    solver = AxisymmetricMRIDNSJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state0, _ = solver.seed_linear_eigenmode(kz_mode=1, amp=1.0e-8)
    references = tc_axisymmetric_mri_dns_reference(
        steps=TC_DNS_PARITY_STEPS,
        amp=1.0e-8 * TC_AXISYM_MRI_REFERENCE_AMP_SCALE,
        include_coefficients=True,
    )

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for key in ("Ekin", "Emag", "E"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-10, abs=1.0e-24
            )
        for key in ("divu", "divb"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-8, abs=1.0e-12
            )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
            got_layout = _shenfun_tc_rfft_coeff_layout(
                got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
            )
            assert np.allclose(
                got_layout,
                _tc_axisymmetric_reference_layout(expected, axial_n=solver.Nz),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        p_layout = _shenfun_tc_rfft_coeff_layout(
            state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        assert np.allclose(
            p_layout,
            _tc_axisymmetric_reference_layout(ref_coeffs["p"], axial_n=solver.Nz),
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_tc_axisymmetric_mri_dns_finite_amplitude_coeffs_match_live_shenfun():
    solver = AxisymmetricMRIDNSJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state0, _ = solver.seed_linear_eigenmode(kz_mode=1, amp=TC_MHD_NONLINEAR_PARITY_AMP)
    reference = tc_axisymmetric_mri_dns_reference(
        steps=TC_MHD_NONLINEAR_PARITY_STEPS,
        amp=TC_MHD_NONLINEAR_PARITY_AMP * TC_AXISYM_MRI_REFERENCE_AMP_SCALE,
        include_coefficients=True,
    )[0]

    state = solver.solve(state0, reference["steps"])
    diag = solver.diagnostics(state)
    for key in ("Ekin", "Emag", "E"):
        assert float(diag[key]) == pytest.approx(
            reference[key], rel=1.0e-10, abs=1.0e-12
        )
    ref_coeffs = reference["coefficients"]
    for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
        got_layout = _shenfun_tc_rfft_coeff_layout(
            got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        _assert_active_coefficients_close(
            got_layout, _tc_axisymmetric_reference_layout(expected, axial_n=solver.Nz)
        )
    p_layout = _shenfun_tc_rfft_coeff_layout(
        state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
    )
    _assert_active_coefficients_close(
        p_layout, _tc_axisymmetric_reference_layout(ref_coeffs["p"], axial_n=solver.Nz)
    )


def test_tc_3d_mri_dns_matches_live_shenfun_diagnostics_and_coeffs():
    solver = TaylorCouetteMRIDNSJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state0, _ = solver.seed_linear_eigenmode(m=1, kz_mode=1, amp=1.0e-8)
    references = tc_3d_mri_dns_reference(
        steps=TC_DNS_PARITY_STEPS, include_coefficients=True
    )

    for reference in references:
        state = solver.solve(state0, reference["steps"])
        diag = solver.diagnostics(state)
        for key in ("Ekin", "Emag", "E"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-10, abs=1.0e-24
            )
        for key in ("divu", "divb"):
            assert float(diag[key]) == pytest.approx(
                reference[key], rel=1.0e-8, abs=1.0e-12
            )

        ref_coeffs = reference["coefficients"]
        for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
            got_layout = _shenfun_tc3d_rfft_coeff_layout(
                got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
            )
            assert np.allclose(
                got_layout,
                _tc_3d_reference_layout(
                    expected, axial_n=solver.Nz, azimuthal_n=solver.Ntheta
                ),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
        p_layout = _shenfun_tc3d_rfft_coeff_layout(
            state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        assert np.allclose(
            p_layout,
            _tc_3d_reference_layout(
                ref_coeffs["p"], axial_n=solver.Nz, azimuthal_n=solver.Ntheta
            ),
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_tc_3d_mri_dns_finite_amplitude_coeffs_match_live_shenfun():
    solver = TaylorCouetteMRIDNSJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        Nr=8,
        Ntheta=4,
        Nz=6,
        dt=1.0e-3,
        dealias=1.0,
    )
    state0, _ = solver.seed_linear_eigenmode(
        m=1, kz_mode=1, amp=TC_MHD_NONLINEAR_PARITY_AMP
    )
    reference = tc_3d_mri_dns_reference(
        steps=TC_MHD_NONLINEAR_PARITY_STEPS,
        amp=TC_MHD_NONLINEAR_PARITY_AMP,
        include_coefficients=True,
    )[0]

    state = solver.solve(state0, reference["steps"])
    diag = solver.diagnostics(state)
    for key in ("Ekin", "Emag", "E"):
        assert float(diag[key]) == pytest.approx(
            reference[key], rel=1.0e-10, abs=1.0e-12
        )
    ref_coeffs = reference["coefficients"]
    for got, expected in zip(state.x, ref_coeffs["x"], strict=True):
        got_layout = _shenfun_tc3d_rfft_coeff_layout(
            got, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
        )
        _assert_active_coefficients_close(
            got_layout,
            _tc_3d_reference_layout(
                expected, axial_n=solver.Nz, azimuthal_n=solver.Ntheta
            ),
        )
    p_layout = _shenfun_tc3d_rfft_coeff_layout(
        state.p, radial_n=solver.Nr, axial_n=solver.Nz, mask_nyquist=True
    )
    _assert_active_coefficients_close(
        p_layout,
        _tc_3d_reference_layout(
            ref_coeffs["p"], axial_n=solver.Nz, azimuthal_n=solver.Ntheta
        ),
    )


@pytest.mark.parametrize("family", ["L", "C"], ids=["legendre", "chebyshev"])
def test_tc_linear_operator_parts_match_live_shenfun(family):
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.002, N=8, family=family)
    got = solver.assemble_parts(m=1, kz=2.0)
    ref = tc_linear_operator_parts(n=8, m=1, kz=2.0, family=family)

    for name, matrix in zip(("L0", "Lv", "M"), got, strict=True):
        assert np.allclose(
            matrix, _nested_complex(ref[name]), rtol=1.0e-10, atol=1.0e-12
        ), name


def test_tc_linear_matches_live_shenfun_eigenvalues_and_nonmodal():
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.002, N=12, family="L")

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)
    _assert_eigenvalue_multiset_close(
        w, tc_linear_eigenvalues(), rtol=1.0e-11, atol=1.0e-11
    )

    rows = solver.nonmodal_growth(m=0, kz=3.0, times=[0.0, 0.5], n_modes=12)
    ref_rows = tc_linear_nonmodal()
    for row, ref in zip(rows, ref_rows, strict=True):
        assert row["t"] == pytest.approx(ref["t"], abs=0.0)
        assert row["gain"] == pytest.approx(ref["gain"], rel=1.0e-10, abs=1.0e-10)


def _assert_scalar_dict_close(got, ref, *, rel=1.0e-10, abs=1.0e-10):
    assert got.keys() == ref.keys()
    for key, value in got.items():
        assert value == pytest.approx(ref[key], rel=rel, abs=abs), key


def test_tc_linear_critical_scan_matches_live_shenfun():
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.001, N=8, family="L")
    kz_c, nu_c = solver.critical_over_kz(
        m=0, kz_list=np.array([2.0, 3.0, 4.0]), iters=8
    )
    got = {
        "kz_c": kz_c,
        "nu_c": nu_c,
        "Re_c": solver.base.Omega1 * solver.base.R1 * solver.base.gap / nu_c,
        "a_c": kz_c * solver.base.gap,
    }

    _assert_scalar_dict_close(got, tc_linear_critical_scan(n=8, iters=8))


@pytest.mark.parametrize(
    ("magnetic_bc", "m"),
    [("conducting", 1), ("insulating", 0)],
    ids=["conducting-primitive", "insulating-flux"],
)
def test_tc_mri_operator_parts_match_live_shenfun(magnetic_bc, m):
    solver = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=8,
        family="L",
        magnetic_bc=magnetic_bc,
    )
    got = solver.assemble_parts(m=m, kz=2.0)
    ref = tc_mri_operator_parts(magnetic_bc=magnetic_bc, n=8, m=m, kz=2.0)

    for name, matrix in zip(("L0", "Lnu", "Leta", "M"), got, strict=True):
        assert np.allclose(
            matrix, _nested_complex(ref[name]), rtol=1.0e-10, atol=1.0e-12
        ), name


@pytest.mark.parametrize("magnetic_bc", ["conducting", "insulating"])
def test_tc_mri_critical_scans_match_live_shenfun(magnetic_bc):
    solver = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=8,
        family="L",
        magnetic_bc=magnetic_bc,
    )
    kzs = np.array([2.0, 3.0])
    got = {
        "fixed_B0_nu": solver.critical_Rm_fixed_B0_nu(0, kzs, iters=8),
        "fixed_controls": solver.critical_Rm(0, kzs, iters=8),
    }
    ref = tc_mri_critical_scans(magnetic_bc=magnetic_bc, n=8, iters=8)

    assert got.keys() == ref.keys()
    for scan, values in got.items():
        assert values is not None
        _assert_scalar_dict_close(values, ref[scan])


@pytest.mark.parametrize("magnetic_bc", ["conducting", "insulating"])
def test_tc_mri_matches_live_shenfun_eigenvalues_and_nonmodal(magnetic_bc):
    solver = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=12,
        family="L",
        magnetic_bc=magnetic_bc,
    )

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)
    _assert_eigenvalue_multiset_close(
        w,
        tc_mri_eigenvalues(magnetic_bc=magnetic_bc),
        rtol=1.0e-11,
        atol=1.0e-11,
    )

    solver_small = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=10,
        family="L",
        magnetic_bc=magnetic_bc,
    )
    rows = solver_small.nonmodal_growth(
        m=0, kz=3.0, times=[0.0, 0.25], n_modes=10, energy="total"
    )
    ref_rows = tc_mri_nonmodal(magnetic_bc=magnetic_bc, n=10, energy="total")
    for row, ref in zip(rows, ref_rows, strict=True):
        assert row["t"] == pytest.approx(ref["t"], abs=0.0)
        assert row["gain"] == pytest.approx(ref["gain"], rel=1.0e-8, abs=1.0e-8)


def test_radial_polynomial_dealiasing_matches_live_shenfun_product():
    n = 8
    F = FunctionSpace(n, Fourier, domain=Domain(0.0, 2.0 * np.pi))
    S = FunctionSpace(n, Legendre, domain=Domain(1.0, 2.0))
    T = TensorProduct(F, S)
    Tp = T.get_dealiased((1.5, 1.5))
    u = jnp.zeros(T.num_dofs, dtype=complex)
    v = jnp.zeros(T.num_dofs, dtype=complex)
    u = u.at[0, 1].set(0.5).at[1, 2].set(0.75 + 0.25j)
    v = v.at[0, 3].set(-0.4).at[1, 1].set(-0.2 + 0.1j)

    h = Tp.forward(Tp.backward(u) * Tp.backward(v))

    assert np.allclose(h, tc_radial_dealias_product(n=n), rtol=1.0e-12, atol=1.0e-12)
