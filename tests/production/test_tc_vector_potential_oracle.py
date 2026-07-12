"""Vector-potential Taylor-Couette MHD/MRI production contract.

The TC counterpart of tests/production/test_vector_potential_oracle.py: the
magnetic representation is ``B = B0 e_z + curl(A)`` in full 3D (theta, z, r),
so the solenoidal witness must hold its resolution floor for the *whole*
horizon at finite amplitude -- the invariant the primitive-b TC family loses
(its retained golden ends at div b ~ 8e-4).  Physics anchors: the seeded DNS
must reproduce the linear MRI eigenvalue for conducting walls (m=0 and the
non-axisymmetric m=1 mode) and for insulating walls (m=0 flux eigensolver),
and the nonlinear terms must match the primitive solver trajectory at finite
amplitude once the axisymmetric family's missing 2*pi azimuthal factor is
accounted for.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from production.oracles import load_resume_checkpoint, run_supported_spec

# Solenoidal ceiling for the TC vector-potential family.  The reported witness
# is the divergence of the forward-projected coefficient representation of
# b = curl(A); it carries the (spectrally convergent) quadrature error of the
# cylindrical 1/r projections, measured at ~1e-19 for m=0 and ~4e-15 for m=1
# at Nr=40, versus the primitive family's finite ~1e-4..1e-3 drift.  The gate
# is a few orders above the measured floor and far below the primitive regime.
SOLENOIDAL_CEIL = 1.0e-12

_CONTINUATION_KEYS = (
    "kinetic_energy",
    "magnetic_energy",
    "total_energy",
    "divergence_u_l2",
    "divergence_b_l2",
    "reynolds_stress",
    "maxwell_stress_rt",
    "total_stress",
    "mean_bz",
)


def _tc_vp_spec(magnetic_bc: str = "conducting", **overrides):
    spec = {
        "problem_id": f"tc_mri_vp_{magnetic_bc}_smoke",
        "spec_hash": f"tc-vp-{magnetic_bc}-smoke-hash",
        "numerics_contract_version": 2,
        "geometry": "taylor_couette",
        "physics": "mri",
        "representation": "vector_potential",
        "expected_oracle": {"type": "tc_mri_saturation_ladder"},
        "boundary_conditions": {
            "velocity": {"type": "no_slip_rotating_cylinders"},
            "magnetic": {"type": magnetic_bc},
        },
        "domain": {
            "r": [1.0, 2.0],
            "theta_period": 2.0 * np.pi,
            "z_period": 2.0 * np.pi / 3.0,
        },
        "nondimensional_groups": {
            "R1": 1.0,
            "R2": 2.0,
            "Omega1": 1.0,
            "Omega2": float(0.5**1.5),
            "B0": 0.1,
            "nu": 1.0e-3,
            "eta_mag": 1.0e-3,
        },
        "time": {"integrator": "CNAB2", "dt": 1.0e-3, "final_time": 0.01},
        "resolution": {
            "Nr": 20,
            "Ntheta": 4,
            "Nz": 8,
            "family": "L",
            "dealias": 1.5,
        },
        "initial_condition": {
            "type": "linear_eigenmode",
            "amplitude": 1.0e-4,
            "seeded_kz_mode": 1,
            "azimuthal_mode": 0,
        },
        "forcing": {"B0": 0.1},
        "golden": {
            "artifact_id": f"tc_mri_vp_{magnetic_bc}_smoke",
            "regeneration_command": "test-only spec; no committed golden",
        },
    }
    spec.update(overrides)
    return spec


def _max_series(out, key):
    return max(row[key] for row in out["time_series"] if key in row)


def test_tc_vp_conducting_oracle_is_solenoidal_for_the_whole_horizon():
    jax.config.update("jax_enable_x64", True)
    out = run_supported_spec(_tc_vp_spec(), steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["representation"] == "vector_potential"
    assert sc["magnetic_bc"] == "conducting"
    assert sc["energy_convention"] == "half_integral_abs2_annulus"
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert _max_series(out, "divergence_b_l2") < SOLENOIDAL_CEIL
    assert np.isfinite(sc["growth_rate_linear"])
    # Net-flux run: the volume-mean axial field is the imposed B0.
    assert sc["mean_bz"] == pytest.approx(0.1, rel=1e-6)


def test_tc_vp_insulating_oracle_holds_divergence_and_matching_rows():
    jax.config.update("jax_enable_x64", True)
    # Exercise the production seeding path end to end: m=0 eigenmode plus the
    # non-axisymmetric symmetry-breaking perturbation (the m=0 mode alone
    # would stay axisymmetric forever and never touch the m != 0
    # vacuum-matching rows).  The test amplitude is small so the m=1 modes'
    # resolution-dependent projection floor (amplitude-proportional, measured
    # ~3e-14 here) stays far under the 1e-12 gate at the smoke resolution.
    spec = _tc_vp_spec("insulating")
    spec["initial_condition"]["symmetry_break_amplitude"] = 1.0e-8
    spec["initial_condition"]["symmetry_break_m"] = 1
    out = run_supported_spec(spec, steps=4, diagnostics_every=2)
    sc = out["scalars"]
    assert sc["magnetic_bc"] == "insulating"
    assert sc["divergence_b_l2"] < SOLENOIDAL_CEIL
    assert _max_series(out, "divergence_b_l2") < SOLENOIDAL_CEIL
    # The per-mode vacuum-matching tau rows are enforced exactly by every
    # implicit stage, so evolved states sit at roundoff.  The t=0 row is the
    # eigensolver seed and carries that solver's own projection error at the
    # smoke resolution, so it is excluded from the roundoff gate.
    assert sc["insulating_bc_residual"] < SOLENOIDAL_CEIL
    t0 = out["time_series"][0]["t"]
    evolved = max(
        row["insulating_bc_residual"]
        for row in out["time_series"]
        if "insulating_bc_residual" in row and row["t"] > t0
    )
    assert evolved < SOLENOIDAL_CEIL


@pytest.mark.integration
def test_tc_vp_symmetry_breaking_populates_nonaxisymmetric_modes():
    """The m=0 eigenmode seed is invariant under nonlinear evolution, so the
    non-axisymmetric insulating dynamics (including the m != 0 Bessel rows)
    are only exercised through the symmetry-breaking perturbation.  The
    perturbation profile satisfies every wall row identically at t = 0
    (w = w' = 0 at both cylinders), stays gated over the horizon, and the
    MRI coupling must transfer m = 1 content into the velocity field."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=20,
        Ntheta=4,
        Nz=8,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="insulating",
    )
    import jax.numpy as jnp

    state, _ = solver.seed_linear_eigenmode(m=0, kz_mode=1, amp=1e-4)
    residual_seed = float(solver.insulating_bc_residual(state))
    # Amplitude keeps the m=1 projection floor (amplitude-proportional) far
    # below the solenoidal gate at this resolution; symmetry breaking needs
    # no more, since the MRI amplifies the m=1 content exponentially.
    state = solver.add_symmetry_breaking_perturbation(state, 1e-8, m=1, kz_mode=1)

    def m1_content(arrays):
        return max(float(jnp.max(jnp.abs(a[1, :, :]))) for a in arrays)

    # The perturbation populates the magnetic m=1 slots immediately and
    # satisfies the vacuum-matching rows identically (w = w' = 0 at both
    # cylinders), so the wall residual -- dominated by the m=0 eigenmode
    # seed's own projection error at this resolution -- must not increase.
    assert m1_content(state.A) > 0.0
    residual_perturbed = float(solver.insulating_bc_residual(state))
    assert residual_perturbed <= residual_seed + 1.0e-14
    assert m1_content(state.u) == 0.0

    state = solver.solve(state, 10)
    diag = solver.diagnostics(state)
    assert float(diag["divb_l2"]) < SOLENOIDAL_CEIL
    assert float(diag["insulating_bc_residual"]) < SOLENOIDAL_CEIL
    # MRI coupling feeds the non-axisymmetric velocity modes.
    assert m1_content(state.u) > 0.0


@pytest.mark.integration
def test_tc_vp_checkpoint_resume_matches_straight_run(tmp_path):
    jax.config.update("jax_enable_x64", True)
    spec = _tc_vp_spec()

    straight_dir = tmp_path / "straight"
    straight = run_supported_spec(
        spec, steps=4, out_dir=straight_dir, checkpoint_every=2
    )
    parent_dir = tmp_path / "parent"
    run_supported_spec(spec, steps=2, out_dir=parent_dir, checkpoint_every=2)
    record = load_resume_checkpoint(parent_dir)
    assert record.tstep == 2
    assert str(record.attrs["state_kind"]) == "tc_vector_potential_mhd_saturation"
    resumed = run_supported_spec(spec, steps=4, resume_checkpoint=record)

    expected_final_time = 4 * float(spec["time"]["dt"])
    assert straight["time_series"][-1]["t"] == pytest.approx(expected_final_time)
    assert resumed["time_series"][-1]["t"] == pytest.approx(expected_final_time)
    assert [row["t"] for row in resumed["time_series"]] == sorted(
        row["t"] for row in resumed["time_series"]
    )

    for key in _CONTINUATION_KEYS:
        assert np.isclose(
            resumed["scalars"][key],
            straight["scalars"][key],
            rtol=1e-10,
            atol=1e-14,
        ), key


def test_tc_vp_rejects_unsupported_magnetic_bc():
    from production.oracles import ProductionOracleNotImplementedError

    jax.config.update("jax_enable_x64", True)
    with pytest.raises(ProductionOracleNotImplementedError, match="conducting or"):
        run_supported_spec(_tc_vp_spec("pseudo_vacuum"), steps=1)


@pytest.mark.slow
def test_tc_vp_conducting_m0_growth_matches_linear_eigenvalue():
    """The seeded DNS must reproduce the conducting linear MRI growth rate."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=24,
        Ntheta=4,
        Nz=12,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="conducting",
    )
    state, ev = solver.seed_linear_eigenmode(m=0, kz_mode=1, amp=1e-6)
    assert ev.real > 0.1  # MRI-unstable anchor point
    state = solver.solve(state, 100)
    e0 = float(solver.energy(state))
    divbs = []
    for _ in range(3):
        state = solver.solve(state, 100)
        divbs.append(float(solver.diagnostics(state)["divb_l2"]))
    e1 = float(solver.energy(state))
    gamma = 0.5 * np.log(e1 / e0) / (300 * solver.dt)
    assert gamma == pytest.approx(ev.real, rel=1e-5)
    assert max(divbs) < SOLENOIDAL_CEIL


@pytest.mark.slow
def test_tc_vp_nonaxisymmetric_m1_growth_matches_linear_eigenvalue():
    """Full 3D: the m=1 non-axisymmetric mode is evolved, not truncated."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    # Ntheta=4 resolves the seeded m=1 mode (2|m| < Ntheta); at amplitude 1e-6
    # the truncated m=2 harmonics are ~1e-12 and irrelevant to the growth
    # measurement.  Kept lean so four xdist workers fit the CI runner memory.
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=32,
        Ntheta=4,
        Nz=8,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="conducting",
    )
    state, ev = solver.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-6)
    assert ev.real > 0.05
    state = solver.solve(state, 50)
    e0 = float(solver.energy(state))
    state = solver.solve(state, 250)
    e1 = float(solver.energy(state))
    gamma = 0.5 * np.log(e1 / e0) / (250 * solver.dt)
    assert gamma == pytest.approx(ev.real, rel=1e-5)
    assert float(solver.diagnostics(state)["divb_l2"]) < SOLENOIDAL_CEIL


@pytest.mark.slow
def test_tc_vp_insulating_m0_growth_matches_flux_eigensolver():
    """Insulating anchor: the m=0 vacuum-matched DNS growth rate must match
    the independent flux-function (chi, b_theta) linear eigensolver."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=24,
        Ntheta=4,
        Nz=12,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="insulating",
    )
    state, ev = solver.seed_linear_eigenmode(m=0, kz_mode=1, amp=1e-6)
    assert ev.real > 0.1
    state = solver.solve(state, 100)
    e0 = float(solver.energy(state))
    state = solver.solve(state, 300)
    e1 = float(solver.energy(state))
    gamma = 0.5 * np.log(e1 / e0) / (300 * solver.dt)
    assert gamma == pytest.approx(ev.real, rel=1e-5)
    diag = solver.diagnostics(state)
    assert float(diag["divb_l2"]) < SOLENOIDAL_CEIL
    assert float(diag["insulating_bc_residual"]) < SOLENOIDAL_CEIL


@pytest.mark.slow
def test_tc_vp_nonlinear_parity_with_primitive_family():
    """Finite-amplitude cross-representation parity.

    Both solvers evolve the same conducting m=0 eigenmode seed at finite
    amplitude; the physics is identical (the VP conducting rows are the
    on-shell resistive perfect-conductor set), so the energy trajectories
    must agree once the axisymmetric family's missing 2*pi azimuthal factor
    is applied.  This validates every nonlinear term (advection, Lorentz,
    EMF) of the vector-potential implementation against the independently
    validated primitive solver.
    """
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_dns_jax import AxisymmetricMRIDNSJax
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    kw = dict(B0=0.1, nu=1e-3, eta_mag=1e-3, dt=1e-3, family="L", dealias=1.5)
    vp = TaylorCouetteVPMRIDNSJax(
        base, Nr=28, Ntheta=2, Nz=16, **kw, magnetic_bc="conducting"
    )
    pr = AxisymmetricMRIDNSJax(base, Nr=28, Nz=16, **kw)
    svp, _ = vp.seed_linear_eigenmode(m=0, kz_mode=1, amp=3e-2)
    spr, _ = pr.seed_linear_eigenmode(kz_mode=1, amp=3e-2)
    two_pi = 2.0 * np.pi
    for _ in range(4):
        svp = vp.solve(svp, 50)
        spr = pr.solve(spr, 50)
        dv = vp.diagnostics(svp)
        dp = pr.diagnostics(spr)
        assert float(dv["Ekin"]) / two_pi == pytest.approx(float(dp["Ekin"]), rel=1e-7)
        assert float(dv["Emag"]) / two_pi == pytest.approx(float(dp["Emag"]), rel=1e-7)
        # The vector-potential representation stays solenoidal while the
        # primitive representation is already drifting at this amplitude.
        assert float(dv["divb_l2"]) < SOLENOIDAL_CEIL


@pytest.mark.slow
def test_tc_vp_finite_amplitude_divergence_does_not_grow():
    """The user-facing invariant: div b must NOT grow over a long nonlinear
    horizon at finite amplitude (the primitive family's failure mode)."""
    jax.config.update("jax_enable_x64", True)
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_vp_jax import TaylorCouetteVPMRIDNSJax

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=0.1,
        nu=1e-3,
        eta_mag=1e-3,
        Nr=28,
        Ntheta=4,
        Nz=16,
        dt=1e-3,
        family="L",
        dealias=1.5,
        magnetic_bc="conducting",
    )
    state, _ = solver.seed_linear_eigenmode(m=0, kz_mode=1, amp=3e-2)
    divbs = []
    for _ in range(6):
        state = solver.solve(state, 50)
        divbs.append(float(solver.diagnostics(state)["divb_l2"]))
    assert max(divbs) < SOLENOIDAL_CEIL
    # Non-growth over the horizon: the late-window maximum must not exceed a
    # small multiple of the early-window maximum (roundoff jitter allowance).
    assert max(divbs[3:]) < 10.0 * max(divbs[:3]) + 1.0e-15
