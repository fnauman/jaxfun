"""FJ-09: pseudo-vacuum magnetic wall family for the primitive PCF-MRI DNS.

Pseudo-vacuum enforces ``b_y = b_z = 0`` at the walls (Dirichlet) with the
compatible normal condition ``d_x b_x = 0`` (Neumann), the physically distinct
alternative to the perfect-conductor family (``b_x = 0``,
``d_x b_y = d_x b_z = 0``). These tests verify the basis actually imposes the
boundary condition, keeps the field solenoidal, and conserves finiteness under
a nonlinear step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mri_primitive_jax import PCFMRIDNSJax, _canonical_magnetic_bc


def _seed_velocity_state(solver):
    X, Y, Z = solver.X, solver.Y, solver.Z
    Ly = solver.Ly
    Lz = solver.Lz
    wall = 1.0 - X**2  # vanishes at the walls x = +/-1
    amp = 1e-2
    u0 = amp * wall * jnp.sin(2.0 * jnp.pi * Y / Ly) * jnp.cos(2.0 * jnp.pi * Z / Lz)
    u1 = amp * wall * jnp.cos(2.0 * jnp.pi * Y / Ly) * jnp.sin(2.0 * jnp.pi * Z / Lz)
    u2 = amp * wall * jnp.sin(4.0 * jnp.pi * Y / Ly) * jnp.cos(2.0 * jnp.pi * Z / Lz)
    zero = jnp.zeros_like(u0)
    return solver.state_from_physical((u0, u1, u2, zero, zero, zero))


def test_magnetic_bc_canonicalization():
    assert _canonical_magnetic_bc("conducting") == "perfect_conductor"
    assert _canonical_magnetic_bc("perfect_conductor") == "perfect_conductor"
    assert _canonical_magnetic_bc("pseudo_vacuum") == "pseudo_vacuum"
    with pytest.raises(ValueError):
        _canonical_magnetic_bc("insulating")  # not implemented for primitive DNS


def _stepped(magnetic_bc):
    solver = PCFMRIDNSJax(
        S=1.0,
        omega=2.0 / 3.0,
        B0=0.05,
        nu=2e-2,
        eta_mag=2e-2,
        Nx=16,
        Ny=4,
        Nz=8,
        Ly=4.0,
        Lz=1.0,
        dt=1e-3,
        dealias=1.0,
        magnetic_bc=magnetic_bc,
    )
    state = _seed_velocity_state(solver)
    for _ in range(3):  # induction generates a magnetic field from u x B0
        state = solver.step(state)
    return solver, state


def test_pseudo_vacuum_selects_dirichlet_tangential_spaces():
    """FJ-09: the pseudo-vacuum family uses Dirichlet b_y,b_z and Neumann b_x."""
    solver = PCFMRIDNSJax(Nx=8, Ny=4, Nz=4, dt=1e-3, magnetic_bc="pseudo_vacuum")
    assert solver.magnetic_bc == "pseudo_vacuum"
    assert solver.Tbx is solver.TN  # b_x Neumann (d_x b_x = 0)
    assert solver.Tby is solver.TD and solver.Tbz is solver.TD  # b_y=b_z=0
    pc = PCFMRIDNSJax(Nx=8, Ny=4, Nz=4, dt=1e-3, magnetic_bc="perfect_conductor")
    assert pc.Tbx is pc.TD and pc.Tby is pc.TN and pc.Tbz is pc.TN


def test_pseudo_vacuum_runs_and_is_finite():
    jax.config.update("jax_enable_x64", True)
    solver, state = _stepped("pseudo_vacuum")
    diag = solver.diagnostics(state)
    assert float(diag["Emag"]) > 0.0
    assert bool(jnp.isfinite(jnp.asarray(diag["E"])))
    assert bool(jnp.isfinite(jnp.asarray(diag["divb"])))


def test_pseudo_vacuum_solenoidality_matches_perfect_conductor():
    """The BC swap must not degrade solenoidality: from the same (non-div-free)
    seed, pseudo-vacuum tracks perfect-conductor div_b closely."""
    jax.config.update("jax_enable_x64", True)
    pv, pv_state = _stepped("pseudo_vacuum")
    pc, pc_state = _stepped("perfect_conductor")
    divb_pv = float(pv.diagnostics(pv_state)["divb"])
    divb_pc = float(pc.diagnostics(pc_state)["divb"])
    assert divb_pv == pytest.approx(divb_pc, rel=0.5)


def test_bc_families_are_physically_distinct():
    """The two wall families evolve the same seed into different magnetic states."""
    jax.config.update("jax_enable_x64", True)
    pv, pv_state = _stepped("pseudo_vacuum")
    pc, pc_state = _stepped("perfect_conductor")
    emag_pv = float(pv.diagnostics(pv_state)["Emag"])
    emag_pc = float(pc.diagnostics(pc_state)["Emag"])
    assert emag_pv > 0.0 and emag_pc > 0.0
    # distinct boundary conditions -> distinct magnetic energy from the same IC
    assert abs(emag_pv - emag_pc) > 1e-12 * max(emag_pv, emag_pc)
