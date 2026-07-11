"""FJ-08: step-doubling temporal-order gate for the production JAX CNAB2 block.

The plan requires a committed float64 regression that the production primitive
solver (``examples.pcf_mri_primitive_jax.PCFMRIDNSJax``, which advances the coupled
MRI block with CNAB2) stays *second order* with viscosity, resistivity, rotation,
shear, and the imposed field all active. We seed the coupled linear MRI eigenmode
and Richardson-extrapolate over dt, dt/2, dt/4: the ratio of successive-difference
norms must approach 2^2 = 4.
"""

from __future__ import annotations

import math

import jax
import pytest

pytestmark = pytest.mark.slow


def _state_l2(a, b) -> float:
    import jax.numpy as jnp

    total = 0.0
    for ca, cb in zip(a.x, b.x, strict=True):
        total += float(jnp.sum(jnp.abs(ca - cb) ** 2))
    return math.sqrt(total)


def _run(dt, steps, *, seed_kwargs):
    from examples.pcf_mri_primitive_jax import PCFMRIDNSJax

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
        dt=dt,
        dealias=1.0,
    )
    state, _ = solver.seed_linear_eigenmode(**seed_kwargs)
    return solver.solve(state, steps)


def test_pcf_mri_cnab2_is_second_order():
    jax.config.update("jax_enable_x64", True)
    seed = dict(ky_mode=0, kz_mode=1, amp=1.0e-3)  # coupled MRI block, k_y=0
    T = 0.4
    base_steps = 64
    dt = T / base_steps

    u1 = _run(dt, base_steps, seed_kwargs=seed)
    u2 = _run(dt / 2, base_steps * 2, seed_kwargs=seed)
    u4 = _run(dt / 4, base_steps * 4, seed_kwargs=seed)

    e1 = _state_l2(u1, u2)
    e2 = _state_l2(u2, u4)
    assert e2 > 0.0
    order = math.log2(e1 / e2)
    # Second-order CNAB2 gives ratio ~4 -> order ~2. Allow a generous band.
    assert 1.7 <= order <= 2.3, (
        f"observed temporal order {order:.3f} (e1={e1:.3e}, e2={e2:.3e})"
    )
