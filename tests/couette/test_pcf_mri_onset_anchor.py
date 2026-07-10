"""FJ-10: committed PCF-MRI conducting-wall onset anchor.

Verified reference (independent recomputation, plan FJ-10): under ``h=1``, ``S=1``,
``Omega=2/3``, full box ``(2, 2, 0.5)``, ``Re_h=400``, ``B0=0.025`` the conducting-wall
marginal resistive Reynolds number is ``Rm_h,c = 415.288`` for the ``k_y=0`` vertical
``n=1`` mode, with ``gamma(Rm_h=420) = 3.17887e-3``. The ``n=2`` vertical mode only
onsets near ``Rm_h = 3172`` and the ``k_y in {pi, 2pi}`` modes stay stable to
``Rm_h = 8000``.

Convention traps encoded here: ``S=+1`` with ``U'=-S`` via ``shearpy(shear_rate=1)``;
full-width ``[-1, 1]`` domain so ``h=1`` (operator built directly, not via an
``a=0.5`` atlas); imposed vertical field magnitude ``B0 = bz``. The dense eigensolves
are ~10 s each, so the whole module is ``@slow``.
"""

from __future__ import annotations

import math

import jax
import pytest

pytestmark = pytest.mark.slow

_LZ = 0.5
_KZ1 = 2.0 * math.pi / _LZ  # vertical mode n=1
_KZ2 = 2.0 * math.pi * 2 / _LZ  # vertical mode n=2


def _op(Rm, *, nx=64):
    from examples.pcf_linear_jax import PlaneCouetteLinear

    return PlaneCouetteLinear.shearpy(
        nx=nx, Re=400.0, Rm=Rm, shear_rate=1.0, omega=2.0 / 3.0,
        bz=0.025, magnetic_bc="conducting",
    )


def test_anchor_growth_rate_at_Rm_420():
    jax.config.update("jax_enable_x64", True)
    gamma = float(_op(420.0).growth_rate(0.0, _KZ1))
    assert gamma == pytest.approx(3.17887e-3, rel=1e-4)
    # marginal at the committed critical Rm
    gamma_c = float(_op(415.288).growth_rate(0.0, _KZ1))
    assert abs(gamma_c) < 1e-5


def test_anchor_critical_Rm_bisection():
    jax.config.update("jax_enable_x64", True)
    from production.onset import critical_Rm

    res = critical_Rm(
        lambda Rm: _op(Rm), ky=0.0, kz=_KZ1, Rm_lo=350.0, Rm_hi=500.0, tol=1e-3
    )
    assert res.critical_Rm == pytest.approx(415.288, rel=2e-3)


def test_anchor_resolution_robustness():
    jax.config.update("jax_enable_x64", True)
    # The marginal point is unchanged across wall resolutions 48 and 96.
    for nx in (48, 96):
        gamma = float(_op(420.0, nx=nx).growth_rate(0.0, _KZ1))
        assert gamma == pytest.approx(3.17887e-3, rel=5e-3)


def test_anchor_competing_modes():
    jax.config.update("jax_enable_x64", True)
    # n=2 vertical mode stays stable well past n=1 onset, marginal near Rm=3172.
    assert float(_op(420.0).growth_rate(0.0, _KZ2)) < 0.0
    assert float(_op(2000.0).growth_rate(0.0, _KZ2)) < 0.0
    assert abs(float(_op(3172.0).growth_rate(0.0, _KZ2))) < 1e-3
    # k_y = pi, 2pi non-axisymmetric modes remain stable even at Rm=8000.
    assert float(_op(8000.0).growth_rate(math.pi, _KZ1)) < 0.0
    assert float(_op(8000.0).growth_rate(2.0 * math.pi, _KZ1)) < 0.0
