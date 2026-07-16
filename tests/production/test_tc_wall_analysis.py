import math

import jax.numpy as jnp
import pytest

from examples.taylor_couette_dns_jax import AxisymmetricTCDNSJax, CircularCouette
from production.oracles import _tc_inner_torque


def test_tc_inner_torque_uses_exact_chebyshev_wall_trace() -> None:
    solver = AxisymmetricTCDNSJax(
        CircularCouette(), Nr=8, Nz=4, Lz=1.5, nu=0.002, family="C", dealias=1.0
    )
    amplitude = 0.2
    perturbation = jnp.broadcast_to(
        amplitude * (solver.R - solver.base.R1) * (solver.base.R2 - solver.R),
        solver.T0.num_quad_points,
    )
    zero = jnp.zeros_like(perturbation)
    state = solver.state_from_physical((zero, perturbation, zero))

    exact_perturbation_shear = amplitude * (solver.base.R2 - solver.base.R1)
    base_shear = -2.0 * solver.base.b / solver.base.R1**2
    expected = (
        2.0
        * math.pi
        * solver.nu
        * solver.base.R1**2
        * abs(base_shear + exact_perturbation_shear)
    )
    assert _tc_inner_torque(solver, state) == pytest.approx(expected, rel=1.0e-12)
