"""Third-order and mode-local Taylor--Couette solver regressions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.taylor_couette_dns_jax import (
    AxisymmetricMRIDNSJax,
    AxisymmetricTCDNSJax,
    CircularCouette,
    TaylorCouetteDNSJax,
    TaylorCouetteMRIDNSJax,
)


def _keplerian_base() -> CircularCouette:
    return CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)


@pytest.mark.parametrize("integrator", ["CNAB2", "SBDF3", "IMEXRK3"])
@pytest.mark.parametrize(
    "factory",
    [
        lambda integrator: AxisymmetricTCDNSJax(
            CircularCouette(),
            Nr=8,
            Nz=4,
            dealias=1.0,
            time_integrator=integrator,
        ),
        lambda integrator: TaylorCouetteDNSJax(
            CircularCouette(),
            Nr=8,
            Ntheta=4,
            Nz=4,
            dealias=1.0,
            time_integrator=integrator,
        ),
        lambda integrator: AxisymmetricMRIDNSJax(
            _keplerian_base(),
            Nr=8,
            Nz=4,
            dealias=1.0,
            time_integrator=integrator,
        ),
        lambda integrator: TaylorCouetteMRIDNSJax(
            _keplerian_base(),
            Nr=8,
            Ntheta=4,
            Nz=4,
            dealias=1.0,
            time_integrator=integrator,
        ),
    ],
    ids=["axisymmetric-hydro", "3d-hydro", "axisymmetric-mhd", "3d-mhd"],
)
def test_tc_integrator_dispatch_preserves_zero_constraints(factory, integrator) -> None:
    solver = factory(integrator)
    state = solver.solve(solver.zero_state(), 3)
    diagnostics = solver.diagnostics(state)

    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(state))
    assert float(diagnostics["E"]) == pytest.approx(0.0, abs=1.0e-15)
    if integrator == "SBDF3":
        assert float(state.history_steps) == 2.0


@pytest.mark.skipif(
    not bool(jax.config.read("jax_enable_x64")),
    reason="third-order TC self-convergence uses float64",
)
@pytest.mark.slow
@pytest.mark.parametrize("integrator", ["SBDF3", "IMEXRK3"])
def test_tc_third_order_integrators_have_third_order_self_convergence(
    integrator,
) -> None:
    final_time = 0.2
    solutions = []
    for steps in (8, 16, 32, 64):
        solver = AxisymmetricTCDNSJax(
            CircularCouette(),
            nu=0.02,
            Nr=8,
            Nz=4,
            dt=final_time / steps,
            dealias=1.0,
            time_integrator=integrator,
        )
        initial, _ = solver.seed_linear_eigenmode(kz_mode=1, amp=1.0e-3)
        out = solver.solve(initial, steps)
        solutions.append(np.concatenate([np.asarray(value).ravel() for value in out.u]))

    errors = [
        np.linalg.norm(solutions[index] - solutions[index + 1]) for index in range(3)
    ]
    orders = [math.log2(errors[index] / errors[index + 1]) for index in range(2)]
    assert all(2.7 <= order <= 3.3 for order in orders), (errors, orders)


def test_tc_rejects_unknown_time_integrator() -> None:
    with pytest.raises(ValueError, match="time_integrator"):
        AxisymmetricTCDNSJax(CircularCouette(), time_integrator="RK-ish")
