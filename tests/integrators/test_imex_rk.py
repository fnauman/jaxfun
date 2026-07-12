import jax.numpy as jnp
import pytest
import sympy as sp

from jaxfun.galerkin.arguments import TestFunction, TrialFunction
from jaxfun.galerkin.Fourier import Fourier as FourierSpace
from jaxfun.galerkin.functionspace import FunctionSpace
from jaxfun.integrators import IMEXRK011, IMEXRK3, IMEXRK111, IMEXRK222, IMEXRK443

pytestmark = pytest.mark.integration


def _linear_problem(cls, lam=-1.25, dt=0.07):
    V = FunctionSpace(8, FourierSpace, name="V", fun_str="E")
    v = TestFunction(V, name="v")
    u = TrialFunction(V, name="u", transient=True)
    (x,) = V.system.base_scalars()
    t = V.system.base_time()
    weak_form = v * (u.diff(t) - sp.Float(lam) * u)
    initial = sp.cos(x) + sp.Rational(1, 2) * sp.sin(2 * x)
    integrator = cls(V, weak_form, time=(0.0, dt), initial=initial, sparse=True)
    return integrator, integrator.initial_coefficients()


def _ars_linear_factor(cls, z):
    a, _, _ = cls.stages()
    gamma = float(a[1, 1]) if cls.steps() else 0.0
    u_stage = 1.0
    linear = []
    for rk in range(cls.steps()):
        rhs = 1.0
        if rk > 0:
            linear.append(u_stage)
            for j in range(rk):
                rhs += float(a[rk + 1, j + 1]) * z * linear[j]
        u_stage = rhs if gamma == 0.0 else rhs / (1.0 - gamma * z)
    return u_stage


def _imexrk3_linear_factor(z):
    a, b, _ = IMEXRK3.stages()
    u_stage = 1.0
    for rk in range(IMEXRK3.steps()):
        gamma = float((a[rk] + b[rk]) / 2.0)
        u_stage = (u_stage + gamma * z * u_stage) / (1.0 - gamma * z)
    return u_stage


@pytest.mark.parametrize("cls", [IMEXRK011, IMEXRK111, IMEXRK222, IMEXRK443])
def test_ars_imex_linear_stage_contract(cls) -> None:
    lam = -1.25
    dt = 0.07
    integrator, u0 = _linear_problem(cls, lam=lam, dt=dt)

    out = integrator.solve(dt=dt, steps=1, progress=False)
    expected = _ars_linear_factor(cls, lam * dt) * u0

    assert jnp.allclose(out, expected, rtol=2e-5, atol=2e-6)


def test_imexrk3_linear_stage_contract() -> None:
    lam = -1.25
    dt = 0.07
    integrator, u0 = _linear_problem(IMEXRK3, lam=lam, dt=dt)

    out = integrator.solve(dt=dt, steps=1, progress=False)
    expected = _imexrk3_linear_factor(lam * dt) * u0

    assert jnp.allclose(out, expected, rtol=2e-5, atol=2e-6)


def test_imexrk222_tableau_values() -> None:
    a, b, c = IMEXRK222.stages()
    gamma = (2.0 - 2.0**0.5) / 2.0
    delta = 1.0 - 1.0 / (2.0 * gamma)

    assert IMEXRK222.steps() == 2
    assert jnp.allclose(c, jnp.asarray((0.0, gamma, 1.0)))
    assert jnp.allclose(a[1:, 1:], jnp.asarray(((gamma, 0.0), (1.0 - gamma, gamma))))
    assert jnp.allclose(b[1:, :2], jnp.asarray(((gamma, 0.0), (delta, 1.0 - delta))))
