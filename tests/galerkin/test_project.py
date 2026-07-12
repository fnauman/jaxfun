import importlib

import jax.numpy as jnp
import pytest

from jaxfun import Domain, Dx
from jaxfun.galerkin import FunctionSpace, Project, TensorProduct, TrialFunction
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.Legendre import Legendre


def test_cached_project_reproduces_shenfun_derivative_doc_example(monkeypatch):
    T = FunctionSpace(8, Chebyshev, domain=Domain(-1.0, 1.0))
    u = TrialFunction(T, name="u")
    projector = Project(Dx(u, 0, 1), T)
    coeffs = jnp.zeros(T.num_dofs).at[1].set(1.0)

    inner_mod = importlib.import_module("jaxfun.galerkin.inner")

    def fail_inner(*_args, **_kwargs):
        raise AssertionError("Project.__call__ must not reassemble forms")

    monkeypatch.setattr(inner_mod, "inner", fail_inner)
    out = projector(coeffs)

    expected = jnp.zeros(T.num_dofs).at[0].set(1.0)
    assert jnp.allclose(out, expected, atol=1.0e-12)


def test_cached_project_cross_space_derivative_matches_forward_projection():
    B = FunctionSpace(8, Legendre, bc=(0, 0, 0, 0), domain=Domain(-1.0, 1.0))
    C = FunctionSpace(8, Legendre, domain=Domain(-1.0, 1.0))
    F1 = FunctionSpace(6, Fourier, domain=Domain(0.0, 2.0 * jnp.pi))
    F2 = FunctionSpace(6, Fourier, domain=Domain(0.0, 2.0 * jnp.pi))
    TB = TensorProduct(B, F1, F2, name="TB")
    TC = TensorProduct(C, F1, F2, name="TC")
    u = TrialFunction(TB, name="u_b")
    projector = Project(Dx(u, 0, 1), TC)
    coeffs = jnp.zeros(TB.num_dofs, dtype=complex)
    coeffs = coeffs.at[1, 1, 0].set(0.25 + 0.5j)
    coeffs = coeffs.at[2, 0, 1].set(-0.125j)

    out = projector(coeffs)
    expected = TC.forward(TB.backward_primitive(coeffs, (1, 0, 0)))

    assert out.shape == TC.num_dofs
    assert jnp.allclose(out, expected, atol=1.0e-11, rtol=1.0e-11)


def test_cached_project_rejects_nonlinear_expression():
    T = FunctionSpace(6, Legendre)
    u = TrialFunction(T, name="u")
    with pytest.raises(ValueError):
        Project(u * u, T)
