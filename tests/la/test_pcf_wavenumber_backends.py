"""GPU qualification for the actual PCF implicit operator shapes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from examples.pcf_mhd_jax import PlaneCouetteMHDJax

pytestmark = pytest.mark.integration


def _solver(family: str, monkeypatch, backend: str, resolution=(17, 8, 8)):
    monkeypatch.setenv("JAXFUN_WAVENUMBER_SOLVER", backend)
    return PlaneCouetteMHDJax(
        N=resolution,
        family=family,
        Re=200.0,
        Rm=200.0,
        dt=1.0e-3,
        time_integrator="CNAB2",
        padding_factor=(1.0, 1.0, 1.0),
        perturbation_amplitude=0.0,
        magnetic_amplitude=0.0,
    )


def _rhs(shape, n_rhs=3):
    index = jnp.arange(int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64).reshape(
        shape
    )
    base = (jnp.sin(0.17 * index) + 1j * jnp.cos(0.11 * index)) * 1.0e-3
    return jnp.stack(
        [(1.0 + 0.1 * lane + 0.03j * lane) * base for lane in range(n_rhs)]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["C", "L"], ids=["chebyshev", "legendre"])
def test_actual_pcf_operators_pallas_match_jax_multi_rhs(family, monkeypatch) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("pallas-triton requires a GPU")
    reference = _solver(family, monkeypatch, "jax")
    pallas = _solver(family, monkeypatch, "pallas-triton")

    for name in ("Su", "Sg", "SA"):
        operator = getattr(reference, name)
        reference_factor = getattr(reference, f"{name}_factor")
        pallas_factor = getattr(pallas, f"{name}_factor")
        rhs = _rhs(reference_factor.shape)
        expected = reference_factor.solve_many(rhs)
        actual = pallas_factor.solve_many(rhs)
        assert jnp.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12), name
        residual = jax.vmap(lambda x, b, op=operator: op @ x - b)(actual, rhs)
        relative = jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)
        assert float(relative) < 2.0e-11, (name, float(relative))

    # S00 is a radial-only factor rather than a wavenumber solver, but its two
    # production mean-flow right-hand sides are qualified in the same test.
    means_rhs = jnp.real(_rhs((reference.S00.shape[0],), n_rhs=2))
    means = jax.vmap(reference.S00_factor.solve)(means_rhs)
    mean_residual = jax.vmap(lambda x, b: reference.S00 @ x - b)(means, means_rhs)
    assert float(jnp.linalg.norm(mean_residual) / jnp.linalg.norm(means_rhs)) < 2.0e-12


@pytest.mark.gpu
@pytest.mark.slow
def test_pallas_sg_large_bounded_pcf_residual(monkeypatch) -> None:
    """Largest local qualification case stays within the 128^3 cap."""

    if jax.default_backend() != "gpu":
        pytest.skip("pallas-triton requires a GPU")
    solver = _solver("C", monkeypatch, "pallas-triton", resolution=(65, 64, 64))
    rhs = _rhs(solver.Sg_factor.shape)
    solution = solver.Sg_factor.solve_many(rhs)
    residual = jax.vmap(lambda x, b: solver.Sg @ x - b)(solution, rhs)
    relative = jnp.linalg.norm(residual) / jnp.linalg.norm(rhs)
    assert float(relative) < 2.0e-10
