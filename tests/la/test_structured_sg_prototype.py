import jax.numpy as jnp
import pytest

from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
from production.structured_sg_prototype import qualify_sg_prototype


@pytest.mark.integration
@pytest.mark.parametrize("family", ["C", "L"])
def test_compact_sg_prototype_matches_production_and_reduces_factor_storage(family):
    solver = PlaneCouetteFluctuationJax(
        N=(17, 8, 8),
        family=family,
        dt=1.0e-3,
        padding_factor=(1.0, 1.0, 1.0),
    )
    index = jnp.arange(int(jnp.prod(jnp.asarray(solver.Sg_factor.shape)))).reshape(
        solver.Sg_factor.shape
    )
    rhs = (jnp.sin(0.13 * index) + 1j * jnp.cos(0.07 * index)) * 1.0e-3
    report = qualify_sg_prototype(solver, rhs)

    assert report.factor_to_compact_ratio > 1.5
    assert report.relative_residual < 2.0e-11
    assert report.relative_solution_error < 2.0e-11
