import jax.numpy as jnp
import pytest

from jaxfun import Domain
from jaxfun.diagnostics import coefficient_wall_linf
from jaxfun.galerkin import FunctionSpace, TensorProduct
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier


def test_coefficient_wall_linf_transforms_periodic_modes_to_physical_space() -> None:
    periodic = FunctionSpace(8, Fourier, domain=Domain(0.0, 2.0 * jnp.pi))
    radial = FunctionSpace(6, Chebyshev, domain=Domain(1.0, 2.0))
    space = TensorProduct(periodic, radial)
    theta, r = space.mesh()
    values = jnp.broadcast_to(1.0 + jnp.cos(theta), space.num_quad_points)
    coefficients = space.forward(values)

    # The constant and cosine Fourier modes add in phase at theta=0. Taking
    # max(abs(mode)) after only the radial trace would report 1 instead of 2.
    assert float(coefficient_wall_linf((coefficients,), (space,))) == pytest.approx(
        2.0, abs=1.0e-12
    )
