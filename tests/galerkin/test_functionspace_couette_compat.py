import jax.numpy as jnp
import numpy as np
import pytest

from jaxfun.coordinates import CartCoordSys, x
from jaxfun.galerkin import FunctionSpace, InnerKind, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Legendre import Legendre
from tests._parity import shenfun_basis_stencils


def test_shenfun_tuple_bc_alias_matches_dict_bcs():
    tuple_space = FunctionSpace(8, Legendre, bc=(0, 0))
    dict_space = FunctionSpace(8, Legendre, bcs={"left": {"D": 0}, "right": {"D": 0}})

    assert tuple_space.dim == dict_space.dim == 6
    assert tuple_space.bcs == dict_space.bcs
    assert jnp.allclose(
        tuple_space.mass_matrix().todense(), dict_space.mass_matrix().todense()
    )


def test_shenfun_clamped_tuple_bc_alias():
    space = FunctionSpace(8, Legendre, bc=(0, 0, 0, 0))

    assert space.dim == 4
    assert space.bcs == {
        "left": {"D": 0, "N": 0},
        "right": {"D": 0, "N": 0},
    }


def test_unconstrained_functionspace_preserves_system_for_cross_space_forms():
    system = CartCoordSys("R", (x,))
    velocity = FunctionSpace(8, Legendre, bc=(0, 0), domain=(1, 2), system=system)
    pressure = FunctionSpace(6, Legendre, domain=(1, 2), system=system)

    assert pressure.system is system
    assert pressure.system is velocity.system

    q = TestFunction(pressure)
    u = TrialFunction(velocity)
    r = system.base_scalars()[0]
    A = inner(q * (u / r), kind=InnerKind.BILINEAR, num_quad_points=8)

    assert A.shape == (6, 6)
    assert jnp.linalg.norm(A.todense()) > 0


def test_homogeneous_robin_bc_preserves_robin_coefficient():
    space = FunctionSpace(
        8,
        Legendre,
        bc={
            "left": {"R": (2.0, 0)},
            "right": {"R": (4.0, 0)},
        },
    )

    assert space.dim == 6
    assert space.bcs == {
        "left": {"R": (2.0, 0)},
        "right": {"R": (4.0, 0)},
    }


@pytest.mark.integration
def test_tuple_bc_stencils_match_live_shenfun_dirichlet_and_biharmonic():
    references = shenfun_basis_stencils(n=8)
    cases = (
        ("L_2", Legendre, (0, 0), "ShenDirichlet"),
        ("L_4", Legendre, (0, 0, 0, 0), "ShenBiharmonic"),
        ("C_2", Chebyshev, (0, 0), "ShenDirichlet"),
        ("C_4", Chebyshev, (0, 0, 0, 0), "ShenBiharmonic"),
    )

    for key, family, bc, reference_type in cases:
        space = FunctionSpace(8, family, bc=bc)
        reference = references[key]
        expected = np.asarray(reference["stencil"], dtype=float)[: space.dim, :]
        actual = np.asarray(space.S.todense())

        assert reference["type"] == reference_type
        assert reference["dim"] == space.dim
        assert actual.shape == expected.shape
        assert np.allclose(actual, expected, rtol=0.0, atol=1.0e-13)
