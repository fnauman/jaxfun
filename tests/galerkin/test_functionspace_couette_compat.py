import jax.numpy as jnp

from jaxfun.coordinates import CartCoordSys, x
from jaxfun.galerkin import FunctionSpace, InnerKind, TestFunction, TrialFunction, inner
from jaxfun.galerkin.Legendre import Legendre


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
