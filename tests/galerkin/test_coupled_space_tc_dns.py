import jax.numpy as jnp
import sympy as sp

from jaxfun import Domain
from jaxfun.galerkin import (
    CoupledSpace,
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import inner, integrate
from jaxfun.galerkin.Legendre import Legendre


def _tc_spaces(Nr=10, Nz=8):
    dom = Domain(1.0, 2.0)
    F = FunctionSpace(Nz, Fourier, domain=Domain(0.0, 2.0 * float(sp.pi)))
    SD = FunctionSpace(Nr, Legendre, bc=(0, 0), domain=dom)
    SP = FunctionSpace(Nr, Legendre, domain=dom, num_dofs=Nr - 2)
    TD = TensorProduct(F, SD)
    TP = TensorProduct(F, SP)
    return TD, TP, SP


def test_truncated_pressure_space_keeps_quadrature_but_reduces_dofs() -> None:
    Nr = 10
    _TD, _TP, SP = _tc_spaces(Nr=Nr)
    values = jnp.ones(SP.num_quad_points)
    coeffs = SP.forward(values)

    assert Nr == SP.N
    assert SP.num_quad_points == Nr
    assert SP.num_dofs == Nr - 2
    assert coeffs.shape == (Nr - 2,)
    assert SP.mass_matrix().shape == (Nr - 2, Nr - 2)


def test_coupled_space_pack_unpack_for_tc_dns_vq() -> None:
    TD, TP, _SP = _tc_spaces()
    VQ = CoupledSpace((TD, TD, TD, TP), name="VQ")
    coeffs = tuple(jnp.ones(space.num_dofs) * (i + 1) for i, space in enumerate(VQ))

    flat = VQ.flatten(coeffs)
    unpacked = VQ.unflatten(flat)

    assert VQ.num_dofs == (TD.num_dofs, TD.num_dofs, TD.num_dofs, TP.num_dofs)
    assert VQ.block_sizes == (TD.dim, TD.dim, TD.dim, TP.dim)
    assert flat.shape == (4 * TD.dim,)
    assert all(jnp.allclose(a, b) for a, b in zip(coeffs, unpacked, strict=True))


def test_coupled_space_component_transforms_and_integrate() -> None:
    TD, TP, _SP = _tc_spaces()
    VQ = CoupledSpace((TD, TP), name="VP")
    z, r = TD.mesh()
    wall_vanishing = (r - 1.0) * (2.0 - r) + 0.0 * z
    values = (wall_vanishing, 2.0 + 0.0 * z + 0.0 * r)

    coeffs = VQ.forward(values)
    recovered = VQ.backward(coeffs)

    assert tuple(c.shape for c in coeffs) == VQ.num_dofs
    assert all(
        jnp.allclose(v, u, atol=2e-6) for v, u in zip(values, recovered, strict=True)
    )
    expected = integrate(wall_vanishing, TD) + 2.0 * (2.0 * float(sp.pi)) * 1.0
    assert jnp.allclose(integrate(values, VQ), expected, rtol=2e-6, atol=2e-6)


def test_tc_dns_continuity_blocks_are_square_compatible() -> None:
    TD, TP, _SP = _tc_spaces(Nr=10, Nz=8)
    q = TestFunction(TP)
    ur = TrialFunction(TD)
    uz = TrialFunction(TD)
    z, r = TD.system.base_scalars()

    dr = inner(q * ur.diff(r), sparse=True)
    invr = inner(q * (1 / r) * ur, sparse=True)
    dz = inner(q * uz.diff(z), sparse=True)

    assert dr.shape == (TP.dim, TD.dim)
    assert invr.shape == (TP.dim, TD.dim)
    assert dz.shape == (TP.dim, TD.dim)
