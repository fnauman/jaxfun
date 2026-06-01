import jax
import jax.numpy as jnp

from jaxfun.galerkin import Array, Chebyshev, TensorProduct, TestFunction, TrialFunction
from jaxfun.galerkin.arguments import JAXFunction, ScalarFunction, VectorFunction


def test_jaxfunction_doit_and_matmul_rank1():
    C = Chebyshev.Chebyshev(4)
    T = TensorProduct(C, C)
    coeffs = jax.random.normal(jax.random.PRNGKey(10), shape=T.num_dofs)
    jf = JAXFunction(coeffs, T, name="A")
    expr = jf.doit(linear=True)
    # Should produce Jaxc * TrialFunction structure
    assert hasattr(expr, "args")
    assert expr.args[0].__class__.__name__ == "Jaxc"
    a = jnp.ones(T.num_dofs)
    _ = jf @ a
    _ = a @ jf


def test_scalar_vector_function_latex_rank1():
    C = Chebyshev.Chebyshev(3)
    s = ScalarFunction("f", C.system)
    v = VectorFunction("g", C.system)
    _ = s._latex(), v._latex()


def test_trial_test_function_str_repr_symmetry():
    C = Chebyshev.Chebyshev(3)
    v = TestFunction(C)
    u = TrialFunction(C)
    assert str(v) != str(u)  # ensure distinct naming
    assert v.functionspace is C and u.functionspace is C


def test_jaxfunction_no_bold_for_rank0():
    C = Chebyshev.Chebyshev(4)
    coeffs = jax.random.normal(jax.random.PRNGKey(11), shape=(C.N,))
    jf = JAXFunction(coeffs, C)
    assert "mathbf" not in jf._latex()


def test_physical_array_forward_returns_jaxfunction():
    C = Chebyshev.Chebyshev(7)
    T = TensorProduct(C, C)
    coeffs = jnp.zeros(T.num_dofs).at[1, 2].set(0.25).at[3, 1].set(-0.5)

    values = T.backward(coeffs)
    physical = Array(T, values, name="u")
    projected = physical.forward()

    assert isinstance(projected, JAXFunction)
    assert projected.functionspace is T
    assert projected.name == "u"
    assert jnp.allclose(projected.array, coeffs)
    assert jnp.allclose(physical.coefficients(), coeffs)
    assert jnp.allclose(physical.scalar_product(), T.scalar_product(values))
    assert physical.shape == values.shape


def test_physical_array_from_coefficients_and_pytree_roundtrip():
    C = Chebyshev.Chebyshev(6)
    coeffs = jnp.zeros(C.num_dofs).at[2].set(1.0)

    physical = Array.from_coefficients(coeffs, C, name="u")
    leaves, treedef = jax.tree.flatten(physical)
    restored = jax.tree.unflatten(treedef, leaves)

    assert isinstance(restored, type(physical))
    assert restored.functionspace is C
    assert restored.name == "u"
    assert jnp.allclose(restored.backward(), C.backward(coeffs))
    assert jnp.allclose(restored.forward().array, coeffs)
