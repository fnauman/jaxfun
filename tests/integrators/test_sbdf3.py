import jax
import jax.numpy as jnp
import numpy as np

from jaxfun.integrators.sbdf3 import IMPLICIT_SCALE, sbdf3_rhs


def test_sbdf3_rhs_supports_matching_pytrees() -> None:
    mass = {
        "u": jnp.asarray([3.0, 6.0]),
        "nested": (jnp.asarray([2.0]),),
    }
    previous = jax.tree.map(lambda value: 0.5 * value, mass)
    older = jax.tree.map(lambda value: 0.25 * value, mass)
    nonlinear = jax.tree.map(lambda value: 0.1 * value, mass)
    nonlinear_previous = jax.tree.map(lambda value: 0.5 * value, nonlinear)
    nonlinear_older = jax.tree.map(lambda value: 0.25 * value, nonlinear)

    actual = sbdf3_rhs(
        mass,
        previous,
        older,
        nonlinear,
        nonlinear_previous,
        nonlinear_older,
        0.2,
    )
    expected = jax.tree.map(
        lambda m, n: (18.0 * m - 9.0 * 0.5 * m + 2.0 * 0.25 * m) / 11.0
        + 0.2 * (18.0 * n - 18.0 * 0.5 * n + 6.0 * 0.25 * n) / 11.0,
        mass,
        nonlinear,
    )
    assert all(
        bool(jnp.allclose(a, b))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)
    )


def _integrate_split_linear(dt: float, final_time: float = 1.0) -> float:
    implicit_rate = -1.25
    explicit_rate = 0.4
    total_rate = implicit_rate + explicit_rate
    steps = int(round(final_time / dt))
    values = [jnp.exp(total_rate * (index * dt)) for index in range(3)]
    nonlinear = [explicit_rate * value for value in values]
    denominator = 1.0 - IMPLICIT_SCALE * dt * implicit_rate
    for _ in range(2, steps):
        rhs = sbdf3_rhs(
            values[-1],
            values[-2],
            values[-3],
            nonlinear[-1],
            nonlinear[-2],
            nonlinear[-3],
            dt,
        )
        value = rhs / denominator
        values.append(value)
        nonlinear.append(explicit_rate * value)
    return abs(float(values[-1]) - float(np.exp(total_rate * final_time)))


def test_sbdf3_split_linear_equation_has_third_order_convergence() -> None:
    dts = np.asarray([1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0, 1.0 / 160.0])
    errors = np.asarray([_integrate_split_linear(float(dt)) for dt in dts])
    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    assert np.all((orders[-2:] > 2.7) & (orders[-2:] < 3.3)), (errors, orders)
