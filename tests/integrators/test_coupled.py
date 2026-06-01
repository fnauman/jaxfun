import jax.numpy as jnp

from jaxfun.integrators.coupled import ars_stage_rhs


def test_ars_stage_rhs_accumulates_coupled_pytrees():
    a = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.75, 0.25],
        ]
    )
    b = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    base = (jnp.array([1.0, 2.0]), {"g": jnp.array([3.0])})
    nonlinear = [
        (jnp.array([10.0, 20.0]), {"g": jnp.array([30.0])}),
        (jnp.array([2.0, 4.0]), {"g": jnp.array([6.0])}),
    ]
    linear = [(jnp.array([5.0, 6.0]), {"g": jnp.array([7.0])})]

    out = ars_stage_rhs(base, nonlinear, linear, a, b, dt=0.1, rk=1)

    expected0 = base[0] + 0.1 * (0.5 * nonlinear[0][0] + 0.5 * nonlinear[1][0])
    expected0 = expected0 + 0.1 * 0.75 * linear[0][0]
    expected1 = base[1]["g"] + 0.1 * (
        0.5 * nonlinear[0][1]["g"] + 0.5 * nonlinear[1][1]["g"]
    )
    expected1 = expected1 + 0.1 * 0.75 * linear[0][1]["g"]

    assert jnp.allclose(out[0], expected0)
    assert jnp.allclose(out[1]["g"], expected1)
