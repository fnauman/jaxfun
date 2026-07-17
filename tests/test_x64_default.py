import jax
import jax.numpy as jnp

import jaxfun


def test_jaxfun_import_enables_x64_by_default():
    assert jax.config.read("jax_enable_x64") is True
    assert jax.config.jax_use_simplified_jaxpr_constants is True
    assert jnp.zeros(1).dtype == jnp.float64
    assert jaxfun.galerkin is not None
