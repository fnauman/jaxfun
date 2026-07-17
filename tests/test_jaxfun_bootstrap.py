from _jaxfun_bootstrap import configure_simplified_jaxpr_constants


def test_simplified_constants_bootstrap_defaults_to_true():
    environment = {}

    result = configure_simplified_jaxpr_constants(environment)

    assert result == "true"
    assert environment == {"JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "true"}


def test_simplified_constants_bootstrap_honors_standard_jax_setting():
    environment = {"JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "false"}

    result = configure_simplified_jaxpr_constants(environment)

    assert result == "false"
    assert environment["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] == "false"


def test_jaxfun_setting_overrides_standard_jax_setting():
    environment = {
        "JAXFUN_USE_SIMPLIFIED_JAXPR_CONSTANTS": "1",
        "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "0",
    }

    result = configure_simplified_jaxpr_constants(environment)

    assert result == "1"
    assert environment["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] == "1"
