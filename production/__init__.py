"""Production contract and golden-comparison helpers for jaxfun."""

from _jaxfun_bootstrap import configure_simplified_jaxpr_constants

# Production modules import JAX through several solver/diagnostic paths. Apply
# the shared environment precedence before any of those imports.
configure_simplified_jaxpr_constants()


from .problem_spec import (  # noqa: E402 - configure environment first
    ProblemSpecError,
    UnsupportedSpecError,
    load_spec,
    spec_hash,
    validate_spec,
)

__all__ = [
    "ProblemSpecError",
    "UnsupportedSpecError",
    "load_spec",
    "spec_hash",
    "validate_spec",
]
