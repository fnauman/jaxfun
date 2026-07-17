"""Production contract and golden-comparison helpers for jaxfun."""

import sys
from pathlib import Path

try:
    from _jaxfun_bootstrap import configure_simplified_jaxpr_constants
except ModuleNotFoundError as exc:  # pragma: no cover - clean-checkout fallback
    if exc.name != "_jaxfun_bootstrap":
        raise
    # ``python -m production.<entrypoint>`` imports this package before the
    # entrypoint can install its direct-script fallback. Make the sibling src
    # tree visible without importing JAX or mutating its environment first.
    _src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(_src))
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
