"""Environment bootstrap shared by jaxfun and production entry points."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def configure_simplified_jaxpr_constants(
    environ: MutableMapping[str, str] | None = None,
) -> str:
    """Apply jaxfun precedence before JAX is imported and return the value."""

    environment = os.environ if environ is None else environ
    configured = environment.get("JAXFUN_USE_SIMPLIFIED_JAXPR_CONSTANTS")
    if configured is not None:
        environment["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = configured
    else:
        environment.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "true")
    return environment["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"]
