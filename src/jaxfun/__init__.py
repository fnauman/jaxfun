# ruff: noqa: E402
import os

from _jaxfun_bootstrap import configure_simplified_jaxpr_constants

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
configure_simplified_jaxpr_constants()


import jax


def _env_truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


jax.config.update(
    "jax_enable_x64",
    _env_truthy(
        os.environ.get("JAXFUN_ENABLE_X64", os.environ.get("JAX_ENABLE_X64")),
        default=True,
    ),
)

# Hoist closed-over array constants to executable arguments. Wall-bounded
# solvers capture timestep-dependent banded factors; legacy lowering embeds
# and constant-folds those arrays into multi-gigabyte executables. The JAX
# setting preserves numerical semantics while keeping compilation/cache size
# proportional to the program rather than the factor values. The environment
# is set above before importing JAX because this lowering is initialized early.
if hasattr(jax.config, "jax_use_simplified_jaxpr_constants"):
    jax.config.update(
        "jax_use_simplified_jaxpr_constants",
        _env_truthy(
            os.environ.get(
                "JAXFUN_USE_SIMPLIFIED_JAXPR_CONSTANTS",
                os.environ.get("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"),
            ),
            default=True,
        ),
    )


from . import galerkin as galerkin, integrators as integrators, pinns as pinns
from .basespace import BaseSpace as BaseSpace
from .coordinates import CoordSys as CoordSys, get_CoordSys as get_CoordSys
from .operators import (
    Cross as Cross,
    Curl as Curl,
    Div as Div,
    Dot as Dot,
    Dx as Dx,
    Grad as Grad,
    Outer as Outer,
    cross as cross,
    curl as curl,
    divergence as divergence,
    dot as dot,
    gradient as gradient,
    outer as outer,
)
from .utils import (
    Domain as Domain,
    common as common,
    fastgl as fastgl,
    lambdify as lambdify,
)
