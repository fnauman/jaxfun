# ruff: noqa: E402
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
