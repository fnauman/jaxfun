"""Named-axis dealiasing contract (FJ-01).

Production specs express dealiasing padding *semantically* as a per-canonical-axis
map ``{"x": 1.0, "y": 1.5, "z": 1.5}`` (or a single scalar for uniform padding).
Each solver stores its field arrays in a fixed *native* axis order that differs
between formulations:

* ``PCFMRIDNSJax`` (primitive PCF-MRI/MHD DNS): native ``(y, z, x)`` -- streamwise
  Fourier, spanwise/vertical Fourier, wall-normal Chebyshev;
* the KMM velocity family (``PlaneCouetteFluctuationJax``) and the
  vector-potential MRI family (``PlaneCouetteMRIShearpyJax``): native
  ``(x, y, z)`` -- wall-normal Chebyshev, streamwise Fourier, spanwise Fourier.

Passing a *positional* padding tuple to the wrong native order silently leaves a
periodic direction undealiased (the FJ-01 defect: ``[1.0, 1.5, 1.5]`` handed to
the ``(y, z, x)`` primitive solver leaves the streamwise ``y`` nonlinearity
undealiased). :func:`to_native_padding` converts a semantic map to the correct
native tuple for a named solver family, so the *same* spec produces the right
per-axis padding regardless of the solver's array layout.

The bump-able :data:`NUMERICS_CONTRACT_VERSION` marks specs/checkpoints/goldens
created after this fix; artifacts without it (or at a lower version) are pre-FJ-01
and must not seed post-fix production continuations.
"""

from __future__ import annotations

import warnings
from typing import Any

from .problem_spec import ProblemSpecError

CANONICAL_AXES: tuple[str, ...] = ("x", "y", "z")

# Native array axis order (canonical-axis letters) per solver family. This is the
# authoritative contract used to remap semantic padding; it must match each JAX
# solver's actual ``TensorProduct`` / ``KMM`` axis layout.
SOLVER_NATIVE_AXES: dict[str, tuple[str, ...]] = {
    # examples/pcf_mri_primitive_jax.py: TensorProduct(Fy, Fz, SD) -> (y, z, x)
    "pcf_primitive": ("y", "z", "x"),
    # examples/pcf_fluctuations_jax.py (KMM base): (x wall-normal, y, z)
    "pcf_kmm": ("x", "y", "z"),
    # examples/pcf_mhd_mri_shearpy_jax.py (KMM/vector-potential base): (x, y, z)
    "pcf_vector_potential": ("x", "y", "z"),
    # 2-D axisymmetric primitive (ky=0): TensorProduct(Fz, SD) -> (z, x)
    "pcf_primitive_axisymmetric": ("z", "x"),
}


def native_axis_order(solver_family: str, *, dimensions: int | None = None) -> tuple[str, ...]:
    """Return the native canonical-axis-letter order for ``solver_family``."""

    try:
        order = SOLVER_NATIVE_AXES[solver_family]
    except KeyError as exc:  # pragma: no cover - programming error
        raise ProblemSpecError(
            f"unknown solver family {solver_family!r}; "
            f"known: {sorted(SOLVER_NATIVE_AXES)}"
        ) from exc
    if dimensions is not None and len(order) != dimensions:
        raise ProblemSpecError(
            f"solver family {solver_family!r} is {len(order)}-D, "
            f"asked for {dimensions}-D padding"
        )
    return order


def to_native_padding(
    dealias: Any,
    native_order: tuple[str, ...],
    *,
    context: str = "resolution.dealias",
) -> tuple[float, ...]:
    """Convert a semantic/scalar/legacy dealias to a native-order padding tuple.

    ``dealias`` may be:

    * a mapping ``{"x": .., "y": .., "z": ..}`` -- remapped to ``native_order``
      (order-invariant: permuting the dict keys cannot change the result);
    * a scalar -- uniform padding on every axis;
    * a positional ``list``/``tuple`` (legacy) -- interpreted in CANONICAL
      ``(x, y, z)`` order and remapped to ``native_order`` with a
      :class:`DeprecationWarning`. Interpreting legacy tuples as canonical order
      matches the evident author intent (specs stated padding as x, y, z) and so
      *corrects* an un-migrated spec instead of silently mis-dealiasing it.
    """

    n = len(native_order)
    if isinstance(dealias, dict):
        missing = [axis for axis in native_order if axis not in dealias]
        if missing:
            raise ProblemSpecError(
                f"{context}: semantic padding is missing axis/axes {missing}; "
                f"native order is {native_order}"
            )
        return tuple(_finite_padding(dealias[axis], f"{context}.{axis}") for axis in native_order)
    if isinstance(dealias, (list, tuple)):
        values = tuple(_finite_padding(item, context) for item in dealias)
        if len(values) != n:
            raise ProblemSpecError(
                f"{context}: expected {n} padding values, got {len(values)}"
            )
        if len(set(values)) > 1:
            warnings.warn(
                f"{context}: positional padding tuple {values} is deprecated; "
                "use a semantic {'x': .., 'y': .., 'z': ..} map. Interpreting the "
                f"tuple in canonical {CANONICAL_AXES[:n]} order.",
                DeprecationWarning,
                stacklevel=2,
            )
            canonical = dict(zip(CANONICAL_AXES[:n], values, strict=True))
            return tuple(canonical[axis] for axis in native_order)
        # Uniform padding is order-invariant; pass it through verbatim.
        return values
    return (_finite_padding(dealias, context),) * n


def native_padding_for_solver(
    resolution: dict[str, Any],
    *,
    solver_family: str,
    dimensions: int | None = None,
) -> tuple[float, ...]:
    """Return the native padding tuple for ``solver_family`` from a resolution block."""

    native = native_axis_order(solver_family, dimensions=dimensions)
    dealias = resolution.get("dealias", 1.0)
    return to_native_padding(
        dealias, native, context=f"resolution.dealias[{solver_family}]"
    )


def is_semantic_or_scalar(dealias: Any) -> bool:
    """True when ``dealias`` is a semantic map or a scalar (contract-v2 forms)."""

    if isinstance(dealias, dict):
        return True
    return not isinstance(dealias, (list, tuple))


def _finite_padding(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ProblemSpecError(f"{label} padding must be numeric") from exc
    if not (out >= 1.0):
        raise ProblemSpecError(f"{label} padding must be >= 1.0, got {out}")
    return out
