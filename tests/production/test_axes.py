"""FJ-01: named-axis dealiasing contract tests."""

from __future__ import annotations

import warnings

import pytest

from production.axes import (
    SOLVER_NATIVE_AXES,
    native_padding_for_solver,
    to_native_padding,
)
from production.problem_spec import ProblemSpecError


def test_semantic_map_permutation_invariant():
    """FJ-01 acceptance: permuting the semantic input order cannot change padding."""
    native = SOLVER_NATIVE_AXES["pcf_primitive"]  # (y, z, x)
    a = to_native_padding({"x": 1.0, "y": 1.5, "z": 1.5}, native)
    b = to_native_padding({"z": 1.5, "x": 1.0, "y": 1.5}, native)
    assert a == b == (1.5, 1.5, 1.0)


def test_same_semantic_maps_to_each_solver_native_order():
    """One semantic spec -> correct per-solver native tuple."""
    semantic = {"x": 1.0, "y": 1.5, "z": 1.5}
    # primitive (y, z, x): streamwise y and spanwise z padded, wall-normal x not.
    assert to_native_padding(semantic, SOLVER_NATIVE_AXES["pcf_primitive"]) == (
        1.5,
        1.5,
        1.0,
    )
    # KMM (x, y, z): wall-normal x not padded, periodic y, z padded.
    assert to_native_padding(semantic, SOLVER_NATIVE_AXES["pcf_kmm"]) == (
        1.0,
        1.5,
        1.5,
    )


def test_legacy_positional_tuple_is_corrected_and_warns():
    """FJ-01: a legacy positional [1.0, 1.5, 1.5] is read as canonical (x, y, z)
    and remapped, so the primitive (y, z, x) solver is correctly dealiased."""
    native = SOLVER_NATIVE_AXES["pcf_primitive"]
    with pytest.warns(DeprecationWarning):
        out = to_native_padding([1.0, 1.5, 1.5], native)
    assert out == (1.5, 1.5, 1.0)


def test_legacy_uniform_tuple_is_order_invariant_and_silent():
    native = SOLVER_NATIVE_AXES["pcf_primitive"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # would raise if a warning fired
        out = to_native_padding([1.0, 1.0, 1.0], native)
    assert out == (1.0, 1.0, 1.0)


def test_scalar_is_uniform():
    assert to_native_padding(1.5, SOLVER_NATIVE_AXES["pcf_primitive"]) == (
        1.5,
        1.5,
        1.5,
    )
    assert to_native_padding(1.0, SOLVER_NATIVE_AXES["pcf_kmm"]) == (1.0, 1.0, 1.0)


def test_missing_axis_in_semantic_map_raises():
    with pytest.raises(ProblemSpecError):
        to_native_padding({"x": 1.0, "y": 1.5}, SOLVER_NATIVE_AXES["pcf_primitive"])


def test_padding_below_one_rejected():
    with pytest.raises(ProblemSpecError):
        to_native_padding(
            {"x": 1.0, "y": 0.5, "z": 1.5}, SOLVER_NATIVE_AXES["pcf_primitive"]
        )


def test_native_padding_for_solver_reads_resolution():
    resolution = {"dealias": {"x": 1.0, "y": 1.5, "z": 1.5}}
    assert native_padding_for_solver(resolution, solver_family="pcf_primitive") == (
        1.5,
        1.5,
        1.0,
    )
    assert native_padding_for_solver(resolution, solver_family="pcf_kmm") == (
        1.0,
        1.5,
        1.5,
    )
