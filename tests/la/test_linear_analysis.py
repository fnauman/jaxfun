import numpy as np
import pytest

from jaxfun.la import (
    finite_eigensystem,
    parse_times,
    physical_eigensystem,
    transient_growth_from_eigs,
)


def test_parse_times_accepts_strings_and_rejects_empty_or_negative():
    assert np.allclose(parse_times("0, 1;2"), [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        parse_times("")
    with pytest.raises(ValueError):
        parse_times([0.0, -1.0])


def test_finite_eigensystem_and_transient_growth_match_reference_helper():
    from couette._linear_analysis import (
        finite_eigensystem as reference_finite_eigensystem,
        transient_growth_from_eigs as reference_transient_growth_from_eigs,
    )

    L = np.array(
        [
            [0.5, 2.0, 0.0],
            [0.0, -0.25, 0.0],
            [0.0, 0.0, 1.0e12],
        ],
        dtype=complex,
    )
    M = np.diag([1.0, 1.0, 0.0]).astype(complex)
    Q = np.diag([2.0, 1.0, 0.0]).astype(complex)
    times = [0.0, 0.5, 1.0]

    w, V = finite_eigensystem(L, M, finite_cap=1.0e8)
    rw, rV = reference_finite_eigensystem(L, M, finite_cap=1.0e8)
    assert np.allclose(w, rw, rtol=0.0, atol=1.0e-12)
    assert np.allclose(np.abs(V), np.abs(rV), rtol=0.0, atol=1.0e-12)

    rows = transient_growth_from_eigs(w, V, Q, times)
    ref_rows = reference_transient_growth_from_eigs(rw, rV, Q, times)
    assert rows == pytest.approx(ref_rows, rel=1.0e-12, abs=1.0e-12)


def test_physical_eigensystem_filters_metric_null_modes_in_both_twins():
    from couette._linear_analysis import (
        physical_eigensystem as reference_physical_eigensystem,
    )

    L = np.diag([1.0e5, 0.5, -0.25]).astype(complex)
    M = np.eye(3, dtype=complex)
    Q = np.diag([0.0, 2.0, 1.0]).astype(complex)

    for solve in (physical_eigensystem, reference_physical_eigensystem):
        values, vectors = solve(L, M, Q, n_return=2)
        assert values == pytest.approx([0.5, -0.25])
        assert vectors.shape == (3, 2)
