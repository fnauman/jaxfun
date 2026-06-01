from __future__ import annotations

from typing import overload

import numpy as np
from scipy.linalg import eig as scipy_eig


def finite_eigenvalues(
    eigenvalues, *, finite_cap: float = 1.0e6, sort: bool = True
) -> np.ndarray:
    """Return finite generalized eigenvalues, optionally growth-sorted.

    This mirrors the filtering used by the shenfun Couette references, where
    the singular pressure/continuity mass block produces eigenvalues at
    infinity.  See couette/taylor_couette_linear.py:239-252.
    """
    w = np.asarray(eigenvalues)
    mask = np.isfinite(w) & (np.abs(w) < finite_cap)
    w = w[mask]
    if sort and len(w):
        w = w[np.argsort(-w.real)]
    return w


@overload
def generalized_eig(
    L,
    M,
    *,
    vectors: bool = False,
    finite_cap: float = 1.0e6,
    sort: bool = True,
) -> np.ndarray: ...


@overload
def generalized_eig(
    L,
    M,
    *,
    vectors: bool,
    finite_cap: float = 1.0e6,
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...


def generalized_eig(
    L,
    M,
    *,
    vectors: bool = False,
    finite_cap: float = 1.0e6,
    sort: bool = True,
):
    """Solve and filter L q = lambda M q as dense host arrays.

    Reference: couette/taylor_couette_linear.py:239-252 and
    couette/taylor_couette_mri.py use scipy.linalg.eig followed by
    finite-eigenvalue filtering for singular generalized mass matrices.
    """
    L_np = np.asarray(L)
    M_np = np.asarray(M)
    if not vectors:
        w = scipy_eig(L_np, M_np, right=False)
        return finite_eigenvalues(w, finite_cap=finite_cap, sort=sort)

    w, V = scipy_eig(L_np, M_np)
    mask = np.isfinite(w) & (np.abs(w) < finite_cap)
    w = w[mask]
    V = V[:, mask]
    if sort and len(w):
        order = np.argsort(-w.real)
        w = w[order]
        V = V[:, order]
    return w, V
