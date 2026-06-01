from __future__ import annotations

from typing import overload

import numpy as np
from scipy.linalg import eig as scipy_eig, svdvals

MODAL_FINITE_CAP = 1.0e6
NONMODAL_FINITE_CAP = 1.0e8


def _growth_order(w: np.ndarray) -> np.ndarray:
    """Return deterministic descending-growth order for complex eigenvalues."""
    w = np.asarray(w)
    if w.size == 0:
        return np.asarray([], dtype=int)
    # np.lexsort uses the last key as primary.  This sorts by descending real
    # part, then descending imaginary part for reproducible conjugate-pair order.
    return np.lexsort((-w.imag, -w.real))


def parse_times(value) -> np.ndarray:
    """Parse comma/semicolon-separated non-modal times into a 1D array."""
    if isinstance(value, str):
        times = [
            float(item)
            for item in value.replace(";", ",").split(",")
            if item.strip()
        ]
    else:
        times = list(value)
    if not times:
        raise ValueError("at least one time is required")
    arr = np.asarray(times, dtype=float)
    if np.any(arr < 0):
        raise ValueError("non-modal times must be non-negative")
    return arr


def finite_eigenvalues(
    eigenvalues, *, finite_cap: float = MODAL_FINITE_CAP, sort: bool = True
) -> np.ndarray:
    """Return finite generalized eigenvalues, optionally growth-sorted.

    This mirrors the filtering used by the shenfun Couette references, where
    singular pressure/continuity mass blocks produce eigenvalues at infinity.
    ``MODAL_FINITE_CAP`` intentionally remains tighter than the non-modal cap
    used by :func:`finite_eigensystem`; see ``COUETTE_IMPLEMENTATION_PLAN.md``
    M8/T8.0.
    """
    w = np.asarray(eigenvalues)
    mask = np.isfinite(w) & (np.abs(w) < finite_cap)
    w = w[mask]
    if sort and len(w):
        w = w[_growth_order(w)]
    return w


def finite_eigensystem(
    L, M, *, finite_cap: float = NONMODAL_FINITE_CAP, n_return: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite generalized eigenpairs sorted by descending growth.

    Reference: ``couette/_linear_analysis.py:15-87``.  This helper keeps the
    larger ``1e8`` non-modal cap distinct from the default modal eigenvalue cap
    because transient-growth bases need the strongly damped finite modes that
    the quick modal listing may discard.
    """
    w, V = scipy_eig(np.asarray(L), np.asarray(M))
    mask = np.isfinite(w) & (np.abs(w) < finite_cap)
    w = w[mask]
    V = V[:, mask]
    order = _growth_order(w)
    w = w[order]
    V = V[:, order]
    if n_return is not None:
        w = w[:n_return]
        V = V[:, :n_return]
    return w, V


def transient_growth_from_eigs(
    evals, evecs, metric, times, *, metric_rtol: float = 1.0e-10
) -> list[dict[str, float]]:
    r"""Return optimal transient growth in the metric-induced energy norm.

    The modal state is ``q(t) = V exp(Lambda t) a``.  Gauge/null directions
    that have negligible metric norm are removed before taking the largest
    singular value of the metric-weighted propagator.  Reference:
    ``couette/_linear_analysis.py:41-87``.
    """
    evals = np.asarray(evals, dtype=complex)
    V = np.asarray(evecs, dtype=complex)
    Q = np.asarray(metric, dtype=complex)
    times = parse_times(times)
    if evals.size == 0 or V.size == 0:
        raise ValueError("no finite eigenvectors available for non-modal growth")

    Qm = V.conj().T @ Q @ V
    Qm = 0.5 * (Qm + Qm.conj().T)
    d, U = np.linalg.eigh(Qm)
    scale = float(np.max(np.abs(d))) if d.size else 0.0
    keep = d > max(metric_rtol * scale, np.finfo(float).eps)
    if not np.any(keep):
        raise ValueError("energy metric is singular on all retained modes")

    d = d[keep]
    U = U[:, keep]
    sqrt_d = np.sqrt(d)
    inv_sqrt_d = 1.0 / sqrt_d
    left = sqrt_d[:, None] * U.conj().T

    out: list[dict[str, float]] = []
    for t in times:
        exp_diag = np.exp(evals * float(t))
        middle = exp_diag[:, None] * U
        prop = (left @ middle) * inv_sqrt_d[None, :]
        gain = float(svdvals(prop)[0] ** 2)
        out.append(
            {
                "t": float(t),
                "gain": gain,
                "amplification": float(np.sqrt(gain)),
            }
        )
    return out


def print_eigenvalues(evals, header: str = "leading eigenvalues") -> None:
    print(header)
    for value in evals:
        print(f"   s = {value.real:+.6e}  {value.imag:+.6e} i")


def print_transient_growth(
    rows: list[dict[str, float]], header: str = "non-modal transient growth"
) -> None:
    print(header)
    for row in rows:
        print(
            f"   t={row['t']:.6g}  G={row['gain']:.6e}  "
            f"sqrt(G)={row['amplification']:.6e}"
        )


@overload
def generalized_eig(
    L,
    M,
    *,
    vectors: bool = False,
    finite_cap: float = MODAL_FINITE_CAP,
    sort: bool = True,
) -> np.ndarray: ...


@overload
def generalized_eig(
    L,
    M,
    *,
    vectors: bool,
    finite_cap: float = MODAL_FINITE_CAP,
    sort: bool = True,
) -> tuple[np.ndarray, np.ndarray]: ...


def generalized_eig(
    L,
    M,
    *,
    vectors: bool = False,
    finite_cap: float = MODAL_FINITE_CAP,
    sort: bool = True,
):
    """Solve and filter ``L q = lambda M q`` as dense host arrays.

    Reference: ``couette/taylor_couette_linear.py:239-252`` and
    ``couette/taylor_couette_mri.py``.  Sorting is deterministic even when
    conjugate pairs share the same growth rate.
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
        order = _growth_order(w)
        w = w[order]
        V = V[:, order]
    return w, V
