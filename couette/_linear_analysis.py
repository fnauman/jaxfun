"""Shared dense linear-analysis helpers for demo eigen/non-modal variants."""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig, svdvals

# Single shared magnitude cap for the *non-modal* (transient-growth) modal
# basis.  Generalised eigenproblems ``L q = s M q`` with singular ``M`` (the
# incompressibility / div(b)=0 constraint rows) place spurious eigenvalues at
# infinity; these must be dropped before the modal expansion.  The cap has to
# sit *above* the most strongly damped physical mode (whose decay scales like
# ``nu * N**4`` for an N-point spectral grid) yet below the spurious branch.
# ``1e8`` clears both for every demo here; verified insensitive between 1e6 and
# 1e12 on the Taylor-Couette transient-growth benchmark.
FINITE_CAP = 1.0e8


def parse_times(value):
    """Parse comma-separated times, or return a 1D float array unchanged."""
    if isinstance(value, str):
        times = [float(item) for item in value.replace(";", ",").split(",") if item.strip()]
    else:
        times = list(value)
    if not times:
        raise ValueError("at least one time is required")
    arr = np.asarray(times, dtype=float)
    if np.any(arr < 0):
        raise ValueError("non-modal times must be non-negative")
    return arr


def finite_eigensystem(L, M, finite_cap=FINITE_CAP, n_return=None):
    """Return finite generalized eigenpairs sorted by descending growth."""
    w, V = eig(L, M)
    good = np.isfinite(w) & (np.abs(w) < finite_cap)
    w, V = w[good], V[:, good]
    order = np.argsort(-w.real)
    w, V = w[order], V[:, order]
    if n_return is not None:
        w, V = w[:n_return], V[:, :n_return]
    return w, V


def transient_growth_from_eigs(evals, evecs, metric, times, metric_rtol=1.0e-10):
    r"""Energy-norm transient growth from a finite modal expansion.

    The state is expanded as ``q(t) = V exp(Lambda t) a``.  For the Hermitian
    positive semi-definite state metric ``Q``, the modal metric is
    ``Qm = V^H Q V``.  Gauge/null directions with negligible ``Qm`` norm are
    removed before computing the largest singular value of the propagator in the
    induced energy norm.
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

    left = sqrt_d[:, None] * U.conj().T          # fixed across times; build once
    out = []
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


def print_eigenvalues(evals, header="leading eigenvalues"):
    print(header)
    for s in evals:
        print(f"   s = {s.real:+.6e}  {s.imag:+.6e} i")


def print_transient_growth(rows, header="non-modal transient growth"):
    print(header)
    for row in rows:
        print(
            f"   t={row['t']:.6g}  G={row['gain']:.6e}  "
            f"sqrt(G)={row['amplification']:.6e}"
        )
