"""Shared dense linear-analysis helpers for demo eigen/non-modal variants."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig, svdvals
from scipy.optimize import linear_sum_assignment

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
        times = [
            float(item) for item in value.replace(";", ",").split(",") if item.strip()
        ]
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

    left = sqrt_d[:, None] * U.conj().T  # fixed across times; build once
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


def match_eigenvalues(reference, candidates):
    r"""Nearest-neighbour match each reference eigenvalue to a candidate set.

    The plane-Couette and Taylor-Couette spectra must be compared as *sets*, not
    by their single "leading" eigenvalue.  A (near-)conjugate-symmetric spectrum
    has growth rates that are degenerate between the ``+`` and ``-`` frequency
    members of a conjugate pair, so which one sorts first is an arbitrary
    numerical tie-break.  Plane Couette at small ``ky`` is essentially
    conjugate-symmetric while the Taylor-Couette ``m`` term lifts the degeneracy,
    so the two "leading" picks can disagree in the *sign* of the frequency even
    though the full spectra agree.  Matching each reference eigenvalue to its
    nearest neighbour in the complex plane is tie-break- and orientation-robust
    and yields a genuine full-complex comparison.

    The matching is **one-to-one**: a minimum-cost (Hungarian) assignment
    consumes each candidate at most once, so two references can never collapse
    onto the same nearest candidate and silently hide an unmatched outlier --
    that outlier instead surfaces as a large residual (or, when there are fewer
    candidates than references, as a ``nan`` match).

    Returns a list of dicts ``{'ref', 'match', 'delta'}`` (one per reference
    eigenvalue), where ``delta = match - ref`` is the complex residual.
    """
    ref = np.asarray(reference, dtype=complex)
    cand = np.asarray(candidates, dtype=complex)
    nan = complex(float("nan"), float("nan"))
    out = [{"ref": complex(z), "match": nan, "delta": nan} for z in ref]
    if ref.size == 0 or cand.size == 0:
        return out
    cost = np.abs(ref[:, None] - cand[None, :])  # (n_ref, n_cand)
    rows, cols = linear_sum_assignment(cost)  # one-to-one min-cost match
    for r, c in zip(rows, cols):
        out[int(r)] = {
            "ref": complex(ref[r]),
            "match": complex(cand[c]),
            "delta": complex(cand[c] - ref[r]),
        }
    return out


def imex_tableau(name):
    """Return the shenfun IMEXRK ``(a, b)`` Butcher tableau by scheme name.

    Shared by the plane-Couette and Taylor-Couette linear IMEXRK steppers so the
    two geometries are advanced by the *same* time-integration coefficients.
    """
    name = name.upper()
    if name == "IMEXRK111":
        a = np.array([[0, 0], [0, 1]], dtype=float)
        b = np.array([[0, 0], [1, 0]], dtype=float)
    elif name == "IMEXRK222":
        gamma = (2 - np.sqrt(2)) / 2
        delta = 1 - 1 / (2 * gamma)
        a = np.array([[0, 0, 0], [0, gamma, 0], [0, 1 - gamma, gamma]], dtype=float)
        b = np.array([[0, 0, 0], [gamma, 0, 0], [delta, 1 - delta, 0]], dtype=float)
    elif name == "IMEXRK443":
        a = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1 / 2, 0, 0, 0],
                [0, 1 / 6, 1 / 2, 0, 0],
                [0, -1 / 2, 1 / 2, 1 / 2, 0],
                [0, 3 / 2, -3 / 2, 1 / 2, 1 / 2],
            ],
            dtype=float,
        )
        b = np.array(
            [
                [0, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, 0],
                [11 / 18, 1 / 18, 0, 0, 0],
                [5 / 6, -5 / 6, 1 / 2, 0, 0],
                [1 / 4, 7 / 4, 3 / 4, -7 / 4, 0],
            ],
            dtype=float,
        )
    else:
        raise ValueError("scheme must be IMEXRK111, IMEXRK222, or IMEXRK443")
    return a, b


def imexrk_step(q0, Aimp, Aexp, M, a, b, dt, solve_stage):
    """Advance one IMEXRK descriptor-system step ``M q' = (Aimp + Aexp) q``.

    ``Aimp`` is treated implicitly (diffusion + saddle-point constraint rows),
    ``Aexp`` explicitly (advection and the remaining couplings).  ``solve_stage``
    is an operator-specific callback ``solve_stage(gamma, rhs)`` that solves
    ``(M - dt*gamma*Aimp) x = rhs`` with the operator's constraint rows pinned.

    Shared between the plane-Couette and Taylor-Couette linear steppers so a
    DNS-style time-stepped comparison uses identical integrator logic on both.
    """
    q0 = np.asarray(q0, dtype=complex)
    stages = []
    nstages = a.shape[0] - 1
    Mq0 = M @ q0
    for rk in range(nstages):
        rhs = np.array(Mq0, dtype=complex, copy=True)
        for j in range(rk + 1):
            source = q0 if j == 0 else stages[j - 1]
            rhs = rhs + dt * b[rk + 1, j] * (Aexp @ source)
        for j in range(rk):
            rhs = rhs + dt * a[rk + 1, j + 1] * (Aimp @ stages[j])
        stages.append(solve_stage(a[rk + 1, rk + 1], rhs))
    return stages[-1]


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
