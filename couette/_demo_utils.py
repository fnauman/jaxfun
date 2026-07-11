"""Small helpers shared by demo scripts."""

from __future__ import annotations

import os


def default_thread_cap(default="2"):
    """Keep demo CLI runs laptop-friendly unless the caller configured BLAS."""
    cap = os.environ.get("SHENFUN_DEMO_THREADS", default)
    if not cap:
        return
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, cap)
