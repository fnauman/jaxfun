"""Plane Couette divergence diagnostic using the jaxfun KMM solver.

This is the jaxfun counterpart of couette/pcf_fluctuations_divV.py.  It reuses
PlaneCouetteFluctuationJax and reports the L2 divergence after time stepping.
"""

from __future__ import annotations

import argparse

import sympy as sp

try:
    from examples.pcf_fluctuations_jax import PlaneCouetteFluctuationJax
except ModuleNotFoundError:  # direct script execution from examples/
    from pcf_fluctuations_jax import PlaneCouetteFluctuationJax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs=3, default=(17, 16, 16))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--Re", type=float, default=600.0)
    parser.add_argument("--family", choices=("L", "C"), default="C")
    parser.add_argument("--amp", type=float, default=0.05)
    args = parser.parse_args()

    domain = ((-1.0, 1.0), (0.0, 4.0 * float(sp.pi)), (0.0, 2.0 * float(sp.pi)))
    solver = PlaneCouetteFluctuationJax(
        N=tuple(args.N),
        domain=domain,
        dt=args.dt,
        Re=args.Re,
        family=args.family,
        perturbation_amplitude=args.amp,
    )
    state = solver.solve(solver.initial_state(), args.steps)
    diag = solver.diagnostics(state)
    print(f"divL2={float(diag['divL2']):.6e} Epert={float(diag['Epert']):.6e}")


if __name__ == "__main__":
    main()
