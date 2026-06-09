# Divergence-Free Plane Couette MHD Notes

Date: 2026-05-28

> **See also.**  This is the *nonlinear DNS* (vector-potential `B=curl(A)`)
> solver.  Its linear-stability siblings — the dense collocation
> `_pcf_linear.PlaneCouetteLinear` and Galerkin
> `pcf_galerkin_linear.PlaneCouetteGalerkinLinear` (primitive `b` + magnetic
> pressure) — plus the apples-to-apples PCF↔Taylor-Couette comparison
> (`thin_gap_compare.py`) are documented in `pcf_algorithms.md` §13 and
> `README_Couette.md`.

## Implemented solver

The new solver is `demo/pcf_mhd_divfree.py`. It keeps the existing `ChannelFlow.KMM` Plane Couette velocity formulation, but it does not advance the magnetic field components directly. Instead it advances a vector potential `A` and derives

```text
dA/dt = U x B + eta*lap(A)
B = curl(A)
J = curl(B)
```

where `U = U_wall*x*e_y + u_prime`. The magnetic invariant is the compatible-space identity `div(curl(A)) = 0`. In shenfun spaces the implemented chain is:

```text
A in [TD, TD, TD]
B = curl(A) in [TD, TC, TC]
J = curl(B) in [TC, TD, TD]
```

This is the key intervention. `B` is recomputed from `A` before every nonlinear evaluation and diagnostic, so the code never relies on a componentwise magnetic update to preserve `div(B)`.

The wall model is the simple vector-potential and no-normal-flux model implied by `A_y = A_z = 0` at the walls, so `B_x = d_y A_z - d_z A_y = 0` at `x = +/-1`. This is enough for a local PCF MHD integrator with a discrete solenoidal magnetic field.

## Why earlier direct-B attempts failed

*(Historical: the `pcf_mhd.py` / `pcf_mhd_fixed.py` exploratory files referenced
below were removed once the vector-potential solver landed; this section is kept
because it explains why the surviving `pcf_mhd_divfree.py` uses `B = curl(A)`.)*

`pcf_mhd.py` evolved `B` directly in a TD-based vector space and also stored curl-like quantities in TD components. That is incompatible with the wall-normal derivative mapping: an x-derivative of a Dirichlet field naturally lives in the unconstrained `TC` space, not another `TD` space. Those extra projections break the operator identities that the scheme needs.

`pcf_mhd_fixed.py` fixed part of that by storing `J = curl(B)` in the mixed curl space `[TD, TC, TC]` and by using symbolic `curl(U x B)` for the induction source. The diagnostic `div(curl(U x B))` was then near roundoff, which confirms that the source curl was compatible. But the solver still advanced `B` component by component in `TD^3`; the implicit diffusion solves and wall-space projection do not form an exact H(div) evolution. Evidence from the control run:

```text
pcf_mhd_fixed.py, N=(8,8,8), family=L, dt=0.001, t=0.002
initial divB L2 = 2.22e-17
final divB L2   = 5.92e-07
div(curl(UxB))  = O(1e-18)
```

So the remaining failure was not the explicit induction curl; it was the direct componentwise `B` integration in an incompatible magnetic space.

The vector-potential and Coulomb attempts were pointed in the right direction, but they mixed in extra gauge and boundary-condition machinery before the simpler invariant was verified. They also forced the shenfun optimization mode to `numba`; in the current environment, Chebyshev matrix paths need the installed cython backend, while small Chebyshev biharmonic tests need enough wall-normal modes. The new file leaves the installed backend alone by default and exposes `--family L` for small tests.

## Verification performed

Focused pytest suite:

```text
/home/nauman/miniconda3/envs/shenfun/bin/python -m pytest -q demo/test_pcf_mhd_divfree.py
..                                                                       [100%]
2 passed in 2.79s
```

Tiny Legendre invariant test:

```text
N=(8,8,8), family=L, Re=400, Rm=400, dt=0.001, t=0.003
final divU L2 = 9.41e-17
final divB L2 = 3.05e-21
final divB relative RMS = 2.97e-20
```

Chebyshev compatible-space test:

```text
N=(16,16,16), family=C, Re=400, Rm=400, dt=0.001, t=0.001
final divU L2 = 9.03e-17
final divB L2 = 4.71e-21
final divB relative RMS = 4.59e-20
```

Near-transition finite-amplitude PCF MHD run:

```text
N=(24,48,24), family=C, Re=400, Rm=400, dt=0.005, t=5.0
velocity perturbation amplitude = 0.20
magnetic amplitude = 0.05
per-step assertions: divB L2 < 1e-9 and divU L2 < 1e-9
final Epert = 9.991126056224221
final Emag  = 0.06065490963089308
final divU L2 = 2.84e-16
final divB L2 = 2.05e-16
final divB relative RMS = 8.32e-16
```

The Re=400 case is the relevant near-transition finite-amplitude check for this workspace. It remained finite, retained magnetic energy, and kept both velocity and magnetic divergence at roundoff-level values through 1000 asserted IMEX steps with nonlinear advection, Lorentz force, and induction coupling active.

## Practical run commands

Small invariant check:

```bash
/home/nauman/miniconda3/envs/shenfun/bin/python /home/nauman/cfd/fn_shenfun/demo/pcf_mhd_divfree.py --family L --nx 8 --ny 8 --nz 8 --dt 0.001 --end-time 0.003 --perturbation-amplitude 0.05 --magnetic-amplitude 0.02 --max-divb-l2 1e-12 --max-divu-l2 1e-12 --assert-every-step
```

Near-transition check:

```bash
/home/nauman/miniconda3/envs/shenfun/bin/python /home/nauman/cfd/fn_shenfun/demo/pcf_mhd_divfree.py --family C --nx 24 --ny 48 --nz 24 --Re 400 --Rm 400 --dt 0.005 --end-time 5.0 --perturbation-amplitude 0.20 --magnetic-amplitude 0.05 --max-divb-l2 1e-9 --max-divu-l2 1e-9 --assert-every-step
```
