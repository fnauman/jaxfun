# Keplerian MRI in wall-bounded Plane Couette: setup and extensions

Date: 2026-05-30

> **See also.**  The wall-bounded MRI shearpy *DNS* here can be cross-checked
> against its linear operators — `_pcf_linear.PlaneCouetteLinear.shearpy(...)`
> (dense collocation; used in `couette_linear_benchmarks.md`) and
> `pcf_galerkin_linear` — and against the Taylor-Couette MRI in the thin-gap
> limit via `thin_gap_compare.py --limit shearing --mhd` (the rotating,
> Rayleigh-stable limit).  See `pcf_algorithms.md` §13 and `README_Couette.md`.

Companion to `demo/pcf_mhd_mri_shearpy.py` (the solver) and
`demo/pcf_mhd_divfree_notes.md` (the divergence-free vector-potential scheme it
is built on). This note documents the physical setup of the MRI case and the
realistic ways to extend it.

## 1. What the solver actually solves

It is the wall-bounded Plane Couette analogue of a local shearing box. The
velocity solver is `ChannelFlow.KMM` (Chebyshev/Legendre walls in `x`, Fourier
in `y, z`); the magnetic field is `B = B0 + curl(A)` with the vector potential
`A` advanced in the Weyl gauge, so `div(B) = 0` holds to machine precision by
construction (see the divfree notes).

Coordinate / direction map (shearing-box convention):

| index | coordinate | role | numerics |
|------|-----------|------|----------|
| 0 | `x` | radial, shear-gradient, **wall-normal** | Dirichlet (no-slip walls) |
| 1 | `y` | azimuthal, streamwise (wall motion) | Fourier (periodic) |
| 2 | `z` | vertical, **rotation axis** `Omega = Omega e_z` | Fourier (periodic) |

Base flow `U_b(x) = -S x e_y`, rotation `Omega e_z`, imposed uniform field
`B0 = (0, by, bz)`. The implemented source terms (verified analytically and
against the linear MRI dispersion relation):

```text
du_x/dt += 2*Omega*u_y                      (Coriolis)
du_y/dt += (S - 2*Omega)*u_x                (Coriolis + base-flow shear)
dB_y/dt += -S*B_x                           (shear induction, via U_b x B)
+ J x B_total  Lorentz force, J = curl(curl(A))   (B0 carries no current)
```

Key parameters and the dimensionless groups printed at startup:

- `q = S/Omega` — shear parameter. Keplerian `q = 3/2` (default `S=1`,
  `Omega=2/3`). Rayleigh-stable for `q < 2`.
- `kappa^2 = 2*Omega*(2*Omega - S) = 2*Omega^2*(2 - q)` — epicyclic frequency
  squared; `> 0` (hydrodynamically stable) for `q < 2`.
- `Re = U/nu`, `Rm = U/eta`, magnetic Prandtl `Pm = Rm/Re = nu/eta`.
- `v_A = bz` (vertical Alfven speed with rho = mu0 = 1).

## 2. Choosing parameters so the MRI is actually unstable

For a net **vertical** field and axisymmetric modes (`k_y = 0`, vertical
wavenumber `k_z = 2*pi*n / Lz`), the ideal incompressible MRI is unstable when

```text
0 < (k_z v_A)^2 < 2*q*Omega^2               [= 3*Omega^2 for q = 3/2]
```

(the Balbus--Hawley vertical-field cutoff `-d Omega^2/d ln r = 2 q Omega^2`, matching
the Taylor--Couette note's `3 Omega^2`; the earlier `4 Omega^2 (q - 1)` was wrong)
so the largest field that still leaves the `n = 1` mode unstable is roughly
`bz < sqrt(2*q)*Omega*Lz/(2*pi)`. The maximum growth rate is
`gamma_max = (q/2)*Omega = 0.75*Omega` for Keplerian, reached near
`k_z v_A ≈ (sqrt(15)/4)*Omega ≈ 0.97*Omega`, i.e. at wavelength

```text
lambda_MRI ≈ 2*pi*v_A / (0.97*Omega)
```

Design rule: pick `bz` and `Lz` so that `lambda_MRI` fits a few times inside
`Lz`. The shipped defaults (`bz = 0.025`, `Omega = 2/3`, `Lz = 1`) give
`lambda_MRI ≈ 0.24`, so ~4 wavelengths fit and the `n = 1` mode grows at
`gamma ≈ 0.20` (well below `gamma_max` because `k_z v_A` is small — increase
`bz` toward `~0.1` to approach the fastest-growing mode). The shipped tiny-box
defaults are CI-sized; raise `N`, `Ly`, and integration time for science.

**Seeding.** The fastest mode is axisymmetric, so `initialize()` now seeds
`k_y = 0`, `k_z = {1,2,3}` channel-mode velocity plus a little non-axisymmetric
content. A purely non-axisymmetric seed (the previous default) only reaches the
same saturated state after a long nonlinear incubation, because sheared
non-axisymmetric modes grow only transiently.

## 3. Domain-size extensions

- **`Lz` (vertical, `--lz`)** — sets the fundamental `k_z` and therefore which
  MRI modes exist. Larger `Lz` admits more unstable channel modes and richer
  saturation; too small stabilizes the box (fundamental above the marginal
  `k_z`). This is the single most important box dimension for net-flux MRI.
- **`Ly` (azimuthal, `--ly`)** — sets the smallest `k_y`. Needed for
  non-axisymmetric / parasitic (Kelvin–Helmholtz / tearing) modes that break up
  the channel solution and sustain turbulence. Linear axisymmetric growth is
  independent of `Ly`, but turbulent saturation is not. Standard shearing boxes
  use `Ly >= Lx` (e.g. aspect `1 : pi : 1` or `1 : 4 : 1`).
- **`Lx` (radial gap, `--lx`)** — wall-to-wall distance. Controls the radial
  mode content and the wall boundary layers. The default `(-2, 2)` is a wide
  gap; narrowing it strengthens wall effects, widening it approaches the local
  (gap-independent) limit.
- Useful aspect-ratio presets to try: `1:pi:1`, `1:2:2`, `1:4:1`. The current
  default `lx=4, ly=4, lz=1` is vertically short; `lz` is usually the dimension
  to grow first.

## 4. Magnetic-field wall boundary conditions

This is the most physically consequential choice and the hardest to change.
The field BC is set entirely by the function space of `A` (currently
`CD = VectorSpace(self.TD)`, i.e. all three components Dirichlet `A = 0` at the
walls).

- **Perfectly conducting wall (current).** `A_y = A_z = 0` gives
  `B_x = d_y A_z - d_z A_y = 0` at the walls: **no normal field**, tangential
  field free. (`A_x = 0` is just a convenient gauge fix.) This is the simplest
  consistent MHD wall and what the solver uses. The wall-bounded channel mode
  must then have `k_x != 0`, which lowers growth rates relative to the
  unbounded shearing box — a real physical difference from periodic-box MRI.
- **Pseudo-vacuum / vertical-field wall (`B_y = B_z = 0`, `B_x` free).** Common
  in dynamo and solar work. Requires tangential `B` to vanish instead of
  normal `B`; in this formulation that means changing which `A` components are
  Dirichlet vs Neumann (roughly: `A_x` Dirichlet, tangential `A` Neumann) and
  re-deriving the curl spaces. Moderate change.
- **Insulating / vacuum exterior (current-free outside).** Physically correct
  for laboratory MRI (Taylor–Couette) and many astrophysical surfaces: `B`
  matches a potential field decaying outside the walls. This couples the wall
  values nonlocally across Fourier modes (an exterior Laplace solve per
  `(k_y, k_z)`), which is why earlier insulator attempts in this repo did not
  work in the spectral-Galerkin setting. Hardest extension; needs a dedicated
  per-wavenumber boundary operator.
- **Thin / finite-conductivity wall (Hollerbach-type).** A mixed/Robin BC that
  interpolates between perfectly conducting and insulating, parameterised by a
  wall conductance. Good compromise if a full vacuum solve is too costly.

Practical note: any new field BC must keep `B = curl(A)` in compatible spaces so
the `div(curl(A)) = 0` invariant survives — verify with the existing
`--assert-every-step --max-divb-l2` machinery after the change.

## 5. Field configuration: net flux vs zero net flux

- **Net vertical flux** (`--by 0 --bz>0`, default): linearly unstable, cleanest
  MRI, transport scales with `bz`. The classic benchmark.
- **Net azimuthal flux** (`--by>0 --bz 0`): toroidal-field MRI; the dominant
  instability is non-axisymmetric, so it needs adequate `Ly` resolution and a
  non-axisymmetric seed.
- **Zero net flux** (`--by 0 --bz 0` with `--magnetic-amplitude>0`): no imposed
  field; sustaining turbulence is a subcritical MRI-dynamo problem. It runs and
  stays divergence-free, but whether it self-sustains depends strongly on `Rm`,
  `Pm`, resolution, and box size — expect decay at low `Rm`. Treat as a research
  target, not a guaranteed turbulent state.

## 6. Other physics extensions

- **Magnetic Prandtl number.** Already supported: set `--Re` and `--Rm`
  independently (`Pm = Rm/Re`). MRI saturation is sensitive to `Pm`; scans
  around `Pm = 1` are standard.
- **Vertical gravity / stratification.** Currently unstratified. Add a buoyancy
  term (Boussinesq) and an entropy/temperature equation to study stratified MRI,
  convection–MRI interplay, and butterfly dynamo cycles. This is a substantial
  addition (new scalar field + its wall BCs).
- **Non-ideal terms.** Hall and ambipolar diffusion can be added to the
  induction (vector-potential) RHS for protoplanetary-disc regimes.
- **Explicit shearing-periodic box (major).** Replace the wall-normal Dirichlet
  `x` with a shearing-periodic radial direction (Fourier in `x` with the
  time-dependent wavenumber remap). This turns the wall-bounded PCF into the
  canonical local MRI box and makes results directly comparable to the shearing-
  box literature (Hawley–Gammie–Balbus and successors). shenfun has no native
  shearing-periodic remap, so this is a significant rewrite rather than a flag.

## 7. Diagnostics worth adding

The current diagnostics track energies and divergence. For MRI turbulence add:

- **Maxwell stress** `<-B_x B_y>` and **Reynolds stress** `<u_x u_y>`, and the
  transport coefficient `alpha = (R_xy^Reynolds + M_xy^Maxwell)/p0` (or
  normalised by `B0^2`). These are the headline MRI outputs.
- **Butterfly diagram**: time–height (or time–`x`) plot of the horizontally
  averaged `B_y` to see field reversals / dynamo cycles.
- **Energy spectra** in `(k_y, k_z)` and time-averaged saturated energies.
- Time-averaging of all of the above over many orbital times `2*pi/Omega`.

## 8. Quick recipes

Net-flux MRI, longer run that clearly grows and saturates:

```bash
/home/nauman/miniconda3/envs/shenfun/bin/python /home/nauman/cfd/fn_shenfun/demo/pcf_mhd_mri_shearpy.py \
  --family L --nx 32 --ny 32 --nz 32 --lx 4 --ly 4 --lz 1 \
  --Re 1000 --Rm 1000 --shear 1.0 --omega 0.6667 --by 0 --bz 0.025 \
  --dt 0.005 --end-time 60 --moderror 200 --perturbation-amplitude 1e-3
```

Zero-net-flux smoke test (expect decay unless Rm/box are pushed):

```bash
/home/nauman/miniconda3/envs/shenfun/bin/python /home/nauman/cfd/fn_shenfun/demo/pcf_mhd_mri_shearpy.py \
  --family L --nx 16 --ny 16 --nz 16 --by 0 --bz 0 \
  --magnetic-amplitude 0.05 --perturbation-amplitude 1e-2 \
  --dt 0.005 --end-time 5 --moderror 50
```
