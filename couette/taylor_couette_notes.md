# Taylor-Couette hydro & MHD/MRI solvers (shenfun)

Cylindrical-annulus companions to the Cartesian plane-Couette MHD/MRI shearing-box
demos (`pcf_mhd_divfree.py`, `pcf_mhd_mri_shearpy.py`). Instead of a *local*
shearing box these use the **true annulus** `r ∈ [R1, R2]` and the exact
circular-Couette base flow, so the curvature physics (the `Ω(r)=a+b/r²` profile,
the Rayleigh criterion, epicyclic frequency `κ²`, global boundary conditions) is
captured directly.

> **See also (added after this note was first written).**  Besides the
> shenfun-Galerkin linear solvers and CNAB2 DNS described here, the directory now
> also has a dense Chebyshev-**collocation** linear operator
> (`taylor_couette_collocation.py`), an **IMEXRK** linear stepper
> (`taylor_couette_imexrk.py`) and an axisymmetric **IMEXRK DNS** companion
> (`taylor_couette_imexrk_dns.py`), plus the apples-to-apples PCF↔TC comparison
> (`thin_gap_compare.py`).  These are documented in `taylor_couette_algorithms.md`
> (§0, §2.8-2.9, §3.8, §5) and `README_Couette.md` ("Which approach to use").

| file | what it is | status |
|------|------------|--------|
| `taylor_couette_linear.py` | hydrodynamic linear-stability eigensolver | **validated** |
| `taylor_couette_mri.py`    | resistive MHD linear-stability (standard MRI) + ideal local dispersion | **validated** |
| `taylor_couette_dns.py`    | nonlinear DNS: hydro (axisymmetric + 3D) and MHD/MRI (axisymmetric + 3D) | **validated** |
| `test_taylor_couette.py`   | pytest validation of the linear solvers vs published benchmarks | — |
| `test_taylor_couette_dns.py` | pytest validation of the DNS (fixed point, linear growth, saturation) | — |

Both are **generalized-eigenvalue stability solvers**: they linearise about the
laminar circular-Couette state and return the complex growth rate `s` for a mode
`q(r) exp(s t + i m θ + i kz z)`. This directly delivers the "classic laminar
profile tests" (the base flow is the analytically-known steady state) and the
"MRI regimes" (onset, growth rates, critical parameters) without the cost and
near-marginal noise of a time-stepping DNS. A nonlinear DNS is the planned next
layer (see *Roadmap*).

## Method

Geometry: `eta = R1/R2`, `mu = Omega2/Omega1`, gap `d = R2-R1`. Base flow

```
Omega(r) = a + b/r^2,   V(r) = r*Omega(r)
a = Omega1 (mu - eta^2)/(1 - eta^2)
b = Omega1 R1^2 (1 - mu)/(1 - eta^2)
```

Only the radial direction is discretised — `(m, kz)` are parameters — with a 1D
Chebyshev/Legendre Galerkin basis on `[R1, R2]` (the `OrrSommerfeld_eigs.py`
strong-form / plain-measure pattern; the cylindrical `1/r`, `1/r²` factors are
carried as explicit sympy coefficients of the radial symbol `x`). The coupled
operator blocks are extracted to dense complex matrices and the generalized
problem `L q = s M q` is solved with `scipy.linalg.eig`. `M` is singular (the
continuity row / pressure column carry no `s`), so non-physical eigenvalues sit
at infinity and are filtered (`|s| < cap`).

- **Velocity / `b_r`**: no-slip / `b_r=0` → Dirichlet basis (`N-2` modes).
- **Pressure**: orthogonal basis sliced to `N-2` → the inf-sup-stable
  `P_N`–`P_{N-2}` pair.
- **Perfectly-conducting magnetic walls** (`taylor_couette_mri.py`):
  `b_r = 0` (Dirichlet), `d(r b_θ)/dr = 0` (Robin), `b_z' = 0` (Neumann).
  NB shenfun's Robin `{'R': (c, d)}` means `u + c·u'_ref = d` with the derivative
  in the *reference* coordinate, so the conducting condition `b_θ + r b_θ'_phys=0`
  needs `c = r_wall / J`, `J = (R2-R1)/2`. Getting this wrong yields a spurious
  growing magnetic mode (the magnetic analogue of a spurious pressure mode).

The hydro solver also splits `L = L0 + nu*Lv` so a critical-viscosity bisection
reuses a single assembly (`critical_nu`, `critical_reynolds`); the MHD solver
splits `L = L0 + nu*Lnu + eta_mag*Leta` likewise (`critical_eta_mag`,
`critical_Rm`).

## Validated benchmarks

Hydrodynamic (`taylor_couette_linear.py`):

| quantity | this solver | reference |
|----------|-------------|-----------|
| Re_c, η=0.5, μ=0 (N=32≡48) | **Re_c=68.187, a_c=3.154** | 68.19, 3.16 (Fasel & Booz 1984) |
| narrow gap η=0.95 | a_c=3.125 | a_c→3.117 (Chandrasekhar) |
| exchange of stabilities (m=0) | Im(s_lead)~1e-16 | stationary onset (classic) |
| Rayleigh line μ=η²=0.25 | growth≈−0.006 (marginal) | μ=η² (Rayleigh 1917) |

MHD / MRI (`taylor_couette_mri.py`):

| quantity | this solver | reference |
|----------|-------------|-----------|
| ideal local Keplerian s_max/Ω | **0.7500** | 0.75 = (3/4)Ω (Balbus-Hawley) |
| ideal local optimum (k vA)²/Ω² | **0.9377** | 15/16 = 0.9375 |
| Keplerian hydro null (B0=0) | stable (magnetic modes decay) | Rayleigh-stable |
| Keplerian + axial field, high Rm | MRI-**unstable** | standard MRI |
| MRI onset Rm, η=0.5 quasi-Kep, conducting, S=4.11 | **Pm=1→95.3, 0.1→32.9, 0.02→26.7** | Rm_min=24.7 at Pm=1e-5 (Rüdiger 2023) |
| MRI onset Rm, η=0.5 quasi-Kep, **insulating** (flux fn), S=5.21 | **Pm=0.1→28.2** (vs conducting 32.3) — insulating destabilises more easily | Rm_min=16.5 < conducting 24.7 at Pm→0 (Rüdiger 2023) |
| insulating-wall eigenmode `div(b)` | **~1e-16** (flux function `b_r=-(ikz/r)χ`, `b_z=(1/r)χ'`) | solenoidal by construction |

> The ideal `s_max=0.75 Ω` channel-mode limit is *not* obtained by shrinking the
> gap — no-slip walls force a radial wavenumber `~π/d`, which suppresses the
> channel mode in a thin gap. It is therefore validated **analytically** to
> machine precision (`mri_local_growth` / `mri_keplerian_optimum`).
>
> The global resistive onset is a 2-parameter (Rm, S) threshold. The published
> `Rm_min=24.7` is for `Pm=1e-5`; in the convenient `Pm=1` regime the extra
> viscous damping `ν kz² = (Pm/Rm) kz²` raises the threshold, and it relaxes
> toward the literature value as `Pm` is decreased — the onset Rm at S=4.11 goes
> `95.3 (Pm=1) → 32.9 (Pm=0.1) → 26.7 (Pm=0.02)`, converging on the published
> `24.7`. Nondimensionalize as `eta_mag=Omega1 R1 d/Rm`, `B0=S eta_mag/d`,
> `nu=Pm eta_mag`. `TaylorCouetteMRI.critical_Rm` now follows this fixed-`Pm`,
> fixed-`S` path; use `critical_Rm_fixed_B0_nu` only for ad hoc fixed-dimensional
> `B0`/`nu` scans.

## Usage

```bash
CONDA=shenfun   # the env with shenfun 4.2.2

# hydro: critical Taylor onset (outer cylinder at rest)
conda run -n $CONDA python demo/taylor_couette_linear.py --R1 1 --R2 2 --Omega1 1 --Omega2 0 --m 0

# hydro: pick an explicit (m, kz)
conda run -n $CONDA python demo/taylor_couette_linear.py --m 0 --kz 3.14 --nu 2e-3

# MRI: ideal local Keplerian optimum (analytic, instant)
conda run -n $CONDA python demo/taylor_couette_mri.py --local-check

# MRI: global growth, quasi-Keplerian (Omega2 defaults to Keplerian mu=eta^1.5)
conda run -n $CONDA python demo/taylor_couette_mri.py --R1 1 --R2 2 --B0 0.1 --nu 1e-3 --eta-mag 1e-3
```

Library use:

```python
from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
base = CircularCouette(R1=1, R2=2, Omega1=1, Omega2=0)
s = TaylorCouetteLinear(base, N=48)
print(s.critical_reynolds(m=0))          # {'Re_c': 68.19, 'a_c': 3.15, ...}

from taylor_couette_mri import TaylorCouetteMRI
m = TaylorCouetteMRI(base_kep, B0=0.1, nu=1e-3, eta_mag=1e-3, N=40)
kz, growth, _ = m.max_growth_over_kz(0, np.linspace(0.5, 8, 30))
```

## Findings & shenfun gotchas (recorded so they aren't rediscovered)

These came out of building/validating the solvers and are easy to trip over:

1. **Robin BC uses the *reference* derivative.** shenfun's `FunctionSpace(..., bc={'R':(c,d)})`
   imposes `u + c·u'_ref = d`, where `u'_ref` is `d/d(reference coord)`, **not** physical
   `d/dr`. On `domain=(R1,R2)` the map is linear with `J = dr_phys/dr_ref = (R2-R1)/2`, so a
   *physical* condition `u + γ·u'_phys = 0` needs `c = γ/J`. The perfectly-conducting wall
   `d(r b_θ)/dr = 0` (⇔ `b_θ + r·b_θ'_phys = 0`) therefore needs `c = r_wall/J`, not `r_wall`.
   **Getting this wrong gives a spurious _growing_ magnetic mode** — purely magnetic (`|u|=0`),
   scaling exactly with `eta_mag` — i.e. the magnetic analogue of a spurious pressure mode. It
   silently breaks both the B0=0 null test and any critical-Rm search. Guarded by
   `test_conducting_btheta_bc_is_satisfied`.
2. **`inner(v, coeff*u)` returns a *list*** of `SpectralMatrix` when `coeff` has several additive
   terms (e.g. `Ω = a + b/r²`); a single term returns one matrix. Always sum the list before
   `.diags().toarray()` (see `_dense` / `_blk`). A symbolically-zero coefficient (e.g. `i·m` at
   `m=0`) must be short-circuited to a zero block.
3. **The radial coordinate symbol is physical.** A space on `domain=(R1,R2)` uses sympy `x` as the
   *physical* `r ∈ [R1,R2]`; `inner(1, x)` integrates to `(R2²-R1²)/2`. So cylindrical `1/r`, `1/r²`
   coefficients are written directly as `1/x`, `1/x**2` (the OrrSommerfeld strong-form pattern).
4. **The annulus (`R1>0`) needs no `m=0` special-casing.** Unlike the pipe/disc demos
   (`pipe_poisson.py`, `unitdisc_helmholtz.py`) where `r=0` makes `m=0` a singular 1-D BVP needing
   `bc=(None,0)`, every Fourier mode on an annulus is a regular two-point BVP.
5. **Pressure inf-sup:** the `P_N`(velocity, Dirichlet)–`P_{N-2}`(pressure) pair is obtained with
   `SP.slice = lambda: slice(0, N-2)` on an orthogonal pressure space; the continuity row carries
   no `s`, so `M` is singular and the QZ solver (`scipy.linalg.eig`) returns infinite eigenvalues
   that are filtered by `|s| < cap`.
6. **The ideal `s_max=0.75 Ω` channel mode is NOT a thin-gap limit.** No-slip walls force a radial
   wavenumber `~π/d`; a *thin* gap maximises `k_r` and therefore *suppresses* the channel mode.
   Validate `s_max` analytically (`mri_local_growth`); validate the global solver against the
   resistive critical Rm instead.
7. **Insulating (vacuum) walls need a flux-function formulation, NOT per-component BCs.** In the
   primitive `(b_r,b_theta,b_z)` form the vacuum match is *coupled* — `b_r=(kappa/i kz)b_z` at the
   wall (`kappa` = the modified-Bessel `I_0`/`K_0` log-derivative of the exterior potential) — so
   imposing single-field Robin BCs on `b_r` and `b_z` separately leaves `div(b)~O(1)` at the wall.
   The fix (`magnetic_bc="insulating"`, m=0): a **poloidal flux function** `chi` with
   `b_r=-(i kz/r)chi`, `b_z=(1/r)chi'` makes `div(b)=0` identical, and the poloidal induction reduces
   to the Stokes operator `Lchi = d^2 - (1/r)d - kz^2`. Then a conducting wall is just `chi=0`
   (the chi equation at the wall forces `chi''=chi'/r`, i.e. `b_z'=0`, *automatically*), and an
   insulating wall is the single-field Robin `chi'=(kz^2/kappa)chi`; `b_theta` is Robin (conducting)
   or Dirichlet `b_theta=0` (insulating, vacuum toroidal field vanishes). The flux operator
   reproduces the primitive conducting eigenvalues to ~1e-6 and gives `div(b)~1e-16`. (m!=0 couples
   the poloidal/toroidal scalars at the wall and is not yet done.) NB **insulating + Keplerian is
   physically stable in the highly resistive limit** (Liu/Goodman/Herron/Ji 2006), so no-growth at
   very small Pm is expected physics, not a bug.
8. **Env:** `conda run -n shenfun python - <<'HEREDOC'` does **not** forward stdin — always run a
   real file. shenfun 4.2.2 lives in the `shenfun` conda env.

## Nonlinear DNS — `taylor_couette_dns.py`

Two classes: `AxisymmetricTCDNS` (axisymmetric, `m=0`) and `TaylorCouetteDNS`
(full 3D, azimuthal Fourier `m != 0`).  Both integrate the perturbation about the
exact circular-Couette base flow with the same IMEX method.

### Axisymmetric — `AxisymmetricTCDNS`

Time-steps the incompressible Navier-Stokes equations for **axisymmetric**
(`d/dtheta = 0`) flow in the annulus, keeping all three velocity components
(swirl retained). It integrates the *perturbation* `u` about the exact
circular-Couette base flow

```
W(r,z,t) = V(r) e_theta + u(r,z,t),   V(r) = a r + b/r
```

so the walls are homogeneous (`u = 0`) and the laminar state is the exact fixed
point `u = 0`. Subtracting the base balance (`dP_base/dr = V^2/r`) gives, for the
perturbation, the only linear base coupling as the algebraic centrifugal/Coriolis
pair (`+2 Omega u_theta` in r, `-2a u_r` in theta — the same blocks that drive the
Taylor instability in the linear solver), with quadratic self-advection carrying
the cylindrical metric terms (`-u_theta^2/r`, `+u_r u_theta/r`).

**Method.** Same discretisation philosophy as the linear solver — only the radial
operators carry the `1/r` factors explicitly (plain Cartesian measure; the axial
direction is Fourier, velocity a no-slip Dirichlet basis, pressure the inf-sup
`P_N`-`P_{N-2}` pair). Time stepping is **IMEX**: Crank-Nicolson for the linear
(viscous + coupling + pressure) operator, 2nd-order Adams-Bashforth for the
nonlinear advection (CNAB2, IMEX-Euler bootstrap). Each step is a single coupled
velocity-pressure `BlockMatrix` solve per axial Fourier mode (so incompressibility
is enforced *exactly* — no fractional-step splitting error), with the `k=0`
pressure null space removed by one constraint. The quadratic nonlinearity uses
optional 3/2-rule dealiasing.

**Validated** (`test_taylor_couette_dns.py`):

| check | result |
|-------|--------|
| laminar fixed point (`u=0` stays `u=0`) | exact |
| linear growth of a seeded eigenmode vs `TaylorCouetteLinear` | **rel. err ~1e-9** (e.g. `nu=1e-2,kz=3.13`: both `s=+0.127304`) |
| Rayleigh-stable seed (`mu=0.5>eta^2`) | decays |
| `div(u)` after the coupled solve | **~5e-14** (machine roundoff) |
| supercritical `Re=100` (`>Re_c~68`) | grows at the linear rate `sigma=0.127`, saturates to a steady Taylor-vortex state (`E~0.028`, radial jets `|u_r|~0.16`) |

```bash
# axisymmetric DNS: supercritical Taylor vortices (eta=0.5, mu=0, Re=100)
conda run -n shenfun python demo/taylor_couette_dns.py --nu 1e-2 --Nr 40 --Nz 16 \
    --Lz 2.007 --dt 4e-3 --end-time 80 --moderror 50
```

```python
from taylor_couette_dns import AxisymmetricTCDNS
dns = AxisymmetricTCDNS(base, nu=1e-2, Nr=40, Nz=16, Lz=2.007, dt=4e-3)
s = dns.seed_linear_eigenmode(kz_mode=1, amp=1e-4)   # returns linear eigenvalue
dns.run(80.0, moderror=50)                            # grows then saturates
```

> **shenfun gotcha (DNS).** The base-flow `Omega(r)` and all `1/r` factors must be
> written in the *2D space's radial symbol* `T.coors.psi[1]` (= `y`, axis 1), **not**
> the linear solver's global `x` — in a 2D `TensorProductSpace`, `x` is axis 0 (the
> *axial* coordinate `z`). Using `x` silently applies the curvature along the wrong
> axis (and `b/z^2` blows up at `z=0`), reducing the run to pure viscous decay.
> Separately, the divergence diagnostic must sum the three terms as *separate*
> derivative projections in physical space: combining `Dx(f_hat,...)` with a sympy
> `(1/r)*f_hat` coefficient inside one `inner`/`project` mis-evaluates and reports a
> spurious O(amplitude) "divergence" (the real `div(u)` is ~1e-13).

### Full 3D — `TaylorCouetteDNS`

Adds an azimuthal Fourier direction `theta in [0, 2 pi)` (complex Fourier; `z`
real Fourier; `r` Dirichlet), so non-axisymmetric modes (`m != 0`: wavy/spiral
vortices) are captured.  The linear operator now carries every coupling that
`TaylorCouetteLinear.assemble_parts(m, kz)` has: base-shear advection
`-Omega d/dtheta` (`-i m Omega`), viscous cross-coupling `-/+ (2/r**2) d/dtheta u_{theta,r}`
(`-/+ 2 i m/r**2`), the full scalar Laplacian (with `(1/r**2) d^2/dtheta^2`), and the
`(i m/r) u_theta` continuity / `(i m/r) p` pressure-gradient terms.  All of these are
treated **implicitly** (Crank-Nicolson) in the per-`(m, kz)` coupled block solve;
the quadratic self-advection (with `theta`-advection `(u_theta/r) d/dtheta` and the
metric pieces) is explicit (AB2), pseudo-spectral with 3/2 dealiasing in all three
directions.  Setting `Ntheta` so only `m=0` is present reproduces the axisymmetric
solver exactly.

**Validated** (`test_taylor_couette_dns.py`):

| check | result |
|-------|--------|
| `m=0` mode through the 3D code vs `TaylorCouetteLinear` | **rel. err ~1e-9** (both `s=+0.127304`) |
| `m=1` eigenmode growth vs `TaylorCouetteLinear` (after transient) | `Re(s)=+0.18128`, DNS `+0.18228` (**0.5%**) |
| `div(u)` for a resolved eigenmode (`m=0` and `m=1`) | ~1e-16 (roundoff) |
| `div(u)` for a crude `m=1` seed (inf-sup top-2-mode residual) | small, converges spectrally (~6e-3 at Nr=32 → ~7e-4 at Nr=48) |

```bash
# full 3D run, seed an m=1 perturbation (Ntheta>0 selects the 3D solver)
conda run -n shenfun python demo/taylor_couette_dns.py --Ntheta 16 --m 1 \
    --nu 6e-3 --Nr 40 --Nz 16 --dt 1e-3 --end-time 5 --moderror 50
```

```python
from taylor_couette_dns import TaylorCouetteDNS
dns = TaylorCouetteDNS(base, nu=6e-3, Nr=40, Ntheta=16, Nz=8, dt=1e-3)
s = dns.seed_linear_eigenmode(m=1, kz_mode=1, amp=1e-6)  # real Re[q e^{i(m th+kz z)}]
dns.run(5.0)
```

> **3D seeding note.** A single azimuthal mode `m>=1` injected as raw Fourier
> coefficients is *complex* in physical space (the `z`-real-FFT alone does not make
> `theta` real), which breaks the real-field energy/nonlinear paths.  Seed it as the
> *real* field `Re[q exp(i(m theta + kz z))]` (`seed_linear_eigenmode`) and measure
> growth after a short transient; `m=0` may be injected directly (real via the
> `z`-real-FFT) and matches linear theory instantly.

### MHD / MRI (axisymmetric) — `AxisymmetricMRIDNS`

Resistive-MHD DNS with an imposed uniform axial field `B0 e_z` (Alfven units,
`v_A = B0`): the nonlinear, time-stepping companion to the MRI eigensolver
`taylor_couette_mri.py`.  It advances the perturbation `(u, b)` about
`W = V(r) e_theta`, `B = B0 e_z`, and the **standard magnetorotational
instability** lives here -- a quasi-Keplerian profile that is *Rayleigh-stable*
(hydrodynamically inert) is destabilised by the axial field.

**Method.** Same IMEX/coupled-solve machinery as the hydro DNS, extended to the
7-field system `(u_r, u_theta, u_z, Pi, b_r, b_theta, b_z)`.  The **total pressure**
`Pi = p + B0 b_z` absorbs the imposed-field magnetic pressure, so the linear
Lorentz / induction couplings reduce to `+B0 db/dz` (momentum) and `+B0 du/dz`
(induction) with the field-stretching source `r Omega' b_r -> b_theta`; this linear
operator is *identical* to `TaylorCouetteMRI.assemble_parts(0, kz)` and is advanced
implicitly (Crank-Nicolson).  The quadratic Maxwell/Reynolds advection
`(u.grad)u - (b.grad)b` and the induction EMF curl `curl(u x b)` are explicit (AB2),
pseudo-spectral with 3/2 dealiasing.  Perfectly-conducting walls use the same radial
bases as the eigensolver: `b_r = 0` (Dirichlet), `d(r b_theta)/dr = 0` (Robin),
`b_z' = 0` (Neumann).  `div(u) = 0` is enforced exactly by the total-pressure block
solve; `div(b) = 0` is preserved by the induction dynamics (the seeded eigenmode is
solenoidal and the discrete operator keeps it so) and is monitored every step.

> Note: this uses **direct** `b`-components (not a `B = curl(A)` vector potential).
> The earlier roadmap suggested `A` for a `div(B)=0` identity, but in practice the
> direct-component induction keeps `div(b)` at roundoff (~1e-17 for a resolved
> eigenmode, ~1e-13 with the nonlinear EMF active), so the extra `A`-gauge / wall-BC
> machinery was unnecessary for conducting walls.  Because `b` is **never**
> pressure-projected (unlike `u`), a non-eigenmode IC must *start* solenoidal *and*
> keep `u` divergence-free (else the imposed-field induction injects
> `div(b) ~ dt*B0*d_z div(u)`): `set_random` seeds a div-free velocity (Stokes
> stream function) plus a toroidal `b_theta` (`b_r = b_z = 0`, exactly solenoidal
> for axisymmetric flow since `div(b)` ignores `b_theta`); the poloidal field then
> develops dynamically.  `seed_linear_eigenmode` gives a fully solenoidal
> poloidal+toroidal seed.

**Validated** (`test_taylor_couette_dns.py`):

| check | result |
|-------|--------|
| linear MRI growth of a seeded eigenmode vs `TaylorCouetteMRI` | **rel. err 3.9e-8** (Keplerian, `Rm=1000`, `S=100`, `kz=6`: both `s=+0.340397`) |
| `div(u)`, `div(b)` (resolved eigenmode) | **~1e-17** (machine roundoff) |
| `B0=0` quasi-Keplerian | energy **decays** (Rayleigh-stable, no MRI) |
| nonlinear (Lorentz + EMF, dealiased) `div(b)` | ~1e-13 |
| nonlinear MRI saturation (Keplerian + `B0`) | grows at the MRI rate `sigma=0.34`, **field amplified ~1e8x** (`E_mag: 1.5e-10 -> 1.6e-2`), saturates ~`t=32` into a fluctuating MHD-turbulent state (`E_kin ~ E_mag`); `div(b)` stays `<~1e-5` |

```bash
# axisymmetric MRI: quasi-Keplerian + axial field, watch it grow & saturate
conda run -n shenfun python - <<'PY'
import sys; sys.path.insert(0, "demo")
from taylor_couette_linear import CircularCouette
from taylor_couette_dns import AxisymmetricMRIDNS
import math
base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)            # Keplerian
dns = AxisymmetricMRIDNS(base, B0=0.1, nu=1e-3, eta_mag=1e-3,
                         Nr=40, Nz=24, Lz=2*math.pi/6.0, dt=2e-3)
s = dns.seed_linear_eigenmode(kz_mode=1, amp=1e-5)         # returns eigenvalue
dns.run(40.0, moderror=200)
PY
```

### MHD / MRI (full 3D) — `TaylorCouetteMRIDNS`

The non-axisymmetric resistive-MHD DNS: the 3D azimuthal-Fourier machinery of
`TaylorCouetteDNS` combined with the imposed-axial-field MHD physics of
`AxisymmetricMRIDNS`, i.e. the nonlinear companion to `TaylorCouetteMRI` for
`m != 0` perturbations.  Fields depend on `(theta, z, r)` (`theta`, `z` Fourier;
`r` radial); the full linear operator of `TaylorCouetteMRI.assemble_parts(m, kz)`
is reproduced -- base-shear advection `-Omega d/dtheta` on every component, the
viscous/resistive `+-(2/r^2) d(.)/dtheta` cross-coupling between the `r` and
`theta` components of `u` and of `b`, the `(1/r^2) d^2/dtheta^2` Laplacian piece,
the `(1/r) dPi/dtheta` pressure gradient and `(1/r) du_theta/dtheta` continuity
term -- plus the imposed-field Lorentz/induction `+-B0 d/dz` and the MRI source
`r Omega' b_r`.  The quadratic terms add the cylindrical `(u_theta/r) d/dtheta`
and `(b_theta/r) d/dtheta` advection and the azimuthal `(1/r) d/dtheta` pieces of
the EMF curl `-curl(u x b)`.  Time stepping (IMEX CNAB2), the coupled per-`(m,kz)`
7-field block solve (so `div(u)=0` exactly), the total pressure `Pi = p + B0 b_z`,
and 3/2 dealiasing are exactly as in the axisymmetric MRI solver.

> **Conducting walls are `m`-independent.** `b_r = 0` on the wall forces
> `d b_r/dtheta = 0` there, so the tangential-`E` conditions collapse to the same
> `d(r b_theta)/dr = 0` (Robin) and `b_z' = 0` (Neumann) for every `m` -- the 3D
> solver reuses the axisymmetric radial bases.
>
> **3D eigenmode seeding** is the real field `Re[hat q(r) e^{i(m theta + kz z)}]`.
> The MRI eigenvector is genuinely **complex even at `m=0`** (`u` and `b` are out of
> phase), so the radial profile must be evaluated real/imag-separately and
> recombined -- assigning the complex eigenvector into a real-dtype radial
> `Function` silently drops `Im(q)`, which destroys the radial/axial balance that
> makes the mode divergence-free (it would leave `div(b) ~ 1e-6` instead of
> roundoff).  `set_random` seeds an **exactly divergence-free** axisymmetric
> velocity (Stokes stream function `psi=g(r)*rand(z)`, `g=sin^2`) and leaves
> `b = 0`; this matters because `b` is never pressure-projected, so a
> non-solenoidal `u` would inject `div(b) ~ dt*B0*d_z div(u)` through the
> imposed-field induction (a toroidal-only `b` is also *not* solenoidal for
> `m != 0`).  `seed_linear_eigenmode` rejects azimuthal modes the grid cannot
> resolve (`2|m| >= Ntheta`, which would alias the sampled phase to another
> mode while still reporting the requested mode's eigenvalue).

**Validated** (`test_taylor_couette_dns.py`):

| check | result |
|-------|--------|
| `m=0` channel growth vs `TaylorCouetteMRI` | **rel. err 3.9e-8** (= the axisymmetric solver; `s=+0.34040`) |
| non-axisymmetric `m=1` (travelling wave, `Im(s) != 0`) | **rel. err 4e-7** (`s=+0.29068-0.67393i`) |
| non-axisymmetric `m=2` | **rel. err 2e-6** (`s=+0.15916-1.40503i`) |
| `div(u)`, `div(b)` (seeded complex eigenmode) | **~1e-12** (roundoff) |
| `B0=0` quasi-Keplerian | energy **decays** (Rayleigh-stable, no MRI) |
| nonlinear saturation: seeded `m=1` eigenmode | grows at the linear rate `sigma=0.29`, **field amplified ~1e5x** (`E_mag: 8e-7 -> 0.087` peak ~`t=26`), then the Maxwell/Reynolds + EMF nonlinearities saturate it into a sustained fluctuating MHD state; a strong `m=0` zonal mean develops yet the state stays genuinely 3D (`m != 0` energy fraction settles ~`0.19`); `div(b)` stays `<~7e-4` |

> **Nonlinear note.** At `Rm=1000`, `Pm=1`, conducting walls, the saturated
> *axisymmetric* MRI channel is stable to non-axisymmetric perturbations
> (parasitic `m != 0` modes decay rather than growing), so 3D structure is
> demonstrated by seeding a genuinely non-axisymmetric (`m=1`) eigenmode and
> following it through nonlinear saturation, not via parasitic breakdown of the
> `m=0` channel.

```bash
# full 3D MRI: seed a non-axisymmetric (m=1) MRI eigenmode (Ntheta>0 -> 3D solver)
conda run -n shenfun python demo/taylor_couette_dns.py --mhd --Ntheta 8 \
    --Omega1 1.0 --Omega2 0.35355 --B0 0.1 --nu 1e-3 --eta-mag 1e-3 \
    --Nr 32 --Nz 16 --Lz 1.0472 --m 1 --kz-mode 1 --amp 1e-6 \
    --end-time 4.0 --moderror 200
```

## Roadmap (remaining DNS layers)

### Insulating walls

Both the axisymmetric and 3D MRI DNS use perfectly-conducting walls.  The
remaining magnetic-BC option is **insulating walls** (a vacuum field-matching
radial BC instead of conducting; physically stable for Keplerian in the resistive
limit, so a later feature, not a first target).  The non-axisymmetric MHD layer is
now implemented (`TaylorCouetteMRIDNS`) -- the linear operator is
`TaylorCouetteMRI.assemble_parts(m, kz)` and reduces to the axisymmetric solver
for `m = 0`.

### Validation gates (met)

The DNS suite checks: laminar/zero-perturbation fixed point; hydro and MRI
small-amplitude growth matching the linear solvers; `div(u)`, `div(b)`, energies,
and non-finite guards every step; and nonlinear saturation.

### Development resource limits

The first usable DNS should include automated checks for:

- laminar circular Couette remains steady for `u=A=0`;
- hydro small-amplitude growth/decay matches `TaylorCouetteLinear` at the same
  `(m,kz,Re)`;
- MHD small-amplitude MRI growth matches `TaylorCouetteMRI` at the same
  `(m,kz,Re,Rm,B0)`;
- `div(u)`, `div(B)`, wall residuals, kinetic/magnetic energy, and non-finite
  values are checked every step in smoke tests;
- angular-momentum transport diagnostics include wall torque and radial flux.

### 4. Development resource limits

Default runs must be laptop-friendly. Every DNS CLI should set BLAS/OpenMP thread
defaults before importing NumPy/SciPy unless the user already set them:

```
SHENFUN_DEMO_THREADS=2
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=2
```

Expose `--threads` and `--max-ranks`; default `--threads=2` and reject MPI runs
with more than `--max-ranks` ranks unless the user explicitly raises it. Smoke
tests should use serial or at most two ranks, tiny grids, short end times, and no
expensive output by default. Mark larger DNS checks as `slow`.
