# Taylor–Couette solvers — algorithm & numerics reference

Audit-oriented description of the numerical methods used by the
cylindrical-annulus (Taylor–Couette) demos in this directory:

| file | class(es) | what it computes | method |
|------|-----------|------------------|--------|
| `taylor_couette_linear.py` | `CircularCouette`, `TaylorCouetteLinear` | hydrodynamic linear stability (growth rate `s`, non-modal growth) | global generalized eigenproblem, **shenfun-Galerkin** |
| `taylor_couette_mri.py` | `TaylorCouetteMRI` (+ local WKB helpers) | resistive-MHD linear stability (MRI) | global generalized eigenproblem, **shenfun-Galerkin** |
| `taylor_couette_collocation.py` | `TaylorCouetteCollocationLinear` | hydro + MHD linear stability (cross-check / thin-gap partner) | dense **Chebyshev-collocation** |
| `taylor_couette_imexrk.py` | `TaylorCouetteIMEXRKLinearStepper` | DNS-style linear time stepping of the Galerkin operators | dense descriptor-system **IMEXRK** |
| `taylor_couette_dns.py` | `AxisymmetricTCDNS`, `TaylorCouetteDNS`, `AxisymmetricMRIDNS`, `TaylorCouetteMRIDNS` | nonlinear time-stepping DNS (hydro & MHD, 2-D & 3-D) | IMEX **CNAB2** pseudo-spectral |
| `taylor_couette_imexrk_dns.py` | `AxisymmetricTCIMEXRKDNS`, `AxisymmetricMRIIMEXRKDNS` | axisymmetric nonlinear DNS companion (hydro + conducting MHD) | IMEX **IMEXRK** pseudo-spectral |
| `test_taylor_couette.py` | — | pytest vs published linear benchmarks | — |
| `test_taylor_couette_dns.py` | — | pytest vs linear theory & saturation | — |
| `test_thin_gap_comparison.py` | — | pytest: collocation↔Galerkin agreement + apples-to-apples PCF↔TC | — |

These are the cylindrical companions to the Cartesian plane-Couette demos
(`pcf_*.py`): they use the **true annulus** `r ∈ [R1, R2]` and the exact
circular-Couette base flow, so curvature physics (`Ω(r)=a+b/r²`, the Rayleigh
criterion, epicyclic frequency `κ²`, global wall BCs) is captured directly.
Physics/validation history lives in `taylor_couette_notes.md`; this file is
about *how the operators are discretized and solved*.

## 0. Solution approaches (orientation)

Several tasks have **more than one implementation**; the alternatives agree
where they overlap (checked by `test_thin_gap_comparison.py`):

- **Linear discretisation:** shenfun composite-basis **Galerkin**
  (`taylor_couette_linear.py`, `taylor_couette_mri.py`; §2.1-2.7) vs dense
  **Chebyshev-collocation** (`taylor_couette_collocation.py`; §2.8). Galerkin is
  the production path (critical-Re/Rm, default); collocation is the thin-gap
  cross-check and the reliable MHD transient-growth partner (it carries a
  magnetic-pressure projection the Galerkin conducting operator lacks).
- **Linear time stepping:** the eigen/non-modal route above, vs the DNS-style
  IMEXRK stepper (`taylor_couette_imexrk.py`; §2.9).
- **Nonlinear DNS scheme:** **CNAB2** (`taylor_couette_dns.py`; all four classes,
  3-D supported; §3) vs **IMEXRK** (`taylor_couette_imexrk_dns.py`; axisymmetric
  only; §3.8). Both nonlinear DNS solvers are **conducting-wall only** for MHD.
- **MHD walls:** `conducting` (any `m`) vs `insulating` (vacuum match,
  flux-function, `m=0` only; §2.6). **Insulating is a linear-analysis capability
  only** — the linear operators (`taylor_couette_mri`, `taylor_couette_collocation`)
  and the DNS scripts' `--linear-analysis --magnetic-bc insulating` flag (which
  runs the linear operator, not the time-stepping DNS).
- **PCF↔TC apples-to-apples** comparison: `thin_gap_compare.py` (§5).

See `README_Couette.md` ("Which approach to use") for a one-glance decision table.

---

## 1. Base flow (`CircularCouette`)

Circular-Couette ("ideal Couette") profile between cylinders of radii
`R1 < R2` rotating at `Ω1, Ω2` (`taylor_couette_linear.py:74-125`):

```
Ω(r) = a + b/r² ,   V(r) = Ω(r) r = a r + b/r
a = (Ω2 R2² − Ω1 R1²)/(R2² − R1²)
b = (Ω1 − Ω2) R1² R2² /(R2² − R1²)
```
so `V(R1)=Ω1 R1`, `V(R2)=Ω2 R2`. Identities used throughout the linearization
(all carried as **sympy** expressions in the radial symbol):

```
2Ω + r Ω' = 2a               (constant)          → shear2a
r Ω'       = −2b/r²                               → rOmega_p_sym
κ²(r)      = (1/r³) d(r²Ω)²/dr = 4 a Ω(r)         (epicyclic)
```
Control numbers: radius ratio `η = R1/R2`, rotation ratio `μ = Ω2/Ω1`, gap
`d = R2 − R1`, local shear `q = −d ln Ω/d ln r`. The **Rayleigh criterion**
(inviscid centrifugal stability) is `κ²>0` everywhere; a Keplerian profile
`Ω ∝ r^{−3/2}` (`μ = η^{3/2}`) has `κ² = Ω² > 0` — Rayleigh-stable yet
MRI-unstable.

---

## 2. Linear stability solvers — discretization

### 2.1 Modal ansatz and what is discretized

Perturbations are `q(r) · exp(s t + i m θ + i k_z z)` with azimuthal mode `m`
and axial wavenumber `k_z` as **fixed parameters**. Only the **radial**
direction is discretized; `θ, z` are analytic Fourier factors. This turns the
PDE stability problem into a 1-D generalized eigenproblem per `(m, k_z)`:

```
L q = s M q ,    solved with scipy.linalg.eig
```
`M` is singular (the continuity row and pressure column carry no time
derivative), so spurious eigenvalues appear at infinity and are filtered:
`w[isfinite(w) & (abs(w) < 1e6)]`, then sorted by descending `Re(s)`.

### 2.2 Radial bases (shenfun)

`family ∈ {'L' (Legendre), 'C' (Chebyshev)}`, `N` modes on `domain=(R1,R2)`:

```python
SD/SDv = FunctionSpace(N, family, bc=(0, 0), domain=dom)   # velocity, no-slip; also b_r=0
SP     = FunctionSpace(N, family,             domain=dom)   # pressure, orthogonal
SP.slice = lambda: slice(0, N - 2)                          # keep N-2 pressure modes
```

- Velocity uses the **Dirichlet** composite basis (`bc=(0,0)`): every basis
  function satisfies `φ(R1)=φ(R2)=0`, so no-slip is built into the space
  (dimension `n = N − 2`).
- Pressure uses the **orthogonal** basis truncated to `N − 2` modes. The pair
  velocity-`P_N` / pressure-`P_{N-2}` is the **inf-sup (LBB) stable** spectral
  element choice — it suppresses spurious pressure modes and makes the
  saddle-point operator well-posed. (`assert SP.dim() == SD.dim()`.)

### 2.3 Strong-form assembly with explicit `1/r` (OrrSommerfeld pattern)

The cylindrical metric factors are **not** absorbed into a weighted measure.
Instead the integrals use the plain Cartesian measure and every `1/r`, `1/r²`
appears as an explicit sympy coefficient of the radial symbol `x`:

```python
inner(test, coeff(x) * Dx(trial, 0, order))   #  ∫ test · coeff · d^order(trial)/dr^order
```
`_blk`/`_A`/`_Avp`/`_Aqu` (`taylor_couette_*`): each builds one dense `(n,n)`
block from a list of `(coeff, derivative_order)` terms. Note that
`inner` returns a **list** of `SpectralMatrix` objects when the coefficient is a
sum (e.g. `Ω = a + b/r²`); the helpers sum the list. Symbolically-zero
coefficients are skipped.

The scalar cylindrical Laplacian block is
```
Lp = ∂_rr + (1/r)∂_r − (m²/r² + k_z²)
```
and the vector-Laplacian diagonal piece adds `−1/r²` (the `±2im/r²` `r`–`θ`
cross terms are kept as off-diagonal `couple` blocks).

### 2.4 Hydrodynamic operator (`TaylorCouetteLinear`)

Four field blocks `(u_r, u_θ, u_z, p)`, each size `n`. The linearized equations
(`taylor_couette_linear.py:223-291`):

```
s u_r = −imΩ u_r + 2Ω u_θ − ∂p/∂r + ν(Lp − 1/r²) u_r
s u_θ = −imΩ u_θ − 2a u_r − (im/r)p + ν(Lp − 1/r²) u_θ   (+ 2im/r² coupling)
s u_z = −imΩ u_z − i k_z p + ν Lp u_z
  0   = (∂_r + 1/r) u_r + (im/r) u_θ + i k_z u_z          (continuity)
```
The viscous part is split out: `assemble_parts` returns `(L0, Lv, M)` with
`L = L0 + ν·Lv`, so a critical-viscosity bisection reuses one assembly and only
re-runs the cheap `eig`.

### 2.5 MHD/MRI operator (`TaylorCouetteMRI`)

Imposed uniform axial field `B0 = B0 e_z` in **Alfvén units** (`v_A = B0`). A
**total pressure** `Π = p + B0 b_z` absorbs `grad(B0 b_z)`, so the imposed-field
Lorentz force reduces to `i k_z B0 b` per component. Conducting walls use the
**primitive 7-field** system `(u_r, u_θ, u_z, Π, b_r, b_θ, b_z)` for any `m`
(`assemble_parts`); the diffusion is split three ways:
```
L = L0 + ν·Lnu + η·Leta            (η = eta_mag, magnetic diffusivity)
```
Induction terms: `b_r` is only advected (no shear source); `b_θ` is generated
from `b_r` at rate `r Ω'` — the MRI field-stretching term.

The local ideal-MRI dispersion relation (analytic reference, `mri_local_growth`,
`mri_keplerian_optimum`) gives the Balbus–Hawley targets `s_max = 0.75Ω` at
`(k_z v_A)² = 0.9375 Ω²`, cutoff `3Ω²`.

### 2.6 Magnetic boundary conditions

| wall type | `b_r` | `b_θ` | `b_z` |
|-----------|-------|-------|-------|
| **conducting** | `=0` (Dirichlet) | `d(r b_θ)/dr = 0` (Robin) | `b_z' = 0` (Neumann) |
| **insulating** | flux-function `χ` | `=0` (Dirichlet) | flux-function `χ` |

shenfun encodings (`taylor_couette_mri.py:159-183`, `316-358`):
- Robin `d(r b_θ)/dr = 0 ⇔ b_θ + r b_θ' = 0` → `bc={'left':{'R':(R1/J,0)},
  'right':{'R':(R2/J,0)}}` where `J = (R2−R1)/2` is the reference→physical
  Jacobian (shenfun's Robin coefficient is in reference-coordinate derivative).
- Neumann `b_z'=0` → `bc={'left':{'N':0},'right':{'N':0}}`.

**Insulating (vacuum) walls — poloidal flux function** (`m = 0` only). The
primitive per-component vacuum BCs are coupled and would leave `div(b) ~ O(1)`.
Instead an `m=0` poloidal flux `χ` is used:
```
b_r = −(i k_z/r) χ ,   b_z = (1/r) χ'   ⇒   div(b) = 0 identically
```
giving a **6-field** system `(u_r, u_θ, u_z, Π, χ, b_θ)`. The vacuum match
becomes a single-field **Robin** on `χ`: `χ'/χ = k_z²/κ`, where `κ` is the
log-derivative of the current-free exterior potential — modified Bessel
`I_0`/`I_1` inside, `K_0`/`K_1` outside (`scipy.special.iv, kv`). The conducting
limit is `χ=0` (Dirichlet), from which `b_z'=0` emerges automatically at the
wall. These flux bases depend only on `k_z`, so they are cached by `k_z`
(`_flux_bases`). `m≠0` insulating walls couple poloidal/toroidal scalars at the
wall and raise `NotImplementedError`; `k_z=0` raises `ValueError` (no exterior
Bessel limit).

### 2.7 Critical-parameter search

`critical_nu` / `critical_eta_mag` bisect the sign of the leading `Re(s)`
(growth decreases monotonically with diffusivity), reusing a single
`assemble_parts`. The MRI threshold uses **geometric** bisection (`Rm` spans
decades) at **fixed `Pm` and Lundquist `S`** (`critical_Rm`): `η` is varied
while `ν = Pm·η` and `B0 = S·η/d` are updated consistently — the parameter path
for standard MRI threshold comparisons (conducting `Rm_min ≈ 24.7`, insulating
`≈ 16.5` as `Pm → 0`).

### 2.8 Dense Chebyshev-collocation operator (`taylor_couette_collocation.py`)

`TaylorCouetteCollocationLinear` is the collocation counterpart to the
shenfun-Galerkin §2.4-2.5 operators: a **strong-form** discretisation on
Chebyshev–Lobatto points (`_pcf_linear.cheb_lobatto`, mapped to `[R1,R2]`), with
the cylindrical `1/r`, `1/r²` carried as `diag(1/r)` matrices and BC rows
imposed by row-replacement. It is **NumPy/SciPy only** (no shenfun; SciPy only
for `eig` and the insulating Bessel `iv,kv`). Intended as a thin-gap cross-check,
not a replacement for the Galerkin solvers.

Two differences from the Galerkin MRI operator matter:

- **Conducting MHD** is an **8-block** system `(u_r,u_θ,u_z,p,b_r,b_θ,b_z,φ)` —
  it adds a **magnetic-pressure multiplier `φ`** (mirroring the PCF operators)
  that projects `b` divergence-free. The Galerkin `TaylorCouetteMRI` conducting
  operator is 7-field with **no** such multiplier (it relies on the conducting
  BCs). The leading eigenvalues agree either way, but only the φ-projected
  collocation operator gives reliable MHD **transient growth** — hence the
  collocation↔collocation rule for non-modal MHD (§5, `README_Couette.md`).
- **Insulating** reuses the same `m=0` poloidal flux-function `χ`/`b_θ` 6-block
  form as the Galerkin operator (Bessel vacuum match).

API mirrors the Galerkin operators: `.eigs(m,kz)`, `.growth_rate`,
`.nonmodal_growth(m,kz,times,energy=)` (insulating energy needs an explicit
`kz`), `.energy_matrix`. The dense energy norm is the **nodal quadrature**
`diag(weights·r)` (vs the Galerkin Gram matrix) — so non-modal growth is only
comparable collocation↔collocation.

### 2.9 DNS-style linear time stepping (`taylor_couette_imexrk.py`)

`TaylorCouetteIMEXRKLinearStepper` advances the **same** generalized systems
assembled by `TaylorCouetteLinear` / `TaylorCouetteMRI` (via `assemble_parts`,
so `L = L0 + ν Lnu + η Leta`, `M` singular) as a dense descriptor system
`M q' = (Aimp + Aexp) q` using the shenfun IMEXRK tableaux. It shares
`imex_tableau` and the step core `imexrk_step` with the PCF stepper
(`pcf_imexrk_linear.py`) through `_linear_analysis.py`, which is what makes the
`--dns` comparison advance both geometries with **identical** integrator logic.

`--split diffusion` (default) treats diffusion plus the pressure/continuity
saddle-point rows implicitly and the base-flow couplings explicitly; `--split
full` is a stiff fully-implicit reference. `--scheme` is `IMEXRK111/222/443`
(default 222); the pressure block (index 3) is pinned per stage; `integrate`
requires an integer number of `dt` steps. Seeding the leading eigenmode recovers
its `Re(s)` — the time-integration consistency check, and the TC side of
`thin_gap_compare --dns`.

---

## 3. Nonlinear DNS

`taylor_couette_dns.py` provides four **CNAB2** classes (§3.1-3.7);
`taylor_couette_imexrk_dns.py` provides an axisymmetric **IMEXRK** companion
(§3.8) that reuses the same formulation, spaces and nonlinear terms. Four CNAB2
classes, increasing capability, all sharing the same algorithmic skeleton:

```
AxisymmetricTCDNS   (hydro,  r–z, ∂_θ=0)
TaylorCouetteDNS    (hydro,  r–θ–z, full 3D)
AxisymmetricMRIDNS  (MHD,    r–z, imposed B0 e_z)
TaylorCouetteMRIDNS (MHD,    r–θ–z, full 3D)
```

### 3.1 Perturbation formulation

Integrate the perturbation `u` (and `b`) about the exact base state
`U = V(r) e_θ` (`B = B0 e_z` for MHD), so the walls are homogeneous and `u=0`
(`b=0`) is the exact fixed point. The axisymmetric momentum perturbation
(`taylor_couette_dns.py:21-37`):

```
∂u_r/∂t  = −∂p/∂r + ν(L − 1/r²) u_r + 2Ω u_θ − N_r
∂u_θ/∂t  =          ν(L − 1/r²) u_θ − 2a u_r  − N_θ
∂u_z/∂t  = −∂p/∂z + ν L u_z                   − N_z
   0     = ∂u_r/∂r + u_r/r + ∂u_z/∂z                         (div u = 0)
```
with axisymmetric scalar Laplacian `L f = f_rr + f_r/r + f_zz`. The quadratic
self-advection carries the cylindrical metric terms, e.g.
`N_r = u_r u_{r,r} + u_z u_{r,z} − u_θ²/r`. The 3-D solvers add the
`−Ω ∂_θ` base-shear advection (`−imΩ` per mode), the `±2im/r²` viscous
`r`–`θ` cross-coupling, the `f_{θθ}/r²` Laplacian piece, and the azimuthal
continuity term `u_θ,θ/r`. The MHD solvers add the imposed-field couplings
`+B0 ∂_z b` (Lorentz) and `+B0 ∂_z u` (induction) plus the field-stretching
source `r Ω' b_r → b_θ`. **The DNS linear operators are exactly
`TaylorCouette*.assemble_parts(m, k_z)` term-for-term** — that is the design
constraint that makes the DNS reproduce linear growth rates to spectral
accuracy.

### 3.2 Spectral spaces

Radial: Dirichlet `SD` (velocity, `b_r`), orthogonal `S0` (derivative range &
nonlinear products), pressure `SP` sliced to `N−2` (same inf-sup pair as the
eigensolver), plus the magnetic Robin `Sbt` (`b_θ`) and Neumann `Sbz` (`b_z`)
bases. Periodic: Fourier `z` (real, `dtype='d'`); for 3-D an additional Fourier
`θ` (complex, `dtype='D'`, domain `[0,2π)`). Assembled into
`TensorProductSpace`s with the **radial axis as the solve axis** (`axes` puts it
last), and grouped into `CompositeSpace`s:

```
VV / VE = [TD, TD, TD]               (velocity, or 6 evolving MHD fields)
VQ      = [TD, TD, TD, TP, …]        (velocity/fields + pressure, for the coupled solve)
```
Conducting walls are **`m`-independent** in 3-D: `b_r=0` forces `∂_θ b_r=0` at
the wall, so the same Robin/Neumann radial bases serve every `m`.

### 3.3 Time integration — IMEX CNAB2

Second-order semi-implicit (`taylor_couette_dns.py:42-45, 288-313`):

- **Crank–Nicolson** (`½` implicit / `½` explicit) for the **linear** operator
  `A` = viscous + resistive + all base-flow couplings + pressure gradient;
- **2nd-order Adams–Bashforth** (`1.5 Nⁿ − 0.5 Nⁿ⁻¹`) for the **nonlinear**
  advection / Lorentz / EMF terms;
- **IMEX-Euler bootstrap** on the very first step (`_have_old=False`).

The implicit and explicit halves are pre-assembled once as block operators:
```
Limp = la.BlockMatrixSolver(  M/dt − ½A + grad p ;  div u = 0 )   # over VQ
Lexp = BlockMatrix(           M/dt + ½A )                          # over VV/VE
```
One step:
```
nonlinear(N_hat)                              # explicit terms, dealiased
rhs = Lexp · uⁿ − (1.5 Nⁿ − 0.5 Nⁿ⁻¹)        # AB2 (Euler on step 1)
sol = Limp(rhs, constraints=((3,0,0),))       # coupled velocity–pressure solve
```

**Coupled solve ⇒ exact incompressibility.** Velocity *and* pressure are solved
together in one `BlockMatrix` system per Fourier mode (a saddle-point solve),
so there is **no fractional-step / pressure-projection splitting error** and
`div(u)` is enforced to roundoff (~1e-13). The `k=0` pressure null space is
removed by the constraint `((3, 0, 0))` (pin pressure block 3, mode 0, to 0).

**Viscosity/resistivity are implicit** (the Crank–Nicolson `±½ ν L`, `±½ η L`
terms); the quadratic nonlinearities are explicit. The base-flow couplings
(Coriolis `2Ω`, shear `−2a`, `−imΩ` advection, field-stretching `rΩ'`,
imposed-field `B0 ∂_z`) are linear and so are also integrated implicitly via
Crank–Nicolson.

CNAB2 is not the only stepper: `taylor_couette_imexrk_dns.py` (§3.8) keeps this
exact formulation but advances with shenfun IMEXRK tableaux (axisymmetric only),
which is also what the PCF DNS uses.

### 3.4 Nonlinear term — pseudo-spectral, 3/2 dealiased

`nonlinear(out)` (`:251`, `:649`, `:1001`, `:1437`):
1. Backward-transform each field and its `∂_r, ∂_θ, ∂_z` to the **padded**
   physical grid (`get_dealiased((dealias,…))`; **all** axes padded, including
   radial, default `dealias=1.5`).
2. Form the cylindrical products in physical space:
   - momentum `N_u = (u·∇)u − (b·∇)b` (Reynolds − Maxwell, with metric terms);
   - induction `N_b = −curl(ε)`, `ε = u×b` (the EMF), with the cylindrical curl
     including `(1/r)∂_θ` and `ε_θ/r` pieces in 3-D.
3. **Dealias**: forward-transform the product through the padded space (drops
   aliased high modes), copy the truncated coefficients into a clean base-space
   `Function`, transform back to the standard grid (`_dealias`/`_set_hat`).
4. Weak-projection `inner(test, ·)` onto each component's test space (note the
   `b_θ`, `b_z` tests live in the Robin/Neumann spaces, so the magnetic
   nonlinearity is projected consistently with its BCs).

**Performance / correctness invariants** (`:884-903`, `:1308-1325`): the radial
derivative projections are cached `Project` objects assembled once (a fresh
`project()` per call rebuilds the per-mode solver and dominates runtime,
~108 vs ~14 ms/step). Each cached `Project` captures `self.x[i]` / `_eps[k]`
**symbolically**, so `step()` must update state by **in-place** numpy
item-assignment into the composite array — never a rebind that allocates a new
array — or the caches would evaluate stale fields. The EMF-curl terms must be
materialized to numpy arrays **before** the `_eps` buffers are reused for the
momentum dealiasing (an explicit ordering invariant in the code).

### 3.5 Divergence handling

- `div(u) = 0` to roundoff via the coupled solve (no projection splitting).
- `div(b) = 0` is **not** enforced by a solve — the magnetic field is never
  pressure-projected. It is preserved because induction keeps `b` a curl, and is
  **monitored** (`divergences()`). Consequence for initial conditions: magnetic
  seeds must be solenoidal by construction (the random IC seeds only a toroidal
  `b_θ`, for which axisymmetric `div(b)` is identically zero; the 3-D random IC
  seeds a divergence-free Stokes-streamfunction velocity with `b=0`). A
  non-solenoidal velocity seed would inject `div(b) ~ dt B0 ∂_z div(u)` through
  the imposed-field induction.
- `divergence_linf` computes the radial and axial derivative terms by **separate
  projections** summed in physical space; combining them into one mixed
  `Dx`/sympy-`1/r` expression mis-evaluates in shenfun and reports a spurious
  O(amplitude) divergence (documented gotcha, `:396-411`).

### 3.6 Initial conditions and eigenmode seeding

- `set_perturbation` / `set_random`: small wall-vanishing seeds; the meridional
  flow is built from a Stokes stream function `ψ = g(r)·(…)` with
  `g = sin²(π(r−R1)/d)` (so `g=g'=0` at both walls) and
  `u_r = −(1/r)ψ_z, u_z = (1/r)ψ_r` → divergence-free *and* no-slip exactly.
- `seed_linear_eigenmode`: injects the **exact discrete eigenvector** from
  `TaylorCouetteLinear`/`TaylorCouetteMRI` at the matching `(Nr, family, ν,
  η, B0)` into Fourier mode `(m, k_z)`. Because the radial bases coincide, the
  measured DNS growth rate matches linear theory immediately (no transient) —
  the sharpest consistency check. For 3-D the complex eigenvector is evaluated
  **real/imag separately** before recombining (the radial-only eval space is
  real-dtype; a plain complex→real cast would drop `Im(q)` and break the
  poloidal/axial balance that makes the mode solenoidal). A seed mode the grid
  cannot resolve (`2|m| ≥ Ntheta`) raises `ValueError`.

### 3.7 `run()` time bookkeeping (gotcha)

`run(end_time)` **accumulates** `self._t`/`self._tstep` across successive calls
(lazy-init). So a growth-rate measured across two calls must use the elapsed
time `d2['t'] − d1['t']`, not `end_time` — the tests do exactly this.

### 3.8 IMEXRK companion (`taylor_couette_imexrk_dns.py`)

An additive alternative to the CNAB2 stepper of §3.3 that keeps the formulation,
spectral spaces, nonlinear terms and coupled velocity-pressure (saddle-point)
solve identical, but advances in time with the shenfun **IMEXRK** tableaux
(`IMEXRK111/222/443`, `--timestepper`) instead of Crank–Nicolson + AB2. Each RK
stage does its own coupled implicit solve (diffusion + base couplings + pressure
implicit; nonlinear advection/Lorentz/EMF explicit), with the stage LHS factor
cached by the stage coefficient γ.

Scope: **axisymmetric only** — `AxisymmetricTCIMEXRKDNS` (hydro, subclasses
`AxisymmetricTCDNS`) and `AxisymmetricMRIIMEXRKDNS` (conducting-wall MHD/MRI,
subclasses `AxisymmetricMRIDNS`); there is no 3-D (`m≠0`) IMEXRK class. It
supports the same `--seed-linear` eigenmode seeding and reports the measured
growth, so it both cross-checks the CNAB2 result and matches the PCF DNS
integrator (the shared `imex_tableau` from `taylor_couette_imexrk.py`). Use
CNAB2 (§3) for 3-D / non-axisymmetric runs. Neither nonlinear DNS does insulating
walls — those are a linear-analysis-only capability (§2.6).

---

## 4. Tests

### `test_taylor_couette.py` (linear solvers)
- **Base flow / Rayleigh** (algebraic): wall rotation recovery, `2Ω+rΩ'=2a`,
  Rayleigh line `μ=η²`, Keplerian Rayleigh-stable with `q≈3/2`.
- **Hydro stability:** exchange of stabilities (`m=0` onset is stationary,
  `Im(s)≈0`); critical Reynolds `Re_c=68.19`, `a_c≈3.16` at `η=0.5` (Fasel &
  Booz 1984); Keplerian hydro null (stable); `m→−m` spectral mirror symmetry.
- **Ideal local MRI:** `s_max=0.75Ω`, `(k v_A)²=0.9375Ω²`, band/cutoff
  (Balbus & Hawley 1991).
- **Global MHD/MRI:** `B0=0` Keplerian stable; leading mode `div(b)=0` to 1e-8
  (primitive **and** flux formulations); conducting `b_θ` Robin BC satisfied;
  MRI-unstable with axial field; flux formulation reproduces conducting
  eigenvalues; insulating mode solenoidal by construction; insulating onset
  easier than conducting (`Rm_min` ordering); `m≠0` and `k_z=0` insulating
  guards raise.

### `test_taylor_couette_dns.py` (nonlinear DNS)
- **Fixed point / decay:** zero perturbation stays exactly zero; Rayleigh-stable
  seed decays.
- **Linear-growth consistency** (sharpest): seeded eigenmode growth matches the
  eigensolver to `<1e-3` relative — for axisymmetric, 3-D `m=0`, and genuinely
  non-axisymmetric travelling-wave `m=1,2` (`Im(s)≠0`) modes, hydro and MRI.
- **Incompressibility:** `div(u)` at roundoff (coupled solve, no splitting);
  weak continuity exact, pointwise residual converges spectrally with `Nr`.
- **MHD specifics:** random magnetic seed solenoidal; Alfvénic cancellation
  (`u=b` ⇒ Reynolds/Maxwell force and EMF curl vanish — sign check on
  `(u·∇)u−(b·∇)b`); cylindrical curvature signs (`−u_θ²/r` vs `+b_θ²/r`); 3-D
  EMF-curl metric signs; div-free random IC keeps `div(b)` small through a run.
- **Saturation (slow):** supercritical hydro grows then saturates to a steady
  Taylor-vortex state; seeded MRI amplifies the field by orders of magnitude
  then saturates (axisymmetric and non-axisymmetric), `div(b)` bounded.

### `test_thin_gap_comparison.py` (cross-checks + apples-to-apples)
- **Collocation ↔ Galerkin agreement:** `TaylorCouetteCollocationLinear` matches
  `TaylorCouetteLinear` (hydro) and `TaylorCouetteMRI` (MHD conducting and
  insulating) leading eigenvalues; insulating guards (`m≠0`, `kz=0`) raise.
- **Shared IMEXRK stepper:** `TaylorCouetteIMEXRKLinearStepper` (hydro / MHD /
  insulating) recovers the seeded eigenmode growth; integer-step guard.
- **Apples-to-apples PCF↔TC:** eigenvalue set-matching; shearing-limit hydro/MHD
  match in the local frame; plane-limit curvature convergence; Rayleigh flag;
  collocation↔collocation MHD non-modal agreement; DNS-style growth agreement.

---

## 5. Thin-gap apples-to-apples comparison

`thin_gap_compare.py` places the plane-Couette (`pcf_*`) and Taylor-Couette
linear operators on a common footing and compares them at three levels with
**matched discretisation**. `thin_gap_common.py` (no shenfun) supplies the
scalings (`ShearScales`) and the annulus maps
(`annulus_for_plane_couette_limit`, `annulus_for_shearing_box_limit`) that turn
a plane/shearing-box mid-gap state into a circular annulus; the TC eigenvalues
are reported in the **local frame** rotating at the mid-gap `Ω0` (analytic
Doppler shift `+i m Ω0`).

1. **Eigenvalues** — matched as *sets* (`_linear_analysis.match_eigenvalues`,
   one-to-one Hungarian assignment), robust to the `±`-frequency tie-break of a
   near-conjugate-symmetric spectrum and to the PCF `+ky` vs TC `+m` orientation.
2. **Non-modal growth** (`--nonmodal`) — only apples-to-apples *within* a
   discretisation: Galerkin↔Galerkin (modal Gram norm) and
   collocation↔collocation (nodal quadrature norm). MHD magnetic/total transient
   growth is reliable only in the collocation pairing (§2.8); for the Galerkin
   pairing use `--energy kinetic`.
3. **DNS-style** (`--dns`) — both geometries advanced by the same IMEXRK
   integrator (§2.9 + `pcf_imexrk_linear.py`), comparing measured eigenmode growth.

`--limit` defaults to **`plane`**, the *singular* non-rotating limit: the annulus
is centrifugally (Taylor) unstable at any finite curvature, so the residual is
`O(curvature)` and the tool prints a Rayleigh-stability warning. Use
`--limit shearing` (Rayleigh-stable) as the natural test, or a very small
`--curvature` (`1e-4` or below) for the plane limit.

---

## 6. Running

```bash
# from demo/, in the shenfun conda environment
# linear stability (kz scan or single kz):
python taylor_couette_linear.py --R1 1 --R2 2 --Omega1 1 --Omega2 0 --m 0
python taylor_couette_mri.py --magnetic-bc conducting --B0 0.1 --eta-mag 1e-3 --local-check
# alternative discretisation / time stepper:
python taylor_couette_collocation.py --m 0 --kz 3.16 --nu 1e-3        # dense collocation
python taylor_couette_imexrk.py --m 0 --kz 3.0 --end-time 0.05        # DNS-style IMEXRK (linear)

# nonlinear DNS (axisymmetric default; --Ntheta>0 for full 3D; --mhd for MRI):
python taylor_couette_dns.py --Omega2 0 --nu 1e-2 --Nr 48 --Nz 32 --end-time 2
python taylor_couette_dns.py --mhd --Ntheta 8 --B0 0.1 --eta-mag 1e-3 --m 1
python taylor_couette_imexrk_dns.py --seed-linear --kz-mode 1 --end-time 0.1   # axisym IMEXRK companion

# apples-to-apples PCF <-> TC comparison:
python thin_gap_compare.py --limit shearing --mhd --B0 0.05 --nonmodal --dns
```

Tests: `conda run -n shenfun pytest -q demo/test_taylor_couette.py
demo/test_taylor_couette_dns.py demo/test_thin_gap_comparison.py` (slow
benchmarks are marked `@pytest.mark.slow`).
