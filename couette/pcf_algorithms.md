# Plane-Couette (MHD) solvers — algorithm & numerics reference

Audit-oriented description of the numerical methods used by the Cartesian
plane-Couette flow (PCF) **nonlinear DNS** solvers in this directory.  Sections
1-12 describe those KMM solvers; the **linear-stability and apples-to-apples
comparison layer** (added later) is in [§13](#13-linear-stability-and-apples-to-apples-comparison-layer).

| file | class | physics | base class |
|------|-------|---------|------------|
| `pcf_fluctuations_corrected.py` | `PlaneCouetteFluctuation` | hydrodynamic PCF (fluctuation form) | `ChannelFlow.KMM` |
| `pcf_fluctuations_divV.py` | `PlaneCouetteFluctuation` | hydro PCF variant, extra `div(u)` diagnostics | `KMM` |
| `pcf_mhd_divfree.py` | `PlaneCouetteMHDDivFree` | resistive MHD, divergence-free `B` via vector potential | `KMM` |
| `pcf_mhd_mri_shearpy.py` | `PlaneCouetteMRIShearpy` | MHD + rotation/shear + imposed net field (MRI) | `PlaneCouetteMHDDivFree` |
| `test_pcf_mhd_divfree.py` | — | pytest: `div(B)`/`div(u)` at roundoff | — |
| `test_pcf_mhd_mri_shearpy.py` | — | pytest: net-flux MRI exponential growth | — |

> **Multiple approaches.**  PCF is solved/analysed here several complementary
> ways.  *Nonlinear DNS:* the KMM velocity-vorticity solvers in this table
> (Sections 1-12).  *Linear stability* (eigenvalues / optimal transient growth):
> a dense Chebyshev-collocation operator (`_pcf_linear.PlaneCouetteLinear`) **and**
> a shenfun-Galerkin operator (`pcf_galerkin_linear.PlaneCouetteGalerkinLinear`)
> -- same primitive variables, different discretisation; the DNS scripts also
> expose the collocation operator through a `--linear {eigs,nonmodal}` flag.
> *DNS-style linear time stepping:* `pcf_imexrk_linear.PlaneCouetteIMEXRKLinearStepper`.
> All of these, plus the thin-gap PCF↔Taylor-Couette comparison, are covered in
> [§13](#13-linear-stability-and-apples-to-apples-comparison-layer).  See
> `README_Couette.md` ("Which approach to use") for a decision table.

The three KMM solvers form an inheritance chain
(`KMM → PlaneCouetteFluctuation`, and `KMM → PlaneCouetteMHDDivFree →
PlaneCouetteMRIShearpy`), so the velocity discretization and time integration
are **identical** across all three; the MHD solvers only add a magnetic field
and extra source terms. This document therefore describes the shared `KMM`
machinery once, then the deltas added by each subclass.

Companion physics notes (setup, validation, history) live in
`pcf_mhd_divfree_notes.md` and `pcf_mhd_mri_notes.md`; this file is about *how
the equations are discretized and stepped*.

---

## 1. Geometry, coordinates, and field layout

All three solvers use the Kim–Moser–Moin (KMM) channel geometry
(Kim, Moser & Moin 1987, *JFM* 177:133):

| index | coordinate | direction | numerics |
|------:|:----------:|-----------|----------|
| 0 | `x` | wall-normal (shear-gradient) | Chebyshev or Legendre, no-slip walls at `x = ±1` (or `±Lx/2`) |
| 1 | `y` | streamwise (wall-motion / azimuthal) | Fourier, periodic |
| 2 | `z` | spanwise (vertical / rotation axis in the MRI case) | Fourier, periodic |

Default domain `((-1, 1), (0, 4π), (0, 2π))`. The base (laminar) flow is
carried analytically and only the **fluctuation** `u'` is time-stepped:

- hydro / divfree: `U_b(x) = U_wall · x · e_y`, constant shear `dU_b/dx = U_wall`;
- MRI (shearpy): `U_b(x) = −S · x · e_y`, constant shear `dU_b/dx = −S`.

The base flow is never inserted into the spectral solution vector; it enters
only through the convection term (Section 5), so the laminar state is the exact
fixed point `u' = 0` and the walls are homogeneous (`u' = 0` there).

---

## 2. Spectral spaces (shenfun)

### 2.1 One-dimensional bases

Built in `KMM.__init__` (`ChannelFlow.py:82-97`). `family` is `'C'` (Chebyshev)
or `'L'` (Legendre); `N = (Nx, Ny, Nz)` are the physical quadrature counts.

```python
B0 = FunctionSpace(N[0], family, bc=(0, 0, 0, 0), domain=domain[0])  # wall-normal velocity
D0 = FunctionSpace(N[0], family, bc=(0, 0),       domain=domain[0])  # streamwise/spanwise velocity
C0 = FunctionSpace(N[0], family,                  domain=domain[0])  # orthogonal (no BC)
F1 = FunctionSpace(N[1], 'F', dtype='D', domain=domain[1])           # Fourier y (complex)
F2 = FunctionSpace(N[2], 'F', dtype='d', domain=domain[2])           # Fourier z (real)
```

The `bc=(...)` tuple is shenfun's **composite (Galerkin) boundary basis**: the
basis functions are linear combinations of orthogonal polynomials constructed so
that *every* basis function satisfies the homogeneous BCs exactly. Hence the BCs
are built into the trial/test space and never appear as separate tau rows.

- `bc=(0, 0)` → Dirichlet `φ(±1) = 0`. Dimension `N − 2`.
- `bc=(0, 0, 0, 0)` → clamped/biharmonic `φ(±1) = φ'(±1) = 0`. Dimension `N − 4`.
  This is the space for the wall-normal velocity, whose KMM equation is 4th
  order; the two extra conditions are no-penetration *and* the
  continuity-implied `∂u_x/∂x = 0` at the walls.
- The Fourier `dtype='D'` (complex) on `y` and `dtype='d'` (real) on `z` is the
  standard real-field FFT layout: the last transformed axis (`z`) is halved by
  Hermitian symmetry.

### 2.2 Tensor-product and vector spaces

```python
TB = TensorProductSpace(comm, (B0, F1, F2), ...)   # wall-normal velocity component
TD = TensorProductSpace(comm, (D0, F1, F2), ...)   # Dirichlet components / general TD
TC = TensorProductSpace(comm, (C0, F1, F2), ...)   # unconstrained (derivative range)
BD = VectorSpace([TB, TD, TD])                      # velocity (u_x, u_y, u_z)
CD = VectorSpace(TD)            # = [TD, TD, TD]     # convection / vector potential
CC = VectorSpace([TD, TC, TC])                      # curl space
TDp = TD.get_dealiased(padding_factor)              # padded space (Section 6)
```

`collapse_fourier=False, slab=True` keep the data as a 3-D slab decomposition for
MPI (the wall-normal `x` axis stays local on every rank, which the diagnostics
and the `(0,0)`-mode solve rely on).

**Why three radial spaces.** A wall-normal derivative maps a Dirichlet field out
of its space: `∂_x` of a `D0` (Dirichlet) field no longer satisfies `φ(±1)=0`,
so its natural home is the unconstrained `C0`. Putting curl/derivative outputs
in `TC` rather than `TD` is what makes the discrete vector-calculus identities
(`div curl = 0`) hold — central to the divergence-free MHD scheme (Section 8).

---

## 3. Velocity formulation (the KMM `v`–`g` system)

The incompressible momentum equation is reduced to two scalar evolution
equations plus a recovery step, eliminating pressure exactly. This is the
classic wall-normal-velocity / wall-normal-vorticity formulation.

**Evolved scalars** (`ChannelFlow.py:147-164`):

1. Wall-normal velocity `u_x` via a **4th-order (biharmonic)** equation whose
   time-derivative variable is `∇²u_x`:

   ```
   ∂/∂t (∇²u_x) = ν ∇⁴ u_x
                  + ∂²N_y/∂x∂y + ∂²N_z/∂x∂z − ∂²N_x/∂y² − ∂²N_x/∂z²
   ```
   Test function `v ∈ TB`; implicit operator `ν·div(grad(·))` acting on the
   `div(grad(u_x))` variable (so the implicit solve is biharmonic).

2. Wall-normal vorticity `g = (∇×u)_x = ∂_y u_z − ∂_z u_y` via a **Helmholtz**
   equation:
   ```
   ∂g/∂t = ν ∇² g + ∂N_y/∂z − ∂N_z/∂y
   ```
   Stored as `g_ = curl[0]`; initialized `g_ = i k_y u_z − i k_z u_y`.

Here `N = (N_x, N_y, N_z) = H_` is the (negative) nonlinear convection vector
(Section 5).

**Recovery of `u_y, u_z`** (`compute_vw`, `ChannelFlow.py:227-251`). For every
Fourier mode `(k_y, k_z) ≠ (0,0)`, continuity `∇·u = 0` plus the definition of
`g` give an algebraic 2×2 solve:

```
f = ∂u_x/∂x
u_y = i (k_y f + k_z g) / (k_y² + k_z²)
u_z = i (k_z f − k_y g) / (k_y² + k_z²)
```

The single `(0,0)` mode has no such constraint (the horizontal mean is
divergence-free for any `u_y(x), u_z(x)`), so its `u_y, u_z` are advanced by two
**separate 1-D Helmholtz PDEs** on `D00` (a 1-D Dirichlet basis), with the mean
pressure-gradient source `dpdy` (= 0 for PCF):

```
∂v/∂t = ν ∂²v/∂x² − N_y − dp/dy
∂w/∂t = ν ∂²w/∂x² − N_z
```

This `(0,0)` solve runs only on MPI rank 0 (`if comm.Get_rank() == 0`).

---

## 4. Implicit/explicit split and time integration

### 4.1 Operator split

Every PDE object is built as `PDE(test, u, linear_op, rhs, dt, solver, …)`
where `linear_op = lambda f: ν·div(grad(f))` (or `η·div(grad(f))` for the
magnetic potential). The split is:

- **Implicit (linear, stiff):** the viscous operator `ν∇²`/`ν∇⁴` and the
  resistive operator `η∇²`. Treated implicitly every stage → no viscous CFL
  limit.
- **Explicit (nonlinear, non-stiff):** the convection `N`, the Lorentz force,
  the rotation/shear source terms, and the induction EMF `U×B`. Assembled in
  physical space (Section 5) and supplied as the PDE right-hand side.

This is a standard IMEX (implicit–explicit) split: the only matrices that must
be inverted are the constant-coefficient Helmholtz/biharmonic operators, which
are pre-factored once per Fourier mode in `assemble()`.

### 4.2 Time stepper

`self.PDE = globals().get(timestepper)` selects a shenfun IMEX additive
Runge–Kutta integrator by name. The base `KMM` constructor default is
`IMEXRK3`, but the PCF subclasses here set **`IMEXRK222`** (the 2-stage,
2nd-order, L-stable Ascher–Ruuth–Spiteri (2,2,2) scheme) as their default; the
`--timestepper` CLI flag overrides it (`IMEXRK111/222/3/443` accepted). The
driver loop is
(`pcf_mhd_divfree.py:458-495`, mirroring `KMM.solve`):

```
assemble()                               # factor implicit operators once
while t < end_time:
    for rk in range(PDE.steps()):        # RK stages
        prepare_step(rk)                 # -> convection(): build N, Lorentz, EMF
        for eq in pdes:    eq.compute_rhs(rk)
        for eq in pdesA:   eq.compute_rhs(rk)     # (MHD only)
        for eq in pdes:    eq.solve_step(rk)      # implicit solves: u_x, g
        compute_vw(rk)                            # recover u_y, u_z
        for eq in pdesA:   eq.solve_step(rk)      # (MHD only) advance A
    t += dt; tstep += 1
    update(...); checkpoint.update(...)
```

`PDE.steps()` returns the number of RK stages. Each stage does its own implicit
solve; the explicit terms are re-evaluated per stage from the stage state.

### 4.3 Linear solvers

Per-Fourier-mode 1-D radial solves (`ChannelFlow.py:144-145`):

- **Chebyshev:** dense radial operators → tailored fast solvers
  `chebyshev.la.Biharmonic` (for `u_x`) and `chebyshev.la.Helmholtz`
  (for `g`, `v`, `w`, and the vector potential).
- **Legendre:** the Dirichlet/biharmonic stiffness/mass matrices are sparse
  (few nonzero diagonals) → the generic banded solver `la.SolverGeneric1ND`.

The MHD induction solver is chosen the same way
(`pcf_mhd_divfree.py:160`): `chebyshev.la.Helmholtz` or `la.SolverGeneric1ND`.

---

## 5. Nonlinear term (pseudo-spectral, dealiased)

Implemented in `convection()` and stored in `H_`; only `conv=0` (the
**convective / advective form** `u·∇u`) is implemented (`conv=1` rotational form
is rejected). The procedure (`pcf_mhd_divfree.py:295-349`):

1. Inverse-transform `u'` and all nine velocity gradients to the **padded**
   physical grid via `backward(padding_factor=…)`.
2. Form the products in physical space:
   ```
   N_x = u'·∇u'_x ,  N_y = u'·∇u'_y ,  N_z = u'·∇u'_z
   ```
3. Add base-flow advection and shear production (base flow `U_b(x) e_y`):
   ```
   N_x += U_b ∂_y u'_x
   N_y += U_b ∂_y u'_y + u'_x · dU_b/dx     (shear production)
   N_z += U_b ∂_y u'_z
   ```
4. Forward-transform back to spectral space on the standard grid
   (`TDp.forward`), then zero the Nyquist mode with `mask_nyquist`.

The result is stored as `H_ = N`, and the KMM `u_x`/`g` equations take the
curl-of-`N` combinations shown in Section 3. The MHD subclasses add the Lorentz
force and the vector-potential EMF inside the same routine (Sections 7–8).

---

## 6. Dealiasing (3/2 rule)

`padding_factor = (1, 1.5, 1.5)`. The two periodic directions are padded by
3/2 (`get_dealiased`), so quadratic products are aliasing-free by Orszag's
3/2 rule; the **wall-normal `x` direction is not padded** (factor `1`) because
the products are evaluated on the Chebyshev/Legendre Gauss grid where the
polynomial nonlinear interaction is handled by the modal truncation plus
Nyquist masking rather than zero-padding.

Mechanics:
- `u_.backward(padding_factor=self.padding_factor)` evaluates on the enlarged
  grid `TDp`;
- products are formed there;
- `self.TDp.forward(product, out)` transforms back and truncates to the
  resolved modes;
- `mask_nyquist(self.mask)` sets the (unresolved-sign) Nyquist Fourier mode to
  zero after every nonlinear/curl evaluation.

---

## 7. Magnetic field — `pcf_mhd_divfree.py`

Adds resistive MHD to the KMM velocity solver while guaranteeing a discretely
solenoidal magnetic field. **The magnetic field is never advanced directly.**
Instead a magnetic **vector potential** `A` is integrated and `B = ∇×A` is
recomputed whenever needed, so `div(B) = div(curl(A)) = 0` is a compatible-space
identity rather than a quantity to be controlled.

### 7.1 Compatible spaces

```
A in CD = [TD, TD, TD]   (A_x = A_y = A_z = 0 at the walls)
B = curl(A)  in CC = [TD, TC, TC]
J = curl(B)  in JS = [TC, TD, TD]
```
The space assignment is forced by where each derivative lands:
- `B_x = ∂_y A_z − ∂_z A_y` (Fourier derivatives only) → `TD`;
- `B_y = ∂_z A_x − ∂_x A_z`, `B_z = ∂_x A_y − ∂_y A_x` (a wall-normal
  derivative of a Dirichlet field) → `TC`.

These are wired with `Project` objects (`pcf_mhd_divfree.py:141-151`):
`projBx/projBy/projBz` recompute `B`, and `curlb0/1/2` recompute `J`. `A_y = A_z
= 0` at the walls gives `B_x = 0` at the walls (zero normal magnetic flux).

### 7.2 Induction equation

```
∂A/∂t = U×B + η ∇²A ,   η = U_wall / Rm
```
Advanced as **three** scalar PDEs (`pdesA`, one per component, test function in
`TD`) with the resistive operator `η·div(grad(·))` treated **implicitly** by the
same IMEX RK stepper, and the EMF `U×B` (using the *total* velocity `U = U_b +
u'`) supplied explicitly via `HA_`, evaluated pseudo-spectrally on the padded
grid (`pcf_mhd_divfree.py:340-349`).

### 7.3 Lorentz force

`J×B` is computed in physical space on the padded grid and **subtracted** from
the velocity nonlinear storage (`N ← N − J×B`, `pcf_mhd_divfree.py:325-333`),
because the KMM velocity equations apply `−H` after the implicit solve. `B` and
`J` are refreshed from `A` (`update_B_from_A`, `update_J_from_B`) before every
use, so no componentwise magnetic update can spoil `div(B)`.

### 7.4 Resistivity treatment

Implicit, identical in spirit to the viscous term: `η∇²A` is the linear operator
inside each `pdesA` PDE; only the EMF is explicit. `Rm` (magnetic Reynolds) sets
`η = U_wall/Rm`; default `Rm = Re`, i.e. magnetic Prandtl `Pm = 1`.

---

## 8. MRI extension — `pcf_mhd_mri_shearpy.py`

Wall-bounded plane-Couette analogue of a local shearing box (subclass of the
divfree solver). Same discretization; the additions are:

1. **Shearing base flow** `U_b(x) = −S x e_y` (replaces the Couette profile;
   `pcf_mhd_mri_shearpy.py:102-105`).
2. **Rotation (Coriolis) source** `2Ω`, added inside `convection()`:
   `N_x += −2Ω u'_y`, `N_y += +2Ω u'_x`. Because KMM applies `−H`, these realize
   the shearing-box source terms `+2Ω u'_y` (x-momentum) and `−2Ω u'_x`
   (y-momentum); combined with the shear-production term they give the standard
   epicyclic coupling (`pcf_mhd_mri_shearpy.py:347-348`).
3. **Imposed uniform field** `B0 = (0, b_y, b_z)` carried **separately** from
   `curl(A)`. The Lorentz force and the induction EMF use the *total* field
   `B_tot = B0 + curl(A)`. Because `B0` is uniform it carries no current
   (`J = curl(curl A)` only) but still participates in `J×B` and `U×B`. Keeping
   it out of `A` preserves the `div(curl A)=0` invariant while reproducing
   net-flux MRI (default `b_z = 0.025`).

Derived control parameters: shear `q = S/Ω`, epicyclic `κ² = 2Ω(2Ω − S)`,
Lundquist/Alfvén speed `v_A = b_z`. The `−S B_x` azimuthal-field induction is
*not* added as a separate term — it arises automatically from `B·∇U_b` inside
the EMF `U_b×B`.

**Diagnostics** (`mean_profiles`, `channel_amplitude`, `parasite_diagnostic`,
Reynolds/Maxwell stresses, transport `α`): MPI-safe `y,z` reductions via
`allreduce`; the `z`-FFT channel/parasite projections assume `(y,z)` are local
on each rank (serial or `x`-slab decomposition).

---

## 9. Divergence control — summary

| quantity | mechanism | measured value |
|----------|-----------|----------------|
| `div(u')` | KMM `v`–`g` recovery enforces continuity per Fourier mode | ~1e-12 (roundoff) |
| `div(B)` | `B = curl(A)` in compatible spaces `[TD,TC,TC]` | ~1e-12 (roundoff) |

`div(u)` is checked with the `divu` projection; `div(B)` with the `divb`
projection of `div(b_)`. Both are reported relative to the RMS field amplitude
in the diagnostics (`divu_rel`, `divb_rel`).

---

## 10. Hydrodynamic solver — `pcf_fluctuations_corrected.py`

Same `KMM` velocity engine, no magnetic field. Differences worth noting for an
audit:

- Backend selection (`_select_backend_and_family`): prefers the Numba-optimized
  shenfun kernels; if Chebyshev biharmonic assembly is unavailable in the active
  backend it **falls back from Chebyshev to Legendre** automatically.
- `Re = U_wall·h/ν` with half-gap `h = 1` (domain `x ∈ [−1,1]`).
- Rich offline analysis: plane-averaged mean/shear profiles, RMS statistics,
  `k_y`/`k_z` energy spectra (`_compute_spectra`, folded & `allreduce`-d across
  ranks), and self-sustaining-process (SSP) streak/roll diagnostics. These are
  diagnostic only and do not affect the time integration.

---

## 11. Tests

### `test_pcf_mhd_divfree.py`
Runs the solver in a subprocess (clean process per case) and parses a
`JSON_DIAG` line. Two cases:
- tiny Legendre `8³`, and Chebyshev `16³`, both with velocity + magnetic seeds;
- `--assert-every-step` with `--max-divb-l2 1e-12 --max-divu-l2 1e-12`.

Asserts `Emag > 0` (field actually present) and both divergences `< 1e-12`,
i.e. solenoidality is held to **roundoff every step**, in both basis families —
the regression guard on the compatible-space construction.

### `test_pcf_mhd_mri_shearpy.py`
- A tiny sanity run (checks `q = S/Ω`, mean shear `= −S`, `B0z`, divergences).
- `test_netflux_mri_magnetic_energy_grows`: integrates the net-flux MRI case to
  `t = 3`, samples the perturbation magnetic energy, and asserts it **grows
  exponentially** (`E_mag(t=3) > 2·E_mag(t=1)`, expected ~7×), grows
  monotonically once the axisymmetric channel mode dominates, and that
  `div(B) < 1e-10` throughout the growth. This is the discriminating test that
  the solver reproduces real MRI rather than merely not crashing.

Both drive the solver directly via the in-process `run(...)`/class API rather
than the CLI, to sample intermediate state.

---

## 12. Running

```bash
# from demo/, in the shenfun conda environment
python pcf_mhd_divfree.py --family L --nx 16 --ny 16 --nz 16 --dt 1e-3 --end-time 0.1
python pcf_mhd_mri_shearpy.py --bz 0.025 --shear 1.0 --omega 0.6667 --end-time 3
python pcf_fluctuations_corrected.py --no-save-plots
```

Common flags: `--family {C,L}`, `--nx/--ny/--nz`, `--Re`, `--Rm`, `--dt`,
`--timestepper`, `--perturbation-amplitude`, `--magnetic-amplitude`,
`--max-divb-l2/--max-divu-l2` with `--assert-every-step`. MPI: `mpirun -np K
python …` (wall-normal `x` stays local; the `(0,0)`-mode solve and the
spectral/parasite diagnostics assume an `x`-slab decomposition).

All three DNS CLIs also accept `--linear {dns,eigs,nonmodal}` (default `dns` =
normal time integration). With `eigs`/`nonmodal` the script does **not** step in
time; it builds a dense `_pcf_linear.PlaneCouetteLinear` at the matching
parameters and prints the leading spectrum / optimal transient growth — see §13.

---

## 13. Linear-stability and apples-to-apples comparison layer

Sections 1-12 cover the nonlinear KMM DNS. The linear analysis is done by
separate, lighter operators that share a common backend with the Taylor-Couette
linear solvers — by design, so the two geometries can be compared term-for-term.

### 13.1 Two PCF linear operators (collocation and Galerkin)

Both solve the primitive-variable perturbation problem for `U(x) = Uoffset +
Uprime·x` with `q(x)·exp(s t + i k_y y + i k_z z)` (requires `k_y²+k_z² > 0`),
returning the generalized spectrum `L q = s M q` and energy-norm optimal
transient growth. Velocity is no-slip Dirichlet; pressure is a Lagrange
multiplier; MHD adds primitive `b=(b_x,b_y,b_z)` with a **magnetic-pressure**
multiplier `phi` enforcing `div(b)=0`, and an imposed uniform `B0=(0,b_y,b_z)`.

| | `_pcf_linear.PlaneCouetteLinear` | `pcf_galerkin_linear.PlaneCouetteGalerkinLinear` |
|---|---|---|
| discretisation | dense **Chebyshev-collocation** (`cheb_lobatto`) | **shenfun-Galerkin** (`inner`/`Dx`, sympy `1/r`-free Cartesian coeffs) |
| dependencies | NumPy + SciPy only (no shenfun) | shenfun + sympy |
| energy norm | nodal quadrature (`diag(weights)`) | modal Gram matrix (`inner`) |
| BC walls (`magnetic_bc`) | `conducting` (`b_x=0`, Neumann `b_y'=b_z'=0`) / `dirichlet` (`b=0`, diagnostic) | same |
| API | `.couette(...)`, `.shearpy(...)`, `.eigs(ky,kz)`, `.growth_rate`, `.nonmodal_growth(...,energy=)` | `.couette(...)`, `.shearbox(...)`, `.eigs`, `.nonmodal_growth`, `.assemble_parts` |

They agree to spectral accuracy where they overlap; the *non-modal* gain differs
by the norm discretisation (Gram vs nodal), so transient growth is only
apples-to-apples **within** one discretisation (Galerkin↔Galerkin or
collocation↔collocation). The collocation operator is what the DNS `--linear`
flag wraps.

### 13.2 Shared backend (`_linear_analysis.py`)

`FINITE_CAP = 1e8` (drops the spurious infinite eigenvalues from the singular-`M`
constraint rows; insensitive 1e6-1e12); `finite_eigensystem` (finite filter +
sort by `Re(s)`); `physical_eigensystem` (also rejects multiplier/gauge vectors
below the `sqrt(machine epsilon)` physical-energy fraction before modal seeding);
`transient_growth_from_eigs` (finite modal expansion → energy-norm
propagator SVD); `parse_times`; `match_eigenvalues` (one-to-one Hungarian
set-match of two spectra — orientation/tie-break robust); and the IMEXRK
`imex_tableau` / `imexrk_step` shared with §13.3. Pure NumPy/SciPy.

### 13.3 DNS-style linear time stepping (`pcf_imexrk_linear.py`)

`PlaneCouetteIMEXRKLinearStepper` advances the **Galerkin** operator with the
same IMEXRK tableaux as the PCF DNS (`IMEXRK111/222/443`, default 222), using the
shared `imexrk_step`. Split `diffusion` (default: viscous/resistive diffusion +
the pressure/continuity and magnetic-pressure/solenoidal saddle-point rows
implicit; advection/coupling explicit) mirrors the DNS IMEX split; `full` is a
stiff implicit reference. The constraint (zero-mass) blocks are pinned per stage:
`pblocks=[3,7]` for MHD (`p` and `phi`) or `[3]` for hydro. Seeding the leading
eigenmode and integrating recovers `Re(s)` to ~1e-8 — this is the only path that
exercises the time integrator itself, and the PCF side of `thin_gap_compare --dns`.

### 13.4 Thin-gap apples-to-apples comparison

`thin_gap_common.py` (no shenfun) maps a plane/shearing-box state to a circular
annulus (`ShearScales`, `annulus_for_plane_couette_limit`,
`annulus_for_shearing_box_limit`); `thin_gap_compare.py` then compares the PCF
operators above against the Taylor-Couette operators at three levels —
eigenvalues (set-matched), `--nonmodal` (Galerkin↔Galerkin and
collocation↔collocation pairings), and `--dns` (shared IMEXRK). The non-rotating
`--limit plane` is a **singular** limit (the annulus is Rayleigh/Taylor-unstable
at finite curvature); `--limit shearing` is the natural test. See
`taylor_couette_algorithms.md` §5 and `README_Couette.md`.

### 13.5 Linear-layer tests

`test_couette_linear.py` (operator unit tests: half-gap reference scaling,
eig/non-modal sanity, the synthetic transient-growth check) and
`test_thin_gap_comparison.py` (the apples-to-apples regressions: set-matching,
plane-limit curvature convergence, Rayleigh flag, PCF IMEXRK eigenmode growth,
DNS-style PCF↔TC agreement, the two non-modal pairings).
