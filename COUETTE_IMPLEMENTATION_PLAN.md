# Implementation Plan — Plane Couette & Taylor–Couette flows in `jaxfun`

**Status:** design / ready for implementation
**Audience:** junior developers and coding agents. This document is self-contained: you should not need to already understand spectral Galerkin methods, `shenfun`, or the physics to start a task. Read §1–§4 once, then pick a milestone in §7.
**Goal:** make the seven `shenfun` reference scripts in `couette/` run *inside* `jaxfun` — i.e. on top of `jaxfun`'s differentiable, JAX-backed, multi-backend (CPU/GPU/TPU) Galerkin core — reproducing the `shenfun` results to numerical parity.

Paths in this document are relative to the workspace root `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/`:
- `shenfun/` — the reference implementation (CPU/NumPy/MPI). **Read-only ground truth.**
- `couette/` — the seven working `shenfun` scripts we must reproduce.
- `jaxfun/` — the JAX reimplementation we are extending. **All new code goes here.** (Active work is on branch `couette-jax-implementation` in `fork_jaxfun/`.)

---

> ## ⚠️ STATUS UPDATE — 2026-06-01 (READ FIRST)
>
> Implementation is **underway** (~14k lines on branch `couette-jax-implementation`). A full audit is in **Part II (§11–§12)** at the end of this document. Parts I (§1–§10) remain the architectural reference; Part II records what is built and adds the missing scope. Headlines:
>
> - **All seven scripts have `*_jax.py` ports**; **22/22 jaxfun unit tests + 7 solver smoke-tests pass** (CPU, float64). Implemented: IMEX-RK integrators (T3.2/T3.3), dense `eig` (T2.4), `CoupledSpace` + truncated pressure (T6.1), `integrate()` (T1.1), ragged vectors, `get_dealiased`, `mask_nyquist`, `Dx`, the BC adapter (T0.5), and runnable KMM-PCF + axisymmetric-hydro-TC-DNS solvers.
> - **⚠️ NOTHING is validated against `shenfun`.** No test imports `shenfun`; the six `matches_shenfun_reference` tests compare to **self-captured literals** and are `@skipif(not x64)` — and **x64 is OFF by default**, so plain `uv run pytest` **SKIPS every parity test and still reports green**. Read every "done/partial" below as *internally self-consistent, NOT parity-validated*.
> - **⚠️ Taylor–Couette DNS = 1 of 4 quadrants only** (axisymmetric-hydro). 3D-hydro, axisymmetric-MHD, and 3D-MHD are absent in jax — the `shenfun` reference has all four classes. New milestones **M9–M13 (§12)** add them.
> - **⚠️ `IMEXRK3` (the documented KMM default) is rejected by the KMM driver** (`channelflow_kmm.py` only accepts `PDEIMEXRK` subclasses). The coupled multi-equation driver (T3.4), cached `Project` (T1.2), named Helmholtz/Biharmonic solvers (T2.2), and CNAB2 (T6.3) are **inlined in examples, not reusable library modules**.
> - **Decisions taken since Part I:** the **full-complex Fourier** option (T0.3 option B) is in use (see `docs/couette_fourier_layout.md`); `couette/_linear_analysis.py` and `couette/_pcf_linear.py` now **exist** (a prior "missing file" note is stale) and ship a modal+non-modal stability layer that has **no jax port and was unscoped** — added as **M8 (§12)**.
>
> **Do §12 "M0b" (float64-at-import + a live `shenfun` parity harness) FIRST.** Until it lands, no parity claim anywhere in this document is meaningful.

---

## 0. How to use this document

- **§1** lists the seven target scripts and what "done" means.
- **§2** explains the three solver architectures (KMM channel flow; cylindrical block-coupled DNS; linear-stability eigenproblems) in enough depth to implement them. Do not skip this.
- **§3** is the inventory of what `jaxfun` *already* provides. **Read it before writing anything** — most of the hard spectral machinery already exists; we are adding a thin, well-defined layer.
- **§4** is the set of engineering rules (JAX idioms) every task must follow.
- **§5** is the master gap table (41 items).
- **§6** is the dependency graph and milestone map.
- **§7** is the work: milestones M0–M7, each a sequence of self-contained tasks. **Every task has: files to touch, the `shenfun` reference, what to build, how, what it depends on, and an acceptance test.**
- **§8** is the validation/parity harness used by every acceptance test.
- **§9** is the per-script definition-of-done.
- **§10** are appendices: the IMEX Butcher tableaux (copy verbatim), the KMM equations, the Taylor–Couette saddle-point operator, the eigenvalue assembly, and a file index.

A task is **DONE** only when its acceptance test passes against `shenfun` at the stated tolerance in float64.

---

## 1. Goal & deliverables

### 1.1 The two flow families

**Plane Couette flow (PCF)** — fluid between two parallel walls; the top and bottom walls slide in opposite directions, shearing the fluid. We evolve *fluctuations* about the laminar base flow `U_b(x) = U_wall · x`. Wall-normal direction is `x` (Chebyshev/Legendre), streamwise `y` and spanwise `z` are periodic (Fourier).

**Taylor–Couette flow (TC)** — fluid in the annular gap between two concentric rotating cylinders. Radial direction `r` (Chebyshev/Legendre on `[R1, R2]`), axial `z` periodic (Fourier), optionally azimuthal `θ`. The interesting dynamics (Taylor vortices, MRI) live here.

Both have magnetohydrodynamic (MHD) variants that add a magnetic field via a vector potential `A` (with `B = curl(A)`, `J = curl(B)`).

### 1.2 The seven scripts (deliverables)

| # | Script (`couette/`) | Lines | What it is | Hardest dependency |
|---|---|---|---|---|
| 1 | `pcf_fluctuations_corrected.py` | 672 | PCF fluctuation DNS (KMM) | KMM + IMEX-RK |
| 2 | `pcf_fluctuations_divV.py` | 245 | PCF, focused on `div(u)=0` diagnostics | KMM + diagnostics |
| 3 | `pcf_mhd_divfree.py` | 616 | PCF + MHD vector potential, divergence-free `B` | compatible-space curl projections |
| 4 | `pcf_mhd_mri_shearpy.py` | 609 | PCF + MHD + magnetorotational instability | Lorentz/EMF coupling |
| 5 | `taylor_couette_linear.py` | 430 | TC linear stability (generalized eigenproblem) | dense `eig` + `1/r` forms |
| 6 | `taylor_couette_mri.py` | 634 | TC + MHD MRI eigenproblem (insulating BCs) | Bessel BCs + `eig` |
| 7 | `taylor_couette_dns.py` | 1740 | TC nonlinear DNS (coupled `u–p` block solve) | mixed space + block solver + CNAB2 |

All scripts inherit from a small number of shared base classes:
- Scripts 1–4 inherit `ChannelFlow.KMM` (`shenfun/demo/ChannelFlow.py`).
- Scripts 5,6 inherit `CircularCouette` (`couette/taylor_couette_linear.py`).
- Script 7 imports `CircularCouette` for the base flow and defines its own DNS classes.

### 1.3 Recommended delivery order (cheapest end-to-end wins first)

1. **Taylor–Couette linear stability** (script 5) — no time-stepping, no dealiasing, no block solver. Just dense matrix assembly + `eig`. *This is the cheapest complete win and produces eigenmodes used to validate the DNS later.*
2. **Plane Couette fluctuations** (scripts 1, 2) — exercises the full KMM + IMEX foundation.
3. **PCF MHD** (scripts 3, 4) — adds the magnetic vector potential on top of working PCF.
4. **TC MRI eigenproblem** (script 6) — linear stability + Bessel insulating BCs.
5. **Taylor–Couette DNS** (script 7) — the deepest chain; do last.

---

## 2. The three solver architectures (read this)

### 2.1 Plane Couette: the KMM velocity–vorticity formulation

**Reference:** `shenfun/demo/ChannelFlow.py`, class `KMM` (read it in full — it is only 331 lines and is the spine of scripts 1–4).

The incompressible Navier–Stokes equations in a channel are reformulated (Kim–Moin–Moser 1987) to *eliminate pressure* and automatically satisfy incompressibility. Instead of evolving `(u, v, w, p)`, KMM evolves **two scalar fields**:

- **`u` = the wall-normal velocity** `u_x`, governed by a **4th-order (biharmonic)** equation:
  ```
  ∂/∂t (∇²u) = ν ∇⁴u + [nonlinear convection terms]
  ```
  Discretely (ChannelFlow.py:149–155): the linear operator is `ν·div(grad(·))` applied to `div(grad(u_x))`, and the nonlinear RHS is the `Dx`-combination
  `∂²N_y/∂x∂y + ∂²N_z/∂x∂z − ∂²N_x/∂y² − ∂²N_x/∂z²`.
  This needs a **biharmonic basis** (`u = u' = 0` at both walls, 4 boundary conditions) and a **biharmonic fast solver**.

- **`g` = the wall-normal vorticity** `g = curl_x(u) = ∂u_z/∂y − ∂u_y/∂z`, governed by a **2nd-order (Helmholtz)** equation:
  ```
  ∂g/∂t = ν ∇²g + ∂N_y/∂z − ∂N_z/∂y
  ```
  (ChannelFlow.py:157–163). Needs a **Dirichlet basis** and a **Helmholtz fast solver**.

- **`v = u_y` and `w = u_z` are then reconstructed algebraically** from `u` and `g` using the incompressibility constraint, for every Fourier mode `(k_y, k_z) ≠ (0,0)` (ChannelFlow.py:227–251):
  ```
  f = ∂u_x/∂x
  K_over_K2[i] = K[i+1] / (k_y² + k_z²)     # with the (0,0) mode guarded against /0
  u_y = 1j·(K_over_K2[0]·f + K_over_K2[1]·g)
  u_z = 1j·(K_over_K2[1]·f − K_over_K2[0]·g)
  ```

- **The `(k_y, k_z) = (0, 0)` mode** (the plane-averaged mean flow) is singular in the reconstruction above, so it is advanced separately by **two 1D Helmholtz momentum equations** on a standalone 1D space `D00` (ChannelFlow.py:174–197, 242–249).

**Time stepping** is **IMEX Runge–Kutta**: the viscous term `L = ν∇²` is treated implicitly (it is stiff), the convection `N` explicitly. The default scheme is `IMEXRK3`; the scripts run with `IMEXRK222`. See §10.1 for the exact tableaux. The `PDE(...)` objects in `ChannelFlow.py` *are* IMEX stepper instances (`self.PDE = globals()[timestepper]`); their `.assemble()/.compute_rhs(rk)/.solve_step(rk)` methods are the per-stage machinery (see `shenfun/shenfun/utilities/integrators.py` class `PDEIMEXRK`, lines 702–817).

**Convection** (ChannelFlow.py:199–225) is pseudo-spectral with **3/2-rule dealiasing**: transform velocity and its 9 gradients to a 1.5×-padded physical grid, multiply, transform back and truncate. For the PCF *fluctuation* scripts, the base flow adds two extra terms (script 1, the `convection` override): base-flow advection `U_b·∂(·)/∂y` on all components, and shear production `u_x·dU_b/dx` on the streamwise component.

**Diagnostics** (ChannelFlow.py:270–279) are volume integrals `inner(1, f)`: kinetic energies `inner(1, u_i²)`, divergence `sqrt(inner(1, divu²))`.

**Component convention (critical — match it exactly):** component `0` = wall-normal (`x`), `1` = streamwise (`y`), `2` = spanwise (`z`).

### 2.2 Taylor–Couette DNS: cylindrical block-coupled CNAB2

**Reference:** `couette/taylor_couette_dns.py`, class `AxisymmetricTCDNS` (lines 100–345 are the core; read `_build_operators` 182–236 and `step` 288–313).

Unlike KMM, the TC DNS keeps pressure and solves a **coupled velocity–pressure saddle-point system**:

- **Spaces** (taylor_couette_dns.py:127–139): Fourier in `z`; Chebyshev/Legendre Dirichlet `SD` for velocity; an **orthogonal pressure space `SP` truncated to `Nr−2` modes** (`SP.slice = lambda: slice(0, Nr-2)` — this is the inf-sup-stable `P_N / P_{N-2}` pair). The unknown is a **mixed `CompositeSpace([TD,TD,TD,TP])`** = `(u_r, u_θ, u_z, p)`.

- **Cylindrical operators are written with EXPLICIT `1/r` and `1/r²` coefficients on a *Cartesian* space** (taylor_couette_dns.py:178–180):
  ```
  _lap(u) = Dx(u,1,2) + (1/r)·Dx(u,1,1) + Dx(u,0,2)     # axis 1 = radial, axis 0 = axial
  ```
  **Important:** the TensorProductSpace is built with **no `coordinates=` argument** — the measure is plain Cartesian `dr dz`, and every metric factor (`1/r`, `1/r²`, the `+u_r/r` continuity term, the volume `r`-weight) is written by hand as a SymPy coefficient. The radial symbol is `r = TD.coors.psi[1]`. *Do not* use a curvilinear/polar coordinate system for this script — it would double-apply the metric (see §2.4).

- **The implicit operator** `Limp` (taylor_couette_dns.py:193–218) is the saddle-point block
  ```
  [ M/dt − ½A    grad ]
  [   div         0   ]
  ```
  assembled as a `BlockMatrix` and solved **per axial Fourier mode** with a **pressure null-space constraint** `constraints=((3,0,0),)` (pin pressure dof 0 of mode 0). `A` = viscous + Coriolis/curvature couplings.

- **Time stepping is CNAB2** (taylor_couette_dns.py:288–313): Crank–Nicolson (θ=½) on the linear+pressure block, Adams–Bashforth-2 (`1.5 N^n − 0.5 N^{n-1}`) on the nonlinear convection, with an **IMEX-Euler bootstrap** on the first step (`−N^n`, when no history exists). The continuity RHS row is forced to 0.

- **Nonlinear term** (taylor_couette_dns.py:251–272) is the cylindrical advection with metric terms `−u_θ²/r`, `+u_r u_θ/r`, evaluated pseudo-spectrally on the (optionally radially-padded) grid.

### 2.3 Taylor–Couette linear stability / MRI: generalized eigenproblems

**Reference:** `couette/taylor_couette_linear.py` class `CircularCouette` (lines 145–250) and `taylor_couette_mri.py`.

These are **not time-dependent**. For each axial/azimuthal wavenumber `(m, k_z)` they assemble dense matrix blocks `L` and `M` and solve the **generalized eigenvalue problem** `L φ = λ M φ` with `scipy.linalg.eig`. The growth rate is `Re(λ)`.

The assembly pattern (taylor_couette_linear.py:164–221):
```
inner(test, coeff(r)·Dxⁿ(trial))  →  a SpectralMatrix (or list, for multi-term coeffs)
.diags().toarray()                →  a dense complex block
```
Coefficients are SymPy expressions in the radial symbol `x`: `1/x`, `1/x²`, `m²/x²`. The MRI variant adds insulating/vacuum magnetic boundary conditions using modified Bessel functions `scipy.special.iv/kv`.

**These scripts are the easiest to port** (no time-stepping, no dealiasing, no block solver, no MHD coupling for the hydro case). Start here.

### 2.4 Why "explicit `1/r`" and not curvilinear coordinates?

`jaxfun` *does* support curvilinear coordinates (it auto-applies the metric `√det g` in `inner`). But the `shenfun` TC scripts deliberately use a Cartesian space + manual `1/r` coefficients. To match them bit-for-bit, **build the TC spaces Cartesian and put `1/r` in the forms explicitly.** A "principled curvilinear" path (let `jaxfun` apply `√det g = r` automatically) is an *optional* alternative documented in T6.6 — do not mix the two or you will double-count the metric.

---

## 3. Starting point: what `jaxfun` already has (do not reinvent)

Read these before implementing. Most spectral machinery exists.

| Capability | Where | Notes |
|---|---|---|
| Orthogonal + Fourier bases | `src/jaxfun/galerkin/{Chebyshev,Legendre,Jacobi,Ultraspherical,ChebyshevU,Fourier}.py` | forward/backward transforms present |
| Composite (BC) bases via stencils | `src/jaxfun/galerkin/composite.py` | `BoundaryConditions`, `Composite`, `BCGeneric`, `get_stencil_matrix` |
| **Biharmonic basis already works** | `composite.py:744–752` (the `'LDLNRDRN'` case) + `examples/biharmonic2D.py` | clamped `u=u'=0`, dim `N−4`; closed-form Legendre/Chebyshev 5-diagonal stencils. The 2D biharmonic *solve* is proven in `biharmonic2D.py`. |
| TensorProductSpace, TensorProduct | `src/jaxfun/galerkin/tensorproductspace.py` | separable per-axis transforms via `vmap` |
| **Heterogeneous `VectorTensorProductSpace`** | `tensorproductspace.py:503–729` | accepts different basis per component (no same-basis assertion). *Caveat: transforms currently `jnp.stack` components — breaks on ragged shapes; see T0.2/T4.1.* |
| `inner` with variable SymPy coefficients | `src/jaxfun/galerkin/inner.py` | `assemble_multivar`, `contains_sympy_symbols`; multi-term coeffs return a list |
| **`inner` applies the curvilinear measure** | `inner.py:243` (`measure = test_space.system.sg`; `split(expr*measure)`) | so `1/r` variable-coefficient forms already assemble |
| Div/Grad/Curl/Cross/Dot/Outer operators | `src/jaxfun/operators.py` | SymPy-based, curvilinear-aware; `biharmonic2D.py` proves `Div(Grad(Div(Grad(u))))` |
| `project1D` / internal `project` | `inner.py:972, 994` | **re-assembles every call** — not a cached object; see T1.2 |
| `la` matrices & solvers | `src/jaxfun/la/{tpmatrix,blocktpmatrix,diamatrix,matrix,pinned}.py` | `TPMatrix`, `TPMatrices`, `TPMatricesWavenumberSolver` (**vmap over all Fourier wavenumbers — the analog of `SolverGeneric1ND`**), `BlockTPMatrix` (RCM + banded LU), `PinnedSystem` |
| Time integrators | `src/jaxfun/integrators/{rk4,etdrk4,backward_euler}.py` + `base.py` | **only** RK4 / ETDRK4 / BackwardEuler. `BackwardEuler.setup` already builds `mass − dt·linear` (the `c=1` IMEX special case). |
| Curvilinear coordinates | `src/jaxfun/coordinates.py` + `examples/{poisson2D_curv,sphere_helmholtz}.py`, `examples/notebooks/{annulus,polar}.py` | metric, `√det g` |
| **Multi-backend SPMD sharding** | `src/jaxfun/sharding.py` | 1D device mesh `('k',)`, `shard_map` + `all_to_all` separable transforms, GPU/TPU/CPU. **This is how we replace MPI.** |

**The existing `examples/pcf_fluctuations.py` is a raw-JAX collocation prototype that bypasses the Galerkin framework** (finite-difference-style Chebyshev differentiation matrices + FFT + dense per-mode pressure solve). **We are not extending it.** It is a reference for the physics only. The real work builds the KMM solver on top of `jaxfun`'s spectral core.

**Absent entirely** (must build): IMEX-RK / CNAB2 schemes; mixed velocity–pressure space; per-mode coupled saddle-point solver; cached `Project`; `inner(1,·)`; physical `Array` wrapper; `mask_nyquist`; HDF5 I/O; dense generalized `eig` helper; float64 enablement at import.

---

## 4. Cross-cutting engineering conventions (every task obeys these)

These are the JAX idioms that make the port differentiable and multi-backend. Violating them is a bug even if the numbers are right.

1. **Float64 everywhere.** Spectral parity needs double precision. `jax_enable_x64` must be on *before any array is created* (see T0.1). All test tolerances assume float64.
2. **Functional / immutable.** No in-place mutation. `shenfun` writes `u[1] = ...`, `H[0] = ...`, `rhs[1] += ...`, `self.N_old[:] = ...`. In JAX these become `.at[i].set(...)` / returning new pytrees / loop-carried values. **Never** translate a `shenfun` in-place write literally.
3. **`vmap` over wavenumbers, never Python loops over modes.** Per-Fourier-mode banded solves are one batched XLA call (`TPMatricesWavenumberSolver` already does this). Loops over the `≤3` *components* of a vector are fine (unrolled at trace time).
4. **Assemble once, step many.** Matrix assembly and factorization happen at setup time on host; the jitted `step()` only does matvecs + cached banded solves. Mirror `BackwardEuler.setup` / `ETDRK4`.
5. **`jit` + `lax.scan`/`fori_loop` over timesteps; static stage loops unrolled.** The `≤4` IMEX stages are Python-unrolled (static) so XLA sees a fixed graph and the per-stage solver (a Python list) is selected at trace time.
6. **No MPI.** `shenfun`'s `comm`, slab decomposition, and rank-0 gating collapse to: the whole field is logically local; the `(0,0)` mode is *global index 0* (guarded with `jnp.where`, not a rank check); reductions are `jnp.sum` (single device) or `jax.lax.psum`/`all_gather` over the `'k'` sharding mesh (multi-device). See T7.3.
7. **Heterogeneous vectors are pytrees, not stacked arrays.** The velocity `BD = [TB, TD, TD]` has components of *different* modal length (`TB` is `N−4`, `TD` is `N−2`); `jnp.stack` will raise. Represent such vectors as a tuple/pytree of per-component arrays (T0.2).
8. **SymPy for differentiation, never `jax.grad` for spatial derivatives.** Spatial derivatives are realized by differentiating basis polynomials (already in `jaxfun`). `jax.grad` is only for autodiff *through* the solver (the "differentiable solver" goal), which works automatically as long as rules 1–7 hold.
9. **Keep I/O and Python-side diagnostics outside jitted regions** via host callbacks; device→host transfers must not happen inside `lax.fori_loop` (T7.2).
10. **Cite your source.** Every new function gets a docstring naming the `shenfun` reference (`file:line`) it reproduces, so parity can be re-checked.

---

## 5. Master gap map (41 items)

Status: `present` (works, maybe needs an adapter) · `partial` (pieces exist) · `missing` (build from scratch). Effort: S/M/L/XL. Milestone: see §6.

| # | Gap | Status | Effort | Target file | Milestone |
|---|---|---|---|---|---|
| 1 | float64 enablement at import | missing | S | `src/jaxfun/__init__.py` | M0 |
| 2 | Ragged heterogeneous-vector pytree (`TB` N−4 vs `TD` N−2) + `g_=curl[0]` functional aliasing | missing | M | `tensorproductspace.py` | M0 |
| 3 | Real-`z` Fourier decision (rfft vs full-complex) | decision | M | `Fourier.py` | M0 |
| 4 | Deterministic IC builder + shenfun parity harness (steps>0) | missing | M | `tests/`, `tests/_parity.py` | M0 |
| 5 | Tuple→dict BC adapter (`bc=(0,0,0,0)`, `bc=(0,0)`) | partial | S | `galerkin/functionspace.py` | M0 |
| 6 | `inner(1,·)` / `integrate()` volume-integral primitive (DEDUP: was specced 3×) | missing | S | `galerkin/inner.py` (or `diagnostics.py`) | M1 |
| 7 | Cached `Project` operator object (reusable, hoisted assembly) | partial | M | `galerkin/inner.py` | M1 |
| 8 | Cross-space + per-component `project` (`div(u)→TC`; `[TD,TC,TC]` curl) | partial | M | `galerkin/inner.py` | M1 |
| 9 | `Dx(u, axis, k)` ergonomic wrapper + verify `coeff·Dx` assembly | present | S | `operators.py` | M1 |
| 10 | `mask_nyquist` utility | missing | S | `galerkin/tensorproductspace.py` | M1 |
| 11 | Scaled broadcast wavenumber grid `K` + `K_over_K2` helper | partial | S | `galerkin/tensorproductspace.py` | M1 |
| 12 | Physical `Array` wrapper dual to `JAXFunction` (`.forward()`) | missing | M | `galerkin/arguments.py` | M1 |
| 13 | `get_dealiased` / padded forward–backward pair (3/2 rule) | partial | L | `orthogonal.py`,`Fourier.py`,`tensorproductspace.py` | M1 |
| 14 | `backward(mesh='uniform')` (visualization/IO mesh) | partial | S | `orthogonal.py`,`tensorproductspace.py` | M1 |
| 15 | Pivoted/robust per-mode banded LU (biharmonic, indefinite) | partial | L | `la/diamatrix.py`,`la/tpmatrix.py` | M2 |
| 16 | Named Helmholtz/Biharmonic fast-solver constructors | partial | M | `la/` (e.g. `solvers.py`) | M2 |
| 17 | Per-mode constraint / null-space pinning in wavenumber solver | partial | M | `la/tpmatrix.py`,`la/pinned.py` | M2 |
| 18 | Dense generalized `eig` assembly + singular-`M` filtering | missing | L | `la/eig.py` (new) | M2 |
| 19 | IMEX operator-split helper `build_implicit_operator(c,dt)` | partial | S | `integrators/base.py` | M3 |
| 20 | ARS IMEX-RK family (`PDEIMEXRK` + IMEXRK011/111/222/443) | missing | M | `integrators/imex_rk.py` (new) | M3 |
| 21 | Spalart low-storage `IMEXRK3` | missing | M | `integrators/imex_rk.py` | M3 |
| 22 | Multi-equation coupled IMEX driver skeleton | missing | L | `integrators/coupled.py` (new) | M3 |
| 23 | Heterogeneous `VectorTensorProductSpace` transforms (ragged) | present | M | `tensorproductspace.py` | M4 |
| 24 | KMM velocity–vorticity orchestrator class | missing | XL | `examples/channelflow_kmm.py` (new) | M4 |
| 25 | `compute_vw`: `K_over_K2` reconstruction + (0,0)-mode 1D Helmholtz | missing | M | `examples/channelflow_kmm.py` | M4 |
| 26 | PCF base-flow convection override (`U_b·∂/∂y` + shear production) | missing | M | `examples/pcf_fluctuations_jax.py` (new) | M4 |
| 27 | Volume-integral diagnostics (energy/div/Reynolds/Maxwell/wall/shear) | missing | M | `examples/` + `diagnostics.py` | M4 |
| 28 | MHD: `A` diffusion eqs + `B=curl(A)`/`J=curl(B)` compatible-space projections | missing | L | `examples/pcf_mhd_jax.py` (new) | M4b |
| 29 | Cross-product rotational convection `u×curl(u)` / EMF in physical space | missing | M | `integrators/nonlinear.py`,`operators.py` | M4b |
| 30 | Radial-coordinate symbol accessor (`TD.coors.psi[1]`) | partial | S | `coordinates.py`,`tensorproductspace.py` | M5 |
| 31 | Explicit cylindrical `1/r`, `1/r²` variable-coeff bilinear `inner` | present | M | (verify) `galerkin/inner.py` | M5 |
| 32 | Dense block extraction (`.diags().toarray()` analog) | partial | S | `la/matrix.py` | M5 |
| 33 | Robin/Neumann radial bases + Bessel (`iv/kv`) insulating BC coefficient | partial | M | `galerkin/composite.py`, example | M5/M6mri |
| 34 | TC linear/MRI eigensolver example scripts | missing | M | `examples/taylor_couette_linear_jax.py` (new) | M5 |
| 35 | Mixed velocity–pressure `CompositeSpace` + truncated-orthogonal pressure | missing | XL | `tensorproductspace.py` | M6 |
| 36 | Per-Fourier-mode coupled saddle-point block solve + pressure gauge | missing | XL | `la/blocktpmatrix.py` | M6 |
| 37 | CNAB2 integrator (CN + AB2 + IMEX-Euler bootstrap) | missing | L | `integrators/cnab2.py` (new) | M6 |
| 38 | Padded radial dealiasing parity spike + `inv_r_p` + cylindrical nonlinear terms | missing | M | `examples/taylor_couette_dns_jax.py` (new) | M6 |
| 39 | `r`-weighted volume-integral diagnostics for TC | missing | M | `diagnostics.py` | M6 |
| 40 | HDF5 field output + checkpoint/restart + XDMF | missing | M | `src/jaxfun/io/__init__.py` (new) | M7 |
| 41 | MPI→sharding (remove rank gating; `psum`/`all_gather`; multi-device parity) | partial | L | `examples/`, `sharding.py` | M7 |

Plus four **critic-flagged small owned items** folded into tasks: inhomogeneous updating Neumann pressure BC for the optional pressure recovery (T4.8); BC-lifting note for moving walls (T0.5 note — PCF avoids it by evolving fluctuations); composite-basis polynomial-padding parity spike (T6.5); host-callback diagnostic/IO cadence (T7.2).

---

## 6. Dependency graph & milestone overview

```
M0  Foundations: x64, ragged-pytree, real-FFT decision, parity harness, BC adapter
        │  (gates EVERYTHING)
        ▼
M1  Shared spectral primitives: integrate(), Project (cached + cross-space),
    Dx wrapper, mask_nyquist, K/K_over_K2, physical Array, get_dealiased, uniform backward
        │
        ├────────────────────────────┬───────────────────────────────┐
        ▼                            ▼                               ▼
M2  Solver tier:               M3  IMEX time integration:      M5  TC LINEAR track
    pivoted banded LU,             operator-split helper,          (independent! cheapest win)
    Helmholtz/Biharmonic,         PDEIMEXRK + IMEXRK222/443,       radial symbol, 1/r forms,
    per-mode constraints,         IMEXRK3, coupled driver          dense eig, Bessel BCs,
    dense eig (→M5)                   │                            scripts 5 (+6 after M2 eig)
        │                            │
        └──────────┬─────────────────┘
                   ▼
M4  PLANE-COUETTE track: ragged VTPS transforms, KMM orchestrator,
    compute_vw, base-flow convection, diagnostics → scripts 1,2
                   │
                   ▼
M4b PCF MHD: A-diffusion, compatible-space curl projections, Lorentz/EMF,
    physical cross-product → scripts 3,4
                   │
                   ▼   (TC DNS needs M1,M2,M3 + its own deep chain)
M6  TAYLOR-COUETTE DNS: mixed u–p space → block saddle solver →
    CNAB2 → radial dealiasing parity → cylindrical nonlinear → script 7
                   │
                   ▼
M7  I/O & multi-backend: HDF5/checkpoint/XDMF, diagnostic cadence, sharding parity
```

**Critical paths:**
- **First runnable PCF** = M0 → M1 → M2 → M3 → M4.
- **First runnable TC (linear)** = M0 → M1 (1/r assembly) → M2 (dense eig only) → M5. *Largely independent of the KMM chain — staff it in parallel.*
- **TC DNS** = the longest chain; the **mixed velocity–pressure space (T6.1) is the single most important predecessor** and is XL — start it early in the TC track even though the DNS lands last.

**Hard ordering rules the critic flagged:** the mixed space (T6.1) **must** precede the block solver (T6.2); the biharmonic *basis* (present, T0.5) is distinct from the biharmonic *solver* (missing, T2.2). De-duplicate `inner(1,·)`, `mask_nyquist`, and the wavenumber grid into the single M1 implementations — do not let three milestones each build their own.

---

## 7. Milestones & tasks

Each task: **[ID] Title** — *status/effort* · files · depends-on. Then **Build / How / Accept**.

---

### M0 — Foundations & parity harness

> These four cross-cutting prerequisites gate every parity claim. Without them the team will chase phantom float32 mismatches and pytree-shape errors. Do these first, fully.

#### [T0.1] Enable float64 at library import — *missing/S* · `src/jaxfun/__init__.py`
- **Build:** ensure `jax.config.update("jax_enable_x64", True)` runs at `import jaxfun` (before any array is created), or document a required `JAX_ENABLE_X64=1` env var and assert it. Today the only x64 enablement is in an example, not the library.
- **How:** add the config call at the top of `src/jaxfun/__init__.py`; add a guard that warns if arrays were already created in x32. Verify it threads through space/solver construction.
- **Accept:** `import jaxfun; jnp.zeros(1).dtype == float64`. A space's quadrature points are float64. A regression test asserts x64 is on.

#### [T0.2] Ragged heterogeneous-vector pytree + `g_=curl[0]` functional aliasing — *missing/M* · `tensorproductspace.py`
- **Why:** the velocity `BD=[TB,TD,TD]` has component 0 of modal length `N−4` (biharmonic) and components 1,2 of length `N−2` (Dirichlet). `jnp.stack` raises. `shenfun`'s `g_ = curl[0]` is a mutating view; JAX is immutable.
- **Build:** a representation for heterogeneous vector fields as a registered **pytree of per-component coefficient arrays** (not a stacked tensor). Provide a named accessor for "component 0 of curl" that reads/writes `state` functionally (`.at`-style on the relevant leaf).
- **How:** register the vector Function as a JAX pytree whose leaves are the component arrays; in `VectorTensorProductSpace` transforms, return a tuple when component shapes differ (only `jnp.stack` in the homogeneous case for speed). Re-express `g_=curl[0]` as an explicit dependency `g = 1j·K[1]·u[2] − 1j·K[2]·u[1]` recomputed/stored in the state pytree, never an alias.
- **Depends on:** none. **Publishes a contract** consumed by M3 (state pytrees), M4 (KMM state), M7 (checkpoint serialization).
- **Accept:** build `BD=VTPS((TB,TD,TD))`, store a field with analytic components, round-trip `forward(backward(u))` per component to 1e-12; `jnp.stack` is never called on ragged shapes (assert no shape error). `jit` caching is stable across steps (leaf structure constant).

#### [T0.3] Real-`z` Fourier strategy decision — *decision/M* · `Fourier.py`
- **Why:** `shenfun`'s KMM uses `F1=Fourier(dtype='D')` (complex, streamwise `y`) and `F2=Fourier(dtype='d')` (**real** spanwise `z` → rfft half-spectrum). `jaxfun`'s `Fourier` uses full `jnp.fft.fft`. The Nyquist masking, the spanwise energy spectrum (must *not* fold over negative `k_z`), and dealias padding all depend on this.
- **Build:** **Decide and document** one of: (A) add an rfft-backed real-Fourier space matching `shenfun`'s half-spectrum layout, or (B) commit to full-complex `z` and re-derive the Nyquist/spectra indexing and the `(0,0)`-mode handling. **Recommendation: (B) full-complex** for simplicity/differentiability, with a documented mapping to `shenfun`'s rfft layout used only in the parity harness. Whichever you pick, write it down — every later transform/spectra task references this decision.
- **Accept:** a short ADR (architecture decision record) committed to `docs/`; a unit test transforming a real field and comparing the retained spectrum to `shenfun`'s rfft output (modulo the documented layout map) to 1e-12.

#### [T0.4] Deterministic IC builder + shenfun parity harness — *missing/M* · `tests/_parity.py`, extend `tests/test_pcf_fluctuations.py`
- **Why:** the existing `tests/test_pcf_fluctuations.py` only checks `steps=0`. Every later milestone needs a *time-evolution* parity check. The PCF IC is purely deterministic (sin/cos × wall-damping; the `np.random.seed(42)` in script 1 is unused — confirm).
- **Build:** (1) a deterministic IC builder reproducing script 1's initial field exactly; (2) a reusable harness `compare_to_shenfun(jax_solver, shen_solver, steps, fields, tol)` that runs both from the same IC and compares spectral coefficients and the six diagnostics (`Epert, Etot, divL2, u_top, u_bot, mean_shear`) after `1, 5, 50` steps.
- **How:** reuse the `shenfun`-loading shim already in `examples/pcf_fluctuations.py:_load_shenfun_demo` (disables `ShenfunFile`/`Checkpoint`). Run `shenfun` on CPU, `jaxfun` on the available backend, compare `np.asarray` arrays.
- **Accept:** harness runs end-to-end against a trivial/linear case (e.g. pure viscous decay, no convection) and reports per-field rtol. It becomes the acceptance vehicle for T4.x, T5.x, T6.x.

#### [T0.5] Tuple→dict boundary-condition adapter — *partial/S* · `galerkin/functionspace.py`
- **Why:** `shenfun` spells the clamped biharmonic as `bc=(0,0,0,0)` and Dirichlet as `bc=(0,0)`; `jaxfun`'s `FunctionSpace` only accepts the dict form `{'left':{'D':..,'N':..},'right':{...}}`. The biharmonic *basis* already exists (`composite.py:744–752`, `biharmonic2D.py`).
- **Build:** `normalize_bc(bc)` mapping `(a,b) → {'left':{'D':a},'right':{'D':b}}` and `(a,b,c,d) → {'left':{'D':a,'N':b},'right':{'D':c,'N':d}}` (verify the 4-tuple order `(LeftD, LeftN, RightD, RightN)` against `shenfun` `spectralbase.py` `BoundaryConditions`). Call it from `FunctionSpace` when `bc` is a tuple/list.
- **Note (moving-wall BC lifting):** PCF evolves *fluctuations* about `U_b=U_wall·x` with **homogeneous** BCs, so inhomogeneous BC lifting (`DirectSum`+`BCGeneric`) is **not** needed for scripts 1–4. Document this; do not build lifting unless a future non-fluctuation script needs it.
- **Accept:** `FunctionSpace(N,'C',bc=(0,0,0,0))` produces a space whose stencil matrix matches `shenfun` `ShenBiharmonic.stencil_matrix().diags().toarray()` to 1e-13, and whose `dim == N−4`; `bc=(0,0)` matches `ShenDirichlet` (`dim == N−2`). Verify both Legendre and Chebyshev stencils (algebraic equality already checked: `−(2n+4)/(n+3) ≡ 2(−n−2)/(n+3)`).

---

### M1 — Shared spectral primitives (de-duplicated)

> Implement each primitive **once**. The analysis independently specced `inner(1,·)`, `mask_nyquist`, and the wavenumber grid in multiple clusters — those are the *same* function. Build here, reuse everywhere.

#### [T1.1] `integrate()` / `inner(1,·)` volume-integral primitive — *missing/S* · `galerkin/inner.py` (+ expose weights in `orthogonal.py`)
- **Why:** every diagnostic in every script is `inner(1, f)` over a physical field. `jaxfun`'s `inner` hard-requires a `TestFunction` (`inner.py:239–240`) — there is no scalar-`1` path.
- **Build:** `integrate(u_phys, V) -> scalar` = quadrature sum `Σ_ijk u_ijk · w0_i·w1_j·w2_k`, divided by the product of `domain_factor`s (the reference→physical Jacobian, as `shenfun` does at `forms/inner.py:198`), and multiplied by the measure `√det g = V.system.sg` **only when the space is curvilinear**. Expose `OrthogonalSpace.integration_weights()` returning `w_j/domain_factor` (fold `sg` exactly as `scalar_product` does at `orthogonal.py:262–269`).
- **How:** `jnp.einsum('ijk,i,j,k->', u, w0, w1, w2)` (static `d≤3`, unrolled). VTPS wrapper sums component integrals. Make it jittable (static `V` via `static_argnums`) and differentiable (pure `jnp`, no `.item()`). **Do not** shoehorn into the TestFunction-required `inner`; expose `integrate()` and have a numeric-`1` first-arg branch of `inner` delegate to it.
- **TC caveat:** for the Cartesian TC scripts the script multiplies the integrand by `rphys` *manually* (`inner(1, (...)·rphys)`), so `integrate` must **not** auto-apply `sg` for a Cartesian space — take `sg` behavior from `V.system`, and build TC spaces Cartesian.
- **Accept:** `integrate(ones, T) == domain volume`; reproduce the `shenfun` `inner.py` docstring area `12.566370614`; a separable analytic `f` matches `shenfun inner(1, Array(T, buffer=f))` to 1e-12. Wire into the T0.4 harness (it already compares `divL2/Epert/Etot` at steps=0).

#### [T1.2] Cached `Project` operator object — *partial/M* · `galerkin/inner.py`
- **Why:** KMM builds ~12 `Project` objects once and calls a subset every RK stage. `jaxfun`'s `project()` **re-assembles the mass system on every call** (the exact perf cliff `taylor_couette_dns.py:886–887` flags: ~108 ms vs ~14 ms/step).
- **Build:** `class Project(uh_expr, T)` mirroring `shenfun/shenfun/forms/project.py:173`: assemble the target mass matrix `B = inner(u·v)` and the RHS derivative matrices `A` from `inner_items(v·uh)` **once**; `B.lu_factor()` once and stash it; `__call__(uh_coeffs) -> coeff array` does the `A`-matvec then `B_lu.solve`. Preserve `shenfun`'s identity optimization (`project.py:218–225`): when `B` and `A` share the non-diagonal factor, skip the solve.
- **How:** `__call__` is `@jax.jit` with `self` carried via a `_CacheBox` (pattern already in `la/`). All per-call ops are `jnp` matmuls/banded solves — no host re-assembly inside `__call__`. The `shenfun` `output_array=` buffer-reuse becomes functional: `__call__` returns the array; the caller threads it into the state pytree (so `g_=curl[0]` is "store curl component 0 in state", per T0.2).
- **Depends on:** T0.2 (functional state).
- **Accept:** `shenfun`'s `project.py` docstring example (`Project(Dx(u,0,1),T)` with `u[1]=1` → `[1,0,...]`) reproduced to 1e-12. A 3D KMM-like test: `Project(Dx(u_,0,1),TC)` and `Project(div(u_),TC)` coefficient arrays match `shenfun` for the same input. Benchmark: repeated `__call__` does **not** re-trigger host assembly.

#### [T1.3] Cross-space + per-component `project` — *partial/M* · `galerkin/inner.py`
- **Why:** `divu = Project(div(u_), TC)` projects a field whose components live in `TB/TD` (with BCs) into the orthogonal `TC` (no BC) — *test/mass space ≠ source field space*. MHD needs **per-component** projection into different target spaces (`B`→`[TD,TC,TC]`, `J`→`[TC,TD,TD]`) for the discrete `div(curl)=0` invariant.
- **Build:** (a) ensure `inner` can assemble a linear form `inner(v_target, div(source_JAXFunction))` where test space ≠ the JAXFunction's space (backward-transform the source derivative onto the shared quadrature mesh, then scalar-product against the target basis). (b) Extend `Project` to accept a *list* of `(component_expr, component_target_space)` and build one cached scalar projector per component (each an independent banded/diagonal solve).
- **Depends on:** T1.2.
- **Accept:** divergence-free analytic field → `max|Project(div(u_),TC)|` ≈ machine eps, matching script 2's `_divergence_stats`. Compatible-space test: seed `A`, compute `B=curl(A)` into `[TD,TC,TC]`, then `div(B)` into `TC`; assert `divb_l2 < 1e-12` and equals `pcf_mhd_divfree.py` diagnostics. Per-component arrays match `shenfun`'s `projBx/projBy/projBz` to 1e-12.

#### [T1.4] `Dx(u, axis, k)` ergonomic wrapper + verify `coeff·Dx` — *present/S* · `operators.py`
- **Build:** `Dx(expr, axis, k=1) -> sp.Derivative(expr, (system.base_scalars()[axis], k))` so ported scripts read like `shenfun`. Verify `(1/r)·Dx(u,1,1)` assembles correctly (the coefficient lambdify path at `inner.py:819/834` already supports it).
- **Accept:** `div(grad(u))·v` and `Dx(u,0,2)·v` matrices match `shenfun` for identical Chebyshev×Fourier×Fourier spaces to 1e-12; the TC continuity row `inner(q,Dx(ur,1,1)) + inner(q,(1/r)·ur) + inner(q,Dx(uz,0,1))` matches `taylor_couette_dns.py:215–217`; `biharmonic2D.py` still assembles after the change.

#### [T1.5] `mask_nyquist` utility — *missing/S* · `tensorproductspace.py`
- **Build:** `get_mask_nyquist()` returning a 0/1 mask zeroing the Nyquist frequency of each Fourier axis, and `mask_nyquist(field, mask)` = elementwise multiply (functional). Mirrors `shenfun` `TB.get_mask_nyquist()` / `H_.mask_nyquist(mask)`.
- **Depends on:** T0.3 (Fourier layout).
- **Accept:** masking a field zeros exactly the Nyquist modes; matches `shenfun`'s mask on the same grid.

#### [T1.6] Scaled wavenumber grid `K` + `K_over_K2` helper — *partial/S* · `tensorproductspace.py`
- **Build:** `local_wavenumbers(scaled=True, broadcast=True)` giving the broadcast `K[i]` grids, and a helper `K_over_K2(K)` = `K[i+1]/where(K2==0,1,K2)` for the `compute_vw` reconstruction (guards the singular `(0,0)` mode).
- **Accept:** `K` matches `shenfun`'s `TD.local_wavenumbers(scaled=True)`; `K_over_K2` finite everywhere and matches `ChannelFlow.py:168–171`.

#### [T1.7] Physical `Array` wrapper dual to `JAXFunction` — *done/M* · `galerkin/arguments.py`
- **Why:** `shenfun` distinguishes `Function` (spectral coefficients) from `Array` (physical-space values, with `.forward()`). KMM and TC build physical products as `Array`s then `forward` them.
- **Build:** implemented as `PhysicalArray`/`Array`, carrying physical values and its function space; `.forward()` returns a `JAXFunction`; `.coefficients()` exposes raw coefficients; registered as a pytree.
- **Depends on:** T0.2.
- **Accept:** covered by `tests/galerkin/test_arguments_extras.py` for tensor/scalar round-trips and pytree restoration. Live vector-space parity remains a broader validation item.

#### [T1.8] `get_dealiased` / padded forward–backward pair (3/2 rule) — *partial/L* · `orthogonal.py`,`Fourier.py`,`tensorproductspace.py`
- **Why:** convection is dealiased on a 1.5×-padded grid. The per-axis padding *mechanics* exist (`Fourier.backward` mid-array zero-pad; polynomial `backward(c,N)` over-sampling) but there is **no `get_dealiased()` convenience** and no cached padded space.
- **Build:** `get_dealiased(padding_factor)` on `OrthogonalSpace`/`Fourier` returning a thin wrapper recording `Np = floor(pf·N)` (match `shenfun`'s `floor`) with `backward(c)=self.backward(c,N=Np)` and `forward(u)=truncate(self.forward(u))`. For `pf=1` (the wall-normal axis in PCF) it is a no-op. `TensorProductSpace.get_dealiased(tuple)` composes per-axis; `VectorTensorProductSpace.get_dealiased` does per-component.
- **How:** fully functional — padded backward is `ifft` on a zero-padded coeff array; forward truncation is an index-gather. Snap sizes with `int(np.floor(pf·N))` to match `shenfun` bit-for-bit.
- **Risk / spike (PCF-easy, TC-hard):** PCF uses `padding_factor=(1,1.5,1.5)` (only Fourier axes padded) — the **easy** path. TC pads the **radial polynomial** axis (`dealias=1.5`), which uses a *different* physical mesh than `shenfun`'s padded DCT — this needs the parity spike in **T6.5**. Get the PCF Fourier path solid here; defer radial-padding parity to M6.
- **Depends on:** T0.3.
- **Accept:** padded `backward` then `forward` returns the original coefficients (de-aliased projection idempotence) to 1e-13; a quadratic product `u·v` of band-limited fields formed on the padded grid then truncated matches `shenfun` `TD.get_dealiased()` `H` coefficients to 1e-12; `floor(1.5·Ny)` equals `shenfun` `TDp` shape; `padding_factor=(1,1.5,1.5)` leaves the wall-normal count unchanged.

#### [T1.9] `backward(mesh='uniform')` — *partial/S* · `orthogonal.py`,`tensorproductspace.py`
- **Build:** evaluate the spectral series on a *uniform* (non-quadrature) mesh for visualization/IO, matching `shenfun`'s `u_.backward(mesh='uniform')`.
- **Accept:** physical values on the uniform mesh match `shenfun` to 1e-12.

---

### M2 — Linear solver tier

> `jaxfun` already has the *generic* per-wavenumber banded solver (`TPMatricesWavenumberSolver`, the functional analog of `shenfun`'s `SolverGeneric1ND`). Helmholtz and Biharmonic are just wider-band cases of the same no-pivot banded LU. Here we make it robust, named, constrained, and add dense `eig`.

#### [T2.1] Pivoted/robust per-mode banded LU — *partial/L* · `la/diamatrix.py`,`la/tpmatrix.py`
- **Why:** the existing wavenumber solver uses no-pivot banded LU. The Chebyshev **biharmonic** operator and the **saddle-point** blocks are indefinite/wide-band and can lose diagonal dominance; constraint indenting (T2.3) destroys it further.
- **Build:** a pivoted (or even/odd-decoupled, as `shenfun`'s `chebyshev/la.py` Biharmonic does) per-mode banded LU, vmapped over wavenumbers. Prefer `jax.scipy.linalg.lu_factor`/`lu_solve` batched over the mode axis if a structured banded path is too fragile.
- **Current coverage:** `TPMatricesWavenumberSolver(pivot=True)` and `tpmats_wavenumber_factor(..., pivot=True)` route through batched dense `jax.scipy.linalg.lu_factor`/`lu_solve`; tests cover parity against assembled Kronecker solves and a zero-diagonal pivot stress case.
- **Accept:** solve a 1D biharmonic MMS with known solution to 1e-10 for all wavenumbers; the Chebyshev biharmonic operator (ill-conditioned at high `k`) solves stably; matches `shenfun` `chebyshev.la.Biharmonic` coefficients to 1e-10.

#### [T2.2] Named Helmholtz / Biharmonic fast-solver constructors — *partial/M* · `la/solvers.py`
- **Build:** thin named wrappers `Helmholtz(matrices)` and `Biharmonic(matrices)` over `TPMatricesWavenumberSolver` (+ pivoted LU from T2.1), so the KMM equations and the `(0,0)`-mode 1D solves read like `shenfun` (which picks `chebyshev.la.Helmholtz/Biharmonic` for Chebyshev and `la.SolverGeneric1ND` otherwise). They route to the fast per-mode banded path, not a dense fallback.
- **Accept:** `Helmholtz` and `Biharmonic` constructors solve their MMS problems to 1e-10 and match `shenfun`. Assert the factorization type is the wavenumber/banded path (not dense) for a Fourier×Legendre space.

#### [T2.3] Per-mode constraint / null-space pinning — *partial/M* · `la/tpmatrix.py`,`la/pinned.py`
- **Why:** the pressure-Poisson and saddle systems are singular at `(0,0)`; `shenfun` pins one dof of mode 0 (`constraints=((0,0,0),)` / `((3,0,0),)`). `PinnedSystem` works on one global matrix; `TPMatricesWavenumberSolver` now also supports per-mode pins.
- **Build:** `constraints: tuple[(flat_mode_index, row, value), ...]` is accepted by the wavenumber solver and factory. At factor construction, constrained mode rows are replaced by identity rows; at solve time, matching RHS entries are set to the requested pin values.
- **Depends on:** T2.1.
- **Current coverage:** `TPMatricesWavenumberSolver` and `tpmats_wavenumber_factor` accept per-mode `constraints=((flat_mode, row, value), ...)`; tests cover constrained row replacement and RHS pinning against an explicitly pinned dense batch.
- **Accept:** a Poisson/Helmholtz with singular `(0,0)` solved with `constraints` matches `shenfun` `SolverGeneric1ND(...)(b, constraints=((0,0,0),))` to 1e-12; the pinned mean is exactly 0.

#### [T2.4] Dense generalized `eig` assembly + singular-`M` filtering — *missing/L* · `la/eig.py`
- **Why:** TC linear stability solves `L φ = λ M φ` densely.
- **Build:** (1) a way to extract an assembled `inner` result as a dense complex matrix (`SpectralMatrix.diags().toarray()` analog — see T3.5 for the `.diags()` part); (2) `generalized_eig(L, M)` returning eigenvalues/vectors with filtering of spurious modes from the singular pressure block (large/inf eigenvalues). Host `scipy.linalg.eig` is acceptable (these are small dense problems, run once); optionally convert to a standard EVP `M⁻¹L` when `M` is nonsingular.
- **Accept:** for a small annulus problem the leading eigenvalue matches `shenfun`'s `taylor_couette_linear.py` (via `scipy.linalg.eig`) to 1e-8; spurious infinite modes are filtered.

---

### M3 — IMEX time integration (shared)

> Copy the Butcher tableaux **verbatim** from §10.1. The schemes are simple once the implicit solver (M2) exists.

#### [T3.1] IMEX operator-split helper `build_implicit_operator(c, dt)` — *partial/S* · `integrators/base.py`
- **Why:** all IMEX schemes need `S(c) = M − c·dt·L` and its factorization; `BackwardEuler.setup` already does the `c=1` case.
- **Build:** `BaseIntegrator.build_implicit_operator(c, dt)` returning `mass_operator − (c·dt)·linear_operator` with `_warm_operator_solve_cache` applied. **Add a sibling `apply_linear_scalar_product(u) = linear_operator @ u`** (no mass inverse) for the *explicit* linear history — `linear_rhs` currently mass-inverts (`base.py:240–245`); using it in an IMEX scheme would double-apply `M⁻¹`.
- **Accept:** `S(c)@x == mass@x − c·dt·(linear@x)` for random `x`; `S(1,dt)` reproduces `BackwardEuler`; `S(c).solve(S(c)@x)==x` to 1e-10; for a Fourier×Legendre space `S(c)` routes to the wavenumber/banded solver (assert `lu_factor()` type).

#### [T3.2] ARS IMEX-RK family `PDEIMEXRK` + tableaux — *missing/M* · `integrators/imex_rk.py`
- **Build:** `PDEIMEXRK(BaseIntegrator)` with class-attribute tableaux `a,b,c` and `steps()`; subclasses `IMEXRK011/111/222/443` (tableaux in §10.1). `setup(dt)`: build the **single** implicit operator `S = M − dt·a[1,1]·L` (constant active diagonal per ARS condition 2.3 — assert it) and warm its cache. `step`: carry stage history as stacked arrays `K` (shape `(steps, *state)`) and `L_hist` (`(steps-1, *state)`), updated out-of-place with `.at[rk].set`. Stage `rk`: `K=K.at[rk].set(scalar_product N(u))`; `rhs = u0 + Σ_j dt·b[rk+1,j]·K[j]`; if `rk>0`: `L_hist=L_hist.at[rk-1].set(linear@u)`; `rhs += Σ_j dt·a[rk+1,j+1]·L_hist[j]`; `u = S.solve(rhs)`. Python-unroll the stage loop; `@jax.jit(static_argnums=...)`.
- **Critical:** the index offsets `a[rk+1,j+1]` (implicit, col 0 is zeros) vs `b[rk+1,j]` (explicit) — off-by-one silently degrades order. Reproduce exactly.
- **Depends on:** T3.1, T2.2.
- **Accept:** import `shenfun`'s `IMEXRK222().stages()` and assert elementwise tableau equality. Integrate a 1D `du/dt = ν u_xx + f`: match `shenfun`'s `PDEIMEXRK` trajectory coefficient-by-coefficient to 1e-10 over 10 steps; 2nd-order (IMEXRK222) / ~3rd-order (IMEXRK443) convergence via `dt`-halving.

#### [T3.3] Spalart low-storage `IMEXRK3` — *missing/M* · `integrators/imex_rk.py`
- **Why:** `IMEXRK3` is the KMM **default** stepper. Unlike `PDEIMEXRK` it uses a **different implicit factor per stage** `(a[rk]+b[rk])·dt/2` and a 2-register low-storage nonlinear history.
- **Build:** tableaux `a=(8/15,5/12,3/4)`, `b=(0,−17/60,−5/12)`, `c=(0,8/15,2/3,1)` (§10.1). `setup`: precompute **three** implicit operators `S[rk]=M − ((a[rk]+b[rk])·dt/2)·L` (warm each) and three explicit-linear matvec closures `Lexp_rk(u)=M@u + ((a[rk]+b[rk])·dt/2)·(L@u)`. `step`: carry `w_prev` (init zeros, reset each step); per stage `w0=scalar_product N(u)`; `rhs = dt·(a[rk]·w0 + b[rk]·w_prev) + Lexp_rk(u)`; `w_prev=w0`; `u=S[rk].solve(rhs)`.
- **Accept:** match `shenfun` `IMEXRK3` on a 1D linear-diffusion mode and a forced nonlinear scalar PDE coefficient-by-coefficient; 3rd-order `dt`-convergence; the three per-stage factors equal `shenfun`'s.

#### [T3.4] Multi-equation coupled IMEX driver skeleton — *missing/L* · `integrators/coupled.py`
- **Why:** KMM advances `{u, g}` (+ the `(0,0)`-mode equations) *together* per stage with the ordering **all-`compute_rhs` → all-`solve_step` → algebraic `compute_vw`**. `BaseIntegrator` drives exactly one equation.
- **Build:** `CoupledIMEXDriver` taking a tuple of per-equation functional steppers plus two user hooks: `prepare_step(states)->nonlinear_terms` (the convection) and `between_solves(states)->states` (the `compute_vw` reconstruction). Per stage (Python-unrolled): `prepare_step` once → each equation `compute_rhs` → each equation `solve_step` → `between_solves`. States are a **pytree** (dict of arrays, possibly different shapes); all updates functional. `solve` reuses `BaseIntegrator`'s batched `lax.fori_loop` over the pytree.
- **Depends on:** T3.2, T0.2.
- **Accept:** a synthetic 2-equation linearly-coupled system (`du/dt=L1 u + c·g`, `dg/dt=L2 g`) with analytic solution, driven with `IMEXRK222` per equation + identity `between_solves`, matches the analytic solution to scheme order; a 3-equation version where `between_solves` sets field 3 from fields 1–2 reproduces the rhs-before-solve, hook-after-solve ordering.

---

### M4 — Plane-Couette track (scripts 1, 2)

#### [T4.1] Ragged `VectorTensorProductSpace` transforms — *present/M* · `tensorproductspace.py`
- **Build:** make `VTPS.forward/backward/scalar_product/get_dealiased` handle components of unequal modal length (return tuples/pytrees, only `jnp.stack` in the homogeneous case). Depends on and completes T0.2.
- **Accept:** `BD=VTPS((TB,TD,TD))` round-trips per component to 1e-12; consumed correctly by `Project`, `integrate`, and the IMEX driver.

#### [T4.2] KMM velocity–vorticity orchestrator — *missing/XL* · `examples/channelflow_kmm.py`
- **Build:** a `KMM` class (the JAX analog of `ChannelFlow.py`) that:
  1. Builds spaces `B0`(biharmonic), `D0`(Dirichlet), `C0`(orthogonal), `F1`(complex `y`), `F2`(`z` per T0.3); tensor products `TB,TD,TC`; heterogeneous vectors `BD=[TB,TD,TD]`, `CC=[TD,TC,TC]`; padded `TDp`.
  2. Assembles the **`u` biharmonic** equation (`∂∇²u/∂t = ν∇⁴u + Dx-combination of H`) and the **`g` Helmholtz** equation (`∂g/∂t = ν∇²g + ∂N_y/∂z − ∂N_z/∂y`) as `TPMatrices` (Fourier×Fourier×banded-poly), pre-factorized via T2.2 (one vmapped call over all `(k_y,k_z)`).
  3. Implements `convection()` (conv==0): backward-transform `u` and its 9 gradients to the padded grid (cached `Project` from T1.2 + dealiased backward from T1.8), form the products, `TDp.forward` back, `mask_nyquist` (T1.5).
  4. Wires equations into `CoupledIMEXDriver` (T3.4) with `prepare_step=convection`, `between_solves=compute_vw`.
- **How:** state pytree carries `(u_hat[3], g_hat, H_hat[3], stage history)`; `g` is `1j·K[1]·u_hat[2] − 1j·K[2]·u_hat[1]` (functional, per T0.2). Single jitted `step`; `lax.scan` over stages. No MPI; `(0,0)` is global index 0.
- **Depends on:** M1, M2, M3, T4.1.
- **Accept:** see T4.5 (this class is exercised by the PCF example parity test).

#### [T4.3] `compute_vw`: divergence-constraint reconstruction + (0,0)-mode 1D Helmholtz — *missing/M* · `examples/channelflow_kmm.py`
- **Build:** for all modes: `f=∂u_x/∂x` (cached `Project`); `u_hat[1]=1j·(K_over_K2[0]·f + K_over_K2[1]·g)`, `u_hat[2]=1j·(K_over_K2[1]·f − K_over_K2[0]·g)` (T1.6). For the `(0,0)` mode: build standalone 1D Dirichlet `D00` + its 1D Helmholtz operator (factorized via T2.2), solve `v0` (source `−dpdy`, =0 for PCF) and `w0` (no source), write into `u_hat[...,0,0]` via `.at[...].set` (no rank gating). Called after the `u,g` solves each stage.
- **Depends on:** T2.2, T2.3, T1.6, T1.2.
- **Accept:** after `compute_vw`, `Project(div(u_),TC)` → `inner(1,divu²)` ≈ machine eps for non-`(0,0)` modes (script 2's check); `(0,0)`-mode `v00/w00` profiles match `shenfun`'s `pdes1d` solves to 1e-10. Get the `f=−∂u_x/∂x` sign convention right (the `shenfun` note).

#### [T4.4] PCF base-flow convection override — *missing/M* · `examples/pcf_fluctuations_jax.py`
- **Build:** subclass the KMM orchestrator; override `convection()` to add (in physical space) base-flow advection `U_b·∂(·)/∂y` on all three components and shear production `u_x·dU_b/dx` on the streamwise component, where `U_b=U_wall·x`, `dU_b/dx=U_wall`. Match script 1's component convention (0=wall-normal, 1=streamwise, 2=spanwise).
- **Depends on:** T4.2.
- **Accept:** part of T4.5.

#### [T4.5] PCF diagnostics + example script + parity test — *missing/M* · `examples/pcf_fluctuations_jax.py`, `tests/test_pcf_fluctuations.py`
- **Build:** the six diagnostics via `integrate()` (T1.1): `Epert=Σ integrate(u_i²)`, `Etot` (with base flow), `divL2=sqrt(integrate(divu²))`, `u_top/u_bot`, `mean_shear`. A runnable `main()` mirroring script 1's config. Extend the T0.4 harness to `steps>0`.
- **Depends on:** T4.2, T4.3, T4.4, T1.1.
- **Accept (the PCF definition-of-done):** from the same deterministic IC (`amp=0.1`, e.g. `N=(17,16,16)`), compare `u_hat` coefficient arrays after **1, 5, 50** `IMEXRK222` steps to rtol **1e-8** (float64) against `shenfun`'s `PlaneCouetteFluctuation`; the six diagnostics match to 1e-10. Script 2's `div(u)` stays at machine precision.

#### [T4.6] (optional) `compute_pressure` — inhomogeneous updating Neumann BC — *missing/M* · `examples/channelflow_kmm.py`
- **Why:** `shenfun` can recover pressure via a Neumann Poisson solve whose BC data `dp/dn = ν ∂²u/∂n²` updates every step (`ChannelFlow.py:253–268`). Not needed for the energy/divergence parity tests, but required if a script reports pressure.
- **Build:** a `BCGeneric` Neumann space with **field-valued, time-updating** BC data (project `ν·Dx(u_x,0,2)` each call), assemble the Neumann pressure-Poisson with `constraints=((0,0,0),)` (T2.3). Defer unless a target script needs pressure output.
- **Accept:** pressure field matches `shenfun` `compute_pressure()` to 1e-8 on a fixed state.

---

### M4b — PCF MHD (scripts 3, 4)

> Build *after* plain PCF works; reuse the coupled driver and the curl-projection machinery.

#### [T4.7] Magnetic vector-potential equations + compatible-space curl projections — *missing/L* · `examples/pcf_mhd_jax.py`
- **Build:** subclass the PCF solver; add the vector potential `A` (space `CD=[TD,TD,TD]`) with diffusion equations `∂A/∂t = η·div(grad(A)) + …` (a second `pdesA` equation set solved *after* `compute_vw`, via the coupled driver). Compute `B=curl(A)` into `[TD,TC,TC]` and `J=curl(B)` into `[TC,TD,TD]` using **per-component cross-space `Project`** (T1.3) — the compatible-space pairing is what makes `div(B)=0` hold discretely.
- **Depends on:** T1.3, T3.4, T4.2.
- **Accept:** seed `A`; `div(B)` L2 `< 1e-12` and equals `pcf_mhd_divfree.py` diagnostics; per-component `B` arrays match `shenfun`'s `projBx/By/Bz` to 1e-12.

#### [T4.8] Lorentz/EMF coupling + physical cross-product convection — *missing/M* · `integrators/nonlinear.py`,`operators.py`
- **Build:** `physical_cross(a,b)` (pointwise `jnp` cross of physical vector Arrays). Fold the Lorentz force `J×B` into the velocity nonlinear storage and the EMF `U×B` into the `A` equation (matching script 3's `H = N − J×B` convention). For script 4 (MRI): add Coriolis `2Ω` coupling, imposed uniform field `B0=(0,by,bz)`, shearing-box base `U_b=−S·x`.
- **Note:** scripts 1–4 *velocity* convection uses `conv==0` (grad form), so the velocity `u×curl(u)` path is **not** on the PCF critical path — but the MHD `U×B`/`J×B` cross-products are. Build `physical_cross` here.
- **Depends on:** T1.8 (dealiased transforms), T4.7.
- **Accept:** `physical_cross` matches `numpy.cross`; the projected `J×B`/`U×B` terms match `shenfun` for a seeded mode to 1e-10; script 4's Reynolds `<u_x u_y>` and Maxwell `<−b_x b_y>` stresses match to 1e-10.

---

### M5 — Taylor–Couette linear/MRI track (scripts 5, 6) — *independent, do early*

> No time-stepping, no dealiasing, no block solver. The cheapest complete win. Needs only M1's `1/r` form assembly and M2's dense `eig`.

#### [T5.1] Radial-coordinate symbol accessor + explicit `1/r` form verification — *partial/S + present/M* · `coordinates.py`, verify `inner.py`
- **Build:** expose `T.coors.psi[axis]` (the radial SymPy symbol) so scripts can write `r = TD.coors.psi[1]` and `(1/r)·Dx(u,1,1)`. **Verify** that `inner(test, coeff(r)·Dxⁿ(trial))` with `coeff ∈ {1/r, 1/r², m²/r²}` assembles correctly on a **Cartesian** radial space (the measure must *not* auto-apply `√det g`; the `1/r` is in the integrand explicitly). This is mostly verification — `jaxfun`'s `inner` already handles variable coefficients and multi-term coefficient lists.
- **Accept:** the cylindrical Laplacian block `_A(None,2) + _A(1/x,1) − _A(m²/x²+k_z²,0)` (taylor_couette_linear.py:215–221) matches `shenfun`'s assembled matrix to 1e-12.

#### [T5.2] Dense block extraction (`.diags().toarray()` analog) — *partial/S* · `la/matrix.py`
- **Build:** a method to materialize an assembled `inner` result (a `jaxfun` matrix, or a list of them for multi-term coefficients) as a dense complex `jnp`/`numpy` array, summing the list — mirroring the script's `_dense` helper (taylor_couette_linear.py:164–180).
- **Accept:** dense blocks match `shenfun`'s `SpectralMatrix.diags().toarray()` to 1e-13.

#### [T5.3] TC linear eigensolver example — *missing/M* · `examples/taylor_couette_linear_jax.py`
- **Build:** port `CircularCouette` + the eigenvalue assembly: build velocity Dirichlet space + truncated pressure space (`dim Nr−2`), assemble `L,M` blocks via T5.1/T5.2, solve `L φ = λ M φ` via T2.4, return growth rates.
- **Depends on:** T2.4, T5.1, T5.2.
- **Accept (script 5 definition-of-done):** leading eigenvalue(s) for a standard TC case match `shenfun` `taylor_couette_linear.py` to 1e-8; the critical Reynolds/Taylor number matches.

#### [T5.4] TC MRI eigensolver + insulating Bessel BCs — *missing/M (+M2)* · `examples/taylor_couette_mri_jax.py`
- **Build:** extend T5.3 with the magnetic fields and **insulating/vacuum magnetic BCs** using modified Bessel functions `scipy.special.iv/kv` (log-derivative matching at the walls, taylor_couette_mri.py). Robin/Neumann radial bases via the composite stencil machinery.
- **Depends on:** T5.3, T0.5 (Neumann bases).
- **Accept (script 6 definition-of-done):** MRI growth rates and critical parameters match `shenfun` `taylor_couette_mri.py` to 1e-6 (eig tolerance); the vacuum-match BC reproduces the reference dispersion relation.

---

### M6 — Taylor–Couette DNS (script 7) — *deepest chain, do last*

> The mixed velocity–pressure space (T6.1) is the single most important predecessor — **start it early in the TC track** even though the DNS lands last.

#### [T6.1] Mixed velocity–pressure `CompositeSpace` + truncated-orthogonal pressure — *missing/XL* · `tensorproductspace.py`
- **Build:** (A) a `CoupledSpace(list[TensorProductSpace])` mirroring `shenfun` `CompositeSpace`: stack arbitrary unlike sub-spaces with different modal dims; `forward/backward/scalar_product` map each component through its own space (return pytree, no `jnp.stack`); expose `block_sizes`/`slices` for flat↔block packing. (B) a **first-class truncated-orthogonal pressure space** (`FunctionSpace(Nr, family, num_dofs=Nr-2)` or a dedicated `PressureSpace`) giving the inf-sup-stable `P_N/P_{N−2}` pair — replace `shenfun`'s runtime `SP.slice` monkey-patch (`jaxfun`'s `num_dofs` is a read-only `@property`, so add a constructor option).
- **Accept:** build `VQ=[TD,TD,TD,TP]`; assemble the coupled operator from `inner` over the mixed space; pressure space `dim == velocity dim == Nr−2`; the assembled block system is square. (Solve checked in T6.2.)

#### [T6.2] Per-Fourier-mode coupled saddle-point block solver + pressure gauge — *missing/XL* · `la/blocktpmatrix.py`
- **Build:** a wavenumber-aware block solver: for each axial Fourier mode `k`, build the small dense coupled matrix `[block-dofs × block-dofs]` from the per-block per-axis factors at `k`; stack into `(n_modes, M, M)` and `vmap` `jax.scipy.linalg.lu_factor/lu_solve` over modes (one XLA call). Pin the `k=0` pressure dof via T2.3 (`constraints=((3,0,0),)`). Provide flat↔(block,mode,dof) scatter/gather. (`jaxfun`'s current `BlockTPMatrix.solve` assembles one global RCM+LU — functionally correct but not per-mode; replace it for the DNS.)
- **Depends on:** T6.1, T2.3, T2.1.
- **Accept:** steady Stokes-in-annulus MMS: velocity+pressure match `shenfun` `BlockMatrixSolver` for the same `(Nr, family, k_z)` to 1e-10; `div(u)=0` to machine precision; `k=0` mean pressure pinned to 0.

#### [T6.3] CNAB2 integrator — *missing/L* · `integrators/cnab2.py`
- **Build:** `setup(dt)`: `Limp = (1/dt)·M − ½A (+ pressure-coupling blocks)`, `Lexp = (1/dt)·M + ½A`; warm `Limp`'s solver. `step`: `N_hat=scalar_product N(u)`; `rhs = Lexp@u − (1.5·N_hat − 0.5·N_old)`; set continuity rows to 0 (`.at[...].set(0)`); `u_new = Limp.solve(rhs, constraints=pressure_pin)`; return `(u_new, N_hat)`. **IMEX-Euler bootstrap** for step 0: either run step 0 eagerly with weights `(1.0, 0.0)` then `lax.scan` the rest with `(1.5, −0.5)` (recommended), or carry `have_old` in the scan carry and select weights with `jnp.where`. `N_old` is loop-carried state, never a mutated attribute. Note `M/dt` (mass scaled by `1/dt`), so changing `dt` re-runs `setup`.
- **Depends on:** T6.2, T6.1.
- **Accept:** 2nd-order `dt`-convergence on scalar advection–diffusion with analytic solution; the first-step bootstrap preserves global 2nd order; on a small TC setup the CNAB2 trajectory (velocity+pressure) matches `taylor_couette_dns.py` for the first ~5 steps to 1e-8; divergence stays at machine precision.

#### [T6.4] Cylindrical nonlinear convection + `inv_r_p` — *missing/S* · `examples/taylor_couette_dns_jax.py`
- **Build:** the nonlinear term with cylindrical metric pieces (`n_r = u_r ∂u_r/∂r + u_z ∂u_r/∂z − u_θ²/r`, etc., taylor_couette_dns.py:251–272), evaluated on the (radially-padded) grid using the padded radial reciprocal `inv_r_p = 1/r_padded`.
- **Depends on:** T1.8, T6.5.
- **Accept:** nonlinear coefficients match `shenfun` for a seeded eigenmode to 1e-10.

#### [T6.5] Radial polynomial dealiasing parity spike — *done/M* · `orthogonal.py` (+ test)
- **Why (highest TC numerical risk):** TC pads the **radial polynomial** axis. `shenfun` zero-pads the *orthogonal* coefficients then re-applies the composite stencil; `jaxfun`'s polynomial `backward(c,N)` over-samples on a *finer Gauss mesh* — a *different* physical mesh. A product formed on `jaxfun`'s grid then truncated must still equal `shenfun`'s to round-off.
- **Build:** a focused spike: form a quadratic product of band-limited radial fields both ways (`jaxfun` finer-Gauss vs `shenfun` padded-DCT-on-orthogonal-coeffs), truncate, compare. If they diverge, implement the coefficient-zero-pad-on-orthogonal-then-restencil path to match `shenfun`.
- **Accept:** `tests/couette/test_live_shenfun_parity.py::test_radial_polynomial_dealiasing_matches_live_shenfun_product` matches live `shenfun` to 1e-12 away from the full-complex Fourier Nyquist convention; solver products mask Nyquist.

#### [T6.6] TC DNS example + eigenmode-seeded parity — *missing/M* · `examples/taylor_couette_dns_jax.py`
- **Build:** the `AxisymmetricTCDNS` analog assembling `Limp`/`Lexp` (T6.1/T6.2), CNAB2 stepping (T6.3), cylindrical nonlinear (T6.4), `r`-weighted diagnostics (T6.7). Seed the leading eigenmode from T5.3 to validate the growth rate.
- **(Optional) principled curvilinear path:** as an alternative, build the spaces with a polar/cylindrical `CoordSys` (auto `√det g = r`) and drop the explicit `1/r`. Keep it separate and clearly labeled — do not mix with the explicit-`1/r` path.
- **Depends on:** all of M6, T5.3.
- **Accept (script 7 definition-of-done):** seeded with a `TaylorCouetteLinear` eigenmode, the DNS linear growth rate matches `shenfun` over 100 CNAB2 steps to 1e-6; `k=0` pressure pinned; `div(u)=0` to machine precision throughout.

#### [T6.7] `r`-weighted volume-integral diagnostics — *done/M* · `diagnostics.py`
- **Build:** implemented `jaxfun.diagnostics` helpers for cylindrical kinetic/magnetic/component energies and wall norms; TC DNS examples use explicit `rphys` weights.
- **Accept:** `tests/couette/test_taylor_couette_dns_jax.py::test_tc_diagnostics_helpers_match_solver_outputs` covers hydro and MRI helper parity with solver diagnostics.

---

### M7 — I/O, diagnostics cadence & multi-backend

#### [T7.1] HDF5 field output + checkpoint/restart + XDMF — *done/M* · `src/jaxfun/io/__init__.py`
- **Build:** implemented optional `h5py` writer under the `io` extra: uniform snapshots from `JAXFunction`/physical `Array`/raw coefficient+space pairs, exact pytree checkpoints with `t/tstep` attrs, `read_checkpoint`, and `generate_xdmf`. Single-device host IO only.
- **Accept:** `tests/io/test_hdf5.py` covers exact write→read, bit-identical restart continuation for 20 steps, uniform snapshot output, and XDMF generation. Live `shenfun` file-output parity remains unclaimed.

#### [T7.2] Host-callback diagnostic/IO cadence — *partial/M* · `src/jaxfun/io/__init__.py`, `examples/`
- **Build:** generic `Cadence` + `run_with_cadence(advance, state, ...)` runs compiled stepping blocks and invokes diagnostics/snapshot/checkpoint callbacks on exact host-side cadence boundaries. PCF and TC DNS examples now expose `--moderror`/`--block-size` and call `solve_with_cadence`; graceful-stop behavior remains open.
- **Accept:** small PCF and TC cadence tests show cadence-blocked solves match direct solves and fire diagnostics on exact step boundaries. A 200-step CLI cadence/no-recompile benchmark remains unclaimed.

#### [T7.3] MPI→sharding: remove rank gating, sharded reductions, multi-device parity — *partial/L* · `examples/`, `sharding.py`
- **Build:** (1) remove all rank gating — the `(0,0)` mode is global index 0 (done in T4.3); (2) energy/spectra reductions = `jnp.sum` (single device) or `jax.lax.psum(_,'k')` inside `shard_map` (multi-device); (3) plane-averaged profiles = mean over homogeneous directions locally then `psum`, or `all_gather` over the wall-normal-distributed axis + deterministic reorder; (4) ensure the banded (wall-normal/radial) axis is **not** the sharded axis for the per-mode solver (`jaxfun`'s sharding shards a Fourier axis; satisfy the `poly_axis != 0`, `n_F divisible by device count` constraints).
- **Current coverage:** `tests/couette/test_sharding_parity_jax.py` covers two-device TC diagnostics and physical transforms for axisymmetric/full-3D hydro and axisymmetric/full-3D MHD states on a Fourier-sharded axis.
- **Accept:** the same problem on 1 device and on N devices (or N CPU shards via `XLA_FLAGS`) yields **bit-identical** `u_hat` and diagnostics after stepping; single-device spectra match `shenfun`'s MPI-allreduced spectra to 1e-10. Runs on GPU and TPU unchanged (`jax.devices()` abstracts the backend — no GPU/TPU branching).

---

## 8. Validation & parity harness (used by every acceptance test)

The harness from **T0.4** (`tests/_parity.py`) is the backbone:

1. **Same IC, both solvers.** Build the deterministic IC once; instantiate the `shenfun` reference (using the `_load_shenfun_demo` shim from `examples/pcf_fluctuations.py`, which null-routes `ShenfunFile`/`Checkpoint`) and the `jaxfun` port.
2. **Run to N steps** with identical `dt`, `ν`, scheme, resolution.
3. **Compare** (a) spectral coefficient arrays per field (`np.asarray`, `rtol` per the task), and (b) the scalar diagnostics.
4. **Tolerances:** unit primitives 1e-12/1e-13; assembled operators 1e-12; time-evolution trajectories 1e-8 to 1e-10 (float64); eigenvalues 1e-6 to 1e-8.
5. **Always float64.** A test that needs float32 to pass is failing.

Run the suite with `uv run pytest`. Mark the slow shenfun-comparison tests `@pytest.mark.slow` (per `jaxfun`'s existing convention; `uv run pytest -m "slow or not slow"` runs them).

**Differentiability check (the project's headline goal):** for at least one PCF and one TC case, assert `jax.grad` of a scalar diagnostic (e.g. final `Epert`) w.r.t. a control (e.g. `U_wall`, `Re`, or the IC amplitude) returns finite values and matches a finite-difference estimate to a few digits. This proves the solver is end-to-end differentiable.

---

## 9. Per-script definition of done

| Script | Done when… | Gating tasks |
|---|---|---|
| `taylor_couette_linear.py` (5) | leading eigenvalue + critical Ta match `shenfun` to 1e-8 | M0, T5.1–T5.3, T2.4 |
| `pcf_fluctuations_corrected.py` (1) | `u_hat` matches to 1e-8 at 1/5/50 IMEXRK222 steps; 6 diagnostics to 1e-10 | M0–M3, T4.1–T4.5 |
| `pcf_fluctuations_divV.py` (2) | `div(u)` stays at machine precision; diagnostics match | + T4.3 |
| `pcf_mhd_divfree.py` (3) | `div(B)<1e-12`; per-component `B` matches to 1e-12 | + T4.7, T4.8 |
| `pcf_mhd_mri_shearpy.py` (4) | Reynolds/Maxwell stresses match to 1e-10 | + T4.8 |
| `taylor_couette_mri.py` (6) | MRI growth rates match to 1e-6; insulating BC reproduces dispersion | + T5.4 |
| `taylor_couette_dns.py` (7) | eigenmode-seeded growth rate matches over 100 steps to 1e-6; `div(u)`≈0 | + all M6 |
| All | `jax.grad` of a diagnostic is finite & FD-consistent; runs on CPU/GPU/TPU; multi-device bit-identical | + M7 |

---

## 10. Appendices

### 10.1 IMEX Butcher tableaux — copy verbatim (`shenfun/shenfun/utilities/integrators.py`)

**`PDEIMEXRK` stage update** (the contract for IMEXRK011/111/222/443):
```
assemble:   S = M − dt·a[1,1]·L           # single implicit operator (a[1,1] constant per ARS 2.3)
compute_rhs(rk):
    K[rk] = scalar_product( N(u_stage) )                     # nonlinear, NOT mass-inverted
    rhs   = M·u^n                                            # u0_rhs, computed once at rk=0
    rhs  += Σ_{j=0..rk}   dt·b[rk+1, j]   · K[j]             # explicit nonlinear history
    if rk>0:
        L[rk-1] = scalar_product( L·u_stage )               # explicit linear history (no M⁻¹)
        rhs    += Σ_{j=0..rk-1} dt·a[rk+1, j+1] · L[j]
    mask_nyquist(rhs)
solve_step(rk):  u_stage = S.solve(rhs)
```

**IMEXRK222** (2 stages; default in the run scripts):
```
γ = (2 − √2)/2 ;  δ = 1 − 1/(2γ)
a = [[0,0,0],[0,γ,0],[0,1−γ,γ]]          # implicit (DIRK), active diagonal = γ
b = [[0,0,0],[γ,0,0],[δ,1−δ,0]]          # explicit
c = (0, γ, 1)
```

**IMEXRK443** (4 stages; a[1,1]=1/2):
```
a = [[0,0,0,0,0],[0,1/2,0,0,0],[0,1/6,1/2,0,0],[0,−1/2,1/2,1/2,0],[0,3/2,−3/2,1/2,1/2]]
b = [[0,0,0,0,0],[1/2,0,0,0,0],[11/18,1/18,0,0,0],[5/6,−5/6,1/2,0,0],[1/4,7/4,3/4,−7/4,0]]
c = (0, 1/2, 2/3, 1/2, 1)
```

**IMEXRK3** (Spalart low-storage, 3 stages; KMM default; **different implicit factor per stage**):
```
a = (8/15, 5/12, 3/4) ;  b = (0, −17/60, −5/12) ;  c = (0, 8/15, 2/3, 1)
per stage rk:  S_rk = M − ((a[rk]+b[rk])·dt/2)·L      # three factorizations
               w0   = scalar_product( N(u_stage) )
               rhs  = dt·(a[rk]·w0 + b[rk]·w_prev) + ( M·u_stage + ((a[rk]+b[rk])·dt/2)·(L·u_stage) )
               w_prev = w0 ;  u_stage = S_rk.solve(rhs)
```

**CNAB2** (TC DNS; θ=½ CN + AB2 + IMEX-Euler bootstrap):
```
Limp = M/dt − ½A (+ grad/div pressure block) ;  Lexp = M/dt + ½A
N_hat = nonlinear(u^n)
rhs_i = (Lexp·u^n)_i − ( 1.5·N_hat_i − 0.5·N_old_i )     # step 0: − N_hat_i  (bootstrap)
rhs_continuity = 0
u^{n+1} = Limp.solve(rhs, constraints=((3,0,0),))         # pin pressure dof 0 of mode 0
N_old ← N_hat
```

### 10.2 KMM equations (the two evolved scalars)

```
u (wall-normal velocity), biharmonic:
  ∂/∂t (∇²u) = ν ∇⁴u + ∂²N_y/∂x∂y + ∂²N_z/∂x∂z − ∂²N_x/∂y² − ∂²N_x/∂z²
  basis B0: bc=(0,0,0,0)  (u=u'=0 both walls, dim N−4)   solver: Biharmonic

g (wall-normal vorticity g = ∂u_z/∂y − ∂u_y/∂z), Helmholtz:
  ∂g/∂t = ν ∇²g + ∂N_y/∂z − ∂N_z/∂y
  basis D0: bc=(0,0)  (dim N−2)                           solver: Helmholtz

reconstruction (k_y,k_z)≠(0,0):  f = ∂u_x/∂x ;  K_over_K2[i] = K[i+1]/(k_y²+k_z²)
  u_y = 1j·(K_over_K2[0]·f + K_over_K2[1]·g)
  u_z = 1j·(K_over_K2[1]·f − K_over_K2[0]·g)
(0,0) mode: two 1D Helmholtz momentum equations on standalone space D00.
PCF base flow: U_b = U_wall·x ; add U_b·∂(·)/∂y to all comps, u_x·U_wall to streamwise comp.
```

### 10.3 Taylor–Couette saddle-point operator (Cartesian space, explicit `1/r`)

```
unknown VQ = (u_r, u_θ, u_z, p) on CompositeSpace([TD,TD,TD,TP]), TP = orthogonal truncated to Nr−2
cylindrical Laplacian:  _lap(u) = Dx(u,1,2) + (1/r)·Dx(u,1,1) + Dx(u,0,2)     # axis1=radial, axis0=axial
implicit block (per axial Fourier mode), Limp:
  vr:  ur/dt − ½ν·_lap(ur) + ½ν·(1/r²)·ur − ½·2Ω·uθ + Dx(p,1,1)
  vθ:  uθ/dt − ½ν·_lap(uθ) + ½ν·(1/r²)·uθ − ½·(−2a)·ur
  vz:  uz/dt − ½ν·_lap(uz) + Dx(p,0,1)
  q (continuity): Dx(ur,1,1) + (1/r)·ur + Dx(uz,0,1) = 0
explicit block Lexp = velocity-velocity (M/dt + ½A), matvec only.
Ω(r) = a + b/r² ;  pin pressure with constraints=((3,0,0),).
```

### 10.4 Taylor–Couette eigenvalue assembly (linear stability)

```
for each (m, k_z):
  block(coeff, order) = inner(test, coeff(r)·Dx(trial,0,order)) → SpectralMatrix(.diags().toarray())
                        (sum the list when coeff has several additive terms, e.g. Ω=a+b/r²)
  Laplacian = block(1,2) + block(1/r,1) − block(m²/r² + k_z², 0)
  assemble L (= L0 + ν·Lv) and M ; solve  L φ = λ M φ  via scipy.linalg.eig ; growth = Re(λ)
  MRI insulating walls: match log-derivative to vacuum modified-Bessel solution (scipy.special iv/kv).
```

### 10.5 Key file index

| Concern | shenfun reference | jaxfun target |
|---|---|---|
| Channel-flow base class | `shenfun/demo/ChannelFlow.py` (KMM) | `jaxfun/examples/channelflow_kmm.py` |
| IMEX integrators | `shenfun/shenfun/utilities/integrators.py` (PDEIMEXRK, IMEXRK222/3) | `jaxfun/src/jaxfun/integrators/imex_rk.py`, `cnab2.py`, `coupled.py` |
| Fast banded solvers | `shenfun/shenfun/chebyshev/la.py`, `shenfun/shenfun/la.py` (SolverGeneric1ND, BlockMatrixSolver) | `jaxfun/src/jaxfun/la/{tpmatrix,diamatrix,blocktpmatrix,solvers,eig}.py` |
| Forms / project / operators | `shenfun/shenfun/forms/{inner,project,operators,arguments}.py` | `jaxfun/src/jaxfun/galerkin/inner.py`, `jaxfun/src/jaxfun/operators.py` |
| Spaces / BCs / dealiasing | `shenfun/shenfun/{spectralbase,tensorproductspace}.py`, `chebyshev/bases.py` | `jaxfun/src/jaxfun/galerkin/{composite,functionspace,tensorproductspace,orthogonal,Fourier}.py` |
| Coordinates / measure | `shenfun/shenfun/coordinates.py` | `jaxfun/src/jaxfun/coordinates.py` |
| I/O | `shenfun/shenfun/io/__init__.py` | `jaxfun/src/jaxfun/io/__init__.py` |
| Parallelism | `mpi4py` `comm` throughout shenfun | `jaxfun/src/jaxfun/sharding.py` |
| TC base + linear/DNS | `couette/taylor_couette_{linear,mri,dns}.py` | `jaxfun/examples/taylor_couette_{linear,mri,dns}_jax.py` |

---

*End of Part I. Part II (below) records implementation status as of 2026-06-01 and extends the plan to the four Taylor–Couette DNS quadrants and the stability-analysis layer.*

---
---

# Part II — Status Review & Extended Plan (2026-06-01)

This part supersedes Part I's status claims. It is based on a full audit of branch `couette-jax-implementation`. **Golden rule for this part:** an item marked *done/partial* is **internally self-consistent only**; the phrase "**not shenfun-validated**" means no test compares it to a live `shenfun` run (see the M0b blocker). Do not silently upgrade "done" to "parity-validated" downstream.

> **Implementation update (2026-06-01):** after this audit, the branch added the reusable CNAB2/coupled-IMEX helpers, named Helmholtz/Biharmonic solvers, cached `Project`, physical `Array`, optional HDF5/checkpoint/XDMF IO, host-side cadence runner, differentiability checks, and live `shenfun` parity coverage for PCF and PCF-MHD (divergence-free + shearpy/MRI) 1/5/50-step diagnostics plus mapped coefficient fields, axisymmetric and full-3D hydro/MHD TC DNS 1/5/50-step diagnostics plus mapped coefficient fields, TC linear/MRI including non-modal growth, and the radial dealiased product. Older status tables below are retained for context where not explicitly updated.

## 11. Status review

### 11.1 Implemented & internally tested — *NOT shenfun-validated*

| Plan ref | Item | Evidence | Caveat |
|---|---|---|---|
| T0.5 | tuple→dict BC adapter (`bc=(0,0,0,0)`/`(0,0)`) + biharmonic | `galerkin/functionspace.py`; `test_functionspace_couette_compat.py` | Live `shenfun` stencil parity now compares Legendre/Chebyshev `ShenDirichlet` and `ShenBiharmonic` compact rows to 1e-13. |
| T1.4 | `Dx(u, axis, k)` wrapper | `operators.py`; `test_dx_operator.py` | matrices not compared to `shenfun` |
| T2.4 | dense `generalized_eig` + singular-`M` filtering | `la/eig.py`; `test_eig.py` | only a 2×2 toy test; `finite_cap=1e6` here vs reference `FINITE_CAP=1e8` — **keep these two caps distinct & documented** (modal filter vs non-modal cap) |
| T5.1 | radial symbol + explicit `1/r` form assembly | `taylor_couette_linear_jax.py` | — |
| T5.2 | dense block extraction (`.diags().toarray()` analog) | `la/matrix.py` todense | — |
| T6.1 | mixed `CoupledSpace` + truncated-orthogonal pressure (`P_N/P_{N-2}`) | `tensorproductspace.py`; `test_coupled_space_tc_dns.py` | — |

**Runnable solver examples** (correct-looking, smoke/self-consistency tested, **not** `shenfun`-parity):
- **KMM PCF** (`channelflow_kmm.py`, `pcf_fluctuations_jax.py`): real biharmonic-`u` + Helmholtz-`g`, vmapped per-mode banded LU, `compute_vw`, base-flow convection. Live `shenfun` integration coverage now compares IMEXRK222 1/5/50-step diagnostics, reconstructed physical velocity fields, and mapped coefficient fields. The coefficient test pads compact jax composite coefficients and slices the nonnegative spanwise modes to match `shenfun`'s rfft half-spectrum storage.
- **PCF-MHD** (`pcf_mhd_jax.py`, `pcf_mhd_mri_shearpy_jax.py`): A-diffusion + compatible-space curl projections; `divB ≈ 2e-16` (weak). Both the divergence-free PCF-MHD path and the shearpy/MRI transport extension now have live `shenfun` 1/5/50-step diagnostic and mapped `(u,g,A)` coefficient parity.
- **Taylor-Couette hydro DNS** (`taylor_couette_dns_jax.py`): CNAB2 + per-mode vmapped saddle solves + cylindrical nonlinear + r-weighted energy. Live `shenfun` coverage now compares axisymmetric and full-3D 1/5/50-step diagnostics plus mapped `(u,p)` coefficient fields; eigenmode growth remains covered by the jax linear solver consistency tests.
- **Taylor-Couette MHD DNS** (`taylor_couette_dns_jax.py`): axisymmetric and full-3D conducting-wall MRI solvers now have live `shenfun` 1/5/50-step diagnostics plus mapped `(u,b,p)` coefficient parity for linear-eigenmode seeds.
- **TC linear** (`taylor_couette_linear_jax.py`) and **TC MRI** (`taylor_couette_mri_jax.py`) eigensolvers: conducting + insulating Bessel BCs; live `shenfun` coverage compares eigenvalues and non-modal growth rows.

### 11.2 Partial — finish these (grouped)

**A. Refactor inline example code into reusable library modules** (the MHD/3D quadrants must *reuse*, not re-inline):
- **Done after audit:** T3.4 coupled multi-equation IMEX stage helper (`integrators/coupled.py`), T6.3 reusable CNAB2 stepping (`integrators/cnab2.py`), T3.1 shared implicit-operator helpers (`integrators/base.py`), T2.2 named `Helmholtz`/`Biharmonic` constructors (`la/solvers.py`), T1.2 cached `Project`, and T1.6 `K_over_K2`.
- **Done after audit:** T6.7 r-weighted TC diagnostics are factored into `jaxfun.diagnostics`; TC DNS examples call the reusable helpers.

**B. Validation gaps:** live `shenfun` parity now covers PCF IMEXRK222 diagnostics/physical velocity/coefficient fields at 1/5/50 steps, both PCF-MHD variants' diagnostics/coefficient fields at 1/5/50 steps, axisymmetric and full-3D hydro/MHD TC DNS diagnostics/coefficient fields at 1/5/50 steps, TC linear/MRI eigenvalues plus non-modal growth, and the radial dealiased product. Remaining validation work is broader-size/dealiased production coverage and low-level operator stencil cross-checking; the tuple-BC basis stencil check now covers `ShenDirichlet`/`ShenBiharmonic` parity.

**C. Correctness gaps in the batched solver tier:**
- **Done after audit:** T2.1/T2.3 are now plumbed into `TPMatricesWavenumberSolver` and `tpmats_wavenumber_factor` as opt-in `pivot=True` and `constraints=((flat_mode, row, value), ...)` dense batched LU paths. Regression coverage includes pivoted parity vs assembled Kronecker solve, a zero-diagonal pivot stress case, and constrained mode-row/RHS pinning.
- **Done after audit:** T3.1 `BaseIntegrator.build_implicit_operator(coefficient, dt)` and `apply_linear_scalar_product` now provide the named base API; BackwardEuler, ARS IMEX-RK, and IMEXRK3 route through it.
- **T1.8/T6.5** radial polynomial dealiasing now has a live `shenfun` regression for a padded radial/Fourier quadratic product. The remaining convention caveat is the full-complex Fourier Nyquist mode; solver nonlinear products call `mask_nyquist`.

### 11.3 Missing — not started

| Plan ref | Item | Note |
|---|---|---|
| T4.6 | `compute_pressure` (Neumann pressure recovery) | optional/deferred; pressure-free KMM diagnostics do not require it |
| T7.3 | stepped all-quadrant sharding parity for Couette workflows | TC axisymmetric/full-3D hydro/MHD diagnostics and physical-transform parity are covered with `--num-devices=2`; bit-identical stepped coefficient parity remains to validate |

### 11.4 Known correctness bugs / risks (fix early)

1. **No-pivot batched LU at production N:** `TPMatricesWavenumberSolver` still defaults to `vmap(_lu_banded_no_pivot_kernel)` for the fast path, with an opt-in dense pivoted path now available. Production-size validation must decide when KMM/biharmonic workloads should stay on the fast path versus request `pivot=True`.
2. **Couette sharding parity gap:** TC axisymmetric/full-3D hydro/MHD transform and diagnostic parity has a two-device test; all-quadrant stepped coefficient parity on multiple devices is still not a completed acceptance gate.
3. **`finite_cap` split:** modal filter `1e6` (`eig.py`) vs non-modal `1e8` (`_linear_analysis.py`) must stay **distinct and documented** — unifying them silently changes which large-but-finite modes the modal filter discards.

### 11.5 Corrections to Part I

- **T0.3 decided → full-complex Fourier (option B).** `jaxfun` Fourier uses single-axis `jnp.fft.fft`; `mask_nyquist`, K-scaling, the `(0,0)` mode, and the saddle solve all assume it. The documented rfft-layout parity mapping for comparing to `shenfun`'s real-`z` half-spectrum is still owed (`docs/couette_fourier_layout.md`).
- **`couette/_linear_analysis.py` EXISTS** (not missing): `FINITE_CAP=1e8`, `finite_eigensystem`, `transient_growth_from_eigs` (SVD-of-propagator energy-norm growth), `parse_times`, `print_*`. `couette/_pcf_linear.py` exists (`PlaneCouetteLinear`, `energy_matrix(kind)`, `nonmodal_growth`). These define a **stability-analysis layer with no jax port** → **M8**.
- **§5 master gap map:** mark items 5, 9, 18, 30, 32, 35 (T0.5, T1.4, T2.4, T5.1, T5.2, T6.1) as **present (internally tested, not shenfun-validated)**.
- **§1.2 / §6 / §9:** script 7 (`taylor_couette_dns.py`) is **four** reference classes, not one — split M6 and add the three missing quadrants (§12).

---

## 12. Extended milestones

### 12.0 Scope: the 2×2 Taylor–Couette DNS matrix

The `shenfun` reference defines **four** DNS classes; the jax port has **only the top-left**:

| | **Hydrodynamic** | **MHD (MRI)** |
|---|---|---|
| **Axisymmetric** (`m=0`) | `AxisymmetricTCDNS` (ref :91) — **PORTED** (`AxisymmetricTCDNSJax`) | `AxisymmetricMRIDNS` (ref :788) — **M12** |
| **3D** (`m≠0`) | `TaylorCouetteDNS` (ref :487) — **M10** | `TaylorCouetteMRIDNS` (ref :1196) — **M13** |

Two **shared prerequisites** keep the four quadrants from duplicating work:
- **M9 — azimuthal Fourier (`m≠0`) machinery** + two-Fourier-axis per-`(m,kz)` block solve → shared by **M10** and **M13**.
- **M11 — magnetic induction/Lorentz/EMF machinery** (primitive `b`, imposed axial `B0`, total pressure `Π=p+B0·b_z`, **conducting** walls) → shared by **M12** and **M13**.

**Dependency edges:** M13 = M9 ∪ M11. M10 needs M9; M12 needs M11; both need the finished axisymmetric-hydro substrate (M6) + CNAB2 module (T6.3).

**Out of scope (do not build):** *insulating-wall MHD DNS*. The `shenfun` reference MHD **DNS** classes are conducting-wall only (`b_r=0` Dirichlet, `b_θ` Robin `d(r·b_θ)/dr=0`, `b_z` Neumann). The `--magnetic-bc insulating` flag routes to the **linear** MRI eigensolver (`m=0` only), which is already ported (T5.4). Building an insulating DNS path is wasted effort.

> All Part-I tasks (T*.b excepted) keep their original IDs. New tasks use the milestone's number. Every new quadrant/analysis deliverable **must carry an x64 eigenmode-growth (or live-`shenfun`-coefficient) PARITY test, not a smoke test** — and those become meaningful only after M0b lands.

---

### M0b — Foundation hardening (DO FIRST; gates every parity claim)

**[T0.1b] Enable float64 at import — *S*** · `src/jaxfun/__init__.py`, `tests/conftest.py`
- **What:** add `jax.config.update("jax_enable_x64", True)` at the top of `src/jaxfun/__init__.py` (before any array creation) with a warn-if-already-x32 guard; **remove** the conftest `--float64` gating so x64 is default. **Where:** library init + conftest + a new x64 regression test.
- **Accept:** `import jaxfun; jnp.zeros(1).dtype == float64`; a space's quadrature points are float64; default `uv run pytest` no longer skips the couette parity tests; **and the FULL suite (not just couette) is green at x64** (guard the ~692 generic tests currently running at float32/2e-6 against dtype-sensitive regressions).

**[T0.4b] Live `shenfun` parity harness — *L*** · `tests/_parity.py` (new) + rewire `tests/couette/`
- **What:** a deterministic IC builder reproducing script-1's field, and `compare_to_shenfun(jax_solver, shen_solver, steps, fields, tol)` running **both** solvers from the same IC (shenfun on CPU via a `_load_shenfun_demo`-style shim) comparing `u_hat` coefficient arrays + the six diagnostics after 1/5/50 steps. Replace the six hardcoded golden literals with live or provenance-documented references. **Depends:** T0.1b. **shenfun ref:** §8; `couette/pcf_fluctuations_corrected.py`.
- **Accept:** runs end-to-end on a viscous-decay case and on PCF at 1/5/50 `IMEXRK222` steps; PCF `u_hat` matches `shenfun` to **1e-8** and the six diagnostics to **1e-10**; TC linear/MRI eigenvalues match a live `scipy.linalg.eig` `shenfun` run to **1e-8/1e-6** with deterministic conjugate-pair ordering (consumes T8.0).

**[T0.4c] Differentiability smoke test — *M*** · `tests/couette/`
- **What:** `jax.grad`/`value_and_grad` of a scalar diagnostic (final `Epert`) w.r.t. a control (`U_wall`/`Re`/IC amplitude) for one PCF and one axisymmetric-TC case. **Depends:** T0.1b.
- **Accept:** gradient is finite and FD-consistent to a few digits for both cases (the headline differentiable-solver goal; currently zero coverage).

---

### M8 — Stability-analysis layer: modal eig + non-modal transient growth (PCF & TC, hydro & MHD)

*Rationale:* the reference ships `couette/_linear_analysis.py` (SVD-of-propagator transient growth) and `couette/_pcf_linear.py` (energy-norm choice); the jax side has only modal `eig`. *Depends:* M2 (T2.4), M5. **Note:** M8 *code* depends only on M2/M5, but M8 *acceptance* depends on T0.4b (live reference) and on `couette/_linear_analysis.py` importing cleanly in the venv.

**[T8.0] Shared jax linear-analysis primitives — *M*** · `la/eig.py` (extend) + `la/__init__.py`
- **What:** `finite_eigensystem(L,M,finite_cap)->(w,V)` (finite eigenpairs, descending growth, **deterministic secondary Im sort key** — fixes the MRI ordering bug); `transient_growth_from_eigs(evals,evecs,metric,times)` = largest singular value of the metric-weighted modal propagator `V·exp(Λt)·a` with gauge/null-direction removal; `parse_times`. Reconcile `FINITE_CAP=1e8` for the non-modal cap while keeping the `1e6` modal-filter default **documented and distinct**. **shenfun ref:** `couette/_linear_analysis.py:15-87`.
- **Accept:** `finite_eigensystem` eigenvalues match `couette/_linear_analysis.finite_eigensystem` elementwise to 1e-12 with stable conjugate-pair order; `transient_growth_from_eigs` G(t) matches the reference to 1e-10 for t∈{0,1,5}.

**[T8.1] Energy-norm metric (total/kinetic/magnetic) — *M*** · TC eigen examples + new PCF linear module
- **What:** `energy_matrix(kind)` building the Hermitian PSD state metric `Q` (radial-quadrature-weighted velocity vs magnetic blocks), feeding T8.0. **shenfun ref:** `couette/_pcf_linear.py:247-270`, `couette/taylor_couette_mri.py:482-531`. **Depends:** T8.0.
- **Accept:** kinetic-norm non-modal growth reduces to the hydro result at `B0=0` to 1e-10; `Q` matches the reference `energy_matrix(kind)` to 1e-12 for all three norms.

**[T8.2] Wire `--nonmodal`/`--linear-energy` into TC eigen examples — *M*** · `taylor_couette_{linear,mri}_jax.py`
- **What:** `nonmodal_growth(m,kz,times,n_modes,energy=...)` + CLI flags + `print_transient_growth`. **shenfun ref:** `couette/taylor_couette_linear.py:337-341`, `taylor_couette_mri.py:533-545`. **Depends:** T8.1.
- **Accept:** non-modal gains match `couette/taylor_couette_{linear,mri}.py` to 1e-8 (x64); modal `--eigs` still matches to 1e-11.

**[T8.3] Plane-Couette linear/MHD eigen + non-modal operator — *L*** · `examples/pcf_linear_jax.py` (new) + `--linear` flags in PCF examples
- **What:** primitive-variable Chebyshev PCF linear operator (hydro + MHD + rotating-shear/MRI) with `eigs`/`nonmodal`/`energy_matrix` and conducting/dirichlet magnetic wall BCs; `--linear {dns,eigs,nonmodal}` + `--linear-by/-bz/-magnetic-bc/-energy`. **shenfun ref:** `couette/_pcf_linear.py:59-289`. **Depends:** T8.1.
- **Accept:** PCF leading eigenvalues and G(t) match `couette/_pcf_linear.PlaneCouetteLinear` (all three energy norms, both magnetic BCs) to 1e-8 (x64).

**[T8.4] Critical-parameter scans — *M*** · `taylor_couette_{mri,linear}_jax.py`
- **What:** port `critical_eta_mag`, `critical_Rm_fixed_B0_nu`, `critical_Rm`, the kz-scan, and wire TC-hydro `critical_reynolds` into a tested CLI branch. **shenfun ref:** `couette/taylor_couette_mri.py:470-571`, `taylor_couette_linear.py:343-417`. **Depends:** T8.2.
- **Accept:** conducting/insulating critical-Rm match the reference scan to 1e-4 (reproduce η=0.5 quasi-Keplerian conducting `Rm_min≈24.7`, insulating `Rm_min≈16.5`); TC-hydro critical Reynolds/Taylor matches to 1e-8.

---

### M9 — Azimuthal Fourier (`m≠0`) machinery (shared by M10, M13)

**[T9.1] 3-axis `CoupledSpace` with a complex azimuthal Fourier axis — *M*** · `taylor_couette_dns_jax.py` + `galerkin/Fourier.py`
- **What:** TC spaces on `(θ` complex-Fourier `(0,2π)`, `z` Fourier, `r` Dirichlet/orthogonal`)` as a `CoupledSpace`; thread the azimuthal symbol so `Dx(·, θ_axis, 1)` gives `i·m` per mode; add the `_require_resolved_m` guard (`2|m|<Ntheta`). **shenfun ref:** `couette/taylor_couette_dns.py:525-540`, `:73-88`. **Depends:** T6.1.
- **Accept:** `forward(backward(u))` per component to 1e-12 on a 3-axis field; `Dx(u,θ,1)` gives `i·m·u`; the guard raises for `2|m|≥Ntheta`.

**[T9.2] Per-`(m,kz)` two-Fourier-axis block solve — *L*** · `taylor_couette_dns_jax.py:147-162` or `la/blocktpmatrix.py`
- **What:** generalize `_mode_indices`/`_extract_mode_matrices` so each `(m,kz)` pair is one batched dense `lu_factor`/`lu_solve` over the combined mode axis, with the `(0,0)` pressure pin at the correct global index. **shenfun ref:** `couette/taylor_couette_dns.py:510-565`. **Depends:** T9.1, T6.2.
- **⚠️ HARD prerequisite (critic):** before trusting any 3D quadrant, **validate the no-pivot batched LU at production `N` on the indefinite per-`(m,kz)` saddle blocks** (cf. risk §11.4.4); add pivoting (T2.1) to the batched path if it loses accuracy. A silently inaccurate no-pivot LU corrupts every 3D result and is very hard to diagnose.
- **Accept:** a steady-Stokes-in-annulus 3D MMS solves per `(m,kz)` and matches `shenfun` `BlockMatrixSolver` to 1e-10 for several `(m,kz)`; `(0,0)` pressure pinned; mode indexing verified against a reference enumeration.

---

### M10 — 3D hydrodynamic TC DNS (`TaylorCouetteDNS`, `m≠0`)

**[T10.1] 3D cylindrical linear operator with `m`-couplings — *L*** · `taylor_couette_dns_jax.py` (new `TaylorCouetteDNSJax`)
- **What:** `m`-dependent Laplacian `L f = f_rr + f_r/r + (1/r²)f_θθ + f_zz`; base-shear advection `−Ω·d/dθ` on every component; viscous r/θ cross-coupling `∓(2/r²)d u_{θ/r}/dθ`; azimuthal continuity `(1/r)d u_θ/dθ` and pressure gradient `(1/r)dΠ/dθ`, into the per-`(m,kz)` Limp/Lexp. **shenfun ref:** `taylor_couette_dns.py:568-571, :573-596, :611-618`. **Depends:** T9.2.
- **Accept:** assembled 3D linear blocks for a few `m` match `shenfun`'s matrices to 1e-12.

**[T10.2] 3D cylindrical nonlinear + CNAB2 + 3D divergence — *L*** · `taylor_couette_dns_jax.py`
- **What:** add `(u_θ/r)d/dθ` advection on all three components (plus existing `−u_θ²/r`, `+u_r u_θ/r`); 3D divergence with `(1/r)du_θ/dθ`; 3D eigenmode seeding `exp(i(mθ+kz·z))` with real-field reconstruction; drive with CNAB2 (T6.3). **shenfun ref:** `:649-666, :752-759, :687-715`. **Depends:** T10.1, T6.3, T6.4.
- **Accept:** **x64 eigenmode-seeded growth-rate parity over 100 CNAB2 steps to 1e-6** vs a 3D linear eigenmode (and vs `shenfun` once the harness exists); `div(u)` at machine precision; **not** a smoke test.

---

### M11 — Magnetic induction/Lorentz/EMF machinery (shared by M12, M13; conducting walls)

**[T11.1] Conducting magnetic bases + 7-field coupled space — *M*** · `examples/taylor_couette_mri_dns_jax.py` (new shared base)
- **What:** conducting bases `b_r` Dirichlet, `b_θ` Robin `d(r·b_θ)/dr=0`, `b_z` Neumann `b_z'=0` (reuse the Robin/Neumann machinery already in `taylor_couette_mri_jax.py`); 7-field implicit space `VQ=[TD,TD,TD,TP,TD,Tbt,Tbz]` (total pressure `Π`) and 6-field evolving `VE`. **shenfun ref:** `taylor_couette_dns.py:835-857`. **Depends:** T6.1.
- **Accept:** conducting magnetic stencils match `shenfun` to 1e-13; the 7-field block system is square with expected per-block dims.

**[T11.2] Linear induction + Lorentz block — *L*** · `taylor_couette_mri_dns_jax.py`
- **What:** `η(L−1/r²)b` induction diffusion; `B0·db/dz` Lorentz momentum coupling and `B0·du/dz` induction coupling under the total-pressure formulation; MRI field-stretching source `r·Ω'·b_r → b_θ`. **shenfun ref:** `taylor_couette_dns.py:917-951`. **Depends:** T11.1.
- **Accept:** assembled induction+Lorentz blocks match `shenfun` to 1e-12 (axisymmetric).

**[T11.3] EMF `curl(u×b)` + Maxwell `−(b·∇)b` + `divb` monitor — *L*** · `taylor_couette_mri_dns_jax.py` + `integrators/nonlinear.py`
- **What:** pseudo-spectral EMF `ε=u×b` (reuse `physical_cross`, T4.8); induction RHS `−curl(ε)`; Maxwell `−(b·∇)b`; dealiasing + (ideally cached) cross-space `Project`s for curl pieces; `magnetic_divergence_l2` monitor. **shenfun ref:** `taylor_couette_dns.py:1001-1048, :1163-1164`. **Depends:** T11.2, T4.8.
- **Accept:** EMF/Maxwell terms match `shenfun` to 1e-10 for a seeded mode; `divb < 1e-12` throughout (x64).

---

### M12 — Axisymmetric MHD (MRI) TC DNS (`AxisymmetricMRIDNS`)

**[T12.1] `AxisymmetricMRIDNSJax` — *L*** · `examples/taylor_couette_mri_dns_jax.py`
- **What:** combine the axisymmetric-hydro CNAB2/block-solve/cylindrical-nonlinear with the M11 induction/Lorentz block + EMF/Maxwell nonlinear; pin `k=0` total pressure; seed from the linear MRI eigenmode (`taylor_couette_mri_jax.py`). **shenfun ref:** `taylor_couette_dns.py:788-1190`. **Depends:** T11.3, T6.3, T6.4.
- **Accept:** **x64 eigenmode-seeded MRI growth-rate parity over 100 CNAB2 steps to 1e-6** (vs linear MRI eigenmode; vs `shenfun` once the harness exists); `div(u)` and `div(b)` at machine precision; total-pressure `k=0` pinned. Carry an x64 eigenmode-growth (not smoke) parity test.

---

### M13 — 3D MHD (full MRI) TC DNS (`TaylorCouetteMRIDNS`) — deepest quadrant

**[T13.1] 3D MHD linear couplings — *L*** · `examples/taylor_couette_mri_dns_jax.py`
- **What:** `−Ω·d/dθ` on every `b`-component; `∓(2/r²)d(·)_{θ/r}/dθ` viscous **and resistive** cross-coupling for both `u` and `b`; 3D continuity/pressure azimuthal terms, into the per-`(m,kz)` 7-field block. Conducting `b_r=0` makes wall BCs `m`-independent (reuse M11 bases). **shenfun ref:** `taylor_couette_dns.py:1340-1385, :1398-1404, :1263-1267`. **Depends:** T9.2, T11.2.
- **Accept:** assembled 3D MHD blocks for a few `m` match `shenfun` to 1e-12.

**[T13.2] 3D EMF curl (azimuthal `(1/r)d/dθ`) + complex seeding + step — *L*** · `examples/taylor_couette_mri_dns_jax.py`
- **What:** 3D EMF curl nonlinear with `(1/r)d/dθ` in all three curl components; complex eigenmode seeding split into real/imag radial Functions; CNAB2 over the combined `(m,kz)` modes. **shenfun ref:** `taylor_couette_dns.py:1463-1470, :1512-1554`. **Depends:** T13.1, T11.3, T10.2.
- **Accept:** **x64 eigenmode-seeded 3D MRI growth-rate parity over 100 steps to 1e-6**; `div(u)` and `div(b)` at machine precision. Carry an x64 eigenmode-growth (not smoke) parity test.

---

### 12.x Build order (dependency-correct, 6 phases)

```
PHASE 0  M0b  float64-at-import (T0.1b) → live shenfun harness (T0.4b)   ← unblocks ALL parity
         + cheap gating partials: K_over_K2 → library (T1.6); commit the full-complex Fourier ADR (T0.3)
PHASE 1  finish correctness-blocking partials BEFORE trusting TC quadrants:
         T6.5 radial dealiasing parity · T8.0 stable eig ordering · validate no-pivot
         biharmonic at production N and switch workloads to `pivot=True` where needed ·
         build reusable modules: integrators/coupled.py (T3.4), integrators/cnab2.py (T6.3),
         la/solvers.py (T2.2), cached Project (T1.2)
PHASE 2  M8 stability layer (independent; staff in parallel). Acceptance needs T0.4b.
PHASE 3  M9 azimuthal machinery (T9.2 = highest-risk shared prereq) → M10 3D-hydro
PHASE 4  M11 magnetic machinery (can start parallel to Phase 3; needs only M6+T4.8) → M12 axi-MHD
PHASE 5  M13 3D-MHD  = M9 ∪ M11  (needs Phases 3 & 4)
PHASE 6  M7 I/O (T7.1/T7.2) + sharding parity (T7.3) + differentiability (T0.4c) across all quadrants
```

### 12.y Updated definition-of-done (extends §9)

| Deliverable | Done when… | Gating |
|---|---|---|
| (foundation) | default `uv run pytest` runs x64 & a live `shenfun` parity test passes for one PCF + one TC case | M0b |
| TC linear + non-modal (hydro & MHD) | eig to 1e-8/1e-6 & transient-growth gains to 1e-8 vs reference; critical-Rm/Re reproduced | M8 |
| **3D hydro TC DNS** | x64 eigenmode-growth parity over 100 steps to 1e-6; `div(u)`≈0; reduces to axisymmetric when `m`-content=0 | M9→M10 |
| **Axisymmetric MHD TC DNS** | x64 eigenmode-growth parity to 1e-6; `div(u)`,`div(b)`≈0; total-`p` pinned | M11→M12 |
| **3D MHD TC DNS** | x64 eigenmode-growth parity to 1e-6; `div(u)`,`div(b)`≈0 | M9,M11→M13 |
| All quadrants | `jax.grad` of a diagnostic finite & FD-consistent; HDF5 I/O; multi-device bit-identical | M7 |

---

*End of plan (Parts I + II). Original scope: 41 gaps / 8 milestones (Part I). Extended scope adds M0b (foundation hardening), M8 (stability analysis), and M9–M13 (the three missing Taylor–Couette DNS quadrants + their shared azimuthal & magnetic prerequisites). The current implementation runs all seven scripts as internally-consistent jax ports but is **not yet `shenfun`-validated**; the axisymmetric-hydro TC DNS is the only one of four quadrants complete. Critical path: land M0b first (so parity means something), then finish the Phase-1 correctness partials, then build outward to the 3D and MHD quadrants.*
