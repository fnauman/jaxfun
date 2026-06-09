---
title: "Wall-Bounded Shear-Flow Solvers Across Spectral and Finite-Difference Families"
subtitle: "Algorithms, MHD/MRI, validation, and gap analysis for plane-Couette, Taylor–Couette, and pipe flow — a hand-off specification for autonomous reimplementation"
author: "cfd repository · automated solver survey"
date: "7 June 2026"
abstract: |
  This report is a self-contained, implementation-ready specification of three independent solver families for incompressible and MHD wall-bounded shear flows — plane Couette / plane channel, Taylor–Couette, and pipe — together with a comprehensive cross-family test suite and a prioritized gap analysis. The three families are **A = shenfun** (spectral Galerkin, Python/CPU/MPI), **B = torch** (Fourier + high-order finite difference, PyTorch/GPU/autograd), and **C = jax** (spectral Galerkin, JAX/TPU/JIT/autograd). Part I documents each family in enough algorithmic detail — governing equations, nondimensionalization, spatial discretization, divergence-free/pressure treatment, boundary-condition enforcement, time integrators (with explicit tableaux), MHD induction/Lorentz formulation, and rotation/shear (MRI) source terms — to reimplement it from scratch. Part II is a feature × family gap matrix; Part III is a prioritized roadmap to close those gaps (MHD/MRI everywhere including insulating magnetic walls; coordinate/sign unification; geometry coverage including pipe-in-JAX; and compute/autograd parity). Part IV is the test suite, organized as a foundational floor required in every family, temporal- and spatial-discretization order tests tailored to each family's numerics, and shear/rotation/MHD regime tests with analytic oracles. Part V is a compute/portability/autograd appendix and the reference bibliography. Source anchors and benchmark-style claims are audited in Appendix A with `VERIFIED` / `PARTIAL` / `UNSUPPORTED` status; current-tree line anchors are checked mechanically, while literature-only or semantically ambiguous items remain marked partial until independently rechecked.
---

# How to read this document

**Provenance.** Generated 2026-06-07 from (i) a direct reading of the three solver codebases in the `cfd` repository and (ii) the classic literature. Load-bearing implementation claims carry `file:line` anchors into the source where available, and numerical benchmarks are quoted from code, notes, or a cited paper. Where a fact could not be verified against an opened source it is flagged inline.

**The three families and where they live** (all paths relative to the repository root `/home/nauman/cfd/`):

| Family | Numerics | Backend | Location |
|---|---|---|---|
| **A — shenfun** | spectral Galerkin (Chebyshev/Legendre × Fourier) | Python, CPU, MPI | `fn_shenfun/demo/` |
| **B — torch** | Fourier (periodic) + high-order finite difference (wall-normal/radial) | PyTorch, CPU/GPU, autograd | `fn_openpipeflow-122/torch{channel,couette,pipeflow}/` |
| **C — jax** | spectral Galerkin (tensor-product spaces) | JAX, CPU/GPU/TPU, JIT + autograd | `shenfun_jaxfun_spectralDNS/fork_jaxfun/` |

**Document map.**

- **Part 0** — Scope, the unified coordinate/sign convention with per-family adapters, nondimensionalization, and symbol glossary. *Read this first; everything downstream uses its canonical frame.*
- **Part I** — Per-family algorithmic specification: **I.A** shenfun, **I.B** torch, **I.C** jax. Each is independently sufficient to reimplement that family.
- **Part II** — Cross-family gap matrix (feature × family).
- **Part III** — Prioritized closure recommendations with target files and acceptance gates.
- **Part IV** — Test suite: foundational floor, temporal/spatial discretization order, and shear/rotation/MHD regime tests.
- **Part V** — Compute/portability/autograd appendix and the reference bibliography (cited throughout by keys such as [KMM87], [Willis17], [BH91]).

**Conventions for the reader.** Math set in `$…$` is typeset; expressions in `monospace` are either code or a verbatim transcription of a formula as it appears in the source. The resistive diffusivity is written `η_mag` to avoid collision with the Kolmogorov/wall-unit `η`. Citation keys resolve in Part V.3.

**Verification audit: current tree vs. closure specification (2026-06-07).** This document deliberately mixes two things: an audit of what is implemented now and a specification for what must be added to reach cross-family parity. Treat the following as the current-state override when a later test row is phrased normatively:

- **Implemented and source-supported now:** A and C have PCF MHD/MRI source terms; B channel/couette MHD advances induction and Lorentz coupling, but B MRI parameters are metadata only (`torchchannel/mhd.py:71-74`); A and B have pipe-hydro solvers while C has no pipe; A and C have Taylor-Couette insulating/vacuum wall treatment only in the linear `m=0` flux-function path; pipe MHD is absent everywhere.
- **Planned, not implemented now:** the cross-family file bridge `fn_openpipeflow-122/parity/{conventions.py,observables.py,...}` and `to_canonical()` adapter do **not** exist in the current tree. They are Phase 0 requirements. Current parity evidence is partial: A has internal PCF↔TC checks (`thin_gap_compare.py`) and C has live shenfun parity tests, but B has no cross-boundary parity harness.
- **Missing test coverage now:** no target-tree Taylor-Green/TGV harness was found in `fn_shenfun/demo`, `fn_openpipeflow-122/torch*`, or `fork_jaxfun/{examples,tests}`. F6 remains a required foundational test to add, not evidence of current coverage. F3/F4 energy and symmetry checks are also unevenly implemented and should be made explicit per family.
- **Local note aliases used below:** `PLAN…` = `PLAN_openpipeflow_vs_fnshenfun.md`; `mhd_parity_plan.md` = `fn_shenfun/demo/mhd_parity_plan.md`; `pcf_mhd_mri_notes.md` = `fn_shenfun/demo/pcf_mhd_mri_notes.md` unless explicitly the JAX mirror; `pipe_flow_notes.md` = `fn_shenfun/demo/pipe_flow_notes.md`; `couette_linear_benchmarks.md` = `fn_shenfun/demo/couette_linear_benchmarks.md`; `C parity plan` = `shenfun_jaxfun_spectralDNS/jaxfun_pcf_parity_plan_2026-05-28.md`. Family-local labels such as `A-PCF`, `B-MHD`, `B-PIPE`, `B-CORE`, `C-FRAMEWORK`, and `C-SOLVERS` are shorthand for the corresponding Part I family subsections, not separate source files.

\newpage


\newpage

# Part 0 — Scope & Foundations

This part is the foundation the rest of the survey builds on. It states the problem, introduces the three solver families at a glance, fixes a single **canonical coordinate and sign convention** that every later section maps to, gives the per-family adapter that resolves the plane-Couette discrepancy, defines all nondimensional groups, and collects the symbol glossary. Read it before any of Parts I–V: the conventions and adapters here are load-bearing for every cross-family comparison made later.

---

## 0.1 Scope and the three families at a glance

This survey is an implementation-ready specification of **three independent solver families** for incompressible, wall-bounded shear flows — plane Couette/channel, Taylor–Couette, and pipe — extended with **MHD** (induction + Lorentz) and **rotation/shear (MRI)** physics. The intended reader is an autonomous coding agent reimplementing any of these solvers from scratch, plus a researcher cross-checking the results. Each family is a distinct, working codebase with its own numerical method, backend, and tradeoffs:

- **A = shenfun** — spectral Galerkin (composite/Shen bases that bake boundary conditions into the trial space), Python on CPU with MPI (`mpi4py`/`mpi4py-fft`), float64. This is the **spectral oracle**: exponential convergence, divergence to roundoff (~$10^{-16}$), and golden linear eigenvalues asserted to $10^{-12}$. No GPU, no JIT, no autograd. Code root: `/home/nauman/cfd/fn_shenfun/demo/`.
- **B = torch** — Fourier (periodic directions) + dense Taylor/Vandermonde finite differences (wall-normal/radial), PyTorch on CPU/GPU with full autograd, complex128 default. This is the **FD + GPU + autograd** family: device-agnostic, fully differentiable through the projection, BC influence matrix, and MHD coupling. Code roots: `/home/nauman/cfd/fn_openpipeflow-122/torch{channel,couette,pipeflow}/`.
- **C = jax** — spectral Galerkin built on the `jaxfun` form-language toolkit, JAX on CPU/GPU/TPU with `jax.jit`/`nnx.jit` and `value_and_grad` autograd, x64-by-default. This is the **spectral + JIT + autograd + TPU** family: JAX-native DNS solvers (KMM channel, Taylor–Couette) that are differentiable and shardable, with the linear-stability layer kept as dense NumPy/SciPy (a documented caveat, §I.C.2). Code root: `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/`.

The families are intentionally **not** identical discretizations. Family A is the reference against which B (4th-order-family FD, ~8th-order interior) and C (spectral, JAX-native) are validated. Parity means agreement on *formulation-independent physical observables* (growth rates, energies, stresses) within each family's truncation error — never to roundoff across families (§Part II, §Part IV tolerance ladder).

| | **A = shenfun** | **B = torch** | **C = jax** |
|---|---|---|---|
| Language / backend | Python / NumPy+SciPy, CPU+MPI | Python / PyTorch, CPU+GPU | Python / JAX (XLA), CPU+GPU+TPU |
| Wall-normal basis | spectral Galerkin (Chebyshev/Legendre Shen composite) | dense Taylor/Vandermonde FD (`KL=4` ⇒ 9-pt) | spectral Galerkin (Chebyshev/Legendre Shen composite) |
| Periodic basis | Fourier (real/complex) | Fourier | Fourier |
| Default integrator | IMEXRK222 (PCF/channel); CNAB2 (TC/pipe) | semi-implicit θ-method predictor/corrector (θ=0.51) | IMEXRK222 (KMM/PCF); CNAB2 (TC) |
| Div-free / pressure | KMM velocity–vorticity (PCF) or coupled saddle-point (TC/pipe), div~$10^{-16}$ | influence matrix + pressure projection + dense pinv cleanup, div~$10^{-7}$ | KMM (PCF) or pinned saddle-point (TC), div~$10^{-17}$ |
| Geometries | PCF/channel, Taylor–Couette, **pipe** | PCF/channel, Taylor–Couette, **pipe** | PCF/channel, Taylor–Couette (**no pipe**) |
| MHD | vector potential $B=\nabla\times A$, Lorentz prefactor 1 | induced field $b$, prefactor $Ha^2/(Re\,Rm)$ or $Ha^2/Pm$ | vector potential $B=\nabla\times A$, prefactor 1 |
| MRI (rotation+shear) | **present** (Coriolis + base-shear + shear-induction) | **stub** — metadata only, no source terms | **present** (Coriolis + shear-induction) |
| Insulating walls | present (TC linear, $m=0$ flux-fn) | absent | present (TC linear, $m=0$ flux-fn) |
| Autograd / JIT / GPU / TPU | – / – / – / – | ✓ / – / ✓ / – | ✓ / ✓ / ✓ / ✓ |
| Precision floor | float64 | complex128 (float32 validated) | x64-by-default |

Run environments are **disjoint** (no live cross-import): A → `/home/nauman/miniconda3/envs/shenfun/bin/python` (shenfun 4.2.2, no torch); B → `/home/nauman/miniconda3/envs/huggingface/bin/python` (torch, no shenfun); C → `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/.venv` (uv venv, system Python 3.12, local JAX 0.10.1; `pyproject.toml` defines a `cuda13` optional extra). Cross-family comparison therefore uses **file-based committed goldens** (JSON/HDF5) and never live cross-imports; golden generation may run each family in its own subprocess or CI job (§V.2).

---

## 0.2 Unified coordinate & sign convention + per-family adapters

### 0.2.1 The canonical frame (the single reference all families map to)

All cross-family observables in this survey are expressed in **one canonical frame**. The planned parity bridge must ship a `to_canonical()` adapter that maps each family's native axes/sign onto this frame before any comparison; that adapter is a Phase 0 requirement, not a current-tree file.

- **Axes.**
  - $x$ = wall-normal / radial / shear-gradient direction. This is the Dirichlet / no-slip-wall direction (walls at $x=\pm 1$ in plane geometry).
  - $y$ = streamwise / azimuthal — the wall-motion / shear direction.
  - $z$ = spanwise / axial — **the rotation axis**, $\boldsymbol{\Omega} = \Omega\,\hat{\boldsymbol{z}}$.

  This matches the shenfun shearing-box map (`pcf_mhd_mri_shearpy.py:7-9`): "component 0 / coordinate x: radial, wall-normal, shear-gradient direction; component 1 / coordinate y: azimuthal, streamwise, wall-motion direction; component 2 / coordinate z: vertical, spanwise direction."

- **Canonical base flow (shear).** A linear shear in the streamwise component as a function of wall-normal $x$:
  $$\boldsymbol{U}_b(x) = \sigma\,x\,\boldsymbol{e}_y, \qquad \frac{dU_b}{dx} = \sigma.$$
  - **Hydro plane Couette:** $\sigma = +U_{\text{wall}}$ (walls at $x=\pm 1$ move at $\pm U_{\text{wall}}$).
  - **Shearing-box / MRI (shearpy):** $\sigma = -S$, i.e. $\boldsymbol{U}_b(x) = -S\,x\,\boldsymbol{e}_y$ (verified at `pcf_mhd_mri_shearpy.py:11`, `:102`: `self.Ub = -self.shear_rate * self.X[0]`, `self.dUb_dx = -self.shear_rate`).
  - **Poiseuille channel:** $U_b(x) = 1-x^2$ in $\boldsymbol{e}_y$ (parabolic, stationary walls).

- **Canonical Lorentz force.** $+\,\boldsymbol{J}\times\boldsymbol{B}$ with prefactor **1**, in Alfvén / Lorentz–Heaviside units $\rho = \mu_0 = 1$ (so $\boldsymbol{B}$ is in velocity units and the Alfvén speed is $v_A = |\boldsymbol{B}|$). Families A and C use this directly; family B carries a dimensional prefactor that must be overridden to 1 for oracle tests (§0.2.3).

- **Canonical nondimensional groups.** $Re = UL/\nu$, $Rm = UL/\eta_{\text{mag}}$, $Pm = \nu/\eta_{\text{mag}} = Rm/Re$, Hartmann $Ha = B_0 L /\sqrt{\nu\,\eta_{\text{mag}}}$, Lundquist $S_L = B_0 L /\eta_{\text{mag}}$. Rotation $\Omega$, shear $S$, shear parameter $q = S/\Omega$, epicyclic frequency $\kappa^2 = 2\Omega(2\Omega - S) = 2\Omega^2(2-q)$ (verified at `pcf_mhd_mri_shearpy.py:107-108`: `self.q_shear = self.shear_rate/self.omega`, `self.kappa2 = 2.0*self.omega*(2.0*self.omega - self.shear_rate)`). Full definitions in §0.3.

### 0.2.2 Per-family mapping table

The cell entries below are the native conventions; the proposed `to_canonical()` adapter (§0.2.3) maps them to §0.2.1. File:line anchors are verbatim from the source where available.

| Aspect | A = shenfun | B = torch | C = jax |
|---|---|---|---|
| **Axis order** | 0 = $x$ wall-normal, 1 = $y$ streamwise, 2 = $z$ spanwise (`ChannelFlow.py:9-11`) | $x$ = streamwise, $y$ = wall-normal, $z$ = spanwise (`base_flow.py:25-50`; **swapped vs canonical**) | 0 = $x$ wall-normal, 1 = $y$ streamwise, 2 = $z$ spanwise (`channelflow_kmm.py:58-59`) |
| **PCF hydro base flow** | $\boldsymbol{U}_b = +U_{\text{wall}}\,x\,\boldsymbol{e}_y$ (`pcf_fluctuations_corrected.py:130-135`) | $U = +y$ (streamwise component = $x$) (`base_flow.py:37-41`) | $\boldsymbol{U}_b = +U_{\text{wall}}\,x\,\boldsymbol{e}_y$ (`pcf_fluctuations_jax.py:65-67`) |
| **Poiseuille base** | $1-x^2$, $dp/dy = -2/Re$ (`OrrSommerfeld.py:14,30`) | $1-y^2$, const-flux $4/3$ (`base_flow.py:42-46`) | $1-x^2$ (OS path, mirrors A) |
| **MRI/shearpy base** | $\boldsymbol{U}_b = -S\,x\,\boldsymbol{e}_y$, $dU_b/dx=-S$ (`pcf_mhd_mri_shearpy.py:11,102`) | **N/A — metadata only** (`mhd.py:71-74`) | $\boldsymbol{U}_b = -S\,x\,\boldsymbol{e}_y$, $dU_b/dx=-S$ (`pcf_mhd_mri_shearpy_jax.py:70-72`) |
| **TC base flow** | $V(r)=ar+b/r$, $a=(\Omega_2 R_2^2-\Omega_1 R_1^2)/(R_2^2-R_1^2)$ (`taylor_couette_linear.py:89-91`) | $u_\theta=ar+b/r$, $a=(Re_o-\eta\,Re_i)/(1+\eta)$ (`base_flow.py:18-25`) | $V(r)=ar+b/r$ (`taylor_couette_linear_jax.py:37-89`) |
| **Pipe base flow** | $u_z=(f_z/4\nu)(R^2-r^2)$ (`pipe_flow_dns.py:473-475`) | $U=1-r^2$, $b_{\text{hpf}}=2r$ (`solver.py:200-203`) | **N/A — no pipe** |
| **Shear sign (canonical $\sigma$)** | $+U_{\text{wall}}$ (hydro) / $-S$ (MRI) — direct | $+y$ ⇒ **flip sign + remap axes in adapter** | $+U_{\text{wall}}$ (hydro) / $-S$ (MRI) — direct |
| **Lorentz prefactor** | 1 (`pcf_mhd_divfree.py:325-333`) | $Ha^2/(Re\,Rm)$ channel (`mhd.py:100-101`); $Ha^2/Pm$ couette (`mhd.py:79`) → **set explicit override = 1 for oracle tests** | 1 (`pcf_mhd_jax.py:175-176`) |
| **Pm / Rm convention** | $Rm=U/\eta$, $Pm=\nu/\eta$; $Rm\leftarrow Re$ if unset (`pcf_mhd_divfree.py:91-102`) | $Rm$=None ⇒ $Rm=Re\,Pm$; else $Pm=Rm/Re$; mag-diff $=1/Rm$ (channel) or $1/Pm$ (couette, viscosity=1) (`mhd.py:89-94,155`) | $Rm\leftarrow Re$ if unset, $\eta=U/Rm$ (`pcf_mhd_jax.py:63-64`) |
| **Rotation/shear symbols** | $\Omega$=`omega`, $S$=`shear_rate`, $\kappa^2=2\Omega(2\Omega-S)$ (`pcf_mhd_mri_shearpy.py:107-108`) | `omega`/`shear_rate` stored, `q_shear` only (NO source) (`mhd.py:102-104`) | `omega`=$\Omega$, `q_shear`=$S/\Omega$, `kappa2`=$2\Omega(2\Omega-S)$ (`pcf_mhd_mri_shearpy_jax.py:73-74`) |
| **Magnetic BC** | conducting ($A\in TD^3$ DNS; Robin/Neumann linear) + insulating flux-fn (TC linear $m=0$) | homogeneous $b=0$ only (no insulating) (`mhd.py:278-282`) | conducting (DNS) + insulating flux-fn (TC linear $m=0$) |

### 0.2.3 Resolution of the plane-Couette coordinate/shear-sign discrepancy (the "silent killer")

The conflict between families is purely **labeling + sign**, not physics:

- A and C put the streamwise velocity in component $y$ as a function of wall-normal $x$: $U_b = U_{\text{wall}}\,x$ along $\boldsymbol{e}_y$.
- B puts the streamwise velocity in component $x$ as a function of wall-normal $y$: $U = y$ along $\boldsymbol{e}_x$ (verified at `torchcouette/base_flow.py` analogue and `torchchannel/base_flow.py:37-41`).

These describe the **same physical flow** — linear shear between walls moving at $\pm U_{\text{wall}}$. The adapter for **B** is:

1. **Relabel axes:** $(x_B, y_B, z_B) \to (y_{\text{can}}, x_{\text{can}}, z_{\text{can}})$ — B's streamwise $x_B$ becomes canonical $y$, B's wall-normal $y_B$ becomes canonical $x$, spanwise $z$ unchanged.
2. **Flip the shear sign for MRI comparisons:** $S \to -S$, so B's would-be shear matches the shearpy convention $\boldsymbol{U}_b = -S\,x\,\boldsymbol{e}_y$.
3. **Override the Lorentz prefactor to 1** for any oracle/parity comparison: set B's `lorentz_prefactor = 1` instead of $Ha^2/(Re\,Rm)$ (channel) / $Ha^2/Pm$ (couette).

All cross-family comparisons should operate on **canonical-frame observables** (growth rates, energies, stresses), which are frame-invariant once the adapter is applied. This should be implemented as `parity/conventions.py::to_canonical()` + `parity/observables.py` (Phase 0 of the closure roadmap, §III.2); those files are not present in the current tree. The eigenvalue set-matching and inertial→local Doppler frame conversion should be **reused** from `_linear_analysis.match_eigenvalues`, not reinvented.

**Two non-physics caveats that ride on this adapter:**

- **B's "4th-order FD" label vs. reality.** B is labeled a 4th-order FD family, but the default half-bandwidth `KL=4` gives a 9-point centered stencil that is formally **8th-order interior** (first/second derivatives accurate to 8th/7th order; polynomial-exact through degree 8, asserted `err_dy, err_dyy < 1e-7` for degrees 0–8 in `torchchannel/tests/test_mesh.py:21-37`). Boundary stencils are one-sided and lower-order and typically set the realized global order. Throughout this survey, B is described as **"4th-order family label; ≥4th-order floor, ~8th interior,"** and convergence tests assert slope $\geq 3.7$ (§Part IV/S2).
- **B's MRI is metadata-only.** In B, `omega`/`shear_rate` set only the diagnostic `q_shear` and add **no** Coriolis / base-shear / shear-induction source terms (`torchchannel/mhd.py:71-74`). Every MRI claim about B is flagged **"stub"**; the rotation/shear tests (§Part IV/SR-1, SR-2) are the acceptance gate for wiring it in (§III.1, Phase 1).

---

## 0.3 Nondimensionalization

The three families share the same physical groups but carry them into the equations differently. The canonical (family-A/C) convention is **velocity-units**: a single velocity scale $U$ and length scale $L$, with $\nu$ and $\eta_{\text{mag}}$ appearing as $1/Re$ and $1/Rm$. Family B's channel solver follows this; B's Couette solver uses **viscous units** (viscosity $\equiv 1$, $Re$ carried into the wall speeds). Magnetic quantities are always in **Alfvén / Lorentz–Heaviside units** ($\rho = \mu_0 = 1$).

### 0.3.1 Hydrodynamic groups

$$Re = \frac{U L}{\nu}.$$

Per geometry / family:

- **Plane Couette / channel (A, C):** velocity scale $U = U_{\text{wall}}$, length $L = h$ (half-gap, $h=1$ for $x\in[-1,1]$). Then $\nu = U_{\text{wall}}\,h/Re$; with defaults $U_{\text{wall}}=h=1$, $\nu = 1/Re$ (`pcf_fluctuations_corrected.py:106-107`; `pcf_fluctuations_jax.py:54`). Poiseuille uses an imposed mean pressure gradient $dp/dy = -2/Re$ (`OrrSommerfeld.py:14`).
- **Plane channel (B):** single parameter $Re$; viscosity enters as $1/Re$ in the diffusion operator (`torchchannel/solver.py:225`, `_build_diffusion_system(1.0/Re)`). Couette base $U=y$; Poiseuille base $U=1-y^2$ at constant flux $4/3$ (`torchchannel/base_flow.py:42-46`).
- **Taylor–Couette (B, viscous units):** radius ratio $\eta = R_1/R_2$ with $r_i = \eta/(1-\eta)$, $r_o = 1/(1-\eta)$ (gap $=1$); inner/outer Reynolds numbers $Re_i, Re_o$ are the wall speeds directly. The diffusion LHS uses coefficient $1/dt$ with **no** $1/Re$ factor (`torchcouette/solver.py:256-260`) — velocities are gap-scaled and $Re$ lives in the wall boundary values $u_\theta(r_i)=Re_i$, $u_\theta(r_o)=Re_o$.
- **Taylor–Couette (A, C, velocity units):** $Re = \Omega_1 R_1 \cdot \text{gap}/\nu$ (`taylor_couette_dns_jax.py:159`), base flow $V(r)=ar+b/r$.
- **Pipe (A):** $u_z = (f_z/4\nu)(R^2-r^2)$ with axial body force $f_z$; Hagen–Poiseuille flux $Q = \pi R^4 f_z/(8\nu)$ (`pipe_flow_dns.py:473-475`). Family C has **no pipe**.

### 0.3.2 Magnetic groups

$$Rm = \frac{U L}{\eta_{\text{mag}}}, \qquad Pm = \frac{\nu}{\eta_{\text{mag}}} = \frac{Rm}{Re}, \qquad Ha = \frac{B_0 L}{\sqrt{\nu\,\eta_{\text{mag}}}}, \qquad S_L = \frac{B_0 L}{\eta_{\text{mag}}}\ \text{(Lundquist)}.$$

- **A:** $Rm = U/\eta_{\text{mag}}$, $\eta_{\text{mag}} = U_{\text{wall}}/Rm$; if $Rm$ unset, $Rm \leftarrow Re$ so $Pm = 1$ (`pcf_mhd_divfree.py:91-102`). Lorentz force $+\boldsymbol{J}\times\boldsymbol{B}$ with **prefactor 1** (`pcf_mhd_divfree.py:325-333`).
- **C:** identical to A — $Rm \leftarrow Re$ if unset, $\eta = U/Rm$ (`pcf_mhd_jax.py:63-64`), Lorentz prefactor 1 (`pcf_mhd_jax.py:175-176`).
- **B (channel):** $Pm$ default 1; if $Rm$ given then $Pm = Rm/Re$, else $Rm = Re\,Pm$ (`torchchannel/mhd.py:89-94`). Magnetic diffusion $\propto 1/Rm$. **Lorentz prefactor** $= Ha^2/(Re\,Rm)$ (`torchchannel/mhd.py:100-101`).
- **B (couette):** magnetic diffusion LHS coefficient $\propto 1/Pm$ (viscosity $\equiv 1$, `torchcouette/mhd.py:155`); **Lorentz prefactor** $= Ha^2/Pm$ (`torchcouette/mhd.py:79`). No separate $Rm$.

For oracle comparisons against A/C, B's Lorentz prefactor is overridden to 1 (§0.2.3).

### 0.3.3 Rotation / shear (MRI) groups

In the shearing-box (shearpy) convention, with rotation along $z$ and base shear $\boldsymbol{U}_b = -S\,x\,\boldsymbol{e}_y$:

$$\Omega \equiv \texttt{omega}, \qquad S \equiv \texttt{shear\_rate}, \qquad q = \frac{S}{\Omega}, \qquad \kappa^2 = 2\Omega(2\Omega - S) = 2\Omega^2(2 - q).$$

Keplerian rotation has $q = 3/2$ (defaults $S=1$, $\Omega = 2/3$). The Alfvén speed of the imposed vertical field is $v_A = B_z$ (with $\rho = \mu_0 = 1$). These are verified at `pcf_mhd_mri_shearpy.py:107-108` (A) and `pcf_mhd_mri_shearpy_jax.py:73-74` (C). Family B stores `omega`/`shear_rate` but adds no rotation/shear source terms (stub).

The implemented shearing-box source terms (A and C; the acceptance target for B, §III.1) are, in canonical axes:

$$
\frac{\partial u_x}{\partial t} \mathrel{+}= 2\Omega\,u_y, \qquad
\frac{\partial u_y}{\partial t} \mathrel{+}= (S - 2\Omega)\,u_x, \qquad
\frac{\partial B_y}{\partial t} \mathrel{+}= -S\,B_x,
$$

i.e. Coriolis $-2\boldsymbol{\Omega}\times\boldsymbol{u}$ (with $\boldsymbol{\Omega}=\Omega\hat{z}$), Coriolis + base-flow shear in the streamwise equation, and shear-induction (stretching $B_x$ into $B_y$) via $\boldsymbol{U}_b\times\boldsymbol{B}$. These are verified verbatim in the shenfun module header (`pcf_mhd_mri_shearpy.py:11-14`). In A/C the magnetic term is realized through the vector-potential induction $\partial A/\partial t = \boldsymbol{U}\times\boldsymbol{B} + \eta\nabla^2 A$, not as a separate component update (`pcf_mhd_mri_shearpy.py:16-21`).

### 0.3.4 Reference / oracle benchmark numbers (frame-invariant, for cross-checking)

These closed-form or published targets are reused throughout Parts IV–V. Values and sources are verbatim.

| Benchmark | Target | Reproduced (source) |
|---|---|---|
| Orr–Sommerfeld leading eigenvalue, $Re=8000$, $\alpha=1$ (A) | — | $c = 0.24707506017508621 + 0.0026644103710965817\,i$, tol $10^{-12}$ (`OrrSommerfeld_eigs.py:183-184`) |
| Orr–Sommerfeld leading eigenvalue, $Re=10000$, $\alpha=1$ (B) | $c_{\text{ref}} = 0.23752649 + 0.00373967\,i$ | tol $10^{-4}$ (`torchchannel/tests/test_linstab_poiseuille.py:7-19`) |
| OS critical Reynolds (cross-family) | $Re_{\text{crit}} = 5772.22$ [Orszag71] | rel $<10^{-2}$ |
| Ideal local Keplerian MRI | $s_{\max}/\Omega = 0.75$, $(k v_A)^2/\Omega^2 = 15/16$ | A/C: $0.7499999944199642$, $0.9373170323757943$ (`couette_linear_benchmarks.md:313`) |
| TC MRI conducting walls ($\eta=0.5$ quasi-Kep, $Rm=24.7$, $S=4.11$) | growth $> 0$ | $+0.003322863594034156$ at best $k_z=1.75$ (`couette_linear_benchmarks.md:352`) |
| TC MRI insulating walls ($Rm=16.5$, $S=5.21$) | growth $< 0$ (sign flip) | $-0.00027582037141390655$ at best $k_z=1.25$ (`couette_linear_benchmarks.md:353`) |

The conducting/insulating sign flip ($+0.00332$ vs $-2.76\times10^{-4}$) is reproduced by A and C only (B has no insulating walls); pin `magnetic_bc` identically on both sides before comparing (§Part IV/SR-6, SR-9).

---

## 0.4 Symbol glossary

| Symbol | Meaning | Notes / canonical value |
|---|---|---|
| $x$ | wall-normal / radial / shear-gradient coordinate (axis 0) | Dirichlet/no-slip walls at $x=\pm 1$ |
| $y$ | streamwise / azimuthal coordinate (axis 1) | wall-motion / shear direction |
| $z$ | spanwise / axial coordinate (axis 2) | **rotation axis**, $\boldsymbol{\Omega}=\Omega\hat{z}$ |
| $\boldsymbol{u}=(u_x,u_y,u_z)$ | velocity perturbation (fluctuation) | base flow carried separately, not in solution vector |
| $\boldsymbol{U}_b$ | base flow | $\sigma\,x\,\boldsymbol{e}_y$; $\sigma=+U_{\text{wall}}$ (hydro), $-S$ (MRI) |
| $\sigma$ | canonical shear, $dU_b/dx$ | $+U_{\text{wall}}$ (PCF) / $-S$ (shearbox) |
| $U_{\text{wall}}$ | plane-Couette wall speed | default 1 |
| $h$, $L$ | half-gap / length scale | $h=1$ for $x\in[-1,1]$ |
| $\nu$ | kinematic viscosity | $=U_{\text{wall}}h/Re$ (velocity units); $\equiv 1$ in B-couette |
| $\eta_{\text{mag}}$ | magnetic diffusivity (resistivity) | $=U L/Rm$ |
| $Re$ | Reynolds number | $UL/\nu$ |
| $Rm$ | magnetic Reynolds number | $UL/\eta_{\text{mag}}$; $\leftarrow Re$ if unset (A/C) |
| $Pm$ | magnetic Prandtl number | $\nu/\eta_{\text{mag}} = Rm/Re$ |
| $Ha$ | Hartmann number | $B_0 L/\sqrt{\nu\,\eta_{\text{mag}}}$ |
| $S_L$ | Lundquist number | $B_0 L/\eta_{\text{mag}}$ |
| $\boldsymbol{B}$, $B_0$, $b$ | magnetic field; imposed uniform field; induced/fluctuation field | $B = \nabla\times A$ (A/C); induced $b$ with background $B_0$ (B) |
| $A$ | magnetic vector potential | $B=\nabla\times A$ ⇒ $\nabla\cdot B=0$ by construction (A/C) |
| $\boldsymbol{J}$ | current density | $\boldsymbol{J}=\nabla\times\boldsymbol{B}$; Lorentz $=+\boldsymbol{J}\times\boldsymbol{B}$, prefactor 1 (canonical) |
| $\chi$ | poloidal flux function | insulating-wall (vacuum) formulation, TC linear $m=0$ |
| $v_A$ | Alfvén speed | $=|\boldsymbol{B}|$ in Alfvén units; $v_A=B_z$ for imposed vertical field |
| $\Omega$ | rotation rate | $\boldsymbol{\Omega}=\Omega\hat{z}$; `omega` in code |
| $S$ | shear rate | $dU_b/dx = -S$ (shearbox); `shear_rate` in code |
| $q$ | shear parameter | $S/\Omega$; Keplerian $q=3/2$ |
| $\kappa^2$ | epicyclic frequency squared | $2\Omega(2\Omega-S) = 2\Omega^2(2-q)$ |
| $q_{\text{shear}}$ | diagnostic shear parameter (B) | $=S/\Omega$; metadata only, no source term |
| $U'$, `Uprime` | base-flow shear in linear operators | $=\sigma$ (e.g. $-S$ for shearbox) |
| $s$ | complex growth rate / eigenvalue | $\propto e^{st}$; $\text{Re}(s)>0$ unstable |
| $g$ | wall-normal vorticity $(\nabla\times u)_x$ | KMM evolved scalar; $g = i k_y u_z - i k_z u_y$ |
| $\theta$ | implicit weight in B's θ-method | default 0.51 (slightly above CN 0.5) |
| $KL$ | FD half-bandwidth (B) | default 4 ⇒ 9-point stencil, ~8th-order interior |
| $\eta$ | TC radius ratio $R_1/R_2$ | distinct from $\eta_{\text{mag}}$ — context disambiguates |
| $a,b$ | TC base-flow coefficients | $V(r)=ar+b/r$ |

**Per-family file-path roots:**

- **A (shenfun):** `/home/nauman/cfd/fn_shenfun/demo/`
- **B (torch):** `/home/nauman/cfd/fn_openpipeflow-122/torch{channel,couette,pipeflow}/`
- **C (jax):** `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/` (solvers in `examples/`, library in `src/jaxfun/`, shenfun reference in `couette/`)

> **Note on $\eta$ overloading.** $\eta_{\text{mag}}$ (magnetic diffusivity) and $\eta$ (TC radius ratio) are distinct quantities sharing a symbol; this survey writes $\eta_{\text{mag}}$ wherever resistivity is meant and reserves bare $\eta$ for the geometric radius ratio in Taylor–Couette contexts. Both usages are standard in the respective literatures and codebases.


\newpage

# Part I.A — The shenfun spectral-Galerkin family (full algorithmic spec)

This part specifies family **A** in enough detail to reimplement from scratch. Family A is a CPU/MPI, NumPy/SciPy + shenfun spectral-Galerkin stack (`float64`/`complex128`, no GPU/JIT/autograd; an optional numba toggle accelerates shenfun's own kernels only, `pcf_mhd_divfree.py:47-57`). All file paths are under `/home/nauman/cfd/fn_shenfun/demo/` and the installed library is `/home/nauman/miniconda3/envs/shenfun/lib/python3.12/site-packages/shenfun/`. Run via `/home/nauman/miniconda3/envs/shenfun/bin/python` (Part V.2). The unified coordinate frame, sign adapters, and nondimensionalization groups are §0.2–§0.4; this part uses those conventions and gives the family-A realization. Cross-family parity is Part II (§3); closure work is Part III; tests are Part IV.

Family A contains **two complementary stacks** throughout: a **nonlinear pseudo-spectral DNS** and a **dense linear-stability / non-modal layer** (collocation and Galerkin) that shares Butcher tableaux with the DNS time-steppers. Family A is the designated **spectral oracle** for cross-family checks because its operator identities (div-free, div(curl)=0) hold to roundoff.

---

## I.A.1 Plane Couette / plane channel (KMM velocity–vorticity)

### I.A.1.1 Governing equations and nondimensionalization

The hydro DNS uses the Kim–Moser–Moin (KMM) velocity–vorticity reduction [KMM87]: pressure is eliminated exactly, and only two scalar fields are advanced — the wall-normal velocity $u_x$ (whose time variable is $\nabla^2 u_x$) and the wall-normal vorticity $g=(\nabla\times u)_x$. From `ChannelFlow.py:149-163` (LaTeX strings verbatim from the code):

$$\frac{\partial (\nabla^2 u_x)}{\partial t} = \nu\,\nabla^4 u_x + \frac{\partial^2 N_y}{\partial x\partial y} + \frac{\partial^2 N_z}{\partial x\partial z} - \frac{\partial^2 N_x}{\partial y^2} - \frac{\partial^2 N_x}{\partial z^2},$$

$$\frac{\partial g}{\partial t} = \nu\,\nabla^2 g + \frac{\partial N_y}{\partial z} - \frac{\partial N_z}{\partial y}, \qquad g = \partial_y u_z - \partial_z u_y .$$

$N=H$ is the convection vector (§I.A.1.6). The exact shenfun source operators are `Dx(Dx(self.H_[1],0,1),1,1)+Dx(Dx(self.H_[2],0,1),2,1)-Dx(self.H_[0],1,2)-Dx(self.H_[0],2,2)` for $u_x$ (`ChannelFlow.py:152`) and `Dx(self.H_[1],2,1)-Dx(self.H_[2],1,1)` for $g$ (`ChannelFlow.py:160`).

**Nondimensionalization (PCF fluctuation solver).** Half-gap $h=1$, domain $x\in[-1,1]$, $\mathrm{Re}=U_{\text{wall}}h/\nu$ so $\nu=U_{\text{wall}}/\mathrm{Re}$ (`pcf_fluctuations_corrected.py:106-107`); default $U_{\text{wall}}=1\Rightarrow\nu=1/\mathrm{Re}$. The Orr–Sommerfeld (Poiseuille) validation case uses $\nu=1/\mathrm{Re}$ with an imposed mean pressure gradient $\mathrm{d}p/\mathrm{d}y=-2/\mathrm{Re}$ (`OrrSommerfeld.py:14`; 2D `OrrSommerfeld2D.py:14`).

**Coordinate/sign convention (canonical, §0.2 row "A").** Axis 0 $=x$ wall-normal (Chebyshev/Legendre, walls at $x=\pm1$); axis 1 $=y$ streamwise (Fourier complex, `dtype='D'`); axis 2 $=z$ spanwise (Fourier real, `dtype='d'`) (`ChannelFlow.py:9-11`). The base flow is carried **analytically in the convection term only** and is never inserted into the spectral solution vector, so $u'=0$ is the exact laminar fixed point. Hydro PCF uses

$$U_b(x) = U_{\text{wall}}\,x\,\mathbf{e}_y, \qquad \frac{\mathrm{d}U_b}{\mathrm{d}x}=U_{\text{wall}}\quad(\text{const}),$$

`pcf_fluctuations_corrected.py:130-135` (`self.Ub = self.U_wall*self.X[0]`, `self.dUb_dx = self.U_wall`). The MRI/shearing-box subclass overrides this to $U_b=-S\,x\,\mathbf{e}_y$ (§I.A.5). For cross-family comparison with family B apply the planned §0.2 `to_canonical()` adapter (axis remap + shear-sign flip).

### I.A.1.2 Spatial discretization — composite (Shen) bases bake in BCs

The 1-D bases (`ChannelFlow.py:82-88`):

```python
B0  = FunctionSpace(N[0], family, bc=(0,0,0,0), domain=domain[0])  # biharmonic/clamped, dim N-4
D0  = FunctionSpace(N[0], family, bc=(0,0),     domain=domain[0])  # Dirichlet,         dim N-2
C0  = FunctionSpace(N[0], family,               domain=domain[0])  # orthogonal, no BC, dim N
F1  = FunctionSpace(N[1], 'F', dtype='D', domain=domain[1])        # Fourier y (complex)
F2  = FunctionSpace(N[2], 'F', dtype='d', domain=domain[2])        # Fourier z (real, Hermitian-halved)
D00 = FunctionSpace(N[0], family, bc=(0,0), domain=domain[0])      # 1-D Dirichlet, the (0,0) mode
C00 = D00.get_orthogonal()
```

- `bc=(0,0)` → composite Dirichlet (Shen) space with every basis function satisfying $\varphi(\pm1)=0$; dimension $N-2$. **No tau rows** — the BC is built into the trial/test space [Shen95].
- `bc=(0,0,0,0)` → clamped/biharmonic space with $\varphi(\pm1)=\varphi'(\pm1)=0$; dimension $N-4$. Used for $u_x$ because its evolution equation is 4th-order; the two extra conditions are no-penetration plus the continuity-implied $\partial_x u_x=0$ at the walls.
- `C0` (orthogonal, dimension $N$) holds $x$-derivatives of Dirichlet fields, which leave the Dirichlet space; placing derivative/curl outputs there makes the discrete identities (div, div curl) exact.

Tensor-product and vector spaces (`ChannelFlow.py:91-99`):

```python
TB  = TensorProductSpace(comm,(B0,F1,F2), collapse_fourier=False, slab=True, modify_spaces_inplace=True)  # wall-normal vel
TD  = TensorProductSpace(comm,(D0,F1,F2), ...)   # Dirichlet components
TC  = TensorProductSpace(comm,(C0,F1,F2), ...)   # unconstrained (derivative range)
BD  = VectorSpace([TB,TD,TD])                     # velocity (u_x,u_y,u_z)
CD  = VectorSpace(TD)                             # convection vector [TD,TD,TD]
CC  = VectorSpace([TD,TC,TC])                     # curl vector space
TDp = TD.get_dealiased(padding_factor)            # padded space
```

`collapse_fourier=False, slab=True` keep a 3-D slab MPI decomposition with wall-normal $x$ local on every rank, which is required for the per-mode 1-D radial solves and the $(0,0)$-mode solve.

**Quadrature.** Gauss by default (Gauss–Chebyshev `GC` / Gauss–Legendre `LG`); the OrrSommerfeld eigensolver uses `quad='GC'` (`OrrSommerfeld_eigs.py:29`).

**Per-mode radial solvers.** For each Fourier mode $(k_y,k_z)\neq(0,0)$, the $u_x$ equation is biharmonic and $g$ is Helmholtz. Chebyshev uses the tailored fast solvers `chebyshev.la.Biharmonic` (for $u_x$) and `chebyshev.la.Helmholtz` (for $g,v,w$); Legendre uses the generic banded `la.SolverGeneric1ND` (`ChannelFlow.py:144-145, 180`).

**2D channel** (`ChannelFlow2D.py`) keeps only $x$ (wall-normal) and $y$ (streamwise Fourier real); the wall-normal velocity basis is built via the explicit clamped dict `bc={'left':{'D':0,'N':0},'right':{'D':0,'N':0}}` (`ChannelFlow2D.py:81`), which is the same clamped biharmonic space. Only $u_x$ is evolved ($\partial_t\nabla^2 u_x=\nu\nabla^4 u_x+\partial^2 N_y/\partial x\partial y-\partial^2 N_x/\partial y^2$, `ChannelFlow2D.py:138-144`); recovery is $u_y=\mathrm{i}\,(\partial_x u_x)/k_y$ (`ChannelFlow2D.py:195`); no $g$ equation is needed.

### I.A.1.3 Divergence-free constraint & pressure handling — exact elimination

The KMM form removes pressure from the time loop entirely (no projection, no influence matrix). For every Fourier mode $(k_y,k_z)\neq(0,0)$, after solving $u_x$ and $g$, the tangential components are recovered **algebraically** from continuity + the definition of $g$ (`compute_vw`, `ChannelFlow.py:236-239`):

$$u_y = \mathrm{i}\,\frac{k_y f + k_z g}{k_y^2+k_z^2}, \qquad u_z = \mathrm{i}\,\frac{k_z f - k_y g}{k_y^2+k_z^2}, \qquad f \equiv \partial_x u_x,$$

with `K_over_K2[i] = K[i+1]/where(K2==0,1,K2)`, $K2=k_y^2+k_z^2$ (`ChannelFlow.py:168-171`). **Sign pitfall:** the code uses $f=+\partial_x u_x$ while a comment notes "paper uses $f=-\mathrm{d}u/\mathrm{d}x$" (`ChannelFlow.py:236`); the recovery formulas above are written for the $+$ convention and must be matched exactly.

The single $(0,0)$ Fourier mode (horizontal mean) has no divergence constraint, so $v_{00}(x), w_{00}(x)$ are advanced as two **separate 1-D Helmholtz PDEs** on `D00`, on MPI rank 0 only, with a mean pressure-gradient source (`ChannelFlow.py:181-196`, LaTeX verbatim):

$$\frac{\partial v}{\partial t} = \nu\,\frac{\partial^2 v}{\partial x^2} - N_y - \frac{\mathrm{d}p}{\mathrm{d}y}, \qquad \frac{\partial w}{\partial t} = \nu\,\frac{\partial^2 w}{\partial x^2} - N_z,$$

with source $-\mathrm{d}p/\mathrm{d}y$ ($=0$ for PCF, $=-(-2/\mathrm{Re})=2/\mathrm{Re}$ source for OS Poiseuille). Measured $\mathrm{div}(u)\approx10^{-12}$ (roundoff), checked via `divu = Project(div(u_), TC)` (`ChannelFlow.py:129`).

An **optional** post-hoc pressure recovery (`compute_pressure`, `ChannelFlow.py:253-268`) solves a Poisson problem $\nabla^2 p=-\mathrm{div}(H)$ with Neumann BC $\partial p/\partial n=\nu\,\partial^2 u_x/\partial n^2$ baked into the basis and the null space pinned by the constraint `((0,0,0),)`.

### I.A.1.4 Boundary conditions

No-slip Dirichlet $u'=0$ at $x=\pm1$, enforced by **basis design** (no tau rows): $u_x\in B0$ (clamped $u_x=\partial_x u_x=0$), $u_y,u_z\in D0$ (Dirichlet). Because only the fluctuation is stepped and the base flow is analytic, walls are homogeneous and the no-slip condition is satisfied exactly. (Linear-layer BC enforcement is §I.A.1.8.)

### I.A.1.5 Time integration in the KMM loop

The driver (`KMM.solve`, `ChannelFlow.py:315-330`) is:

```
assemble()                                   # factor implicit operators once
while t < end_time - 1e-8:
    for rk in range(PDE.steps()):            # RK stages
        prepare_step(rk)                     # -> convection(): build N=H_
        for eq in pdes:   eq.compute_rhs(rk)
        for eq in pdes:   eq.solve_step(rk)  # implicit solves: u_x (biharmonic), g (Helmholtz)
        compute_vw(rk)                       # algebraic recovery u_y,u_z + (0,0) Helmholtz
    t += dt; tstep += 1
```

The base `KMM` constructor default is `IMEXRK3` (`ChannelFlow.py:67`), **but the PCF subclasses and runners override to `IMEXRK222`** (`pcf_fluctuations_corrected.py:689` passes `timestepper='IMEXRK222'`; `pcf_fluctuations_divV.py:58,210`; `ChannelFlow2D.KMM` default `'IMEXRK222'`, `ChannelFlow2D.py:66`). Exact tableaux are in §I.A.6. Diffusion ($\nu\nabla^2$/$\nu\nabla^4$) is implicit; convection is explicit; the implicit LHS depends only on the constant diagonal coefficient $a_{11}$, so the Helmholtz/biharmonic factorizations are computed **once** in `assemble()`. There is no viscous CFL limit and no adaptive `dt` controller; `dt` is fixed by the caller (DNS default $0.01$; OS $0.001$–$0.002$).

### I.A.1.6 Nonlinear term & dealiasing

Only convective (advective) form $u\cdot\nabla u$ is implemented; rotational form (`conv=1`) raises `NotImplementedError` (`pcf_fluctuations_corrected.py:175-176`). The procedure (`pcf_fluctuations_corrected.py:190-202`): backward-transform $u'$ and the nine velocity gradients to the **padded** physical grid; form products; add base-flow advection and shear production

```python
n0 += Ub*dudyp
n1 += Ub*dvdyp + up[0]*self.dUb_dx     # up[0]*dUb_dx is the shear-production term (y-component)
n2 += Ub*dwdyp
```

(`pcf_fluctuations_corrected.py:195-197`, verified verbatim) with $U_b=U_{\text{wall}}\,x$, $\mathrm{d}U_b/\mathrm{d}x=U_{\text{wall}}$; then forward-transform `H[i]=TDp.forward(n_i,H[i])` and zero the Nyquist Fourier mode `H_.mask_nyquist(mask)`.

**Dealiasing (Orszag 3/2, periodic directions only).** `padding_factor=(1,1.5,1.5)` (`ChannelFlow.py:62`): the two Fourier directions ($y,z$) are padded 3/2 so quadratic products are aliasing-free; the wall-normal $x$ is **not padded** (factor 1) — products are evaluated on the Gauss grid and handled by modal truncation + Nyquist masking. For single-mode OrrSommerfeld validation `padding_factor=1` (no dealiasing).

### I.A.1.7 Orr–Sommerfeld validation and golden eigenvalue

`OrrSommerfeld_eigs.py` solves $A\varphi=cB\varphi$ in Shen's biharmonic Chebyshev basis (`bc=(0,0,0,0)`, dim $N-4$); default `alfa=1.0, Re=8000, N=80`. Operators (`OrrSommerfeld_eigs.py:84-99`): with weighted inner products $K=(u'',v)_w$, $K1=((1-x^2)u,v)_w$, $K2=((1-x^2)u'',v)_w$, $Q=(u'''',v)_w$, $M=(u,v)_w$,

$$B=-\mathrm{Re}\,\alpha\,\mathrm{i}\,(K-\alpha^2 M), \quad A = Q - 2\alpha^2 K + (\alpha^4 - 2\alpha\,\mathrm{Re}\,\mathrm{i})M - \mathrm{i}\alpha\mathrm{Re}\,(K2-\alpha^2 K1).$$

**Golden eigenvalue** (self-asserted, `OrrSommerfeld_eigs.py:183-184`): at $\mathrm{Re}=8000$, $\alpha=1$, $N>80$,

$$c = 0.24707506017508621 + 0.0026644103710965817\,\mathrm{i}, \qquad \text{tol } 10^{-12}.$$

The DNS OS path (`OrrSommerfeld.py`/`OrrSommerfeld2D.py`) seeds the eigenmode at amplitude $10^{-7}$ on the Poiseuille base $1-x^2$ and checks energy $\propto\exp(2\,\mathrm{Im}(c)\,t)$ (`OrrSommerfeld.py:47-56`). PCF is linearly stable for all $\mathrm{Re}$ (Romanov); leading rates at $\mathrm{Re}=1000$ and the Butler–Farrell optimal growth ($t^*=139$, $G^*=1165.2$ reference; computed $G=1165.93$ at $t=139$) are the §IV non-modal goldens [RH93] (`couette_linear_benchmarks.md:69-70,111-118`).

### I.A.1.8 Linear-stability layer (dense, primitive variables)

Two operators target the generalized eigenproblem $Lq=sMq$ for perturbations $q(x)\exp(s t+\mathrm{i}k_y y+\mathrm{i}k_z z)$ ($k_y^2+k_z^2>0$), base flow $U(x)=U_{\text{off}}+U'x$ along $\mathbf{e}_y$:

- **Galerkin** (`pcf_galerkin_linear.py`): velocity in Dirichlet `bc=(0,0)`; pressure in the full orthogonal space sliced to `slice(0,N-2)` (`:87-88`, **not** a `bc`-constrained basis); MHD $b_y,b_z$ in Neumann. Blocks via `inner(test, Dx(trial,0,order)).diags().toarray()`. Scalings: $\nu=U_{\text{wall}}h/\mathrm{Re}$, $\eta=U_{\text{wall}}h/\mathrm{Rm}$, $U'=U_{\text{wall}}/h$ (`:103-117`).
- **Collocation** (`_pcf_linear.py`): Chebyshev–Lobatto points $x_j=\cos(\pi j/N)$, dense Trefethen-style $D$, $D2=D@D$, Clenshaw–Curtis quadrature weights for the energy norm; $\mathrm{lap}=D2-k^2 I$; default `nx=64`. Scalings $\nu=U_{\text{wall}}/\mathrm{Re}$, $\eta=U_{\text{wall}}/\mathrm{Rm}$, $U'=U_{\text{wall}}$.

Both treat pressure as a **Lagrange multiplier** enforcing $\mathrm{div}(u)=0$ (saddle-point), with $-\nabla p$ columns and $\mathrm{div}\,u$ rows (`_pcf_linear.py:220-228`); the pressure block has **zero mass** so $M$ is singular and the spurious infinite eigenvalues are dropped with `FINITE_CAP=1e8` in `finite_eigensystem` (`_linear_analysis.py:16,33-42`). MHD adds a magnetic-pressure multiplier $\phi$ enforcing $\mathrm{div}(b)=0$ (§I.A.4). Velocity BCs are imposed by row replacement (identity row at each wall, zeroed in $L$ and $M$). A DNS-style **linear IMEXRK stepper** (`pcf_imexrk_linear.py`) advances the same dense operator with the §I.A.6 tableaux and recovers $\mathrm{Re}(s)$ to $\sim10^{-8}$.

---

## I.A.2 Taylor–Couette

### I.A.2.1 Coordinate convention — explicit-$1/r$ plain measure (NOT curvilinear shenfun)

TC (linear **and** DNS) does **not** use shenfun's `coordinates=` curvilinear machinery. It builds plain Cartesian-measure spaces on `domain=(R1,R2)` and writes every $1/r$, $1/r^2$ as an explicit sympy coefficient of the radial symbol (the "OrrSommerfeld strong-form / plain-measure" pattern). The radial symbol must come from **this space** — `self.r = self.TD.coors.psi[1]` (axisymmetric, axis 1) or `psi[2]` (3D, axis 2) (`taylor_couette_dns.py:141,540`). Using the wrong global symbol "silently applies the curvature along the wrong axis (and $b/z^2$ blows up at $z=0$)" (`taylor_couette_dns.py:187-190`). Axisymmetric scalar Laplacian written out: `Dx(u,1,2)+(1/r)*Dx(u,1,1)+Dx(u,0,2)` (`taylor_couette_dns.py:178-180`); 3D adds `(1/r**2)*Dx(u,0,2)`.

### I.A.2.2 Base flow, perturbation equations, nondimensionalization

The circular-Couette base (`CircularCouette`, `taylor_couette_linear.py:81-132`):

$$\Omega(r)=a+\frac{b}{r^2}, \quad V(r)=\Omega(r)\,r=a r + \frac{b}{r}, \quad a=\frac{\Omega_2 R_2^2-\Omega_1 R_1^2}{R_2^2-R_1^2}, \quad b=\frac{(\Omega_1-\Omega_2)R_1^2 R_2^2}{R_2^2-R_1^2},$$

with $U_{\text{base}}=+V(r)\,\mathbf{e}_\theta$ (positive swirl). Identities: $2\Omega+r\Omega'=2a$ (constant), $r\Omega'=-2b/r^2$, epicyclic $\kappa^2(r)=4a\,\Omega(r)$; radius ratio $\eta=R_1/R_2$, rotation ratio $\mu=\Omega_2/\Omega_1$, gap $d=R_2-R_1$, local shear exponent $q=-\mathrm{d}\ln\Omega/\mathrm{d}\ln r$. Rayleigh-stable $\Leftrightarrow\kappa^2>0$ everywhere; Keplerian ($\mu=\eta^{3/2}$, $\Omega\propto r^{-3/2}$) is Rayleigh-stable but MRI-unstable [BH91]. $\mathrm{Re}=\Omega_1 R_1 d/\nu$ (`taylor_couette_dns.py:123`).

The DNS integrates the **perturbation about the analytic base** (so $u=0$ is the exact fixed point; the base centrifugal balance $\mathrm{d}P_{\text{base}}/\mathrm{d}r=V^2/r$ is subtracted). Axisymmetric (`taylor_couette_dns.py:21-35`, verbatim):

$$\frac{\partial u_r}{\partial t}=-\partial_r p+\nu\!\left(L-\tfrac1{r^2}\right)u_r+2\Omega\,u_\theta-N_r, \quad \frac{\partial u_\theta}{\partial t}=\nu\!\left(L-\tfrac1{r^2}\right)u_\theta-2a\,u_r-N_\theta,$$
$$\frac{\partial u_z}{\partial t}=-\partial_z p+\nu L\,u_z-N_z, \qquad 0=\partial_r u_r+\frac{u_r}{r}+\partial_z u_z,$$

with $Lf=f_{rr}+f_r/r+f_{zz}$, $2a=2\Omega+r\Omega'$ const. Only the centrifugal/Coriolis pair $+2\Omega u_\theta$ (r) and $-2a u_r$ ($\theta$) couples the base flow at axisymmetry. Nonlinear (cylindrical metric):

$$N_r=u_r u_{r,r}+u_z u_{r,z}-\frac{u_\theta^2}{r}, \quad N_\theta=u_r u_{\theta,r}+u_z u_{\theta,z}+\frac{u_r u_\theta}{r}, \quad N_z=u_r u_{z,r}+u_z u_{z,z}.$$

The **3D** solver adds, per azimuthal mode $m$: base-shear advection $-\mathrm{i}m\Omega$ on every component; the viscous cross-couplings $\mp(2/r^2)\partial_\theta u_{\theta/r}$ ($=\mp 2\mathrm{i}m/r^2$); the full $(1/r^2)\partial_\theta^2$ Laplacian term; continuity $u_{\theta,\theta}/r$; pressure $(1/r)\partial_\theta p$ (`taylor_couette_dns.py:573-617`).

### I.A.2.3 Radial basis, mode structure, pressure/div-free

Velocity uses the Dirichlet composite `bc=(0,0)` (dim $N-2$); pressure the orthogonal space sliced to $N-2$ modes (`SP.slice = lambda: slice(0,N-2)`), giving the inf-sup-stable $P_N$/$P_{N-2}$ pair with `assert SP.dim()==SD.dim()`. Family default `'L'` (Legendre) hydro; `'C'` (Chebyshev) for the IMEXRK DNS. Mode layout: axisymmetric `AxisymmetricTCDNS` uses z Fourier real + r Dirichlet, `axes=(1,0)`; 3D `TaylorCouetteDNS` uses $\theta$ Fourier complex + z Fourier real + r Dirichlet, `axes=(2,0,1)`. Dealiasing is the 3/2-rule on **all** axes including radial (`dealias=1.5` default; `dealias=1.0` disables).

**Div-free is exact via a coupled saddle-point solve** (no fractional-step splitting): velocity and pressure are solved together per Fourier mode through `la.BlockMatrixSolver`, so $\mathrm{div}(u)\sim10^{-13}$–$10^{-14}$. The $k=0$ constant-pressure null space is pinned with the constraint `constraints=((3,0,0),)` (block 3, mode 0, value 0). **Diagnostic pitfall:** divergence must be evaluated as separate per-term physical-space projections; combining `Dx(f_hat,...)` with a sympy `(1/r)*f_hat` in one inner/project mis-evaluates and reports spurious $O(\text{amplitude})$ divergence (`taylor_couette_dns.py:396-411`).

### I.A.2.4 Time integrator — CNAB2 (default)

The DNS default is CNAB2 (2nd-order IMEX): Crank–Nicolson for the linear operator $A=$ viscous $+$ all base-flow couplings $+$ pressure gradient; Adams–Bashforth-2 for the nonlinear term; IMEX-Euler bootstrap on the first step. The descriptor form is (`pipe_flow_dns.py:35-37`, identical structure in TC):

$$\frac{u^{n+1}-u^n}{\Delta t}=-\tfrac12 A u^{n+1}-\tfrac12 A u^n-\nabla p^{n+1}-\Big(\tfrac32 N^n-\tfrac12 N^{n-1}\Big)+f, \qquad \mathrm{div}\,u^{n+1}=0.$$

Pre-assembled: `Limp = BlockMatrixSolver(M/dt - 1/2 A + grad p ; div=0)` over the coupled velocity–pressure space and `Lexp = BlockMatrix(M/dt + 1/2 A)`. The exact per-step code (`taylor_couette_dns.py:288-313`, verified verbatim):

```python
self.nonlinear(self.N_hat)
rhs_v = self.Lexp.matvec(self.u_hat, rhs_v)        # (M/dt + 1/2 A) u^n
for i in range(3):
    if self._have_old:
        self.rhs[i] = rhs_v[i] - (1.5*self.N_hat[i] - 0.5*self.N_old[i])  # AB2
    else:
        self.rhs[i] = rhs_v[i] - self.N_hat[i]                            # IMEX-Euler bootstrap
self.rhs[3] = 0.0
self.sol = self.Limp(self.rhs, u=self.sol, constraints=((3,0,0),))        # coupled saddle solve
self.N_old[:] = self.N_hat; self._have_old = True
```

An IMEXRK axisymmetric DNS companion (`taylor_couette_imexrk_dns.py`) does a coupled saddle-point solve **inside each RK stage**, caching the stage LHS factor by $\gamma$ (no 3D IMEXRK class exists). Tableaux as in §I.A.6.

### I.A.2.5 Magnetic BCs (TC MHD)

See §I.A.4 for the MHD equations; the wall conditions are:

- **Conducting** (any $m$): $b_r=0$ (Dirichlet), $\mathrm{d}(r b_\theta)/\mathrm{d}r=0$ i.e. $b_\theta+r b_\theta'=0$ (Robin), $b_z'=0$ (Neumann). The Robin coefficient is in **reference** coordinates: shenfun `bc={"left":{"R":(R1/Jm,0)},"right":{"R":(R2/Jm,0)}}` with $J_m=(R_2-R_1)/2$ (`taylor_couette_dns.py:832,839-841`, verified verbatim). **Load-bearing pitfall:** using $r_{\text{wall}}$ instead of $r_{\text{wall}}/J$ produces a spurious *growing* purely-magnetic mode scaling with $\eta_{\text{mag}}$. Conducting walls are $m$-independent in 3D ($b_r=0\Rightarrow\partial_\theta b_r=0$ on the wall).
- **Insulating** (linear analysis only, $m=0$ only): poloidal flux function $\chi$ (§I.A.4); $m\neq0$ raises `NotImplementedError`, $k_z=0$ raises `ValueError`. No nonlinear DNS supports insulating walls.

`div(b)` is never pressure-projected in TC MHD: $b$ is advanced as direct components (not a vector potential), $\mathrm{div}(b)=0$ is preserved to roundoff by the induction structure and the solenoidal seed, and is only monitored.

---

## I.A.3 Pipe

### I.A.3.1 Coordinate convention — true curvilinear shenfun ($\sqrt{g}=r$)

The pipe DNS **does** use curvilinear shenfun, so the $r\,\mathrm{d}r\,\mathrm{d}\theta\,\mathrm{d}z$ measure is applied automatically: `r,theta,z = sp.symbols('x,y,z', real=True, positive=True)`, `rv=(r*cos(theta), r*sin(theta), z)`, spaces built with `coordinates=(psi,rv)` (`pipe_flow_dns.py:115-140`). Consequence: `inner(1,.)` already carries $\sqrt{g}=r$, so energy/flow-rate must **not** multiply by $r$ again. The scalar Laplacian is the Laplace–Beltrami `div(grad(f))`; the metric vector-Laplacian couplings ($-u_r/r^2$, $\mp(2/r^2)\partial_\theta u$) are still added explicitly on top (`pipe_flow_dns.py:193-201`). Axis order: r axis 0, $\theta$ axis 1 (Fourier complex), z axis 2 (Fourier real). Defaults `Nr=32, Ntheta=8, Nz=8, R=1, Lz=2π`.

### I.A.3.2 Governing equations and $r=0$ pole regularity — KEY MECHANISM

Primitive $(u_r,u_\theta,u_z,p)$ driven by a uniform axial body force $f_z$ (constant $\Rightarrow$ Hagen–Poiseuille; callable $f_z(t)\Rightarrow$ Womersley); $z$ periodic, zero mean pressure gradient. The vector Laplacian (`pipe_flow_dns.py:41-46`):

$$(\nabla^2 u)_r=\nabla^2 u_r-\frac{u_r}{r^2}-\frac{2}{r^2}\partial_\theta u_\theta, \quad (\nabla^2 u)_\theta=\nabla^2 u_\theta-\frac{u_\theta}{r^2}+\frac{2}{r^2}\partial_\theta u_r, \quad (\nabla^2 u)_z=\nabla^2 u_z,$$

with scalar $\nabla^2 f=f_{rr}+f_r/r+f_{\theta\theta}/r^2+f_{zz}$ and continuity $u_{r,r}+u_r/r+u_{\theta,\theta}/r+u_{z,z}$.

The pole is handled with **one** velocity basis `bc=(None,0)` for **all** azimuthal modes (free/regularity at $r=0$, no-slip Dirichlet at $r=R$) — **no $m$-by-$m$ basis split**. The curvilinear weighted ($\sqrt{g}=r$) Galerkin operator carries the singular penalties that select the regular solution: the scalar Laplacian contributes $-m^2/r^2$ (forces $u_z,p\to0$ at the axis for $m\neq0$, leaves $m=0$ free); the vector couplings diagonalize on $u_\pm=u_r\pm\mathrm{i}u_\theta$ to $-(m\mp1)^2/r^2$, reproducing exactly the $r^{|m\mp1|}$ regularity. Verbatim regularity table (`pipe_flow_notes.md:39-43`):

| field | $m=0$ | $\lvert m\rvert=1$ | $\lvert m\rvert\geq2$ |
|---|---|---|---|
| $u_r,u_\theta$ | $\to0$ ($r$) | finite ($r^0$) | $\to0$ ($r^{m-1}$) |
| $u_z$ | finite ($r^0$) | $\to0$ ($r$) | $\to0$ ($r^m$) |

A single `bc=(None,0)` basis reproduces a mixed-$m$ manufactured solution to $1.7\times10^{-13}$; a naive `bc=(0,0)` would force $u_z(0)=0$ — fatal because Hagen–Poiseuille peaks on the axis. Pressure uses the orthogonal $P_N/P_{N-2}$ pair (`p_trunc=2`), with the $m=0$ constant null space pinned by constraint `((3,0,0))`. (The older disc/`pipe_poisson.py`/`unitdisc_helmholtz.py` demos instead use an explicit $m=0$ split [Shen94]; the pipe DNS deliberately uses the unified single-basis approach.)

### I.A.3.3 Time integrator and exact laminar oracles

Same CNAB2 / coupled saddle-point per $(m,k_z)$ as §I.A.2.4. For a **time-dependent** body force (Womersley) the force is evaluated at the midpoint $t^{n+1/2}$ for 2nd-order accuracy (`pipe_flow_dns.py:299-309`). Exact reference solutions (`pipe_flow_dns.py:473-500`, verified verbatim):

- **Hagen–Poiseuille:** $u_z(r)=\dfrac{f_z}{4\nu}(R^2-r^2)$, flow rate $Q=\dfrac{\pi R^4 f_z}{8\nu}$.
- **Womersley** ($-\partial_z p=K\cos\omega t$, $\rho=1$, $\alpha=R\sqrt{\omega/\nu}$, $\mathrm{i}^{3/2}=e^{\mathrm{i}3\pi/4}$): $\displaystyle u_z(r,t)=\mathrm{Re}\!\left\{\frac{K}{\mathrm{i}\omega}\!\left[1-\frac{J_0(\mathrm{i}^{3/2}\alpha r/R)}{J_0(\mathrm{i}^{3/2}\alpha)}\right]e^{\mathrm{i}\omega t}\right\}$ [Womersley55].
- **Bessel viscous decay:** $u_z(r,0)=J_0(j_{0,n}r/R)$, decaying as $\exp(-\nu j_{0,n}^2 t/R^2)$.

Golden tolerances (`test_pipe_flow_dns.py`, `pipe_flow_notes.md:67-71`): Hagen–Poiseuille $\max|u_z-\text{exact}|<10^{-6}$, $|Q-Q_{\text{exact}}|/Q_{\text{exact}}<10^{-10}$ (measured $1.4\times10^{-12}$), $\mathrm{div}_{\infty}<10^{-10}$; Bessel decay rate vs $\nu j_{0,1}^2/R^2=5.78319$ measured $5.78320$ (rel $2.8\times10^{-6}$); Womersley ($\alpha=3,\omega=9$) $\max|u_z-\text{exact}|<5\times10^{-6}$ (measured $8\times10^{-7}$); 3D $\mathrm{div}_{L_2}<10^{-9}$. Hagen–Poiseuille is linearly stable for all $\mathrm{Re}$; the $\mathrm{Re}=3000$, $m=1$, $k=1$ case is strongly non-normal (transient lift-up then bounded decay) [EBHW07]. **There is no pipe MHD in family A** (§3, deferred, low parity value).

---

## I.A.4 MHD

There are two PCF MHD codes plus the TC MRI eigensolver. The **canonical Lorentz prefactor is 1** (Alfvén / Lorentz–Heaviside units, $\rho=\mu_0=1$), uniformly across family A.

### I.A.4.1 PCF DNS — vector-potential formulation $B=\mathrm{curl}(A)$

The induction is advanced for the vector potential $A$ in the Weyl gauge, which makes $\mathrm{div}(B)=0$ a discrete identity by construction (`pcf_mhd_divfree.py:6-14`):

$$\frac{\partial A}{\partial t}=U\times B+\eta\,\nabla^2 A, \qquad B=\mathrm{curl}(A), \qquad J=\mathrm{curl}(B), \qquad U=U_{\text{wall}}\,x\,\mathbf{e}_y+u'.$$

**Compatible-space chain** (the $\mathrm{div}(B)=0$ invariant): $A\in CD=[TD,TD,TD]\to B=\mathrm{curl}(A)\in CC=[TD,TC,TC]\to J=\mathrm{curl}(B)\in JS=[TC,TD,TD]$; $B$ and $J$ are reprojected before every nonlinear evaluation/diagnostic. Each component obeys $\partial_t A_i=\eta\,\mathrm{div}(\mathrm{grad}(A_i))+(U\times B)_i$ with diffusion implicit and EMF explicit, solved by `chebyshev.la.Helmholtz`/`la.SolverGeneric1ND`.

The **Lorentz force** enters the KMM nonlinear store as $N-J\times B$ (`pcf_mhd_divfree.py:325-333`):

$$l_x=J_y B_z-J_z B_y,\quad l_y=J_z B_x-J_x B_z,\quad l_z=J_x B_y-J_y B_x,\qquad n_i \mathrel{-}= l_i,$$

with prefactor 1 (no $1/(\mu_0\rho)$ anywhere) and full $J=\mathrm{curl}(\mathrm{curl}(A))$. The EMF $U\times B$ uses the total Couette velocity. Normalization: $\nu=U_{\text{wall}}/\mathrm{Re}$, $\eta=U_{\text{wall}}/\mathrm{Rm}$, $\mathrm{Pm}=\mathrm{Rm}/\mathrm{Re}=\nu/\eta$; `Rm` defaults to `Re` if unset; default `Re=Rm=400`. Time integrator: IMEXRK222 default.

**PCF magnetic wall (DNS) = perfectly-conducting / no-normal-flux, enforced by basis design.** All three $A$-components are Dirichlet ($A=0$ at walls) because $A\in CD=\mathrm{VectorSpace}(TD)$. With $A_y=A_z=0$ at $x=\pm1$, $B_x=\partial_y A_z-\partial_z A_y=0$ → no normal field, tangential field free; $A_x=0$ is a gauge fix. This is the **only** DNS magnetic wall; pseudo-vacuum / true-insulating / thin-wall Robin are catalogued as unimplemented (parity gaps WS-D/WS-F, Part III.4).

Golden div-control numbers (`pcf_mhd_divfree_notes.md:69-96`): Legendre $N=(8,8,8)$, $\mathrm{Re}=\mathrm{Rm}=400$, $t=0.003$ → $\mathrm{div}\,U_{L_2}=9.41\times10^{-17}$, $\mathrm{div}\,B_{L_2}=3.05\times10^{-21}$; Chebyshev $N=(16,16,16)$ → $\mathrm{div}\,U_{L_2}=9.03\times10^{-17}$, $\mathrm{div}\,B_{L_2}=4.71\times10^{-21}$; near-transition $N=(24,48,24)$, $t=5.0$ → $\mathrm{div}\,B$ rel RMS $=8.32\times10^{-16}$. (The historical *direct-B* scheme leaked $\mathrm{div}\,B$ from $2.2\times10^{-17}$ to $5.9\times10^{-7}$ — the reason the vector-potential form is used.)

### I.A.4.2 PCF dense linear MHD operator

Collocation primitive variables add $(b_x,b_y,b_z,\phi)$, where $\phi$ is a magnetic-pressure Lagrange multiplier enforcing $\mathrm{div}(b)=0$ (`_pcf_linear.py:240-245`). Induction couplings with imposed uniform field $B_0=(0,b_y,b_z)$ (`:230-245`): $k_B=k_y b_y+k_z b_z$, $\mathrm{i}k_B$; symmetric $u\leftrightarrow b$ coupling $\pm\mathrm{i}k_B$; magnetic diffusion $\eta\,\mathrm{lap}$; **shear-induction $L[b_y,b_x]=U'$** (the $\Omega$-effect; with $U'=-S$ this is the $-S B_x$ azimuthal-field generation). Magnetic BCs by tau-style row replacement: `conducting` (default) imposes $b_x=0$ (Dirichlet) + tangential Neumann $b_y'=b_z'=0$; `dirichlet` pins all $b=0$ (diagnostic only). **No insulating wall in PCF linear** (only `conducting`/`dirichlet`). Energy-norm cross-checks at $B_0=0$ confirm the kinetic norm matches hydro to 9 digits while the total norm differs (independent magnetic transient growth from the $\Omega$-effect): hydro $G(50)=3.0741038\times10^{-4}$, MHD kinetic $=3.0741038\times10^{-4}$, magnetic $=3.3750398\times10^{-3}$ (`couette_linear_benchmarks.md:185-194`).

### I.A.4.3 TC MHD/MRI eigensolver

Field in Alfvén units ($v_A=B_{0z}$); total pressure $\Pi=p+B_0 b_z$ absorbs the imposed-field magnetic pressure so the Lorentz force is simply $\mathrm{i}k_z B_0 b$ per component. The linearized equations ($U=r\Omega(r)\mathbf{e}_\theta$, perturbations $\sim e^{st+\mathrm{i}m\theta+\mathrm{i}k_z z}$; `taylor_couette_mri.py:19-34`, verified verbatim):

$$s\,u_r=-\mathrm{i}m\Omega u_r+2\Omega u_\theta-\partial_r\Pi+\nu L_v[u_r]+\mathrm{i}k_z B_0 b_r,$$
$$s\,u_\theta=-\mathrm{i}m\Omega u_\theta-2a u_r-\tfrac{\mathrm{i}m}{r}\Pi+\nu L_v[u_\theta]+\mathrm{i}k_z B_0 b_\theta,$$
$$s\,u_z=-\mathrm{i}m\Omega u_z-\mathrm{i}k_z\Pi+\nu L_p[u_z]+\mathrm{i}k_z B_0 b_z,$$
$$0=\big(\partial_r+\tfrac1r\big)u_r+\tfrac{\mathrm{i}m}{r}u_\theta+\mathrm{i}k_z u_z,$$
$$s\,b_r=\mathrm{i}k_z B_0 u_r-\mathrm{i}m\Omega b_r+\eta L_v[b_r],$$
$$s\,b_\theta=\mathrm{i}k_z B_0 u_\theta+r\Omega' b_r-\mathrm{i}m\Omega b_\theta+\eta L_v[b_\theta],$$
$$s\,b_z=\mathrm{i}k_z B_0 u_z-\mathrm{i}m\Omega b_z+\eta L_p[b_z],$$

with scalar Laplacian $L_p=\partial_{rr}+\tfrac1r\partial_r-(m^2/r^2+k_z^2)$, vector diagonal $L_v=L_p-1/r^2$, cross terms $\pm2\mathrm{i}m/r^2$, and $2a=2\Omega+r\Omega'$ (const). The radial induction has **no** shear source (only advection); the azimuthal field is generated from $b_r$ at rate $r\Omega'=r\,\mathrm{d}\Omega/\mathrm{d}r$ — the MRI field-stretching term. The operator is split $L=L_0+\nu L_\nu+\eta_{\text{mag}}L_\eta$ for cheap critical-parameter bisection. Normalization (Liu/Goodman/Ji): $\mathrm{Pm}=\nu/\eta_{\text{mag}}$, $\mathrm{Re}=\Omega_1 R_1 d/\nu$, $\mathrm{Rm}=\Omega_1 R_1 d/\eta_{\text{mag}}$, Lundquist $S=B_0 d/\eta_{\text{mag}}$, $\mathrm{Ha}=B_0 d/\sqrt{\nu\eta_{\text{mag}}}$.

**Conducting walls** (any $m$): exactly as §I.A.2.5 (Dirichlet $b_r=0$ / Robin $b_\theta+r b_\theta'=0$ / Neumann $b_z'=0$).

**Insulating walls** ($m=0$ only, poloidal flux function $\chi$): with $b_r=-(\mathrm{i}k_z/r)\chi$, $b_z=(1/r)\chi'$ the divergence vanishes identically. The 6-field system $(u_r,u_\theta,u_z,\Pi,\chi,b_\theta)$ has

$$s\,\chi=-B_0 r u_r+\eta L_\chi\chi \;\; (L_\chi=\partial_{rr}-\tfrac1r\partial_r-k_z^2), \qquad s\,b_\theta=\mathrm{i}k_z B_0 u_\theta-\mathrm{i}k_z\Omega'\chi+\eta L_v b_\theta.$$

The vacuum match is a single-field Robin $\chi'/\chi=k_z^2/\kappa$ with $\kappa$ the modified-Bessel log-derivative of the exterior potential: $\kappa_{\text{in}}=k\,I_1(kR_1)/I_0(kR_1)$, $\kappa_{\text{out}}=-k\,K_1(kR_2)/K_0(kR_2)$ → shenfun Robin coefficient $c=-\kappa/(k_z^2 J)$ (`taylor_couette_mri.py:366-371`, verified verbatim); toroidal $b_\theta=0$ (vacuum toroidal field vanishes). $k_z=0$ raises `ValueError`; $m\neq0$ insulating raises `NotImplementedError`.

**Sign-distinguishing golden numbers** ($\eta=0.5$ quasi-Keplerian, $\mathrm{Pm}\to0$, [LL07]/Rüdiger 2023; `couette_linear_benchmarks.md:352-353`): conducting target $\mathrm{Rm}=24.7$, $S=4.11$, best $k_z=1.75$ → $\text{growth}=+0.003322863594034156$; insulating $\mathrm{Rm}=16.5$, $S=5.21$, best $k_z=1.25$ → $\text{growth}=-0.00027582037141390655$ (insulating destabilizes more easily). The local ideal Keplerian WKB MRI [BH91] is reproduced to $\sim10^{-3}$: $s_{\max}/\Omega=0.7499999944$ (theory $0.75$), $(k v_A)^2/\Omega^2=0.93732$ (theory $15/16=0.9375$), cutoff $3\Omega^2$.

---

## I.A.5 MRI source terms (wall-bounded shearing box, PCF)

`pcf_mhd_mri_shearpy.py` is the PCF MHD analogue of the shearpy shearing-box MRI: a wall-bounded box (the radial/shear direction is replaced by no-slip PCF walls, **not** a shearing-periodic remap). It subclasses `pcf_mhd_divfree.py` and overrides the base flow plus adds the rotation/shear source terms. Convention (verified verbatim, `pcf_mhd_mri_shearpy.py:6-15`): $x$ radial/wall-normal/shear-gradient, $y$ azimuthal/streamwise, $z$ vertical/rotation axis. Base flow and source terms:

$$U_b(x)=-S\,x\,\mathbf{e}_y, \qquad \frac{\mathrm{d}U_b}{\mathrm{d}x}=-S,$$
$$\frac{\mathrm{d}u_x}{\mathrm{d}t}\mathrel{+}=2\Omega\,u_y \;\;(\text{Coriolis}), \qquad \frac{\mathrm{d}u_y}{\mathrm{d}t}\mathrel{+}=(S-2\Omega)\,u_x \;\;(\text{Coriolis}+\text{base-flow shear}),$$
$$\frac{\mathrm{d}B_y}{\mathrm{d}t}\mathrel{+}=-S\,B_x \;\;(\text{shear induction}), \qquad +\,J\times B_{\text{total}} \;\;(\text{Lorentz},\; J=\mathrm{curl}(\mathrm{curl}(A))).$$

The override is `self.Ub=-self.shear_rate*self.X[0]`, `self.dUb_dx=-self.shear_rate` (`pcf_mhd_mri_shearpy.py:102-104`, verified verbatim).

**Implementation details.** Because KMM stores $H=N-F$ (the velocity equations apply $-H$ after projection), the Coriolis additions are entered as `n0 += -2*Omega*u_y; n1 += 2*Omega*u_x` (`:346-348`), which yield the desired $+2\Omega u_y$, $-2\Omega u_x$ sources; the $+S u_x$ part of the $\mathrm{d}u_y$ term arises from base-flow advection ($U_b\partial_y$) plus $u_x\,\mathrm{d}U_b/\mathrm{d}x=u_x(-S)$, combining to the net $(S-2\Omega)u_x$. The **shear-induction $\mathrm{d}B_y/\mathrm{d}t=-S B_x$ is not a separate update**: it follows automatically from $\mathrm{d}A/\mathrm{d}t=U\times B$ with $U_b=-S x\,\mathbf{e}_y$.

**Imposed uniform field** $B_0=(0,b_y,b_z)$ is carried separately from $\mathrm{curl}(A)$ so the $\mathrm{div}(\mathrm{curl}\,A)=0$ invariant is preserved; the Lorentz force uses $B_{\text{total}}=\mathrm{curl}(A)+B_0$ but $J=\mathrm{curl}(\mathrm{curl}(A))$ only (the imposed field is current-free); the EMF uses $B_{\text{total}}$.

**Dimensionless groups** (`pcf_mhd_mri_shearpy.py:107-108`, verified verbatim): $q=S/\Omega$ (Keplerian $q=3/2$ at default $S=1,\Omega=2/3$); epicyclic

$$\kappa^2=2\Omega(2\Omega-S)=2\Omega^2(2-q),$$

stable ($\kappa^2>0$) for $q<2$. $\mathrm{Re}=U/\nu$, $\mathrm{Rm}=U/\eta$, $\mathrm{Pm}=\nu/\eta$, $v_A=b_z$. Defaults: $N=(16,32,16)$, $\text{domain}=((-2,2),(0,4),(0,1))$, $\mathrm{Re}=\mathrm{Rm}=1000$, $S=1$, $\Omega=2/3$, $b_y=0$, $b_z=0.025$, $\mathrm{d}t=0.001$. Ideal MRI unstable band $0<(k_z v_A)^2<4\Omega^2(q-1)$ with $\gamma_{\max}=(q/2)\Omega=0.75\Omega$ at $k_z v_A\approx(\sqrt{15}/4)\Omega$; optimal channel mode $K_0=\sqrt{15}/4\cdot\Omega/v_A$. Transport diagnostics: Reynolds $\langle u_x u_y\rangle$, Maxwell $-\langle B_x B_y\rangle$ (total field), $\alpha=(\text{Reynolds}+\text{Maxwell})/v_A^2$, plus a Jackson–Krommes mean-shear cancellation metric.

Validation (`couette_linear_benchmarks.md:313-317`): PCF rotating-shear MRI analogue at $\Omega=2/3$, $b_z=0.025$, $k_z=25.81988897471611$ → leading eigenvalue $s\approx0.498406$ (theory $s=0.5$); DNS netflux case grows $E_{\text{mag}}$ monotonically (test asserts $E_{\text{mag}}[-1]>2 E_{\text{mag}}[0]$, $\max(\mathrm{div}\,b)<10^{-10}$). The DNS MRI growth-rate match to linear theory is a Part III closure item (WS-A); the existing assertion is qualitative (energy growth + solenoidality).

---

## I.A.6 Time integrators (exact tableaux & update equations)

Family A exposes two integrator families. All steppers solve $\partial u/\partial t=N+Lu$ with $L$ (diffusion / linear couplings) treated implicitly and $N$ explicitly.

### I.A.6.1 PDEIMEXRK family (Ascher–Ruuth–Spiteri, condition 2.3) [ARS97]

`stages()` returns $(a,b,c)$ with $a$ the implicit (DIRK) tableau, $b$ the explicit tableau, indexed $[\text{rk}+1,j]$. The per-stage update (`integrators.py:798-817`, verified verbatim):

$$\text{rhs} = u_0\text{-rhs} + \sum_{j=0}^{\text{rk}}\Delta t\,b_{\text{rk}+1,j}\,K_{\text{rhs}}[j] + \sum_{j=0}^{\text{rk}-1}\Delta t\,a_{\text{rk}+1,j+1}\,L_{\text{rhs}}[j],$$

then a single implicit solve with the **once-factored** operator (the implicit LHS is $\mathrm{inner}(v,\,u_l-\Delta t\,a_{11}\,L(u_l))$, reused because the diagonal $a_{11}$ is constant across stages, `integrators.py:787,816`). The Nyquist mode is masked on the RHS. The tableaux (verbatim from `integrators.py`; identical copies in `_linear_analysis.py:128-160` used by the linear steppers and by family C):

**IMEXRK111** — 1 stage, 1st order (`integrators.py:836-850`):
$$a=\begin{bmatrix}0&0\\0&1\end{bmatrix},\quad b=\begin{bmatrix}0&0\\1&0\end{bmatrix},\quad c=(0,1),\quad\text{steps}=1.$$

**IMEXRK222** — 2 stages, 2nd order, L-stable (**DEFAULT for PCF subclasses, PCF MHD, and the TC IMEXRK stepper**) (`integrators.py:852-870`). With $\gamma=(2-\sqrt2)/2\approx0.2928932188$ and $\delta=1-1/(2\gamma)\approx-0.7071067812$:
$$a=\begin{bmatrix}0&0&0\\0&\gamma&0\\0&1-\gamma&\gamma\end{bmatrix},\quad b=\begin{bmatrix}0&0&0\\\gamma&0&0\\\delta&1-\delta&0\end{bmatrix},\quad c=(0,\gamma,1),\quad\text{steps}=2.$$
(Note $\delta<0$.)

**IMEXRK443** — 4 stages, 3rd order (`integrators.py:872-892`):
$$a=\begin{bmatrix}0&0&0&0&0\\0&\tfrac12&0&0&0\\0&\tfrac16&\tfrac12&0&0\\0&-\tfrac12&\tfrac12&\tfrac12&0\\0&\tfrac32&-\tfrac32&\tfrac12&\tfrac12\end{bmatrix},\quad b=\begin{bmatrix}0&0&0&0&0\\\tfrac12&0&0&0&0\\\tfrac{11}{18}&\tfrac1{18}&0&0&0\\\tfrac56&-\tfrac56&\tfrac12&0&0\\\tfrac14&\tfrac74&\tfrac34&-\tfrac74&0\end{bmatrix},\quad c=(0,\tfrac12,\tfrac23,\tfrac12,1),\;\text{steps}=4.$$

(Also `IMEXRK011`, `integrators.py:819-833`: $a=[[0,0],[0,0]]$, $b=[[0,0],[1,0]]$, $c=(1,0)$, steps $=1$.)

### I.A.6.2 IMEXRK3 — the base-`KMM` default (semi-implicit RK3, Spalart-style) [SMR91]

A **separate** class (not PDEIMEXRK), `integrators.py:603-700`, 3 stages, 3rd order. Coefficients (`integrators.py:665-669`, verified verbatim):
$$a=(\tfrac{8}{15},\,\tfrac{5}{12},\,\tfrac34),\qquad b=(0,\,-\tfrac{17}{60},\,-\tfrac{5}{12}),\qquad c=(0,\,\tfrac{8}{15},\,\tfrac23,\,1),\qquad\text{steps}=3.$$
The implicit LHS per stage is Crank–Nicolson-like, $\mathrm{inner}(v,\,u_l-(a_{\text{rk}}+b_{\text{rk}})\tfrac{\Delta t}{2}L(u_l))$, and a **separate solver is assembled per stage** (`integrators.py:678-680`). The per-stage RHS combines an Adams–Bashforth-like explicit nonlinear term with the CN linear term (`integrators.py:692-693`, verified verbatim):
$$\text{rhs} = \mathrm{inner}\!\Big(v,\,u+(a_{\text{rk}}+b_{\text{rk}})\tfrac{\Delta t}{2}L(u)\Big) + \Delta t\,\big(a_{\text{rk}}\,w_0 + b_{\text{rk}}\,\text{rhs}_0\big),$$
where $w_0$ is the current nonlinear inner product and $\text{rhs}_0$ the previous one. This is the closest "CNAB-style" path in the KMM stack.

### I.A.6.3 CNAB2 (default for all TC and pipe DNS)

2nd-order IMEX: Crank–Nicolson ($\pm\tfrac12$ implicit/explicit) for the full linear operator $A$ (viscous/resistive $+$ all base-flow couplings: Coriolis $2\Omega$, shear $-2a$, $-\mathrm{i}m\Omega$ advection, $r\Omega'$ field-stretching, $B_0\partial_z$) $+$ pressure gradient; Adams–Bashforth-2 ($\tfrac32 N^n-\tfrac12 N^{n-1}$) for the nonlinear term; IMEX-Euler bootstrap on the first step. The exact step code is in §I.A.2.4. Pre-assembled `Limp = BlockMatrixSolver(M/\Delta t-\tfrac12 A; \text{div}=0)` and `Lexp = BlockMatrix(M/\Delta t+\tfrac12 A)`; one coupled saddle-point solve per Fourier mode with the pressure null space pinned by `constraints=((3,0,0),)`. CNAB2 is the closest family-A counterpart to family B's $\theta$-method PC (Part I.B.2) and to family C's CNAB2 (Part I.C.5).

### I.A.6.4 Implicit/explicit split summary

| Term | Treatment | Where assembled |
|---|---|---|
| Viscous $\nu\nabla^2$ / $\nu\nabla^4$, resistive $\eta\nabla^2$ | **Implicit** | once in `assemble()` / `Limp` |
| Pressure gradient + continuity (saddle rows) | Implicit (CNAB2/coupled solve) or eliminated (KMM) | `Limp` / KMM recovery |
| Linear base couplings (Coriolis, shear $-2a$, $-\mathrm{i}m\Omega$, $r\Omega'$, $B_0\partial_z$) | Implicit in CNAB2; explicit in IMEXRK `--split diffusion` | per-mode block / RHS |
| Nonlinear convection, Lorentz $J\times B$, EMF $U\times B$ | **Explicit** (AB2 in CNAB2; $b$-tableau in IMEXRK) | per-step `nonlinear()` |

The IMEXRK linear steppers support `--split diffusion` (diffusion + pressure/continuity implicit, base couplings explicit) vs `--split full` (everything implicit, stiff reference). The descriptor-system stage solve is $(M-\Delta t\,\gamma\,A_{\text{imp}})x=\text{rhs}$ with constraint rows pinned (`_linear_analysis.py:163-186`).

---

## I.A.7 Reimplementation checklist (load-bearing facts)

1. KMM evolves $\nabla^2 u_x$ (not $u_x$) → the implicit $u_x$ solve is **biharmonic**; $u_x\in$ clamped `bc=(0,0,0,0)`, $g\in$ Dirichlet `bc=(0,0)`.
2. Recovery uses $f=+\partial_x u_x$ (code) with $u_y=\mathrm{i}(k_y f+k_z g)/k^2$, $u_z=\mathrm{i}(k_z f-k_y g)/k^2$; $g=\mathrm{i}k_y u_z-\mathrm{i}k_z u_y$.
3. The $(0,0)$ Fourier mode is special — two 1-D Helmholtz PDEs on rank 0 with source $-\mathrm{d}p/\mathrm{d}y$.
4. Default integrator is **IMEXRK222** in the PCF/MHD subclasses (base `KMM` default is `IMEXRK3`); $\delta\approx-0.7071$ in IMEXRK222.
5. Dealiasing is Fourier-only at 3/2; wall-normal $x$ unpadded; Nyquist masked after every nonlinear/curl evaluation.
6. Base flow lives only in the convection term; the shear-production term is `up[0]*dUb_dx`.
7. TC uses explicit-$1/r$ plain measure; pipe uses curvilinear $\sqrt{g}=r$ — build $\Omega(r)$ / $1-r^2$ factors in **the space's own** radial symbol.
8. Pipe pole regularity is automatic from singular operator penalties with one `bc=(None,0)` basis (no $m$-split).
9. The conducting $b_\theta$ Robin coefficient is in reference coordinates: $c=r_{\text{wall}}/J$, $J=(R_2-R_1)/2$.
10. MHD is in Alfvén units (Lorentz prefactor 1); PCF DNS uses $B=\mathrm{curl}(A)$ (div-free by construction); TC uses direct $b$-components, $\mathrm{div}(b)$ monitored not projected; seed solenoidally.
11. Coupled saddle-point solve = exact incompressibility; pin the pressure null space with `((3,0,0))`; divergence diagnostics need separate per-term projections.
12. Linear operators have singular mass $M$ (zero pressure/$\phi$ mass) → filter infinite eigenvalues with `FINITE_CAP=1e8`.


\newpage

# Part I.B — torch finite-difference family: channel, couette, pipe

This part gives the complete algorithmic specification of **family B** (`torch`): a set of three PyTorch solvers — plane channel/Couette (`torchchannel`), Taylor–Couette annulus (`torchcouette`), and pipe (`torchpipeflow`). All three share one architecture: **Fourier spectral** in the periodic directions, **dense (channel/couette) or banded (pipe) finite differences** in the wall-normal/radial direction, an **influence/capacitance-matrix** boundary correction with pressure-Poisson projection enforcing div-free, and a **semi-implicit θ-method predictor–corrector** time integrator. Channel and couette carry full-induction MHD; the pipe is hydrodynamic only. The entire family is device-agnostic PyTorch (GPU via CUDA), `complex128` by default, and **fully autograd-differentiable**.

Two contradictions from §0.2 are honored throughout: (i) family B's axis labels are *swapped* relative to the canonical frame (B uses streamwise=`x`, wall-normal=`y`; the canonical frame uses wall-normal=`x`, streamwise=`y`) — apply the planned §0.2 `to_canonical()` adapter before any cross-family comparison; (ii) the "4th-order FD" family label is conservative — the default `KL=4` 9-point stencil is formally ~8th-order in the interior (see §I.B.1). The MRI capability is a **metadata-only stub** (§I.B.6), the headline gap closed by Part IV/SR-1,SR-2.

Package roots (all paths absolute):
`/home/nauman/cfd/fn_openpipeflow-122/torchchannel/torchchannel/`,
`/home/nauman/cfd/fn_openpipeflow-122/torchcouette/torchcouette/`,
`/home/nauman/cfd/fn_openpipeflow-122/torchpipeflow/torchpipeflow/`.

---

## I.B.1 Channel & Couette core (`torchchannel`, `torchcouette`)

### Governing equations and nondimensionalization

**Channel/plane-Couette.** Incompressible Navier–Stokes on a box periodic in $(x,z)$ with no-slip walls at $y=\pm 1$:
$$
\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u = -\nabla p + \frac{1}{Re}\,\nabla^2\mathbf u,
\qquad \nabla\cdot\mathbf u = 0 .
$$
The *only* hydrodynamic nondimensional parameter is $Re=UL/\nu$; viscosity enters the implicit Helmholtz solve as the single diffusivity $1/Re$ (`solver.py:225`, `_build_diffusion_system(1.0/self.Re)`). $Re$ must be positive and finite. There are **no rotation/shear/MRI source terms** in the hydro path. The axes are: $x$ = streamwise, $y$ = wall-normal, $z$ = spanwise. **This is the swapped convention of §0.2** — wall-normal is $y$ here, not the canonical $x$.

**Taylor–Couette.** Cylindrical incompressible NS in **viscous units**: the radial Laplacian carries unit viscosity (operator coefficient `c2 = -implicit` with no $1/Re$, `solver.py:258-260`), and the wall speeds carry the Reynolds numbers $Re_i, Re_o$ directly. Geometry is gap-scaled (gap $=1$):
$$
r_i = \frac{\eta}{1-\eta},\qquad r_o=\frac{1}{1-\eta}\qquad(\texttt{base\_flow.py:10-15}).
$$
Components are $(r,\theta,z)$ = (radial, azimuthal, axial). The nondimensionalization differs from channel: it pushes $Re$ into the boundary values, not the diffusion coefficient.

### Coordinate system, base flows, sign conventions (VERBATIM)

| Flow | Base profile | Derivatives | Walls | Source |
|---|---|---|---|---|
| Plane Couette | $U(y)=y$ | $U'=1,\ U''=0$ | $u(\mp1)=\mp1$ | `base_flow.py:38-41` |
| Poiseuille | $U(y)=1-y^2$ | $U'=-2y,\ U''=-2$ | stationary $(0,0)$ | `base_flow.py:43-46` |
| Taylor–Couette | $u_\theta(r)=ar+b/r$ | — | $(Re_i,Re_o)$ | `base_flow.py:18-25` |

The circular-Couette coefficients are (`base_flow.py:23-24`):
$$
a=\frac{Re_o-\eta\,Re_i}{1+\eta},\qquad
b=\frac{\eta\,(Re_i-\eta\,Re_o)}{(1-\eta)(1-\eta^2)} .
$$
For canonical comparison, B's plane-Couette $U=y\,\mathbf e_x$ describes the *same physical flow* as A/C's $U_b=U_{\text{wall}}\,x\,\mathbf e_y$; the §0.2 adapter relabels $(x_B,y_B,z_B)\to(y_{\text{can}},x_{\text{can}},z_{\text{can}})$ (and flips shear sign for MRI comparisons).

### Spatial discretization: Fourier × finite differences

**Periodic directions** use Fourier. The wall-normal/radial direction uses **dense FD matrices** built from a Taylor/Vandermonde system, *not* Chebyshev differentiation matrices.

The FD weights mirror OpenPipeFlow's `mes_weights` (`mesh.py:12-45`). For target $x_0$ and stencil $\{x_j\}$, build $A_{:,0}=1$, $A_{:,j}=A_{:,j-1}\cdot(x-x_0)/j$ so that $A_{j}=(x-x_0)^j/j!$, then solve the **transposed** system $A^{\mathsf T}\mathbf w=\mathbf e_{\text{deriv}}$ for the weights (`mesh.py:43-45` — solving the transpose avoids forming the ill-conditioned inverse; the couette variant `mesh.py:25` uses `inv(A)[deriv]` directly).

**Stencil width and order.** `KL` is the half-bandwidth; the interior stencil is $\min(2\,KL+1,\,N)$ centered points (`mesh.py:109-117`), default `KL=4` $\Rightarrow$ **9-point centered stencil**. With 9 points the interior FD is polynomial-exact through degree 8, i.e. formally **~8th-order interior** despite the "4th-order FD" family label. Boundary stencils are one-sided (`_stencil_bounds`, `mesh.py:103-106`) and lower order, setting the realized global floor. Tests assert polynomial exactness for degrees 0..8 to $<10^{-7}$ (`tests/test_mesh.py:21-37`). **Acceptance posture:** state "4th-order family label; ≥4th-order floor, ~8th interior"; convergence tests assert slope $\ge 3.7$ (Part IV/S2).

**Critical operator identity.** The second-derivative matrix is *not* an independent stencil — it is the square of the first:
$$
W_{dy2} = W_{dy1}\,W_{dy1}\qquad(\texttt{mesh.py:189}).
$$
This is deliberate so that $\operatorname{div}(\operatorname{grad}p)\equiv\nabla^2 p$ holds *exactly* for the pressure projection. (Couette stores an independent $W_{dr2}$ but the projection uses $W_{dr1}^2$; see §I.B.3.) The couette radial Laplacian is baked in as $W_{\text{radlap}}=W_{dr2}+(1/r)\,W_{dr1}$ (`mesh.py:155`), the $\partial_{rr}+(1/r)\partial_r$ part of the cylindrical Laplacian.

**Mesh points.** Channel: Chebyshev extrema $y=-\cos(\pi j/(N-1))$ when `clust=0`, else nsCouette stretching $y=\arcsin(-c\cos\theta)/\arcsin(c)$; endpoints pinned exactly to $\pm1$ (`mesh.py:80-100`). Couette: ascending radial grid pinned to $[r_i,r_o]$.

**Mode counts (differ between channel and couette).**
- Channel: full symmetric double Fourier, $K1=K-1$, $Kc=2K1+1$, $Mc=2M1+1$, with $k\in[-K1,K1]$, $m\in[-M1,M1]$ (`operators.py:35-43`). Wavenumbers $k_\alpha=k\alpha$, $m_\beta=m\beta$, $k^2=k_\alpha^2+m_\beta^2$ (`operators.py:44-46`).
- Couette: nsCouette storage — axial slots $[0..K,-K{+}1..{-}1]$ with $Kc=2K$; azimuthal stored non-negative $[0..M-1]$ with Hermitian completion (`spectral.py:32-44`).

**Quadrature.** Channel uses plain $\int_{-1}^{1}$ weights `inty` with low-order moments corrected exactly ($\int1=2,\ \int y=0,\ \int y^2=2/3$, `mesh.py:159-164`). Couette uses the cylindrical Jacobian $\int f(r)\,r\,dr$ via `intrdr` (`mesh.py:58-81`).

### FFT layout and dealiasing

Both use **3× padded** dealiasing (more conservative than the 2/3 rule). Padded physical grid $Z=\texttt{dealias\_mult}\cdot K$, $Th=\texttt{dealias\_mult}\cdot M$, default `dealias_mult=3` (`spectral.py:23,37-38`). Retained modes are scattered into the zero-padded spectrum, `ifft2` to physical, products formed pointwise on the fine grid, `fft2` back, then truncated (`coll_to_phys`/`phys_to_coll`, `spectral.py:91-114`).

**Normalization differs between the two solvers** (a reimplementation trap):
- Channel uses plain `torch.fft.fft2`/`ifft2` (no `norm=`), so a physical constant $c(y)$ lives in the mean coefficient as $c(y)\cdot(Z\cdot Th)$; `physical_scale = Z*Th` (`solver.py:115`).
- Couette uses `norm="forward"` (`spectral.py:87,96`), so the forward FFT carries the $1/N$ scaling.

Reality is enforced by $A[-k,-m]=\overline{A[k,m]}$ with $A[0,0]$ real (`spectral.py:65-79` channel; `enforce_m0_reality` couette `spectral.py:55-64`).

### Differential operators (`operators.py`)

Channel (Cartesian): $\partial_x f=i k_\alpha f$, $\partial_z f=i m_\beta f$, wall-normal $\partial_y,\partial_{yy}$ via `einsum("ij,...jkm->...ikm", W, f)`; $\nabla^2 f=\partial_{yy}f-k^2 f$; $\nabla\!\cdot\!\mathbf u=\partial_x u+\partial_y v+\partial_z w$; curl $\boldsymbol\omega=(\partial_y w-\partial_z v,\ \partial_z u-\partial_x w,\ \partial_x v-\partial_y u)$ (`operators.py:78-110`). Validated $\operatorname{curl}\operatorname{grad}=0$, $\operatorname{div}\operatorname{curl}=0$, $\operatorname{div}\operatorname{grad}=\nabla^2$ (`VALIDATION.md:35-37`).

Couette (cylindrical) uses the $\pm$ decomposition $u_\pm=u_r\pm i u_\theta$ (`operators.py:12-19`) which diagonalizes the $(u_r,u_\theta)$ Helmholtz coupling. The angular+axial spectral diagonal is
$$
\text{mode\_diagonal}(pm)=-\frac{(m+pm)^2}{r^2}-k_z^2,\quad pm\in\{0,+1,-1\}\quad(\texttt{operators.py:85-98}),
$$
and $\text{radlap}(f,pm)=W_{\text{radlap}}f+\text{mode\_diagonal}(pm)\,f$ (`operators.py:100-102`).

### Time integrator — semi-implicit θ-method predictor/corrector (WRITE THE UPDATE)

The scheme is a **one-stage θ-method** (Crank–Nicolson-like) with **diffusion split implicit/explicit** and **nonlinear + base-flow advection explicit**, wrapped in a fixed-point predictor/corrector. The implicit fraction is $\theta=\texttt{implicit}$, **default 0.51** (`solver.py:87`), validated $0\le\theta\le1$. At $\theta=0.51$ the scheme is **formally first-order** in time (second-order only at exactly $\theta=\tfrac12$, and even then the explicit base-flow advection caps the realized order at first-order); the slight over-implicitness ($0.51>0.5$) biases the viscous term to unconditional damping.

The per-mode diffusion system, with $L=W_{dy2}-k^2 I$ and diffusivity $\nu_{\rm eff}$ (`solver.py:215-216`):
$$
\boxed{\;L_{\rm lhs}=\tfrac{1}{\Delta t}I-\theta\,\nu_{\rm eff}\,L,\qquad
L_{\rm rhs}=\tfrac{1}{\Delta t}I+(1-\theta)\,\nu_{\rm eff}\,L\;}
$$
with $\nu_{\rm eff}=1/Re$ for velocity (`_build_diffusion_system(1/Re)`, `solver.py:225`) and $\nu_{\rm eff}=1/Rm$ for magnetic field (§I.B.5). Wall rows of $L_{\rm lhs}$ are overwritten with Dirichlet rows (`_write_*_boundary_row` order 0, `solver.py:217-218`). Each component is one implicit solve per corrector sweep.

The RHS assembly (`_rhs_for_state`, `solver.py:571-577`) for $c\in\{u,v,w\}$ is
$$
\text{rhs}_c = L_{\rm rhs}\,c^{\,\text{old}} + b_c + N_c,
$$
where the explicit base-flow coupling (perturbation form only, `_base_coupling_terms`, `solver.py:534-540`) is
$$
b_u=-U\,\partial_x u - U'\,v,\qquad b_v=-U\,\partial_x v,\qquad b_w=-U\,\partial_x w
$$
(the $-U'\,v$ term is the lift-up coupling), and $N_c$ is the explicit nonlinear term (default **rotational** $\mathbf u\times\boldsymbol\omega$, `solver.py:542-550`; or **convective** $-(\mathbf u\cdot\nabla)\mathbf u$, `solver.py:552-561`), formed pseudospectrally on the dealiased grid.

The full step (`step`, `solver.py:594-606`):
```
N_old  = compute_nonlinear(old)
rhs    = L_rhs·old + base_coupling + N_old
new    = solve_project_correct(rhs)             # predictor
repeat corrector_iterations times:              # default 1 (channel)
    N_new   = compute_nonlinear(new)
    blended = θ·N_new + (1−θ)·N_old             # solver.py:603
    rhs     = L_rhs·old + base_coupling + blended
    new     = solve_project_correct(rhs)        # corrector
```
The corrector blends the **nonlinear term** with the same weight $\theta$ (an Adams-style blend about the old state). Default `corrector_iterations=1` (channel, `solver.py:84-85`).

**Couette uses an iterated/adaptive corrector** instead of a fixed count (`step_with_info`, `solver.py:553-594`): it loops up to `max_corrector_iters` (default 3), blends $N=\theta N_1+(1-\theta)N_0$ (`solver.py:574-577`), measures $\text{err}=\max|\,\text{new}-\text{cur}\,|$ over components, breaks when $\text{err}<$ `tol` (default $10^{-10}$), and returns `StepInfo(iterations, err_history, converged)`. The couette operator coefficients are $c_1=1/\Delta t$, $c_2=-\theta$ (unit viscosity, `solver.py:257-260`), with RHS $\text{rhs}=N+\tfrac{1}{\Delta t}b+(1-\theta)[\text{radlap}(b)+d_{pm}\,b]$ (`_rhs_meshmult`, `solver.py:467-470`). The couette explicit convective nonlinearity carries the cylindrical metric terms $-u_\theta^2/r$ (radial centrifugal) and $+u_r u_\theta/r$ (azimuthal) (`solver.py:496-498`).

**The ordered solve-project-correct sequence is load-bearing** (`_solve_project_correct`, `solver.py:579-592`):
1. `project_rhs` — remove interior divergence (§I.B.3)
2. `_apply_velocity_boundary_rhs` — zero/impose wall RHS rows
3. solve Helmholtz for $u,v,w$
4. `correct_bc` — 8×8 influence-matrix BC correction
5. `enforce_constraints` — dense min-norm projection
6. `_adjust_flux` — constant-flux correction (Poiseuille)
7. `enforce_constraints` again
8. `enforce_mean_mode_cleanup` — mean $v=0$; mean $u,w$ real

**CFL.** $\Delta t=\text{cfl}\cdot\min$ over directions: streamwise $(2\pi/|\alpha|)/Z/\max|u|$, wall-normal $\Delta y_{\min}/\max|v|$, spanwise $(2\pi/|\beta|)/Th/\max|w|$ (`cfl_dt`, `solver.py:676-697`). Because base-flow advection is explicit, $\Delta t$ must respect the base-flow CFL (`README.md:29-31`). **Constant-flux** Poiseuille corrects the mean mode to hold flux $4/3$ via a precomputed shape $U_i$ (`_adjust_flux`, `solver.py:635-644`).

### Linear stability (channel) — Orr–Sommerfeld/Squire

`orr_sommerfeld_squire` (`linstab.py:61-136`) solves $A\mathbf q=\lambda B\mathbf q$ on the same FD matrices ($D=W_{dy1}$, $D2=W_{dy2}$) via `scipy.linalg.eig` (`linstab.py:109`). With $L=D2-k^2I$, $L2=L\,L$, $k^2=\alpha^2+\beta^2$ (`linstab.py:99-104`):
$$
A_{\rm OS}=-i\alpha(U L)+i\alpha U''+\tfrac1{Re}L2,\quad B_{\rm OS}=L,\quad
A_{\rm Sq}=-i\alpha U+\tfrac1{Re}L,\quad \text{coupling}=-i\beta U' .
$$
Eigenvalues are temporal rates $\lambda$ for $\exp(i\alpha x+i\beta z+\lambda t)$; phase speed $c=i\lambda/\alpha$ (`linstab.py:124-125`). OS BC rows clamp $v=Dv=0$ at both walls; Squire rows clamp $\eta=0$ (`linstab.py:42-58`).

---

## I.B.2 Influence/capacitance matrix, projection, and divergence-free enforcement

The div-free constraint is enforced by **three nested mechanisms**, all in `solver.py`.

**(a) Pressure-Poisson projection** `project_rhs` (`solver.py:495-502`): compute interior $\operatorname{div}$, zero its wall rows, solve $p=L_{\rm pois}^{-1}(\operatorname{div})$, subtract $\nabla p$. Because $W_{dy2}=W_{dy1}^2$, this removes *interior* divergence exactly.

**(b) Poisson operator** `L_pois` (`solver.py:226-242`): per-mode $W_{dy2}-k^2I$ with **Neumann** wall rows ($\partial_y$ stencil, `_write_*_boundary_row` order 1, `solver.py:232-233`). The mean mode $(k{=}0,m{=}0)$ is pinned for nonsingularity (`L_pois[h,-1,:]=0; [...,−1,−1]=1`, `solver.py:235-240`) but **masked out** of the actual solve (`_mask_nonmean`, `solver.py:129,263-272`). Pressure is never a prognostic variable — it is a projection auxiliary.

**(c) 8×8 real influence (capacitance) matrix.** Velocity BCs are no-slip Dirichlet; for `state_form="full"` the moving Couette walls are imposed as mean-mode wall values $\text{lo}\cdot Z\,Th$, $\text{hi}\cdot Z\,Th$ (`_apply_velocity_boundary_rhs`, `solver.py:331-339`). The residual vector per mode is 8 entries (`eval_bc`, `solver.py:295-314`):
$$
[\,u,v,w,\ \text{cont}\,]_{\rm lo},\ [\,u,v,w,\ \text{cont}\,]_{\rm hi},\qquad
\text{cont}=\partial_y v + i k\,u + i m\,w .
$$
`_build_influence_matrix` (`solver.py:357-420`) assembles 8 basis responses (velocity-unit wall sources for $u,v,w$ at each wall, plus a pressure-gradient response from a unit wall Poisson source). The complex residual is phase-transformed to a **real 8×8 system** — row scales $[1,-i,1,-i,1,-i,1,-i]$, column phases $[1,i,1,i,1,i,1,i]$ (`solver.py:316-320`) — and `ABC` is inverted per non-mean mode (`solver.py:416-419`); the mean mode is identity.

**Authoritative final cleanup** `enforce_constraints` (`solver.py:474-482`): builds a dense per-mode constraint matrix $C$ (`_build_constraint_pinv`, `solver.py:422-463`) of $(N-2)$ interior-divergence rows plus 8 wall rows (3 velocity + 1 continuity per wall), takes `pinv`, and applies a minimum-norm correction $\delta=-\,\text{pinv}\cdot\text{residual}$. This drives wall-inclusive divergence and wall residuals to roundoff (div $\sim10^{-7}$ for the pinv path; cf. §3 gap matrix).

**Couette** uses the analogue: `_pressure_radial_matrix = W_dr1@W_dr1 + W_dr1·(1/r)` (`solver.py:225-227`, again $W_{dr1}^2$ for exact $\operatorname{div}\operatorname{grad}$), a Neumann pressure Poisson `LNp` with the mean-mode outer row pinned (`solver.py:252-253`), and a **12-column basis → real 8×8** influence matrix (`_build_influence_matrix`, `solver.py:325-383`) with Fortran row order $[u_+^i,u_+^o,u_-^i,u_-^o,i u_z^i,i u_z^o,\text{div}^i,\text{div}^o]$ (`solver.py:288-320`). Solves are LU-factored and cached (`lu_factor`/`lu_solve`, `solver.py:262-265`) with complex RHS split into real/imag stacks (`_solve_batched`, `solver.py:269-274`).

---

## I.B.3 Pipe — banded-LU solver (`torchpipeflow`)

`torchpipeflow` is a serial, GPU-friendly, differentiable port of the **OpenPipeFlow** [Willis17] core. **Hydrodynamic only — no MHD** (verified: grep for magnetic/induction/Lorentz/Rm/Pm/Ha returns 0 substantive hits across the package). State holds only $(u_r,u_\theta,u_z)$ shaped $(N,Kc,M)$ complex (`solver.py:57-74`).

### Geometry, base flow, mode layout

Cylindrical $(r,\theta,z)$, $r\in[0,1]$ (wall at $r[-1]=1$, near-axis at $r[0]\approx0$). Spectral in $(z,\theta)$, banded FD in $r$. Axial modes $k\in[-K1,K1]$, $K1=K-1$, $Kc=2K1+1$; azimuthal $m\in[0,M-1]$ stored non-negative (Hermitian), with actual wavenumbers $k\alpha$ (axial) and $m\cdot Mp$ (azimuthal, $Mp$ = fold factor) (`operators.py:45-47`). $Re$ is based on centerline velocity and pipe radius; $\nu=1/Re$ (`solver.py:296`). No rotation, no shear, no MHD.

Base flow = **Hagen–Poiseuille**, written explicitly (`solver.py:200-203`):
$$
U(r)=1-r^2,\qquad U'(r)=-2r,\qquad b_{\rm hpf}=-U'=2r .
$$
**Womersley is not present** — there is no oscillatory/time-periodic pressure forcing, only steady constant-flux or constant-pressure-gradient driving. (This is the §3 gap-matrix "F1e skip" for B.)

Default constructor (`solver.py:110-131`): `N=64, K=18, M=32, Mp=1, alpha=0.75, Re=4000.0, dt=1e-3, implicit=0.5, KL=4, const_flux=True, nonlinearity_form="rotational", state_form="perturbation", device="cpu", dtype=complex128`. (Note: torch default `implicit=0.5` vs the Fortran reference `0.51`.)

### Radial mesh, axis parity folding, and the spectral diagonal

Mesh points are a modified Chebyshev-extrema mesh on $[0,1]$ with 10 shift iterations pushing $r[0]\to0$, then $r[-1]$ pinned to 1 (`mesh.py:60-79`, mirrors OpenPipeFlow `mes_rdom_init`/`cheby.f`). FD weights via the same Taylor/Vandermonde scheme (`mesh.py:26-57`). Stencil bandwidth $=KL$ on each side; $W_{dr1},W_{dr2}$ are $(N,N{+}KL)$ banded matrices (default $KL=4$ → 9-point, bandwidth 4). Radial Laplacian $W_{\text{radlap}}=W_{dr2}+(1/r)W_{dr1}$ (`mesh.py:165`). Integration weights `intrdr` already include the $r$ factor (`mesh.py:180-206`).

**Axis treatment is parity folding, not L'Hôpital.** Regularity at $r=0$ is enforced by reflecting through the axis: ghost points at negative radius $r_{\rm ext}[:KL]=-\text{flip}(r[:KL])$ (`mesh.py:134-140`) carry physical values with a parity sign. The parity sign (`operators.py:61-70`) is $\text{sign}=-1$ if $\bmod(m\cdot Mp+S,2)=1$ else $+1$, where $S=1$ for radial-type fields ($\partial_r$, div's $\partial_r u_r$, $\text{curl}_z$) and $S=0$ for scalar/axial ($\partial_r u_z$, gradient $p_r$). In LHS operator matrices the ghost columns are folded back onto physical columns (`_build_radlap_fold_parts`, `solver.py:245-259`); the LHS parity uses $\text{add}=|PM|$ (`solver.py:278`), matching Fortran `tim_lumesh_init`'s $S=1-2\,\bmod(m\,Mp+|PM|,2)$ (timestep.f90:84-88).

The $(k,m)$ spectral part is added as a diagonal $d$-field per PM-component (`solver.py:222-228`; `build_d_fields` `banded.py:76-94`), with $m_{\rm act}=m\,Mp$, $k\alpha=k\,\alpha$:
$$
d_0=-\frac{m_{\rm act}^2}{r^2}-(k\alpha)^2,\quad
d_+=-\frac{(m_{\rm act}{+}1)^2}{r^2}-(k\alpha)^2,\quad
d_-=-\frac{(m_{\rm act}{-}1)^2}{r^2}-(k\alpha)^2 .
$$
The $\pm$ velocity variables are $u_+=u_r+iu_\theta$, $u_-=u_r-iu_\theta$ (`operators.py:12-24`), decoupling the $(u_r,u_\theta)$ Helmholtz problems into scalar problems for $u_+,u_-,u_z$.

### Banded operator construction and LU solve

The dense build (`_build_operator_matrix`, `solver.py:266-292`) forms $A=c_2(\text{radlap\_folded}+\operatorname{diag}(d))+c_1 I$ then overwrites the **last row** (the wall) with the BC stencil $dr1[:,BC]$. Operators (`_build_linear_operators`, `solver.py:294-310`): $LDp,LDm,LDz$ (Dirichlet Helmholtz for $u_+,u_-,u_z$, $c_1=1/\Delta t$, $c_2=-\theta\nu$, with $d_+,d_-,d_0$), and $LNp$ (pressure Poisson, Neumann BC $dr1[:,1]$, $c_1=0$, $c_2=1$, $d_0$). RHS coefficients $c_1^{\rm rhs}=1/\Delta t$, $c_2^{\rm rhs}=(1-\theta)\nu$.

The **banded LU** path is the OpenPipeFlow-faithful core. LAPACK GBTRF band layout (`dense_to_banded`, `banded.py:10-28`): leading dimension
$$
\boxed{\,\text{ldab}=2\,kl+ku+1\,}
$$
main diagonal in row $kl+ku$, with $A[i,j]\to AB[kl+ku+(i-j),\,j]$. With $kl=ku=KL=4$, $\text{ldab}=13$ — exactly matching Fortran's $3\cdot i\_KL+1$ (timestep.f90:96). `build_operator_banded` (`banded.py:109-172`) fills the band directly without forming the dense matrix; `build_banded_operator_set` (`banded_operators.py:29-58`) assembles $LDp,LDm,LDz,LNp$ with $\nu=1/Re$, $c_1=1/\Delta t$, $c_2=-\theta\nu$ and parity signs `add=1` for $u_\pm$, `add=0` for $u_z/p$. Banded↔dense equivalence is tested to atol $10^{-12}$ (`test_banded.py:36-52`).

The banded LU solve (`banded_solvers.py:85-155`) **batches the factorization over all $H=Kc\cdot M$ modes** and solves all modes at once. Backends: `"dense"` (default; reconstructs dense and uses `torch.linalg.lu_factor`/`lu_solve`) and `"torch-bandedlu"`/`"bandedlu"` (a C++/CUDA extension `gbtrf`/`gbtrs`). For $LNp$ (Poisson) the extension **forces dense fallback** because it is ill-conditioned for some modes (`banded_solvers.py:118-122`). Complex RHS is stacked $[\text{real},\text{imag}]$ into $(H,N,2)$ and solved at once. The mean mode $(k{=}0,m{=}0)$ is excluded from the Poisson solve via `_mask_nonmean` (matches Fortran's `if(BC==1.and.k==0.and.m==0) cycle`, timestep.f90:99).

The `torch_bandedlu` extension implements pivoted banded LU: CPU `gbtrf_cpu`/`gbtrs_cpu` with partial pivoting within the band and float32/64 only; CUDA kernels parallelize one matrix per thread (batch) and `batch*nrhs` (solve). Autograd supports gradient **w.r.t. the RHS only** ($\text{grad}_b=\text{solve}(A^{\mathsf T},\text{grad}_x)$; no grads through the discrete pivoting). The subclass `PipeFlowSolverTorchBanded` (`solver_banded.py:10-111`) overrides only the operator build and the Helmholtz/Poisson solves; projection, influence matrix, nonlinearity, and time step are **inherited unchanged** from the dense solver. Step-level equivalence to the dense path is verified to rtol $10^{-10}$/atol $10^{-12}$ (`test_banded.py:79-100`).

### Pipe div-free, projection, influence matrix

Same OpenPipeFlow PPE-projection + influence (capacitance) matrix, in cylindrical $\pm$ variables (`operators.py:132-159`): $\operatorname{div}(u_r,u_\theta,u_z)=u_r/r+\partial_r u_r+i(m_{\rm act}/r)u_\theta+ik\alpha\,u_z$. Projection `_project_rhs` (`solver.py:731-740`, = `vel_adjPPE(1)`): compute div, zero the wall row, solve Neumann Poisson $p=LNp^{-1}\operatorname{div}$ (mean mode pinned to 0), subtract $\nabla p$. The influence matrix (`solver.py:329-445`, = `vel_adjPPE(2)`) precomputes six basis solutions $U1..U6$ and a **4×4 per-mode matrix** enforcing the four wall conditions $u_+(1)=u_-(1)=u_z(1)=0$ (no-slip) and $\partial_r u_r(1)=0$ (divergence-consistency). After each step the mean mode is cleaned ($u_r=0$, $u_\theta,u_z$ real) and `enforce_m0_reality` keeps the stored $k<0,m=0$ slots Hermitian (`spectral.py:61-81`).

### Pipe forcing and time integrator

Base-flow (HPF) coupling (perturbation form, `_apply_hpf_coupling`, `solver.py:881-909`, = `vel_addHPF`), with $U_z=1-r^2$, $-U_z'=2r$:
$$
N_r \mathrel{+}= -ik\alpha\,U_z u_r,\quad
N_\theta \mathrel{+}= -ik\alpha\,U_z u_\theta,\quad
N_z \mathrel{+}= -ik\alpha\,U_z u_z + 2r\,u_r .
$$
Constant-flux driving adds a mean-mode axial pressure correction $\propto 4/Re$ (the laminar HPF pressure gradient) and a post-step flux adjustment to zero mean axial disturbance flux (`solver.py:911-919, 785-798`). `const_flux=False` ⇒ constant pressure gradient.

The time integrator is the same family θ-method predictor/corrector (`_step_with_history`, `solver.py:1028-1114`): RHS $\text{rhs}=N+\tfrac{1}{\Delta t}b+(1-\theta)\nu[\text{radlap}(b)+d\,b]$ (`_rhs_meshmult`, `solver.py:678-693`); predictor uses old $N_0$; each corrector recomputes $N_1$ and blends $N=\theta N_1+(1-\theta)N_0$ (= `tim_nlincorr`, timestep.f90:185-195). Defaults `max_corrector_iters=3`, `tol=1e-10`; the correction metric is the absolute $L_\infty$ of the corrector delta (not normalized). A variable-$\Delta t$ controller (`run_driver.py:376-430`) uses CFL capping plus $\text{corr\_dt}=\Delta t\sqrt{d\_dterr/dterr}$. **Formal order:** first-order-in-time IMEX predictor/corrector (CN diffusion at $\theta=0.5$); no Butcher table, no AB/BDF multistep.

Pipe golden regression: against the Fortran reference, energy $=3.0132822082797048\times10^{-7}$ within rel $5\times10^{-5}$ for $N{=}16,K{=}4,M{=}4,Re{=}4000,\alpha{=}0.75,dt{=}10^{-3},\theta{=}0.5$ (`test_fortran_regression.py:15-16`).

---

## I.B.4 MHD — channel & couette (induction + Lorentz, conducting walls)

Both channel and couette carry **full-induction MHD** in primitive induced field $\mathbf b$ (NOT vector potential). The stored field is the induced $\mathbf b$; a current-free imposed background $\mathbf B_0$ is added only for EMF/Lorentz/diagnostics.

### Channel MHD (`torchchannel/mhd.py`)

`ChannelMHDSolver(ChannelSolver)`; state `{velocity, bx, by, bz}` (`mhd.py:32-53`). Constructor defaults (`mhd.py:76-86`): `Pm=1.0, Rm=None, Ha=0.0, background_b=(0,1,0), lorentz_prefactor=None, omega=0.0, shear_rate=0.0`. The default `background_b=(0,1,0)` is a uniform field along $y$ (B's wall-normal axis).

**Rm/Pm coupling** (`mhd.py:89-94`): if `Rm is None`, $Pm$ is taken and $Rm=Re\cdot Pm$; else $Rm$ is taken and $Pm=Rm/Re$. Magnetic diffusion uses diffusivity $1/Rm$ in the *same* Helmholtz template as velocity (`_build_magnetic_matrices` → `_build_diffusion_system(1.0/self.Rm)`, `mhd.py:115-117`), giving per mode
$$
L_{\rm lhs}^{B}=\tfrac1{\Delta t}I-\theta\tfrac1{Rm}L,\qquad
L_{\rm rhs}^{B}=\tfrac1{\Delta t}I+(1-\theta)\tfrac1{Rm}L .
$$

**Lorentz prefactor** (`mhd.py:100-101`, quoted verbatim):
```python
self.lorentz_prefactor = (self.Ha * self.Ha) / (self.Re * self.Rm) if prefactor is None else prefactor
```
so the canonical channel value is
$$
\boxed{\,C_L=\dfrac{Ha^2}{Re\cdot Rm}\,}\qquad(\text{unless overridden}).
$$
For oracle cross-checks against the canonical $+\mathbf J\times\mathbf B$ with prefactor 1 (§0.2), pass `lorentz_prefactor=1` explicitly.

**Induction** $\partial_t\mathbf b\big|_{\rm expl}=\operatorname{curl}(\mathbf u_{\rm tot}\times\mathbf B_{\rm tot})$ computed pseudospectrally, where $\mathbf u_{\rm tot}$ includes base flow (perturbation form) and $\mathbf B_{\rm tot}=\mathbf b+\mathbf B_0$ (`mhd.py:196-214`). **Lorentz force** $=C_L\,(\operatorname{curl}\mathbf b)\times\mathbf B_{\rm tot}=C_L\,\mathbf J\times\mathbf B_{\rm tot}$ with current $\mathbf J=\operatorname{curl}\mathbf b$ from the **induced** field only (`mhd.py:216-228`). If `lorentz_prefactor==0.0` the force short-circuits to zeros (so $Ha=0$ reproduces pure hydro and ignores NaN passive fields). Validated: zero-Lorentz step matches `ChannelSolver` step to $<10^{-10}$ (`test_mhd.py:109-132`).

**Coupled time step** (`step`, `mhd.py:353-370`): velocity step (with Lorentz added to the explicit nonlinear), then `_magnetic_step` (induction implicit-diffusion solve), with the corrector blending **both** momentum and induction explicit terms at weight $\theta$. The magnetic step (`_magnetic_step`, `mhd.py:340-351`) does: build RHS, `project_magnetic_rhs` (solenoidal projection, reuses `project_rhs`), zero wall rows, solve each component, then `enforce_magnetic_constraints`.

**Magnetic walls — conducting/homogeneous $\mathbf b=0$ ONLY (no insulating).** Wall RHS rows are zeroed ($\text{rhs}[0]=\text{rhs}[-1]=0$, `mhd.py:278-282`) and $\operatorname{div}\mathbf b=0$ is enforced via the **same** constraint pinv as velocity. The code comment (`mhd.py:304-307`) states this is intentional reuse of the no-slip/divergence machinery, with one obvious place for future boundary variants. **There is no vacuum/insulating Bessel-matching** — this is the §3 gap-matrix "insulating walls absent" for B.

### Couette MHD (`torchcouette/mhd.py`)

`TaylorCouetteMHDSolver(TaylorCouetteSolver)`; state `{velocity, br, bt, bz}` (`mhd.py:13-50`, with a `.to(device,dtype)` mover). Constructor defaults (`mhd.py:62-69`): `Pm=1.0, Ha=0.0, background_bt=1.0, background_bz=0.0, lorentz_prefactor=None`. **Crucially, the couette MHD class has NO `Rm`, NO `omega`, NO `shear_rate`, NO `q_shear`** (verified by grep) — it carries no MRI metadata at all.

**Lorentz prefactor** (`mhd.py:79-81`, quoted verbatim):
```python
self.lorentz_prefactor = (self.Ha * self.Ha) / self.Pm if lorentz_prefactor is None else validate_finite_real(...)
```
so the couette value is
$$
\boxed{\,C_L=\dfrac{Ha^2}{Pm}\,}
$$
**different from channel** because the couette momentum equation is in viscous units (viscosity $=1$, $Re$ absorbed into $Re_i,Re_o$).

**Pm / magnetic diffusion** (`_build_magnetic_operators`, `mhd.py:153-163`): $c_1=1/\Delta t$, LHS $c_2=-\theta/Pm$, RHS $(1-\theta)/Pm$, i.e. magnetic diffusivity $=1/Pm$ relative to unit-viscosity momentum (so $Pm=\nu/\eta$; larger $Pm$ → smaller magnetic diffusivity). LU-factored once.

**Background field is current-free** (`mhd.py:134-141`): toroidal $B_\theta(r)=\text{background\_bt}\cdot r_i/r$ (the $\propto1/r$ current-free profile) and axial $B_z=\text{background\_bz}$, set on the mean mode; defaults $(1.0,0.0)$.

**Induction** $=\operatorname{curl}(\mathbf u\times\mathbf B_{\rm tot})$ (`mhd.py:316-337`); **Lorentz** $=C_L\,\mathbf J\times\mathbf B_{\rm tot}$, $\mathbf J=\operatorname{curl}(\text{induced }\mathbf b)$ (`mhd.py:339-370`). **Magnetic walls** are homogeneous $\mathbf b=0$ enforced via a dedicated **magnetic influence/capacitance matrix** (`_build_magnetic_influence_matrix`, `mhd.py:179-231`, per-mode real 8×8 over non-mean modes) plus the velocity Poisson projection — the docstring (`mhd.py:53-60`) explicitly states "Exact insulating Bessel matching is a boundary-model upgrade behind the same interface" → **insulating walls NOT implemented**. The coupled step (`step_with_info`, `mhd.py:409-439`) is the adaptive predictor/corrector (`max_corrector_iters=3`, `tol=1e-10`, returns `StepInfo`).

### MHD diagnostics

Channel `diagnostics()` (`mhd.py:441-453`) returns `{Epert, Emag, Emag_total, divLinf, divB_Linf, divB_L2, reynolds_xy, maxwell_xy, transport_xy, alpha, q_shear}`. Reynolds stress $=\langle u_x u_y\rangle$; Maxwell stress $=-\langle B_x B_y\rangle$ (explicit minus, `mhd.py:412`), on the **induced** field unless `total=True`; $\alpha=\text{transport}/\sum B_0^2$ (NaN if denom 0). Couette `diagnostics()` (`mhd.py:480-488`) returns exactly `{Emag, Emag_total, divB_Linf, reynolds_rt, maxwell_rt, transport_rt}` with $\text{transport\_rt}=\text{reynolds\_rt}+C_L\cdot\text{maxwell\_rt}$ (note the Lorentz-prefactor weighting, `mhd.py:476-478`).

---

## I.B.5 MRI — metadata-only STUB (documented gap; quoted)

**Family B has no working MRI.** The channel MHD solver stores `omega`/`shear_rate` but adds **no Coriolis, no base-shear, no shear-induction source terms** anywhere in the timestepper. This is the §3/§4 headline gap and the acceptance gate for Part IV/SR-1, SR-2.

The class docstring states it verbatim (`mhd.py:71-74`):
```python
``omega`` and ``shear_rate`` are diagnostic metadata only in this class.
They set ``q_shear`` but do not add Coriolis or background-shear source
terms to the timestepper.
```
The constructor merely records them and forms a diagnostic ratio (`mhd.py:102-104`):
```python
self.omega = validate_finite_float("omega", omega)
self.shear_rate = validate_finite_float("shear_rate", shear_rate)
self.q_shear = math.inf if self.omega == 0.0 else self.shear_rate / self.omega
```
`self.omega` and `self.shear_rate` are referenced **nowhere** in any RHS/timestepping path; `q_shear` is emitted only in `diagnostics()` (`mhd.py:452`). There is **no Coriolis term $2\boldsymbol\Omega\times\mathbf u$, no background-shear coupling, no shear-induction $\partial_t B_y=-S\,B_x$**. The couette MHD class carries no MRI metadata at all.

The *only* shear-like coupling present is the **hydrodynamic** base-flow advection (`_base_coupling_terms`, `solver.py:534-540`): for `state_form="perturbation"`, $b_u=-U\partial_x u-U'v$, $b_v=-U\partial_x v$, $b_w=-U\partial_x w$ with base flow $U=y$ (Couette) or $U=1-y^2$ (Poiseuille). This is plain advection by the laminar profile, **not** a rotation source, and it is **not applied to the induction equation** (the EMF uses $\mathbf u_{\rm tot}$ which includes base flow, but there is no separate shear-induction term).

A test confirms the metadata-only behavior: with `omega=1.0, shear_rate=1.0`, diagnostics return $q_{\rm shear}=1.0$ but the step is otherwise an ordinary MHD step (`test_mhd.py:135-150`). **Planned wiring** (Part III.1, Phase 1): add Coriolis + base-shear as a sibling of `_base_coupling_terms` (consumed in `_rhs_for_state`, `solver.py:571-577`), promote `omega`/`shear_rate` to active params (default OFF), and add shear-induction $\partial_t B_y\mathrel{+}=-S\,B_x$ to `_magnetic_step`. Every MRI claim about B in this survey is flagged **"stub."**

---

## I.B.6 Compute backend, precision, autograd

- **Backend:** pure PyTorch (FFT + dense/banded linear solves), CPU default everywhere, **device-agnostic** (`device` param). CUDA is supported by construction (all ops are tensor ops with explicit `device=`); it is only special-cased in the pipe benchmark harness (`torchpipeflow/benchmarks/benchmark_hotspots.py`, `--device cuda`, `torch.cuda.synchronize`). The optional `torch_bandedlu` extension adds C++ (CPU/OpenMP) and CUDA banded-LU kernels for the pipe. **No `torch.compile`, `torch.jit`, `vmap`, or `functorch`** anywhere (§3 gap-matrix "JIT absent"); mode batching is explicit $(H,N,N)$/$(N,H)$ tensor ops and `einsum`.
- **Precision:** **`complex128` default** (real `float64`) in all three solvers; `complex64`/`float32` supported and validated (channel `test_float32.py`: roundtrip error $<5\times10^{-5}$; solver step preserves `complex64`, div $<10^{-4}$, BC residual $<10^{-4}$). Real linear solves split complex RHS into real/imag stacks solved with real LU/solve.
- **Autograd:** the solvers are **fully differentiable** end-to-end through projection, BC correction, FFTs, and the MHD Lorentz coupling — no `.detach()` in step paths (`torch.no_grad()` only in IO). The banded `lu_solve` is autograd-aware w.r.t. RHS only. Tests assert gradients flow through velocity and magnetic fields and through the **magnetic→velocity Lorentz coupling** (channel `test_mhd.py:183-201`, magnetic→velocity grad $>0$; couette `test_mhd.py:105-130`, `br.grad` max $>0$ with `lorentz_prefactor=2.0`).

### Golden numbers and benchmarks (verbatim)

| Quantity | Value | Config / source |
|---|---|---|
| Poiseuille OS leading $c$ (ref) | $0.23752649+0.00373967\,i$ | $Re{=}10000,\alpha{=}1,\beta{=}0,N{=}101,KL{=}4$; `tests/test_linstab_poiseuille.py:7-19`, asserts $|c-c_{\rm ref}|<10^{-4}$ |
| Poiseuille OS leading $c$ (computed) | $0.23752722198590992+0.0037381198835812705\,i$ | `VALIDATION.md:83-96` |
| Critical-$Re$ sign change | stable $Re{=}5742.22$, unstable $Re{=}5802.22$ | $\alpha{=}1.02,N{=}96$; `test_linstab_poiseuille.py:22-39` |
| Laminar Couette 2000-step decay | $E_{\rm pert}<10^{-20}$, div $<10^{-12}$, $\max|u|<10^{-14}$ | $N{=}9,Re{=}500,dt{=}0.01$; `test_step_decay.py:29-51` |
| Mesh polynomial exactness | err $<10^{-7}$, deg 0..8; $\int1{=}2,\int y{=}0,\int y^2{=}2/3$ | `test_mesh.py:21-48` |
| Channel MHD div-free | div, divB$<10^{-7}$; $Rm{=}100,Pm{=}5\Rightarrow Pm{=}100/Re$ | `test_mhd.py:54-180` |
| Couette nsCouette reference | $m_r{=}32,m_\theta{=}16,m_{z0}{=}16,k_{\theta0}{=}6.0,k_{z0}{=}2.6179938779914944,\eta{=}0.868,Re_i{=}200,Re_o{=}-200$ | `test_fortran_reference_config.py` |
| Couette laminar 1000-step | div $<10^{-8}$, $E_{\rm pert}<10^{-12}$, $Nu_i,Nu_o\approx1$ (atol $10^{-8}$) | `test_integration_laminar.py` |
| Couette MHD | div $<10^{-8}$, divB$_{L\infty}<10^{-8}$, walls $\mathbf b{=}0$ (atol $10^{-12}$) | `test_mhd.py` ($N{=}8,\eta{=}0.868,Re_i{=}20,Re_o{=}-20,Pm{=}2,Ha{=}1$) |
| Pipe Fortran regression energy | $3.0132822082797048\times10^{-7}$ (rel $<5\times10^{-5}$) | `test_fortran_regression.py:15-16` |
| Pipe banded↔dense step | rtol $10^{-10}$/atol $10^{-12}$ | `test_banded.py:79-100` |

Cross-family OS comparisons use the published $Re_{\rm crit}=5772.22$ [Orszag71] at rel $<10^{-2}$ (the within-family golden $c$ above is family-specific and not portable; see the §0 hand-off note). For all MRI/rotation acceptance (SR-1, SR-2), B is the **gate, not a pass**, until the §I.B.5 wiring lands.

### Reimplementation-critical facts (the load-bearing summary)

1. $W_{dy2}=W_{dy1}^2$ (channel) and pressure radial $=W_{dr1}^2+W_{dr1}/r$ (couette/pipe) — required for exact $\operatorname{div}\operatorname{grad}=\nabla^2$ in the projection.
2. FFT normalization differs: **channel = plain norm** (constant → coefficient $\times Z\,Th$); **couette/pipe = `norm="forward"`**.
3. Mode storage differs: channel full-symmetric $(-K1..K1)\times(-M1..M1)$; couette/pipe nsCouette layout.
4. Influence matrix is **real 8×8 per mode** (channel via phase transform; couette via 12-column basis); pipe is **real 4×4**; mean mode masked/identity in all.
5. Corrector differs: **channel fixed** (`corrector_iterations`, default 1); **couette/pipe iterated** to `tol=1e-10` (`max_corrector_iters=3`) with `StepInfo`.
6. **MRI is metadata-only** — no Coriolis, no shear-induction (§I.B.5).
7. `implicit=0.51` default (channel/couette; pipe `0.5`) → formally first-order θ-method.
8. **Channel $C_L=Ha^2/(Re\,Rm)$; couette $C_L=Ha^2/Pm$** — different because couette momentum is in viscous units.
9. Magnetic walls are **homogeneous $\mathbf b=0$ (conducting-style) only**; no insulating walls in either family.
10. **Pipe has no MHD and no Womersley** — a from-scratch pipe must not add either.


\newpage

# Part I.C — jax Galerkin family (`jaxfun`): full algorithmic spec

**Family C** is a JAX port of Mortensen's `shenfun` spectral-Galerkin platform [Mortensen18], living at
`/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/`. It pairs a **SymPy form language** (weak-form
description: `TestFunction`, `TrialFunction`, `Grad`, `Div`, `Curl`, curvilinear metric `√g`) with a **JAX compute
backend** (`jnp`, `jax.scipy.fft`, `jax.jit`, `jax.grad`, `shard_map`). The reusable engine is `src/jaxfun/`; the
wall-bounded shear-flow **solvers** are scripts under `examples/` that import that engine. A read-only NumPy/shenfun
reference (the "ground truth" these scripts port) lives in the sibling `couette/` directory.

Family C is the **second spectral oracle** alongside Family A (§I.A): identical Galerkin discretization, identical
IMEX-RK tableaux, identical Lorentz prefactor (1, Alfvén units), the **same canonical axis convention** as A (axis
0 = wall-normal `x`, axis 1 = streamwise `y`, axis 2 = spanwise/axial `z`; §0.2 conventions table, `channelflow_kmm.py:58-59`)
— so cross-family comparison against A requires no axis remap, unlike Family B. C adds three things A lacks
(§3 gap matrix): **autograd** (`jax.value_and_grad` for minimal-seed/adjoint), **JIT** (`jax.jit`/`nnx.jit`),
and **multi-device sharding** (CPU/GPU/TPU via `shard_map`). Two things C lacks relative to A: a **pipe** geometry
(absent entirely; §I.C.6) and JAX-native **linear** solvers (the `*_linear_jax.py` files assemble with `jaxfun`
but solve on host NumPy/SciPy; §I.C.2).

This section gives the full algorithmic spec: §I.C.1 the framework (bases, inner products, operators, curvilinear
coordinates); §I.C.2 the solvers (plane Couette, Taylor–Couette, channel KMM, MHD, MRI); §I.C.3 the time
integrators; §I.C.4 the JAX-specific compute capabilities; §I.C.5 golden numbers; §I.C.6 the explicit pipe gap and
`jaxfun_missing_parts.md`.

---

## I.C.1 The `jaxfun` framework

### I.C.1.1 Bootstrap, precision, devices

`src/jaxfun/__init__.py` configures JAX at import:

- `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")` (`__init__.py:4`) — disables XLA GPU
  preallocation before JAX is imported.
- **`jax.config.update("jax_enable_x64", True)`** (`__init__.py:8`) — **float64 is enabled globally and
  unconditionally**. Verified by `tests/test_x64_default.py:7-10`, which asserts
  `jax.config.read("jax_enable_x64") is True` and `jnp.zeros(1).dtype == jnp.float64`. Fourier coefficients are
  therefore `complex128`; the entire spectral path runs in double precision by default.

**Devices:** CPU, GPU (CUDA), TPU. Deps `jax>=0.10`, optional extra `cuda13 = ["jax[cuda13]>=0.10.1"]`
(`pyproject.toml:19-37`). There is **no TPU-specific code path** — portability comes from the device-agnostic
FFT layer and the `shard_map` transform (§I.C.4). Eigensolves and Gauss–Jacobi quadrature escape to host
SciPy/NumPy (see §I.C.1.5).

### I.C.1.2 Basis-space class hierarchy

`BaseSpace` (`basespace.py:15-39`) is the abstract root: it attaches a `CoordSys` (defaulting to a 1-D Cartesian
system `CartCoordSys("N",(x,))` if none is given, `basespace.py:39`), carries `name`, `fun_str` (symbol stem,
default `"phi"`), and a class flag `is_transient = False`. Concrete bases:

```
BaseSpace (basespace.py)
└─ OrthogonalSpace (galerkin/orthogonal.py)        is_orthogonal=True
   ├─ Fourier (galerkin/Fourier.py)
   └─ Jacobi  (galerkin/Jacobi.py)                 (alpha, beta)
      ├─ Chebyshev    (alpha=beta=-1/2)
      ├─ ChebyshevU   (alpha=beta=+1/2)
      ├─ Legendre     (alpha=beta=0)
      └─ Ultraspherical (alpha=beta=lambda-1/2)
   └─ Composite (galerkin/composite.py)            is_orthogonal=False  (Shen / BC bases)
      ├─ CGComposite (Chebyshev), LGComposite (Legendre)
      ├─ PGComposite (Petrov–Galerkin test side) → ChebPhi_1/2/4
      └─ BCGeneric (boundary-lift basis, num_dofs == 0)
```

Container spaces (`galerkin/tensorproductspace.py`): `TensorProductSpace`, `DirectSumTPS`,
`VectorTensorProductSpace`, `CoupledSpace`, plus the 1-D `DirectSum` (`Composite ⊕ BCGeneric`,
`composite.py:475`).

### I.C.1.3 Basis families: quadrature, mode counts, transforms

**Orthogonal common interface** (`galerkin/orthogonal.py:57-545`): modes `N`; quadrature points
`_num_quad_points = N` (`:69`); dofs `_num_dofs = N` by default, constrained `0 < num_dofs <= N` (`:70-72`).
`forward(u) = scalar_product(u) / (norm_squared / domain_factor)` by orthogonality (`:265`);
`scalar_product` injects the curvilinear weight `sg = system.sg / domain_factor` (`:278`). The affine map factor
`domain_factor = (d-c)/(b-a)` (reference length / true length, `:361`) handles non-`[-1,1]`/non-`[0,2π]` domains.
**Dealiasing knob:** `get_dealiased(padding_factor=1.5)` (`:315`) deep-copies the space and sets
`_num_quad_points = floor(padding_factor · nq)` while leaving the modal dimension fixed — the padded-physical-length
realization of the **3/2 rule**.

| Family | $\alpha,\beta$ | nodes / weights | fast transform | operator matrices |
|---|---|---|---|---|
| **Chebyshev** ($T_k$) | $-1/2$ | $x_k=\cos(\pi+(2k+1)\pi/2N)$, $w_k=\pi/N$ (`Chebyshev.py:160-180`) | DCT/IDCT (`jax.scipy.fft.dct/idct`, `:236-273`) | **DENSE** upper-triangular (`:409-459`) |
| **Legendre** ($P_k$) | $0$ | Gauss–Legendre (`utils/fastgl.py`) | none (matrix) | dense deriv coupling (`Legendre.py:20-33`) |
| **ChebyshevU** ($U_k$) | $+1/2$ | Gauss–Cheb-2 | DST (`utils/common.py:174-207`) | — |
| **Ultraspherical** $C^{(\lambda)}_k$ | $\lambda-1/2$ | Gauss–Jacobi | — | — |
| **Fourier** ($e^{ikx}$) | — | $x_j=2\pi j/N$, $w=2\pi/N$ (`Fourier.py:80-91`) | FFT (`jnp.fft.fft/ifft`, `:129-184`) | **DIAGONAL** (`:241-268`) |

**Fourier layout** (`galerkin/Fourier.py`): complex-exponential basis $E_k(x)=e^{ikx}$ on default $[0,2\pi]$;
**`N` must be even** (`:51`). Wavenumbers use NumPy `fftfreq` ordering
$k=\text{where}(\text{idx}<\lceil(N+1)/2\rceil,\ \text{idx},\ \text{idx}-N)$ (`:16-22`); the Nyquist mode $k[N/2]$ is
zeroed for odd derivatives (`eliminate_highest_freq`). Transforms use `norm="forward"`. $\|E_k\|^2=2\pi$ per mode
(`:206-208`); `derivative_coeffs` $=(ik)^m\,c$ (`:210-223`). Crucially, jaxfun uses a **full-complex FFT on every
periodic axis** (not shenfun's real half-spectrum), with the Nyquist filtered explicitly via `mask_nyquist`; this
works identically on CPU/GPU/TPU without an rfft-specific branch (`docs/couette_fourier_layout.md:11`), and the
`(0,0)` mode is the global Fourier coefficient.

**Jacobi recurrence engine** (`galerkin/Jacobi.py:66-579`): Gauss–Jacobi nodes via host
`scipy.special.roots_jacobi(N, alpha, beta)` (`:124`); three-term recurrence built symbolically (SymPy `a(i,j)`,
`b(i,j)`), lambdified to JAX, summed via `lax.scan`. Tridiagonal recurrence matrices `A` ($xQ=A^\top Q$),
`B` ($\partial Q=B^\top Q$), `A_(N,k)` (k-th deriv) returned as `DiaMatrix`. Boundary traces `bnd_values(k)` at
$\pm1$ are the algebraic basis for BC/stencil derivation.

### I.C.1.4 Composite (Shen) bases — the BC mechanism

BCs are **baked into the trial basis** (no tau rows, no influence matrix). `BoundaryConditions`
(`composite.py:25-100`) accepts codes `D` (Dirichlet), `N`/`N2`/`N3`/`N4` (k-th Neumann), `R` (Robin, tuple
`(alpha, value)`), `W` (weighted). Shenfun-style tuples are accepted by the `FunctionSpace` factory: `(a,b)` →
Dirichlet left/right, `(a,b,c,d)` → clamped (Dirichlet + Neumann) each side (`functionspace.py:42-58`).

A `Composite` builds a constrained basis $\phi_i = \sum_j S_{ij} P_j$ where $P$ are the underlying Jacobi
polynomials and $S$ (a `DiaMatrix` stencil) encodes the BCs. `get_stencil_matrix` (`composite.py:725-792`)
**symbolically solves** the boundary-trace relations to express $\psi_i = T_i + \sum_k d_k T_{i+k}$ (e.g. the
Neumann Chebyshev result $\psi_i = T_i - \tfrac{i^2}{i^2+4i+4}\,T_{i+2}$; the `LDRD` Dirichlet case is the
special-cased $\{0:1,\,2:-1\}$). The basis dimension drops by the number of BCs:
`dim = orthogonal.dim − stencil_width()` (`:324-327`); the mass matrix is $S\,P_{\text{mass}}\,S^\top$,
LU-factored eagerly. Inhomogeneous wall values (moving Couette walls, imposed flux) enter through a
`DirectSum(C, B)` = homogeneous `Composite` $\oplus$ `BCGeneric` boundary-lift, whose `to_orthogonal` adds the lift
$S\cdot\text{bnd\_vals}$ (`functionspace.py:193-203`, `composite.py:550-556`).

The KMM solver uses exactly two homogeneous wall-normal bases: `bc=(0,0,0,0)` → **clamped biharmonic**
($u=u'=0$ at both walls) and `bc=(0,0)` → **Dirichlet** ($u=0$).

### I.C.1.5 Inner products, tensor products, linear algebra

`inner(expr, sparse, num_quad_points, kind)` (`galerkin/inner.py:126-193`) is the weak-form assembler: it finds
test/trial spaces, multiplies by the metric `system.sg`, splits the SymPy expression into bilinear/linear forms,
and assembles `Matrix`/`DiaMatrix` (1-D), `TPMatrix` (separable d-D), or `TensorMatrix` (non-separable dense). It
uses precomputed sparse operator matrices `matrices(i,(u,j),...)` when the coefficient is scalar, else quadrature.
The companion `integrate(u, V)` (`inner.py:213-253`) is the pure integral $\int u\cdot w\cdot\sqrt g$ — the
backbone of energy/divergence diagnostics. The 3/2-rule is the documented call
`inner(..., num_quad_points=tuple(int(1.5·n)…))` (`inner.py:166`).

`TensorProductSpace` (`tensorproductspace.py:98-633`) realizes transforms as **separable per-axis vmaps**, fused/
JIT-compiled and cached per `(op, N)`. Fourier helpers used by the KMM reconstruction: `wavenumbers(scaled=True)`
returns per-axis $k$ grids ($\times\,2\pi/L$, zero on non-Fourier axes); `K_over_K2(K, axes)` returns
$K_i/|K|^2$ with a zero-mode guard (`:76-95`); `mask_nyquist` zeros the Fourier Nyquist modes.
`VectorTensorProductSpace` (rank-1 vector fields) and `CoupledSpace` (mixed systems, e.g. $(u_r,u_\theta,u_z,p)$
with `flatten`/`unflatten` over `block_slices`) handle the TC saddle systems.

Linear-algebra layer `src/jaxfun/la/`: `DiaMatrix` (`la/diamatrix.py`) scipy-compatible diagonal sparse
(cached banded LU, RCM reordering); `TPMatrix`/`TPMatrices` separable Kronecker solvers including a wavenumber-diagonal
Fourier solver and an `eigh`-based "matrix diagonalization" channel solver; `la/solvers.py` exports the
**`Helmholtz`** and **`Biharmonic`** wrappers the KMM solver depends on; `la/eig.py` provides
`generalized_eig(L,M)` via host `scipy.linalg.eig` with finite-eigenvalue caps `MODAL_FINITE_CAP = 1e6`,
`NONMODAL_FINITE_CAP = 1e8` (`la/eig.py:8-9`) and `transient_growth_from_eigs` via `svdvals`.

### I.C.1.6 Curvilinear coordinates and operators

`coordinates.py:425-1227` is a full Riemannian layer. `get_CoordSys(name, Lambda)` wraps a SymPy
`Lambda((q…),(x,y,z…))` mapping computational → Cartesian (polar example:
`Lambda((r,θ),(r·cosθ, r·sinθ))`). It lazily caches the covariant basis $b_i=\partial r/\partial q^i$, contravariant
basis, covariant/contravariant metric tensors $g_{ij}=b\cdot b^\top$, $g^{ij}=(g_{ij})^{-1}$, the determinant, and
**`sg = √det(g)`** — the volume element injected into every `scalar_product`/`integrate`. Christoffel symbols
$\Gamma^k_{ij}$ and covariant differentiation are provided. `operators.py:118-1097` implements curvilinear
`gradient`/`divergence`/`curl`/`cross`/`dot`/`outer` on the form language; the **Laplacian is composed as
`Div(Grad(u))`** (no standalone class). Module-level monkeypatches make `sympy.vector`'s
`dot/cross/gradient/curl/divergence` use these curvilinear semantics throughout assembly.

> **Note on the TC solvers.** Despite this curvilinear machinery existing, the Taylor–Couette *solvers* (§I.C.2.3)
> deliberately use plain Cartesian tensor-product spaces with **explicit `1/r` factors written into the weak forms**,
> mirroring the shenfun reference — not the curvilinear metric. The metric layer is exercised by the curvilinear
> Poisson demos instead (`examples/poisson2D_curv.py`).

---

## I.C.2 The solvers (`examples/`)

### I.C.2.0 What is JAX-native vs host NumPy

| Geometry / physics | JAX file | Status |
|---|---|---|
| PCF linear (modal + non-modal) | `pcf_linear_jax.py` | host NumPy/SciPy dense (caveat below) |
| PCF nonlinear DNS (hydro) | `pcf_fluctuations_jax.py` + `channelflow_kmm.py` | **JAX-native, differentiable** |
| PCF MHD (vector potential) | `pcf_mhd_jax.py` | JAX-native |
| PCF MHD + rotation/shear (MRI) | `pcf_mhd_mri_shearpy_jax.py` | JAX-native |
| PCF minimal-seed / adjoint | `pcf_minimal_seed_jax.py` | JAX-native (`value_and_grad`) |
| TC linear (hydro eig) | `taylor_couette_linear_jax.py` | jaxfun assembly → host dense GEP |
| TC MRI linear (conducting + insulating) | `taylor_couette_mri_jax.py` | jaxfun assembly → host dense GEP |
| TC DNS (axisym + 3-D hydro) | `taylor_couette_dns_jax.py` | **JAX-native** |
| TC MHD/MRI DNS (axisym + 3-D) | `taylor_couette_dns_jax.py` | **JAX-native** |
| Channel KMM (shared base class) | `channelflow_kmm.py` | **JAX-native** |
| **Pipe flow** | — | **ABSENT** (§I.C.6) |

**Linear-solver caveat.** Despite the `_jax.py` suffix, `pcf_linear_jax.py:14-16` imports only `numpy`; its docstring
(`:8-12`) states it is *"a NumPy/SciPy dense reference workflow … not a differentiable JAX Galerkin port."* The TC
linear/MRI scripts use `jaxfun.galerkin.inner(...).todense()` then immediately cast to
`np.asarray(..., dtype=complex)` (e.g. `taylor_couette_linear_jax.py:152`) and call `generalized_eig`/`scipy`. So the
linear solvers use jaxfun **assembly** but produce **dense NumPy operators**; only the **DNS** solvers are
JAX-native and autodiff-capable.

### I.C.2.1 Channel KMM solver (`channelflow_kmm.py`) — the core engine

Kim–Moin–Moser velocity–vorticity formulation [KMM87], a port of `couette/ChannelFlow.py:44-251`
(`channelflow_kmm.py:56-59`). It evolves the wall-normal velocity $u_0$ on a clamped biharmonic basis and the
wall-normal vorticity $g$ on a Dirichlet basis, then reconstructs $u_1,u_2$ from incompressibility. The
`KMMState` pytree carries `u=(u0,u1,u2)` and `g` (`channelflow_kmm.py:38-52`).

**Coordinate convention (exact):** component **0 = wall-normal $x$** (Chebyshev/Legendre),
**1 = streamwise $y$** (Fourier), **2 = spanwise $z$** (Fourier) (`channelflow_kmm.py:58-59`). Default domain
$((-1,1),(0,4\pi),(0,2\pi))$ (`channelflow_kmm.py:65-69`). This is the **canonical frame** of §0.2 — no adapter needed against A.

**Wall-normal vorticity definition** (`channelflow_kmm.py:352`):
$$g = i K_y\,\hat u_2 - i K_z\,\hat u_1.$$

**Spaces** (`channelflow_kmm.py:88-118`): `B0 = FunctionSpace(N0, family, bc=(0,0,0,0))` (clamped biharmonic);
`D0 = bc=(0,0)` (Dirichlet); `C0` unconstrained; `F1,F2` Fourier. Tensor products `TB,TD,TC`; dealiased
`TBp,TDp = get_dealiased(padding_factor)` with default `padding_factor=(1.0,1.5,1.5)` → **3/2-rule dealiasing in
the two Fourier directions only** (`channelflow_kmm.py:73`). 1-D radial spaces `D00,C00` handle the **(0,0) Fourier mean modes** of
$u_1,u_2$ (`channelflow_kmm.py:113-118`).

**Implicit operators** (`_build_operators`, assembled via `inner`):
- $M_u=\langle v_b,\Delta u_b\rangle$, $L_u=\langle v_b,\nu\Delta^2 u_b\rangle$ (biharmonic, wall-normal momentum);
- $M_g=\langle v_h,h_b\rangle$, $L_g=\langle v_h,\nu\Delta h_b\rangle$ (Helmholtz, vorticity);
- $S_u=\texttt{Biharmonic}(\text{coeff}=\gamma\,dt,\ \text{diff}=\nu)$,
  $S_g=\texttt{Helmholtz}(\text{coeff}=\gamma\,dt,\ \text{diff}=\nu)$ — i.e. $M-\gamma\,dt\,L$ (`la/solvers.py`).
  For `IMEXRK3` there is one operator per stage with $\gamma=(a_{rk}+b_{rk})\,dt/2$.
- $M_{00},L_{00},S_{00}$ for the (0,0)-mode Helmholtz solve, plus an optional mean pressure gradient `dpdy_rhs`.

**Nonlinear / convection** (`convection`, `channelflow_kmm.py:382-396`): **gradient (advective) form** ($conv=0$) $n_i = u\cdot\nabla u_i$,
evaluated on the **padded (dealiased) mesh**, forward-transformed through `TDp`, then `mask_nyquist`. The KMM RHS
combinations (`_nonlinear_rhs`, `channelflow_kmm.py:398-413`) are
$$H_u = \partial^2_{xy}H_1 + \partial^2_{xz}H_2 - \partial^2_{yy}H_0 - \partial^2_{zz}H_0,\qquad
  H_g = \partial_z H_1 - \partial_y H_2,$$
with separate (0,0)-mode terms $N_{v00}=-(M_{00}\,\mathrm{Re}\,H_1[:,0,0])+\texttt{dpdy\_rhs}$,
$N_{w00}=-(M_{00}\,\mathrm{Re}\,H_2[:,0,0])$ (`channelflow_kmm.py:409-413`).

**Velocity reconstruction** (`_reconstruct_velocity`, `channelflow_kmm.py:415-428`): with $f=\mathcal F(\partial_x u_0)$,
$$u_1 = i\,(K_{/K^2}[0]\,f + K_{/K^2}[1]\,g),\qquad
  u_2 = i\,(K_{/K^2}[1]\,f - K_{/K^2}[0]\,g),$$
and the (0,0) modes overwritten with the real mean solves $v_{00},w_{00}$ (`channelflow_kmm.py:422-423`). `K_over_K2` is defined in
`tensorproductspace.py:76`.

**Pressure recovery** (`compute_pressure_coefficients`, `channelflow_kmm.py:238-337`, optional): Poisson $\Delta p = -\nabla\cdot H$
on a Neumann pressure-test space `bc={"left":{"N":0},"right":{"N":0}}` with wall data
$\nu\,\partial^2_x u_{\text{wall-normal}}$ and the (0,0) mode pinned (`matrices.at[0,0,0,0].set(1)`,
`rhs.at[0,0,0].set(0)`).

**Divergence diagnostic** (`divergence_l2`, `:550-556`): $\partial_x u_0+\partial_y u_1+\partial_z u_2$
integrated in `TC`.

### I.C.2.2 Plane Couette hydro (`pcf_fluctuations_jax.py`)

A subclass of `KMM`. Only the **fluctuation** $u'$ is in the state vector; the base flow is never stored
(verified `pcf_fluctuations_jax.py:29-122`).

- **Nondimensionalization:** $\nu = U_{\text{wall}}/Re$ (`:54`). Defaults $Re=600$, $U_{\text{wall}}=1$, $dt=0.01$,
  `family="L"` (Legendre).
- **Base flow (canonical PCF sign):**
  $$U_b(x) = U_{\text{wall}}\,x\;e_y,\qquad \frac{dU_b}{dx}=U_{\text{wall}}\quad(\texttt{:65\text{-}67}).$$
- **Base-flow coupling** (`_add_base_convection`, `:91-98`) — exactly $(U_b\partial_y)u' + (u'\cdot\nabla)U_b$:
  $$n_0 \mathrel{+}= U_{bp}\,\partial_y u_0,\quad
    n_1 \mathrel{+}= U_{bp}\,\partial_y u_1 + u_0\,\frac{dU_b}{dx},\quad
    n_2 \mathrel{+}= U_{bp}\,\partial_y u_2.$$
- **IC** (deterministic, `:69-89`): $\text{wall}=1-x^2$; amplitude-0.05 sinusoids in $u'_0,u'_1,u'_2$ over
  $L_y=4\pi$, $L_z=2\pi$.
- **Diagnostics** (`:104-122`): `Epert`, `Etot`, `divL2`, `u_top`, `u_bot`,
  `mean_shear = mean(∂v/∂x + dU_b/dx)`.

### I.C.2.3 PCF MHD (`pcf_mhd_jax.py`)

Subclass of `PlaneCouetteFluctuationJax`; port of `couette/pcf_mhd_divfree.py`. Identical to Family A's vector-
potential approach.

- **Magnetic unknown = vector potential $A$** in $TD^3$ (Dirichlet all components). Then
  $\mathbf B=\nabla\times\mathbf A$ is **divergence-free by construction** (`update_B_from_A`, in
  `[TD,TC,TC]`); $\mathbf J=\nabla\times\mathbf B$ (`update_J_from_B`, in `[TC,TD,TD]`).
- **Resistive normalization:** $Rm=Re$ if unset; $\eta = U_{\text{wall}}/Rm$ (`:63-64`).
- **$A$-evolution (implicit):** $M_A=\langle h,a\rangle$, $L_A=\langle h,\eta\Delta a\rangle$,
  $S_A=M_A-\gamma\,dt\,L_A$ (Helmholtz, resistive diffusion).
- **Lorentz force (explicit):** $\text{lorentz}=\mathbf J\times\mathbf B$ (physical cross), $n=n-\text{lorentz}$
  (`:175-176`). The Lorentz **prefactor is 1** (Alfvén units, $\rho=\mu_0=1$) — matching the §0.2 canonical
  convention and A; no $Ha^2/(Re\,Rm)$ scaling.
- **Induction (EMF, explicit):** $\text{emf}=\mathbf U_{\text{tot}}\times\mathbf B$ with
  $\mathbf U_{\text{tot}}=(u_0,u_1+U_{bp},u_2)$; drives $dA/dt = \mathbf U\times\mathbf B + \eta\Delta A$.
- **Magnetic BC:** $A\in TD$ (A = 0 at walls); `magnetic_divergence_l2` confirms $\nabla\cdot\mathbf B\approx0$.

### I.C.2.4 PCF MRI / shearpy (`pcf_mhd_mri_shearpy_jax.py`)

Subclass of `PlaneCouetteMHDJax`; port of `couette/pcf_mhd_mri_shearpy.py`. This is a **Cartesian PCF analogue of
the shearing box** [BH91; Lesur15], not a rotating annulus. All formulas below are verified from source.

- **Base flow (MRI/shearpy sign — opposite to plain Couette):**
  $$U_b(x) = -S\,x\;e_y,\qquad \frac{dU_b}{dx}=-S\quad(\texttt{:70\text{-}72}),$$
  with `shear_rate = S`.
- **Rotation/shear parameters** (`:73-74`): $\Omega=$`omega`, $q=S/\Omega$ (`q_shear`), epicyclic
  $$\kappa^2 = 2\Omega(2\Omega - S).$$
- **Coriolis (explicit)** — i.e. $-2\boldsymbol\Omega\times\mathbf u$ with $\boldsymbol\Omega=\Omega\,e_z$
  (`_mhd_convection`, `:130-134`):
  $$n_0 \mathrel{-}= 2\Omega\,u_1,\qquad n_1 \mathrel{+}= 2\Omega\,u_0,\qquad n_2\ \text{unchanged}.$$
  (The base-flow shear $u_0\,dU_b/dx = -S\,u_0$ enters separately through `_add_base_convection`, so the streamwise
  equation carries the canonical $(S-2\Omega)u_0$ combination.)
- **Imposed net field:** `background_b = (0,0,B_z)`, default $B_z=0.1$. The **total** physical field
  $\mathbf B_{\text{tot}}=\nabla\times\mathbf A+\text{background\_b}$ (`:116-118`) is used in both Lorentz and EMF
  (`:138-146`), so imposed $B_z$ shear-couples $b_x\to b_y$ through the shear-induction term $\mathbf U_b\times\mathbf B$.
- **IC seed** (`:80-114`): channel-mode harmonics $\cos/\sin(\text{harmonic}\cdot k_z z)$ (harmonics 1,2,3) plus
  cross-stream perturbations; an $A_x$ seed for the magnetic field.
- **Diagnostics** (`:149-182`): Reynolds stress $\langle u_r u_\theta\rangle$, Maxwell stress
  $-\langle b_r b_\theta\rangle$, transport $\alpha=(R+M)/v_A^2$ with $v_A^2=B_z^2$, plus `q_shear`, `kappa2`,
  `Emag_total`.

### I.C.2.5 Taylor–Couette DNS (`taylor_couette_dns_jax.py`)

**Formulation (deliberately shenfun-faithful):** Cartesian tensor-product spaces with **explicit cylindrical
`1/r` factors** in the weak forms (no curvilinear metric); Dirichlet radial velocity + truncated orthogonal
pressure ($P_N/P_{N-2}$). Component order **$(u_r,u_\theta,u_z)$**; radial $r$ (Chebyshev/Legendre, no-slip),
axial $z$ (Fourier), azimuthal $\theta$ (Fourier, 3-D class only).

- **Base flow — Circular Couette** ($V(r)=ar+b/r$, $\Omega(r)=a+b/r^2$,
  `taylor_couette_linear_jax.py:37-89`):
  $$a=\frac{\Omega_2 R_2^2-\Omega_1 R_1^2}{R_2^2-R_1^2},\qquad
    b=\frac{(\Omega_1-\Omega_2)R_1^2R_2^2}{R_2^2-R_1^2},\qquad
    \kappa^2(r)=4a\,\Omega(r),\qquad q=\frac{2b}{ar^2+b}.$$
  $Re=\Omega_1 R_1\,\text{gap}/\nu$ (`taylor_couette_dns_jax.py:159`).
- **Spaces** (`taylor_couette_dns_jax.py:161-174`): `SD` (Dirichlet radial velocity), `S0` (orthogonal), `SP` (`num_dofs=Nr−2` truncated
  pressure); `CoupledSpace VV=(TD,TD,TD)` (velocity), `VQ=(TD,TD,TD,TP)` (velocity+pressure saddle).
- **Cylindrical Laplacian** (`_lap`, `taylor_couette_dns_jax.py:206-208`, verified):
  $$\Delta u = \partial^2_r u + \tfrac1r\partial_r u + \partial^2_z u.$$
- **Divergence-free / pressure — monolithic saddle-point** (not projection): the coupled
  $(u_r,u_\theta,u_z,p)$ system is solved directly. Continuity rows assembled into `Limp`:
  $q\,\partial_r u_r + q\,\tfrac1r u_r + q\,\partial_z u_z = 0$. **Pressure pinned** at the $k=0$ axial mode
  (`_pin_pressure_modes`). Each axial Fourier mode is an **independent dense radial block**, batch-LU-factored
  (`jax.vmap(jsp_linalg.lu_factor)`, `taylor_couette_dns_jax.py:194-196`) and solved with `lu_solve` — a per-mode dense direct solve, **not**
  an influence/capacitance matrix.
- **Time integrator: CNAB2** (`_build_operators`, `taylor_couette_dns_jax.py:271-348`). $L_{\text{imp}}=M/dt
  -\tfrac12\nu[\Delta-1/r^2]$ plus the Coriolis-like coupling $-\tfrac12(2\Omega)u_\theta$ in the $r$-equation and
  $-\tfrac12(-2a)u_r$ in the $\theta$-equation; $L_{\text{exp}}$ is the explicit ($+\tfrac12$) half; the pressure
  gradient is implicit. The nonlinear advection is AB2.
- **Nonlinear advection (cylindrical, written out, `taylor_couette_dns_jax.py:443-465`, verified):**
  $$n_r = u_r\partial_r u_r + u_z\partial_z u_r - \frac{u_\theta^2}{r},\quad
    n_\theta = u_r\partial_r u_\theta + u_z\partial_z u_\theta + \frac{u_r u_\theta}{r},\quad
    n_z = u_r\partial_r u_z + u_z\partial_z u_z,$$
  evaluated on the dealiased mesh `T0p` (default `dealias=1.5`), then `scalar_product` + `mask_nyquist`.
- **IC seed** (`taylor_couette_dns_jax.py:378-394`): divergence-free streamfunction perturbation $g=\sin^2(\pi(r-R_1)/d)$, or a linear
  eigenmode from `TaylorCouetteLinearJax` (writing both $\pm k$ Fourier coefficients for a real field).
- **Full-3-D class `TaylorCouetteDNSJax`** adds the azimuthal Fourier axis, enforces $2|m|<N_\theta$, and adds the
  azimuthal $1/r$-coupling terms.

### I.C.2.6 Taylor–Couette MHD/MRI DNS

A **7-field total-pressure saddle system** $(u_r,u_\theta,u_z,\Pi,b_r,b_\theta,b_z)$ (`taylor_couette_dns_jax.py:841-847`).

- **Normalizations:** $Re=\Omega_1 R_1\,\text{gap}/\nu$, $Rm=\Omega_1 R_1\,\text{gap}/\eta$, $Pm=\nu/\eta$,
  Lundquist-like $S=B_0\,\text{gap}/\eta$; imposed axial field $B_0$ (default 0.1).
- **Conducting-wall magnetic BCs** (baked into composite bases, `:886-903`, verified): $b_\theta$ uses Robin
  $\text{bc}=\{\text{left}:R=(R_1/J_m,0),\ \text{right}:R=(R_2/J_m,0)\}$ (space `Sbt`); $b_z$ uses Neumann
  (space `Sbz`); $b_r$ Dirichlet (`TD`). No tau rows / no ghosts — the perfect-conductor / pseudo-vacuum set is
  imposed entirely through the trial bases.
- **Linearized MHD coupling** (`_add_mhd_terms`, `:945-984`, verified from source):
  velocity gets $+2\Omega\,u_\theta$ ($r$), $-2a\,u_r$ ($\theta$), viscous $\nu[\Delta-1/r^2]$; magnetic tension
  $+B_0\,\partial_z b_i$ in each velocity equation; induction $+B_0\,\partial_z u_i$ in each magnetic equation;
  resistive $\eta[\Delta-1/r^2]$; and the **$\Omega$-effect / shear-induction**
  $$\frac{\partial b_\theta}{\partial t} \mathrel{+}= (r\,\Omega'(r))\,b_r,\qquad r\,\Omega'(r) = -\frac{2b}{r^2}\quad(\texttt{:959,:978}).$$
- **Nonlinear** (`:1089-1125`): advection minus Lorentz $(\mathbf u\cdot\nabla)\mathbf u-(\mathbf b\cdot\nabla)\mathbf b$
  (cylindrical, with $-u_\theta^2/r$, $-b_\theta^2/r$); induction EMF $\boldsymbol\varepsilon=\mathbf u\times\mathbf b$ then
  $n_b=\nabla\times\boldsymbol\varepsilon$ via radial/axial derivatives
  ($n_{b,r}=\partial_z\varepsilon_\theta$, $n_{b,\theta}=-\partial_z\varepsilon_r+\partial_r\varepsilon_z$,
  $n_{b,z}=-\partial_r\varepsilon_\theta-\varepsilon_\theta/r$).
- **Insulating walls** exist only in the **linear** TC MRI solver (`taylor_couette_mri_jax.py`, flux-function
  formulation, $m=0$ only, via `scipy.special.iv,kv`); the **DNS** is conducting-walls-only. (Matches A; both
  exclude nonlinear insulating DNS.)

### I.C.2.7 TC linear / MRI linear (dense GEP)

`taylor_couette_linear_jax.py` assembles a generalized eigenproblem $(L_0+\nu L_v,\ M)$ per $(m,k_z)$
(`assemble_parts`, `:198-259`), with the explicit cylindrical coupling
$2\Omega=2a+2b/r^2$, shear $=2a$, $r$–$\theta$ coupling $i\,2m/r^2$, advection $-im\Omega(r)$, and
gradient/divergence rows. The energy metric is the $r$-weighted cylindrical mass. Critical-$Re$ comes from a
bisection over $\nu$ and a scan over $k_z$. `taylor_couette_mri_jax.py` assembles the 7-field conducting GEP
$(L_0+\nu L_\nu+\eta L_\eta,\ M)$ with $Ha=B_0\,\text{gap}/\sqrt{\nu\eta}$, plus an ideal local-Keplerian-MRI check
`mri_keplerian_optimum` against theory $s_{\max}/\Omega=0.75$, $(kv_A)^2/\Omega^2=15/16$ [BH91].

---

## I.C.3 Time integrators (`src/jaxfun/integrators/`)

`BaseIntegrator` (`base.py:43`, an `nnx.Module`) splits a weak form into **mass** (time derivative),
**linear** (implicit-eligible), and **nonlinear** (explicit, compiled to a physical-space evaluator). It builds
$M-\gamma\,dt\,L$ once via `build_implicit_operator(γ,dt)` (`base.py:222`), keeps the mass inverse out of the
nonlinear RHS, and batches steps via `jax.lax.fori_loop` with a NaN/Inf divergence guard. The split is the classic
semi-implicit spectral scheme: **viscous/resistive diffusion implicit, advection/Lorentz/EMF explicit**.

The KMM and TC solvers **do not** call `BaseIntegrator.solve`; they hand-roll the stage loop but reuse the
`.stages()` tableaux and the `Biharmonic`/`Helmholtz` wrappers.

### I.C.3.1 ARS additive IMEX-RK (`imex_rk.py`) — exact tableaux

All coefficients verified verbatim from `imex_rk.py`. `PDEIMEXRK` (`:21-93`) uses one DIRK diagonal $\gamma$ and
the per-stage update (`step`, `:73-93`): with $u^0_{\text{rhs}}=M\hat u$,
$$\text{rhs} = u^0_{\text{rhs}} + dt\sum_{j\le rk} B_{rk+1,j}\,N_j + dt\sum_{j<rk} A_{rk+1,j+1}\,L_j,
  \qquad u_{\text{stage}}=(M-\gamma\,dt\,L)^{-1}\,\text{rhs}.$$

| Scheme | $A$ (implicit) | $B$ (explicit) | $C$ (nodes) |
|---|---|---|---|
| **IMEXRK011** | $((0,0),(0,0))$ | $((0,0),(1,0))$ | $(1,0)$ |
| **IMEXRK111** | $((0,0),(0,1))$ | $((0,0),(1,0))$ | $(0,1)$ |
| **IMEXRK222** | $((0,0,0),(0,\gamma,0),(0,1-\gamma,\gamma))$ | $((0,0,0),(\gamma,0,0),(\delta,1-\delta,0))$ | $(0,\gamma,1)$ |
| **IMEXRK443** | 5×5 (`:184-190`) | 5×5 (`:191-197`) | $(0,\tfrac12,\tfrac23,\tfrac12,1)$ |

with **`IMEXRK222`**: $\gamma=(2-\sqrt2)/2$, $\delta=1-\tfrac1{2\gamma}$ (`:176-177`) — the **PCF/KMM default**
(`channelflow_kmm.py:75`). The 443 matrices (verified `:184-198`):
$$A_{443}=\begin{pmatrix}0&0&0&0&0\\0&\tfrac12&0&0&0\\0&\tfrac16&\tfrac12&0&0\\0&-\tfrac12&\tfrac12&\tfrac12&0\\0&\tfrac32&-\tfrac32&\tfrac12&\tfrac12\end{pmatrix},\quad
  B_{443}=\begin{pmatrix}0&0&0&0&0\\\tfrac12&0&0&0&0\\\tfrac{11}{18}&\tfrac1{18}&0&0&0\\\tfrac56&-\tfrac56&\tfrac12&0&0\\\tfrac14&\tfrac74&\tfrac34&-\tfrac74&0\end{pmatrix}.$$

### I.C.3.2 Spalart low-storage IMEXRK3 (`imex_rk.py:96-160`) — exact coefficients

Third-order, **one implicit operator per stage** [SMR91]. Coefficients (verified `:103-105`):
$$A=(\tfrac8{15},\ \tfrac5{12},\ \tfrac34),\qquad B=(0,\ -\tfrac{17}{60},\ -\tfrac5{12}),\qquad
  C=(0,\ \tfrac8{15},\ \tfrac23,\ 1).$$
Per-stage implicit coefficient $\gamma_{rk}=(A_{rk}+B_{rk})\,dt/2$ (`:130,:153`). The stage update
(`step`, `:144-160`, and the KMM `_step_imexrk3`, `channelflow_kmm.py:430-469`):
$$\text{rhs}=M u_{\text{stage}}+\gamma_{rk}\,L u_{\text{stage}}+dt\,(A_{rk}w_0 + B_{rk}w_{\text{prev}}),\quad
  u_{\text{stage}}=(M-\gamma_{rk}\,dt\,L)^{-1}\text{rhs},\quad w_{\text{prev}}\leftarrow w_0.$$

### I.C.3.3 CNAB2 and the explicit menu

`cnab2.py` (Crank–Nicolson + Adams–Bashforth-2, IMEX-Euler bootstrap): `ab2_extrapolate(curr,prev,have_prev)`
$=1.5\,\text{curr}-0.5\,\text{prev}$ after the first step (`:33-45`); `scan_steps` wraps the loop in `jax.lax.scan`
(eager loop on multi-device). **CNAB2 is the Taylor–Couette DNS integrator** (`taylor_couette_dns_jax.py`).
Remaining menu: `BackwardEuler` (1st-order implicit, solves $(M-dt\,L)u=Mu^n+dt\,N$, `backward_euler.py:11-40`),
`RK4` (`rk4.py:10-20`), `ETDRK4` (exponential time-differencing [KT05; CM02], `etdrk4.py`). These are exercised by
the non-shear demos (`diffusion1D_rk4.py`, `cahn_hilliard2D_etdrk4.py`, `nls1D_etdrk4.py`,
`advection1D_backward_euler.py`). `coupled.ars_stage_rhs` provides the coupled-pytree ARS stage accumulation used
by KMM's multi-equation update.

| Solver | Integrator | IMEX split |
|---|---|---|
| KMM / PCF (hydro + MHD + MRI) | ARS **IMEXRK222** (default) or IMEXRK3 | viscous/resistive diffusion implicit (Biharmonic for $u_0$, Helmholtz for $g$ / mean / $A$); convection + base-flow + Lorentz + EMF explicit |
| Taylor–Couette DNS (all variants) | **CNAB2** | linear viscous/coupling CN-split into $L_{\text{imp}}/L_{\text{exp}}$; advection / Lorentz / EMF via AB2 |

---

## I.C.4 JAX compute capabilities

- **Double precision:** float64 by default and unconditional (`__init__.py:8`; §I.C.1.1).
- **JIT:** pervasive `@jax.jit(static_argnums=…)` / `@nnx.jit` on space methods and integrator `step`/`total_rhs`
  (`imex_rk.py:73,144`, `rk4.py:13`, `base.py:262`). Solver state objects are JAX pytrees (`KMMState`,
  `MHDState`, `AxisymmetricTCState`, `AxisymmetricMRIState`), so the stage loop is jittable/scannable.
- **Autograd:** the DNS solvers are differentiable. `pcf_minimal_seed_jax.py` exposes `perturbation_gain` and
  `gain_and_projected_gradient` built on `jax.value_and_grad`, plus an energy-tangent projection for
  adjoint/minimal-seed loops [PWK12], validated against finite differences (`test_differentiability_jax.py`).
  Note: framework-level autograd in the *spectral assembly* path is limited (it appears in `utils/common.py`'s
  `jacn`/`diff` derivative-Vandermonde construction and in PINNs); the **solver-level** autodiff is what makes C
  the autograd oracle.
- **Sharding (multi-device CPU/GPU/TPU)** — `sharding.py:9-21` (verified):
  ```python
  spmd_mesh         = Mesh(jax.devices(), ("k",))
  spectral_sharding = NamedSharding(spmd_mesh, P("k"))        # axis 0 sharded
  physical_sharding = NamedSharding(spmd_mesh, P(None, "k"))  # axis 1 sharded
  ```
  `_build_local_apply_fn` (`:24-40`) is `jax.jit(jax.vmap(1D-transform))` along one axis on a local shard;
  `_apply_separable_spmd_shard_map` (`:43-105`) is the production multi-device transform — a single fused
  `shard_map` with `jax.lax.all_to_all(tiled=True, …, axis_name="k")` to transpose the sharding between phases
  (requires the split axis divisible by #devices, which holds for power-of-2 Fourier / even Chebyshev counts).
  `VectorTensorProductSpace` uses slab sharding `P(None,"k")` (spectral) / `P(None,None,"k")` (physical). SPMD
  tests are gated behind `--num-devices=2 -m spmd`.
- **Host escapes (not jittable):** `scipy.special.roots_jacobi` (Gauss–Jacobi nodes), `scipy.linalg.eig`/`eigh`/
  `svdvals` (eigen/stability), SymPy stencil derivation (compile-time). The dense linear (eig) solvers are
  therefore host NumPy/SciPy regardless of the `_jax` suffix (§I.C.2.0).

Compute gap vs siblings (cross-ref §3): C has **GPU + TPU + autograd + JIT** (A has none of these; B has GPU +
autograd but no JIT/TPU). C still has **no MPI/distributed FFT** — only single-process JAX plus optional
multi-device `shard_map`.

---

## I.C.5 Golden numbers (Family-C, verbatim from `tests/`)

These are the within-family acceptance constants (tolerance ladder per §1 hand-off notes). Use Family-C goldens
only against C; cross-family physics oracles use published values at rel < 1e-2.

**PCF fluctuation diagnostics** (`test_pcf_fluctuations_jax.py:78-95`; $N=(9,8,8)$, Legendre, $dt=10^{-3}$,
amp 0.05, **one step**, x64, `rtol=1e-10`):
```
Epert      = 0.21836099019180652
Etot       = 52.85625108205688
divL2      = 7.183953559387109e-17    (atol 5e-15)
u_top      = 0.968160239435768
u_bot      = -0.9681602394357679
mean_shear = 1.0000000004699001
```

**PCF MHD** (`test_pcf_mhd_jax.py`): `Epert>0`, `Emag>0`, `divL2<1e-4`, `divB_L2<1e-5`; float64-invariant
`magnetic_divergence_l2 < 1e-12`.

**PCF MRI shearpy** (`test_pcf_mhd_mri_shearpy_jax.py:6-23`): `divL2<1e-4`, `divB_L2<1e-5`, finite
`alpha/reynolds_xy/maxwell_xy`, **`q_shear == 1.0`** (at $\Omega=S=1$).

**TC linear leading spectrum** (`test_taylor_couette_linear_jax.py:16-26`; `CircularCouette()` default
$R_1=1,R_2=2,\Omega_1=1,\Omega_2=0$, $\nu=0.002$, $N=12$, Legendre, $m=0$, $k_z=3$, `rtol=1e-11`): leading
eigenvalue $0.36073352898670064 + 4.8\times10^{-22}i$ (5 more in source).

**TC MRI leading spectra** (`test_taylor_couette_mri_jax.py`; Keplerian base
`CircularCouette(1,2,1,0.5**1.5)`, $B_0=0.1$, $\nu=\eta=0.001$, $N=12$, Legendre, $m=0$, $k_z=3$, `rtol=1e-11`):
- conducting leading: $0.25628761535339467 + 1.6\times10^{-16}i$;
- insulating leading: $0.25995005500337837 + 5.2\times10^{-17}i$;
- local Keplerian-MRI optimum: $s_{\max}/\Omega\approx0.75$ (rel 1e-3), $(kv_A)^2/\Omega^2\approx15/16$ (rel 2e-3).

**TC DNS** (`test_taylor_couette_dns_jax.py`): zero-state stays zero; one-step finite with pressure pinned
(`|p[0,0]|<1e-7`); **eigenmode growth-rate matches the linear solver** (axisym hydro `rtol=1e-7`; 3-D hydro/MRI
`rtol=1e-6`; 100 steps, x64); pinned-saddle LU residual `<1e-11`; continuity residual `<1e-18` (x64);
$\nabla\cdot u$, $\nabla\cdot b < 1e-7$.

**Differentiability** (`test_differentiability_jax.py`): `jax.grad` of final perturbation energy vs central FD —
single-step amplitude `rtol=2e-3`, full-state directional `rtol=2e-3`, multi-step finite-amplitude `rtol=5e-3`;
energy-tangent projection orthogonality `atol=1e-10`.

**Shenfun-side benchmark goldens** (`couette/couette_linear_benchmarks.md`, the reference C ports/validates against):
PCF hydro transient growth $Re=1000,\alpha=0,\beta=1.66$: literature $G=1165.2$ vs computed $G=1165.93$ [RH93];
TC hydro onset $\eta=0.5$ outer-stationary $Re_c=68.186$ ($a_c=3.167$, $k_{z,c}=3.167$) [Taylor23]; ideal local
Keplerian MRI $s_{\max}/\Omega=0.7500$, $(kv_A)^2/\Omega^2=0.9373$ (theory 0.75, 15/16) [BH91]; PCF linearly stable
for all $Re$ (Romanov); insulating MRI scan best growth $-2.758\times10^{-4}$.

---

## I.C.6 ABSENT: pipe flow, and `jaxfun_missing_parts.md`

**Pipe flow is ABSENT in Family C.** There is no pipe / Bessel / Openpipeflow code anywhere in the JAX tree (the
only `pipe`/`fastgl` matches are unrelated quadrature/optimizer files). Consequently the pipe-specific tests
**skip** in C (F1d/F1e, S2-pipe; §3 gap matrix, §IV hand-off notes). This is the single geometry gap of C relative
to A and B. Closing it (Phase 5a of §4) means mirroring A's curvilinear $\sqrt g=r$ + `bc=(None,0)` axis-regularity
pipe or B's parity-fold; the framework already has the curvilinear metric machinery (§I.C.1.6) to support it.

**`jaxfun_missing_parts.md`** (`/home/nauman/cfd/jaxfun_missing_parts.md`) is a January note that **predates** the
`examples/` solver work (the JAX solvers are dated June). It argues that PCF/KMM was hard in jaxfun because jaxfun
was then *"a spectral basis/transform and form assembly library (plus PINNs)"* lacking PDE-solver infrastructure.
Its eight listed gaps vs shenfun:

1. **Time-stepping / PDE solver framework** — no IMEX/RK integrator working with spectral operators + BCs.
2. **Channel-flow (KMM) solvers** — *"no Navier–Stokes solver modules at all."*
3. **Fast spectral operator projections** — no precompiled `Project/Dx/curl/div` per-step pipeline.
4. **Implicit Chebyshev/Legendre solvers** — no Helmholtz/Biharmonic wrappers.
5. **Pressure projection + divergence enforcement** — no incompressibility tool, incl. (0,0)-mode + Neumann pressure.
6. **Dealiasing infrastructure** — no padding/filtering tied into transforms.
7. **MPI / parallel decomposition** — no mpi4py-fft / distributed FFT.
8. **IO / checkpointing / diagnostics** — no PDE-specific HDF5 IO.

It concludes a minimal differentiable PCF prototype was *"feasible in days,"* but KMM-level parity *"weeks or more."*

**Reconciliation with the current code (status as of this survey).** Items **1, 2, 4, 5, 6, 8 are now substantially
addressed** in `examples/` + `src/jaxfun`: the IMEX-RK/CNAB2 integrators (`integrators/`), the KMM channel solver
(`channelflow_kmm.py`), the `Biharmonic`/`Helmholtz` wrappers (`la/solvers.py`), KMM pressure recovery + (0,0)-mode
+ Neumann pressure (`compute_pressure_coefficients`), 3/2-rule dealiasing via `get_dealiased`, and HDF5
checkpoint/Cadence IO (`io/__init__.py`). Item **3** is partly addressed (fast separable vmapped transforms +
precomputed sparse operator matrices, though no shenfun-style `Project` cache for every operator). **Still
genuinely missing:** item **7 (MPI / distributed FFT)** — only single-process JAX plus optional multi-device
sharding; **production-scale turbulence runs**; the **pipe geometry** (above); and JAX-native (vs host
NumPy/SciPy) **linear/eig** solvers (§I.C.2.0).

---

### Cross-references
- Conventions & adapters: §0.2 (C needs no axis remap vs A; PCF base flow $+U_{\text{wall}}x\,e_y$, MRI $-Sx\,e_y$).
- Family A (the other spectral oracle, same tableaux/Lorentz=1): §I.A.1–I.A.6.
- Family B (FD, opposite axes, MRI-stub): §I.B (cross-family comparison uses canonical observables).
- Gap matrix: §3 (C: no pipe, no MPI; has GPU/TPU/autograd/JIT). Roadmap: §4 (Phase 5a jax pipe, Phase 5b TPU/autograd parity).
- Tests: §IV (F1/F8 base flows & convergence; SR-1…SR-9 shear/rotation/MHD; C participates in all except pipe-specific skips).


\newpage

# Part II — Cross-family GAP MATRIX

This part summarizes, as a single auditable table, exactly which capabilities each
family ships, which are stubbed, and which are absent — each cell carrying its
`file:line` evidence so an implementer can jump straight to the code. It then walks
through the gaps that matter most for the wall-bounded MHD/MRI program. Conventions,
symbols, and per-family axis maps are defined in §0.2 (Unified Convention Table); the
detailed algorithmic specs that justify each "present" cell are in §I.A (shenfun),
§I.B (torch), and §I.C (jax). The closure roadmap that retires the **S**/**A** cells
is Part III.

Cell legend: **P** = present (working, tested); **S** = stub / partial (present in
name or metadata but not wired into the timestepper, or only one of several needed
variants); **A** = absent (not implemented).

## 3. Gap matrix

| Capability | A (shenfun) | B (torch) | C (jax) |
|---|---|---|---|
| **PCF / channel hydro** | **P** — KMM velocity–vorticity (`ChannelFlow.py:147-164`) | **P** — θ-method predictor/corrector (`torchchannel/solver.py:594-606`) | **P** — KMM, JAX-native (`channelflow_kmm.py:382-516`) |
| **TC hydro** | **P** — coupled saddle-point CNAB2 (`taylor_couette_dns.py:288-313`) | **P** — influence-matrix PC (`torchcouette/solver.py:553-594`) | **P** — pinned saddle, per-mode LU (`taylor_couette_dns_jax.py:271-364`) |
| **Pipe hydro** | **P** — curvilinear √g=r, CNAB2 (`pipe_flow_dns.py:289-321`) | **P** — banded GBTRF/GBTRS LU (`torchpipeflow/banded.py:109-172`) | **A** — no pipe (`jaxfun_missing_parts.md`; C-SOLVERS §1) |
| **PCF MHD** | **P** — `B=curl(A)`, div-free by construction (`pcf_mhd_divfree.py:6-14`) | **P** — induced `b`, full induction (`torchchannel/mhd.py:207-228`) | **P** — `B=curl(A)` (`pcf_mhd_jax.py:108-181`) |
| **TC MHD** | **P** — direct-b conducting, axisym+3D (`taylor_couette_dns.py:788-951`) | **P** — induced b, influence matrix (`torchcouette/mhd.py:179-231`) | **P** — 7-field saddle (`taylor_couette_dns_jax.py:841-984`) |
| **Pipe MHD** | **A** (deferred, low parity value; `PLAN…:182-184`) | **A** — pipe is hydro-only (B-PIPE §11) | **A** (no pipe) |
| **MRI rotation+shear** | **P** — Coriolis + base-shear + shear-induction (`pcf_mhd_mri_shearpy.py:12-15`) | **S** — metadata only, **no source terms** (`torchchannel/mhd.py:71-74`) | **P** — Coriolis + shear-induction (`pcf_mhd_mri_shearpy_jax.py:130-134`) |
| **Insulating / vacuum walls** | **P** — TC linear, m=0, flux-fn (`taylor_couette_mri.py:42-47`) | **A** — homogeneous `b=0` only (`torchchannel/mhd.py:278-282`; B-MHD §3.1) | **P** — TC linear, m=0, flux-fn (`taylor_couette_mri_jax.py`, `_assemble_flux_parts`) |
| **Conducting walls** | **P** — Robin `c=r_wall/J` (`taylor_couette_mri.py:36-40`) | **P** — homogeneous b=0 + influence matrix (`torchcouette/mhd.py:266-276`) | **P** — Robin/Neumann (`taylor_couette_dns_jax.py:886-903`) |
| **Time integrators** | **P** — IMEXRK111/222/3/443, CNAB2 (`integrators.py`) | **S** — θ-method PC only, formally 1st-order (`solver.py:87`) | **P** — IMEXRK222/3, CNAB2, RK4/ETDRK4 (`integrators/`) |
| **Div-free method** | **P** — KMM/saddle-point, div≈1e-16 (`pcf_mhd_divfree_notes.md:69`) | **P** — influence-matrix + pinv, div≈1e-7 (`solver.py:474-482`) | **P** — KMM/pinned saddle, div≈1e-17 (`test_taylor_couette_dns_jax.py`) |
| **GPU** | **A** — CPU/MPI only (A-PCF §10) | **P** — device-agnostic torch (CUDA in benchmarks; B-MHD §6.1) | **P** — JAX/XLA; `cuda13` extra configured (`pyproject.toml`) |
| **TPU** | **A** | **A** | **P** — `shard_map`, device-agnostic (`sharding.py:43-105`) |
| **Autograd** | **A** — NumPy/SciPy (A-PCF §10) | **P** — full, incl. Lorentz coupling (`test_mhd.py:183-201`) | **P** — `value_and_grad` minimal-seed (`pcf_minimal_seed_jax.py`) |
| **JIT** | **A** — numba kernels only | **A** — no `torch.compile`/`jit` (B-MHD §6.4) | **P** — `jax.jit`/`nnx.jit` pervasive (`base.py:262`) |
| **Double precision** | **P** — float64 default | **P** — complex128 default; float32 validated (`solver.py:87`) | **P** — x64-by-default (`__init__.py:8`) |
| **Linear eigensolver** | **P** — `scipy.linalg.eig`, `FINITE_CAP=1e8` (`_linear_analysis.py:16`) | **P** — `scipy.linalg.eig` OS/Squire (`linstab.py:109`) | **P** — `generalized_eig`, dense NumPy (`la/eig.py:168`) |
| **Convergence-order tests** | **P** — golden eig 1e-12, MMS (`OrrSommerfeld_eigs.py:183`) | **P** — OS golden 1e-4, mesh poly-exact deg 8 (`test_mesh.py:21-37`) | **P** — MMS self-asserts (`poisson1D.py:46`) |
| **Cross-family parity** | partial — internal PCF↔TC harness (`thin_gap_compare.py`) | **A** — no cross-boundary test (planned `parity/`; `PLAN…:73-87`) | partial — live shenfun parity (`test_live_shenfun_parity.py`) |

## 3.1 Walkthrough of the load-bearing gaps

Four gaps dominate the closure program; the rest are convenience or coverage items.

**(1) torch MRI is metadata-only — the single highest-value gap.** `ChannelMHDSolver`
accepts `omega` and `shear_rate`, but its own docstring states they "set `q_shear`
but do not add Coriolis or background-shear source terms to the timestepper"
(`torchchannel/mhd.py:71-74`). Verified: `omega`/`shear_rate` are referenced **nowhere**
in any RHS or stage path; `q_shear` is emitted only in `diagnostics()`
(`torchchannel/mhd.py:104,452`). The Taylor–Couette torch MHD class carries **no MRI
metadata at all** (no `omega`, `shear_rate`, or `q_shear`; B-MHD §3.2). By contrast A
and C both ship the verified shearing-box source set — Coriolis $2\Omega\,u_y$ and
$(S-2\Omega)\,u_x$, plus shear-induction $\partial_t B_y = -S\,B_x$ — written into the
nonlinear/EMF path (`pcf_mhd_mri_shearpy.py:12-15`,
`pcf_mhd_mri_shearpy_jax.py:130-134`). The *only* shear-like coupling present in torch
is the **hydrodynamic** base-flow advection $-U\,\partial_x u - U'\,v$
(`torchchannel/solver.py:534-540`); this is plain advection by the laminar profile, not
a rotation/MRI source, and it is not applied to the induction equation. Every MRI claim
about B must therefore be flagged "stub"; the rotation/shear acceptance tests (Part IV
SR-1, SR-2) are the gate that wires it in, not a description of existing behaviour.

**(2) Insulating / vacuum magnetic walls exist only in A and C.** Both A and C
implement the poloidal flux-function (Bessel $I_0/I_1$ inside, $K_0/K_1$ outside)
insulating wall for the Taylor–Couette linear MRI, restricted to $m=0$
(`taylor_couette_mri.py:42-47,332-444`; `taylor_couette_mri_jax.py` `_assemble_flux_parts`).
torch has homogeneous induced-field walls `b=0` only — a perfectly-conducting-style
Dirichlet on all three induced components (`torchchannel/mhd.py:278-282`;
`torchcouette/mhd.py:266-276` with the explicit docstring note that exact insulating
Bessel matching is "a boundary-model upgrade behind the same interface", B-MHD §3.1).
This matters physically because the wall BC **flips the sign of the MRI eigenvalue at
fixed parameters**: conducting gives growth $+0.00332$ (at $\mathrm{Rm}=24.7$, $S=4.11$,
$k_z=1.75$) while insulating gives $-2.76\times10^{-4}$ (at $\mathrm{Rm}=16.5$,
$S=5.21$, $k_z=1.25$), verbatim
`couette_linear_benchmarks.md:34-35,352-353`:

```text
conducting target_Rm 24.7 target_S 4.11 best_kz 1.75 max_growth 0.003322863594034156
insulating target_Rm 16.5 target_S 5.21 best_kz 1.25 max_growth -0.00027582037141390655
```

Any cross-family comparison must therefore pin `magnetic_bc` on both sides and only
compare like-for-like (Part III.2; risk R2 in `PLAN…:252-254`). Note also that
insulating walls remain **linear-only everywhere** — the nonlinear DNS path is
conducting-only in all three families (`mhd_parity_plan.md:35`), so SR-6/SR-7/SR-9 are
A/C-only and *skip* on B.

**(3) Pipe coverage and pipe-MHD are absent where noted.** jax has no pipe solver of
any kind (`jaxfun_missing_parts.md`; C-SOLVERS §1) — the foundational pipe tests
(F1d/F1e, S2-pipe) *skip* for C. Pipe MHD is absent in **all three** families and is an
explicit, documented non-goal: a pipe carries no shear/rotation, so adding MHD surface
yields no MRI physics and therefore low parity value (`PLAN…:182-184,264`). The torch
pipe is a faithful, hydro-only OpenPipeflow port (banded GBTRF/GBTRS LU, ldab
$=2k_l+k_u+1$; B-PIPE §5, §11).

**(4) Compute/autograd/JIT asymmetry.** A is the spectral oracle but has no GPU, no
JIT, and no autograd (NumPy/SciPy linear layer; A-PCF §10). B has full autograd
(gradients verified to flow through projection, BC correction, and the magnetic→velocity
Lorentz coupling, `test_mhd.py:183-201`) and is device-agnostic, but has **no
`torch.compile`/JIT** (batching is explicit `(H,N,N)` tensor ops + `einsum`; B-MHD §6.4)
and its time integrator is a θ-method predictor/corrector that is formally first-order
(`implicit=0.51`; B-CORE §1.9). C has pervasive JIT, `value_and_grad` autograd, x64 by
default, and is the only family with TPU support via `shard_map` (`sharding.py:43-105`) —
but its *linear* solvers are dense NumPy/SciPy rather than JAX-native (C-SOLVERS §2), and
it has no pipe. No single family is complete: A = spectral oracle, B = FD + GPU + autograd
+ banded pipe, C = spectral + JIT + autograd + TPU.

---

# Part III — Prioritized CLOSURE RECOMMENDATIONS

This part is an actionable roadmap that **reuses and extends** the two existing planning
documents: `PLAN_openpipeflow_vs_fnshenfun.md` (A↔B cross-family parity harness and the
torch-MRI wiring) and `mhd_parity_plan.md` (A-internal PCF↔TC parity, the WS-A…WS-J work
streams). Where those plans address only the A↔B or A-internal axes, the recommendations
below **extend the same mechanisms to the jax family C**, so the end state is a single
canonical frame and a single file-based golden bridge spanning all three families.

The four themes are ordered by parity value × tractability, matching the phased
sequencing of `PLAN…:91-188`. A **prerequisite Phase 0** (conventions harness) underpins
all four and is stated first because every cross-family number depends on it.

**Phase 0 (prerequisite for all themes) — conventions harness.** Build the
`to_canonical()` adapter and `conventions.md` mapping every observable to the canonical
frame of §0.2, and unit-test the sign/axis maps before any cross-family number is compared.
*Target files (new):* `fn_openpipeflow-122/parity/conventions.md`,
`fn_openpipeflow-122/parity/observables.py`,
`fn_openpipeflow-122/parity/producers/{shenfun_emit.py,torch_emit.py}`, plus a
`jax_emit.py` to extend the bridge to C (run from `fork_jaxfun/.venv`). Reuse the
existing `_linear_analysis.match_eigenvalues` set-match + Doppler frame conversion — do
**not** reinvent eigenvalue matching (`PLAN…:67-71,120`). *Gate:* the adapter round-trips
A↔B↔C base-flow and growth-rate observables to within each family's truncation band; the
sign/normalization maps pass `test_analytic_parity.py` in every env (`PLAN…:106-107`).
Note the environments are disjoint (A → shenfun conda; B → huggingface conda; C →
`fork_jaxfun/.venv` uv), so the bridge is **committed JSON/HDF5 goldens, not subprocess
calls** (`PLAN…:52-56`).

## III.1 — MHD + MRI completion (highest priority)

This is the headline theme: it retires the **S** in the MRI row for B and the insulating
**A** for B, and it adds the quantitative DNS↔linear validation that A's PCF and all of C
currently lack.

### III.1.a Wire Coriolis + base-shear source terms into torch (closes the MRI stub)

*Steps.* Add a `_rotation_shear_terms` method as a **sibling** of `_base_coupling_terms`
(`torchchannel/solver.py:534-540`), consumed where base coupling already enters
`_rhs_for_state` (`torchchannel/solver.py:571-577`); add the analogous frame-rotation
term in `torchcouette/solver.py` alongside `_compute_rotational_core` (lines 501-514).
Promote `omega`/`shear_rate` from diagnostic metadata to **active** parameters, gated by
a flag that defaults OFF so all existing hydro/MHD tests are byte-for-byte unchanged
(`PLAN…:125-137`). Use the shenfun-verified term set (`pcf_mhd_mri_notes.md:37-42`, also
the C mirror `pcf_mhd_mri_shearpy_jax.py:130-134`), with $\Omega=$`omega`, $S=$`shear_rate`:

$$
\frac{\partial u_x}{\partial t}\mathrel{+}= 2\Omega\,u_y,\qquad
\frac{\partial u_y}{\partial t}\mathrel{+}= (S-2\Omega)\,u_x,\qquad
\frac{\partial B_y}{\partial t}\mathrel{+}= -S\,B_x .
$$

The shear-induction term $\partial_t B_y = -S\,B_x$ goes into the induction RHS of
`torchchannel/mhd.py` (it is added as a separate source on the induced field, unlike A/C
where it arises naturally from $\partial_t A = U_b\times B$ with $U_b=-S\,x\,e_y$;
A-MHD §3, C-SOLVERS §7).

*Target files.* `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`
(`_base_coupling_terms:534`, `_rhs_for_state:571`, `step:594`);
`fn_openpipeflow-122/torchchannel/torchchannel/mhd.py` (stub `:71-104`, induction RHS);
`fn_openpipeflow-122/torchcouette/torchcouette/solver.py` (`_compute_rotational_core:501`);
`fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`.

*Gate (= the first milestone, `PLAN…:191-202`).* With rotation/shear ON and Lorentz OFF,
seed a uniform ($k=0$) mode and assert the epicyclic oscillation **SR-1**:

$$
v_x(t)=\cos(\kappa t),\quad v_y(t)=\frac{S-2\Omega}{\kappa}\sin(\kappa t),\quad
\kappa=\sqrt{2\Omega(2\Omega-S)},
$$

to `rel<1e-2, abs<5e-4` — byte-for-byte the shearpy assertion
`tests/test_theory.py:100-132`. Then **SR-2** isolates the shear-induction term via the
shear-winding relation $B_y(t)=B_{y0}-S\,B_x\,t$ for a uniform mode. This milestone is
geometry- and discretization-independent (uniform mode ⇒ no walls, no spatial
truncation) and has an exact analytic oracle, so no cross-env golden is needed.

### III.1.b Already-runnable torch MHD regressions (no rotation needed)

torch already has magnetic diffusion + full induction + Lorentz, so two analytic MHD
oracles can be turned on **immediately** as torch-only regressions, before the rotation
wiring lands: **SR-3** Ohmic decay $E(t)=E_0\exp(-2\eta k^2 t)$
(`test_theory.py:66-97`), and **SR-4** Alfvén phase $\omega=k\cdot v_A$
(`test_theory.py:135-178`). *Gate:* rate / frequency match to `rel<1e-2` with an explicit
`lorentz_prefactor=1` override (B's default channel prefactor is $\mathrm{Ha}^2/(\mathrm{Re}\,\mathrm{Rm})$,
`torchchannel/mhd.py:101`; couette is $\mathrm{Ha}^2/\mathrm{Pm}$, `torchcouette/mhd.py:79` —
set the override to 1 to match the A/C Alfvén-unit oracle, §0.2).

### III.1.c Insulating / vacuum magnetic walls for torch (and DNS-level for all)

*Steps (linear, torch).* Add an opt-in `magnetic_bc="insulating"` to the two torch MHD
solvers. For each tangential Fourier wavenumber $q=\sqrt{k_y^2+k_z^2}$, replace the
homogeneous tangential rows with the exterior-potential-decay (Robin / Bessel-matching)
relation linking the normal induced field to its tangential structure at the wall,
handling the mean tangential mode explicitly. Pattern from
`taylor_couette_mri.py:142-153,366-371` (the shenfun Robin coefficient
$c=-\kappa/(k_z^2 J)$ with $\kappa$ the modified-Bessel log-derivative). Keep conducting
walls the default. *Gate:* torch insulating-wall linear MRI $\gamma(k_z)$ matches the
shenfun **insulating** golden (like-for-like BC only — conducting↔insulating differ in
sign by design, R2), with `div(B)` at within-family roundoff (`PLAN…:158-168`).

*Steps (A and C, extending `mhd_parity_plan.md`).* A's PCF currently has only
`conducting` + diagnostic `dirichlet` (`_pcf_linear.py:115-116`). Add the
pseudo-vacuum / vertical-field wall ($B_y=B_z=0$, $B_x$ free) as the cheapest real second
BC for PCF (WS-D, `mhd_parity_plan.md:103-120`), keeping the $\varphi$ magnetic-pressure
multiplier so `div(b)=0` survives. Mirror the same flux-function insulating BC into C's
linear path (it is already present for C's TC linear MRI, so the work is to extend it to
PCF). *Gate:* SR-6 / SR-9 conducting↔insulating sign-flip reproduced ($+0.00332$ vs
$-2.76\times10^{-4}$, verbatim above); Robin BC satisfied $<10^{-10}$ (the A test
`test_taylor_couette.py:210-223` precedent). **Insulating nonlinear DNS is net-new for
all three families** (WS-J, `mhd_parity_plan.md:194-198`) and is deferred to a stretch
goal.

### III.1.d MRI/MHD fidelity in jax + PCF growth validation (extends WS-A to C)

*Steps.* A's PCF MRI DNS validation is currently **qualitative** — it only checks that
$E_\mathrm{mag}$ grows monotonically and `div(B)` stays at roundoff
(`test_pcf_mhd_mri_shearpy.py:120-133`), not a growth-rate match (WS-A,
`mhd_parity_plan.md:44-64`). Add a `seed_linear_eigenmode(ky,kz,amp)` that builds the
leading eigenvector from the linear operator, projects it onto the DNS `A`/`u` spaces
(magnetic part via `A` consistent with $B=\mathrm{curl}(A)$, with the complex
eigenvector real/imag split to preserve solenoidality), and returns the linear
eigenvalue; then fit $d/dt\log E$ and assert it matches $\mathrm{Re}(s)$. Extend the
**same** validation to C's PCF MHD/MRI DNS (C already has the seeding machinery for TC,
`taylor_couette_dns_jax.py:396-428`, and differentiable diagnostics). Also add to A the
critical-parameter finder (WS-B) and the local analytic MRI `--local-check` (WS-C) that
TC and C already have. *Target files.* `pcf_mhd_mri_shearpy.py`, `_pcf_linear.py`,
`test_pcf_mhd_mri_shearpy.py` (A); the jax PCF MHD/MRI examples + tests (C). *Gate:* PCF
DNS growth matches the linear eigenvalue to $2\times10^{-3}\cdot|s|$ — the TC precedent
(`taylor_couette_notes.md:401-406`; per-$m$ rel-errs 4e-8/4e-7/2e-6 reused as the WS-A
target, `mhd_parity_plan.md:46`); `div(B)`, `div(u)` at roundoff throughout. Tighten to
$\sim10^{-6}$ after the imposed-field Alfvén coupling is made implicit (WS-G,
`mhd_parity_plan.md:154-169`).

## III.2 — Coordinate / sign unification

This theme is **Phase 0** of `PLAN…` and the prerequisite for every cross-family number,
restated here as a standalone deliverable because it is the documented "silent killer"
(`PLAN…:62-66`).

*Steps.* (i) Build the single canonical frame of §0.2 and a `to_canonical()` adapter per
family in `observables.py`. The non-trivial adapter is **B**: relabel axes
$(x_B,y_B,z_B)\to(y_\mathrm{can},x_\mathrm{can},z_\mathrm{can})$ (B puts streamwise
velocity in component `x` as a function of wall-normal `y`, `base_flow.py:37-41`, opposite
to the A/C convention `U_b=\sigma\,x\,e_y`), and for MRI comparisons flip the shear sign
$S\to-S$ to match the shearpy $U_b=-S\,x\,e_y$ convention. A and C need only sign
selection ($+U_\mathrm{wall}$ hydro / $-S$ MRI), no axis remap. (ii) Pin the Lorentz
prefactor: A and C use 1 (Alfvén units), B uses $\mathrm{Ha}^2/(\mathrm{Re}\,\mathrm{Rm})$
(channel) or $\mathrm{Ha}^2/\mathrm{Pm}$ (couette) — provide an explicit `override=1` for
oracle tests (§0.2; B-MHD §8 item 1). (iii) Write `conventions.md` documenting every
observable's map and review it before comparing any number.

*Target files (new).* `fn_openpipeflow-122/parity/conventions.py`,
`fn_openpipeflow-122/parity/observables.py`,
`fn_openpipeflow-122/parity/conventions.md`. *Mirror for design:*
`fn_shenfun/demo/thin_gap_compare.py`, `fn_shenfun/demo/_linear_analysis.py`
(`match_eigenvalues`, Doppler frame conversion) — extend its philosophy across the env
boundary, do not reinvent it (`PLAN…:67-71`).

*Gate.* The adapter round-trips base-flow profiles and growth-rate observables across
A↔B↔C to within the truncation band, using the existing `match_eigenvalues` set-match
(not index-match) with Doppler frame conversion; `conventions.md` reviewed; sign/axis
maps unit-tested in `test_analytic_parity.py` (`PLAN…:106-107`). All cross-family
comparisons thereafter operate on canonical-frame, frame-invariant observables (growth
rates, energies, stresses) only.

## III.3 — Geometry coverage

*Steps.* (i) **jax pipe solver** (closes the only geometry **A** that is not a documented
non-goal): mirror A's curvilinear $\sqrt{g}=r$ weighting with the unified `bc=(None,0)`
axis-regularity basis (no $m$-split; A-TC-PIPE §3), or B's parity-fold approach
(negative-radius ghosts; B-PIPE §3). C already has the `CoordSys`/Christoffel curvilinear
machinery (C-FRAMEWORK §7) and the CNAB2 saddle-point integrator (C-SOLVERS §3) needed to
host it. *Target files (new):* a `pipe_flow_dns_jax.py` + tests under
`fork_jaxfun/examples/` and `tests/couette/`. *Gate:* Hagen–Poiseuille flow rate
$Q=\pi R^4 f_z/(8\nu)$ to `rel<1e-10` (the A precedent: $Q$ rel-err 1.4e-12,
`pipe_flow_notes.md:67`); Womersley oscillatory solution to `<5e-6`
(A: $\max|u_z-\mathrm{exact}|=8\times10^{-7}$, `pipe_flow_notes.md:69`). Note F1e
(Womersley) *skips* on B because torch pipe has no oscillatory forcing (B-PIPE §1).

(ii) **Pipe MHD** is **explicitly deferred** for all families — a pipe has no
shear/rotation, so MHD adds surface but no MRI physics, hence low parity value
(`PLAN…:182-184,264`; `mhd_parity_plan.md` does not list it as a parity gap). Documented
as a non-goal in `conventions.md`, not built.

## III.4 — Compute / autograd / portability parity

*Steps.* (i) **TPU sharding validation for C:** exercise the `shard_map` separable
transforms (`sharding.py:43-105`) under multi-device, gated behind `--num-devices=2 -m spmd`
(C-FRAMEWORK §9, §14). (ii) **Autograd parity statement across B and C:** B's autograd is
verified end-to-end including the magnetic→velocity Lorentz coupling
(`test_mhd.py:183-201`); C's via `value_and_grad` minimal-seed adjoint
(`pcf_minimal_seed_jax.py`, finite-difference-checked in `test_differentiability_jax.py`).
Add a parity statement / cross-check that both compute the same gain gradient for a
matched setup (canonical-frame observables only). (iii) **Double-precision floors:** all
three already default to float64/complex128/x64; assert this in the harness so a regression
(e.g. an accidental float32 path) is caught — B validates float32 separately with err
`<5e-5` (`test_float32.py`), which must not become the default. (iv) **Document the
asymmetry as non-goals where appropriate:** A has no GPU/JIT/autograd by design (spectral
oracle role); adding them is out of scope.

*Gate.* `shard_map` 2-device parity for C (`-m spmd`); B↔C autograd gain-gradient
agreement within the truncation band on a canonical-frame setup; double-precision asserted
in the harness for all three families.

---

### Closure summary (what each phase retires)

| Phase | Theme | Retires (gap-matrix cell) | Acceptance gate |
|---|---|---|---|
| 0 | Conventions harness | — (prerequisite) | adapter round-trips A↔B↔C; `conventions.md` reviewed |
| III.1.a | torch MRI wiring | MRI row **S→P** (B) | SR-1 epicyclic `rel<1e-2`; SR-2 shear-winding |
| III.1.b | torch MHD regressions | (validation) | SR-3 Ohmic, SR-4 Alfvén `rel<1e-2`, prefactor=1 |
| III.1.c | insulating walls | insulating row **A→P** (B); add PCF (A/C) | SR-6/SR-9 sign-flip $+0.00332$ vs $-2.76\mathrm{e}{-4}$; Robin `<1e-10` |
| III.1.d | jax MRI fidelity + WS-A | (validation, A/C) | PCF DNS growth vs linear `2e-3·\|s\|` |
| III.2 | coordinate/sign unification | cross-family parity (B) | canonical round-trip; `match_eigenvalues` set-match |
| III.3 | geometry coverage | pipe hydro **A→P** (C) | Hagen–Poiseuille `rel<1e-10`; Womersley `<5e-6` |
| III.4 | compute/autograd parity | TPU/autograd statements | `shard_map` 2-device; B↔C gradient parity; fp64 floor |

Explicitly deferred (documented non-goals, `PLAN…:264`, `mhd_parity_plan.md:194-198`):
pipe MHD (no shear/rotation → low parity value); stratification/buoyancy; shenfun
shearing-periodic remap (a rewrite); reproducing published *turbulent saturated* $\alpha$
(resolution/box/Pm-sensitive); insulating-wall nonlinear DNS for all families (WS-J,
low priority).


\newpage

# Part IV — Test suite

This part specifies the executable acceptance suite for all three families (A=shenfun, B=torch, C=jax) defined in §1 and §0.1. It is organized into four subsections that mirror the §1 section map:

- **IV.1 Foundational tests** (`F1`–`F8`) — the *common floor* every family must satisfy: laminar base-flow profiles, the divergence-free / `div(B)` identities, conservation / energy-balance closure, discrete symmetries, the Orr–Sommerfeld / Squire eigenvalue benchmark, and the 2D Taylor–Green decaying-mode oracle.
- **IV.2 Temporal discretization-order tests** (`T1`–`T4`) — per-integrator `Δt`-halving slope extraction plus the IMEX splitting-error probe.
- **IV.3 Spatial discretization-order tests** (`S1`–`S3`) — MMS recipe; spectral exponential (A,C) vs FD ~algebraic (B); the cross-family *floor-meeting* parity definition.
- **IV.4 Shear / rotation / MHD regime tests** (`SR-1`…`SR-9`) — the headline physics: epicyclic frequency, RDT shear-winding, Ohmic decay, Alfvén wave, ideal MRI dispersion, wall-bounded MRI conducting vs insulating, the `Pm`-scan, the magnetic energy/stress budget, and the BC sign-flip regression.

**Which tests form the "foundational floor."** The floor that MUST exist in **all three families after closure** is: `F1a` (Couette `U=y`), `F1b` (Poiseuille `1−x²`), `F1c` (Taylor–Couette `Ar+B/r`), `F2` (div-free, and `div(B)` where MHD exists), `F3` (energy balance), `F4` (geometry symmetries), `F5` (Orr–Sommerfeld/Squire), and `F6` (2D Taylor–Green). The implementation audit found no target-tree F6/TGV harness, so F6 is a required addition in A/B/C rather than current coverage. Pipe-only foundational tests (`F1d` Hagen–Poiseuille, `F1e` Womersley) are **not** part of the universal floor because Family C has no pipe (§1.C.4, GAP MATRIX §3 "Pipe hydro = **A**") and Family B has no Womersley driving; they are recorded as explicit `skip`s, never failures.

**The tolerance ladder (used consistently throughout, per the §"Tolerance ladder" hand-off note).**

| Tolerance class | When it applies | Magnitude |
|---|---|---|
| Within-family operator identity | div-free, `div(B)`, symmetry residual, Robin-BC satisfaction | roundoff: `1e-10…1e-21` (spectral A/C), `1e-7` (B pinv cleanup) |
| Cross-family physical observable | any A↔B↔C comparison of growth rate / energy / eigenvalue | truncation band `max(C·Δx^p, C·Δt^q, ε_{spectral})` — **never roundoff** |
| Closed-form physics oracle | epicyclic κ, Alfvén ω, Ohmic rate, ideal MRI `0.75Ω` | `rel<1e-2` (tighten to `1e-3/1e-4` within-family) |

The cross-family band derivation is fixed by `PLAN_openpipeflow_vs_fnshenfun.md:57-61`: *"Tolerances for any cross-family comparison are derived as `max(C·Δx_FD⁴, C·Δt_FD², ε_spectral)` from the actual grid — never roundoff."* Because Family B is the coarsest (Fourier + FD wall-normal + a formally first-order θ-method at `implicit=0.51`), the band is almost always dominated by B's `Δt` and `Δx`, giving the `rel<1e-2` figure for cross-family physics comparisons.

**Order-extraction recipe (shared by `F6`, `T1`–`T4`, `S1`–`S3`).** Two interchangeable estimators:

```
# log-log slope (oracle available)
p = polyfit(log(dt_or_h), log(error), 1).slope

# three-level Richardson (no oracle needed)
p = log( (u[dt] - u[dt/2]) / (u[dt/2] - u[dt/4]) ) / log(2)
```

Report **both** the least-squares slope over the full ladder and the successive pairwise orders `p_i = log(E_i/E_{i+1})/log 2`; a non-monotone pairwise sequence flags either the roundoff floor (drop coarse points) or a pre-asymptotic regime (refine further). This is the direct analogue of the shearpy RDT slope test (`shearpy/tests/fd_proto/test_rdt.py:12-26`, which extracts `time_order = log(coarse_t/fine_t)/log 4` and `space_order = log(coarse_x/fine_x)/log 2`).

---

## IV.1 Foundational tests (`F1`–`F8`)

Notation: `x` = wall-normal/radial (Dirichlet walls), `y` = streamwise/azimuthal, `z` = spanwise/axial (rotation axis), per the §0.2 canonical frame. `Δx` = wall-normal/radial grid spacing; `p` = spatial order (spectral ⇒ `ε_spectral ≈ 1e-10…1e-12`; FD ⇒ nominal 4, but B's default 9-point `KL=4` stencil is formally 8th-order interior, §0.2 contradiction #2); `q` = temporal order. Family-B axis labels are swapped relative to canonical (`base_flow.py:37-41`): apply the planned `to_canonical()` adapter of §0.2 before any cross-family comparison.

### F1 — Laminar base-flow profiles (exact steady/periodic oracle)

#### F1a. Plane Couette `U_b = σ·x·e_y`

**Description & rationale.** The linear shear is the exact steady laminar fixed point of plane Couette. Two-pronged assertion: (i) the analytic base flow used in the convection term is exactly linear in `x`; (ii) a zero fluctuation stays zero (the laminar state is a true fixed point). Families A/C evolve only the *fluctuation* about the base flow; B may carry the base flow as fluctuation or full field.

**Setup.** Channel `x∈[-1,1]` wall-normal, periodic `y,z`. No-slip moving walls `u(±1)=±U_wall·e_y`. `Re=500`, `U_wall=1`. IC `u'=0`. Resolution A/C `N=(9,8,8)`, B `N=9,K=1,M=1`. `dt=0.01`.
- Family A: `U_b = +U_wall·x·e_y`, `dU_b/dx = U_wall` (`pcf_fluctuations_corrected.py:130-135`); streamwise = axis 1 (`y`).
- Family B: `U=y.clone(), Up=ones, Upp=zeros, walls=(-1,1)` (`base_flow.py:37-41`); streamwise = `x` (swapped — apply adapter).
- Family C: `U_b = +U_wall·x·e_y`, `dU_b/dx = U_wall` (`pcf_fluctuations_jax.py:65-67`).

**Oracle.** `U(x)=x` exactly (`U'=1`, `U''=0`); wall values `±1`. Plane Couette is the canonical linearly-stable base flow for all `Re` [Nagata90]; Romanov (1973) (`couette_linear_benchmarks.md:84-88, 440`).

**Metric & tolerance.**
- *Base-flow identity (roundoff):* `max_j|U(x_j) − x_j| < 1e-13` for A/C (degree-1 polynomial ∈ trial space exactly; B stores the mesh `y` exactly). Roundoff is justified because `U=σx` lies in every family's trial space exactly.
- *Fixed-point check:* `E_pert < 1e-20` after one step from `u'=0` (C reports `E_pert=0` exactly, `test_pcf_fluctuations_jax.py:48-53`; B `perturbation_energy < 1e-20` after 2000 steps, `test_step_decay.py:29-51`).
- *mean-shear cross-check:* `mean_shear ≈ σ`. C golden `mean_shear = 1.0000000004699001` after one step; within-family `rel<1e-8`, cross-family `rel<1e-2`. **Sign note:** in the MRI/shearpy convention `σ = −S`, so the shearpy diagnostic reports `mean_shear = −1` (the test asserts `|mean_shear + 1.0| < 1e-10`, `test_pcf_mhd_mri_shearpy.py:60`).

**Families.** All three. Cross-family compare the *full* streamwise mean profile `⟨u⟩(x) + U(x)`, never the fluctuation (families differ on whether the base flow lives in the state vector).

#### F1b. Plane Poiseuille `U = 1 − x²`

**Description & rationale.** The parabolic channel profile validates the body-force / pressure-gradient term and the constant-flux constraint.

**Setup.** Channel `x∈[-1,1]`, stationary no-slip walls `u(±1)=0`. Mean gradient `dp/dy = −2/Re` (A) imposing the parabola. IC `u=1−x²` (full) or zero fluctuation. `N≈(128,32,4)` for the OS validation case, `N=(33,…)` for the profile check.
- Family A: `U_b=(1−x²)e_y`, `dpdy=−2/Re` (`OrrSommerfeld.py:14,30`).
- Family B: `U=1−y², Up=−2y, Upp=−2, walls=(0,0)`, const-flux target `4/3` (`base_flow.py:42-46`; `_flux_target` `solver.py:621-626`).
- Family C: Poiseuille via the OS path (mirrors A).

**Oracle.** `U(x)=1−x²`, centerline `U(0)=1`, walls `U(±1)=0`, `U''=−2`, flux `∫_{-1}^{1}(1−x²)dx = 4/3` [Orszag71].

**Metric & tolerance.**
- *Profile:* `max_j|U(x_j) − (1−x_j²)| < 1e-12` for A/C (degree-2 ∈ trial space); `< 1e-7` for B (9-point FD polynomial-exact through degree 8, `test_mesh.py:21-37`).
- *Constant-flux oracle (B):* `|flux − 4/3| < 1e-12` (`test_step_decay.py:108-126`).
- *Curvature:* `max_j|U''(x_j) − (−2)| < 1e-10` (A/C), `< 1e-7` (B).

**Families.** All three; cross-family compare centerline value and flux (formulation-independent), band ≈ `1e-7` (B FD-limited).

#### F1c. Taylor–Couette `U_θ = A r + B/r`

**Description & rationale.** Circular-Couette azimuthal profile is the exact steady annular solution; validates the cylindrical `1/r` metric and the wall-rotation BCs.

**Setup.** Annulus `r∈[R1,R2]`, no-slip `U_θ(R1)=Ω1 R1`, `U_θ(R2)=Ω2 R2`. Canonical A case `R1=1,R2=2,Ω1=1,Ω2=0` (`η=0.5`); B default `η=0.868, Re_i=200, Re_o=−200`. IC zero fluctuation. `Nr=12…48`.
- Family A: `V(r)=ar+b/r`, `a=(Ω2 R2²−Ω1 R1²)/(R2²−R1²)`, `b=(Ω1−Ω2)R1²R2²/(R2²−R1²)` (`taylor_couette_linear.py:89-91`).
- Family B: `u_θ=ar+b/r`, `a=(Re_o−η Re_i)/(1+η)`, `b=η(Re_i−η Re_o)/((1−η)(1−η²))` (`base_flow.py:18-30`).
- Family C: same as A (`taylor_couette_linear_jax.py:37-89`).

**Oracle.** For the canonical A case `a=−1/3`, `b=4/3` ⇒ `V(r)=−r/3 + 4/(3r)`; check `V(1)=1=Ω1 R1`, `V(2)=0=Ω2 R2`. The hydro onset for this case is `Re_c=68.18635`, `kz_c=3.1667` (`couette_linear_benchmarks.md:29,227`), available as a stronger linear regression.

**Metric & tolerance.**
- *Wall values (roundoff):* `|V(R1)−Ω1 R1| < 1e-13`, `|V(R2)−Ω2 R2| < 1e-13` (`atol=1e-13`-class assertion, `test_taylor_couette.py`).
- *Constant-shear identity:* `2Ω(r) + r Ω'(r) = 2a` (constant); `max_r|2Ω + rΩ' − 2a| < 1e-12`.
- *Laminar fixed point:* zero perturbation stays zero, `div_linf < 1e-4` (B); `energy()==0.0` (A).
- *Torque/Nusselt oracle (B):* laminar `τ_lam = −2b/r²`, `Nu_i=Nu_o≈1.0` within `atol=1e-8`.

**Families.** All three. A/C use explicit-`1/r` plain-measure (NOT curvilinear shenfun for TC, §1.A.2); B uses cylindrical FD. Base sign `U_base = +V(r)e_θ`. Cross-family sample `U_θ(r)` at common radii, band `max(C·Δx⁴, ε_spectral)`.

#### F1d. Hagen–Poiseuille parabolic pipe *(A, B only — C skip)*

**Description & rationale.** Steady axisymmetric pipe flow under uniform axial forcing — the exact parabola with maximum **on the axis**, which critically validates axis regularity at `r=0` (a naive `u_z(0)=0` BC is fatal, §1.A.3).

**Setup.** Pipe `r∈[0,R]`, `R=1`, periodic `z`, no-slip `u_z(R)=0`, regular axis. `Nr=32, Nθ=8, Nz=8` (A); `N=64,K=18,M=32` (B).
- Family A: `u_z(r)=(f_z/(4ν))(R²−r²)`, `Q=πR⁴f_z/(8ν)` (`pipe_flow_dns.py:473-475`); axis via unified `bc=(None,0)`.
- Family B: `U(r)=1−r²`, `U'=−2r`, `_b_hpf=2r` (`solver.py:200-203`); axis via parity folding (negative-radius ghosts).
- Family C: **N/A — no pipe** (§1.C.4); record as `skip`.

**Oracle.** `u_z(r) ∝ R²−r²`; flow rate `Q=πR⁴f_z/(8ν)`; laminar Darcy friction `f=64/Re`; linearly stable for all Re [EBHW07].

**Metric & tolerance.**
- *Profile:* `max|u_z − exact| < 1e-6` (A golden, `test_pipe_flow_dns.py:33-37`).
- *Flow rate (tight oracle):* `|Q − πR⁴f_z/(8ν)|/Q < 1e-10` (A; notes report rel.err `1.4e-12`, `pipe_flow_notes.md:67`). B drives mean axial flux to `<1e-8` (`test_invariants.py`).
- *Divergence:* `div_l2 < 1e-10` (A), `< 1e-8` (B).

**Families.** A & B only. A uses curvilinear `√g=r` weighting (do NOT re-multiply by `r`, §1.A.3); B uses parity folding. Cross-family compare centerline velocity and `Q`.

#### F1e. Womersley pulsatile pipe (periodic-in-time oracle) *(A only)*

**Description & rationale.** Oscillatory pipe flow under a periodic axial gradient — the **only** non-trivial unsteady analytic foundational benchmark, exercising the time integrator AND the radial operator simultaneously.

**Setup.** Pipe `r∈[0,1]`, forcing `−dp/dz = K cos(ωt)`. Womersley number `α_W=R√(ω/ν)`; A's test uses `α_W=3`, `ω=9`, period `2π/9≈0.698`. `Nr≈32`, `dt` small so `(ωΔt)²` is below tolerance.
- Family A: `u_z(r,t)=Re{(K/(iρω))[1 − J_0(i^{3/2}α_W r/R)/J_0(i^{3/2}α_W)] e^{iωt}}`, `ρ=1`, `i^{3/2}=e^{i3π/4}` (`pipe_flow_dns.py:478-490`).
- Family B: **not implemented** — torchpipeflow has no oscillatory forcing; `skip`.
- Family C: **N/A — no pipe**; `skip`.

**Oracle.** The Womersley Bessel solution above [Womersley55].

**Metric & tolerance.** `max|u_z − exact| < 5e-6` over a full period (A golden; notes report `8e-7`, rel `6e-6`, `pipe_flow_notes.md:69`). Temporal error scales `~(ωΔt)²` (CNAB2, 2nd order); halving `Δt` confirms slope ≈ 2 (see F8). This is the one foundational test that is single-family by necessity — document the gap (B/torchpipeflow has no Womersley; C has no pipe).

### F2 — Divergence-free residual `≈ 0` (and `div(B)≈0` for MHD)

**Description & rationale.** Incompressibility `∇·u=0` (and solenoidality `∇·B=0`) must hold to *within-family roundoff*. This is a pure operator/projection identity, so the tolerance is roundoff — the single most diagnostic foundational test, since a nonzero divergence signals a broken projection, pressure solve, or compatible-space chain.

**Setup.** Any geometry, after one or several steps from a nontrivial divergence-free IC; resolution/`dt` per F1. Measure `‖∇·u‖` and (MHD) `‖∇·B‖`.
- Family A: KMM eliminates pressure exactly, recovers `v,w` enforcing `div(u)=0` (§1.A.1); MHD `B=curl(A)` ⇒ `div(B)=div(curl A)=0` by the discrete identity (§1.A.4); TC/pipe saddle-point enforces `div(u)=0`.
- Family B: influence-matrix + projection + dense `enforce_constraints` pinv cleanup (§1.B.3); MHD induced `b=0` walls + same pinv for `div(b)=0`.
- Family C: KMM reconstruction (PCF) / pinned saddle-point (TC); `B=curl(A)` (PCF MHD).

**Oracle.** Exactly zero (operator identity); achievable floor = roundoff at float64.

**Metric & tolerance (per-family roundoff — verbatim goldens).**

| Family | Geometry | Metric | Golden / gate | Source |
|---|---|---|---|---|
| A | PCF MHD (Legendre `N=(8,8,8)`) | `divU L2` / `divB L2` | `9.41e-17` / `3.05e-21` | `pcf_mhd_divfree_notes.md:69-73` |
| A | PCF MHD (Cheb `N=(16,16,16)`) | `divU L2` / `divB L2` | `9.03e-17` / `4.71e-21` | §1.A.4 |
| A | PCF MHD near-transition | `divU L2` / `divB rel RMS` | `2.84e-16` / `8.32e-16` | §1.A.4 |
| A | shearpy MRI (Legendre `N=(8,8,8)`) | `divb_l2`, `divu_l2` | `< 1e-12` (gate) | `test_pcf_mhd_mri_shearpy.py:61-62` |
| A | TC DNS | `div_linf` | `< 1e-9·umax` | `test_taylor_couette_dns.py` |
| A | Pipe | `divergence_l2` | `1e-13…1e-11`; gate `<1e-9` | §1.A.3 |
| B | Channel | `divergence_norm(include_walls)` | `< 1e-12` | `test_step_decay.py:29-51` |
| B | Channel MHD | `divLinf, divB_Linf, divB_L2` | `< 1e-7` | `test_mhd.py` |
| B | Couette MHD | `div(velocity), divB_Linf` | `< 1e-8` | §1.B.5 |
| B | Pipe | `div_linf` | `< 1e-6` | `test_invariants.py` |
| C | PCF | `divL2` | `7.18e-17` (gate `<5e-15`) | `test_pcf_fluctuations_jax.py:78-95` |
| C | PCF MHD | `divL2`/`divB_L2` | `<1e-4`/`<1e-5`; x64 `divB<1e-12` | §1.C.3 |
| C | TC DNS | continuity residual | `< 1e-18` (x64) | §1.C.2 |

**Tolerance rationale.** This is a *within-family operator identity* → roundoff. Spectral families (A,C) reach `1e-16…1e-21` because the divergence operator and the compatible-space chain are exact; the `div(B)` floor for A is spectacular (`1e-21`) because `B=curl(A)` is a *structural* identity. Family B reaches `1e-7…1e-12` because the influence-matrix/pinv cleanup leaves a small dense-least-squares residual (not an exact null-space projection). **Do NOT apply a cross-family band here** — each family asserts its own roundoff floor. **Extraction note (A gotcha):** use *separate per-term projections* of each divergence contribution, or the residual is spuriously `O(amplitude)` (§1.A.2).

### F3 — Conservation / energy-balance closure

**Description & rationale.** The kinetic-energy budget `dE/dt = P − D` (production minus dissipation) must close to ≈0, and decaying cases must decay monotonically. Validates that the discrete nonlinear term conserves energy (no spurious production) and the viscous term is purely dissipative.

**Sub-cases.**

**F3a — Inviscid energy conservation (high-Re limit).** Channel, `Re=1e12`, rotational nonlinear form, `dt=1e-4`, 50 steps. Golden (B): `worst_rel < 1e-6`, `div < 1e-8` (`test_step_decay.py:246-271`). Assert relative energy drift `< 1e-6`.

**F3b — Monotone viscous decay (no production).** Pure-diffusion problem (Stokes, no base flow): seed a mode, assert energy decreases every step and the decay rate matches the analytic eigenvalue.
- A pipe Stokes `m=1`: strictly monotonic, asymptotic rate `0.178`, plateau `<5%` (`pipe_flow_notes.md:71`).
- A pipe Bessel mode: `E~e^{−2λt}`, `rate=−log(E1/E0)/(2(t1−t0))`, oracle `λ = ν j_{01}²/R² = 5.78319`, measured `5.78320`, rel.err `2.8e-6` (§1.A.3). Assert `|rate − ν j_{01}²/R²|/(ν j_{01}²/R²) < 1e-4`.
- B channel MHD magnetic diffusion: `0 < energy1 < energy0` (`test_mhd.py:163-180`).
- C: TC/PCF decaying-mode energy decreases (§1.C.2).

**F3c — Production–dissipation closure (sheared).** PCF/TC with base flow: compute `dE/dt` by finite difference and `P−D` from the stress/dissipation integrals; assert closure near zero.

**Oracle.** F3a `dE/dt=0` at `ν→0`; F3b `E(t)=E_0 e^{−2λt}` (`λ=ν j_{01}²/R²` Bessel); F3c the Reynolds–Orr equation `dE/dt = P − D`.

**Metric & tolerance.**
- F3a: `|E(t)−E(0)|/E(0) < 1e-6` (within-family energy-conservation property).
- F3b: monotonicity is a hard boolean (every step `E_{n+1}<E_n`); rate match is an analytic oracle → `rel < 1e-4` (A Bessel golden `2.8e-6`).
- F3c: closure `|dE/dt − (P−D)|/E <` truncation band `max(C·Δx^p, C·Δt^q, ε_spectral)` — `~1e-8` for spectral A/C; `~1e-2` for B (θ-method, `q≈1`).

**Tolerance rationale.** F3a/F3b monotonicity = structural property → tight; F3b rate = analytic oracle → `rel<1e-4`; F3c carries a *temporal* component (the FD of `dE/dt`) and so sits in the truncation band, dominated by `Δt^q` (`q=1` for B's θ-method, `q=2` for CNAB2 / IMEX-RK).

**Families.** All three. F3a best demonstrated in B (`Re=1e12`); F3b uses pipe Bessel (A/B) or Stokes mode (all). Energy norm: A/C use the spectral mass-matrix / quadrature inner product (curvilinear `r`-weight for TC/pipe); B uses `intrdr`/`inty` quadrature.

### F4 — Discrete symmetries / invariances of each geometry

**Description & rationale.** Each geometry has exact discrete symmetries the solver must preserve (the equivariance of Navier–Stokes under the geometry's isometry group). Breaking one signals a sign error or an asymmetric stencil/projection.

**Per-geometry setup & oracle.**

**F4a — Channel/PCF: reflection & shift symmetries.** Equivariant under spanwise reflection `z→−z`, streamwise translation, and the Couette point-symmetry `(x,y,z,u)→(−x,−y,−z,−u)`.
- C: mean `(0,0)` modes forced real after a step, `imag < 1e-12` (`test_pcf_fluctuations_jax.py:56-72`). B: `enforce_mean_mode_cleanup` keeps mean `v=0`, mean `u,w` real (§1.B.2). Assert symmetry residual `< 1e-12`.

**F4b — Taylor–Couette: `m↔−m` conjugation & axial-shift.** A real field satisfies `q(−m)=conj(q(m))`.
- A golden: `test_hydro_nonaxisymmetric_mirror_symmetry` to `atol=1e-9`; assert `max|q(−m) − conj(q(m))| < 1e-9`. C evaluates the complex eigenvector real/imag separately to preserve this (§1.C.2).

**F4c — Pipe: Hermitian/parity & rotational symmetry.** Real field ⇒ `f(−k,0)=conj(f(k,0))`, `f(0,0)` real; azimuthal parity folding at `r=0`; optional `m_p`-fold rotation.
- B golden: `enforce_m0_reality` keeps `f(−k,0)=conj(f(k,0))`, `f(0,0)` real, `m0_hermitian_residual < ` roundoff; the discrete symmetries `mirror_z`, `shift_reflect`, `shift_rotate` (keeps `k+m` even) are ported verbatim from OpenPipeFlow (§1.B.4). Assert Hermitian residual `< 1e-12`.

**Metric & tolerance.** Within-family roundoff (symmetry is an exact equivariance of the discrete operators when correctly implemented): `1e-9` for TC `m↔−m` (accumulated arithmetic in dense per-mode blocks), `1e-12` for the FFT-based reality conditions.

**Families.** All three, per geometry. A/C impose reality on the spectral coefficients; B imposes it on the FFT layout. The symmetry *group* differs by geometry — test the geometry-appropriate generator set.

### F5 — Orr–Sommerfeld / Squire linear eigenvalue benchmark

**Description & rationale.** The least-stable eigenvalue of the Orr–Sommerfeld (and coupled Squire) operator is *the* canonical wall-bounded linear-stability benchmark with published golden values; it validates the wall-normal differentiation operators, the clamped BCs, and the eigensolver simultaneously.

**Setup.** Channel, Poiseuille base `U=1−x²`. Operating points:
- **Orszag point:** `Re=10000, α=1, β=0` (or `Re=8000, α=1` for A's golden). Clamped OS BC `v=v'=0` at both walls; Squire `η=0`.
- A `N>80` (Chebyshev biharmonic, `quad='GC'`); B `N=101, KL=4`; C analogous.
- Family A: `OrrSommerfeld_eigs.py` Shen biharmonic basis; eigenvalues ranked by descending `Im`. Family B: `orr_sommerfeld_squire` via `scipy.linalg.eig` on FD matrices `D=W_dy1, D2=W_dy2`; `c=iλ/α`. Family C: jaxfun `inner`-assembled OS operator → dense GEP via `generalized_eig`.

**Oracle (verbatim golden eigenvalues).**
- **A, `Re=8000, α=1`:** leading `c = 0.24707506017508621 + 0.0026644103710965817 i` (positive Im → unstable), tolerance `1e-12` (`OrrSommerfeld_eigs.py:183-184`).
- **B, `Re=10000, α=1, β=0`:** reference `c_ref = 0.23752649 + 0.00373967 i`; computed `0.23752722198590992 + 0.0037381198835812705 i`, abs error `< 1e-4` (`test_linstab_poiseuille.py:7-19`).
- **Published critical Re (Poiseuille):** `Re_crit = 5772.22`, `α_crit ≈ 1.02056`, `c ≈ 0.26400` [Orszag71]. B verifies the sign change: stable at `Re=5742.22, α=1.02`, unstable at `Re=5802.22` (`test_linstab_poiseuille.py:22-39`).
- **Plane Couette:** linearly stable for **all Re** — no unstable OS eigenvalue (Romanov 1973, `couette_linear_benchmarks.md:84-88`).

**Metric & tolerance.**
- *Within-family golden:* A `|c − golden| < 1e-12`; B `|c − c_ref| < 1e-4`. Family-specific because the published reference itself was computed by one method.
- *Cross-family / published oracle:* assert each family's neutral curve crosses `Re(λ)=0` at `Re=5772.22 ± 30` (`rel<5e-3`); B's flanking points (5742/5802) bracket it within ±30.
- *Couette stability:* assert `max Re(λ) < 0` for all tested `(α,β)`. Golden Romanov rates (`couette_linear_benchmarks.md:111-118`): `ky=1,kz=0: −1.179054e-01`; `ky=0,kz=1: −3.467401e-03`; `ky=2,kz=1: −1.905757e-01`. Streamwise-roll analytic oracle `s = −ν(kz² + (π/2)²)`: `kz=1.0: numeric −3.46740110e-3, analytic −3.46740110e-3` (rel.err `5e-10`, `couette_linear_benchmarks.md:117`; `test_couette_linear.py:90-104`) — a *closed-form* check held to `rel<1e-9` because both sides are exact.

**Tolerance rationale.** The within-family golden is asserted at the precision the family achieves (A spectral `1e-12`; B FD `1e-4`, limited by the 9-point stencil truncation). The cross-family `Re_crit=5772.22` uses `rel<1e-2` because the three families have different spatial discretizations and the eigenvalue is an analytic-physics oracle, not an operator identity. The streamwise-roll `s=−ν(kz²+(π/2)²)` is closed-form on both sides → `rel<1e-9`.

**Families.** All three (channel/Couette). A uses Shen biharmonic Galerkin `Aφ = c Bφ`; B uses OS/Squire primitive FD blocks with clamped rows; C uses jaxfun assembly. The Squire coupling `−iβU'` must be present for 3D (`β≠0`). Rank eigenvalues by descending Im; filter spurious infinite (tau) eigenvalues via `FINITE_CAP` (A `1e8`, `test_couette_linear.py:166`; C `1e6`; B homogeneous-denominator rejection).

### F6 — 2D Taylor–Green decaying-mode (spatial + temporal accuracy at once)

**Description & rationale.** The 2D Taylor–Green vortex is an *exact unsteady* solution of incompressible NS on a doubly-periodic box: pure exponential viscous decay with no change in spatial structure. Because the spatial modes are exactly representable in a Fourier basis, the only error is temporal — this isolates **temporal order** while confirming **spatial exactness** and divergence-free decay in one test. **Current implementation status:** repository search found no target-family TGV harness; this section is the required test specification.

**Setup.** Doubly-periodic `[0,2π]²` (or `[−π,π]²`), `ν=1/Re`. IC = exact TG field at `t=0`, integrate to `t=T`, compare. Resolution modest (`16×16` Fourier — TG modes exactly resolved). `dt` ladder `{Δt, Δt/2, Δt/4}` for order extraction.

**Oracle (verbatim formula, [TaylorGreen37]).**

```
u(x,y,t) =  sin(x) cos(y) · F(t)
v(x,y,t) = -cos(x) sin(y) · F(t)
p(x,y,t) = (ρ/4)·(cos(2x) + cos(2y)) · F(t)²
F(t) = exp(-2 ν t)                       # velocity decay factor
```

Kinetic energy decays as `E(t) = E(0)·exp(−4νt)` (twice the velocity rate, `E ∝ |u|²`). Concrete: `ν=0.01, k=1`: `F(t)=e^{−0.02t}`; at `t=10`, `F=e^{−0.2}=0.818731`, energy ratio `E(10)/E(0)=e^{−0.4}=0.670320`.

**Metric & tolerance.**
- *Field error:* `max|u_num − u_exact|` at `t=T`, in the band `max(C·Δt^q, ε_spectral)` (spatial exact for Fourier).
- *Energy-decay oracle:* `|E(T)/E(0) − e^{−4νT}|/e^{−4νT} <` the same band (coarse `dt` → `rel<1e-2`; fine `dt` → roundoff).
- *Divergence:* `‖∇·u‖ <` roundoff throughout (F2 floor).
- **Temporal-order slope (headline deliverable):** run `{Δt, Δt/2, Δt/4}`, fit `log‖error‖` vs `log Δt`; assert slope `q ≈` the scheme's formal order: **q=2** for CNAB2 (A-TC/B-couette/C-TC) and IMEXRK222 (A-PCF/C-PCF); **q=1** for B's θ-method (`implicit=0.51`, formally 1st-order, §1.B.2). Slope tolerance `|q_measured − q_formal| < 0.15`.

**Tolerance rationale.** The TG field error has NO spatial truncation contribution for Fourier families → the band collapses to `max(C·Δt^q, ε_spectral)`, making it a pure temporal-accuracy probe. The energy rate is an analytic oracle (`rel<1e-2` at usable `dt`). The slope test catches a temporal-order regression (e.g. a first-order bootstrap step polluting a second-order scheme).

**Families.** All three should add this as a periodic-box verification. **Adaptation for wall-bounded codes:** run TG in the two periodic directions with a trivial wall-normal dependence, or preferably as a standalone Fourier-only verification (B exposes `torch.fft`; A/C have Fourier `FunctionSpace`). C's jaxfun has direct Fourier spaces and `RK4`/`ETDRK4`/`IMEXRK` integrators (§1.C.5) ideal for a clean 2D-TG harness. Do not cite external spectralDNS/TGV examples as current coverage for these target families; they can only be implementation patterns. **Per-family `q`:** A-PCF/C-PCF IMEXRK222 (q=2), A-TC/C-TC CNAB2 (q=2), B θ-method (q=1) — assert the *family-correct* `q`, never q=2 for B's default θ-stepper. The 3D TG (`Re=1600`, peak dissipation vs time, [BMO83]) is the natural validation-run extension, out of scope for the foundational floor.

### F7 — Cross-cutting conservation diagnostics (attached to every test)

Not a standalone test but a mandatory invariant harness running inside `F1`–`F6`: (1) divergence-free (F2) always; (2) energy/enstrophy budget (F3) — inviscid → roundoff, viscous → matches resolved dissipation; (3) symmetry residual (F4) — geometry-appropriate generators. A regression in any invariant fails the specific physics test that triggered it.

### F8 — Method: extracting convergence order (the shared recipe)

Used by F1e (temporal), F3b (decay rate), F5 (eigenvalue refinement), F6 (temporal slope), and reused by IV.2/IV.3.

**Spatial order.** Refine `N`. *Spectral (A,C):* straight line on **semilog** (`error ~ e^{−cN}`); assert `error < 1e-10` at modest `N` and that it drops `≥2` decades per resolution doubling until the floor. *FD (B):* straight line on **log-log** (`error ~ N^{−p}`), slope `=p`; the 9-point stencil gives interior `p≈8` (`test_mesh.py` polynomial-exact through degree 8), boundary one-sided lower; assert fitted slope `≥ 4` (the "4th-order FD" floor).

**Temporal order.** Fix `N` fine (spatial error ≪ temporal), run `{Δt, Δt/2, Δt/4}`. Observed order `q = log[(u_{Δt}−u_{Δt/2})/(u_{Δt/2}−u_{Δt/4})]/log 2` (three-level Richardson, no oracle) OR the `log‖u−u_exact‖` vs `log Δt` slope when an oracle exists (F6). Assert `|q − q_formal| < 0.15`.

**Eigenvalue refinement (F5).** Increase `N`, confirm the leading eigenvalue converges to the golden monotonically. The Family A MRI pattern (`couette_linear_benchmarks.md:314-316`): `nx=24 → 0.4984075630441907`, `nx=32 → 0.49840694616677383`, `nx=48 → 0.49840620435392047`, converging toward the local theory `s=0.5` (for `Ω=2/3`); extract the Cauchy-difference ratio to confirm spectral convergence.

**Tolerance-band assembly (cross-family).** For any cross-family comparison the band is `max(C·Δx^p, C·Δt^q, ε_spectral)` evaluated on the *actual* grid/`dt` of the coarsest family (almost always B, FD + θ-method): at `Re=500, N~33, dt~0.01`, dominated by `C·Δt^1 ~ 1e-2`, with `C·Δx^8` small → cross-family tolerance ≈ `1e-2`. Within-family identities (F2, F4) stay at roundoff. **Never** use roundoff for a cross-family comparison.

### IV.1 coverage matrix

| Test | A (shenfun) | B (torch) | C (jax) | Notes |
|---|---|---|---|---|
| F1a Couette `U=σx` | ✓ | ✓ | ✓ | base sign `+U_wall·x` (A/C), `U=y` (B) |
| F1b Poiseuille `1−x²` | ✓ | ✓ | ✓ | B uses const-flux `4/3` |
| F1c TC `Ar+B/r` | ✓ | ✓ | ✓ | explicit-`1/r` (A/C) vs cyl-FD (B) |
| F1d Hagen–Poiseuille | ✓ | ✓ | **skip** | C has no pipe |
| F1e Womersley | ✓ | **skip** | **skip** | only A implements pulsatile |
| F2 div(u), div(B) | ✓ | ✓ | ✓ | per-family roundoff; MHD where present |
| F3 energy balance | ✓ | ✓ | ✓ | F3a best in B (`Re=1e12`) |
| F4 symmetries | ✓ | ✓ | ✓ | geometry-specific generators |
| F5 OS/Squire eig | ✓ | ✓ | ✓ | A golden `1e-12`; B golden `1e-4`; `Re_c=5772.22` |
| F6 2D Taylor–Green | **add** | **add** | **add** | required/proposed; no target-tree harness found |

**Floor rows (must exist in all three after closure):** F1a, F1b, F1c, F2, F3, F4, F5, F6. F6 must be added in all three; F3/F4 should be hardened where they are currently only partial or geometry-specific.

---

## IV.2 Temporal discretization-order tests (`T1`–`T4`)

The integrators differ by construction (IMEXRK111/222/3/443 + CNAB2 in A/C, §1.A.6/§1.C.5; single-stage θ-method PC at `implicit=0.51` in B, §1.B.2), so these tests are *tailored per family* — that tailoring is the point. The shared procedure follows the established slope-extraction pattern: halve `dt` repeatedly on a problem with an analytic time solution, at fixed high spatial resolution so spatial error is negligible, and extract the `log(error)` vs `log(dt)` slope — the direct analogue of the shearpy RDT slope test (`fd_proto/test_rdt.py:12-26`, which fits `time_order = log(coarse_t/fine_t)/log 4` and asserts `2.8 ≤ time_order ≤ 3.4`).

### TEST T1 — Single decaying Fourier mode: integrator temporal order *(all families)*

**Description & rationale.** The cleanest temporal oracle is one Fourier mode under linear diffusion: `∂u/∂t = ν ∂²u/∂x²` for `u=û(t)e^{ikx}` gives `dû/dt = −νk² û`, exact `û(t)=û_0 e^{−νk²t}`. This isolates the **implicit** branch of every IMEX integrator (diffusion is the implicitly-treated stiff term: shenfun `linear_op=ν·div(grad)` `ChannelFlow.py:151`; torch `L_lhs=(1/dt)I − θ(1/Re)L` `solver.py:215`; jax `M − γ·dt·L` `base.py:222`). A single wavenumber ⇒ no nonlinear/aliasing contamination and (in a Fourier direction) zero spatial error, so the measured error is purely time-integration error. This is the temporal-accuracy form of the shearpy diffusion oracle (`test_theory.py:66-97`, which asserts `E(t)=E0·exp(−2νk²t)` to `rel=1e-5, abs=1e-8`).

**Setup.** Periodic 1-D (or a periodic direction of the 3-D box with all other modes zeroed). `k=1` (mildest stiffness), `ν=0.01` (`λ=−0.01`). IC `û(0)=1` in mode `k=1`. Integrate to `T=1.0`. `N_x=16` Fourier; for families needing a wall-normal direction, put a smooth Chebyshev/FD field with `N_y≥32` and **disable** the nonlinear term (`nonlinear_form="none"` torch; convection→0 shenfun). `dt ∈ {1/20, 1/40, 1/80, 1/160, 1/320}` (5 halvings; `T/dt` integer at every level).

**Oracle.** `û(T) = e^{−νk²T} = e^{−0.01} = 0.990049833749168…`; `E(dt) = |û_num(T) − e^{−0.01}|`.

**Metric & tolerance (per integrator).** Fit `log E = p·log dt + c` (least squares, 5 points); assert `p` in the per-integrator band:

| Integrator | Family | Formal order | Pass band for `p` | Source |
|---|---|---|---|---|
| IMEXRK111 | A, C | 1 | `[0.8, 1.3]` | `integrators.py:836`; `imex_rk.py:163` |
| CNAB2 | A (TC/pipe), C (TC) | 2 | `[1.8, 2.3]` | `taylor_couette_dns.py:288`; `cnab2.py` |
| torch θ-method PC (θ=0.51) | B | ~1 (see note) | `[0.85, 1.6]` | `solver.py:87, 594-606` |
| IMEXRK222 | A, C | 2 | `[1.8, 2.3]` | `integrators.py:858`; `imex_rk.py` |
| IMEXRK3 (Spalart) | A, C | 3 | `[2.7, 3.3]` | `integrators.py:665`; `imex_rk.py:96` |
| IMEXRK443 | A, C | 3 | `[2.7, 3.3]` | `integrators.py:872`; `imex_rk.py:184` |

**Tolerance rationale.** These are within-family order-verification bands, not roundoff: a 5-point log-log fit on a clean linear ODE recovers the asymptotic slope to ≈±0.15 once in the asymptotic regime, so a ±0.3 half-width accepts the true order while rejecting an off-by-one coding error (e.g. a CNAB2 silently degraded to first order — the classic "CN paired with AB1/Euler" failure). The band is widened on the low side for 2nd/3rd-order schemes because on a pure linear problem the explicit branch is inactive and the implicit DIRK can show its *higher* stage order — guard the top with +0.3.

**θ-method note.** For a scalar linear ODE the θ-method `û_{n+1} = û_n (1+(1−θ)λdt)/(1−θλdt)` is exactly 2nd-order **only at θ=0.5**; at the shipped `θ=0.51` it carries an `O((θ−0.5)·λ·dt)` first-order term, small over this window so the empirical slope often reads ≈2 until the first-order term dominates at the coarsest `dt` — hence the wide `[0.85, 1.6]`. **Adaptation:** run a second variant with `θ=0.5` exactly and assert `p ∈ [1.8, 2.3]`, confirming the integrator IS Crank–Nicolson when asked and isolating the deliberate 0.51 damping bias. Document both numbers.

**Families.** A: `KMM`/PCF subclass, `--timestepper ∈ {IMEXRK111,222,3,443}`, seed mode `k=1` (the `(0,0)` mean path is untouched). C: same tableaux (verified `imex_rk.py` γ=(2−√2)/2, δ=1−1/(2γ) match A), `channelflow_kmm` with nonlinear→0, x64 default. B: `ChannelSolver` with `nonlinearity_form="none"`, single seeded mode; run both θ=0.51 and θ=0.5.

**Extraction.** Least-squares slope + successive pairwise orders `p_i = log(E_i/E_{i+1})/log 2` to confirm monotone approach to the asymptote.

### TEST T2 — Epicyclic oscillation: temporal order of the rotation/shear coupling *(A, C; B gated)*

**Description & rationale.** T1 exercises only the implicit branch. To verify the **explicit** branch carrying the rotation/shear source terms, use the epicyclic oscillation. A uniform (`k=0`) perturbation with Coriolis + base-shear obeys `du_x/dt = 2Ω u_y`, `du_y/dt = (S−2Ω)u_x` (`pcf_mhd_mri_shearpy.py:12-15`), giving `d²u_x/dt² = −κ²u_x` with `κ² = 2Ω(2Ω−S)`. Exact `u_x(t)=u_{x0}cos(κt)`, `u_y(t)=((S−2Ω)/κ)u_{x0}sin(κt)`. This is byte-for-byte the first-milestone assertion (`PLAN_openpipeflow_vs_fnshenfun.md:191-202`), itself a port of shearpy `test_theory.py:100-132`.

**Setup.** Shearing box. Keplerian `Ω=2/3, S=1` ⇒ `q=3/2`, `κ²=2·(2/3)·(1/3)=4/9` ⇒ `κ=2/3` (= Ω, the Keplerian result). Also a non-Keplerian case `Ω=1, S=1` ⇒ `κ²=2`, `κ=√2`. Seed only the `k=0` mode: `u_{x0}=1e-3`, `u_y=u_z=0`. **Lorentz/magnetic OFF.** `ν=0` (pure oscillator) or `ν=1e-6` (the `k=0` mode has no diffusion anyway). Integrate to `T=3·(2π/κ)=9π≈28.274` (three periods). Minimal grid (`N=(8,8,8)`). `dt ∈ {T/200, T/400, T/800, T/1600, T/3200}`.

> **Shearpy reference parametrization.** The shearpy theory test (`test_theory.py:100-132`) uses `omega=1.0, shear=1.5` ⇒ `κ=√(2·1·(2−1.5))=√1=1`, amplitude `(shear−2ω)/κ=(1.5−2)/1=−0.5`, integrating to `T=1.0`, asserting `vx_mean=cos(κT)`, `vy_mean=((S−2Ω)/κ)sin(κT)` to `rel=1e-2, abs=5e-4`. Use this exact case as the cross-port regression; use the Keplerian `Ω=2/3, S=1` case for the family-canonical run.

**Oracle.** `u_x(T)=u_{x0}cos(κT)`. For Keplerian, `κT=6π` ⇒ `u_x(T)=u_{x0}` exactly. Use the whole-trajectory L2 phase error (more sensitive than the endpoint, which sits at a node): `E(dt)=√(Σ_n[(u_x^{num}(t_n)−u_{x0}cos κt_n)² + (u_y^{num}(t_n)−((S−2Ω)/κ)u_{x0}sin κt_n)²]·dt)`.

**Metric & tolerance.** Slope of `log E` vs `log dt`: IMEXRK222 `p ∈ [1.8, 2.3]`; IMEXRK3/443 `[2.7, 3.3]`; IMEXRK111 `[0.8, 1.3]`. Plus a physics-oracle check at the finest `dt`: `|κ_measured − κ_theory|/κ_theory < 1e-2`, `κ_measured` from the zero-crossings / FFT of `u_x(t)` — mirroring the shearpy `rel<1e-2, abs<5e-4` tolerance.

**Tolerance rationale.** The explicit branch carries the source terms, so this catches a mis-wired Coriolis sign or an order-reduced explicit tableau that T1 cannot see. The κ `rel<1e-2` is a physics tolerance asserting the integrator reproduces the correct oscillation frequency independent of per-step truncation. Order band ±0.3 for the same 5-point-fit reason as T1.

**Families.** A: native — `pcf_mhd_mri_shearpy.py:346-348` Coriolis, base shear `dUb_dx=−S`; set magnetic amplitude 0. C: native — `pcf_mhd_mri_shearpy_jax.py:130-134` adds `n_0 −= 2Ω u_1; n_1 += 2Ω u_0`; same κ². **B: CONDITIONAL / GATED** — MRI rotation/shear are *metadata-only* (`mhd.py:71-74`: "do not add Coriolis or background-shear source terms"). T2 **cannot run on stock torch.** Either (i) `pytest.skip`/`xfail` citing `mhd.py:71-74` until the planned wiring lands (Coriolis as a sibling of `_base_coupling_terms`, `PLAN_openpipeflow_vs_fnshenfun.md:125-137`), or (ii) run it as the *first acceptance test* of that new wiring. The harness should detect a `rotation_active` flag and skip otherwise — that gating IS the per-family adaptation. (See SR-1.)

### TEST T3 — Ohmic decay: temporal order of the resistive (magnetic-implicit) branch *(A, C; B with conducting-wall caveat)*

**Description & rationale.** Resistive diffusion is treated implicitly in all MHD families (shenfun on the vector potential `SA = MA − dt·γ·η·LA`; torch magnetic Helmholtz with `1/Rm`, `mhd.py:115-117`). A single magnetic Fourier mode with no flow decays purely resistively: `E_mag(t)=E_0 e^{−2ηk²t}` (amplitude `∝ e^{−ηk²t}`) — the shearpy Ohmic oracle (`test_theory.py:66-97`). Verifies the temporal order and effective numerical resistivity of the magnetic branch independent of the velocity branch.

**Setup.** Single magnetic mode `k=1` in a periodic direction, `b̂(0)=1`; velocity held zero (no EMF — use the fluctuation form so the base flow contributes none, or set base flow off). `η=1/Rm`, `Rm=100` ⇒ `η=0.01`, `λ_mag=−0.01`. `T=1.0`; `dt` ladder `{1/20,…,1/320}`. Lorentz OFF.

**Oracle.** `E_mag(T)/E_mag(0)=e^{−2ηk²T}=e^{−0.02}=0.980198…`; amplitude `b̂(T)=e^{−0.01}=0.990049833…`; `E(dt)=|b̂_num(T) − e^{−0.01}|`.

**Metric & tolerance.** Slope `p`: IMEXRK222 `[1.8, 2.3]`; CNAB2 (TC MHD A/C) `[1.8, 2.3]`; IMEXRK3/443 `[2.7, 3.3]`; torch θ-method `[0.85, 1.6]` (θ=0.51), `[1.8, 2.3]` (θ=0.5 override). Plus physics check `|η_measured − 0.01|/0.01 < 1e-2`, `η_measured = −log(b̂(T))/(k²T)`.

**Tolerance rationale.** Identical logic to T1 on the magnetic implicit branch — catches a resistivity wired with the wrong sign or a magnetic stepper fallen to first order. The `η_measured` physics tolerance directly asserts the discrete scheme reproduces the analytic resistive decay rate.

**Families.** A: `pcf_mhd_divfree.py` (vector potential, η implicit). C: `pcf_mhd_jax.py` analog. Both Alfvén-unit, Lorentz prefactor 1. B: runs (induction + magnetic diffusion implemented), but (i) Lorentz prefactor `Ha²/(Re·Rm)` channel / `Ha²/Pm` couette — set `Ha=0` so Lorentz=0; (ii) magnetic walls are homogeneous `b=0` Dirichlet only, so put the decaying mode in the **periodic (Fourier) direction**, not wall-normal, to avoid wall-BC subtleties. (T3 is one of the two MHD tests that CAN run on torch today — turn it on as a torch-MHD regression; see SR-3.)

### TEST T4 — IMEX splitting-error order-reduction probe *(A, C)*

**Description & rationale.** When implicit (diffusion) and explicit (advection/source) terms are both active and stiff, an IMEX scheme can show **observed order < formal order** — the classic IMEX order-reduction pathology [ARS97]. T1–T3 each isolate one branch and cannot detect splitting error; T4 deliberately activates both via a manufactured oracle with nontrivial implicit AND explicit content.

**Setup (MMS).** Advection–diffusion of a single mode with forcing: `∂u/∂t = ν ∂²u/∂x² − c ∂u/∂x + Q(x,t)`, target `u_exact = sin(kx − ωt)·e^{−αt}`. Compute `Q = ∂_t u_exact − ν∂_{xx}u_exact + c∂_x u_exact` symbolically (sympy), inject as forcing. Treat `ν∂²/∂x²` implicitly, `−c∂u/∂x + Q` explicitly. `k=2, ω=1, α=0.1, ν=0.05, c=1.0`, periodic, `N_x=32`, `T=1.0`, `dt ∈ {1/40,…,1/640}` (start finer than T1; both active branches push the asymptotic regime to smaller `dt`).

**Oracle.** `u_exact(x,T)` pointwise; `E(dt) = ‖u_num(·,T) − u_exact(·,T)‖_{L2}`.

**Metric & tolerance.** Slope `p`. **Order-reduction assertion:** `p ≥ formal_order − 0.5` (IMEXRK222 ⇒ `p≥1.5`; IMEXRK3 ⇒ `p≥2.5`). The `−0.5` margin tolerates mild splitting-induced reduction while rejecting a full integer drop. Report measured `p` as data — a value at `formal−1` is a red flag to investigate. For ARS schemes with stiff explicit content, observed `p` between `formal−0.5` and `formal` is **expected and acceptable**, not a failure.

**Tolerance rationale.** Unlike T1–T3 (clean single-branch ⇒ tight ±0.3), T4 must budget for the known reduction mechanism, hence the one-sided `≥ formal−0.5` — the operationally honest tolerance.

**Families.** A and C only (native multi-stage IMEX-RK with controllable split). B's θ-method is single-stage with no distinct explicit-stage order to reduce, so **T4 is N/A for B** (its order is fully characterized by T1). Document N/A explicitly.

**Extraction.** Log-log slope + per-pair orders to distinguish asymptotic reduction (constant `p<formal`) from pre-asymptotic behaviour (`p` rising toward formal as `dt→0` — refine further, not a true reduction).

---

## IV.3 Spatial discretization-order tests (`S1`–`S3`)

The procedure differs by family because the bases differ. Spectral families (A,C) converge **exponentially** for smooth fields [Boyd01, CHQZ06] (demonstrated in jaxfun's own MMS self-asserts `poisson1D.py:46 error < ulp(1000)`); the FD family (B) converges **algebraically** at a fixed order. You cannot apply one acceptance criterion to both — asserting an algebraic slope on a spectral method would reject a correct method, and vice versa. Each test uses the Method of Manufactured Solutions [SalariKnupp00].

### TEST S1 — Spectral exponential convergence on a smooth MMS *(A, C)*

**Description & rationale.** For shenfun (A) and jaxfun (C) a smooth manufactured solution must converge faster than any algebraic rate — error dropping `>10×` per fixed-`N` increment until the roundoff floor, the defining property of spectral Galerkin [Mortensen18, Shen94, Shen95].

**Setup (MMS).** Solve the wall-normal Helmholtz the channel solvers invert: `α u − ν u'' = f` on `x∈[−1,1]` with the solver's own bases. Manufactured `u_exact(x)=(1−x²)cos(3x)` — smooth, satisfies the Dirichlet wall BC `u(±1)=0` exactly (lives in the Shen Dirichlet basis `bc=(0,0)`). For the biharmonic/clamped basis (`bc=(0,0,0,0)`) use `u_exact=(1−x²)²cos(3x)` (satisfies `u=u'=0` at walls). Compute `f = α u_exact − ν u_exact''` symbolically, project, solve. `α=10, ν=1`. `N_x ∈ {8,12,16,20,24,28,32}` (increments of 4). All other directions trivial.

**Oracle.** `u_exact(x)`; `E(N)=‖u_num − u_exact‖_{L2}` in the family's quadrature.

**Metric & tolerance.** Primary (exponential): for consecutive `N` until the floor, `E(N+4) < E(N)/10`; equivalently fit `log E = −c·N + b` on a **semilog** axis and assert decay constant `c > 0.5`. Floor handling: stop the `>10×` chain once `E(N) < ε_spectral` with `ε_spectral = 1e-11` (a few orders above float64 `ε≈2.2e-16`, accounting for Helmholtz-inverse conditioning and quadrature; consistent with jaxfun's `error < ulp(1000)` self-asserts at `poisson1D.py:46`). Assert the floor is reached by `N=32` and stays flat.

**Tolerance rationale.** The `>10×/+4N` criterion is the operational definition of "spectral" — unattainable by any fixed-order FD scheme (a 4th-order scheme would need a `1.78×` `N` increase, not `+4`, to drop `10×`). `ε_spectral=1e-11` is not roundoff; it is the realistic spectral floor for a float64 Helmholtz solve, set ~5 orders above machine-ε to avoid false failures from conditioning while still proving the method bottoms out at spectral precision.

**Families.** A: native (`chebyshev.la.Helmholtz`/`Biharmonic` or `la.SolverGeneric1ND`); the OS golden `c=0.24707506017508621+0.0026644103710965817j` at `Re=8000` to `1e-12` (`OrrSommerfeld_eigs.py:183`) is the eigenvalue analog. C: native (`la/solvers.py` `Helmholtz`/`Biharmonic`; jaxfun examples `poisson1D.py`, `helmholtz1D.py` ARE this test). B: N/A — FD, see S2.

**Extraction.** Decay constant `c` from the semilog fit; per-increment ratios `E(N)/E(N+4)` to confirm `>10×` until the floor.

### TEST S2 — Algebraic ~4th-order convergence + saturation ceiling on FD MMS *(B)*

**Description & rationale.** For torch (B) the wall-normal/radial direction is finite-difference with a Taylor/Vandermonde stencil. The family is labeled "4th-order FD," but the default `KL=4` 9-point centered stencil is formally **8th-order interior** (`mesh.py:169`; §0.2 contradiction #2; polynomial-exact through degree 8, `err_dy<1e-7` for degrees 0..8, `test_mesh.py:21-37`; pipe `mes_weights` "1st/2nd derivatives to 8th/7th order" [Willis17]). So the test asserts algebraic order with `slope ≥ 3.7` as the **minimum floor**, while documenting that interior order is higher and that **boundary stencils are one-sided and lower-order**, which often sets the realized global order.

**Setup (MMS).** Apply the FD derivative matrices `W_dy1, W_dy2` (`mesh.py:186-189`) to a smooth field and measure derivative error vs analytic — the operator-level test mirroring `test_mesh.py`. Field `u_exact(y)=cos(3y)·e^{0.5y}` on `y∈[−1,1]` (smooth, non-polynomial so it does NOT hit the exact-to-degree-8 trap). `u'_exact = (−3 sin 3y + 0.5 cos 3y)e^{0.5y}`; `u''_exact` analytic. `N_y ∈ {17,33,65,129,257}` (doubling) on the Chebyshev-extrema mesh `y=−cos(πj/(N−1))` (`mesh.py:80-100`). `KL=4`. `E(N)=‖W_dy1·u_exact − u'_exact‖_∞` (and `W_dy2` separately). Pure spatial-operator test (no time stepping).

**Oracle.** `u'_exact(y)`, `u''_exact(y)` analytic.

**Metric & tolerance.** Fit `log E = −p·log N + b` (log-log); assert `p ≥ 3.7` for the first-derivative operator (the documented FD floor; the clustered Chebyshev-extrema mesh makes the boundary-limited global order land between 4 and the 8th interior order — assert the conservative floor). Document measured `p` (often 4–6) as data. **Saturation/roundoff ceiling:** near-wall spacing `h_min ∝ 1/N²` makes the dense FD matrix ill-conditioned at large `N`, so error stops decreasing (or rises) once differentiation roundoff `κ(W_dy1)·ε` exceeds truncation. Assert the algebraic slope holds over `N ∈ {17,33,65}` (`p≥3.7` on this sub-range), and allow `E(257) ≥ E(129)` (do NOT assert monotone decrease past the ceiling). Report the floor `N` and magnitude (`E≈1e-9…1e-11` for `W_dy1`; higher for `W_dy2`, which squares the conditioning, `W_dy2=W_dy1@W_dy1` `mesh.py:189`).

**Tolerance rationale.** `p≥3.7` is the documented FD floor — it rejects a stencil silently fallen to 2nd order (e.g. a boundary-row bug) while accepting the true (≥4, often 8 interior) order. The saturation allowance is physically necessary: clustered-mesh dense FD has a finite roundoff floor `κ·ε`, and asserting monotone convergence past it would falsely fail a correct implementation. **Cross-port anchor:** the shearpy RDT prototype on its compact-FD path asserts `space_order ≥ 5.5` (`fd_proto/test_rdt.py:20-26`), consistent with the high-interior-order behaviour expected of B's stencil family.

**Families.** B: native — channel `W_dy1/W_dy2`, couette `W_dr1/W_radlap`, pipe banded `W_dr1` (banded LU but same Taylor weights `mesh.py:26-57`). For the pipe, the banded and dense builds are verified equal to `1e-12` (`test_banded.py`), so run S2 on the dense path; couette `W_dr2` is an independent stencil (not squared) — test both `W_dr1` and `W_dr2`. A, C: N/A — spectral, see S1.

### TEST S3 — Cross-family floor-meeting (parity) *(A↔B↔C)*

**Description & rationale.** This is *the operational definition of cross-family parity*: at torch's converged resolution, a shared physical observable must sit within **torch's own truncation band** of the spectral (effectively exact) value from A/C. Spectral is the oracle (`PLAN_openpipeflow_vs_fnshenfun.md:110-111`: "shenfun is the oracle for linear eigenvalues … spectral ⇒ effectively exact at modest N"). The cross-family tolerance is **never roundoff** — it is `max(C·Δx_FD^p, C·Δt_FD^q, ε_spectral)` (`PLAN_openpipeflow_vs_fnshenfun.md:57-61`).

**Setup.** Two concrete instances.

**S3a — Orr–Sommerfeld leading eigenvalue (channel, hydro).** Plane Poiseuille, `Re=10000, α=1, β=0`. Spectral oracle (A/C): the OS leading eigenvalue recomputed at high `N` (`N=128` Chebyshev, converged ~`1e-10`). Torch (B) computes it via `orr_sommerfeld_squire` on its FD matrices; torch golden `c = 0.23752649 + 0.00373967j` at `N=101, KL=4` (`test_linstab_poiseuille.py:7-19`). B resolution sweep `N_y ∈ {65,101,151,201}`. (Note `Re=8000` is the *different* case carrying A's `c=0.24707506…` golden — recompute the spectral oracle at the `Re=10000` operating point.)

**S3b — Epicyclic frequency κ or MRI growth `s_max`** (MHD/rotation, A↔C; B only if rotation wired). Ideal local Keplerian oracle `s_max=0.75Ω`, `(k v_A)²=(15/16)Ω²`; the local-MRI computation gives `s_max/Ω = 0.7499999944199642`, `(k v_A)²/Ω² = 0.9373170323757943` (`couette_linear_benchmarks.md:313`). A/C reproduce to `rel~1e-3` (`test_taylor_couette.py`: `|s_max/Ω − 0.75| < 2e-3`).

**Oracle.** S3a: `c_spectral` from A/C at `N=128`. S3b: `s_max/Ω = 0.75` analytic + A/C spectral confirmation.

**Metric & tolerance (the truncation band).** At B's *converged* resolution (where its `N`-refinement has plateaued — confirm `|c_B(201) − c_B(151)|` is below B's truncation estimate), assert `|c_B − c_spectral| ≤ max(C_x·h_FD^p, C_t·dt^q, ε_spectral)`. For S3a (steady eigenproblem, no `dt`): the band is `C_x·h_FD^p` with `p≈4` (conservative FD order from S2). Concretely, with B's golden differing from the spectral value by `~1e-3…1e-4`, set the band to `max(1e-3, ε_spectral)` and assert `|c_B − c_spectral| < 1e-3` (a tightened physics-oracle tolerance because OS is a well-conditioned eigenvalue both families resolve). **Crucially, do NOT use roundoff (`1e-12`) here** — that would falsely fail B (4th–8th-order FD, not spectral). For S3b: `|s_max^B/Ω − 0.75| < 1e-2`, **gated on rotation being active in B** (metadata-only otherwise, `mhd.py:71-74`; skip/xfail).

**Tolerance rationale.** The band `max(C·h^p, C·dt^q, ε_spectral)` is derived from B's *actual* grid, not chosen arbitrarily: at `N=101, KL=4` on the clustered mesh, the FD truncation error in a leading OS eigenvalue is empirically `~1e-3…1e-4`, so a `1e-3` band accepts a correctly-converged FD result while rejecting a genuinely wrong mode. Using roundoff here is the canonical mistake the plan warns against (`PLAN_openpipeflow_vs_fnshenfun.md:60`).

**Families.** A, C produce the oracle (spectral, `N=128`). **A↔C** can additionally be compared at `ε_spectral=1e-10` (both spectral Galerkin with matching IMEX tableaux, `imex_rk.py == integrators.py`) — a within-spectral-class parity sub-assertion at tight tolerance. B is the FD party meeting the floor. All linear-eigenvalue paths use dense `scipy.linalg.eig`; eigenvalue comparison must be **set-matched (not index-matched)** with Doppler/frame conversion — reuse `_linear_analysis.match_eigenvalues` (`PLAN_openpipeflow_vs_fnshenfun.md:120`), do not reinvent.

**Extraction.** Confirm B's eigenvalue converges to the spectral value at FD rate: fit `|c_B(N) − c_spectral|` vs `N` on log-log; slope ≈ S2's `p` (≥3.7), demonstrating B *approaches the spectral floor at its own order* — the cleanest possible statement of floor-meeting.

### IV.2 / IV.3 summary matrix

| Test | Type | Oracle | A | B | C | Key tolerance |
|---|---|---|---|---|---|---|
| T1 decaying mode | temporal | `e^{−νk²T}` | ✓ | ✓ (θ band) | ✓ | per-integrator slope band (e.g. `[1.8,2.3]` 2nd-order) |
| T2 epicyclic | temporal (explicit) | `cos(κt)`, `κ²=2Ω(2Ω−S)` | ✓ | gated (`mhd.py:71-74`) | ✓ | slope band + `|κ−κ_th|/κ_th<1e-2` |
| T3 Ohmic decay | temporal (mag. implicit) | `e^{−2ηk²t}` | ✓ | ✓ (`Ha=0`) | ✓ | slope band + `|η−0.01|/0.01<1e-2` |
| T4 IMEX splitting | temporal (both) | MMS adv-diff | ✓ | N/A (single-stage) | ✓ | `p ≥ formal−0.5` |
| S1 spectral exp. | spatial | smooth MMS Helmholtz | ✓ | N/A | ✓ | `>10×`/`+4N` until `ε_spectral=1e-11` |
| S2 FD algebraic | spatial | MMS derivative | N/A | ✓ | N/A | `p≥3.7` + saturation ceiling allowed |
| S3 floor-meeting | cross-family | spectral value (A/C) | oracle | FD party | oracle | `max(C·h_FD^p, ε_spectral)≈1e-3`; A↔C at `1e-10` |

---

## IV.4 Shear / rotation / MHD regime tests (`SR-1`…`SR-9`)

These are the major focus of the suite; each has an analytic oracle. The physical anchor is the shearing-box / shearpy convention (`pcf_mhd_mri_shearpy.py:11-15`, `pcf_mhd_mri_notes.md:37-42`; [BH91, SG10]); the canonical reference for the test *patterns* is `shearpy/tests/test_theory.py`.

**Canonical conventions (apply to ALL SR tests — the §0.2 frame).** `x`=radial/wall-normal (shear gradient), `y`=azimuthal/streamwise (wall motion), `z`=vertical/rotation axis, `Ω=Ω ẑ`. Base flow `U_b(x)=−S·x·e_y`, `dU_b/dx=−S` (A `pcf_mhd_mri_shearpy.py:11,102`; C `pcf_mhd_mri_shearpy_jax.py:70-72`). **Critical adaptation:** Family B plain Couette uses `U=+y·e_x` (`base_flow.py:38`) with opposite sign and swapped axes — the planned canonical adapter must flip the shear sign and remap axes before any comparison. Shearpy defaults `Ω=2/3, S=1` (`q=3/2`, Keplerian), `v_A=b_z=0.025`, `Re=Rm=1000` (or `1e6` for the inviscid eigenvalue). `κ²=2Ω(2Ω−S)=2Ω²(2−q)`; Keplerian `κ=Ω`. Units: Alfvén / Lorentz–Heaviside (`ρ=μ_0=1`); Lorentz prefactor `=1` in A/C; B uses `C_L=Ha²/(Re·Rm)` (channel) or `Ha²/Pm` (couette) — **set `lorentz_prefactor=1` explicitly** (override, `mhd.py:100-101`) to recover unit prefactor for these oracle tests.

**Family applicability at a glance.** A (shenfun): all SR tests run today (DNS MHD `pcf_mhd_mri_shearpy.py` + dense-linear `_pcf_linear.py`, `taylor_couette_mri.py`). C (jax): all run today (DNS `pcf_mhd_mri_shearpy_jax.py`, `taylor_couette_dns_jax.py` + dense-linear `taylor_couette_mri_jax.py`). B (torch): MRI/rotation is **metadata-only / STUBBED** (`mhd.py:71-74`); SR-1/SR-2 are the **acceptance gate** for wiring it in (the first milestone, `PLAN_openpipeflow_vs_fnshenfun.md:191-202`), SR-3/SR-4 CAN already run (no rotation needed), SR-5 is a pure-algebra gate, and SR-6/SR-7/SR-9 require insulating walls + an MRI operator torch lacks (stay A/C-only).

### SR-1 — Epicyclic frequency (rotation + shear, hydro, `k=0` oscillator)

**Description & rationale.** With rotation Ω and shear S, a uniform (`k=0`) velocity perturbation is a 2-D harmonic oscillator: Coriolis couples `u_x ↔ u_y` at the epicyclic frequency κ. The most basic check that the source terms `+2Ω u_y` (x-eqn) and `+(S−2Ω)u_x` (y-eqn) are wired with the right signs/magnitudes; the hydro foundation that MUST pass before any MHD/MRI test is trusted. Direct analogue of shearpy `test_theory.py:100-132`; Athena conserves epicyclic energy to round-off [SG10].

**Setup.** Triply-periodic dynamics (only `k=0` excited); for wall-bounded families use the `(0,0)`-mode mean equations (`ChannelFlow.py:174-197`). Domain e.g. `((-1,1),(0,2π),(0,2π))`, Lorentz/magnetic OFF. Keplerian `Ω=2/3, S=1` (`κ=2/3`); plus a non-Keplerian `Ω=1, S=1` (`κ²=2·1·(2−1)=2`, `κ=√2`) to exercise `κ≠Ω`. Seed only `k=0`: `u_x(0)=u_{x0}=1e-3`, `u_y=u_z=0` (`(u·∇)u=0` for a uniform field, so linear at any amplitude). `N=(8 or 9,4,4)`. `dt=0.01` (A/C IMEXRK222), `≤0.01` (B θ-method). Integrate to `t=2·(2π/κ)`; Keplerian `2π/κ=9.424777960769381` ⇒ `t_end≈18.85`.

**Oracle.**

```
u_x(t) = u_x0 · cos(κ t)
u_y(t) = u_x0 · ((S − 2Ω)/κ) · sin(κ t)
κ = sqrt(2Ω(2Ω − S))
```

Keplerian (`Ω=2/3,S=1`): `κ=0.6666666666666665`, amplitude ratio `(S−2Ω)/κ=(1−4/3)/(2/3)=−0.5`, period `2π/κ=9.424777960769381`. Non-Keplerian (`Ω=1,S=1`): `κ=√2=1.41421356`, ratio `(1−2)/√2=−0.70710678`.

**Metric & tolerance.** Sample `u_x(t), u_y(t)` at ≥20 times over two periods. `rel_err = max_t|u_x^{num}(t) − u_{x0}cos κt|/u_{x0}` and likewise for `u_y` (normalize by `|u_{x0}(S−2Ω)/κ|`). **Pass if `rel_err < 1e-2`** — a physics-oracle tolerance: the analytic solution is exact in the continuum, so the error is time-integration truncation (`κ·dt` small) plus cross-family scheme differences. `1e-2` absorbs `(κΔt)²` for IMEXRK222 and the θ-method first-order base-advection error over many periods, with margin (cf. shearpy `rel<1e-2, abs<5e-4`, `test_theory.py:131-132`). Secondary: extract κ from the FFT of `u_x(t)`, assert `|κ_meas − κ_th|/κ_th < 1e-2`.

**Families.** A, C: run directly (A `pcf_mhd_mri_shearpy.py:130-134, 346-348`; C `pcf_mhd_mri_shearpy_jax.py:130-134`). B: **acceptance gate** — passes only once `n_x += −2Ω u_y`, `n_y += (S−2Ω)u_x` wiring is added (and apply the canonical sign/axis flip, since B's native `U=+y` is opposite). Convergence (optional): `dt`-halving slope of `log(rel_err)` ≈ 2 (IMEXRK222 A/C), ≈ 1 (θ=0.51 B).

### SR-2 — Shear-winding / RDT (isolated shear-induction term)

**Description & rationale.** Isolates the *new* induction term from the base shear: `dB_y/dt = −S·B_x` (the Ω-effect — azimuthal field generated from radial field by differential rotation). For a uniform `k=0` field with no flow and no resistivity, a constant `B_x` winds linearly into `B_y`. The magnetic analogue of SR-1: a single-term sign/magnitude check. The "RDT test" of the deliverable, anchored on shearpy `fd_proto/test_rdt.py:12-26` (slope extraction; that prototype additionally asserts `2.8 ≤ time_order ≤ 3.4`, `space_order ≥ 5.5`) and swing-amplification winding `k_x(t)=k_{x0}+S k_y t`.

**Setup.** As SR-1, `k=0` magnetic mode only. `Ω=2/3, S=1`; **resistivity OFF** (`η=0` or `Rm=1e12`). Velocity perturbation suppressed / `B_x` small so `J×B` back-reaction is negligible over a short integration. IC uniform `B_x(0)=B_{x0}=0.025`, `B_y(0)=0`, `B_z=0` (A seeds via the vector potential giving uniform `B_x`, or the imposed-field channel `_total_b_components`; C analogously). `N=(8,4,4)`, `dt=0.005`, `t_end=5`.

**Oracle.** `B_y(t) = B_{y0} − S·B_{x0}·t` (with `B_x(t)=B_{x0}` const for `k=0`, no decay). For `B_{y0}=0, B_{x0}=0.025, S=1`: `B_y(t)=−0.025 t`, `B_y(5)=−0.125`; slope `dB_y/dt=−0.025`.

**Metric & tolerance.** Linear fit of `B_y(t)` vs `t`; assert `|slope_fit − (−S·B_{x0})|/|S·B_{x0}| < 1e-2` and `B_x` constant `|B_x(t)−B_{x0}|/B_{x0} < 1e-3`. **Pass if both.** Physics-oracle tolerance: exact in the continuum, error = time-truncation + residual Ohmic decay `e^{−ηk²t}≈1`; with `Rm=1e12` and IMEXRK222 expect `~1e-6`. Tighten to `1e-4` within-family A/C.

**Families.** A: term arises automatically from `dA/dt=U×B` with `U_b=−S x e_y` (`pcf_mhd_mri_shearpy.py:16-21, 366-376`); the linear operator has it explicitly `L[by,bx]=Uprime=−S` (`_pcf_linear.py:239`). C: same via the EMF with `U_b=−S x`. B: **acceptance gate** — torch has NO shear-induction term (`mhd.py:71-74`); wiring `dB_y/dt += −S·B_x` into `_magnetic_step` defines acceptance (apply the canonical shear-sign flip).

### SR-3 — Ohmic decay (single magnetic mode, pure resistive diffusion)

**Description & rationale.** A single magnetic Fourier mode with no flow decays purely resistively, calibrating the effective numerical resistivity and verifying the magnetic-diffusion operator (`η∇²b` / `ηΔA`) is discretized and time-split correctly. Independent of rotation/shear. Oracle pattern from shearpy `test_theory.py:66-97` (which asserts `E(t)=E0·exp(−2νk²t)` to `rel=1e-5, abs=1e-8` for the velocity analogue).

**Setup.** Single mode `b ∝ sin(kz)` in a periodic axis (eigenfunctions in the wall-normal direction are not pure Fourier — use `z`). Flow OFF (`u=0`, freeze the momentum solve). `Ω, S` irrelevant (set `S=0` to avoid winding contaminating a `b_x` seed, or seed `b_z`, which winding does not touch). `η=U/Rm`, `Rm=100` ⇒ `η=0.01` (`U=1`). IC `b_z(z,0)=b_0 sin(kz)`, `L_z=2π` ⇒ `k=1`, `b_0=0.01`. `N_z=16`, `dt=0.01`, `t_end=50` (one e-fold: `2ηk²t=1` ⇒ `t=50`).

**Oracle.** `E(t)=E_0 e^{−2ηk²t}`, `b(t)=b_0 e^{−ηk²t}`. For `η=0.01, k=1`: amplitude rate `0.01`, energy rate `0.02`, `E(50)/E_0=e^{−1}=0.3678794`.

**Metric & tolerance.** Fit `log E(t)` vs `t`; assert `|rate_fit − 2ηk²|/(2ηk²) < 1e-2`. **Pass if holds.** Physics-oracle tolerance: exact decay law, error from time truncation only (implicit diffusion, very accurate); within-family A/C expect `~1e-4` (tighten to `1e-3`). For the wall-bounded variant (non-Fourier radial eigenfunction), use the family's true discrete diffusion eigenvalue as the oracle and compare DNS to it at within-family roundoff `<1e-6`.

**Families.** A: resistive Helmholtz on A (`pcf_mhd_divfree.py:159-192`, `η=U/Rm`). C: same (`pcf_mhd_jax.py:78-85`). B: HAS magnetic diffusion (`mhd.py:115-117`, diffusivity `1/Rm` channel / `1/Pm` couette) — **this test CAN run on torch today** (no rotation needed); torch already checks `0<E1<E0` (`test_mhd.py:163-180`), SR-3 strengthens it to the quantitative rate. B's magnetic walls are `b=0`, so use a periodic-direction mode and map `Rm` correctly (`mhd.py:89-94`).

### SR-4 — Alfvén wave (uniform background field, ideal MHD dispersion)

**Description & rationale.** With a uniform background field `B0` and no rotation/shear, a transverse perturbation propagates as an Alfvén wave with `ω=k·v_A`, `v_A=B0/√ρ=B0` (Alfvén units), verifying the ideal induction–Lorentz coupling (`ik B0` terms) produces the correct magnetic-tension wave speed. Oracle from shearpy `test_theory.py:135-178`.

**Setup.** Periodic propagation direction. Uniform background `B0=B0 ẑ`, transverse perturbation. Rotation/shear OFF (`Ω=S=0`); resistivity OFF or tiny (`Rm=1e6`, damping rate `½ηk²` negligible). `B0=v_A=0.1` (or `0.025`), `k=1` along `z` (`L_z=2π`). IC `δu_y(z,0)=ε v_A sin(kz)`, `δb_y(z,0)=0` (a velocity-only kick makes a standing wave oscillating at ω), `ε=1e-3`. `N_z=16`, `dt=0.05`, `t_end=130` (≥2 periods; `T=2π/ω=62.83`).

**Oracle.** `ω=k·v_A`, `v_A=B0`; `δu_y(t)=ε v_A cos(ωt)`. For `B0=0.1, k=1`: `ω=0.1`, `T=62.832`. For `B0=0.025`: `ω=0.025`.

**Metric & tolerance.** FFT `δu_y(t)` (or fit the oscillation); assert `|ω_meas − k v_A|/(k v_A) < 1e-2`. **Pass if holds.** Physics-oracle tolerance: exact dispersion, error = time truncation + tiny resistive damping; within-family A/C `~1e-4`. Secondary equipartition check: time-averaged kinetic ≈ magnetic energy within `1e-2`. **Cross-port anchor:** shearpy `test_alfven_wave_phase` asserts the propagated mode matches `e^{±iωT}` to `<1.2e-2` (`test_theory.py:135-178`).

**Families.** A: imposed field via `_total_b_components` + EMF/Lorentz with `B_total` (`pcf_mhd_mri_shearpy.py:127-132, 350-376`); the linear operator has `ikB` coupling (`_pcf_linear.py:230-238`). C: same (`pcf_mhd_mri_shearpy_jax.py:116-118, 138-146`). B: HAS induction `curl(u×B_total)` + Lorentz `C_L·J×B_total` (`mhd.py:207-228`) — **this test CAN run on torch today** (no rotation needed), provided `lorentz_prefactor=1` (Alfvén units, override `mhd.py:100-101`). B's background default is `(0,1,0)` — set `background_b=(0,0,B0)` and pick the periodic propagation axis.

### SR-5 — Ideal local MRI dispersion: maximum growth `s_max=(3/4)Ω` (geometry-free 4×4)

**Description & rationale.** The geometry-free analytic heart of the MRI: for a Keplerian shearing box with uniform vertical field, the local 4×4 axisymmetric dispersion (`u_x,u_y,b_x,b_y`) has maximum growth `s_max=(3/4)Ω` at `(k v_A)²=(15/16)Ω²`, with marginal cutoff at `(k v_A)²=3Ω²`. Validates the combined rotation + shear + induction + tension operator against [BH91]. It is the local 4×4 algebraic check (shearpy `test_theory.py:181-193`), **NOT** a wall-bounded eigenvalue.

**Setup.** Run as a **dense-linear / algebraic dispersion check** for the ideal (inviscid, non-resistive) limit: form the 4×4 matrix and find its leading eigenvalue vs `k v_A`. The local biquartic (`taylor_couette_mri.py:91-93`): `s⁴ + 2s²(ω_A² + κ²/2) + ω_A²(ω_A² + dΩ²/dlnr) = 0`, `ω_A=k v_A`, Keplerian `κ²=Ω²`, `dΩ²/dlnr=−3Ω²`. `Ω=2/3` (so `s_max=0.5`); scan `(k v_A)²/Ω² ∈ [0, 3.2]`. No grid/`dt`. The shearpy 4×4 matrix form is the cross-port template (`test_theory.py:181-193`):

```
[[ 0,        2Ω,       i k b0,   0     ],
 [ S−2Ω,     0,        0,        i k b0],
 [ i k b0,   0,        0,        0     ],
 [ 0,        i k b0,  −S,        0     ]]
```

**Oracle.** `s_max=(3/4)Ω` (`Ω=2/3` ⇒ `0.5`), `s_max/Ω=0.75`, argmax at `(k v_A)²/Ω²=15/16=0.9375`, marginal cutoff `(k v_A)²/Ω²=3.0`.

**Metric & tolerance.** Scan, find the max real eigenvalue and its argmax. Assert `|s_max/Ω − 0.75| < 2e-3` and `|(k v_A)²_opt/Ω² − 0.9375| < 5e-3`; for the cutoff, assert growth `>0` just inside `(k v_A)²=3Ω²` and `≈0` (`≤1e-12`) just outside. Tolerance is analytic-oracle but tight (`2e-3/5e-3`) because the dispersion is solved algebraically — error is only the `k v_A` scan resolution; refine to shrink it. The findings already achieve `s_max/Ω=0.7499999944199642` and `(k v_A)²/Ω²=0.9373170323757943` (`couette_linear_benchmarks.md:313`).

**Families.** A: `taylor_couette_mri.py:104-122` `mri_keplerian_optimum`. C: `taylor_couette_mri_jax.py:47-62`. B: torch has **no MRI operator** — to run this as a 4×4 algebraic check, implement the dispersion evaluator (pure linear algebra, no solver); this is the lightest MRI acceptance gate and a good first wiring milestone. All three families share the same biquartic, making this the cleanest cross-family analytic anchor.

### SR-6 — Wall-bounded linear MRI growth `γ(k_z)`: conducting vs insulating BC *(like-for-like only)*

**Description & rationale.** In a wall-bounded geometry the MRI growth `γ(k_z)` depends on the magnetic BC: conducting vs insulating give different — even sign-flipped — growth at the same hydro/resistive parameters. Verifies the BC machinery (Robin/Neumann conducting; flux-function insulating) and the resistive MRI operator against the TC-MRI goldens. **R1 warning: NEVER compare wall-bounded `γ` to the ideal `0.75Ω`** — walls + finite resistivity change the answer entirely (that is SR-5 only). **R2 warning: only compare like-for-like BC** — pin `magnetic_bc` on both sides.

**Setup.** TC annulus, `R1=1,R2=2`, `η=0.5`, quasi-Keplerian (`Ω(r)∝r^{−3/2}` analogue; the bench uses `CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)`, `couette_linear_benchmarks.md:329`). BCs under test: (a) **conducting** — `b_r=0` (Dirichlet), `d(r b_θ)/dr=0` (Robin, coefficient `c=r_wall/J`, `J=(R2−R1)/2`, `taylor_couette_mri.py:36-40,183-187`), `b_z'=0` (Neumann); (b) **insulating** — poloidal flux function χ, Robin from modified-Bessel `I/K` log-derivatives, `m=0` only (`taylor_couette_mri.py:42-47,332-444`). Parameters (the goldens, `couette_linear_benchmarks.md:330-353`): conducting `Rm=24.7, S=4.11, η_mag=1/Rm, B0=S·η_mag, ν=1e-6·η_mag`, scan `kz` near `1.75`; insulating `Rm=16.5, S=5.21`, scan `kz` near `1.25`. `Nr=24…48` (converged by 32). Dense GEP `Lq=sMq` (`scipy.linalg.eig`), filter infinite via `FINITE_CAP=1e8`. No `dt`.

**Oracle (golden numbers).**

```
Conducting:  max_kz Re(s) = +0.003322863594034156  at kz ≈ 1.75   (≈ +0.00332)
Insulating:  max_kz Re(s) = -0.00027582037141390655 at kz ≈ 1.25   (≈ -2.76e-4)
```

(`couette_linear_benchmarks.md:352-353`.) Note the **sign flip**: conducting is (marginally) unstable, insulating (marginally) stable at these BC-specific onset parameters.

**Metric & tolerance.** Compute `max_kz Re(s)` per BC. **Within-family** (A↔A regression, or C reproducing its own golden): `rel < 1e-3` against the stored golden (eigenvalue numbers, spectral-exact at modest `Nr`). **Cross-family A↔C** (both spectral): set-matched `Re(s)` (not index-matched, `PLAN_openpipeflow_vs_fnshenfun.md:120`) with tolerance `max(ε_spectral, 1e-6)` (exponential convergence ⇒ agreement `~1e-6` at `Nr=32`). The C golden for a *related* setup is conducting lead `0.25628761535339467`, insulating `0.25995005500337837` (its own `test_taylor_couette_mri_jax.py:29-51`, different parameters — use as the C regression anchor). **Sign-flip assertion:** `conducting_max > 0 > insulating_max` (the explicit regression, see SR-9).

**Families.** A: `taylor_couette_mri.py` (both BCs). C: `taylor_couette_mri_jax.py` (both BCs, `_assemble_flux_parts` for insulating, `m=0`). B: torch has conducting/homogeneous `b=0` only, **no insulating, no MRI operator** — **cannot run**; documents the known gap (GAP MATRIX §3, "Insulating walls = **A**" for B). Insulating is `m=0` only (both A/C raise `NotImplementedError` for `m≠0`). Convergence: refine `Nr` (24→32→48), `Re(s)` converges spectrally (fit `log|s(Nr)−s(2Nr)|` vs `Nr` → straight line on semilog), converged by `Nr=32` (`couette_linear_benchmarks.md:314-316`).

### SR-7 — `Pm`-scan: `γ(k_z)` / critical-`Rm` vs `Pm ∈ {0.1, 1}` against TC-MRI conducting goldens

**Description & rationale.** The MRI onset Rm depends on the magnetic Prandtl number `Pm=ν/η`. Scanning `Pm` and checking critical `Rm_onset` against the TC-MRI conducting goldens validates the relative viscous/resistive scaling and the critical-parameter finder. Physical anchor: `Pm`-dependence of MRI [LL07] (`α∝Pm^δ`, `δ∈[0.25,0.5]`); resistive MRI `Rm_min → 24.7` (conducting) / `16.5` (insulating) as `Pm→0`.

**Setup.** TC, `η=0.5`, quasi-Keplerian, conducting walls (for the like-for-like scan; `S=4.11`). Fixed Lundquist `S=4.11`; scan `Pm ∈ {0.1, 1}` (optionally `0.02`). For each Pm run the critical-Rm bisection `critical_Rm(Pm,S)` (`taylor_couette_mri.py:553-654`), which varies `η_mag` with `ν=Pm·η_mag`, `B0=S·η_mag/d`. `Nr=32`; eigenproblem (no `dt`).

**Oracle (golden numbers, conducting, `η=0.5` quasi-Kep, `S=4.11`).**

```
Pm = 1    →  Rm_onset = 95.3
Pm = 0.1  →  Rm_onset = 32.9
Pm = 0.02 →  Rm_onset = 26.7
(→ Rm_min ≈ 24.7 as Pm → 0; Rüdiger 2023)
```

Insulating analogue (`S=5.21`): `Pm=0.1 → Rm=28.2`, `Rm_min=16.5`.

**Metric & tolerance.** Per Pm, assert `|Rm_onset − golden|/golden < 1e-2` (goldens quoted to 3 sig figs ⇒ `1e-2` covers quoting + `Nr` convergence); within-family A↔A `< 3e-3`. Cross-family A↔C: set-matched leading eigenvalue at the same `(Pm, Rm, kz)`, tolerance `max(ε_spectral, 1e-6)`. **Monotonicity assertion** (quoting-precision-free, robust): `Rm_onset(Pm=1) > Rm_onset(Pm=0.1) > Rm_onset(Pm=0.02)` (decreasing toward `Rm_min`).

**Families.** A: `critical_Rm(Pm,S)` (`taylor_couette_mri.py`); test `test_critical_rm_uses_fixed_pm_and_lundquist_controls` (`test_taylor_couette.py`). C: `taylor_couette_mri_jax.py` critical-parameter path. B: **cannot run** (no MRI operator, no critical-parameter finder, no insulating). PCF analogue: A has the shearpy `_pcf_linear.py` but **no PCF critical-parameter finder** (closure-roadmap WS-B, §4 Phase 3) — so the `Pm`-scan is a TC-geometry test for A/C only. Refine `Nr` → spectral convergence of `Rm_onset`. The saturated-turbulence `α∝Pm^δ` exponent is an explicit non-goal for quantitative reproduction (§4 deferred) — keep any such check qualitative.

### SR-8 — Energy-balance closure with magnetic terms (DNS, transport α and stresses)

**Description & rationale.** In a nonlinear MRI DNS total energy obeys `dE/dt = Production − Dissipation`, production from Maxwell + Reynolds stresses working against the background shear, dissipation viscous + Ohmic. Closing the budget to within truncation is the strongest integral correctness check on the full nonlinear MHD operator (advection, Lorentz, EMF, shear-source) — it catches sign errors and missing terms that single-mode tests miss. It also defines the transport coefficient `α(t)` and the stresses `⟨u_x u_y⟩` (Reynolds), `⟨−B_x B_y⟩` (Maxwell). Anchor: [HGB95] (stress → α).

**Setup.** Shearpy net-flux MRI run (A/C). Domain `((-2,2),(0,4),(0,1))`, `Re=Rm=1000`, `S=1, Ω=2/3, b_z=0.025` (`test_pcf_mhd_mri_shearpy.py:80-84`). Conducting magnetic BC (A `A∈TD³`; C same). Net vertical flux `B0=b_z`. IC small perturbation (`1e-3`) + channel-mode seed via `A_x`. `N=(16,8,16)`, `dt=0.005`, run to `t=3` (linear-growth window; optionally `t=60` saturation, qualitative). Sample energies/stresses every ~100 steps.

**Oracle.** Two oracles. *Linear-growth phase (quantitative):* magnetic energy grows, `E_mag(t_end) > 2·E_mag(0)` (findings expect ~7× over `t=1..3`), strictly monotone increasing (`test_pcf_mhd_mri_shearpy.py:130-131`). *Energy-balance closure:* `dE_total/dt` (FD of measured total energy) vs `P−D` with production `P = S·(⟨−B_x B_y⟩ + ⟨u_x u_y⟩)·Volume` and dissipation `D = (1/Re)⟨|∇u|²⟩ + (1/Rm)⟨|∇b|²⟩`; assert `|dE/dt − (P−D)|/(|P|+|D|) < tol`. *Transport:* `α(t) = (⟨u_x u_y⟩ + ⟨−B_x B_y⟩)/v_A²` (`pcf_mhd_mri_shearpy.py:385-413`), finite and positive in the growing phase.

**Metric & tolerance.** Energy-balance residual: cross-family / DNS tolerance `< max(C·Δx^p, C·Δt^q, ε_spectral)` — with `dt=0.005` (IMEXRK222, `q=2`) and spectral space, bounded by the time-truncation of the energy FD; use `< 1e-2` (relative) for a coarse run, tightening to `1e-3` as `dt→0`. **Not roundoff** (a nonlinear DNS integral with finite `dt`). Growth-factor check (robust binary): pass if `E_mag[-1] > 2·E_mag[0]` and monotone increasing (`test_pcf_mhd_mri_shearpy.py:130-131`; findings expect ~7×). Stress signs: assert `⟨−B_x B_y⟩ > 0` and `⟨u_x u_y⟩ > 0` in the growing phase (both transport angular momentum outward). div(b) guard: `max|div b| < 1e-10` throughout (`test_pcf_mhd_mri_shearpy.py:133`).

**Families.** A: `pcf_mhd_mri_shearpy.py:385-413` computes Reynolds/Maxwell/α; `test_netflux_mri_magnetic_energy_grows` is the seed. C: `pcf_mhd_mri_shearpy_jax.py:149-182` computes the same + α. B: **acceptance gate** — torch computes Maxwell/Reynolds stresses already (`mhd.py:401-412` channel: `reynolds_xy`, `maxwell_xy=−⟨B_x B_y⟩`, `transport_xy`, `alpha`), but the MRI growth driving them is absent (no shear-source); once SR-1/SR-2 wiring lands, the existing stress diagnostics let SR-8 run. Reconcile B's `maxwell_xy` default (induced field, `total=False`, `mhd.py:407-412`) and α denominator (`Σ background_b²`, `mhd.py:414-415`) with A/C's total-field budget and `v_A²=b_z²`. Convergence: `dt`-halving on the residual → integrator order (2); grid refinement → the early-phase growth rate converges to the linear-MRI eigenvalue, the WS-A quantitative-growth-match gate (§4 Phase 3), tolerance `|γ_dns − Re(s_lin)| < 2e-3·|s_lin|` (the TC precedent `test_mri3d_growth_matches_eigensolver`, per-`m` rel-errs `4e-8/4e-7/2e-6` for `m=0/1/2`).

### SR-9 — Conducting ↔ insulating BC sign-flip (explicit regression)

**Description & rationale.** A focused, fast regression that the magnetic BC *changes the sign* of the marginal MRI growth at fixed-but-BC-specific parameters. Guards against the most insidious BC bug (the wrong Robin Jacobian `c=r_wall/J` producing a spurious growing magnetic mode). The minimal, standalone version of SR-6's sign observation — the canonical "BC flip flips the physics" check (R2).

**Setup.** Identical to SR-6 reduced to a single assertion. TC, `η=0.5`, quasi-Keplerian, `Nr=32`. Two runs: conducting `Rm=24.7, S=4.11, kz=1.75`; insulating `Rm=16.5, S=5.21, kz=1.25`.

**Oracle.**

```
conducting  max growth = +0.003322863594034156   (> 0, unstable)
insulating  max growth = -0.00027582037141390655 (< 0, stable)
```

The load-bearing fact is the **sign flip**: `+0.00332` vs `−2.76e-4` (`couette_linear_benchmarks.md:352-353`).

**Metric & tolerance.** Two-part: (a) **sign assertion** `conducting > 0 > insulating` (binary, no tolerance needed); (b) value regression `|computed − golden|/|golden| < 1e-2` per BC (within-family `< 3e-3`). **Pass if both.** Additionally assert the conducting Robin BC: `|b_θ + r·b_θ'| < 1e-10` at both walls (`test_conducting_btheta_bc_is_satisfied`, `test_taylor_couette.py`) — directly catches the wrong-Jacobian bug; and insulating solenoidality `div(b)/|b| < 1e-10` (`test_insulating_eigenmode_is_solenoidal_by_construction`).

**Families.** A: `taylor_couette_mri.py` (both BCs + the BC-satisfaction test). C: `taylor_couette_mri_jax.py` (both BCs). B: **cannot run** (no insulating walls, no MRI operator) — explicit documentation of torch's gap. This test is A/C only and is the cleanest A↔C parity check on the BC machinery. Convergence: refine `Nr`; the sign must NOT flip with resolution (if it does, the operator/BC is wrong) — spectral convergence of each value as in SR-6.

### IV.4 coverage & gating matrix

| Test | Type | Oracle | A | B | C | Key tolerance |
|---|---|---|---|---|---|---|
| SR-1 epicyclic | temporal/physics | `cos(κt)`, `κ²=2Ω(2Ω−S)` | ✓ | **gate** (`mhd.py:71-74`) | ✓ | `rel<1e-2`; FFT `|κ−κ_th|/κ_th<1e-2` |
| SR-2 shear-winding | physics | `B_y=−S B_x t` | ✓ | **gate** | ✓ | slope `rel<1e-2` (`1e-4` within A/C) |
| SR-3 Ohmic decay | physics | `e^{−2ηk²t}` | ✓ | ✓ (`Ha=0`) | ✓ | `|rate−2ηk²|/(2ηk²)<1e-2` |
| SR-4 Alfvén wave | physics | `ω=k v_A` | ✓ | ✓ (`lorentz_prefactor=1`) | ✓ | `|ω−k v_A|/(k v_A)<1e-2` |
| SR-5 ideal MRI 4×4 | algebraic | `s_max/Ω=0.75`, `(k v_A)²/Ω²=15/16` | ✓ | (4×4 evaluator only) | ✓ | `2e-3` / `5e-3` |
| SR-6 wall-bounded MRI BC | eigenvalue | golden `+0.00332` / `−2.76e-4` | ✓ | **A** (no insul./MRI) | ✓ | within `1e-3`; A↔C `max(ε_spec,1e-6)` |
| SR-7 Pm-scan | eigenvalue | `Rm_onset` table | ✓ | **A** | ✓ | `1e-2` (within `3e-3`) + monotonicity |
| SR-8 energy/stress budget | DNS integral | `dE/dt=P−D`; `E_mag>2×` | ✓ | **gate** (stresses exist) | ✓ | residual `<1e-2`; growth binary |
| SR-9 BC sign-flip | eigenvalue | `+0.00332` vs `−2.76e-4` | ✓ | **A** | ✓ | sign binary + `1e-2` value |

Legend: ✓ runs today; **gate** = the test defines the torch-wiring acceptance milestone (SR-1/SR-2 first, then SR-8); **A** = absent capability in torch (no insulating wall and/or no MRI operator), record as `skip` with the cited gap.

---

## IV.5 Run environments (disjoint, file-based goldens)

Per the §"Disjoint run environments" hand-off note, no live in-process cross-family import is permitted — cross-family parity (S3, the A↔C sub-assertions of SR-6/SR-7) reads committed JSON/HDF5 goldens written by each family in its own environment:

- **A → `/home/nauman/miniconda3/envs/shenfun/bin/python`** (e.g. `conda run -n shenfun pytest -q demo/test_couette_linear.py`, `demo/test_taylor_couette.py`).
- **B → `/home/nauman/miniconda3/envs/huggingface/bin/python`**.
- **C → `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/.venv`** (uv, system Python 3.12.3, local JAX 0.10.1; `cuda13` optional extra configured).

**Per-family "skip not fail" gaps to encode** (so a missing capability never reads as a regression): C has no pipe (`F1d`, `F1e`, any `S2`-pipe variant → `skip`); B has no Womersley (`F1e` → `skip`), no pipe MHD, and MRI metadata-only (`SR-1`/`SR-2` are the acceptance gate, not a pass; `SR-3`/`SR-4` run today); insulating-wall MRI is A/C-only (`SR-6`/`SR-7`/`SR-9` → `skip` for B); A has no GPU/JIT/autograd (out of scope for this physics suite).


\newpage

# Part V — Compute, portability, autograd appendix + References

This part is the cross-family reference layer for everything the three solver families (**A = shenfun**, **B = torch**, **C = jax**; see §0.1) share or diverge on at the *infrastructure* level — hardware targets, floating precision, just-in-time compilation, automatic differentiation, mode-batching, and multi-device parallelism — followed by the run-environment constraints that shape how the families are tested against one another (§V.2), and finally the master reference list (§V.3). Algorithmic content (discretization, integrators, MHD/MRI source terms) lives in Parts I.A/I.B/I.C; this part only catalogs the compute substrate those algorithms run on and the literature they are validated against.

The single recurring theme: **A is the spectral oracle but the compute laggard** (CPU/MPI, float64, no JIT, no autograd); **B is the differentiable GPU workhorse** (PyTorch, CUDA-by-construction, full autograd including the Lorentz coupling, but no compile/JIT and a 1st-order θ-integrator); **C is the JIT+autograd+TPU family** (JAX, x64-by-default, `jax.jit`/`shard_map` pervasive, `value_and_grad` adjoint) at the cost of host escapes for eigensolves and SymPy stencil derivation. These trade-offs are exactly why cross-family testing runs through a **file-based golden bridge across disjoint environments** (§V.2) rather than live cross-import.

## V.1 Backend / precision / autograd / JIT / sharding

### V.1.1 Master comparison table

The cells below are verified against the cited source files. Where a capability is absent the cell reads **none** with the evidence anchor; where it is present-but-constrained the constraint is stated inline.

| Aspect | A = shenfun | B = torch | C = jax |
|---|---|---|---|
| **Language / backend** | Python; NumPy/SciPy + Cython/numba kernels; MPI via `mpi4py-fft` [Mortensen18] | Python; pure PyTorch tensor ops | Python; pure JAX (`jnp`, `jax.scipy.fft`, `jnp.fft`) over XLA |
| **CPU** | yes (primary target) | yes (default `device="cpu"`) | yes (XLA CPU) |
| **GPU** | **none** — CPU/MPI only | yes by construction (device-agnostic tensors; CUDA exercised only in `torchpipeflow/benchmarks/benchmark_hotspots.py:29,39-41`) | yes (JAX/XLA; `cuda13` extra configured in `pyproject.toml:19-37`; local venv reports JAX 0.10.1) |
| **TPU** | **none** | **none** | yes (device-agnostic XLA + `shard_map`; `docs/couette_fourier_layout.md:11`) |
| **Default float precision** | float64 | complex128 / float64 (`torchchannel/.../solver.py:87,103`) | float64 / complex128 (x64 forced at import, `src/jaxfun/__init__.py:8`) |
| **Lower precision validated?** | n/a (always f64) | yes — complex64/float32 path, roundtrip err `<5e-5` (`test_float32.py:21-23`) | x64 is unconditional at import; toggle is global |
| **JIT / compile** | **none** (numba/Cython AOT kernels only) | **none** — no `torch.compile`/`torch.jit`/`vmap` anywhere (grep: zero hits) | yes — `jax.jit`/`nnx.jit` pervasive; transforms compiled & cached |
| **Autograd** | **none** (NumPy/SciPy) | yes — full, end-to-end incl. magnetic→velocity Lorentz coupling (`test_mhd.py:183-201`) | yes — `value_and_grad`/`grad` (`pcf_minimal_seed_jax.py:118,136`) |
| **Mode batching** | vectorized NumPy + MPI mode distribution | explicit batched `(H,N,N)`/`(N,H)` tensor ops + `einsum` (no `vmap`) | `jax.vmap` (separable per-axis transforms, `sharding.py:24-40`) |
| **Multi-device** | MPI (slab/pencil) | **none** (single-device; no distributed primitives in solver) | `shard_map` SPMD (CPU/GPU/TPU), `Mesh(jax.devices(), ("k",))` (`sharding.py:9-11`) |
| **Eigensolver** | `scipy.linalg.eig` (host) | `scipy.linalg.eig` (host) | `scipy.linalg.eig` (host, `la/eig.py:168`) |

### V.1.2 CPU / GPU / TPU support

**A (shenfun)** is CPU-only and scales via **MPI**: `mpi4py-fft` provides automatic **slab** and **pencil** domain decompositions and a global-array redistribution algorithm, and has been run on thousands of cores on supercomputers [Mortensen18]. There is no GPU or TPU path; this is the deliberate trade for being the trusted spectral-Galerkin oracle.

**B (torch)** is **device-agnostic by construction**. Every tensor is created with an explicit `device=` taken from the mesh (`self.mesh.y.device` / `self.mesh.r.device`), and no solver kernel special-cases CUDA. Moving a problem to GPU is therefore a matter of constructing the solver with `device="cuda"`; the only place CUDA is *exercised* in-repo is the pipeflow benchmark harness (`torchpipeflow/benchmarks/benchmark_hotspots.py:29` for the `--device` arg, `:39-41` for `torch.cuda.synchronize`). There is no TPU path (PyTorch/XLA is not used) and no multi-device distribution layer inside the solver. The MHD state objects carry device movers (`TaylorCouetteMHDState.to(device, dtype)`, `torchcouette/.../mhd.py:36-50`).

**C (jax)** targets **CPU, GPU, and TPU** through XLA with no device-specific branches. The local `.venv` reports Python 3.12.3 and JAX 0.10.1, and the CUDA build is configured as an optional dependency (`cuda13 = ["jax[cuda13]>=0.10.1"]`, `pyproject.toml:19-37`). TPU support is asserted by the device-agnostic FFT/sharding layer — `docs/couette_fourier_layout.md:11` states the full-complex FFT layout "works on CPU, GPU and TPU without an rfft-specific branch." At import, `__init__.py:4` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX does not greedily preallocate GPU memory.

### V.1.3 Floating precision and how to enable float64

This is a frequent silent-error source across families, so the canonical floors are spelled out:

- **A:** float64 always; no toggle needed. Complex Fourier coefficients are complex128.
- **B:** the default solver `dtype` is **`torch.complex128`** (`torchchannel/.../solver.py:87`), validated to `complex64`/`complex128` only (`:100-101`); the derived real working dtype is **float64** for complex128 and float32 for complex64 (`real_dtype` at `:103`). To run in single precision, construct with `dtype=torch.complex64` — the spectral roundtrip then keeps complex64 with error `<5e-5` (`test_float32.py:21-23`) and a full solver step preserves complex64 with divergence and BC residual both `<1e-4` (`test_float32.py:44-50`). Real linear solves are done by splitting the complex RHS into stacked real/imag and solving in real arithmetic (`solver.py:251-271`), so the realized precision follows `real_dtype`.
- **C:** float64 is enabled **globally and unconditionally at import** by `jax.config.update("jax_enable_x64", True)` (`src/jaxfun/__init__.py:8`). This is asserted by `tests/test_x64_default.py:7-10` (`jax.config.read("jax_enable_x64") is True` and `jnp.zeros(1).dtype == jnp.float64`). The package therefore does **not** require the usual per-process `JAX_ENABLE_X64` env var or per-call `dtype=` plumbing — importing `jaxfun` is sufficient. A test-time toggle `--float64` exists (`docs/testing.md`).

**Implication for cross-family comparisons (ties to the §3 tolerance ladder):** all three families can meet a float64 floor, so within-family operator identities may be asserted at roundoff (spectral families 1e-10…1e-21; B's pinv cleanup ~1e-7). Cross-family observables must never be compared at roundoff — use `max(C·Δx^p, C·Δt^q, ε_spectral)` — because B's FD spatial error and 1st-order θ-time error (Part I.B.2) dominate over the spectral families' exponential convergence.

### V.1.4 JIT / compile

**A and B have no runtime JIT.** A relies on ahead-of-time Cython/numba kernels [Mortensen18]; B has **no `torch.compile`, no `torch.jit`, no `vmap`** in any of the three packages (verified by grep returning zero hits across `torchchannel`, `torchcouette`, `torchpipeflow`). B's "batching over modes" is therefore done with explicit batched tensor algebra — per-mode operator blocks stacked as `(H, N, N)` and solved with batched `torch.linalg.solve`/`lu_solve`, wall-normal derivatives via `torch.einsum("ij,...jkm->...ikm", W, f)` — with the flattened mode layout `(N, Kc·Mc)` serving as the batch dimension. This is a documented compute gap for B (`B-MHD-COMPUTE §6.4`): wiring in `torch.compile` is a future optimization, not a correctness issue.

**C is JIT-pervasive.** `@jax.jit` (often with `static_argnums`) decorates space methods and integrator steps; a `jit_vmap` helper (`utils/common.py:35-72`) jits and conditionally vmaps based on input rank; matrices and integrators are `flax.nnx` modules/pytrees so `jit`/`scan` can close over solver state. The separable tensor-product transforms are compiled once per `(op, N)` and cached in `_spmd_local_fn_cache`. The only non-jittable escapes are host calls: `scipy.special.roots_jacobi` (Gauss-Jacobi nodes), `scipy.linalg.eig`/`eigh`/`svdvals` (eigen/stability), and SymPy stencil/coefficient derivation (compile-time, not in the hot loop).

### V.1.5 Autograd availability and concrete use-cases

This is the sharpest capability split. **A has no autograd** (NumPy/SciPy stack); any gradient-based task (optimization, adjoint loops) must be done by finite differences or hand-derived adjoints, which is why A is positioned as the forward oracle rather than the optimization engine.

**B is fully differentiable end-to-end**, including through the divergence-free projection, the influence/boundary correction, the FFTs, and — critically — the MHD coupling. There are no `.detach()` calls in the step paths (`torch.no_grad()` appears only in IO/serialization, e.g. `torchpipeflow/field_io.py:161`). The LU factorizations (`torch.linalg.lu_factor`/`lu_solve`) and `torch.linalg.solve`/`inv` are all autograd-differentiable. The load-bearing test is `torchchannel/tests/test_mhd.py:183-201` (`test_mhd_step_preserves_autograd_through_velocity_and_magnetic_fields`): after a step, gradients exist and are finite for `u,v,w,bx,by,bz`, and the **magnetic→velocity coupling gradient is strictly positive**, proving the Lorentz force propagates gradients. The couette analogue (`torchcouette/tests/test_mhd.py:105-130`) asserts `br.grad` finite and `max>0` with `lorentz_prefactor=2.0`.

**C provides JAX autograd with a minimal-seed adjoint.** `examples/pcf_minimal_seed_jax.py` builds the plane-Couette minimal-seed optimization loop: `jax.grad(solver.perturbation_energy)(state)` for the energy gradient (`:118`) and `gain, gradient = jax.value_and_grad(objective)(state)` for the perturbation-gain-and-gradient in one pass (`:136`). The module docstring frames this explicitly as a "fixed-energy normalization, perturbation gain, and tangent ... adjoint loop." Within the jaxfun framework, the only place autograd enters the *spectral assembly* is derivative-Vandermonde construction (`jacn` = repeated `jax.jacfwd` + vmap, `utils/common.py:100-103`); Chebyshev/Legendre/Fourier override this with closed-form coupling matrices, so framework-level autograd is shallow and the *application*-level autograd (minimal seeds, differentiable sweeps) is where it pays off.

**Concrete autograd use-cases enabled (and which family serves each):**

| Use-case | A | B | C | Notes |
|---|---|---|---|---|
| **Adjoint / minimal-seed loops** (touch-the-edge optimization à la [PWK12]) | no | yes (differentiable step) | yes (`pcf_minimal_seed_jax.py` `value_and_grad`) | C ships the explicit minimal-seed helper; B has the differentiable substrate to build one |
| **Newton–GMRES / hookstep ECS continuation** ([GHC09], channelflow-style) | via hand/FD Jacobian-vector products | matrix-free JVPs via autograd | matrix-free JVPs via `jvp`/`linearize` | for [Nagata90]/[Waleffe98] ECS, [FE03]/[WK04] pipe TWs |
| **Differentiable parameter sweeps** (∂(observable)/∂Re, ∂α/∂Pm for the [LL07] δ∈[0.25,0.5] scaling) | FD only | autograd through Lorentz prefactor `Ha²/(Re·Rm)` | autograd through `eta=U/Rm` etc. | B's prefactor and C's `eta` are in the differentiable graph |
| **Linear-stability gradient checks** (OS/Squire) | no | no (`scipy.linalg.eig` host) | no (`scipy.linalg.eig` host) | all three drop to host SciPy for the eig itself |

### V.1.6 Vectorization / batching over modes (vmap)

The three families realize "do the same dense per-mode solve for all Fourier modes" three different ways:

- **A:** NumPy-vectorized operator application with MPI distribution of modes across ranks.
- **B:** **explicit batched tensor ops** — no `vmap`. Per-mode operators are stacked into `(H, N, N)` and solved with batched `torch.linalg.solve`/`lu_solve`; horizontal derivative factors (`kalpha`, `mbeta`, `fk_th`, `fk_z`) are elementwise broadcasts; wall-normal/radial derivative via a single `einsum`. The "batch axis" is the flattened mode index.
- **C:** **`jax.vmap`** is the batching primitive. `_build_local_apply_fn(dim, ax, fn)` returns a `jax.jit(jax.vmap(...))` that applies a 1-D transform along one axis of a local shard (2-D: single vmap; 3-D: nested vmap, `sharding.py:30-40`), compiled once and reused. Separable transforms are thus vmapped per axis and fused into one XLA computation.

### V.1.7 Multi-device sharding / MPI

- **A:** **MPI** is the parallel model — `mpi4py-fft` slab/pencil decompositions with global redistribution [Mortensen18]; no shared-memory device sharding.
- **B:** **single-device.** There is no distributed/sharding primitive in the solver; scaling is intra-device tensor parallelism only.
- **C:** **`shard_map` SPMD.** A global 1-D device mesh `spmd_mesh = Mesh(jax.devices(), ("k",))` (`sharding.py:9`) defines a spectral sharding `P("k")` (axis 0 sharded) and physical sharding `P(None, "k")` (axis 1 sharded) (`sharding.py:10-11`); `get_transposed_sharding` swaps them (`:14-21`). The production multi-device transform `_apply_separable_spmd_shard_map` runs as a single fused `shard_map`: phase 1 transforms the unsharded axes locally, then `jax.lax.all_to_all(tiled=True, axis_name="k")` transposes the sharding, then phase 2 transforms the originally-sharded axes — output sharding being the transpose of input. The `all_to_all` requires the split axis divisible by the device count, which holds for powers-of-2 Fourier and even Chebyshev quadrature counts. `VectorTensorProductSpace` uses slab sharding `P(None,"k")` (spectral) / `P(None,None,"k")` (physical) keeping the vector-component axis unsharded. Multi-device tests are gated behind `--num-devices=2 -m spmd`.

### V.1.8 Autograd / compute parity caveats for cross-family work

- **B and C have full autograd; A has none.** Any "differentiate the solver" parity test (e.g. confirming ∂growth/∂Re agrees) can only be a **B↔C** comparison; A participates only through forward observables. This is captured in the closure roadmap as Phase 5b ("autograd parity statement across B/C").
- **All three host-escape for eigenvalues**, so linear-stability *values* are SciPy-identical in principle; the families differ only in how they assemble the OS/Squire operators (spectral-exact for A/C, FD for B), which is a *discretization* difference (Part IV/S-tests), not a backend one.
- **C's x64-at-import** means a process that imports `jaxfun` is committed to float64 globally; mixing a float32 JAX workload in the same process is not supported, which matters for shared test runners.

## V.2 Run environments (disjoint envs; file-based goldens)

The three families live in **three mutually incompatible Python environments** and cannot be co-imported in one process. This is a hard constraint, not a preference: A is a conda env built around `shenfun`/`mpi4py-fft`; B is a separate conda env built around a specific PyTorch; C is a `uv`-managed virtual environment on system Python 3.12.3 with local JAX 0.10.1 and a configured `cuda13` optional extra. The verified interpreter/venv roots (all present on disk) are:

| Family | Environment | Interpreter / venv root |
|---|---|---|
| **A = shenfun** | conda env `shenfun` | `/home/nauman/miniconda3/envs/shenfun/bin/python` |
| **B = torch** | conda env `huggingface` | `/home/nauman/miniconda3/envs/huggingface/bin/python` |
| **C = jax** | `uv` venv (system Python 3.12.3, JAX 0.10.1 locally verified; `cuda13` extra configured) | `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/.venv` |

**Why this forces a file-based golden bridge.** Because no two families can be imported into the same interpreter, there is **no live cross-import** path for cross-family parity (contrast C's *internal* live-parity test `test_live_shenfun_parity.py`, which only works because shenfun and jaxfun can co-exist in C's env, not because A's solver is imported). Instead, cross-family testing is done by:

1. Running each family's solver in its own environment as a subprocess (or in CI as a separate job), invoked through the per-family interpreter above.
2. Writing the **canonical-frame observables** — growth rates, energies, Reynolds/Maxwell stresses, transport `α`, divergence norms — to a file (a "golden") after applying the planned `to_canonical()` adapter (§0.2) so that B's swapped axes and flipped shear sign are remapped before comparison.
3. Reading the goldens back in a neutral comparison harness and asserting agreement against the **cross-family tolerance band** `max(C·Δx^p, C·Δt^q, ε_spectral)`, never roundoff (§3 tolerance ladder).

This is the intended mechanism behind the Phase-0 prerequisite of the closure roadmap (`parity/conventions.py`, `parity/observables.py`): the adapter and the golden writer should be the only shared code, and they should operate on plain arrays/scalars, not on live solver objects. It also explains the gap-matrix entry "Cross-family parity: B — none (planned `parity/`)" — B has no cross-boundary test yet precisely because the bridge is file-based and must be built.

**Skip-not-fail conventions encoded in the golden harness** (so a missing capability is recorded as a skip, not a spurious failure): C has no pipe (pipe goldens skip for C); B has no Womersley oracle and its MRI is metadata-only (the MRI shear/rotation goldens are an *acceptance gate* for B's Phase-1 wiring, not a passing comparison); insulating-wall MRI goldens are A/C-only (skip B); A produces no GPU/JIT/autograd goldens (those parity tests are B↔C only); and F6/TGV is currently missing across the target solver trees, so it should be tracked as a required addition rather than a passing skip.

## V.3 References (master citation list)

Writers cite by the keys below. Remaining items flagged "(unverified)" had a publisher 403/TLS/scan issue during source gathering; the bibliographic core is stable. URLs are given where a stable landing page exists.

### Plane Couette / channel DNS, stability, ECS

- **KMM87** — Kim, J., Moin, P. & Moser, R. (1987). Turbulence statistics in fully developed channel flow at low Reynolds number. *J. Fluid Mech.* 177, 133–166. DOI:10.1017/S0022112087000892. https://doi.org/10.1017/S0022112087000892 — canonical Re=3300 (Re_τ≈180) channel DNS; ~4×10⁶ points (192×129×160); Fourier(x,z) × Chebyshev(y); semi-implicit CNAB2-type, velocity–vorticity.
- **MKM99** — Moser, R. D., Kim, J. & Mansour, N. N. (1999). DNS of turbulent channel flow up to Re_τ=590. *Phys. Fluids* 11(4), 943–945. DOI:10.1063/1.869966. https://doi.org/10.1063/1.869966 — Re_τ ≈ 180, 395, 590 statistics database.
- **LM15** — Lee, M. & Moser, R. D. (2015). DNS of turbulent channel flow up to Re_τ≈5200. *J. Fluid Mech.* 774, 395–415. DOI:10.1017/jfm.2015.268. arXiv:1410.7809.
- **Orszag71** — Orszag, S. A. (1971). Accurate solution of the Orr–Sommerfeld stability equation. *J. Fluid Mech.* 50(4), 689–703. DOI:10.1017/S0022112071002842. — plane-Poiseuille linear critical **Re_crit = 5772.22** (Chebyshev-tau + QR); the cross-family OS benchmark.
- **RH93** — Reddy, S. C. & Henningson, D. S. (1993). Energy growth in viscous channel flows. *J. Fluid Mech.* 252, 209–238. DOI:10.1017/S0022112093003738. — transient growth O(R²), magnitude O(1000), from operator non-normality.
- **Nagata90** — Nagata, M. (1990). Three-dimensional finite-amplitude solutions in plane Couette flow: bifurcation from infinity. *J. Fluid Mech.* 217, 519–527. DOI:10.1017/S0022112090000829. — first 3-D ECS in PCF (saddle-node from the TC wavy-vortex branch).
- **Waleffe97** — Waleffe, F. (1997). On a self-sustaining process in shear flows. *Phys. Fluids* 9(4), 883–900. DOI:10.1063/1.869185. — rolls→streaks→breakdown SSP cycle.
- **Waleffe98** — Waleffe, F. (1998). Three-dimensional coherent states in plane shear flows. *Phys. Rev. Lett.* 81(19), 4140–4143. DOI:10.1103/PhysRevLett.81.4140.
- **HKW95** — Hamilton, J. M., Kim, J. & Waleffe, F. (1995). Regeneration mechanisms of near-wall turbulence structures. *J. Fluid Mech.* 287, 317–348. DOI:10.1017/S0022112095000978. — minimal-flow-unit streak/vortex regeneration cycle.
- **CB97** — Clever, R. M. & Busse, F. H. (1997). Tertiary and quaternary solutions for plane Couette flow. *J. Fluid Mech.* 344, 137–153. DOI:10.1017/S0022112097005818. https://doi.org/10.1017/S0022112097005818.
- **GHC09** — Gibson, J. F., Halcrow, J. & Cvitanović, P. (2009). Equilibrium and travelling-wave solutions of plane Couette flow. *J. Fluid Mech.* 638, 243–266. DOI:10.1017/S0022112009990863. arXiv:0808.3375. — channelflow ECS catalog; lower-branch saddle-node Re≈125–130; Newton–Krylov–hookstep.
- **Viswanath07** — Viswanath, D. (2007). Recurrent motions within plane Couette turbulence. *J. Fluid Mech.* 580, 339–358. DOI:10.1017/S0022112007005459.

### Pipe flow & Openpipeflow

- **Willis17** — Willis, A. P. (2017). The Openpipeflow Navier–Stokes solver. *SoftwareX* 6, 124–127. DOI:10.1016/j.softx.2017.05.003. arXiv:1705.03838. https://www.openpipeflow.org — Fourier(θ,z) + 9-point Chebyshev-spaced FD(r) (1st/2nd derivs to 8th/7th order), 3/2 dealiasing, 2nd-order predictor–corrector with pressure-Poisson + influence-matrix BC, per-Fourier-mode banded LU, Newton–Krylov–trust-region + multiple shooting + Arnoldi, MPI 2-D split; Re=5300 validates vs [Eggels94].
- **Womersley55** — Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow and viscous drag in arteries when the pressure gradient is known. *J. Physiol.* 127(3), 553–563. DOI:10.1113/jphysiol.1955.sp005276. https://doi.org/10.1113/jphysiol.1955.sp005276 — pulsatile Bessel-function oracle; Womersley number α_W = R√(ω/ν).
- **FE03** — Faisst, H. & Eckhardt, B. (2003). Traveling waves in pipe flow. *Phys. Rev. Lett.* 91(22), 224502. DOI:10.1103/PhysRevLett.91.224502. arXiv:nlin/0304029. — Fourier–Legendre collocation TWs; **C₃ saddle-node at Re≈1250** (Re=2RU/ν).
- **WK04** — Wedin, H. & Kerswell, R. R. (2004). Exact coherent structures in pipe flow: travelling wave solutions. *J. Fluid Mech.* 508, 333–371. DOI:10.1017/S0022112004009346. — independent **m=3 at Re=1251** cross-validation of [FE03].
- **PDK09** — Pringle, C. C. T., Duguet, Y. & Kerswell, R. R. (2009). Highly symmetric travelling waves in pipe flow. *Phil. Trans. R. Soc. A* 367(1888), 457–472. DOI:10.1098/rsta.2008.0236. arXiv:0804.4854. — mirror-symmetric M1 saddle-node Re≈773.
- **PWK12** — Pringle, C. C. T., Willis, A. P. & Kerswell, R. R. (2012). Minimal seeds for shear flow turbulence: using nonlinear transient growth to touch the edge of chaos. *J. Fluid Mech.* 702, 415–443. — the minimal-seed adjoint-optimization problem (the autograd use-case of §V.1.5).
- **EBHW07** — Eckhardt, B., Schneider, T. M., Hof, B. & Westerweel, J. (2007). Turbulence transition in pipe flow. *Annu. Rev. Fluid Mech.* 39, 447–468. DOI:10.1146/annurev.fluid.39.050905.110308. — Hagen-Poiseuille linearly stable at all Re; subcritical transition; edge state.
- **Avila11** — Avila, K., Moxey, D., de Lozar, A., Avila, M., Barkley, D. & Hof, B. (2011). The onset of turbulence in pipe flow. *Science* 333(6039), 192–196. DOI:10.1126/science.1203223. — sustained-turbulence onset **Re=2040±10** (puff splitting vs decay).
- **Eggels94** — Eggels, J. G. M. et al. (1994). Fully developed turbulent pipe flow: DNS vs experiment. *J. Fluid Mech.* 268, 175–209. DOI:10.1017/S002211209400131X. — **Re=5300** turbulent-statistics benchmark.

### Taylor–Couette

- **Taylor23** — Taylor, G. I. (1923). Stability of a viscous liquid contained between two rotating cylinders. *Phil. Trans. R. Soc. A* 223, 289–343. DOI:10.1098/rsta.1923.0008. https://doi.org/10.1098/rsta.1923.0008 — first linear onset; **Ta_c=1708** (narrow-gap), refined Ta_c=1707.76 as μ→1.
- **DiPrima85** — DiPrima, R. C. & Swinney, H. L. (1985). Instabilities and transition in flow between concentric rotating cylinders. In *Hydrodynamic Instabilities and the Transition to Turbulence*, Topics Appl. Phys. 45, 139–180. Springer.
- **Marcus84** — Marcus, P. S. (1984). Simulation of Taylor-Couette flow. Part 1. *J. Fluid Mech.* 146, 45–64. DOI:10.1017/S0022112084001762. — pseudospectral TC with Green-function/capacitance BC enforcement; growth rates/wave speeds vs linear theory.
- **Wendt33** — Wendt, F. (1933). Turbulente Strömungen zwischen zwei rotierenden konaxialen Zylindern. *Ing.-Arch.* 4(6), 577–595. DOI:10.1007/BF02084936 (unverified DOI). — torque G∝Re^α, α≈1.5–1.7, η=0.68/0.85/0.935.
- **EGL07** — Eckhardt, B., Grossmann, S. & Lohse, D. (2007). Torque scaling in turbulent Taylor–Couette flow. *J. Fluid Mech.* 581, 221–250. DOI:10.1017/S0022112007005629. — TC↔Rayleigh-Bénard analogy; conserved current J^ω, Nusselt Nu_ω vs Ta.
- **Lopez20** — Lopez, J. M., Feldmann, D., Rampp, M., Vela-Martín, A., Shi, L. & Avila, M. (2020). nsCouette — a high-performance code for DNS of turbulent Taylor–Couette flow. *SoftwareX* 11, 100395. DOI:10.1016/j.softx.2019.100395. arXiv:1908.00587. — Fourier(θ,z) + high-order explicit FD(r) (default 9-point), predictor–corrector + dynamic CFL, PPE; hybrid MPI+OpenMP + standalone C-CUDA GPU version; wave-speed validation to 1e-4. The architectural template for Family B's `nsCouette`-style layout.

### Time integration

- **ARS97** — Ascher, U. M., Ruuth, S. J. & Spiteri, R. J. (1997). Implicit-explicit Runge-Kutta methods for time-dependent PDEs. *Appl. Numer. Math.* 25(2–3), 151–167. DOI:10.1016/S0168-9274(97)00056-1. — the IMEX-RK family (schemes (1,1,1)…(4,4,3)): DIRK L-stable implicit tableau + explicit tableau; the source of A/C's IMEXRK222/IMEXRK3.
- **SMR91** — Spalart, P. R., Moser, R. D. & Rogers, M. M. (1991). Spectral methods for the Navier-Stokes equations with one infinite and two periodic directions. *J. Comput. Phys.* 96(2), 297–324. DOI:10.1016/0021-9991(91)90238-G. — RK3/CN splitting; pressure-eliminated formulation; the "Spalart" IMEXRK3 lineage (A=(8/15,5/12,3/4), B=(0,−17/60,−5/12)).
- **CK94** — Carpenter, M. H. & Kennedy, C. A. (1994). Fourth-order 2N-storage Runge-Kutta schemes. NASA TM-109112. — five-stage 4th-order 2N-storage RK; Williamson↔Butcher conversion.
- **KS80** — Kleiser, L. & Schumann, U. (1980). Treatment of incompressibility and boundary conditions in 3-D spectral simulations of plane channel flows. *Proc. 3rd GAMM Conf.*, Vieweg, 165–173 (unverified pagination). — influence-matrix method + tau correction (the basis of B's influence/capacitance matrix div-free treatment).
- **KM85** — Kim, J. & Moin, P. (1985). Application of a fractional-step method to incompressible Navier-Stokes equations. *J. Comput. Phys.* 59(2), 308–323. DOI:10.1016/0021-9991(85)90148-2. — fractional-step/projection; consistent intermediate-velocity BCs.
- **CM02** — Cox, S. M. & Matthews, P. C. (2002). Exponential time differencing for stiff systems. *J. Comput. Phys.* 176(2), 430–455. DOI:10.1006/jcph.2002.6995. — ETD/ETDRK4 (C's `ETDRK4` menu option); small-eigenvalue Taylor cutoff.
- **KT05** — Kassam, A.-K. & Trefethen, L. N. (2005). Fourth-order time-stepping for stiff PDEs. *SIAM J. Sci. Comput.* 26(4), 1214–1233. DOI:10.1137/S1064827502410633. — contour-integral stabilization of ETDRK4 (32–64 points); the practical ETDRK4.

### Spectral methods & verification

- **CHQZ06** — Canuto, C., Hussaini, M. Y., Quarteroni, A. & Zang, T. A. (2006). *Spectral Methods: Fundamentals in Single Domains.* Springer. DOI:10.1007/978-3-540-30726-6. https://link.springer.com/book/10.1007/978-3-540-30726-6 — Galerkin/collocation/tau formulations; Fourier/Chebyshev/Legendre.
- **Boyd01** — Boyd, J. P. (2001). *Chebyshev and Fourier Spectral Methods*, 2nd ed. Dover. ISBN 978-0-486-41183-5. — exponential convergence for analytic functions; resolution rules-of-thumb.
- **Tref00** — Trefethen, L. N. (2000). *Spectral Methods in MATLAB.* SIAM. DOI:10.1137/1.9780898719598. — 40 M-files; OS/KdV/Allen-Cahn worked examples; spectral-vs-algebraic convergence demonstration.
- **Mortensen18** — Mortensen, M. (2018). Shenfun: high performance spectral Galerkin computing platform. *J. Open Source Softw.* 3(31), 1071. DOI:10.21105/joss.01071. https://joss.theoj.org/papers/10.21105/joss.01071 — Family A's foundation; FEniCS-like weak forms, tensor-product spaces, direct Poisson/Helmholtz/Biharmonic solvers, MPI via mpi4py-fft (slab/pencil), Cython kernels.
- **Shen94** — Shen, J. (1994). Efficient spectral-Galerkin method I (Legendre). *SIAM J. Sci. Comput.* 15(6), 1489–1505. DOI:10.1137/0915089. — composite Legendre (Shen) bases.
- **Shen95** — Shen, J. (1995). Efficient spectral-Galerkin method II (Chebyshev). *SIAM J. Sci. Comput.* 16(1), 74–87. DOI:10.1137/0916006. — composite Chebyshev (Shen) bases.
- **SalariKnupp00** — Salari, K. & Knupp, P. (2000). Code Verification by the Method of Manufactured Solutions. SAND2000-1444, Sandia. DOI:10.2172/759450. https://www.osti.gov/biblio/759450 — MMS detects any order-of-accuracy coding error.
- **KnuppSalari02** — Knupp, P. & Salari, K. (2002). Code Verification by the Method of Manufactured Solutions. *J. Fluids Eng.* 124(1), 4–10. DOI:10.1115/1.1436090 (unverified, 403) — journal MMS version.
- **Roache98** — Roache, P. J. (1998). *Verification and Validation in Computational Science and Engineering.* Hermosa. ISBN 978-0-913478-08-0. — Verification vs Validation; Grid Convergence Index.
- **TaylorGreen37** — Taylor, G. I. & Green, A. E. (1937). Mechanism of the production of small eddies from large ones. *Proc. R. Soc. A* 158(895), 499–521. DOI:10.1098/rspa.1937.0036. — 2D TGV exact solution `u=sin x cos y·e^{−2νt}`, KE∝e^{−4νt} (temporal-accuracy oracle).
- **BMO83** — Brachet, M. E. et al. (1983). Small-scale structure of the Taylor–Green vortex. *J. Fluid Mech.* 130, 411–452. DOI:10.1017/S0022112083001159. — 3D TGV transition benchmark (≤256³ modes, Re up to 3000+).

### MHD shearing box & MRI

- **BH91** — Balbus, S. A. & Hawley, J. F. (1991). A powerful local shear instability in weakly magnetized disks. I. Linear analysis. *ApJ* 376, 214–222. DOI:10.1086/170270. — MRI linear analysis; max growth ~Ω, field-strength-independent; criterion dΩ²/dR<0.
- **BH98** — Balbus, S. A. & Hawley, J. F. (1998). Instability, turbulence, and enhanced transport in accretion disks. *Rev. Mod. Phys.* 70, 1–53. DOI:10.1103/RevModPhys.70.1. — canonical local MRI dispersion relation.
- **HGB95** — Hawley, J. F., Gammie, C. F. & Balbus, S. A. (1995). Local 3D MHD simulations of accretion disks. *ApJ* 440, 742. DOI:10.1086/175311. — shearing-box MHD; periodic(y,z) + shearing-periodic(x); channel mode is exact nonlinear MRI solution.
- **GX94** — Goodman, J. & Xu, G. (1994). Parasitic instabilities in magnetized, differentially rotating disks. *ApJ* 432, 213–223. DOI:10.1086/174562. — channel-mode parasites set MRI saturation.
- **LL07** — Lesur, G. & Longaretti, P.-Y. (2007). Impact of dimensionless numbers on the efficiency of MRI-induced turbulent transport. *MNRAS* 378(4), 1471–1480. DOI:10.1111/j.1365-2966.2007.11888.x. arXiv:0704.2943. — **α∝Pm^δ, δ∈[0.25,0.5]** (the differentiable Pm-sweep target of §V.1.5); SNOOPY code.
- **LL10** — Longaretti, P.-Y. & Lesur, G. (2010). MRI-driven turbulent transport: dissipation, channel modes and parasites. *A&A* 516, A51. DOI:10.1051/0004-6361/201014093. arXiv:1004.1384.
- **MS08** — Masada, Y. & Sano, T. (2008). Axisymmetric MRI in viscous accretion disks. *ApJ* 689, 1234. arXiv:0808.2338. — viscous/resistive MRI scalings; verified ideal oracles **s_max=(3/4)Ω, (k v_A)²=(15/16)Ω²**, κ²=2Ω(2Ω−S).
- **LFG10** — Latter, H. N., Fromang, S. & Gressel, O. (2010). MRI channel flows in vertically-stratified models of accretion disks. *MNRAS* 406, 848. arXiv:1004.0109. — conducting vs insulating/pseudo-vacuum magnetic BCs and their effect on the channel-mode spectrum.
- **SG10** — Stone, J. M. & Gardiner, T. A. (2010). Implementation of the shearing box approximation in Athena. *ApJS* 189(1), 142. DOI:10.1088/0067-0049/189/1/142. arXiv:1006.0139. — orbital advection + CT; CN for Coriolis/tidal source terms; epicyclic energy conserved to round-off; flux conserved to <0.03%.
- **Lesur15** — Lesur, G. (2015). Snoopy: general purpose spectral solver. ASCL:1505.022. https://ipag.osug.fr/~lesurg/snoopy.html — incompressible MHD/Boussinesq shearing box; Fourier pseudo-spectral + shearing-wave remap; RK3 + integrating-factor dissipation; 2/3 dealiasing; FFTW3, MPI/OpenMP.

<!-- SOLVER_SURVEY_AUDIT_APPENDIX_START -->

\newpage

# Appendix A — Source Anchor and Benchmark Audit

Generated 2026-06-07 from `solver_survey.md` content hash `78031aaed3e6dec5` by `_survey_audit/audit_solver_survey.py`.

This appendix replaces the earlier blanket claim that every load-bearing number had already been cross-checked. The audit is conservative: `VERIFIED` means the cited current-tree file and line(s) resolved and the cited evidence contained a nontrivial token or number from the statement; `PARTIAL` means the line exists or the claim is literature-cited but semantic support was not fully established by this automated pass; `UNSUPPORTED` means the cited file/line was missing, out of range, unreadable, or otherwise failed resolution.

Machine-readable ledgers are written to `_survey_audit/source_anchor_audit.csv` and `_survey_audit/benchmark_claim_audit.csv`; the complete ledgers are also reproduced below so the Markdown, TeX, and PDF artifacts carry the same status labels.

## A.1 Summary Counts

| Ledger | VERIFIED | PARTIAL | UNSUPPORTED | Total |
|---|---:|---:|---:|---:|
| Source-anchor support | 512 | 149 | 0 | 661 |
| Benchmark-style claims | 107 | 310 | 0 | 417 |

## A.2 Anchor Resolution Counts

| Anchor resolution status | Count |
|---|---:|
| `partial:ambiguous-file` | 92 |
| `resolved` | 569 |

### A.3 High-priority partial / unsupported entries

**Source anchors requiring human follow-up.**

- **A0003 [PARTIAL]:** line 95; `pcf_mhd_mri_shearpy.py:11` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0010 [PARTIAL]:** line 109; `base_flow.py:37-41` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0011 [PARTIAL]:** line 109; `pcf_fluctuations_jax.py:65-67` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0014 [PARTIAL]:** line 111; `pcf_mhd_mri_shearpy.py:11,102` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0016 [PARTIAL]:** line 111; `pcf_mhd_mri_shearpy_jax.py:70-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0028 [PARTIAL]:** line 116; `pcf_mhd_jax.py:63-64` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0030 [PARTIAL]:** line 117; `mhd.py:102-104` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0032 [PARTIAL]:** line 118; `mhd.py:278-282` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0037 [PARTIAL]:** line 152; `pcf_fluctuations_jax.py:54` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0042 [PARTIAL]:** line 155; `taylor_couette_dns_jax.py:159` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0046 [PARTIAL]:** line 163; `pcf_mhd_jax.py:63-64` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0076 [PARTIAL]:** line 316; `ChannelFlow2D.py:195` -> `fn_shenfun/demo/ChannelFlow2D.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0090 [PARTIAL]:** line 365; `pcf_fluctuations_corrected.py:195-197` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0132 [PARTIAL]:** line 574; `integrators.py:798-817` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0133 [PARTIAL]:** line 578; `integrators.py:787,816` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0135 [PARTIAL]:** line 580; `integrators.py:836-850` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0136 [PARTIAL]:** line 583; `integrators.py:852-870` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0137 [PARTIAL]:** line 587; `integrators.py:872-892` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0138 [PARTIAL]:** line 590; `integrators.py:819-833` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0139 [PARTIAL]:** line 594; `integrators.py:603-700` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0140 [PARTIAL]:** line 594; `integrators.py:665-669` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0141 [PARTIAL]:** line 596; `integrators.py:678-680` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0142 [PARTIAL]:** line 596; `integrators.py:692-693` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0144 [PARTIAL]:** line 657; `solver.py:225` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0149 [PARTIAL]:** line 673; `base_flow.py:23-24` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0150 [PARTIAL]:** line 684; `mesh.py:12-45` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0151 [PARTIAL]:** line 684; `mesh.py:43-45` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0152 [PARTIAL]:** line 684; `mesh.py:25` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0153 [PARTIAL]:** line 686; `mesh.py:109-117` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0154 [PARTIAL]:** line 686; `mesh.py:103-106` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0155 [PARTIAL]:** line 686; `tests/test_mesh.py:21-37` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0163 [PARTIAL]:** line 704; `spectral.py:23,37-38` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0174 [PARTIAL]:** line 726; `solver.py:215-216` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0175 [PARTIAL]:** line 731; `solver.py:225` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0176 [PARTIAL]:** line 731; `solver.py:217-218` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0181 [PARTIAL]:** line 743; `solver.py:594-606` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0197 [PARTIAL]:** line 785; `solver.py:495-502` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0198 [PARTIAL]:** line 787; `solver.py:226-242` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0199 [PARTIAL]:** line 787; `solver.py:232-233` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0200 [PARTIAL]:** line 787; `solver.py:235-240` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0201 [PARTIAL]:** line 787; `solver.py:129,263-272` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0202 [PARTIAL]:** line 789; `solver.py:331-339` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0203 [PARTIAL]:** line 789; `solver.py:295-314` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0204 [PARTIAL]:** line 794; `solver.py:357-420` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0205 [PARTIAL]:** line 794; `solver.py:316-320` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0206 [PARTIAL]:** line 794; `solver.py:416-419` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0207 [PARTIAL]:** line 796; `solver.py:474-482` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0208 [PARTIAL]:** line 796; `solver.py:422-463` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0209 [PARTIAL]:** line 798; `solver.py:225-227` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0210 [PARTIAL]:** line 798; `solver.py:252-253` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0211 [PARTIAL]:** line 798; `solver.py:325-383` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0212 [PARTIAL]:** line 798; `solver.py:288-320` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0213 [PARTIAL]:** line 798; `solver.py:262-265` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0214 [PARTIAL]:** line 798; `solver.py:269-274` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0218 [PARTIAL]:** line 810; `solver.py:200-203` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0278 [PARTIAL]:** line 932; `mhd.py:102-104` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0279 [PARTIAL]:** line 938; `mhd.py:452` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0281 [PARTIAL]:** line 942; `test_mhd.py:135-150` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0282 [PARTIAL]:** line 942; `solver.py:571-577` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0289 [PARTIAL]:** line 960; `test_mesh.py:21-48` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0296 [PARTIAL]:** line 1020; `tests/test_x64_default.py:7-10` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/test_x64_default.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0300 [PARTIAL]:** line 1052; `composite.py:475` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0320 [PARTIAL]:** line 1079; `docs/couette_fourier_layout.md:11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/docs/couette_fourier_layout.md`; resolved-line-exists-but-semantic-support-not-proven
- **A0327 [PARTIAL]:** line 1103; `functionspace.py:193-203` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/functionspace.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0335 [PARTIAL]:** line 1134; `coordinates.py:425-1227` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/coordinates.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0336 [PARTIAL]:** line 1139; `operators.py:118-1097` -> `partial:ambiguous-file`; partial:ambiguous-file
- **A0345 [PARTIAL]:** line 1187; `channelflow_kmm.py:352` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0348 [PARTIAL]:** line 1194; `channelflow_kmm.py:113-118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0354 [PARTIAL]:** line 1216; `tensorproductspace.py:76` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/tensorproductspace.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0357 [PARTIAL]:** line 1229; `pcf_fluctuations_jax.py:29-122` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0370 [PARTIAL]:** line 1294; `taylor_couette_linear_jax.py:37-89` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0371 [PARTIAL]:** line 1298; `taylor_couette_dns_jax.py:159` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0390 [PARTIAL]:** line 1385; `channelflow_kmm.py:75` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-claim-numbers-not-found
- **A0400 [PARTIAL]:** line 1421; `imex_rk.py:73,144` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0401 [PARTIAL]:** line 1421; `rk4.py:13` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/rk4.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0402 [PARTIAL]:** line 1421; `base.py:262` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0408 [PARTIAL]:** line 1473; `test_taylor_couette_linear_jax.py:16-26` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_taylor_couette_linear_jax.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0409 [PARTIAL]:** line 1565; `ChannelFlow.py:147-164` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/ChannelFlow.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0413 [PARTIAL]:** line 1566; `torchcouette/solver.py:553-594` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; resolved-line-exists-but-semantic-support-not-proven
- **A0423 [PARTIAL]:** line 1571; `pcf_mhd_mri_shearpy.py:12-15` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-semantic-support-not-proven

Additional source-anchor follow-ups are listed in the full ledger below (149 total).


**Benchmark claims requiring human follow-up.**

- **B0001 [PARTIAL]:** line 12; benchmark-style numeric claim without same-line source anchor; claim: Provenance. Generated 2026-06-07 from (i) a direct reading of the three solver codebases in the 'cfd' repository and (ii) the classic literature. Load-bearing implementation claims carry 'file:line' anchors into the source where availa…
- **B0002 [PARTIAL]:** line 37; benchmark-style numeric claim without same-line source anchor; claim: - Missing test coverage now: no target-tree Taylor-Green/TGV harness was found in 'fn-shenfun/demo', 'fn-openpipeflow-122/torch', or 'fork-jaxfun/(examples,tests)'. F6 remains a required foundational test to add, not evidence of curre…
- **B0003 [PARTIAL]:** line 55; benchmark-style numeric claim without same-line source anchor; claim: - A = shenfun — spectral Galerkin (composite/Shen bases that bake boundary conditions into the trial space), Python on CPU with MPI ('mpi4py'/'mpi4py-fft'), float64. This is the spectral oracle: exponential convergence, divergence…
- **B0004 [PARTIAL]:** line 59; benchmark-style numeric claim without same-line source anchor; claim: The families are intentionally not identical discretizations. Family A is the reference against which B (4th-order-family FD, ~8th-order interior) and C (spectral, JAX-native) are validated. Parity means agreement on formulation-indep…
- **B0005 [PARTIAL]:** line 73; benchmark-style numeric claim without same-line source anchor; claim: / Precision floor / float64 / complex128 (float32 validated) / x64-by-default /
- **B0006 [PARTIAL]:** line 75; benchmark-style numeric claim without same-line source anchor; claim: Run environments are disjoint (no live cross-import): A → '/home/nauman/miniconda3/envs/shenfun/bin/python' (shenfun 4.2.2, no torch); B → '/home/nauman/miniconda3/envs/huggingface/bin/python' (torch, no shenfun); C → '/home/nauman/cfd…
- **B0007 [PARTIAL]:** line 98; benchmark-style numeric claim without same-line source anchor; claim: - Canonical Lorentz force. +,boldsymbol(J)timesboldsymbol(B) with prefactor 1, in Alfvén / Lorentz–Heaviside units rho = mu-0 = 1 (so boldsymbol(B) is in velocity units and the Alfvén speed is v-A = /boldsymbol(B)/)…
- **B0009 [PARTIAL]:** line 131; benchmark-style numeric claim without same-line source anchor; claim: 3. Override the Lorentz prefactor to 1 for any oracle/parity comparison: set B's 'lorentz-prefactor = 1' instead of Ha2/(Re,Rm) (channel) / Ha2/Pm (couette).
- **B0010 [PARTIAL]:** line 133; benchmark-style numeric claim without same-line source anchor; claim: All cross-family comparisons should operate on canonical-frame observables (growth rates, energies, stresses), which are frame-invariant once the adapter is applied. This should be implemented as 'parity/conventions.py::to-canonical()'…
- **B0012 [PARTIAL]:** line 167; benchmark-style numeric claim without same-line source anchor; claim: For oracle comparisons against A/C, B's Lorentz prefactor is overridden to 1 (§0.2.3).
- **B0013 [PARTIAL]:** line 177; benchmark-style numeric claim without same-line source anchor; claim: The implemented shearing-box source terms (A and C; the acceptance target for B, §III.1) are, in canonical axes:
- **B0016 [PARTIAL]:** line 195; literature-cited benchmark not independently checked in this run; claim: / OS critical Reynolds (cross-family) / Re-(text(crit)) = 5772.22 [Orszag71] / rel <10(-2) /
- **B0020 [PARTIAL]:** line 200; benchmark-style numeric claim without same-line source anchor; claim: The conducting/insulating sign flip (+0.00332 vs -2.76times10(-4)) is reproduced by A and C only (B has no insulating walls); pin 'magnetic-bc' identically on both sides before comparing (§Part IV/SR-6, SR-9).
- **B0021 [PARTIAL]:** line 234; benchmark-style numeric claim without same-line source anchor; claim: / s / complex growth rate / eigenvalue / propto e(st); text(Re)(s)>0 unstable /
- **B0022 [PARTIAL]:** line 256; benchmark-style numeric claim without same-line source anchor; claim: Family A contains two complementary stacks throughout: a nonlinear pseudo-spectral DNS and a dense linear-stability / non-modal layer (collocation and Galerkin) that shares Butcher tableaux with the DNS time-steppers. Family A…
- **B0024 [PARTIAL]:** line 373; benchmark-style numeric claim without same-line source anchor; claim: B=-mathrm(Re),alpha,mathrm(i),(K-alpha2 M), quad A = Q - 2alpha2 K + (alpha4 - 2alpha,mathrm(Re),mathrm(i))M - mathrm(i)alphamathrm(Re),(K2-alpha2 K1).
- **B0026 [PARTIAL]:** line 377; benchmark-style numeric claim without same-line source anchor; claim: c = 0.24707506017508621 + 0.0026644103710965817,mathrm(i), qquad text(tol ) 10(-12).
- **B0028 [PARTIAL]:** line 383; benchmark-style numeric claim without same-line source anchor; claim: Two operators target the generalized eigenproblem Lq=sMq for perturbations q(x)exp(s t+mathrm(i)k-y y+mathrm(i)k-z z) (k-y2+k-z2>0), base flow U(x)=U-(text(off))+U'x along mathbf(e)-y:
- **B0030 [PARTIAL]:** line 419; benchmark-style numeric claim without same-line source anchor; claim: Velocity uses the Dirichlet composite 'bc=(0,0)' (dim N-2); pressure the orthogonal space sliced to N-2 modes ('SP.slice = lambda: slice(0,N-2)'), giving the inf-sup-stable P-N/P-(N-2) pair with 'assert SP.dim()==SD.dim()'. Family…
- **B0032 [PARTIAL]:** line 484; benchmark-style numeric claim without same-line source anchor; claim: - Hagen–Poiseuille: u-z(r)=dfrac(f-z)(4nu)(R2-r2), flow rate Q=dfrac(pi R4 f-z)(8nu).
- **B0033 [PARTIAL]:** line 485; literature-cited benchmark not independently checked in this run; claim: - Womersley (-partial-z p=Kcosomega t, rho=1, alpha=Rsqrt(omega/nu), mathrm(i)(3/2)=e(mathrm(i)3pi/4)): displaystyle u-z(r,t)=mathrm(Re)!left(frac(K)(mathrm(i)omega)!left[1-frac(J-0(mathrm(i)(3/2)alp…
- **B0034 [PARTIAL]:** line 486; benchmark-style numeric claim without same-line source anchor; claim: - Bessel viscous decay: u-z(r,0)=J-0(j-(0,n)r/R), decaying as exp(-nu j-(0,n)2 t/R2).
- **B0038 [PARTIAL]:** line 530; benchmark-style numeric claim without same-line source anchor; claim: with scalar Laplacian L-p=partial-(rr)+tfrac1rpartial-r-(m2/r2+k-z2), vector diagonal L-v=L-p-1/r2, cross terms pm2mathrm(i)m/r2, and 2a=2Omega+rOmega' (const). The radial induction has no shear source (only advecti…
- **B0041 [PARTIAL]:** line 630; benchmark-style numeric claim without same-line source anchor; claim: 12. Linear operators have singular mass M (zero pressure/phi mass) → filter infinite eigenvalues with 'FINITE-CAP=1e8'.
- **B0042 [PARTIAL]:** line 639; benchmark-style numeric claim without same-line source anchor; claim: Two contradictions from §0.2 are honored throughout: (i) family B's axis labels are swapped relative to the canonical frame (B uses streamwise='x', wall-normal='y'; the canonical frame uses wall-normal='x', streamwise='y') — apply the pl…
- **B0043 [PARTIAL]:** line 684; one or more local source anchors only partially verified; claim: The FD weights mirror OpenPipeFlow's 'mes-weights' ('mesh.py:12-45'). For target x-0 and stencil (x-j), build A-(:,0)=1, A-(:,j)=A-(:,j-1)cdot(x-x-0)/j so that A-(j)=(x-x-0)j/j!, then solve the transposed system A(math…
- **B0044 [PARTIAL]:** line 686; one or more local source anchors only partially verified; claim: Stencil width and order. 'KL' is the half-bandwidth; the interior stencil is min(2,KL+1,,N) centered points ('mesh.py:109-117'), default 'KL=4' Rightarrow 9-point centered stencil. With 9 points the interior FD is polynomia…
- **B0051 [PARTIAL]:** line 888; benchmark-style numeric claim without same-line source anchor; claim: For oracle cross-checks against the canonical +mathbf Jtimesmathbf B with prefactor 1 (§0.2), pass 'lorentz-prefactor=1' explicitly.
- **B0055 [PARTIAL]:** line 948; benchmark-style numeric claim without same-line source anchor; claim: - Backend: pure PyTorch (FFT + dense/banded linear solves), CPU default everywhere, device-agnostic ('device' param). CUDA is supported by construction (all ops are tensor ops with explicit 'device='); it is only special-cased in t…
- **B0056 [PARTIAL]:** line 949; benchmark-style numeric claim without same-line source anchor; claim: - Precision: 'complex128' default (real 'float64') in all three solvers; 'complex64'/'float32' supported and validated (channel 'test-float32.py': roundtrip error <5times10(-5); solver step preserves 'complex64', div <10(-4)…
- **B0062 [PARTIAL]:** line 960; one or more local source anchors only partially verified; claim: / Mesh polynomial exactness / err <10(-7), deg 0..8; int1(=)2,int y(=)0,int y2(=)2/3 / 'test-mesh.py:21-48' /
- **B0064 [PARTIAL]:** line 962; benchmark-style numeric claim without same-line source anchor; claim: / Couette nsCouette reference / m-r(=)32,m-theta(=)16,m-(z0)(=)16,k-(theta0)(=)6.0,k-(z0)(=)2.6179938779914944,eta(=)0.868,Re-i(=)200,Re-o(=)-200 / 'test-fortran-reference-config.py' /
- **B0065 [PARTIAL]:** line 963; benchmark-style numeric claim without same-line source anchor; claim: / Couette laminar 1000-step / div <10(-8), E-(rm pert)<10(-12), Nu-i,Nu-oapprox1 (atol 10(-8)) / 'test-integration-laminar.py' /
- **B0066 [PARTIAL]:** line 964; benchmark-style numeric claim without same-line source anchor; claim: / Couette MHD / div <10(-8), divB-(Linfty)<10(-8), walls mathbf b(=)0 (atol 10(-12)) / 'test-mhd.py' (N(=)8,eta(=)0.868,Re-i(=)20,Re-o(=)-20,Pm(=)2,Ha(=)1) /
- **B0069 [PARTIAL]:** line 968; literature-cited benchmark not independently checked in this run; claim: Cross-family OS comparisons use the published Re-(rm crit)=5772.22 [Orszag71] at rel <10(-2) (the within-family golden c above is family-specific and not portable; see the §0 hand-off note). For all MRI/rotation acceptance (SR-1, S…
- **B0070 [PARTIAL]:** line 976; benchmark-style numeric claim without same-line source anchor; claim: 5. Corrector differs: channel fixed ('corrector-iterations', default 1); couette/pipe iterated to 'tol=1e-10' ('max-corrector-iters=3') with 'StepInfo'.
- **B0071 [PARTIAL]:** line 1006; benchmark-style numeric claim without same-line source anchor; claim: integrators; §I.C.4 the JAX-specific compute capabilities; §I.C.5 golden numbers; §I.C.6 the explicit pipe gap and
- **B0072 [PARTIAL]:** line 1129; benchmark-style numeric claim without same-line source anchor; claim: 'generalized-eig(L,M)' via host 'scipy.linalg.eig' with finite-eigenvalue caps 'MODAL-FINITE-CAP = 1e6',
- **B0073 [PARTIAL]:** line 1425; literature-cited benchmark not independently checked in this run; claim: adjoint/minimal-seed loops [PWK12], validated against finite differences ('test-differentiability-jax.py').
- **B0074 [PARTIAL]:** line 1453; benchmark-style numeric claim without same-line source anchor; claim: These are the within-family acceptance constants (tolerance ladder per §1 hand-off notes). Use Family-C goldens
- **B0075 [PARTIAL]:** line 1454; benchmark-style numeric claim without same-line source anchor; claim: only against C; cross-family physics oracles use published values at rel < 1e-2.
- **B0077 [PARTIAL]:** line 1457; benchmark-style numeric claim without same-line source anchor; claim: amp 0.05, one step, x64, 'rtol=1e-10'):
- **B0078 [PARTIAL]:** line 1459; benchmark-style numeric claim without same-line source anchor; claim: Epert = 0.21836099019180652
- **B0079 [PARTIAL]:** line 1460; benchmark-style numeric claim without same-line source anchor; claim: Etot = 52.85625108205688
- **B0080 [PARTIAL]:** line 1461; benchmark-style numeric claim without same-line source anchor; claim: divL2 = 7.183953559387109e-17 (atol 5e-15)
- **B0081 [PARTIAL]:** line 1462; benchmark-style numeric claim without same-line source anchor; claim: u-top = 0.968160239435768
- **B0082 [PARTIAL]:** line 1463; benchmark-style numeric claim without same-line source anchor; claim: u-bot = -0.9681602394357679
- **B0083 [PARTIAL]:** line 1464; benchmark-style numeric claim without same-line source anchor; claim: mean-shear = 1.0000000004699001
- **B0084 [PARTIAL]:** line 1467; benchmark-style numeric claim without same-line source anchor; claim: PCF MHD ('test-pcf-mhd-jax.py'): 'Epert>0', 'Emag>0', 'divL2<1e-4', 'divB-L2<1e-5'; float64-invariant
- **B0085 [PARTIAL]:** line 1468; benchmark-style numeric claim without same-line source anchor; claim: 'magnetic-divergence-l2 < 1e-12'.
- **B0087 [PARTIAL]:** line 1471; benchmark-style numeric claim without same-line source anchor; claim: 'alpha/reynolds-xy/maxwell-xy', 'q-shear == 1.0' (at Omega=S=1).
- **B0088 [PARTIAL]:** line 1474; benchmark-style numeric claim without same-line source anchor; claim: R-1=1,R-2=2,Omega-1=1,Omega-2=0, nu=0.002, N=12, Legendre, m=0, k-z=3, 'rtol=1e-11'): leading
- **B0089 [PARTIAL]:** line 1475; benchmark-style numeric claim without same-line source anchor; claim: eigenvalue 0.36073352898670064 + 4.8times10(-22)i (5 more in source).
- **B0090 [PARTIAL]:** line 1478; benchmark-style numeric claim without same-line source anchor; claim: 'CircularCouette(1,2,1,0.51.5)', B-0=0.1, nu=eta=0.001, N=12, Legendre, m=0, k-z=3, 'rtol=1e-11'):
- **B0091 [PARTIAL]:** line 1479; benchmark-style numeric claim without same-line source anchor; claim: - conducting leading: 0.25628761535339467 + 1.6times10(-16)i;
- **B0092 [PARTIAL]:** line 1480; benchmark-style numeric claim without same-line source anchor; claim: - insulating leading: 0.25995005500337837 + 5.2times10(-17)i;
- **B0093 [PARTIAL]:** line 1481; benchmark-style numeric claim without same-line source anchor; claim: - local Keplerian-MRI optimum: s-(max)/Omegaapprox0.75 (rel 1e-3), (kv-A)2/Omega2approx15/16 (rel 2e-3).
- **B0094 [PARTIAL]:** line 1484; benchmark-style numeric claim without same-line source anchor; claim: ('/p[0,0]/<1e-7'); eigenmode growth-rate matches the linear solver (axisym hydro 'rtol=1e-7'; 3-D hydro/MRI
- **B0095 [PARTIAL]:** line 1485; benchmark-style numeric claim without same-line source anchor; claim: 'rtol=1e-6'; 100 steps, x64); pinned-saddle LU residual '<1e-11'; continuity residual '<1e-18' (x64);
- **B0096 [PARTIAL]:** line 1486; benchmark-style numeric claim without same-line source anchor; claim: nablacdot u, nablacdot b < 1e-7.
- **B0097 [PARTIAL]:** line 1489; benchmark-style numeric claim without same-line source anchor; claim: single-step amplitude 'rtol=2e-3', full-state directional 'rtol=2e-3', multi-step finite-amplitude 'rtol=5e-3';
- **B0098 [PARTIAL]:** line 1490; benchmark-style numeric claim without same-line source anchor; claim: energy-tangent projection orthogonality 'atol=1e-10'.
- **B0099 [PARTIAL]:** line 1493; literature-cited benchmark not independently checked in this run; claim: PCF hydro transient growth Re=1000,alpha=0,beta=1.66: literature G=1165.2 vs computed G=1165.93 [RH93];
- **B0100 [PARTIAL]:** line 1494; literature-cited benchmark not independently checked in this run; claim: TC hydro onset eta=0.5 outer-stationary Re-c=68.186 (a-c=3.167, k-(z,c)=3.167) [Taylor23]; ideal local
- **B0101 [PARTIAL]:** line 1495; literature-cited benchmark not independently checked in this run; claim: Keplerian MRI s-(max)/Omega=0.7500, (kv-A)2/Omega2=0.9373 (theory 0.75, 15/16) [BH91]; PCF linearly stable
- **B0102 [PARTIAL]:** line 1496; benchmark-style numeric claim without same-line source anchor; claim: for all Re (Romanov); insulating MRI scan best growth -2.758times10(-4).
- **B0103 [PARTIAL]:** line 1504; benchmark-style numeric claim without same-line source anchor; claim: skip in C (F1d/F1e, S2-pipe; §3 gap matrix, §IV hand-off notes). This is the single geometry gap of C relative
- **B0104 [PARTIAL]:** line 1538; benchmark-style numeric claim without same-line source anchor; claim: - Family A (the other spectral oracle, same tableaux/Lorentz=1): §I.A.1–I.A.6.
- **B0105 [PARTIAL]:** line 1570; benchmark-style numeric claim without same-line source anchor; claim: / Pipe MHD / A (deferred, low parity value; 'PLAN…:182-184') / A — pipe is hydro-only (B-PIPE §11) / A (no pipe) /
- **B0106 [PARTIAL]:** line 1576; benchmark-style numeric claim without same-line source anchor; claim: / GPU / A — CPU/MPI only (A-PCF §10) / P — device-agnostic torch (CUDA in benchmarks; B-MHD §6.1) / P — JAX/XLA; 'cuda13' extra configured ('pyproject.toml') /
- **B0107 [PARTIAL]:** line 1580; one or more local source anchors only partially verified; claim: / Double precision / P — float64 default / P — complex128 default; float32 validated ('solver.py:87') / P — x64-by-default ('--init--.py:8') /
- **B0108 [PARTIAL]:** line 1582; one or more local source anchors only partially verified; claim: / Convergence-order tests / P — golden eig 1e-12, MMS ('OrrSommerfeld-eigs.py:183') / P — OS golden 1e-4, mesh poly-exact deg 8 ('test-mesh.py:21-37') / P — MMS self-asserts ('poisson1D.py:46') /
- **B0109 [PARTIAL]:** line 1615; benchmark-style numeric claim without same-line source anchor; claim: fixed parameters: conducting gives growth +0.00332 (at mathrm(Rm)=24.7, S=4.11,
- **B0110 [PARTIAL]:** line 1639; benchmark-style numeric claim without same-line source anchor; claim: (4) Compute/autograd/JIT asymmetry. A is the spectral oracle but has no GPU, no
- **B0111 [PARTIAL]:** line 1669; benchmark-style numeric claim without same-line source anchor; claim: Target files (new): 'fn-openpipeflow-122/parity/conventions.md',
- **B0112 [PARTIAL]:** line 1674; benchmark-style numeric claim without same-line source anchor; claim: not reinvent eigenvalue matching ('PLAN…:67-71,120'). Gate: the adapter round-trips
- **B0113 [PARTIAL]:** line 1709; benchmark-style numeric claim without same-line source anchor; claim: Target files. 'fn-openpipeflow-122/torchchannel/torchchannel/solver.py'
- **B0114 [PARTIAL]:** line 1716; benchmark-style numeric claim without same-line source anchor; claim: seed a uniform (k=0) mode and assert the epicyclic oscillation SR-1:
- **B0115 [PARTIAL]:** line 1723; benchmark-style numeric claim without same-line source anchor; claim: to 'rel<1e-2, abs<5e-4' — byte-for-byte the shearpy assertion
- **B0116 [PARTIAL]:** line 1735; one or more local source anchors only partially verified; claim: ('test-theory.py:135-178'). Gate: rate / frequency match to 'rel<1e-2' with an explicit

Additional benchmark follow-ups are listed in the full ledger below (310 total).


## A.4 Complete Source-Anchor Ledger

- **A0001 [VERIFIED]:** survey line 35; `torchchannel/mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=metadata,terms; claim: - Implemented and source-supported now: A and C have PCF MHD/MRI source terms; B channel/couette MHD advances induction and Lorentz coupling, but B MRI parameters are metadata only ('torchchannel/mhd.py:71-74'); A a…
- **A0002 [VERIFIED]:** survey line 90; `pcf_mhd_mri_shearpy.py:7-9` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=azimuthal,component,coordinate,direction,gradient;numbers=0,1,2; claim: This matches the shenfun shearing-box map ('pcf-mhd-mri-shearpy.py:7-9'): "component 0 / coordinate x: radial, wall-normal, shear-gradient direction; component 1 / coordinate y: azimuthal, streamwise, wall-motion direct…
- **A0003 [PARTIAL]:** survey line 95; `pcf_mhd_mri_shearpy.py:11` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-claim-numbers-not-found; claim: - Shearing-box / MRI (shearpy): sigma = -S, i.e. boldsymbol(U)-b(x) = -S,x,boldsymbol(e)-y (verified at 'pcf-mhd-mri-shearpy.py:11', ':102': 'self.Ub = -self.shear-rate  self.X[0]', 'self.dUb-dx = -self.sh…
- **A0004 [VERIFIED]:** survey line 95; `:102` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=self,shear_rate;numbers=102,0; claim: - Shearing-box / MRI (shearpy): sigma = -S, i.e. boldsymbol(U)-b(x) = -S,x,boldsymbol(e)-y (verified at 'pcf-mhd-mri-shearpy.py:11', ':102': 'self.Ub = -self.shear-rate  self.X[0]', 'self.dUb-dx = -self.sh…
- **A0005 [VERIFIED]:** survey line 100; `pcf_mhd_mri_shearpy.py:107-108` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega,kappa,kappa2,omega,q_shear;numbers=2,2,2,2,2; claim: - Canonical nondimensional groups. Re = UL/nu, Rm = UL/eta-(text(mag)), Pm = nu/eta-(text(mag)) = Rm/Re, Hartmann Ha = B-0 L /sqrt(nu,eta-(text(mag))), Lundquist S-L = B-0 L /eta-(text(mag)).…
- **A0006 [VERIFIED]:** survey line 108; `ChannelFlow.py:9-11` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=normal,spanwise,streamwise;numbers=0,1,0,1; claim: / Axis order / 0 = x wall-normal, 1 = y streamwise, 2 = z spanwise ('ChannelFlow.py:9-11') / x = streamwise, y = wall-normal, z = spanwise ('base-flow.py:25-50'; swapped vs canonical) / 0 = x wall-…
- **A0007 [VERIFIED]:** survey line 108; `base_flow.py:25-50` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; tokens=normal;numbers=0,1,2,0,1; claim: / Axis order / 0 = x wall-normal, 1 = y streamwise, 2 = z spanwise ('ChannelFlow.py:9-11') / x = streamwise, y = wall-normal, z = spanwise ('base-flow.py:25-50'; swapped vs canonical) / 0 = x wall-…
- **A0008 [VERIFIED]:** survey line 108; `channelflow_kmm.py:58-59` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=normal,spanwise,streamwise;numbers=0,1,2,0,1; claim: / Axis order / 0 = x wall-normal, 1 = y streamwise, 2 = z spanwise ('ChannelFlow.py:9-11') / x = streamwise, y = wall-normal, z = spanwise ('base-flow.py:25-50'; swapped vs canonical) / 0 = x wall-…
- **A0009 [VERIFIED]:** survey line 109; `pcf_fluctuations_corrected.py:130-135` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=component,streamwise; claim: / PCF hydro base flow / boldsymbol(U)-b = +U-(text(wall)),x,boldsymbol(e)-y ('pcf-fluctuations-corrected.py:130-135') / U = +y (streamwise component = x) ('base-flow.py:37-41') / boldsymbol(U)-b = +U-(…
- **A0010 [PARTIAL]:** survey line 109; `base_flow.py:37-41` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / PCF hydro base flow / boldsymbol(U)-b = +U-(text(wall)),x,boldsymbol(e)-y ('pcf-fluctuations-corrected.py:130-135') / U = +y (streamwise component = x) ('base-flow.py:37-41') / boldsymbol(U)-b = +U-(…
- **A0011 [PARTIAL]:** survey line 109; `pcf_fluctuations_jax.py:65-67` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / PCF hydro base flow / boldsymbol(U)-b = +U-(text(wall)),x,boldsymbol(e)-y ('pcf-fluctuations-corrected.py:130-135') / U = +y (streamwise component = x) ('base-flow.py:37-41') / boldsymbol(U)-b = +U-(…
- **A0012 [VERIFIED]:** survey line 110; `OrrSommerfeld.py:14,30` -> `fn_shenfun/demo/OrrSommerfeld.py`; numbers=1,2,-2,1,2; claim: / Poiseuille base / 1-x2, dp/dy = -2/Re ('OrrSommerfeld.py:14,30') / 1-y2, const-flux 4/3 ('base-flow.py:42-46') / 1-x2 (OS path, mirrors A) /
- **A0013 [VERIFIED]:** survey line 110; `base_flow.py:42-46` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; tokens=Poiseuille;numbers=1,2,-2,1,2; claim: / Poiseuille base / 1-x2, dp/dy = -2/Re ('OrrSommerfeld.py:14,30') / 1-y2, const-flux 4/3 ('base-flow.py:42-46') / 1-x2 (OS path, mirrors A) /
- **A0014 [PARTIAL]:** survey line 111; `pcf_mhd_mri_shearpy.py:11,102` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / MRI/shearpy base / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-S ('pcf-mhd-mri-shearpy.py:11,102') / N/A — metadata only ('mhd.py:71-74') / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-…
- **A0015 [VERIFIED]:** survey line 111; `mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=metadata; claim: / MRI/shearpy base / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-S ('pcf-mhd-mri-shearpy.py:11,102') / N/A — metadata only ('mhd.py:71-74') / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-…
- **A0016 [PARTIAL]:** survey line 111; `pcf_mhd_mri_shearpy_jax.py:70-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / MRI/shearpy base / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-S ('pcf-mhd-mri-shearpy.py:11,102') / N/A — metadata only ('mhd.py:71-74') / boldsymbol(U)-b = -S,x,boldsymbol(e)-y, dU-b/dx=-…
- **A0017 [VERIFIED]:** survey line 112; `taylor_couette_linear.py:89-91` -> `fn_shenfun/demo/taylor_couette_linear.py`; numbers=2,2,2,2,1; claim: / TC base flow / V(r)=ar+b/r, a=(Omega-2 R-22-Omega-1 R-12)/(R-22-R-12) ('taylor-couette-linear.py:89-91') / u-theta=ar+b/r, a=(Re-o-eta,Re-i)/(1+eta) ('base-flow.py:18-25') / V(r)=ar+b/r ('taylo…
- **A0018 [VERIFIED]:** survey line 112; `base_flow.py:18-25` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; tokens=theta;numbers=2,2,2,2,1; claim: / TC base flow / V(r)=ar+b/r, a=(Omega-2 R-22-Omega-1 R-12)/(R-22-R-12) ('taylor-couette-linear.py:89-91') / u-theta=ar+b/r, a=(Re-o-eta,Re-i)/(1+eta) ('base-flow.py:18-25') / V(r)=ar+b/r ('taylo…
- **A0019 [VERIFIED]:** survey line 112; `taylor_couette_linear_jax.py:37-89` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; numbers=2,2,2,2,1; claim: / TC base flow / V(r)=ar+b/r, a=(Omega-2 R-22-Omega-1 R-12)/(R-22-R-12) ('taylor-couette-linear.py:89-91') / u-theta=ar+b/r, a=(Re-o-eta,Re-i)/(1+eta) ('base-flow.py:18-25') / V(r)=ar+b/r ('taylo…
- **A0020 [VERIFIED]:** survey line 113; `pipe_flow_dns.py:473-475` -> `fn_shenfun/demo/pipe_flow_dns.py`; numbers=4,2,2,1,2; claim: / Pipe base flow / u-z=(f-z/4nu)(R2-r2) ('pipe-flow-dns.py:473-475') / U=1-r2, b-(text(hpf))=2r ('solver.py:200-203') / N/A — no pipe /
- **A0021 [VERIFIED]:** survey line 113; `solver.py:200-203` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; numbers=2,2,1,2,2; claim: / Pipe base flow / u-z=(f-z/4nu)(R2-r2) ('pipe-flow-dns.py:473-475') / U=1-r2, b-(text(hpf))=2r ('solver.py:200-203') / N/A — no pipe /
- **A0022 [VERIFIED]:** survey line 115; `pcf_mhd_divfree.py:325-333` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; tokens=Lorentz;numbers=1,2,2,1,1; claim: / Lorentz prefactor / 1 ('pcf-mhd-divfree.py:325-333') / Ha2/(Re,Rm) channel ('mhd.py:100-101'); Ha2/Pm couette ('mhd.py:79') → set explicit override = 1 for oracle tests / 1 ('pcf-mhd-jax.py:175-176') /
- **A0023 [VERIFIED]:** survey line 115; `mhd.py:100-101` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Lorentz,prefactor;numbers=1,1,1; claim: / Lorentz prefactor / 1 ('pcf-mhd-divfree.py:325-333') / Ha2/(Re,Rm) channel ('mhd.py:100-101'); Ha2/Pm couette ('mhd.py:79') → set explicit override = 1 for oracle tests / 1 ('pcf-mhd-jax.py:175-176') /
- **A0024 [VERIFIED]:** survey line 115; `mhd.py:79` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=1,1,1; claim: / Lorentz prefactor / 1 ('pcf-mhd-divfree.py:325-333') / Ha2/(Re,Rm) channel ('mhd.py:100-101'); Ha2/Pm couette ('mhd.py:79') → set explicit override = 1 for oracle tests / 1 ('pcf-mhd-jax.py:175-176') /
- **A0025 [VERIFIED]:** survey line 115; `pcf_mhd_jax.py:175-176` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; tokens=Lorentz;numbers=1,1,1; claim: / Lorentz prefactor / 1 ('pcf-mhd-divfree.py:325-333') / Ha2/(Re,Rm) channel ('mhd.py:100-101'); Ha2/Pm couette ('mhd.py:79') → set explicit override = 1 for oracle tests / 1 ('pcf-mhd-jax.py:175-176') /
- **A0026 [VERIFIED]:** survey line 116; `pcf_mhd_divfree.py:91-102` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; numbers=1,1,1; claim: / Pm / Rm convention / Rm=U/eta, Pm=nu/eta; Rmleftarrow Re if unset ('pcf-mhd-divfree.py:91-102') / Rm=None ⇒ Rm=Re,Pm; else Pm=Rm/Re; mag-diff =1/Rm (channel) or 1/Pm (couette, viscosity=1) ('m…
- **A0027 [VERIFIED]:** survey line 116; `mhd.py:89-94,155` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=1,1,1; claim: / Pm / Rm convention / Rm=U/eta, Pm=nu/eta; Rmleftarrow Re if unset ('pcf-mhd-divfree.py:91-102') / Rm=None ⇒ Rm=Re,Pm; else Pm=Rm/Re; mag-diff =1/Rm (channel) or 1/Pm (couette, viscosity=1) ('m…
- **A0028 [PARTIAL]:** survey line 116; `pcf_mhd_jax.py:63-64` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; resolved-line-exists-but-claim-numbers-not-found; claim: / Pm / Rm convention / Rm=U/eta, Pm=nu/eta; Rmleftarrow Re if unset ('pcf-mhd-divfree.py:91-102') / Rm=None ⇒ Rm=Re,Pm; else Pm=Rm/Re; mag-diff =1/Rm (channel) or 1/Pm (couette, viscosity=1) ('m…
- **A0029 [VERIFIED]:** survey line 117; `pcf_mhd_mri_shearpy.py:107-108` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega,kappa,kappa2,omega,q_shear;numbers=2,2,2,2,2; claim: / Rotation/shear symbols / Omega='omega', S='shear-rate', kappa2=2Omega(2Omega-S) ('pcf-mhd-mri-shearpy.py:107-108') / 'omega'/'shear-rate' stored, 'q-shear' only (NO source) ('mhd.py:102-104') / 'omega'=…
- **A0030 [PARTIAL]:** survey line 117; `mhd.py:102-104` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Rotation/shear symbols / Omega='omega', S='shear-rate', kappa2=2Omega(2Omega-S) ('pcf-mhd-mri-shearpy.py:107-108') / 'omega'/'shear-rate' stored, 'q-shear' only (NO source) ('mhd.py:102-104') / 'omega'=…
- **A0031 [VERIFIED]:** survey line 117; `pcf_mhd_mri_shearpy_jax.py:73-74` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=Omega,kappa,kappa2,omega,q_shear;numbers=2,2,2,2,2; claim: / Rotation/shear symbols / Omega='omega', S='shear-rate', kappa2=2Omega(2Omega-S) ('pcf-mhd-mri-shearpy.py:107-108') / 'omega'/'shear-rate' stored, 'q-shear' only (NO source) ('mhd.py:102-104') / 'omega'=…
- **A0032 [PARTIAL]:** survey line 118; `mhd.py:278-282` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Magnetic BC / conducting (Ain TD3 DNS; Robin/Neumann linear) + insulating flux-fn (TC linear m=0) / homogeneous b=0 only (no insulating) ('mhd.py:278-282') / conducting (DNS) + insulating flux-fn (TC linea…
- **A0033 [VERIFIED]:** survey line 125; `torchchannel/base_flow.py:37-41` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; tokens=normal; claim: - B puts the streamwise velocity in component x as a function of wall-normal y: U = y along boldsymbol(e)-x (verified at 'torchcouette/base-flow.py' analogue and 'torchchannel/base-flow.py:37-41').
- **A0034 [VERIFIED]:** survey line 137; `torchchannel/tests/test_mesh.py:21-37` -> `fn_openpipeflow-122/torchchannel/tests/test_mesh.py`; tokens=assert,degree,err_dy,err_dyy,exact;numbers=4,4,4,9,8; claim: - B's "4th-order FD" label vs. reality. B is labeled a 4th-order FD family, but the default half-bandwidth 'KL=4' gives a 9-point centered stencil that is formally 8th-order interior (first/second derivatives ac…
- **A0035 [VERIFIED]:** survey line 138; `torchchannel/mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Coriolis,diagnostic,metadata,omega,q_shear;numbers=1,2,1,1; claim: - B's MRI is metadata-only. In B, 'omega'/'shear-rate' set only the diagnostic 'q-shear' and add no Coriolis / base-shear / shear-induction source terms ('torchchannel/mhd.py:71-74'). Every MRI claim about B is…
- **A0036 [VERIFIED]:** survey line 152; `pcf_fluctuations_corrected.py:106-107` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_fluctuations_corrected.py`; tokens=Couette;numbers=1,-1,1,1,1; claim: - Plane Couette / channel (A, C): velocity scale U = U-(text(wall)), length L = h (half-gap, h=1 for xin[-1,1]). Then nu = U-(text(wall)),h/Re; with defaults U-(text(wall))=h=1, nu = 1/Re ('pcf-…
- **A0037 [PARTIAL]:** survey line 152; `pcf_fluctuations_jax.py:54` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-claim-numbers-not-found; claim: - Plane Couette / channel (A, C): velocity scale U = U-(text(wall)), length L = h (half-gap, h=1 for xin[-1,1]). Then nu = U-(text(wall)),h/Re; with defaults U-(text(wall))=h=1, nu = 1/Re ('pcf-…
- **A0038 [VERIFIED]:** survey line 152; `OrrSommerfeld.py:14` -> `fn_shenfun/demo/OrrSommerfeld.py`; numbers=1,1,1,1,-2; claim: - Plane Couette / channel (A, C): velocity scale U = U-(text(wall)), length L = h (half-gap, h=1 for xin[-1,1]). Then nu = U-(text(wall)),h/Re; with defaults U-(text(wall))=h=1, nu = 1/Re ('pcf-…
- **A0039 [VERIFIED]:** survey line 153; `torchchannel/solver.py:225` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_build_diffusion_system,diffusion;numbers=1,1.0,1,2; claim: - Plane channel (B): single parameter Re; viscosity enters as 1/Re in the diffusion operator ('torchchannel/solver.py:225', '-build-diffusion-system(1.0/Re)'). Couette base U=y; Poiseuille base U=1-y2 at co…
- **A0040 [VERIFIED]:** survey line 153; `torchchannel/base_flow.py:42-46` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; tokens=Poiseuille;numbers=1,1.0,1,2,4; claim: - Plane channel (B): single parameter Re; viscosity enters as 1/Re in the diffusion operator ('torchchannel/solver.py:225', '-build-diffusion-system(1.0/Re)'). Couette base U=y; Poiseuille base U=1-y2 at co…
- **A0041 [VERIFIED]:** survey line 154; `torchcouette/solver.py:256-260` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; numbers=1,1,1,1,1; claim: - Taylor–Couette (B, viscous units): radius ratio eta = R-1/R-2 with r-i = eta/(1-eta), r-o = 1/(1-eta) (gap =1); inner/outer Reynolds numbers Re-i, Re-o are the wall speeds directly. The diffusion LHS…
- **A0042 [PARTIAL]:** survey line 155; `taylor_couette_dns_jax.py:159` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: - Taylor–Couette (A, C, velocity units): Re = Omega-1 R-1 cdot text(gap)/nu ('taylor-couette-dns-jax.py:159'), base flow V(r)=ar+b/r.
- **A0043 [VERIFIED]:** survey line 156; `pipe_flow_dns.py:473-475` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=Hagen,Poiseuille,axial;numbers=4,2,2,4; claim: - Pipe (A): u-z = (f-z/4nu)(R2-r2) with axial body force f-z; Hagen–Poiseuille flux Q = pi R4 f-z/(8nu) ('pipe-flow-dns.py:473-475'). Family C has no pipe.
- **A0044 [VERIFIED]:** survey line 162; `pcf_mhd_divfree.py:91-102` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; numbers=1,1; claim: - A: Rm = U/eta-(text(mag)), eta-(text(mag)) = U-(text(wall))/Rm; if Rm unset, Rm leftarrow Re so Pm = 1 ('pcf-mhd-divfree.py:91-102'). Lorentz force +boldsymbol(J)timesboldsymbol(B) with pref…
- **A0045 [VERIFIED]:** survey line 162; `pcf_mhd_divfree.py:325-333` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; tokens=Lorentz,force;numbers=1,1; claim: - A: Rm = U/eta-(text(mag)), eta-(text(mag)) = U-(text(wall))/Rm; if Rm unset, Rm leftarrow Re so Pm = 1 ('pcf-mhd-divfree.py:91-102'). Lorentz force +boldsymbol(J)timesboldsymbol(B) with pref…
- **A0046 [PARTIAL]:** survey line 163; `pcf_mhd_jax.py:63-64` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; resolved-line-exists-but-claim-numbers-not-found; claim: - C: identical to A — Rm leftarrow Re if unset, eta = U/Rm ('pcf-mhd-jax.py:63-64'), Lorentz prefactor 1 ('pcf-mhd-jax.py:175-176').
- **A0047 [VERIFIED]:** survey line 163; `pcf_mhd_jax.py:175-176` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; tokens=Lorentz;numbers=1; claim: - C: identical to A — Rm leftarrow Re if unset, eta = U/Rm ('pcf-mhd-jax.py:63-64'), Lorentz prefactor 1 ('pcf-mhd-jax.py:175-176').
- **A0048 [VERIFIED]:** survey line 164; `torchchannel/mhd.py:89-94` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=1,1,2; claim: - B (channel): Pm default 1; if Rm given then Pm = Rm/Re, else Rm = Re,Pm ('torchchannel/mhd.py:89-94'). Magnetic diffusion propto 1/Rm. Lorentz prefactor = Ha2/(Re,Rm) ('torchchannel/mhd.py:100-…
- **A0049 [VERIFIED]:** survey line 164; `torchchannel/mhd.py:100-101` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Lorentz,prefactor;numbers=1,1; claim: - B (channel): Pm default 1; if Rm given then Pm = Rm/Re, else Rm = Re,Pm ('torchchannel/mhd.py:89-94'). Magnetic diffusion propto 1/Rm. Lorentz prefactor = Ha2/(Re,Rm) ('torchchannel/mhd.py:100-…
- **A0050 [VERIFIED]:** survey line 165; `torchcouette/mhd.py:155` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; numbers=1,1,2; claim: - B (couette): magnetic diffusion LHS coefficient propto 1/Pm (viscosity equiv 1, 'torchcouette/mhd.py:155'); Lorentz prefactor = Ha2/Pm ('torchcouette/mhd.py:79'). No separate Rm.
- **A0051 [VERIFIED]:** survey line 165; `torchcouette/mhd.py:79` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Lorentz,prefactor; claim: - B (couette): magnetic diffusion LHS coefficient propto 1/Pm (viscosity equiv 1, 'torchcouette/mhd.py:155'); Lorentz prefactor = Ha2/Pm ('torchcouette/mhd.py:79'). No separate Rm.
- **A0052 [VERIFIED]:** survey line 175; `pcf_mhd_mri_shearpy.py:107-108` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega,omega,shear,shear_rate;numbers=2,1,2,1; claim: Keplerian rotation has q = 3/2 (defaults S=1, Omega = 2/3). The Alfvén speed of the imposed vertical field is v-A = B-z (with rho = mu-0 = 1). These are verified at 'pcf-mhd-mri-shearpy.py:107-108' (A) and…
- **A0053 [VERIFIED]:** survey line 175; `pcf_mhd_mri_shearpy_jax.py:73-74` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=Omega,omega,shear,shear_rate;numbers=3,2,2,3; claim: Keplerian rotation has q = 3/2 (defaults S=1, Omega = 2/3). The Alfvén speed of the imposed vertical field is v-A = B-z (with rho = mu-0 = 1). These are verified at 'pcf-mhd-mri-shearpy.py:107-108' (A) and…
- **A0054 [VERIFIED]:** survey line 185; `pcf_mhd_mri_shearpy.py:11-14` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega;numbers=2; claim: i.e. Coriolis -2boldsymbol(Omega)timesboldsymbol(u) (with boldsymbol(Omega)=Omegahat(z)), Coriolis + base-flow shear in the streamwise equation, and shear-induction (stretching B-x into B-y) via boldsy…
- **A0055 [VERIFIED]:** survey line 185; `pcf_mhd_mri_shearpy.py:16-21` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=component,induction,magnetic,potential,separate;numbers=2; claim: i.e. Coriolis -2boldsymbol(Omega)timesboldsymbol(u) (with boldsymbol(Omega)=Omegahat(z)), Coriolis + base-flow shear in the streamwise equation, and shear-induction (stretching B-x into B-y) via boldsy…
- **A0056 [VERIFIED]:** survey line 193; `OrrSommerfeld_eigs.py:183-184` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; numbers=8000,1,0.24707506017508621,0.0026644103710965817,10; claim: / Orr–Sommerfeld leading eigenvalue, Re=8000, alpha=1 (A) / — / c = 0.24707506017508621 + 0.0026644103710965817,i, tol 10(-12) ('OrrSommerfeld-eigs.py:183-184') /
- **A0057 [VERIFIED]:** survey line 194; `torchchannel/tests/test_linstab_poiseuille.py:7-19` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=Sommerfeld,alpha,eigenvalue,leading;numbers=10000,1,0.23752649,0.00373967,10; claim: / Orr–Sommerfeld leading eigenvalue, Re=10000, alpha=1 (B) / c-(text(ref)) = 0.23752649 + 0.00373967,i / tol 10(-4) ('torchchannel/tests/test-linstab-poiseuille.py:7-19') /
- **A0058 [VERIFIED]:** survey line 196; `couette_linear_benchmarks.md:313` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Omega;numbers=0.75,2,2,16,0.7499999944199642; claim: / Ideal local Keplerian MRI / s-(max)/Omega = 0.75, (k v-A)2/Omega2 = 15/16 / A/C: 0.7499999944199642, 0.9373170323757943 ('couette-linear-benchmarks.md:313') /
- **A0059 [VERIFIED]:** survey line 197; `couette_linear_benchmarks.md:352` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=conducting,growth;numbers=24.7,4.11,0,+0.003322863594034156,1.75; claim: / TC MRI conducting walls (eta=0.5 quasi-Kep, Rm=24.7, S=4.11) / growth > 0 / +0.003322863594034156 at best k-z=1.75 ('couette-linear-benchmarks.md:352') /
- **A0060 [VERIFIED]:** survey line 198; `couette_linear_benchmarks.md:353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=growth,insulating;numbers=16.5,5.21,0,-0.00027582037141390655,1.25; claim: / TC MRI insulating walls (Rm=16.5, S=5.21) / growth < 0 (sign flip) / -0.00027582037141390655 at best k-z=1.25 ('couette-linear-benchmarks.md:353') /
- **A0061 [VERIFIED]:** survey line 254; `pcf_mhd_divfree.py:47-57` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; tokens=installed,numba,shenfun;numbers=4,2,3; claim: This part specifies family A in enough detail to reimplement from scratch. Family A is a CPU/MPI, NumPy/SciPy + shenfun spectral-Galerkin stack ('float64'/'complex128', no GPU/JIT/autograd; an optional numba toggle…
- **A0062 [VERIFIED]:** survey line 264; `ChannelFlow.py:149-163` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=LaTeX,nabla;numbers=7,2; claim: The hydro DNS uses the Kim–Moser–Moin (KMM) velocity–vorticity reduction [KMM87]: pressure is eliminated exactly, and only two scalar fields are advanced — the wall-normal velocity u-x (whose time variable is nabla…
- **A0063 [VERIFIED]:** survey line 270; `ChannelFlow.py:152` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=self;numbers=1,0,1,1,1; claim: N=H is the convection vector (§I.A.1.6). The exact shenfun source operators are 'Dx(Dx(self.H-[1],0,1),1,1)+Dx(Dx(self.H-[2],0,1),2,1)-Dx(self.H-[0],1,2)-Dx(self.H-[0],2,2)' for u-x ('ChannelFlow.py:152') and 'Dx(se…
- **A0064 [VERIFIED]:** survey line 270; `ChannelFlow.py:160` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=self;numbers=1,0,1,1,1; claim: N=H is the convection vector (§I.A.1.6). The exact shenfun source operators are 'Dx(Dx(self.H-[1],0,1),1,1)+Dx(Dx(self.H-[2],0,1),2,1)-Dx(self.H-[0],1,2)-Dx(self.H-[0],2,2)' for u-x ('ChannelFlow.py:152') and 'Dx(se…
- **A0065 [VERIFIED]:** survey line 272; `pcf_fluctuations_corrected.py:106-107` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; numbers=1,-1,1,1,1; claim: Nondimensionalization (PCF fluctuation solver). Half-gap h=1, domain xin[-1,1], mathrm(Re)=U-(text(wall))h/nu so nu=U-(text(wall))/mathrm(Re) ('pcf-fluctuations-corrected.py:106-107'); default U-(te…
- **A0066 [VERIFIED]:** survey line 272; `OrrSommerfeld.py:14` -> `fn_shenfun/demo/OrrSommerfeld.py`; numbers=1,1,1,1,1; claim: Nondimensionalization (PCF fluctuation solver). Half-gap h=1, domain xin[-1,1], mathrm(Re)=U-(text(wall))h/nu so nu=U-(text(wall))/mathrm(Re) ('pcf-fluctuations-corrected.py:106-107'); default U-(te…
- **A0067 [VERIFIED]:** survey line 272; `OrrSommerfeld2D.py:14` -> `fn_shenfun/demo/OrrSommerfeld2D.py`; numbers=1,1,1,1,1; claim: Nondimensionalization (PCF fluctuation solver). Half-gap h=1, domain xin[-1,1], mathrm(Re)=U-(text(wall))h/nu so nu=U-(text(wall))/mathrm(Re) ('pcf-fluctuations-corrected.py:106-107'); default U-(te…
- **A0068 [VERIFIED]:** survey line 274; `ChannelFlow.py:9-11` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=normal,spanwise,streamwise;numbers=0,1,0; claim: Coordinate/sign convention (canonical, §0.2 row "A"). Axis 0 =x wall-normal (Chebyshev/Legendre, walls at x=pm1); axis 1 =y streamwise (Fourier complex, 'dtype='D''); axis 2 =z spanwise (Fourier real, 'dtyp…
- **A0069 [VERIFIED]:** survey line 278; `pcf_fluctuations_corrected.py:130-135` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=U_wall,dUb_dx,self,shear;numbers=0,5; claim: 'pcf-fluctuations-corrected.py:130-135' ('self.Ub = self.U-wallself.X[0]', 'self.dUb-dx = self.U-wall'). The MRI/shearing-box subclass overrides this to U-b=-S,x,mathbf(e)-y (§I.A.5). For cross-family comparison w…
- **A0070 [VERIFIED]:** survey line 282; `ChannelFlow.py:82-88` -> `fn_shenfun/demo/ChannelFlow.py`; numbers=1; claim: The 1-D bases ('ChannelFlow.py:82-88'):
- **A0071 [VERIFIED]:** survey line 298; `ChannelFlow.py:91-99` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=Tensor,product,spaces,vector; claim: Tensor-product and vector spaces ('ChannelFlow.py:91-99'):
- **A0072 [VERIFIED]:** survey line 312; `OrrSommerfeld_eigs.py:29` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; tokens=quad; claim: Quadrature. Gauss by default (Gauss–Chebyshev 'GC' / Gauss–Legendre 'LG'); the OrrSommerfeld eigensolver uses 'quad='GC'' ('OrrSommerfeld-eigs.py:29').
- **A0073 [VERIFIED]:** survey line 314; `ChannelFlow.py:144-145` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=Biharmonic,Chebyshev,Helmholtz,SolverGeneric1ND,biharmonic;numbers=0,0; claim: Per-mode radial solvers. For each Fourier mode (k-y,k-z)neq(0,0), the u-x equation is biharmonic and g is Helmholtz. Chebyshev uses the tailored fast solvers 'chebyshev.la.Biharmonic' (for u-x) and 'chebysh…
- **A0074 [VERIFIED]:** survey line 316; `ChannelFlow2D.py:81` -> `fn_shenfun/demo/ChannelFlow2D.py`; tokens=left,right,space;numbers=0,0,0,0; claim: 2D channel ('ChannelFlow2D.py') keeps only x (wall-normal) and y (streamwise Fourier real); the wall-normal velocity basis is built via the explicit clamped dict 'bc=('left':('D':0,'N':0),'right':('D':0,'N':0))'…
- **A0075 [VERIFIED]:** survey line 316; `ChannelFlow2D.py:138-144` -> `fn_shenfun/demo/ChannelFlow2D.py`; tokens=nabla,partial;numbers=2,0,0,0,0; claim: 2D channel ('ChannelFlow2D.py') keeps only x (wall-normal) and y (streamwise Fourier real); the wall-normal velocity basis is built via the explicit clamped dict 'bc=('left':('D':0,'N':0),'right':('D':0,'N':0))'…
- **A0076 [PARTIAL]:** survey line 316; `ChannelFlow2D.py:195` -> `fn_shenfun/demo/ChannelFlow2D.py`; resolved-line-exists-but-claim-numbers-not-found; claim: 2D channel ('ChannelFlow2D.py') keeps only x (wall-normal) and y (streamwise Fourier real); the wall-normal velocity basis is built via the explicit clamped dict 'bc=('left':('D':0,'N':0),'right':('D':0,'N':0))'…
- **A0077 [VERIFIED]:** survey line 320; `ChannelFlow.py:236-239` -> `fn_shenfun/demo/ChannelFlow.py`; numbers=0,0; claim: The KMM form removes pressure from the time loop entirely (no projection, no influence matrix). For every Fourier mode (k-y,k-z)neq(0,0), after solving u-x and g, the tangential components are recovered algebra…
- **A0078 [VERIFIED]:** survey line 324; `ChannelFlow.py:168-171` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=K_over_K2,where;numbers=1,0,1,2,2; claim: with 'K-over-K2[i] = K[i+1]/where(K2==0,1,K2)', K2=k-y2+k-z2 ('ChannelFlow.py:168-171'). Sign pitfall: the code uses f=+partial-x u-x while a comment notes "paper uses f=-mathrm(d)u/mathrm(d)x" ('ChannelF…
- **A0079 [VERIFIED]:** survey line 324; `ChannelFlow.py:236` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=paper;numbers=2,2; claim: with 'K-over-K2[i] = K[i+1]/where(K2==0,1,K2)', K2=k-y2+k-z2 ('ChannelFlow.py:168-171'). Sign pitfall: the code uses f=+partial-x u-x while a comment notes "paper uses f=-mathrm(d)u/mathrm(d)x" ('ChannelF…
- **A0080 [VERIFIED]:** survey line 326; `ChannelFlow.py:181-196` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=LaTeX;numbers=0,0,00,00,1; claim: The single (0,0) Fourier mode (horizontal mean) has no divergence constraint, so v-(00)(x), w-(00)(x) are advanced as two separate 1-D Helmholtz PDEs on 'D00', on MPI rank 0 only, with a mean pressure-gradient s…
- **A0081 [VERIFIED]:** survey line 330; `ChannelFlow.py:129` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=Project,divu;numbers=2; claim: with source -mathrm(d)p/mathrm(d)y (=0 for PCF, =-(-2/mathrm(Re))=2/mathrm(Re) source for OS Poiseuille). Measured mathrm(div)(u)approx10(-12) (roundoff), checked via 'divu = Project(div(u-), TC)' ('Chann…
- **A0082 [VERIFIED]:** survey line 332; `ChannelFlow.py:253-268` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=compute_pressure,constraint,pressure,space;numbers=2,2,2,0,0; claim: An optional post-hoc pressure recovery ('compute-pressure', 'ChannelFlow.py:253-268') solves a Poisson problem nabla2 p=-mathrm(div)(H) with Neumann BC partial p/partial n=nu,partial2 u-x/partial n2 b…
- **A0083 [VERIFIED]:** survey line 340; `ChannelFlow.py:315-330` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=solve; claim: The driver ('KMM.solve', 'ChannelFlow.py:315-330') is:
- **A0084 [VERIFIED]:** survey line 353; `ChannelFlow.py:67` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=IMEXRK3,timestepper;numbers=6.; claim: The base 'KMM' constructor default is 'IMEXRK3' ('ChannelFlow.py:67'), but the PCF subclasses and runners override to 'IMEXRK222' ('pcf-fluctuations-corrected.py:689' passes 'timestepper='IMEXRK222''; 'pcf-fluctuati…
- **A0085 [VERIFIED]:** survey line 353; `pcf_fluctuations_corrected.py:689` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=IMEXRK222,timestepper;numbers=22,22,22,6.,2; claim: The base 'KMM' constructor default is 'IMEXRK3' ('ChannelFlow.py:67'), but the PCF subclasses and runners override to 'IMEXRK222' ('pcf-fluctuations-corrected.py:689' passes 'timestepper='IMEXRK222''; 'pcf-fluctuati…
- **A0086 [VERIFIED]:** survey line 353; `pcf_fluctuations_divV.py:58,210` -> `fn_shenfun/demo/pcf_fluctuations_divV.py`; tokens=IMEXRK222,timestepper;numbers=22,22,22,2; claim: The base 'KMM' constructor default is 'IMEXRK3' ('ChannelFlow.py:67'), but the PCF subclasses and runners override to 'IMEXRK222' ('pcf-fluctuations-corrected.py:689' passes 'timestepper='IMEXRK222''; 'pcf-fluctuati…
- **A0087 [VERIFIED]:** survey line 353; `ChannelFlow2D.py:66` -> `fn_shenfun/demo/ChannelFlow2D.py`; tokens=IMEXRK222,timestepper;numbers=22,22,22,6.,2; claim: The base 'KMM' constructor default is 'IMEXRK3' ('ChannelFlow.py:67'), but the PCF subclasses and runners override to 'IMEXRK222' ('pcf-fluctuations-corrected.py:689' passes 'timestepper='IMEXRK222''; 'pcf-fluctuati…
- **A0088 [VERIFIED]:** survey line 357; `pcf_fluctuations_corrected.py:175-176` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=NotImplementedError,conv,implemented;numbers=1; claim: Only convective (advective) form ucdotnabla u is implemented; rotational form ('conv=1') raises 'NotImplementedError' ('pcf-fluctuations-corrected.py:175-176'). The procedure ('pcf-fluctuations-corrected.py:190-202'…
- **A0089 [VERIFIED]:** survey line 357; `pcf_fluctuations_corrected.py:190-202` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=advection,production,shear,transform;numbers=1; claim: Only convective (advective) form ucdotnabla u is implemented; rotational form ('conv=1') raises 'NotImplementedError' ('pcf-fluctuations-corrected.py:175-176'). The procedure ('pcf-fluctuations-corrected.py:190-202'…
- **A0090 [PARTIAL]:** survey line 365; `pcf_fluctuations_corrected.py:195-197` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; resolved-line-exists-but-semantic-support-not-proven; claim: ('pcf-fluctuations-corrected.py:195-197', verified verbatim) with U-b=U-(text(wall)),x, mathrm(d)U-b/mathrm(d)x=U-(text(wall)); then forward-transform 'H[i]=TDp.forward(n-i,H[i])' and zero the Nyquist Fourier m…
- **A0091 [VERIFIED]:** survey line 367; `ChannelFlow.py:62` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=factor,padding_factor;numbers=2,1,1.5,1.5,2; claim: Dealiasing (Orszag 3/2, periodic directions only). 'padding-factor=(1,1.5,1.5)' ('ChannelFlow.py:62'): the two Fourier directions (y,z) are padded 3/2 so quadratic products are aliasing-free; the wall-normal x i…
- **A0092 [VERIFIED]:** survey line 371; `OrrSommerfeld_eigs.py:84-99` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; tokens=inner;numbers=0,0,0,0,4; claim: 'OrrSommerfeld-eigs.py' solves Avarphi=cBvarphi in Shen's biharmonic Chebyshev basis ('bc=(0,0,0,0)', dim N-4); default 'alfa=1.0, Re=8000, N=80'. Operators ('OrrSommerfeld-eigs.py:84-99'): with weighted inner pro…
- **A0093 [VERIFIED]:** survey line 375; `OrrSommerfeld_eigs.py:183-184` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; numbers=8000,1,80; claim: Golden eigenvalue (self-asserted, 'OrrSommerfeld-eigs.py:183-184'): at mathrm(Re)=8000, alpha=1, N>80,
- **A0094 [VERIFIED]:** survey line 379; `OrrSommerfeld.py:47-56` -> `fn_shenfun/demo/OrrSommerfeld.py`; numbers=1,2,2,3; claim: The DNS OS path ('OrrSommerfeld.py'/'OrrSommerfeld2D.py') seeds the eigenmode at amplitude 10(-7) on the Poiseuille base 1-x2 and checks energy proptoexp(2,mathrm(Im)(c),t) ('OrrSommerfeld.py:47-56'). PCF i…
- **A0095 [VERIFIED]:** survey line 379; `couette_linear_benchmarks.md:69-70,111-118` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Romanov,growth,leading;numbers=10,1,2,2,1000; claim: The DNS OS path ('OrrSommerfeld.py'/'OrrSommerfeld2D.py') seeds the eigenmode at amplitude 10(-7) on the Poiseuille base 1-x2 and checks energy proptoexp(2,mathrm(Im)(c),t) ('OrrSommerfeld.py:47-56'). PCF i…
- **A0096 [VERIFIED]:** survey line 385; `:87-88` -> `fn_shenfun/demo/pcf_galerkin_linear.py`; tokens=slice,space;numbers=0,0,0,2,87; claim: - Galerkin ('pcf-galerkin-linear.py'): velocity in Dirichlet 'bc=(0,0)'; pressure in the full orthogonal space sliced to 'slice(0,N-2)' (':87-88', not a 'bc'-constrained basis); MHD b-y,b-z in Neumann. Blocks…
- **A0097 [VERIFIED]:** survey line 385; `:103-117` -> `fn_shenfun/demo/pcf_galerkin_linear.py`; numbers=0,0,0,2,0; claim: - Galerkin ('pcf-galerkin-linear.py'): velocity in Dirichlet 'bc=(0,0)'; pressure in the full orthogonal space sliced to 'slice(0,N-2)' (':87-88', not a 'bc'-constrained basis); MHD b-y,b-z in Neumann. Blocks…
- **A0098 [VERIFIED]:** survey line 388; `_pcf_linear.py:220-228` -> `fn_shenfun/demo/_pcf_linear.py`; numbers=0,0,4,6,0; claim: Both treat pressure as a Lagrange multiplier enforcing mathrm(div)(u)=0 (saddle-point), with -nabla p columns and mathrm(div),u rows ('-pcf-linear.py:220-228'); the pressure block has zero mass so M…
- **A0099 [VERIFIED]:** survey line 388; `_linear_analysis.py:16,33-42` -> `fn_shenfun/demo/_linear_analysis.py`; tokens=FINITE_CAP,finite_eigensystem;numbers=0,0,4,6,0; claim: Both treat pressure as a Lagrange multiplier enforcing mathrm(div)(u)=0 (saddle-point), with -nabla p columns and mathrm(div),u rows ('-pcf-linear.py:220-228'); the pressure block has zero mass so M…
- **A0100 [VERIFIED]:** survey line 396; `taylor_couette_dns.py:141,540` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=coors,radial,self,symbol,sympy;numbers=1,1,2,1,1; claim: TC (linear and DNS) does not use shenfun's 'coordinates=' curvilinear machinery. It builds plain Cartesian-measure spaces on 'domain=(R1,R2)' and writes every 1/r, 1/r2 as an explicit sympy coefficient of t…
- **A0101 [VERIFIED]:** survey line 396; `taylor_couette_dns.py:187-190` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=linear,radial,self,space,symbol;numbers=1,1,2,1,1; claim: TC (linear and DNS) does not use shenfun's 'coordinates=' curvilinear machinery. It builds plain Cartesian-measure spaces on 'domain=(R1,R2)' and writes every 1/r, 1/r2 as an explicit sympy coefficient of t…
- **A0102 [VERIFIED]:** survey line 396; `taylor_couette_dns.py:178-180` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=self;numbers=1,1,2,1,1; claim: TC (linear and DNS) does not use shenfun's 'coordinates=' curvilinear machinery. It builds plain Cartesian-measure spaces on 'domain=(R1,R2)' and writes every 1/r, 1/r2 as an explicit sympy coefficient of t…
- **A0103 [VERIFIED]:** survey line 400; `taylor_couette_linear.py:81-132` -> `fn_shenfun/demo/taylor_couette_linear.py`; tokens=CircularCouette,Couette,circular; claim: The circular-Couette base ('CircularCouette', 'taylor-couette-linear.py:81-132'):
- **A0104 [VERIFIED]:** survey line 404; `taylor_couette_dns.py:123` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=Omega;numbers=2,2,2,2,2; claim: with U-(text(base))=+V(r),mathbf(e)-theta (positive swirl). Identities: 2Omega+rOmega'=2a (constant), rOmega'=-2b/r2, epicyclic kappa2(r)=4a,Omega(r); radius ratio eta=R-1/R-2, rotation ratio mu…
- **A0105 [VERIFIED]:** survey line 406; `taylor_couette_dns.py:21-35` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=Axisymmetric,centrifugal,perturbation;numbers=0,2; claim: The DNS integrates the perturbation about the analytic base (so u=0 is the exact fixed point; the base centrifugal balance mathrm(d)P-(text(base))/mathrm(d)r=V2/r is subtracted). Axisymmetric ('taylor-couett…
- **A0106 [VERIFIED]:** survey line 415; `taylor_couette_dns.py:573-617` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=continuity,couplings,theta,viscous;numbers=3,2,2,2,2; claim: The 3D solver adds, per azimuthal mode m: base-shear advection -mathrm(i)mOmega on every component; the viscous cross-couplings mp(2/r2)partial-theta u-(theta/r) (=mp 2mathrm(i)m/r2); the full (1/…
- **A0107 [VERIFIED]:** survey line 421; `taylor_couette_dns.py:396-411` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=amplitude,combining,coupled,divergence,evaluates;numbers=0,-13,10,0,3; claim: Div-free is exact via a coupled saddle-point solve (no fractional-step splitting): velocity and pressure are solved together per Fourier mode through 'la.BlockMatrixSolver', so mathrm(div)(u)sim10(-13)–10(-14…
- **A0108 [VERIFIED]:** survey line 425; `pipe_flow_dns.py:35-37` -> `fn_shenfun/demo/pipe_flow_dns.py`; numbers=2,2; claim: The DNS default is CNAB2 (2nd-order IMEX): Crank–Nicolson for the linear operator A= viscous + all base-flow couplings + pressure gradient; Adams–Bashforth-2 for the nonlinear term; IMEX-Euler bootstrap on the fir…
- **A0109 [VERIFIED]:** survey line 429; `taylor_couette_dns.py:288-313` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=Lexp,Limp,coupled,pressure,space;numbers=1,2,0,1,2; claim: Pre-assembled: 'Limp = BlockMatrixSolver(M/dt - 1/2 A + grad p ; div=0)' over the coupled velocity–pressure space and 'Lexp = BlockMatrix(M/dt + 1/2 A)'. The exact per-step code ('taylor-couette-dns.py:288-313', verifie…
- **A0110 [VERIFIED]:** survey line 450; `taylor_couette_dns.py:832,839-841` -> `fn_shenfun/demo/taylor_couette_dns.py`; tokens=Robin,left,right,theta;numbers=0,0,0,0,0; claim: - Conducting (any m): b-r=0 (Dirichlet), mathrm(d)(r b-theta)/mathrm(d)r=0 i.e. b-theta+r b-theta'=0 (Robin), b-z'=0 (Neumann). The Robin coefficient is in reference coordinates: shenfun 'bc=("lef…
- **A0111 [VERIFIED]:** survey line 461; `pipe_flow_dns.py:115-140` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=Fourier,Ntheta,True,already,coordinates;numbers=1,2,2,2,0; claim: The pipe DNS does use curvilinear shenfun, so the r,mathrm(d)r,mathrm(d)theta,mathrm(d)z measure is applied automatically: 'r,theta,z = sp.symbols('x,y,z', real=True, positive=True)', 'rv=(rcos(theta), rs…
- **A0112 [VERIFIED]:** survey line 461; `pipe_flow_dns.py:193-201` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=inner,theta;numbers=1,2,2,2,0; claim: The pipe DNS does use curvilinear shenfun, so the r,mathrm(d)r,mathrm(d)theta,mathrm(d)z measure is applied automatically: 'r,theta,z = sp.symbols('x,y,z', real=True, positive=True)', 'rv=(rcos(theta), rs…
- **A0113 [VERIFIED]:** survey line 465; `pipe_flow_dns.py:41-46` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=theta; claim: Primitive (u-r,u-theta,u-z,p) driven by a uniform axial body force f-z (constant Rightarrow Hagen–Poiseuille; callable f-z(t)Rightarrow Womersley); z periodic, zero mean pressure gradient. The vector Laplac…
- **A0114 [VERIFIED]:** survey line 471; `pipe_flow_notes.md:39-43` -> `fn_shenfun/demo/pipe_flow_notes.md`; numbers=0,0,2,2,0; claim: The pole is handled with one velocity basis 'bc=(None,0)' for all azimuthal modes (free/regularity at r=0, no-slip Dirichlet at r=R) — no m-by-m basis split. The curvilinear weighted (sqrt(g)=r) G…
- **A0115 [VERIFIED]:** survey line 482; `pipe_flow_dns.py:299-309` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=Womersley,accuracy,dependent,force,midpoint;numbers=1,2,2; claim: Same CNAB2 / coupled saddle-point per (m,k-z) as §I.A.2.4. For a time-dependent body force (Womersley) the force is evaluated at the midpoint t(n+1/2) for 2nd-order accuracy ('pipe-flow-dns.py:299-309'). Exact…
- **A0116 [VERIFIED]:** survey line 482; `pipe_flow_dns.py:473-500` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=Exact,Womersley;numbers=1,2,2; claim: Same CNAB2 / coupled saddle-point per (m,k-z) as §I.A.2.4. For a time-dependent body force (Womersley) the force is evaluated at the midpoint t(n+1/2) for 2nd-order accuracy ('pipe-flow-dns.py:299-309'). Exact…
- **A0117 [VERIFIED]:** survey line 488; `pipe_flow_notes.md:67-71` -> `fn_shenfun/demo/pipe_flow_notes.md`; tokens=Bessel,Hagen,Poiseuille,Womersley,decay;numbers=10,-6,10,-10,1.4; claim: Golden tolerances ('test-pipe-flow-dns.py', 'pipe-flow-notes.md:67-71'): Hagen–Poiseuille max/u-z-text(exact)/<10(-6), /Q-Q-(text(exact))//Q-(text(exact))<10(-10) (measured 1.4times10(-12)), mathrm(div)-…
- **A0118 [VERIFIED]:** survey line 498; `pcf_mhd_divfree.py:6-14` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; tokens=discrete,identity,potential;numbers=0; claim: The induction is advanced for the vector potential A in the Weyl gauge, which makes mathrm(div)(B)=0 a discrete identity by construction ('pcf-mhd-divfree.py:6-14'):
- **A0119 [VERIFIED]:** survey line 504; `pcf_mhd_divfree.py:325-333` -> `fn_shenfun/demo/pcf_mhd_divfree.py`; tokens=Lorentz,enters,force,nonlinear; claim: The Lorentz force enters the KMM nonlinear store as N-Jtimes B ('pcf-mhd-divfree.py:325-333'):
- **A0120 [VERIFIED]:** survey line 512; `pcf_mhd_divfree_notes.md:69-96` -> `fn_shenfun/demo/pcf_mhd_divfree_notes.md`; tokens=Chebyshev;numbers=8,8,8,400,0.003; claim: Golden div-control numbers ('pcf-mhd-divfree-notes.md:69-96'): Legendre N=(8,8,8), mathrm(Re)=mathrm(Rm)=400, t=0.003 → mathrm(div),U-(L-2)=9.41times10(-17), mathrm(div),B-(L-2)=3.05times10(-21); Che…
- **A0121 [VERIFIED]:** survey line 516; `_pcf_linear.py:240-245` -> `fn_shenfun/demo/_pcf_linear.py`; numbers=0,0,0,0,0; claim: Collocation primitive variables add (b-x,b-y,b-z,phi), where phi is a magnetic-pressure Lagrange multiplier enforcing mathrm(div)(b)=0 ('-pcf-linear.py:240-245'). Induction couplings with imposed uniform field…
- **A0122 [VERIFIED]:** survey line 516; `:230-245` -> `fn_shenfun/demo/_pcf_linear.py`; numbers=0,0,230,0,0; claim: Collocation primitive variables add (b-x,b-y,b-z,phi), where phi is a magnetic-pressure Lagrange multiplier enforcing mathrm(div)(b)=0 ('-pcf-linear.py:240-245'). Induction couplings with imposed uniform field…
- **A0123 [VERIFIED]:** survey line 516; `couette_linear_benchmarks.md:185-194` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Energy,Magnetic,hydro,imposed,kinetic;numbers=0,0,0,0,0; claim: Collocation primitive variables add (b-x,b-y,b-z,phi), where phi is a magnetic-pressure Lagrange multiplier enforcing mathrm(div)(b)=0 ('-pcf-linear.py:240-245'). Induction couplings with imposed uniform field…
- **A0124 [VERIFIED]:** survey line 520; `taylor_couette_mri.py:19-34` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Field,Omega,equations,field,theta;numbers=0; claim: Field in Alfvén units (v-A=B-(0z)); total pressure Pi=p+B-0 b-z absorbs the imposed-field magnetic pressure so the Lorentz force is simply mathrm(i)k-z B-0 b per component. The linearized equations (U=rOmega(r…
- **A0125 [VERIFIED]:** survey line 538; `taylor_couette_mri.py:366-371` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Robin,kappa,shenfun;numbers=2,2,0,0; claim: The vacuum match is a single-field Robin chi'/chi=k-z2/kappa with kappa the modified-Bessel log-derivative of the exterior potential: kappa-(text(in))=k,I-1(kR-1)/I-0(kR-1), kappa-(text(out))=-k,K-1(kR…
- **A0126 [VERIFIED]:** survey line 540; `couette_linear_benchmarks.md:352-353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=conducting,growth,insulating,target;numbers=7,24.7,4.11,1.75,+0.003322863594034156; claim: Sign-distinguishing golden numbers (eta=0.5 quasi-Keplerian, mathrm(Pm)to0, [LL07]/Rüdiger 2023; 'couette-linear-benchmarks.md:352-353'): conducting target mathrm(Rm)=24.7, S=4.11, best k-z=1.75 → te…
- **A0127 [VERIFIED]:** survey line 546; `pcf_mhd_mri_shearpy.py:6-15` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=azimuthal,direction,gradient,normal,radial; claim: 'pcf-mhd-mri-shearpy.py' is the PCF MHD analogue of the shearpy shearing-box MRI: a wall-bounded box (the radial/shear direction is replaced by no-slip PCF walls, not a shearing-periodic remap). It subclasses 'pcf-m…
- **A0128 [VERIFIED]:** survey line 552; `pcf_mhd_mri_shearpy.py:102-104` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=dUb_dx,self,shear_rate;numbers=0; claim: The override is 'self.Ub=-self.shear-rateself.X[0]', 'self.dUb-dx=-self.shear-rate' ('pcf-mhd-mri-shearpy.py:102-104', verified verbatim).
- **A0129 [VERIFIED]:** survey line 554; `:346-348` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega,shear;numbers=-2,2,346,+2,-2; claim: Implementation details. Because KMM stores H=N-F (the velocity equations apply -H after projection), the Coriolis additions are entered as 'n0 += -2Omegau-y; n1 += 2Omegau-x' (':346-348'), which yield the de…
- **A0130 [VERIFIED]:** survey line 558; `pcf_mhd_mri_shearpy.py:107-108` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Omega;numbers=2,1,2; claim: Dimensionless groups ('pcf-mhd-mri-shearpy.py:107-108', verified verbatim): q=S/Omega (Keplerian q=3/2 at default S=1,Omega=2/3); epicyclic
- **A0131 [VERIFIED]:** survey line 564; `couette_linear_benchmarks.md:313-317` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Omega,shear,theory;numbers=2,3,25.81988897471611,.498406,-1; claim: Validation ('couette-linear-benchmarks.md:313-317'): PCF rotating-shear MRI analogue at Omega=2/3, b-z=0.025, k-z=25.81988897471611 → leading eigenvalue sapprox0.498406 (theory s=0.5); DNS netflux case grows…
- **A0132 [PARTIAL]:** survey line 574; `integrators.py:798-817` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: 'stages()' returns (a,b,c) with a the implicit (DIRK) tableau, b the explicit tableau, indexed [text(rk)+1,j]. The per-stage update ('integrators.py:798-817', verified verbatim):
- **A0133 [PARTIAL]:** survey line 578; `integrators.py:787,816` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: then a single implicit solve with the once-factored operator (the implicit LHS is mathrm(inner)(v,,u-l-Delta t,a-(11),L(u-l)), reused because the diagonal a-(11) is constant across stages, 'integrators.py:7…
- **A0134 [VERIFIED]:** survey line 578; `_linear_analysis.py:128-160` -> `fn_shenfun/demo/_linear_analysis.py`; tokens=linear,steppers;numbers=11,11; claim: then a single implicit solve with the once-factored operator (the implicit LHS is mathrm(inner)(v,,u-l-Delta t,a-(11),L(u-l)), reused because the diagonal a-(11) is constant across stages, 'integrators.py:7…
- **A0135 [PARTIAL]:** survey line 580; `integrators.py:836-850` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: IMEXRK111 — 1 stage, 1st order ('integrators.py:836-850'):
- **A0136 [PARTIAL]:** survey line 583; `integrators.py:852-870` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: IMEXRK222 — 2 stages, 2nd order, L-stable (DEFAULT for PCF subclasses, PCF MHD, and the TC IMEXRK stepper) ('integrators.py:852-870'). With gamma=(2-sqrt2)/2approx0.2928932188 and delta=1-1/(2gamma)appr…
- **A0137 [PARTIAL]:** survey line 587; `integrators.py:872-892` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: IMEXRK443 — 4 stages, 3rd order ('integrators.py:872-892'):
- **A0138 [PARTIAL]:** survey line 590; `integrators.py:819-833` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (Also 'IMEXRK011', 'integrators.py:819-833': a=[[0,0],[0,0]], b=[[0,0],[1,0]], c=(1,0), steps =1.)
- **A0139 [PARTIAL]:** survey line 594; `integrators.py:603-700` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: A separate class (not PDEIMEXRK), 'integrators.py:603-700', 3 stages, 3rd order. Coefficients ('integrators.py:665-669', verified verbatim):
- **A0140 [PARTIAL]:** survey line 594; `integrators.py:665-669` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: A separate class (not PDEIMEXRK), 'integrators.py:603-700', 3 stages, 3rd order. Coefficients ('integrators.py:665-669', verified verbatim):
- **A0141 [PARTIAL]:** survey line 596; `integrators.py:678-680` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The implicit LHS per stage is Crank–Nicolson-like, mathrm(inner)(v,,u-l-(a-(text(rk))+b-(text(rk)))tfrac(Delta t)(2)L(u-l)), and a separate solver is assembled per stage ('integrators.py:678-680'). The per-s…
- **A0142 [PARTIAL]:** survey line 596; `integrators.py:692-693` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The implicit LHS per stage is Crank–Nicolson-like, mathrm(inner)(v,,u-l-(a-(text(rk))+b-(text(rk)))tfrac(Delta t)(2)L(u-l)), and a separate solver is assembled per stage ('integrators.py:678-680'). The per-s…
- **A0143 [VERIFIED]:** survey line 613; `_linear_analysis.py:163-186` -> `fn_shenfun/demo/_linear_analysis.py`; tokens=IMEXRK,constraint,couplings,descriptor,diffusion; claim: The IMEXRK linear steppers support '--split diffusion' (diffusion + pressure/continuity implicit, base couplings explicit) vs '--split full' (everything implicit, stiff reference). The descriptor-system stage solve is …
- **A0144 [PARTIAL]:** survey line 657; `solver.py:225` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The only hydrodynamic nondimensional parameter is Re=UL/nu; viscosity enters the implicit Helmholtz solve as the single diffusivity 1/Re ('solver.py:225', '-build-diffusion-system(1.0/self.Re)'). Re must be pos…
- **A0145 [VERIFIED]:** survey line 659; `solver.py:258-260` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; tokens=implicit,operator;numbers=1,1; claim: Taylor–Couette. Cylindrical incompressible NS in viscous units: the radial Laplacian carries unit viscosity (operator coefficient 'c2 = -implicit' with no 1/Re, 'solver.py:258-260'), and the wall speeds carry…
- **A0146 [VERIFIED]:** survey line 669; `base_flow.py:38-41` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; numbers=1,0; claim: / Plane Couette / U(y)=y / U'=1, U''=0 / u(mp1)=mp1 / 'base-flow.py:38-41' /
- **A0147 [VERIFIED]:** survey line 670; `base_flow.py:43-46` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; numbers=1,2,-2,-2,0; claim: / Poiseuille / U(y)=1-y2 / U'=-2y, U''=-2 / stationary (0,0) / 'base-flow.py:43-46' /
- **A0148 [VERIFIED]:** survey line 671; `base_flow.py:18-25` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; tokens=Couette,theta; claim: / Taylor–Couette / u-theta(r)=ar+b/r / — / (Re-i,Re-o) / 'base-flow.py:18-25' /
- **A0149 [PARTIAL]:** survey line 673; `base_flow.py:23-24` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The circular-Couette coefficients are ('base-flow.py:23-24'):
- **A0150 [PARTIAL]:** survey line 684; `mesh.py:12-45` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The FD weights mirror OpenPipeFlow's 'mes-weights' ('mesh.py:12-45'). For target x-0 and stencil (x-j), build A-(:,0)=1, A-(:,j)=A-(:,j-1)cdot(x-x-0)/j so that A-(j)=(x-x-0)j/j!, then solve the transpose…
- **A0151 [PARTIAL]:** survey line 684; `mesh.py:43-45` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The FD weights mirror OpenPipeFlow's 'mes-weights' ('mesh.py:12-45'). For target x-0 and stencil (x-j), build A-(:,0)=1, A-(:,j)=A-(:,j-1)cdot(x-x-0)/j so that A-(j)=(x-x-0)j/j!, then solve the transpose…
- **A0152 [PARTIAL]:** survey line 684; `mesh.py:25` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The FD weights mirror OpenPipeFlow's 'mes-weights' ('mesh.py:12-45'). For target x-0 and stencil (x-j), build A-(:,0)=1, A-(:,j)=A-(:,j-1)cdot(x-x-0)/j so that A-(j)=(x-x-0)j/j!, then solve the transpose…
- **A0153 [PARTIAL]:** survey line 686; `mesh.py:109-117` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Stencil width and order. 'KL' is the half-bandwidth; the interior stencil is min(2,KL+1,,N) centered points ('mesh.py:109-117'), default 'KL=4' Rightarrow 9-point centered stencil. With 9 points the inte…
- **A0154 [PARTIAL]:** survey line 686; `mesh.py:103-106` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Stencil width and order. 'KL' is the half-bandwidth; the interior stencil is min(2,KL+1,,N) centered points ('mesh.py:109-117'), default 'KL=4' Rightarrow 9-point centered stencil. With 9 points the inte…
- **A0155 [PARTIAL]:** survey line 686; `tests/test_mesh.py:21-37` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Stencil width and order. 'KL' is the half-bandwidth; the interior stencil is min(2,KL+1,,N) centered points ('mesh.py:109-117'), default 'KL=4' Rightarrow 9-point centered stencil. With 9 points the inte…
- **A0156 [VERIFIED]:** survey line 692; `mesh.py:155` -> `fn_openpipeflow-122/torchcouette/torchcouette/mesh.py`; tokens=radlap;numbers=2,2,1,1; claim: This is deliberate so that operatorname(div)(operatorname(grad)p)equivnabla2 p holds exactly for the pressure projection. (Couette stores an independent W-(dr2) but the projection uses W-(dr1)2; see §I.B.3…
- **A0157 [VERIFIED]:** survey line 694; `mesh.py:80-100` -> `fn_openpipeflow-122/torchchannel/torchchannel/mesh.py`; tokens=Channel,clust,points,theta;numbers=1,0; claim: Mesh points. Channel: Chebyshev extrema y=-cos(pi j/(N-1)) when 'clust=0', else nsCouette stretching y=arcsin(-ccostheta)/arcsin(c); endpoints pinned exactly to pm1 ('mesh.py:80-100'). Couette: ascendin…
- **A0158 [VERIFIED]:** survey line 697; `operators.py:35-43` -> `fn_openpipeflow-122/torchchannel/torchchannel/operators.py`; numbers=1,2,+1,2,+1; claim: - Channel: full symmetric double Fourier, K1=K-1, Kc=2K1+1, Mc=2M1+1, with kin[-K1,K1], min[-M1,M1] ('operators.py:35-43'). Wavenumbers k-alpha=kalpha, m-beta=mbeta, k2=k-alpha2+m-beta2 ('oper…
- **A0159 [VERIFIED]:** survey line 697; `operators.py:44-46` -> `fn_openpipeflow-122/torchchannel/torchchannel/operators.py`; tokens=alpha;numbers=2,2,2,2,2; claim: - Channel: full symmetric double Fourier, K1=K-1, Kc=2K1+1, Mc=2M1+1, with kin[-K1,K1], min[-M1,M1] ('operators.py:35-43'). Wavenumbers k-alpha=kalpha, m-beta=mbeta, k2=k-alpha2+m-beta2 ('oper…
- **A0160 [VERIFIED]:** survey line 698; `spectral.py:32-44` -> `fn_openpipeflow-122/torchcouette/torchcouette/spectral.py`; tokens=negative;numbers=0.,1.,1,2,0.; claim: - Couette: nsCouette storage — axial slots [0..K,-K(+)1..(-)1] with Kc=2K; azimuthal stored non-negative [0..M-1] with Hermitian completion ('spectral.py:32-44').
- **A0161 [VERIFIED]:** survey line 700; `mesh.py:159-164` -> `fn_openpipeflow-122/torchcouette/torchcouette/mesh.py`; tokens=order,weights;numbers=-1,1,2,0,2; claim: Quadrature. Channel uses plain int-(-1)(1) weights 'inty' with low-order moments corrected exactly (int1=2, int y=0, int y2=2/3, 'mesh.py:159-164'). Couette uses the cylindrical Jacobian int f(r),r,d…
- **A0162 [VERIFIED]:** survey line 700; `mesh.py:58-81` -> `fn_openpipeflow-122/torchcouette/torchcouette/mesh.py`; tokens=Couette,Quadrature,intrdr,order,weights;numbers=-1,1,2,0,2; claim: Quadrature. Channel uses plain int-(-1)(1) weights 'inty' with low-order moments corrected exactly (int1=2, int y=0, int y2=2/3, 'mesh.py:159-164'). Couette uses the cylindrical Jacobian int f(r),r,d…
- **A0163 [PARTIAL]:** survey line 704; `spectral.py:23,37-38` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Both use 3× padded dealiasing (more conservative than the 2/3 rule). Padded physical grid Z=texttt(dealias-mult)cdot K, Th=texttt(dealias-mult)cdot M, default 'dealias-mult=3' ('spectral.py:23,37-38'). Ret…
- **A0164 [VERIFIED]:** survey line 704; `spectral.py:91-114` -> `fn_openpipeflow-122/torchchannel/torchchannel/spectral.py`; tokens=coll_to_phys,fft2,ifft2,phys_to_coll,physical;numbers=3,2,3,3; claim: Both use 3× padded dealiasing (more conservative than the 2/3 rule). Padded physical grid Z=texttt(dealias-mult)cdot K, Th=texttt(dealias-mult)cdot M, default 'dealias-mult=3' ('spectral.py:23,37-38'). Ret…
- **A0165 [VERIFIED]:** survey line 707; `solver.py:115` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=physical,physical_scale; claim: - Channel uses plain 'torch.fft.fft2'/'ifft2' (no 'norm='), so a physical constant c(y) lives in the mean coefficient as c(y)cdot(Zcdot Th); 'physical-scale = ZTh' ('solver.py:115').
- **A0166 [VERIFIED]:** survey line 708; `spectral.py:87,96` -> `fn_openpipeflow-122/torchcouette/torchcouette/spectral.py`; tokens=forward,norm;numbers=1; claim: - Couette uses 'norm="forward"' ('spectral.py:87,96'), so the forward FFT carries the 1/N scaling.
- **A0167 [VERIFIED]:** survey line 710; `spectral.py:65-79` -> `fn_openpipeflow-122/torchchannel/torchchannel/spectral.py`; tokens=Reality;numbers=0,0; claim: Reality is enforced by A[-k,-m]=overline(A[k,m]) with A[0,0] real ('spectral.py:65-79' channel; 'enforce-m0-reality' couette 'spectral.py:55-64').
- **A0168 [VERIFIED]:** survey line 710; `spectral.py:55-64` -> `fn_openpipeflow-122/torchchannel/torchchannel/spectral.py`; numbers=0,0; claim: Reality is enforced by A[-k,-m]=overline(A[k,m]) with A[0,0] real ('spectral.py:65-79' channel; 'enforce-m0-reality' couette 'spectral.py:55-64').
- **A0169 [VERIFIED]:** survey line 714; `operators.py:78-110` -> `fn_openpipeflow-122/torchchannel/torchchannel/operators.py`; tokens=alpha,omega;numbers=2,2,0,0,2; claim: Channel (Cartesian): partial-x f=i k-alpha f, partial-z f=i m-beta f, wall-normal partial-y,partial-(yy) via 'einsum("ij,...jkm->...ikm", W, f)'; nabla2 f=partial-(yy)f-k2 f; nabla!cdot!mathbf u=…
- **A0170 [VERIFIED]:** survey line 714; `VALIDATION.md:35-37` -> `fn_openpipeflow-122/torchchannel/VALIDATION.md`; tokens=Cartesian;numbers=0,0; claim: Channel (Cartesian): partial-x f=i k-alpha f, partial-z f=i m-beta f, wall-normal partial-y,partial-(yy) via 'einsum("ij,...jkm->...ikm", W, f)'; nabla2 f=partial-(yy)f-k2 f; nabla!cdot!mathbf u=…
- **A0171 [VERIFIED]:** survey line 716; `operators.py:12-19` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/operators.py`; tokens=theta; claim: Couette (cylindrical) uses the pm decomposition u-pm=u-rpm i u-theta ('operators.py:12-19') which diagonalizes the (u-r,u-theta) Helmholtz coupling. The angular+axial spectral diagonal is
- **A0172 [VERIFIED]:** survey line 720; `operators.py:100-102` -> `fn_openpipeflow-122/torchcouette/torchcouette/operators.py`; tokens=_diagonal,radlap; claim: and text(radlap)(f,pm)=W-(text(radlap))f+text(mode-diagonal)(pm),f ('operators.py:100-102').
- **A0173 [VERIFIED]:** survey line 724; `solver.py:87` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; numbers=2; claim: The scheme is a one-stage θ-method (Crank–Nicolson-like) with diffusion split implicit/explicit and nonlinear + base-flow advection explicit, wrapped in a fixed-point predictor/corrector. The implicit fracti…
- **A0174 [PARTIAL]:** survey line 726; `solver.py:215-216` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The per-mode diffusion system, with L=W-(dy2)-k2 I and diffusivity nu-(rm eff) ('solver.py:215-216'):
- **A0175 [PARTIAL]:** survey line 731; `solver.py:225` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: with nu-(rm eff)=1/Re for velocity ('-build-diffusion-system(1/Re)', 'solver.py:225') and nu-(rm eff)=1/Rm for magnetic field (§I.B.5). Wall rows of L-(rm lhs) are overwritten with Dirichlet rows ('-write--b…
- **A0176 [PARTIAL]:** survey line 731; `solver.py:217-218` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: with nu-(rm eff)=1/Re for velocity ('-build-diffusion-system(1/Re)', 'solver.py:225') and nu-(rm eff)=1/Rm for magnetic field (§I.B.5). Wall rows of L-(rm lhs) are overwritten with Dirichlet rows ('-write--b…
- **A0177 [VERIFIED]:** survey line 733; `solver.py:571-577` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_rhs_for_state; claim: The RHS assembly ('-rhs-for-state', 'solver.py:571-577') for cin(u,v,w) is
- **A0178 [VERIFIED]:** survey line 737; `solver.py:534-540` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_base_coupling_terms,coupling,perturbation; claim: where the explicit base-flow coupling (perturbation form only, '-base-coupling-terms', 'solver.py:534-540') is
- **A0179 [VERIFIED]:** survey line 741; `solver.py:542-550` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=nonlinear,omega,rotational; claim: (the -U',v term is the lift-up coupling), and N-c is the explicit nonlinear term (default rotational mathbf utimesboldsymbolomega, 'solver.py:542-550'; or convective -(mathbf ucdotnabla)mathbf u…
- **A0180 [VERIFIED]:** survey line 741; `solver.py:552-561` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=convective,nonlinear; claim: (the -U',v term is the lift-up coupling), and N-c is the explicit nonlinear term (default rotational mathbf utimesboldsymbolomega, 'solver.py:542-550'; or convective -(mathbf ucdotnabla)mathbf u…
- **A0181 [PARTIAL]:** survey line 743; `solver.py:594-606` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The full step ('step', 'solver.py:594-606'):
- **A0182 [VERIFIED]:** survey line 754; `solver.py:84-85` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=corrector,corrector_iterations;numbers=1; claim: The corrector blends the nonlinear term with the same weight theta (an Adams-style blend about the old state). Default 'corrector-iterations=1' (channel, 'solver.py:84-85').
- **A0183 [VERIFIED]:** survey line 756; `solver.py:553-594` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; tokens=Couette,StepInfo,converged,corrector,couette;numbers=3,1,10,-10,1; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577…
- **A0184 [VERIFIED]:** survey line 756; `solver.py:574-577` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; numbers=1,1,1,1; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577…
- **A0185 [VERIFIED]:** survey line 756; `solver.py:257-260` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; tokens=operator;numbers=1,1,1,1,2; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577…
- **A0186 [VERIFIED]:** survey line 756; `solver.py:467-470` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; tokens=_rhs_meshmult,radial;numbers=1,1,1,1,2; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577…
- **A0187 [VERIFIED]:** survey line 756; `solver.py:496-498` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=operator; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577…
- **A0188 [VERIFIED]:** survey line 758; `solver.py:579-592` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_solve_project_correct,bearing,correct,project,solve; claim: The ordered solve-project-correct sequence is load-bearing ('-solve-project-correct', 'solver.py:579-592'):
- **A0189 [VERIFIED]:** survey line 768; `solver.py:676-697` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=alpha,cfl_dt;numbers=2,2,4,3; claim: CFL. Delta t=text(cfl)cdotmin over directions: streamwise (2pi//alpha/)/Z/max/u/, wall-normal Delta y-(min)/max/v/, spanwise (2pi//beta/)/Th/max/w/ ('cfl-dt', 'solver.py:676-697'). Because base-…
- **A0190 [VERIFIED]:** survey line 768; `README.md:29-31` -> `fn_openpipeflow-122/torchchannel/README.md`; tokens=advection,cfl_dt,explicit,respect;numbers=2,2,3; claim: CFL. Delta t=text(cfl)cdotmin over directions: streamwise (2pi//alpha/)/Z/max/u/, wall-normal Delta y-(min)/max/v/, spanwise (2pi//beta/)/Th/max/w/ ('cfl-dt', 'solver.py:676-697'). Because base-…
- **A0191 [VERIFIED]:** survey line 768; `solver.py:635-644` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_adjust_flux,streamwise;numbers=2,2,4,3; claim: CFL. Delta t=text(cfl)cdotmin over directions: streamwise (2pi//alpha/)/Z/max/u/, wall-normal Delta y-(min)/max/v/, spanwise (2pi//beta/)/Th/max/w/ ('cfl-dt', 'solver.py:676-697'). Because base-…
- **A0192 [VERIFIED]:** survey line 772; `linstab.py:61-136` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=alpha,orr_sommerfeld_squire;numbers=2,2,2,2; claim: 'orr-sommerfeld-squire' ('linstab.py:61-136') solves Amathbf q=lambda Bmathbf q on the same FD matrices (D=W-(dy1), D2=W-(dy2)) via 'scipy.linalg.eig' ('linstab.py:109'). With L=D2-k2I, L2=L,L, k2=alph…
- **A0193 [VERIFIED]:** survey line 772; `linstab.py:109` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=linalg,scipy; claim: 'orr-sommerfeld-squire' ('linstab.py:61-136') solves Amathbf q=lambda Bmathbf q on the same FD matrices (D=W-(dy1), D2=W-(dy2)) via 'scipy.linalg.eig' ('linstab.py:109'). With L=D2-k2I, L2=L,L, k2=alph…
- **A0194 [VERIFIED]:** survey line 772; `linstab.py:99-104` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=alpha;numbers=2,2,2,2; claim: 'orr-sommerfeld-squire' ('linstab.py:61-136') solves Amathbf q=lambda Bmathbf q on the same FD matrices (D=W-(dy1), D2=W-(dy2)) via 'scipy.linalg.eig' ('linstab.py:109'). With L=D2-k2I, L2=L,L, k2=alph…
- **A0195 [VERIFIED]:** survey line 777; `linstab.py:124-125` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=alpha,phase,speed;numbers=0,0; claim: Eigenvalues are temporal rates lambda for exp(ialpha x+ibeta z+lambda t); phase speed c=ilambda/alpha ('linstab.py:124-125'). OS BC rows clamp v=Dv=0 at both walls; Squire rows clamp eta=0 ('linstab.py…
- **A0196 [VERIFIED]:** survey line 777; `linstab.py:42-58` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=Squire;numbers=0,0; claim: Eigenvalues are temporal rates lambda for exp(ialpha x+ibeta z+lambda t); phase speed c=ilambda/alpha ('linstab.py:124-125'). OS BC rows clamp v=Dv=0 at both walls; Squire rows clamp eta=0 ('linstab.py…
- **A0197 [PARTIAL]:** survey line 785; `solver.py:495-502` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (a) Pressure-Poisson projection 'project-rhs' ('solver.py:495-502'): compute interior operatorname(div), zero its wall rows, solve p=L-(rm pois)(-1)(operatorname(div)), subtract nabla p. Because W-(dy2)=…
- **A0198 [PARTIAL]:** survey line 787; `solver.py:226-242` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (b) Poisson operator 'L-pois' ('solver.py:226-242'): per-mode W-(dy2)-k2I with Neumann wall rows (partial-y stencil, '-write--boundary-row' order 1, 'solver.py:232-233'). The mean mode (k(=)0,m(=)0) is…
- **A0199 [PARTIAL]:** survey line 787; `solver.py:232-233` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (b) Poisson operator 'L-pois' ('solver.py:226-242'): per-mode W-(dy2)-k2I with Neumann wall rows (partial-y stencil, '-write--boundary-row' order 1, 'solver.py:232-233'). The mean mode (k(=)0,m(=)0) is…
- **A0200 [PARTIAL]:** survey line 787; `solver.py:235-240` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (b) Poisson operator 'L-pois' ('solver.py:226-242'): per-mode W-(dy2)-k2I with Neumann wall rows (partial-y stencil, '-write--boundary-row' order 1, 'solver.py:232-233'). The mean mode (k(=)0,m(=)0) is…
- **A0201 [PARTIAL]:** survey line 787; `solver.py:129,263-272` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (b) Poisson operator 'L-pois' ('solver.py:226-242'): per-mode W-(dy2)-k2I with Neumann wall rows (partial-y stencil, '-write--boundary-row' order 1, 'solver.py:232-233'). The mean mode (k(=)0,m(=)0) is…
- **A0202 [PARTIAL]:** survey line 789; `solver.py:331-339` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (c) 8×8 real influence (capacitance) matrix. Velocity BCs are no-slip Dirichlet; for 'state-form="full"' the moving Couette walls are imposed as mean-mode wall values text(lo)cdot Z,Th, text(hi)cdot Z,Th (…
- **A0203 [PARTIAL]:** survey line 789; `solver.py:295-314` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: (c) 8×8 real influence (capacitance) matrix. Velocity BCs are no-slip Dirichlet; for 'state-form="full"' the moving Couette walls are imposed as mean-mode wall values text(lo)cdot Z,Th, text(hi)cdot Z,Th (…
- **A0204 [PARTIAL]:** survey line 794; `solver.py:357-420` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: '-build-influence-matrix' ('solver.py:357-420') assembles 8 basis responses (velocity-unit wall sources for u,v,w at each wall, plus a pressure-gradient response from a unit wall Poisson source). The complex residual…
- **A0205 [PARTIAL]:** survey line 794; `solver.py:316-320` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: '-build-influence-matrix' ('solver.py:357-420') assembles 8 basis responses (velocity-unit wall sources for u,v,w at each wall, plus a pressure-gradient response from a unit wall Poisson source). The complex residual…
- **A0206 [PARTIAL]:** survey line 794; `solver.py:416-419` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: '-build-influence-matrix' ('solver.py:357-420') assembles 8 basis responses (velocity-unit wall sources for u,v,w at each wall, plus a pressure-gradient response from a unit wall Poisson source). The complex residual…
- **A0207 [PARTIAL]:** survey line 796; `solver.py:474-482` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Authoritative final cleanup 'enforce-constraints' ('solver.py:474-482'): builds a dense per-mode constraint matrix C ('-build-constraint-pinv', 'solver.py:422-463') of (N-2) interior-divergence rows plus 8 wall…
- **A0208 [PARTIAL]:** survey line 796; `solver.py:422-463` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Authoritative final cleanup 'enforce-constraints' ('solver.py:474-482'): builds a dense per-mode constraint matrix C ('-build-constraint-pinv', 'solver.py:422-463') of (N-2) interior-divergence rows plus 8 wall…
- **A0209 [PARTIAL]:** survey line 798; `solver.py:225-227` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0210 [PARTIAL]:** survey line 798; `solver.py:252-253` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0211 [PARTIAL]:** survey line 798; `solver.py:325-383` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0212 [PARTIAL]:** survey line 798; `solver.py:288-320` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0213 [PARTIAL]:** survey line 798; `solver.py:262-265` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0214 [PARTIAL]:** survey line 798; `solver.py:269-274` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Couette uses the analogue: '-pressure-radial-matrix = W-dr1@W-dr1 + W-dr1·(1/r)' ('solver.py:225-227', again W-(dr1)2 for exact operatorname(div)operatorname(grad)), a Neumann pressure Poisson 'LNp' with the…
- **A0215 [VERIFIED]:** survey line 804; `solver.py:57-74` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=State,complex;numbers=7,0; claim: 'torchpipeflow' is a serial, GPU-friendly, differentiable port of the OpenPipeFlow [Willis17] core. Hydrodynamic only — no MHD (verified: grep for magnetic/induction/Lorentz/Rm/Pm/Ha returns 0 substantive hits a…
- **A0216 [VERIFIED]:** survey line 808; `operators.py:45-47` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/operators.py`; tokens=actual,alpha;numbers=2; claim: Cylindrical (r,theta,z), rin[0,1] (wall at r[-1]=1, near-axis at r[0]approx0). Spectral in (z,theta), banded FD in r. Axial modes kin[-K1,K1], K1=K-1, Kc=2K1+1; azimuthal min[0,M-1] stored non-…
- **A0217 [VERIFIED]:** survey line 808; `solver.py:296` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; numbers=0,1,1,0,1; claim: Cylindrical (r,theta,z), rin[0,1] (wall at r[-1]=1, near-axis at r[0]approx0). Spectral in (z,theta), banded FD in r. Axial modes kin[-K1,K1], K1=K-1, Kc=2K1+1; azimuthal min[0,M-1] stored non-…
- **A0218 [PARTIAL]:** survey line 810; `solver.py:200-203` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; resolved-line-exists-but-semantic-support-not-proven; claim: Base flow = Hagen–Poiseuille, written explicitly ('solver.py:200-203'):
- **A0219 [VERIFIED]:** survey line 816; `solver.py:110-131` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=True,alpha,complex128,const_flux,device;numbers=64,18,32,1,0.75; claim: Default constructor ('solver.py:110-131'): 'N=64, K=18, M=32, Mp=1, alpha=0.75, Re=4000.0, dt=1e-3, implicit=0.5, KL=4, const-flux=True, nonlinearity-form="rotational", state-form="perturbation", device="cpu", dtype=com…
- **A0220 [VERIFIED]:** survey line 820; `mesh.py:60-79` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=Chebyshev,Radial,cheby,extrema,iterations;numbers=0,1,10,0,-1; claim: Mesh points are a modified Chebyshev-extrema mesh on [0,1] with 10 shift iterations pushing r[0]to0, then r[-1] pinned to 1 ('mesh.py:60-79', mirrors OpenPipeFlow 'mes-rdom-init'/'cheby.f'). FD weights via the sa…
- **A0221 [VERIFIED]:** survey line 820; `mesh.py:26-57` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=Taylor,Vandermonde,mirrors,weights;numbers=0,1,0,-1,1; claim: Mesh points are a modified Chebyshev-extrema mesh on [0,1] with 10 shift iterations pushing r[0]to0, then r[-1] pinned to 1 ('mesh.py:60-79', mirrors OpenPipeFlow 'mes-rdom-init'/'cheby.f'). FD weights via the sa…
- **A0222 [VERIFIED]:** survey line 820; `mesh.py:165` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=radlap;numbers=0,1,0,1,1; claim: Mesh points are a modified Chebyshev-extrema mesh on [0,1] with 10 shift iterations pushing r[0]to0, then r[-1] pinned to 1 ('mesh.py:60-79', mirrors OpenPipeFlow 'mes-rdom-init'/'cheby.f'). FD weights via the sa…
- **A0223 [VERIFIED]:** survey line 820; `mesh.py:180-206` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=intrdr,mes_rdom_init;numbers=0,1,0,1,4; claim: Mesh points are a modified Chebyshev-extrema mesh on [0,1] with 10 shift iterations pushing r[0]to0, then r[-1] pinned to 1 ('mesh.py:60-79', mirrors OpenPipeFlow 'mes-rdom-init'/'cheby.f'). FD weights via the sa…
- **A0224 [VERIFIED]:** survey line 822; `mesh.py:134-140` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=Fortran,ghost,physical;numbers=0,1,+1,1,0; claim: Axis treatment is parity folding, not L'Hôpital. Regularity at r=0 is enforced by reflecting through the axis: ghost points at negative radius r-(rm ext)[:KL]=-text(flip)(r[:KL]) ('mesh.py:134-140') carry phys…
- **A0225 [VERIFIED]:** survey line 822; `operators.py:61-70` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/operators.py`; tokens=Fortran,matching,parity;numbers=0,-1,2,1,+1; claim: Axis treatment is parity folding, not L'Hôpital. Regularity at r=0 is enforced by reflecting through the axis: ghost points at negative radius r-(rm ext)[:KL]=-text(flip)(r[:KL]) ('mesh.py:134-140') carry phys…
- **A0226 [VERIFIED]:** survey line 822; `solver.py:245-259` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=Fortran,_build_radlap_fold_parts,columns,folded,ghost;numbers=0,-1,2,1,+1; claim: Axis treatment is parity folding, not L'Hôpital. Regularity at r=0 is enforced by reflecting through the axis: ghost points at negative radius r-(rm ext)[:KL]=-text(flip)(r[:KL]) ('mesh.py:134-140') carry phys…
- **A0227 [VERIFIED]:** survey line 822; `solver.py:278` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=parity;numbers=2,2; claim: Axis treatment is parity folding, not L'Hôpital. Regularity at r=0 is enforced by reflecting through the axis: ghost points at negative radius r-(rm ext)[:KL]=-text(flip)(r[:KL]) ('mesh.py:134-140') carry phys…
- **A0228 [VERIFIED]:** survey line 824; `solver.py:222-228` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=alpha; claim: The (k,m) spectral part is added as a diagonal d-field per PM-component ('solver.py:222-228'; 'build-d-fields' 'banded.py:76-94'), with m-(rm act)=m,Mp, kalpha=k,alpha:
- **A0229 [VERIFIED]:** survey line 824; `banded.py:76-94` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded.py`; tokens=alpha,build_d_fields,field; claim: The (k,m) spectral part is added as a diagonal d-field per PM-component ('solver.py:222-228'; 'build-d-fields' 'banded.py:76-94'), with m-(rm act)=m,Mp, kalpha=k,alpha:
- **A0230 [VERIFIED]:** survey line 830; `operators.py:12-24` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/operators.py`; tokens=theta; claim: The pm velocity variables are u-+=u-r+iu-theta, u--=u-r-iu-theta ('operators.py:12-24'), decoupling the (u-r,u-theta) Helmholtz problems into scalar problems for u-+,u--,u-z.
- **A0231 [VERIFIED]:** survey line 834; `solver.py:266-292` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=Dirichlet,Helmholtz,Neumann,_build_operator_matrix,build;numbers=1,1,0,1,1; claim: The dense build ('-build-operator-matrix', 'solver.py:266-292') forms A=c-2(text(radlap-folded)+operatorname(diag)(d))+c-1 I then overwrites the last row (the wall) with the BC stencil dr1[:,BC]. Operators ('…
- **A0232 [VERIFIED]:** survey line 834; `solver.py:294-310` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=Neumann,Operators,Poisson,_build_linear_operators,_build_operator_matrix;numbers=1,1,0,1,1; claim: The dense build ('-build-operator-matrix', 'solver.py:266-292') forms A=c-2(text(radlap-folded)+operatorname(diag)(d))+c-1 I then overwrites the last row (the wall) with the BC stencil dr1[:,BC]. Operators ('…
- **A0233 [VERIFIED]:** survey line 836; `banded.py:10-28` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded.py`; tokens=GBTRF,LAPACK,banded,dense_to_banded; claim: The banded LU path is the OpenPipeFlow-faithful core. LAPACK GBTRF band layout ('dense-to-banded', 'banded.py:10-28'): leading dimension
- **A0234 [VERIFIED]:** survey line 840; `banded.py:109-172` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded.py`; tokens=Banded,build_operator_banded;numbers=4,13,3,1,1; claim: main diagonal in row kl+ku, with A[i,j]to AB[kl+ku+(i-j),,j]. With kl=ku=KL=4, text(ldab)=13 — exactly matching Fortran's 3cdot i-KL+1 (timestep.f90:96). 'build-operator-banded' ('banded.py:109-172') fill…
- **A0235 [VERIFIED]:** survey line 840; `banded_operators.py:29-58` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded_operators.py`; tokens=Banded,build_banded_operator_set,build_operator_banded;numbers=4,3,1,1,1; claim: main diagonal in row kl+ku, with A[i,j]to AB[kl+ku+(i-j),,j]. With kl=ku=KL=4, text(ldab)=13 — exactly matching Fortran's 3cdot i-KL+1 (timestep.f90:96). 'build-operator-banded' ('banded.py:109-172') fill…
- **A0236 [VERIFIED]:** survey line 840; `test_banded.py:36-52` -> `fn_openpipeflow-122/torchpipeflow/tests/test_banded.py`; tokens=Banded,build_banded_operator_set,dense;numbers=4,3,1,1,1; claim: main diagonal in row kl+ku, with A[i,j]to AB[kl+ku+(i-j),,j]. With kl=ku=KL=4, text(ldab)=13 — exactly matching Fortran's 3cdot i-KL+1 (timestep.f90:96). 'build-operator-banded' ('banded.py:109-172') fill…
- **A0237 [VERIFIED]:** survey line 842; `banded_solvers.py:85-155` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded_solvers.py`; tokens=Complex,banded,gbtrf,solve;numbers=2,0,0,1.,0.; claim: The banded LU solve ('banded-solvers.py:85-155') batches the factorization over all H=Kccdot M modes and solves all modes at once. Backends: '"dense"' (default; reconstructs dense and uses 'torch.linalg.lu-factor…
- **A0238 [VERIFIED]:** survey line 842; `banded_solvers.py:118-122` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded_solvers.py`; tokens=Poisson,banded,bandedlu,conditioned,dense;numbers=2,0,0,1.,0.; claim: The banded LU solve ('banded-solvers.py:85-155') batches the factorization over all H=Kccdot M modes and solves all modes at once. Backends: '"dense"' (default; reconstructs dense and uses 'torch.linalg.lu-factor…
- **A0239 [VERIFIED]:** survey line 844; `solver_banded.py:10-111` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver_banded.py`; tokens=Helmholtz,PipeFlowSolverTorchBanded,Poisson,banded,solve;numbers=2,64,10,10; claim: The 'torch-bandedlu' extension implements pivoted banded LU: CPU 'gbtrf-cpu'/'gbtrs-cpu' with partial pivoting within the band and float32/64 only; CUDA kernels parallelize one matrix per thread (batch) and 'batchnrhs'…
- **A0240 [VERIFIED]:** survey line 844; `test_banded.py:79-100` -> `fn_openpipeflow-122/torchpipeflow/tests/test_banded.py`; tokens=PipeFlowSolverTorchBanded,banded,build,dense,solve;numbers=2,10,-10,10,-12; claim: The 'torch-bandedlu' extension implements pivoted banded LU: CPU 'gbtrf-cpu'/'gbtrs-cpu' with partial pivoting within the band and float32/64 only; CUDA kernels parallelize one matrix per thread (batch) and 'batchnrhs'…
- **A0241 [VERIFIED]:** survey line 848; `operators.py:132-159` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/operators.py`; tokens=alpha;numbers=1,0,2,4,4; claim: Same OpenPipeFlow PPE-projection + influence (capacitance) matrix, in cylindrical pm variables ('operators.py:132-159'): operatorname(div)(u-r,u-theta,u-z)=u-r/r+partial-r u-r+i(m-(rm act)/r)u-theta+ikalpha,u…
- **A0242 [VERIFIED]:** survey line 848; `solver.py:731-740` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=Poisson,Projection,_project_rhs,projection,solve;numbers=1,-1,0,2,4; claim: Same OpenPipeFlow PPE-projection + influence (capacitance) matrix, in cylindrical pm variables ('operators.py:132-159'): operatorname(div)(u-r,u-theta,u-z)=u-r/r+partial-r u-r+i(m-(rm act)/r)u-theta+ikalpha,u…
- **A0243 [VERIFIED]:** survey line 848; `solver.py:329-445` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=basis,compute,influence,matrix,solutions;numbers=1,-1,0,2,4; claim: Same OpenPipeFlow PPE-projection + influence (capacitance) matrix, in cylindrical pm variables ('operators.py:132-159'): operatorname(div)(u-r,u-theta,u-z)=u-r/r+partial-r u-r+i(m-(rm act)/r)u-theta+ikalpha,u…
- **A0244 [VERIFIED]:** survey line 848; `spectral.py:61-81` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/spectral.py`; tokens=Hermitian,enforce_m0_reality;numbers=1,-1,0,2,4; claim: Same OpenPipeFlow PPE-projection + influence (capacitance) matrix, in cylindrical pm variables ('operators.py:132-159'): operatorname(div)(u-r,u-theta,u-z)=u-r/r+partial-r u-r+i(m-(rm act)/r)u-theta+ikalpha,u…
- **A0245 [VERIFIED]:** survey line 852; `solver.py:881-909` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=_apply_hpf_coupling,coupling,vel_addHPF;numbers=1,2,2; claim: Base-flow (HPF) coupling (perturbation form, '-apply-hpf-coupling', 'solver.py:881-909', = 'vel-addHPF'), with U-z=1-r2, -U-z'=2r:
- **A0246 [VERIFIED]:** survey line 858; `solver.py:911-919` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=const_flux,correction;numbers=4; claim: Constant-flux driving adds a mean-mode axial pressure correction propto 4/Re (the laminar HPF pressure gradient) and a post-step flux adjustment to zero mean axial disturbance flux ('solver.py:911-919, 785-798'). 'co…
- **A0247 [VERIFIED]:** survey line 860; `solver.py:1028-1114` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=_dterr,_step_with_history,corrector,dterr,max_corrector_iters;numbers=1,1,1,3,1e-10; claim: The time integrator is the same family θ-method predictor/corrector ('-step-with-history', 'solver.py:1028-1114'): RHS text(rhs)=N+tfrac(1)(Delta t)b+(1-theta)nu[text(radlap)(b)+d,b] ('-rhs-meshmult', 'solver.p…
- **A0248 [VERIFIED]:** survey line 860; `solver.py:678-693` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=_rhs_meshmult,radlap;numbers=1,1,1,3; claim: The time integrator is the same family θ-method predictor/corrector ('-step-with-history', 'solver.py:1028-1114'): RHS text(rhs)=N+tfrac(1)(Delta t)b+(1-theta)nu[text(radlap)(b)+d,b] ('-rhs-meshmult', 'solver.p…
- **A0249 [VERIFIED]:** survey line 860; `run_driver.py:376-430` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/run_driver.py`; tokens=Delta,_dterr,delta,dterr;numbers=1,1,1,3; claim: The time integrator is the same family θ-method predictor/corrector ('-step-with-history', 'solver.py:1028-1114'): RHS text(rhs)=N+tfrac(1)(Delta t)b+(1-theta)nu[text(radlap)(b)+d,b] ('-rhs-meshmult', 'solver.p…
- **A0250 [VERIFIED]:** survey line 862; `test_fortran_regression.py:15-16` -> `fn_openpipeflow-122/torchpipeflow/tests/test_fortran_regression.py`; tokens=Fortran,energy,reference;numbers=3.0132822082797048,0,5,0,-5; claim: Pipe golden regression: against the Fortran reference, energy =3.0132822082797048times10(-7) within rel 5times10(-5) for N(=)16,K(=)4,M(=)4,Re(=)4000,alpha(=)0.75,dt(=)10(-3),theta(=)0.5 ('test-fortran-regr…
- **A0251 [VERIFIED]:** survey line 872; `mhd.py:32-53` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=field,state,velocity;numbers=1.0,0.0,0,1,0; claim: 'ChannelMHDSolver(ChannelSolver)'; state '(velocity, bx, by, bz)' ('mhd.py:32-53'). Constructor defaults ('mhd.py:76-86'): 'Pm=1.0, Rm=None, Ha=0.0, background-b=(0,1,0), lorentz-prefactor=None, omega=0.0, shear-rate=0.…
- **A0252 [VERIFIED]:** survey line 872; `mhd.py:76-86` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=None,background_b,lorentz_prefactor,omega,shear_rate;numbers=1.0,0.0,0,1,0; claim: 'ChannelMHDSolver(ChannelSolver)'; state '(velocity, bx, by, bz)' ('mhd.py:32-53'). Constructor defaults ('mhd.py:76-86'): 'Pm=1.0, Rm=None, Ha=0.0, background-b=(0,1,0), lorentz-prefactor=None, omega=0.0, shear-rate=0.…
- **A0253 [VERIFIED]:** survey line 874; `mhd.py:89-94` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=None,self;numbers=1,1.0; claim: Rm/Pm coupling ('mhd.py:89-94'): if 'Rm is None', Pm is taken and Rm=Recdot Pm; else Rm is taken and Pm=Rm/Re. Magnetic diffusion uses diffusivity 1/Rm in the same Helmholtz template as velocity ('-buil…
- **A0254 [VERIFIED]:** survey line 874; `mhd.py:115-117` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Magnetic,_build_diffusion_system,_build_magnetic_matrices,diffusion,self;numbers=1,1.0; claim: Rm/Pm coupling ('mhd.py:89-94'): if 'Rm is None', Pm is taken and Rm=Recdot Pm; else Rm is taken and Pm=Rm/Re. Magnetic diffusion uses diffusivity 1/Rm in the same Helmholtz template as velocity ('-buil…
- **A0255 [VERIFIED]:** survey line 880; `mhd.py:100-101` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Lorentz,prefactor; claim: Lorentz prefactor ('mhd.py:100-101', quoted verbatim):
- **A0256 [VERIFIED]:** survey line 890; `mhd.py:196-214` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Induction;numbers=0.0,0,10; claim: Induction partial-tmathbf bbig/-(rm expl)=operatorname(curl)(mathbf u-(rm tot)timesmathbf B-(rm tot)) computed pseudospectrally, where mathbf u-(rm tot) includes base flow (perturbation form) and m…
- **A0257 [VERIFIED]:** survey line 890; `mhd.py:216-228` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Lorentz,current,force,lorentz_prefactor,zeros;numbers=0.0,0; claim: Induction partial-tmathbf bbig/-(rm expl)=operatorname(curl)(mathbf u-(rm tot)timesmathbf B-(rm tot)) computed pseudospectrally, where mathbf u-(rm tot) includes base flow (perturbation form) and m…
- **A0258 [VERIFIED]:** survey line 890; `test_mhd.py:109-132` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=ChannelSolver,Lorentz,field,hydro,lorentz_prefactor;numbers=0.0,0,10,-10; claim: Induction partial-tmathbf bbig/-(rm expl)=operatorname(curl)(mathbf u-(rm tot)timesmathbf B-(rm tot)) computed pseudospectrally, where mathbf u-(rm tot) includes base flow (perturbation form) and m…
- **A0259 [VERIFIED]:** survey line 892; `mhd.py:353-370` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Coupled,_magnetic_step,corrector,explicit,implicit; claim: Coupled time step ('step', 'mhd.py:353-370'): velocity step (with Lorentz added to the explicit nonlinear), then '-magnetic-step' (induction implicit-diffusion solve), with the corrector blending both momentum a…
- **A0260 [VERIFIED]:** survey line 892; `mhd.py:340-351` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=_magnetic_step,component,enforce_magnetic_constraints,induction,magnetic; claim: Coupled time step ('step', 'mhd.py:353-370'): velocity step (with Lorentz added to the explicit nonlinear), then '-magnetic-step' (induction implicit-diffusion solve), with the corrector blending both momentum a…
- **A0261 [VERIFIED]:** survey line 894; `mhd.py:278-282` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Magnetic;numbers=0,0,-1,0,0; claim: Magnetic walls — conducting/homogeneous mathbf b=0 ONLY (no insulating). Wall RHS rows are zeroed (text(rhs)[0]=text(rhs)[-1]=0, 'mhd.py:278-282') and operatorname(div)mathbf b=0 is enforced via the sam…
- **A0262 [VERIFIED]:** survey line 894; `mhd.py:304-307` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Magnetic,boundary,constraint,divergence,future;numbers=0,0,0,0,3; claim: Magnetic walls — conducting/homogeneous mathbf b=0 ONLY (no insulating). Wall RHS rows are zeroed (text(rhs)[0]=text(rhs)[-1]=0, 'mhd.py:278-282') and operatorname(div)mathbf b=0 is enforced via the sam…
- **A0263 [VERIFIED]:** survey line 898; `mhd.py:13-50` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=class,couette,device,dtype,state;numbers=1.0,0.0,1.0,0.0; claim: 'TaylorCouetteMHDSolver(TaylorCouetteSolver)'; state '(velocity, br, bt, bz)' ('mhd.py:13-50', with a '.to(device,dtype)' mover). Constructor defaults ('mhd.py:62-69'): 'Pm=1.0, Ha=0.0, background-bt=1.0, background-bz=…
- **A0264 [VERIFIED]:** survey line 898; `mhd.py:62-69` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=None,background_bt,background_bz,lorentz_prefactor;numbers=1.0,0.0,1.0,0.0; claim: 'TaylorCouetteMHDSolver(TaylorCouetteSolver)'; state '(velocity, br, bt, bz)' ('mhd.py:13-50', with a '.to(device,dtype)' mover). Constructor defaults ('mhd.py:62-69'): 'Pm=1.0, Ha=0.0, background-bt=1.0, background-bz=…
- **A0265 [VERIFIED]:** survey line 900; `mhd.py:79-81` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Lorentz,prefactor; claim: Lorentz prefactor ('mhd.py:79-81', quoted verbatim):
- **A0266 [VERIFIED]:** survey line 910; `mhd.py:153-163` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=_build_magnetic_operators,magnetic;numbers=1,1,1; claim: Pm / magnetic diffusion ('-build-magnetic-operators', 'mhd.py:153-163'): c-1=1/Delta t, LHS c-2=-theta/Pm, RHS (1-theta)/Pm, i.e. magnetic diffusivity =1/Pm relative to unit-viscosity momentum (so Pm=nu…
- **A0267 [VERIFIED]:** survey line 912; `mhd.py:134-141` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Background,background,field,profile;numbers=1.0,0.0; claim: Background field is current-free ('mhd.py:134-141'): toroidal B-theta(r)=text(background-bt)cdot r-i/r (the propto1/r current-free profile) and axial B-z=text(background-bz), set on the mean mode; defau…
- **A0268 [VERIFIED]:** survey line 914; `mhd.py:316-337` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Induction,Magnetic,magnetic,velocity;numbers=0,8,8,3; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339…
- **A0269 [VERIFIED]:** survey line 914; `mhd.py:339-370` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Lorentz,Magnetic,magnetic,velocity;numbers=0,8,8,3; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339…
- **A0270 [VERIFIED]:** survey line 914; `mhd.py:179-231` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Magnetic,_build_magnetic_influence_matrix,influence,magnetic,matrix;numbers=0,8,8,3; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339…
- **A0271 [VERIFIED]:** survey line 914; `mhd.py:53-60` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Bessel,Exact,Induction,Magnetic,behind;numbers=0,8,8,3; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339…
- **A0272 [VERIFIED]:** survey line 914; `mhd.py:409-439` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=StepInfo,corrector,max_corrector_iters,step_with_info;numbers=0,8,8,3,1e-10; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339…
- **A0273 [VERIFIED]:** survey line 918; `mhd.py:441-453` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Emag,Emag_total,Epert,Maxwell,Reynolds;numbers=2,0; claim: Channel 'diagnostics()' ('mhd.py:441-453') returns '(Epert, Emag, Emag-total, divLinf, divB-Linf, divB-L2, reynolds-xy, maxwell-xy, transport-xy, alpha, q-shear)'. Reynolds stress =langle u-x u-yrangle; Maxwell stre…
- **A0274 [VERIFIED]:** survey line 918; `mhd.py:412` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=2,0; claim: Channel 'diagnostics()' ('mhd.py:441-453') returns '(Epert, Emag, Emag-total, divLinf, divB-Linf, divB-L2, reynolds-xy, maxwell-xy, transport-xy, alpha, q-shear)'. Reynolds stress =langle u-x u-yrangle; Maxwell stre…
- **A0275 [VERIFIED]:** survey line 918; `mhd.py:480-488` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Couette,Emag,Emag_total,Maxwell,Reynolds;numbers=2,0; claim: Channel 'diagnostics()' ('mhd.py:441-453') returns '(Epert, Emag, Emag-total, divLinf, divB-Linf, divB-L2, reynolds-xy, maxwell-xy, transport-xy, alpha, q-shear)'. Reynolds stress =langle u-x u-yrangle; Maxwell stre…
- **A0276 [VERIFIED]:** survey line 918; `mhd.py:476-478` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=Couette,Lorentz,Maxwell,Reynolds,maxwell; claim: Channel 'diagnostics()' ('mhd.py:441-453') returns '(Epert, Emag, Emag-total, divLinf, divB-Linf, divB-L2, reynolds-xy, maxwell-xy, transport-xy, alpha, q-shear)'. Reynolds stress =langle u-x u-yrangle; Maxwell stre…
- **A0277 [VERIFIED]:** survey line 926; `mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=class; claim: The class docstring states it verbatim ('mhd.py:71-74'):
- **A0278 [PARTIAL]:** survey line 932; `mhd.py:102-104` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: The constructor merely records them and forms a diagnostic ratio ('mhd.py:102-104'):
- **A0279 [PARTIAL]:** survey line 938; `mhd.py:452` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: 'self.omega' and 'self.shear-rate' are referenced nowhere in any RHS/timestepping path; 'q-shear' is emitted only in 'diagnostics()' ('mhd.py:452'). There is no Coriolis term 2boldsymbolOmegatimesmathbf u, n…
- **A0280 [VERIFIED]:** survey line 940; `solver.py:534-540` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_base_coupling_terms,coupling,perturbation,state_form; claim: The only shear-like coupling present is the hydrodynamic base-flow advection ('-base-coupling-terms', 'solver.py:534-540'): for 'state-form="perturbation"', b-u=-Upartial-x u-U'v, b-v=-Upartial-x v, b-w=-U…
- **A0281 [PARTIAL]:** survey line 942; `test_mhd.py:135-150` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: A test confirms the metadata-only behavior: with 'omega=1.0, shear-rate=1.0', diagnostics return q-(rm shear)=1.0 but the step is otherwise an ordinary MHD step ('test-mhd.py:135-150'). Planned wiring (Part III.1…
- **A0282 [PARTIAL]:** survey line 942; `solver.py:571-577` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: A test confirms the metadata-only behavior: with 'omega=1.0, shear-rate=1.0', diagnostics return q-(rm shear)=1.0 but the step is otherwise an ordinary MHD step ('test-mhd.py:135-150'). Planned wiring (Part III.1…
- **A0283 [VERIFIED]:** survey line 950; `test_mhd.py:183-201` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=Autograd,Lorentz,assert,autograd,channel;numbers=0,0,2.0; claim: - Autograd: the solvers are fully differentiable end-to-end through projection, BC correction, FFTs, and the MHD Lorentz coupling — no '.detach()' in step paths ('torch.no-grad()' only in IO). The banded 'lu-sol…
- **A0284 [VERIFIED]:** survey line 950; `test_mhd.py:105-130` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=Lorentz,assert,channel,couette,lorentz_prefactor;numbers=0,0,2.0; claim: - Autograd: the solvers are fully differentiable end-to-end through projection, BC correction, FFTs, and the MHD Lorentz coupling — no '.detach()' in step paths ('torch.no-grad()' only in IO). The banded 'lu-sol…
- **A0285 [VERIFIED]:** survey line 956; `tests/test_linstab_poiseuille.py:7-19` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=Poiseuille,alpha,leading;numbers=0.23752649,+0.00373967,10000,1,0; claim: / Poiseuille OS leading c (ref) / 0.23752649+0.00373967,i / Re(=)10000,alpha(=)1,beta(=)0,N(=)101,KL(=)4; 'tests/test-linstab-poiseuille.py:7-19', asserts /c-c-(rm ref)/<10(-4) /
- **A0286 [VERIFIED]:** survey line 957; `VALIDATION.md:83-96` -> `fn_openpipeflow-122/torchchannel/VALIDATION.md`; tokens=Poiseuille,computed,leading;numbers=0.23752722198590992,+0.0037381198835812705; claim: / Poiseuille OS leading c (computed) / 0.23752722198590992+0.0037381198835812705,i / 'VALIDATION.md:83-96' /
- **A0287 [VERIFIED]:** survey line 958; `test_linstab_poiseuille.py:22-39` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=Critical,alpha,change,stable,unstable;numbers=5742.22,5802.22,1.02,96; claim: / Critical-Re sign change / stable Re(=)5742.22, unstable Re(=)5802.22 / alpha(=)1.02,N(=)96; 'test-linstab-poiseuille.py:22-39' /
- **A0288 [VERIFIED]:** survey line 959; `test_step_decay.py:29-51` -> `fn_openpipeflow-122/torchchannel/tests/test_step_decay.py`; tokens=Couette,Laminar;numbers=2000,-20,-12,-14,9; claim: / Laminar Couette 2000-step decay / E-(rm pert)<10(-20), div <10(-12), max/u/<10(-14) / N(=)9,Re(=)500,dt(=)0.01; 'test-step-decay.py:29-51' /
- **A0289 [PARTIAL]:** survey line 960; `test_mesh.py:21-48` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Mesh polynomial exactness / err <10(-7), deg 0..8; int1(=)2,int y(=)0,int y2(=)2/3 / 'test-mesh.py:21-48' /
- **A0290 [VERIFIED]:** survey line 961; `test_mhd.py:54-180` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; numbers=5; claim: / Channel MHD div-free / div, divB<10(-7); Rm(=)100,Pm(=)5Rightarrow Pm(=)100/Re / 'test-mhd.py:54-180' /
- **A0291 [VERIFIED]:** survey line 965; `test_fortran_regression.py:15-16` -> `fn_openpipeflow-122/torchpipeflow/tests/test_fortran_regression.py`; tokens=Fortran,energy;numbers=3.0132822082797048,0,5,0,-5; claim: / Pipe Fortran regression energy / 3.0132822082797048times10(-7) (rel <5times10(-5)) / 'test-fortran-regression.py:15-16' /
- **A0292 [VERIFIED]:** survey line 966; `test_banded.py:79-100` -> `fn_openpipeflow-122/torchpipeflow/tests/test_banded.py`; tokens=banded,dense;numbers=10,-10,10,-12; claim: / Pipe banded↔dense step / rtol 10(-10)/atol 10(-12) / 'test-banded.py:79-100' /
- **A0293 [VERIFIED]:** survey line 997; `channelflow_kmm.py:58-59` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=normal,spanwise,streamwise;numbers=0,1,2; claim: 0 = wall-normal 'x', axis 1 = streamwise 'y', axis 2 = spanwise/axial 'z'; §0.2 conventions table, 'channelflow-kmm.py:58-59')
- **A0294 [VERIFIED]:** survey line 1017; `__init__.py:4` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; tokens=XLA_PYTHON_CLIENT_PREALLOCATE,environ,false,setdefault; claim: - 'os.environ.setdefault("XLA-PYTHON-CLIENT-PREALLOCATE", "false")' ('--init--.py:4') — disables XLA GPU
- **A0295 [VERIFIED]:** survey line 1019; `__init__.py:8` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; tokens=True,config,jax_enable_x64,update;numbers=4,4; claim: - 'jax.config.update("jax-enable-x64", True)' ('--init--.py:8') — float64 is enabled globally and
- **A0296 [PARTIAL]:** survey line 1020; `tests/test_x64_default.py:7-10` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/test_x64_default.py`; resolved-line-exists-but-semantic-support-not-proven; claim: unconditionally. Verified by 'tests/test-x64-default.py:7-10', which asserts
- **A0297 [VERIFIED]:** survey line 1025; `pyproject.toml:19-37` -> `shenfun_jaxfun_spectralDNS/jaxfun/pyproject.toml`; tokens=device; claim: ('pyproject.toml:19-37'). There is no TPU-specific code path — portability comes from the device-agnostic
- **A0298 [VERIFIED]:** survey line 1031; `basespace.py:15-39` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/basespace.py`; tokens=BaseSpace,Cartesian,CoordSys;numbers=1; claim: 'BaseSpace' ('basespace.py:15-39') is the abstract root: it attaches a 'CoordSys' (defaulting to a 1-D Cartesian
- **A0299 [VERIFIED]:** survey line 1032; `basespace.py:39` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/basespace.py`; tokens=CartCoordSys,system; claim: system 'CartCoordSys("N",(x,))' if none is given, 'basespace.py:39'), carries 'name', 'fun-str' (symbol stem,
- **A0300 [PARTIAL]:** survey line 1052; `composite.py:475` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'composite.py:475').
- **A0301 [VERIFIED]:** survey line 1056; `galerkin/orthogonal.py:57-545` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=Orthogonal; claim: Orthogonal common interface ('galerkin/orthogonal.py:57-545'): modes 'N'; quadrature points
- **A0302 [VERIFIED]:** survey line 1057; `:69` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=_num_quad_points;numbers=69; claim: '-num-quad-points = N' (':69'); dofs '-num-dofs = N' by default, constrained '0 < num-dofs <= N' (':70-72').
- **A0303 [VERIFIED]:** survey line 1057; `:70-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=_num_dofs,num_dofs;numbers=0,70; claim: '-num-quad-points = N' (':69'); dofs '-num-dofs = N' by default, constrained '0 < num-dofs <= N' (':70-72').
- **A0304 [VERIFIED]:** survey line 1058; `:265` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=forward;numbers=265; claim: 'forward(u) = scalar-product(u) / (norm-squared / domain-factor)' by orthogonality (':265');
- **A0305 [VERIFIED]:** survey line 1059; `:278` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=domain_factor,factor,system;numbers=278; claim: 'scalar-product' injects the curvilinear weight 'sg = system.sg / domain-factor' (':278'). The affine map factor
- **A0306 [VERIFIED]:** survey line 1060; `:361` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=domain_factor;numbers=361,1; claim: 'domain-factor = (d-c)/(b-a)' (reference length / true length, ':361') handles non-'[-1,1]'/non-'[0,2π]' domains.
- **A0307 [VERIFIED]:** survey line 1061; `:315` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/orthogonal.py`; tokens=get_dealiased,padding_factor;numbers=1.5,315; claim: Dealiasing knob: 'get-dealiased(padding-factor=1.5)' (':315') deep-copies the space and sets
- **A0308 [VERIFIED]:** survey line 1067; `Chebyshev.py:160-180` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Chebyshev.py`; tokens=Chebyshev;numbers=-1,2,2,1,2; claim: / Chebyshev (T-k) / -1/2 / x-k=cos(pi+(2k+1)pi/2N), w-k=pi/N ('Chebyshev.py:160-180') / DCT/IDCT ('jax.scipy.fft.dct/idct', ':236-273') / DENSE upper-triangular (':409-459') /
- **A0309 [VERIFIED]:** survey line 1067; `:236-273` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Chebyshev.py`; tokens=Chebyshev;numbers=2,2,1,2,236; claim: / Chebyshev (T-k) / -1/2 / x-k=cos(pi+(2k+1)pi/2N), w-k=pi/N ('Chebyshev.py:160-180') / DCT/IDCT ('jax.scipy.fft.dct/idct', ':236-273') / DENSE upper-triangular (':409-459') /
- **A0310 [VERIFIED]:** survey line 1067; `:409-459` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Chebyshev.py`; tokens=Chebyshev,DENSE;numbers=2,2,1,2,409; claim: / Chebyshev (T-k) / -1/2 / x-k=cos(pi+(2k+1)pi/2N), w-k=pi/N ('Chebyshev.py:160-180') / DCT/IDCT ('jax.scipy.fft.dct/idct', ':236-273') / DENSE upper-triangular (':409-459') /
- **A0311 [VERIFIED]:** survey line 1068; `Legendre.py:20-33` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Legendre.py`; tokens=Legendre,coupling,dense,deriv,matrix;numbers=0; claim: / Legendre (P-k) / 0 / Gauss–Legendre ('utils/fastgl.py') / none (matrix) / dense deriv coupling ('Legendre.py:20-33') /
- **A0312 [VERIFIED]:** survey line 1069; `utils/common.py:174-207` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/utils/common.py`; numbers=+1,2,2; claim: / ChebyshevU (U-k) / +1/2 / Gauss–Cheb-2 / DST ('utils/common.py:174-207') / — /
- **A0313 [VERIFIED]:** survey line 1071; `Fourier.py:80-91` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; numbers=2,2; claim: / Fourier (e(ikx)) / — / x-j=2pi j/N, w=2pi/N ('Fourier.py:80-91') / FFT ('jnp.fft.fft/ifft', ':129-184') / DIAGONAL (':241-268') /
- **A0314 [VERIFIED]:** survey line 1071; `:129-184` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; numbers=2,2,129; claim: / Fourier (e(ikx)) / — / x-j=2pi j/N, w=2pi/N ('Fourier.py:80-91') / FFT ('jnp.fft.fft/ifft', ':129-184') / DIAGONAL (':241-268') /
- **A0315 [VERIFIED]:** survey line 1071; `:241-268` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; tokens=DIAGONAL,Fourier;numbers=2,2,241; claim: / Fourier (e(ikx)) / — / x-j=2pi j/N, w=2pi/N ('Fourier.py:80-91') / FFT ('jnp.fft.fft/ifft', ':129-184') / DIAGONAL (':241-268') /
- **A0316 [VERIFIED]:** survey line 1074; `:51` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; numbers=51; claim: 'N' must be even (':51'). Wavenumbers use NumPy 'fftfreq' ordering
- **A0317 [VERIFIED]:** survey line 1075; `:16-22` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; tokens=where;numbers=1,2,16,2; claim: k=text(where)(text(idx)<lceil(N+1)/2rceil, text(idx), text(idx)-N) (':16-22'); the Nyquist mode k[N/2] is
- **A0318 [VERIFIED]:** survey line 1077; `:206-208` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; numbers=206; claim: (':206-208'); 'derivative-coeffs' =(ik)m,c (':210-223'). Crucially, jaxfun uses a full-complex FFT on every
- **A0319 [VERIFIED]:** survey line 1077; `:210-223` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Fourier.py`; tokens=derivative_coeffs;numbers=210; claim: (':206-208'); 'derivative-coeffs' =(ik)m,c (':210-223'). Crucially, jaxfun uses a full-complex FFT on every
- **A0320 [PARTIAL]:** survey line 1079; `docs/couette_fourier_layout.md:11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/docs/couette_fourier_layout.md`; resolved-line-exists-but-semantic-support-not-proven; claim: works identically on CPU/GPU/TPU without an rfft-specific branch ('docs/couette-fourier-layout.md:11'), and the
- **A0321 [VERIFIED]:** survey line 1082; `galerkin/Jacobi.py:66-579` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Jacobi.py`; tokens=Jacobi; claim: Jacobi recurrence engine ('galerkin/Jacobi.py:66-579'): Gauss–Jacobi nodes via host
- **A0322 [VERIFIED]:** survey line 1083; `:124` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/Jacobi.py`; tokens=alpha,beta,roots_jacobi;numbers=124; claim: 'scipy.special.roots-jacobi(N, alpha, beta)' (':124'); three-term recurrence built symbolically (SymPy 'a(i,j)',
- **A0323 [VERIFIED]:** survey line 1091; `composite.py:25-100` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; tokens=Dirichlet,Neumann,codes,tuple; claim: ('composite.py:25-100') accepts codes 'D' (Dirichlet), 'N'/'N2'/'N3'/'N4' (k-th Neumann), 'R' (Robin, tuple
- **A0324 [VERIFIED]:** survey line 1093; `functionspace.py:42-58` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/functionspace.py`; tokens=Dirichlet,Neumann,clamped,right; claim: Dirichlet left/right, '(a,b,c,d)' → clamped (Dirichlet + Neumann) each side ('functionspace.py:42-58').
- **A0325 [VERIFIED]:** survey line 1096; `composite.py:725-792` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; tokens=get_stencil_matrix,stencil; claim: polynomials and S (a 'DiaMatrix' stencil) encodes the BCs. 'get-stencil-matrix' ('composite.py:725-792')
- **A0326 [VERIFIED]:** survey line 1100; `:324-327` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; tokens=orthogonal,stencil_width;numbers=324; claim: 'dim = orthogonal.dim − stencil-width()' (':324-327'); the mass matrix is S,P-(text(mass)),Stop,
- **A0327 [PARTIAL]:** survey line 1103; `functionspace.py:193-203` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/functionspace.py`; resolved-line-exists-but-semantic-support-not-proven; claim: Scdottext(bnd-vals) ('functionspace.py:193-203', 'composite.py:550-556').
- **A0328 [VERIFIED]:** survey line 1103; `composite.py:550-556` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/composite.py`; tokens=_vals; claim: Scdottext(bnd-vals) ('functionspace.py:193-203', 'composite.py:550-556').
- **A0329 [VERIFIED]:** survey line 1110; `galerkin/inner.py:126-193` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/inner.py`; tokens=expr,inner,kind,num_quad_points,sparse; claim: 'inner(expr, sparse, num-quad-points, kind)' ('galerkin/inner.py:126-193') is the weak-form assembler: it finds
- **A0330 [VERIFIED]:** survey line 1114; `inner.py:213-253` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/inner.py`; tokens=integrate; claim: The companion 'integrate(u, V)' ('inner.py:213-253') is the pure integral int ucdot wcdotsqrt g — the
- **A0331 [VERIFIED]:** survey line 1116; `inner.py:166` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/inner.py`; tokens=num_quad_points,tuple;numbers=1.5; claim: 'inner(..., num-quad-points=tuple(int(1.5·n)…))' ('inner.py:166').
- **A0332 [VERIFIED]:** survey line 1118; `tensorproductspace.py:98-633` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/tensorproductspace.py`; tokens=TensorProductSpace,transforms; claim: 'TensorProductSpace' ('tensorproductspace.py:98-633') realizes transforms as separable per-axis vmaps, fused/
- **A0333 [VERIFIED]:** survey line 1121; `:76-95` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/tensorproductspace.py`; tokens=Fourier,guard,zeros;numbers=2,76; claim: K-i//K/2 with a zero-mode guard (':76-95'); 'mask-nyquist' zeros the Fourier Nyquist modes.
- **A0334 [VERIFIED]:** survey line 1130; `la/eig.py:8-9` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/la/eig.py`; tokens=NONMODAL_FINITE_CAP; claim: 'NONMODAL-FINITE-CAP = 1e8' ('la/eig.py:8-9') and 'transient-growth-from-eigs' via 'svdvals'.
- **A0335 [PARTIAL]:** survey line 1134; `coordinates.py:425-1227` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/coordinates.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'coordinates.py:425-1227' is a full Riemannian layer. 'get-CoordSys(name, Lambda)' wraps a SymPy
- **A0336 [PARTIAL]:** survey line 1139; `operators.py:118-1097` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Gammak-(ij) and covariant differentiation are provided. 'operators.py:118-1097' implements curvilinear
- **A0337 [VERIFIED]:** survey line 1169; `pcf_linear_jax.py:14-16` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_linear_jax.py`; tokens=numpy; claim: Linear-solver caveat. Despite the '-jax.py' suffix, 'pcf-linear-jax.py:14-16' imports only 'numpy'; its docstring
- **A0338 [VERIFIED]:** survey line 1170; `:8-12` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_linear_jax.py`; tokens=Galerkin,NumPy,SciPy,dense,differentiable;numbers=8; claim: (':8-12') states it is "a NumPy/SciPy dense reference workflow … not a differentiable JAX Galerkin port." The TC
- **A0339 [VERIFIED]:** survey line 1172; `taylor_couette_linear_jax.py:152` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; tokens=asarray,complex,dtype; claim: 'np.asarray(..., dtype=complex)' (e.g. 'taylor-couette-linear-jax.py:152') and call 'generalized-eig'/'scipy'. So the
- **A0340 [VERIFIED]:** survey line 1178; `couette/ChannelFlow.py:44-251` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/ChannelFlow.py`; numbers=7; claim: Kim–Moin–Moser velocity–vorticity formulation [KMM87], a port of 'couette/ChannelFlow.py:44-251'
- **A0341 [VERIFIED]:** survey line 1179; `channelflow_kmm.py:56-59` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=normal,velocity; claim: ('channelflow-kmm.py:56-59'). It evolves the wall-normal velocity u-0 on a clamped biharmonic basis and the
- **A0342 [VERIFIED]:** survey line 1181; `channelflow_kmm.py:38-52` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=KMMState,pytree; claim: 'KMMState' pytree carries 'u=(u0,u1,u2)' and 'g' ('channelflow-kmm.py:38-52').
- **A0343 [VERIFIED]:** survey line 1184; `channelflow_kmm.py:58-59` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=spanwise,streamwise;numbers=1,2; claim: 1 = streamwise y (Fourier), 2 = spanwise z (Fourier) ('channelflow-kmm.py:58-59'). Default domain
- **A0344 [VERIFIED]:** survey line 1185; `channelflow_kmm.py:65-69` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; numbers=-1,1,0,4,0; claim: ((-1,1),(0,4pi),(0,2pi)) ('channelflow-kmm.py:65-69'). This is the canonical frame of §0.2 — no adapter needed against A.
- **A0345 [PARTIAL]:** survey line 1187; `channelflow_kmm.py:352` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-semantic-support-not-proven; claim: Wall-normal vorticity definition ('channelflow-kmm.py:352'):
- **A0346 [VERIFIED]:** survey line 1190; `channelflow_kmm.py:88-118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=FunctionSpace,family;numbers=0,0,0,0; claim: Spaces ('channelflow-kmm.py:88-118'): 'B0 = FunctionSpace(N0, family, bc=(0,0,0,0))' (clamped biharmonic);
- **A0347 [VERIFIED]:** survey line 1193; `channelflow_kmm.py:73` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; numbers=1,0,0,0,0; claim: the two Fourier directions only ('channelflow-kmm.py:73'). 1-D radial spaces 'D00,C00' handle the (0,0) Fourier mean modes of
- **A0348 [PARTIAL]:** survey line 1194; `channelflow_kmm.py:113-118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-semantic-support-not-proven; claim: u-1,u-2 ('channelflow-kmm.py:113-118').
- **A0349 [VERIFIED]:** survey line 1204; `channelflow_kmm.py:382-396` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=convection,gradient;numbers=0; claim: Nonlinear / convection ('convection', 'channelflow-kmm.py:382-396'): gradient (advective) form (conv=0) n-i = ucdotnabla u-i,
- **A0350 [VERIFIED]:** survey line 1206; `channelflow_kmm.py:398-413` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=_nonlinear_rhs; claim: combinations ('-nonlinear-rhs', 'channelflow-kmm.py:398-413') are
- **A0351 [VERIFIED]:** survey line 1210; `channelflow_kmm.py:409-413` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; numbers=0,00,0,0; claim: N-(w00)=-(M-(00),mathrm(Re),H-2[:,0,0]) ('channelflow-kmm.py:409-413').
- **A0352 [VERIFIED]:** survey line 1212; `channelflow_kmm.py:415-428` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=Velocity,_reconstruct_velocity; claim: Velocity reconstruction ('-reconstruct-velocity', 'channelflow-kmm.py:415-428'): with f=mathcal F(partial-x u-0),
- **A0353 [VERIFIED]:** survey line 1215; `channelflow_kmm.py:422-423` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; numbers=0,0,00,00; claim: and the (0,0) modes overwritten with the real mean solves v-(00),w-(00) ('channelflow-kmm.py:422-423'). 'K-over-K2' is defined in
- **A0354 [PARTIAL]:** survey line 1216; `tensorproductspace.py:76` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/tensorproductspace.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'tensorproductspace.py:76'.
- **A0355 [VERIFIED]:** survey line 1218; `channelflow_kmm.py:238-337` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=Pressure,compute_pressure_coefficients,optional,recovery; claim: Pressure recovery ('compute-pressure-coefficients', 'channelflow-kmm.py:238-337', optional): Poisson Delta p = -nablacdot H
- **A0356 [VERIFIED]:** survey line 1223; `:550-556` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=Divergence,divergence_l2;numbers=550; claim: Divergence diagnostic ('divergence-l2', ':550-556'): partial-x u-0+partial-y u-1+partial-z u-2
- **A0357 [PARTIAL]:** survey line 1229; `pcf_fluctuations_jax.py:29-122` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: (verified 'pcf-fluctuations-jax.py:29-122').
- **A0358 [VERIFIED]:** survey line 1231; `:54` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; numbers=54; claim: - Nondimensionalization: nu = U-(text(wall))/Re (':54'). Defaults Re=600, U-(text(wall))=1, dt=0.01,
- **A0359 [VERIFIED]:** survey line 1235; `:91-98` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; tokens=_add_base_convection;numbers=91; claim: - Base-flow coupling ('-add-base-convection', ':91-98') — exactly (U-bpartial-y)u' + (u'cdotnabla)U-b:
- **A0360 [VERIFIED]:** survey line 1239; `:69-89` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; tokens=amplitude,deterministic;numbers=69,1,2; claim: - IC (deterministic, ':69-89'): text(wall)=1-x2; amplitude-0.05 sinusoids in u'-0,u'-1,u'-2 over
- **A0361 [VERIFIED]:** survey line 1241; `:104-122` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; tokens=Diagnostics,Epert,Etot,divL2,u_bot;numbers=104; claim: - Diagnostics (':104-122'): 'Epert', 'Etot', 'divL2', 'u-top', 'u-bot',
- **A0362 [VERIFIED]:** survey line 1252; `:63-64` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_divfree.py`; numbers=63; claim: - Resistive normalization: Rm=Re if unset; eta = U-(text(wall))/Rm (':63-64').
- **A0363 [VERIFIED]:** survey line 1256; `:175-176` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_divfree.py`; numbers=175,1,1; claim: (':175-176'). The Lorentz prefactor is 1 (Alfvén units, rho=mu-0=1) — matching the §0.2 canonical
- **A0364 [VERIFIED]:** survey line 1270; `:73-74` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=73; claim: - Rotation/shear parameters (':73-74'): Omega='omega', q=S/Omega ('q-shear'), epicyclic
- **A0365 [VERIFIED]:** survey line 1273; `:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=130; claim: ('-mhd-convection', ':130-134'):
- **A0366 [VERIFIED]:** survey line 1278; `:116-118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=116; claim: mathbf B-(text(tot))=nablatimesmathbf A+text(background-b) (':116-118') is used in both Lorentz and EMF
- **A0367 [VERIFIED]:** survey line 1279; `:138-146` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; tokens=shear;numbers=138; claim: (':138-146'), so imposed B-z shear-couples b-xto b-y through the shear-induction term mathbf U-btimesmathbf B.
- **A0368 [VERIFIED]:** survey line 1280; `:80-114` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=80,1,2,3; claim: - IC seed (':80-114'): channel-mode harmonics cos/sin(text(harmonic)cdot k-z z) (harmonics 1,2,3) plus
- **A0369 [VERIFIED]:** survey line 1282; `:149-182` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=149; claim: - Diagnostics (':149-182'): Reynolds stress langle u-r u-thetarangle, Maxwell stress
- **A0370 [PARTIAL]:** survey line 1294; `taylor_couette_linear_jax.py:37-89` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'taylor-couette-linear-jax.py:37-89'):
- **A0371 [PARTIAL]:** survey line 1298; `taylor_couette_dns_jax.py:159` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: Re=Omega-1 R-1,text(gap)/nu ('taylor-couette-dns-jax.py:159').
- **A0372 [VERIFIED]:** survey line 1299; `taylor_couette_dns_jax.py:161-174` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=num_dofs;numbers=2; claim: - Spaces ('taylor-couette-dns-jax.py:161-174'): 'SD' (Dirichlet radial velocity), 'S0' (orthogonal), 'SP' ('num-dofs=Nr−2' truncated
- **A0373 [VERIFIED]:** survey line 1301; `taylor_couette_dns_jax.py:206-208` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=_lap; claim: - Cylindrical Laplacian ('-lap', 'taylor-couette-dns-jax.py:206-208', verified):
- **A0374 [VERIFIED]:** survey line 1307; `taylor_couette_dns_jax.py:194-196` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=jsp_linalg,lu_factor,vmap; claim: ('jax.vmap(jsp-linalg.lu-factor)', 'taylor-couette-dns-jax.py:194-196') and solved with 'lu-solve' — a per-mode dense direct solve, not
- **A0375 [VERIFIED]:** survey line 1309; `taylor_couette_dns_jax.py:271-348` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=CNAB2,_build_operators; claim: - Time integrator: CNAB2 ('-build-operators', 'taylor-couette-dns-jax.py:271-348'). L-(text(imp))=M/dt
- **A0376 [VERIFIED]:** survey line 1313; `taylor_couette_dns_jax.py:443-465` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=Nonlinear,advection,cylindrical; claim: - Nonlinear advection (cylindrical, written out, 'taylor-couette-dns-jax.py:443-465', verified):
- **A0377 [VERIFIED]:** survey line 1318; `taylor_couette_dns_jax.py:378-394` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=divergence,perturbation,streamfunction;numbers=2; claim: - IC seed ('taylor-couette-dns-jax.py:378-394'): divergence-free streamfunction perturbation g=sin2(pi(r-R-1)/d), or a linear
- **A0378 [VERIFIED]:** survey line 1325; `taylor_couette_dns_jax.py:841-847` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=field,pressure,system,theta,total;numbers=7; claim: A 7-field total-pressure saddle system (u-r,u-theta,u-z,Pi,b-r,b-theta,b-z) ('taylor-couette-dns-jax.py:841-847').
- **A0379 [VERIFIED]:** survey line 1329; `:886-903` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; numbers=886; claim: - Conducting-wall magnetic BCs (baked into composite bases, ':886-903', verified): b-theta uses Robin
- **A0380 [VERIFIED]:** survey line 1333; `:945-984` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=_add_mhd_terms;numbers=945; claim: - Linearized MHD coupling ('-add-mhd-terms', ':945-984', verified from source):
- **A0381 [VERIFIED]:** survey line 1338; `:1089-1125` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=Nonlinear;numbers=1089; claim: - Nonlinear (':1089-1125'): advection minus Lorentz (mathbf ucdotnabla)mathbf u-(mathbf bcdotnabla)mathbf b
- **A0382 [VERIFIED]:** survey line 1350; `:198-259` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; tokens=assemble_parts;numbers=198; claim: ('assemble-parts', ':198-259'), with the explicit cylindrical coupling
- **A0383 [VERIFIED]:** survey line 1361; `base.py:43` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; tokens=BaseIntegrator,Module; claim: 'BaseIntegrator' ('base.py:43', an 'nnx.Module') splits a weak form into mass (time derivative),
- **A0384 [VERIFIED]:** survey line 1363; `base.py:222` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; tokens=build_implicit_operator; claim: M-gamma,dt,L once via 'build-implicit-operator(γ,dt)' ('base.py:222'), keeps the mass inverse out of the
- **A0385 [VERIFIED]:** survey line 1372; `:21-93` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=PDEIMEXRK;numbers=21; claim: All coefficients verified verbatim from 'imex-rk.py'. 'PDEIMEXRK' (':21-93') uses one DIRK diagonal gamma and
- **A0386 [VERIFIED]:** survey line 1373; `:73-93` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=stage,step;numbers=73,0; claim: the per-stage update ('step', ':73-93'): with u0-(text(rhs))=Mhat u,
- **A0387 [VERIFIED]:** survey line 1382; `:184-190` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; numbers=5,5,184,5,5; claim: / IMEXRK443 / 5×5 (':184-190') / 5×5 (':191-197') / (0,tfrac12,tfrac23,tfrac12,1) /
- **A0388 [VERIFIED]:** survey line 1382; `:191-197` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; numbers=5,5,5,5,191; claim: / IMEXRK443 / 5×5 (':184-190') / 5×5 (':191-197') / (0,tfrac12,tfrac23,tfrac12,1) /
- **A0389 [VERIFIED]:** survey line 1384; `:176-177` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=delta,gamma;numbers=2,2,1,2,176; claim: with 'IMEXRK222': gamma=(2-sqrt2)/2, delta=1-tfrac1(2gamma) (':176-177') — the PCF/KMM default
- **A0390 [PARTIAL]:** survey line 1385; `channelflow_kmm.py:75` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; resolved-line-exists-but-claim-numbers-not-found; claim: ('channelflow-kmm.py:75'). The 443 matrices (verified ':184-198'):
- **A0391 [VERIFIED]:** survey line 1385; `:184-198` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; numbers=184; claim: ('channelflow-kmm.py:75'). The 443 matrices (verified ':184-198'):
- **A0392 [VERIFIED]:** survey line 1389; `imex_rk.py:96-160` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=IMEXRK3,Spalart,storage; claim: ### I.C.3.2 Spalart low-storage IMEXRK3 ('imex-rk.py:96-160') — exact coefficients
- **A0393 [VERIFIED]:** survey line 1391; `:103-105` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; numbers=1,103; claim: Third-order, one implicit operator per stage [SMR91]. Coefficients (verified ':103-105'):
- **A0394 [VERIFIED]:** survey line 1395; `:144-160` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=step;numbers=144; claim: ('step', ':144-160', and the KMM '-step-imexrk3', 'channelflow-kmm.py:430-469'):
- **A0395 [VERIFIED]:** survey line 1395; `channelflow_kmm.py:430-469` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=_step_imexrk3,step; claim: ('step', ':144-160', and the KMM '-step-imexrk3', 'channelflow-kmm.py:430-469'):
- **A0396 [VERIFIED]:** survey line 1402; `:33-45` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/cnab2.py`; tokens=first,scan;numbers=1.5,33; claim: =1.5,text(curr)-0.5,text(prev) after the first step (':33-45'); 'scan-steps' wraps the loop in 'jax.lax.scan'
- **A0397 [VERIFIED]:** survey line 1404; `backward_euler.py:11-40` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/backward_euler.py`; tokens=BackwardEuler,implicit,order;numbers=1; claim: Remaining menu: 'BackwardEuler' (1st-order implicit, solves (M-dt,L)u=Mun+dt,N, 'backward-euler.py:11-40'),
- **A0398 [VERIFIED]:** survey line 1405; `rk4.py:10-20` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/rk4.py`; numbers=5,2; claim: 'RK4' ('rk4.py:10-20'), 'ETDRK4' (exponential time-differencing [KT05; CM02], 'etdrk4.py'). These are exercised by
- **A0399 [VERIFIED]:** survey line 1419; `__init__.py:8` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; numbers=4; claim: - Double precision: float64 by default and unconditional ('--init--.py:8'; §I.C.1.1).
- **A0400 [PARTIAL]:** survey line 1421; `imex_rk.py:73,144` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; resolved-line-exists-but-semantic-support-not-proven; claim: ('imex-rk.py:73,144', 'rk4.py:13', 'base.py:262'). Solver state objects are JAX pytrees ('KMMState',
- **A0401 [PARTIAL]:** survey line 1421; `rk4.py:13` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/rk4.py`; resolved-line-exists-but-semantic-support-not-proven; claim: ('imex-rk.py:73,144', 'rk4.py:13', 'base.py:262'). Solver state objects are JAX pytrees ('KMMState',
- **A0402 [PARTIAL]:** survey line 1421; `base.py:262` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; resolved-line-exists-but-semantic-support-not-proven; claim: ('imex-rk.py:73,144', 'rk4.py:13', 'base.py:262'). Solver state objects are JAX pytrees ('KMMState',
- **A0403 [VERIFIED]:** survey line 1429; `sharding.py:9-21` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=Sharding,device; claim: - Sharding (multi-device CPU/GPU/TPU) — 'sharding.py:9-21' (verified):
- **A0404 [VERIFIED]:** survey line 1435; `:24-40` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=_build_local_apply_fn,along,shard,vmap;numbers=24,1; claim: '-build-local-apply-fn' (':24-40') is 'jax.jit(jax.vmap(1D-transform))' along one axis on a local shard;
- **A0405 [VERIFIED]:** survey line 1436; `:43-105` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=_apply_separable_spmd_shard_map,device,single,transform;numbers=43; claim: '-apply-separable-spmd-shard-map' (':43-105') is the production multi-device transform — a single fused
- **A0406 [VERIFIED]:** survey line 1456; `test_pcf_fluctuations_jax.py:78-95` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_pcf_fluctuations_jax.py`; tokens=diagnostics,fluctuation;numbers=9,8,8,10,-3; claim: PCF fluctuation diagnostics ('test-pcf-fluctuations-jax.py:78-95'; N=(9,8,8), Legendre, dt=10(-3),
- **A0407 [VERIFIED]:** survey line 1470; `test_pcf_mhd_mri_shearpy_jax.py:6-23` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_pcf_mhd_mri_shearpy_jax.py`; tokens=divB_L2,divL2,finite,shearpy; claim: PCF MRI shearpy ('test-pcf-mhd-mri-shearpy-jax.py:6-23'): 'divL2<1e-4', 'divB-L2<1e-5', finite
- **A0408 [PARTIAL]:** survey line 1473; `test_taylor_couette_linear_jax.py:16-26` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_taylor_couette_linear_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: TC linear leading spectrum ('test-taylor-couette-linear-jax.py:16-26'; 'CircularCouette()' default
- **A0409 [PARTIAL]:** survey line 1565; `ChannelFlow.py:147-164` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/ChannelFlow.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / PCF / channel hydro / P — KMM velocity–vorticity ('ChannelFlow.py:147-164') / P — θ-method predictor/corrector ('torchchannel/solver.py:594-606') / P — KMM, JAX-native ('channelflow-kmm.py:382-516') /
- **A0410 [VERIFIED]:** survey line 1565; `torchchannel/solver.py:594-606` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=channel,corrector,predictor; claim: / PCF / channel hydro / P — KMM velocity–vorticity ('ChannelFlow.py:147-164') / P — θ-method predictor/corrector ('torchchannel/solver.py:594-606') / P — KMM, JAX-native ('channelflow-kmm.py:382-516') /
- **A0411 [VERIFIED]:** survey line 1565; `channelflow_kmm.py:382-516` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/channelflow_kmm.py`; tokens=channel,velocity; claim: / PCF / channel hydro / P — KMM velocity–vorticity ('ChannelFlow.py:147-164') / P — θ-method predictor/corrector ('torchchannel/solver.py:594-606') / P — KMM, JAX-native ('channelflow-kmm.py:382-516') /
- **A0412 [VERIFIED]:** survey line 1566; `taylor_couette_dns.py:288-313` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/taylor_couette_dns.py`; tokens=coupled; claim: / TC hydro / P — coupled saddle-point CNAB2 ('taylor-couette-dns.py:288-313') / P — influence-matrix PC ('torchcouette/solver.py:553-594') / P — pinned saddle, per-mode LU ('taylor-couette-dns-jax.py:271…
- **A0413 [PARTIAL]:** survey line 1566; `torchcouette/solver.py:553-594` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / TC hydro / P — coupled saddle-point CNAB2 ('taylor-couette-dns.py:288-313') / P — influence-matrix PC ('torchcouette/solver.py:553-594') / P — pinned saddle, per-mode LU ('taylor-couette-dns-jax.py:271…
- **A0414 [VERIFIED]:** survey line 1566; `taylor_couette_dns_jax.py:271-364` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=CNAB2; claim: / TC hydro / P — coupled saddle-point CNAB2 ('taylor-couette-dns.py:288-313') / P — influence-matrix PC ('torchcouette/solver.py:553-594') / P — pinned saddle, per-mode LU ('taylor-couette-dns-jax.py:271…
- **A0415 [VERIFIED]:** survey line 1567; `pipe_flow_dns.py:289-321` -> `fn_shenfun/demo/pipe_flow_dns.py`; tokens=CNAB2;numbers=1; claim: / Pipe hydro / P — curvilinear √g=r, CNAB2 ('pipe-flow-dns.py:289-321') / P — banded GBTRF/GBTRS LU ('torchpipeflow/banded.py:109-172') / A — no pipe ('jaxfun-missing-parts.md'; C-SOLVERS §1) /
- **A0416 [VERIFIED]:** survey line 1567; `torchpipeflow/banded.py:109-172` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/banded.py`; tokens=banded;numbers=1; claim: / Pipe hydro / P — curvilinear √g=r, CNAB2 ('pipe-flow-dns.py:289-321') / P — banded GBTRF/GBTRS LU ('torchpipeflow/banded.py:109-172') / A — no pipe ('jaxfun-missing-parts.md'; C-SOLVERS §1) /
- **A0417 [VERIFIED]:** survey line 1568; `pcf_mhd_divfree.py:6-14` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_divfree.py`; tokens=curl; claim: / PCF MHD / P — 'B=curl(A)', div-free by construction ('pcf-mhd-divfree.py:6-14') / P — induced 'b', full induction ('torchchannel/mhd.py:207-228') / P — 'B=curl(A)' ('pcf-mhd-jax.py:108-181') /
- **A0418 [VERIFIED]:** survey line 1568; `torchchannel/mhd.py:207-228` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=curl,induction; claim: / PCF MHD / P — 'B=curl(A)', div-free by construction ('pcf-mhd-divfree.py:6-14') / P — induced 'b', full induction ('torchchannel/mhd.py:207-228') / P — 'B=curl(A)' ('pcf-mhd-jax.py:108-181') /
- **A0419 [VERIFIED]:** survey line 1568; `pcf_mhd_jax.py:108-181` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; tokens=curl; claim: / PCF MHD / P — 'B=curl(A)', div-free by construction ('pcf-mhd-divfree.py:6-14') / P — induced 'b', full induction ('torchchannel/mhd.py:207-228') / P — 'B=curl(A)' ('pcf-mhd-jax.py:108-181') /
- **A0420 [VERIFIED]:** survey line 1569; `taylor_couette_dns.py:788-951` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/taylor_couette_dns.py`; tokens=axisym,field;numbers=3,7; claim: / TC MHD / P — direct-b conducting, axisym+3D ('taylor-couette-dns.py:788-951') / P — induced b, influence matrix ('torchcouette/mhd.py:179-231') / P — 7-field saddle ('taylor-couette-dns-jax.py:841-984'…
- **A0421 [VERIFIED]:** survey line 1569; `torchcouette/mhd.py:179-231` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=influence,matrix;numbers=3,7; claim: / TC MHD / P — direct-b conducting, axisym+3D ('taylor-couette-dns.py:788-951') / P — induced b, influence matrix ('torchcouette/mhd.py:179-231') / P — 7-field saddle ('taylor-couette-dns-jax.py:841-984'…
- **A0422 [VERIFIED]:** survey line 1569; `taylor_couette_dns_jax.py:841-984` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; tokens=axisym,conducting,field;numbers=3,7; claim: / TC MHD / P — direct-b conducting, axisym+3D ('taylor-couette-dns.py:788-951') / P — induced b, influence matrix ('torchcouette/mhd.py:179-231') / P — 7-field saddle ('taylor-couette-dns-jax.py:841-984'…
- **A0423 [PARTIAL]:** survey line 1571; `pcf_mhd_mri_shearpy.py:12-15` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / MRI rotation+shear / P — Coriolis + base-shear + shear-induction ('pcf-mhd-mri-shearpy.py:12-15') / S — metadata only, no source terms ('torchchannel/mhd.py:71-74') / P — Coriolis + shear-induction…
- **A0424 [VERIFIED]:** survey line 1571; `torchchannel/mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=Coriolis,metadata,shear,terms; claim: / MRI rotation+shear / P — Coriolis + base-shear + shear-induction ('pcf-mhd-mri-shearpy.py:12-15') / S — metadata only, no source terms ('torchchannel/mhd.py:71-74') / P — Coriolis + shear-induction…
- **A0425 [PARTIAL]:** survey line 1571; `pcf_mhd_mri_shearpy_jax.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / MRI rotation+shear / P — Coriolis + base-shear + shear-induction ('pcf-mhd-mri-shearpy.py:12-15') / S — metadata only, no source terms ('torchchannel/mhd.py:71-74') / P — Coriolis + shear-induction…
- **A0426 [VERIFIED]:** survey line 1572; `taylor_couette_mri.py:42-47` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/taylor_couette_mri.py`; tokens=Insulating,_assemble_flux_parts,vacuum,walls;numbers=0,0,0; claim: / Insulating / vacuum walls / P — TC linear, m=0, flux-fn ('taylor-couette-mri.py:42-47') / A — homogeneous 'b=0' only ('torchchannel/mhd.py:278-282'; B-MHD §3.1) / P — TC linear, m=0, flux-fn ('taylor-c…
- **A0427 [VERIFIED]:** survey line 1572; `torchchannel/mhd.py:278-282` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=0,0,0; claim: / Insulating / vacuum walls / P — TC linear, m=0, flux-fn ('taylor-couette-mri.py:42-47') / A — homogeneous 'b=0' only ('torchchannel/mhd.py:278-282'; B-MHD §3.1) / P — TC linear, m=0, flux-fn ('taylor-c…
- **A0428 [VERIFIED]:** survey line 1573; `taylor_couette_mri.py:36-40` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/taylor_couette_mri.py`; tokens=Conducting,Neumann,Robin,walls;numbers=0; claim: / Conducting walls / P — Robin 'c=r-wall/J' ('taylor-couette-mri.py:36-40') / P — homogeneous b=0 + influence matrix ('torchcouette/mhd.py:266-276') / P — Robin/Neumann ('taylor-couette-dns-jax.py:886-90…
- **A0429 [VERIFIED]:** survey line 1573; `torchcouette/mhd.py:266-276` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=walls;numbers=0; claim: / Conducting walls / P — Robin 'c=r-wall/J' ('taylor-couette-mri.py:36-40') / P — homogeneous b=0 + influence matrix ('torchcouette/mhd.py:266-276') / P — Robin/Neumann ('taylor-couette-dns-jax.py:886-90…
- **A0430 [VERIFIED]:** survey line 1573; `taylor_couette_dns_jax.py:886-903` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; numbers=0; claim: / Conducting walls / P — Robin 'c=r-wall/J' ('taylor-couette-mri.py:36-40') / P — homogeneous b=0 + influence matrix ('torchcouette/mhd.py:266-276') / P — Robin/Neumann ('taylor-couette-dns-jax.py:886-90…
- **A0431 [VERIFIED]:** survey line 1574; `solver.py:87` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; numbers=1; claim: / Time integrators / P — IMEXRK111/222/3/443, CNAB2 ('integrators.py') / S — θ-method PC only, formally 1st-order ('solver.py:87') / P — IMEXRK222/3, CNAB2, RK4/ETDRK4 ('integrators/') /
- **A0432 [PARTIAL]:** survey line 1575; `pcf_mhd_divfree_notes.md:69` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_divfree_notes.md`; resolved-line-exists-but-claim-numbers-not-found; claim: / Div-free method / P — KMM/saddle-point, div≈1e-16 ('pcf-mhd-divfree-notes.md:69') / P — influence-matrix + pinv, div≈1e-7 ('solver.py:474-482') / P — KMM/pinned saddle, div≈1e-17 ('test-taylor-couette-…
- **A0433 [PARTIAL]:** survey line 1575; `solver.py:474-482` -> `fn_openpipeflow-122/torchcouette/torchcouette/solver.py`; resolved-line-exists-but-claim-numbers-not-found; claim: / Div-free method / P — KMM/saddle-point, div≈1e-16 ('pcf-mhd-divfree-notes.md:69') / P — influence-matrix + pinv, div≈1e-7 ('solver.py:474-482') / P — KMM/pinned saddle, div≈1e-17 ('test-taylor-couette-…
- **A0434 [PARTIAL]:** survey line 1577; `sharding.py:43-105` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / TPU / A / A / P — 'shard-map', device-agnostic ('sharding.py:43-105') /
- **A0435 [PARTIAL]:** survey line 1578; `test_mhd.py:183-201` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Autograd / A — NumPy/SciPy (A-PCF §10) / P — full, incl. Lorentz coupling ('test-mhd.py:183-201') / P — 'value-and-grad' minimal-seed ('pcf-minimal-seed-jax.py') /
- **A0436 [PARTIAL]:** survey line 1579; `base.py:262` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; resolved-line-exists-but-claim-numbers-not-found; claim: / JIT / A — numba kernels only / A — no 'torch.compile'/'jit' (B-MHD §6.4) / P — 'jax.jit'/'nnx.jit' pervasive ('base.py:262') /
- **A0437 [VERIFIED]:** survey line 1580; `solver.py:87` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex128;numbers=28,2; claim: / Double precision / P — float64 default / P — complex128 default; float32 validated ('solver.py:87') / P — x64-by-default ('--init--.py:8') /
- **A0438 [PARTIAL]:** survey line 1580; `__init__.py:8` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Double precision / P — float64 default / P — complex128 default; float32 validated ('solver.py:87') / P — x64-by-default ('--init--.py:8') /
- **A0439 [VERIFIED]:** survey line 1581; `_linear_analysis.py:16` -> `fn_shenfun/demo/_linear_analysis.py`; tokens=FINITE_CAP; claim: / Linear eigensolver / P — 'scipy.linalg.eig', 'FINITE-CAP=1e8' ('-linear-analysis.py:16') / P — 'scipy.linalg.eig' OS/Squire ('linstab.py:109') / P — 'generalized-eig', dense NumPy ('la/eig.py:168') /
- **A0440 [VERIFIED]:** survey line 1581; `linstab.py:109` -> `fn_openpipeflow-122/torchchannel/torchchannel/linstab.py`; tokens=linalg,scipy; claim: / Linear eigensolver / P — 'scipy.linalg.eig', 'FINITE-CAP=1e8' ('-linear-analysis.py:16') / P — 'scipy.linalg.eig' OS/Squire ('linstab.py:109') / P — 'generalized-eig', dense NumPy ('la/eig.py:168') /
- **A0441 [VERIFIED]:** survey line 1581; `la/eig.py:168` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/la/eig.py`; tokens=generalized_eig; claim: / Linear eigensolver / P — 'scipy.linalg.eig', 'FINITE-CAP=1e8' ('-linear-analysis.py:16') / P — 'scipy.linalg.eig' OS/Squire ('linstab.py:109') / P — 'generalized-eig', dense NumPy ('la/eig.py:168') /
- **A0442 [VERIFIED]:** survey line 1582; `OrrSommerfeld_eigs.py:183` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; numbers=8; claim: / Convergence-order tests / P — golden eig 1e-12, MMS ('OrrSommerfeld-eigs.py:183') / P — OS golden 1e-4, mesh poly-exact deg 8 ('test-mesh.py:21-37') / P — MMS self-asserts ('poisson1D.py:46') /
- **A0443 [PARTIAL]:** survey line 1582; `test_mesh.py:21-37` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Convergence-order tests / P — golden eig 1e-12, MMS ('OrrSommerfeld-eigs.py:183') / P — OS golden 1e-4, mesh poly-exact deg 8 ('test-mesh.py:21-37') / P — MMS self-asserts ('poisson1D.py:46') /
- **A0444 [PARTIAL]:** survey line 1582; `poisson1D.py:46` -> `fn_shenfun/demo/poisson1D.py`; resolved-line-exists-but-claim-numbers-not-found; claim: / Convergence-order tests / P — golden eig 1e-12, MMS ('OrrSommerfeld-eigs.py:183') / P — OS golden 1e-4, mesh poly-exact deg 8 ('test-mesh.py:21-37') / P — MMS self-asserts ('poisson1D.py:46') /
- **A0445 [VERIFIED]:** survey line 1592; `torchchannel/mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=omega,shear_rate; claim: ('torchchannel/mhd.py:71-74'). Verified: 'omega'/'shear-rate' are referenced nowhere
- **A0446 [VERIFIED]:** survey line 1594; `torchchannel/mhd.py:104,452` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=torch; claim: ('torchchannel/mhd.py:104,452'). The Taylor–Couette torch MHD class carries no MRI
- **A0447 [PARTIAL]:** survey line 1598; `pcf_mhd_mri_shearpy.py:12-15` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; resolved-line-exists-but-semantic-support-not-proven; claim: nonlinear/EMF path ('pcf-mhd-mri-shearpy.py:12-15',
- **A0448 [PARTIAL]:** survey line 1599; `pcf_mhd_mri_shearpy_jax.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'pcf-mhd-mri-shearpy-jax.py:130-134'). The only shear-like coupling present in torch
- **A0449 [PARTIAL]:** survey line 1601; `torchchannel/solver.py:534-540` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; resolved-line-exists-but-semantic-support-not-proven; claim: ('torchchannel/solver.py:534-540'); this is plain advection by the laminar profile, not
- **A0450 [VERIFIED]:** survey line 1609; `taylor_couette_mri.py:42-47,332-444` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/taylor_couette_mri.py`; tokens=_assemble_flux_parts; claim: ('taylor-couette-mri.py:42-47,332-444'; 'taylor-couette-mri-jax.py' '-assemble-flux-parts').
- **A0451 [PARTIAL]:** survey line 1611; `torchchannel/mhd.py:278-282` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; resolved-line-exists-but-semantic-support-not-proven; claim: Dirichlet on all three induced components ('torchchannel/mhd.py:278-282';
- **A0452 [PARTIAL]:** survey line 1612; `torchcouette/mhd.py:266-276` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'torchcouette/mhd.py:266-276' with the explicit docstring note that exact insulating
- **A0453 [PARTIAL]:** survey line 1618; `couette_linear_benchmarks.md:34-35,352-353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; resolved-line-exists-but-semantic-support-not-proven; claim: 'couette-linear-benchmarks.md:34-35,352-353':
- **A0454 [PARTIAL]:** survey line 1628; `mhd_parity_plan.md:35` -> `fn_shenfun/demo/mhd_parity_plan.md`; resolved-line-exists-but-claim-numbers-not-found; claim: conducting-only in all three families ('mhd-parity-plan.md:35'), so SR-6/SR-7/SR-9 are
- **A0455 [PARTIAL]:** survey line 1642; `test_mhd.py:183-201` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Lorentz coupling, 'test-mhd.py:183-201') and is device-agnostic, but has no
- **A0456 [PARTIAL]:** survey line 1646; `sharding.py:43-105` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: default, and is the only family with TPU support via 'shard-map' ('sharding.py:43-105') —
- **A0457 [VERIFIED]:** survey line 1690; `torchchannel/solver.py:534-540` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=coupling; claim: ('torchchannel/solver.py:534-540'), consumed where base coupling already enters
- **A0458 [VERIFIED]:** survey line 1691; `torchchannel/solver.py:571-577` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=_rhs_for_state; claim: '-rhs-for-state' ('torchchannel/solver.py:571-577'); add the analogous frame-rotation
- **A0459 [PARTIAL]:** survey line 1695; `pcf_mhd_mri_notes.md:37-42` -> `fn_shenfun/demo/pcf_mhd_mri_notes.md`; resolved-line-exists-but-claim-numbers-not-found; claim: ('PLAN…:125-137'). Use the shenfun-verified term set ('pcf-mhd-mri-notes.md:37-42', also
- **A0460 [VERIFIED]:** survey line 1696; `pcf_mhd_mri_shearpy_jax.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=Omega,omega; claim: the C mirror 'pcf-mhd-mri-shearpy-jax.py:130-134'), with Omega='omega', S='shear-rate':
- **A0461 [VERIFIED]:** survey line 1711; `:71-104` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=71; claim: 'fn-openpipeflow-122/torchchannel/torchchannel/mhd.py' (stub ':71-104', induction RHS);
- **A0462 [PARTIAL]:** survey line 1724; `tests/test_theory.py:100-132` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: 'tests/test-theory.py:100-132'. Then SR-2 isolates the shear-induction term via the
- **A0463 [PARTIAL]:** survey line 1734; `test_theory.py:66-97` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: ('test-theory.py:66-97'), and SR-4 Alfvén phase omega=kcdot v-A
- **A0464 [PARTIAL]:** survey line 1735; `test_theory.py:135-178` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: ('test-theory.py:135-178'). Gate: rate / frequency match to 'rel<1e-2' with an explicit
- **A0465 [PARTIAL]:** survey line 1737; `torchchannel/mhd.py:101` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; resolved-line-exists-but-claim-numbers-not-found; claim: 'torchchannel/mhd.py:101'; couette is mathrm(Ha)2/mathrm(Pm), 'torchcouette/mhd.py:79' —
- **A0466 [PARTIAL]:** survey line 1737; `torchcouette/mhd.py:79` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; resolved-line-exists-but-claim-numbers-not-found; claim: 'torchchannel/mhd.py:101'; couette is mathrm(Ha)2/mathrm(Pm), 'torchcouette/mhd.py:79' —
- **A0467 [VERIFIED]:** survey line 1747; `taylor_couette_mri.py:142-153,366-371` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Robin,shenfun; claim: 'taylor-couette-mri.py:142-153,366-371' (the shenfun Robin coefficient
- **A0468 [VERIFIED]:** survey line 1754; `_pcf_linear.py:115-116` -> `fn_shenfun/demo/_pcf_linear.py`; tokens=conducting,dirichlet; claim: 'conducting' + diagnostic 'dirichlet' ('-pcf-linear.py:115-116'). Add the
- **A0469 [VERIFIED]:** survey line 1756; `mhd_parity_plan.md:103-120` -> `fn_shenfun/demo/mhd_parity_plan.md`; tokens=keeping,magnetic,pressure; claim: BC for PCF (WS-D, 'mhd-parity-plan.md:103-120'), keeping the varphi magnetic-pressure
- **A0470 [PARTIAL]:** survey line 1761; `test_taylor_couette.py:210-223` -> `fn_shenfun/demo/test_taylor_couette.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'test-taylor-couette.py:210-223' precedent). Insulating nonlinear DNS is net-new for
- **A0471 [PARTIAL]:** survey line 1762; `mhd_parity_plan.md:194-198` -> `fn_shenfun/demo/mhd_parity_plan.md`; resolved-line-exists-but-semantic-support-not-proven; claim: all three families (WS-J, 'mhd-parity-plan.md:194-198') and is deferred to a stretch
- **A0472 [VERIFIED]:** survey line 1769; `test_pcf_mhd_mri_shearpy.py:120-133` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/test_pcf_mhd_mri_shearpy.py`; tokens=growth; claim: ('test-pcf-mhd-mri-shearpy.py:120-133'), not a growth-rate match (WS-A,
- **A0473 [VERIFIED]:** survey line 1770; `mhd_parity_plan.md:44-64` -> `fn_shenfun/demo/mhd_parity_plan.md`; tokens=builds,seed_linear_eigenmode; claim: 'mhd-parity-plan.md:44-64'). Add a 'seed-linear-eigenmode(ky,kz,amp)' that builds the
- **A0474 [PARTIAL]:** survey line 1776; `taylor_couette_dns_jax.py:396-428` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_dns_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: 'taylor-couette-dns-jax.py:396-428', and differentiable diagnostics). Also add to A the
- **A0475 [VERIFIED]:** survey line 1781; `taylor_couette_notes.md:401-406` -> `fn_shenfun/demo/taylor_couette_notes.md`; numbers=4e-7,2e-6; claim: ('taylor-couette-notes.md:401-406'; per-m rel-errs 4e-8/4e-7/2e-6 reused as the WS-A
- **A0476 [PARTIAL]:** survey line 1782; `mhd_parity_plan.md:46` -> `fn_shenfun/demo/mhd_parity_plan.md`; resolved-line-exists-but-semantic-support-not-proven; claim: target, 'mhd-parity-plan.md:46'); 'div(B)', 'div(u)' at roundoff throughout. Tighten to
- **A0477 [PARTIAL]:** survey line 1784; `mhd_parity_plan.md:154-169` -> `fn_shenfun/demo/mhd_parity_plan.md`; resolved-line-exists-but-semantic-support-not-proven; claim: 'mhd-parity-plan.md:154-169').
- **A0478 [VERIFIED]:** survey line 1795; `base_flow.py:37-41` -> `fn_openpipeflow-122/torchchannel/torchchannel/base_flow.py`; tokens=normal; claim: velocity in component 'x' as a function of wall-normal 'y', 'base-flow.py:37-41', opposite
- **A0479 [PARTIAL]:** survey line 1828; `pipe_flow_notes.md:67` -> `fn_shenfun/demo/pipe_flow_notes.md`; resolved-line-exists-but-claim-numbers-not-found; claim: 'pipe-flow-notes.md:67'); Womersley oscillatory solution to '<5e-6'
- **A0480 [VERIFIED]:** survey line 1829; `pipe_flow_notes.md:69` -> `fn_shenfun/demo/pipe_flow_notes.md`; tokens=exact;numbers=8,-7; claim: (A: max/u-z-mathrm(exact)/=8times10(-7), 'pipe-flow-notes.md:69'). Note F1e
- **A0481 [PARTIAL]:** survey line 1840; `sharding.py:43-105` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: transforms ('sharding.py:43-105') under multi-device, gated behind '--num-devices=2 -m spmd'
- **A0482 [PARTIAL]:** survey line 1843; `test_mhd.py:183-201` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: ('test-mhd.py:183-201'); C's via 'value-and-grad' minimal-seed adjoint
- **A0483 [PARTIAL]:** survey line 1872; `mhd_parity_plan.md:194-198` -> `fn_shenfun/demo/mhd_parity_plan.md`; resolved-line-exists-but-claim-numbers-not-found; claim: Explicitly deferred (documented non-goals, 'PLAN…:264', 'mhd-parity-plan.md:194-198'):
- **A0484 [VERIFIED]:** survey line 1900; `PLAN_openpipeflow_vs_fnshenfun.md:57-61` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Family,Tolerances,_spectral,actual,comparison; claim: The cross-family band derivation is fixed by 'PLAN-openpipeflow-vs-fnshenfun.md:57-61': "Tolerances for any cross-family comparison are derived as 'max(C·Δx-FD⁴, C·Δt-FD², ε-spectral)' from the actual grid — never roun…
- **A0485 [VERIFIED]:** survey line 1912; `shearpy/tests/fd_proto/test_rdt.py:12-26` -> `shearpy-jimenez/shearpy/tests/fd_proto/test_rdt.py`; tokens=coarse,coarse_t,coarse_x,fine_t,fine_x;numbers=1,2,4,2; claim: Report both the least-squares slope over the full ladder and the successive pairwise orders 'p-i = log(E-i/E-(i+1))/log 2'; a non-monotone pairwise sequence flags either the roundoff floor (drop coarse points) or a…
- **A0486 [VERIFIED]:** survey line 1918; `base_flow.py:37-41` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; numbers=4,9,4,8; claim: Notation: 'x' = wall-normal/radial (Dirichlet walls), 'y' = streamwise/azimuthal, 'z' = spanwise/axial (rotation axis), per the §0.2 canonical frame. 'Δx' = wall-normal/radial grid spacing; 'p' = spatial order (spectral…
- **A0487 [VERIFIED]:** survey line 1927; `pcf_fluctuations_corrected.py:130-135` -> `fn_shenfun/demo/pcf_fluctuations_corrected.py`; tokens=U_wall,dU_b,streamwise;numbers=1; claim: - Family A: 'U-b = +U-wall·x·e-y', 'dU-b/dx = U-wall' ('pcf-fluctuations-corrected.py:130-135'); streamwise = axis 1 ('y').
- **A0488 [VERIFIED]:** survey line 1928; `base_flow.py:37-41` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; numbers=1; claim: - Family B: 'U=y.clone(), Up=ones, Upp=zeros, walls=(-1,1)' ('base-flow.py:37-41'); streamwise = 'x' (swapped — apply adapter).
- **A0489 [VERIFIED]:** survey line 1929; `pcf_fluctuations_jax.py:65-67` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_fluctuations_jax.py`; tokens=U_wall; claim: - Family C: 'U-b = +U-wall·x·e-y', 'dU-b/dx = U-wall' ('pcf-fluctuations-jax.py:65-67').
- **A0490 [VERIFIED]:** survey line 1931; `couette_linear_benchmarks.md:84-88` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Romanov;numbers=1,0,1,0,1973; claim: Oracle. 'U(x)=x' exactly ('U'=1', 'U''=0'); wall values '±1'. Plane Couette is the canonical linearly-stable base flow for all 'Re' [Nagata90]; Romanov (1973) ('couette-linear-benchmarks.md:84-88, 440').
- **A0491 [VERIFIED]:** survey line 1935; `test_pcf_fluctuations_jax.py:48-53` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_pcf_fluctuations_jax.py`; numbers=0,0; claim: - Fixed-point check: 'E-pert < 1e-20' after one step from 'u'=0' (C reports 'E-pert=0' exactly, 'test-pcf-fluctuations-jax.py:48-53'; B 'perturbation-energy < 1e-20' after 2000 steps, 'test-step-decay.py:29-51').
- **A0492 [VERIFIED]:** survey line 1935; `test_step_decay.py:29-51` -> `fn_openpipeflow-122/torchchannel/tests/test_step_decay.py`; tokens=perturbation_energy,steps;numbers=1e-20,0,0,1e-20,2000; claim: - Fixed-point check: 'E-pert < 1e-20' after one step from 'u'=0' (C reports 'E-pert=0' exactly, 'test-pcf-fluctuations-jax.py:48-53'; B 'perturbation-energy < 1e-20' after 2000 steps, 'test-step-decay.py:29-51').
- **A0493 [VERIFIED]:** survey line 1936; `test_pcf_mhd_mri_shearpy.py:60` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=mean_shear,shear;numbers=1.0000000004699001,1,1.0,1e-10; claim: - mean-shear cross-check: 'mean-shear ≈ σ'. C golden 'mean-shear = 1.0000000004699001' after one step; within-family 'rel<1e-8', cross-family 'rel<1e-2'. Sign note: in the MRI/shearpy convention 'σ = −S', so the s…
- **A0494 [VERIFIED]:** survey line 1945; `OrrSommerfeld.py:14,30` -> `fn_shenfun/demo/OrrSommerfeld.py`; tokens=dpdy;numbers=1,2; claim: - Family A: 'U-b=(1−x²)e-y', 'dpdy=−2/Re' ('OrrSommerfeld.py:14,30').
- **A0495 [PARTIAL]:** survey line 1946; `base_flow.py:42-46` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: - Family B: 'U=1−y², Up=−2y, Upp=−2, walls=(0,0)', const-flux target '4/3' ('base-flow.py:42-46'; '-flux-target' 'solver.py:621-626').
- **A0496 [PARTIAL]:** survey line 1946; `solver.py:621-626` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: - Family B: 'U=1−y², Up=−2y, Upp=−2, walls=(0,0)', const-flux target '4/3' ('base-flow.py:42-46'; '-flux-target' 'solver.py:621-626').
- **A0497 [PARTIAL]:** survey line 1952; `test_mesh.py:21-37` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: - Profile: 'max-j/U(x-j) − (1−x-j²)/ < 1e-12' for A/C (degree-2 ∈ trial space); '< 1e-7' for B (9-point FD polynomial-exact through degree 8, 'test-mesh.py:21-37').
- **A0498 [VERIFIED]:** survey line 1953; `test_step_decay.py:108-126` -> `fn_openpipeflow-122/torchchannel/tests/test_step_decay.py`; tokens=flux;numbers=4,3,1e-12; claim: - Constant-flux oracle (B): '/flux − 4/3/ < 1e-12' ('test-step-decay.py:108-126').
- **A0499 [VERIFIED]:** survey line 1963; `taylor_couette_linear.py:89-91` -> `fn_shenfun/demo/taylor_couette_linear.py`; numbers=2,1,1,2; claim: - Family A: 'V(r)=ar+b/r', 'a=(Ω2 R2²−Ω1 R1²)/(R2²−R1²)', 'b=(Ω1−Ω2)R1²R2²/(R2²−R1²)' ('taylor-couette-linear.py:89-91').
- **A0500 [VERIFIED]:** survey line 1964; `base_flow.py:18-30` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; tokens=Re_i,Re_o;numbers=1,1,1; claim: - Family B: 'u-θ=ar+b/r', 'a=(Re-o−η Re-i)/(1+η)', 'b=η(Re-i−η Re-o)/((1−η)(1−η²))' ('base-flow.py:18-30').
- **A0501 [PARTIAL]:** survey line 1965; `taylor_couette_linear_jax.py:37-89` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_linear_jax.py`; resolved-line-exists-but-semantic-support-not-proven; claim: - Family C: same as A ('taylor-couette-linear-jax.py:37-89').
- **A0502 [VERIFIED]:** survey line 1967; `couette_linear_benchmarks.md:29,227` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Re_c,hydro,kz_c,onset;numbers=1,3,4,3,3; claim: Oracle. For the canonical A case 'a=−1/3', 'b=4/3' ⇒ 'V(r)=−r/3 + 4/(3r)'; check 'V(1)=1=Ω1 R1', 'V(2)=0=Ω2 R2'. The hydro onset for this case is 'Re-c=68.18635', 'kz-c=3.1667' ('couette-linear-benchmarks.md:29,227'…
- **A0503 [VERIFIED]:** survey line 1982; `pipe_flow_dns.py:473-475` -> `fn_shenfun/demo/pipe_flow_dns.py`; numbers=4,0; claim: - Family A: 'u-z(r)=(f-z/(4ν))(R²−r²)', 'Q=πR⁴f-z/(8ν)' ('pipe-flow-dns.py:473-475'); axis via unified 'bc=(None,0)'.
- **A0504 [VERIFIED]:** survey line 1983; `solver.py:200-203` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/solver.py`; tokens=_b_hpf;numbers=1,2,2; claim: - Family B: 'U(r)=1−r²', 'U'=−2r', '-b-hpf=2r' ('solver.py:200-203'); axis via parity folding (negative-radius ghosts).
- **A0505 [VERIFIED]:** survey line 1989; `test_pipe_flow_dns.py:33-37` -> `fn_shenfun/demo/test_pipe_flow_dns.py`; tokens=exact;numbers=1e-6; claim: - Profile: 'max/u-z − exact/ < 1e-6' (A golden, 'test-pipe-flow-dns.py:33-37').
- **A0506 [VERIFIED]:** survey line 1990; `pipe_flow_notes.md:67` -> `fn_shenfun/demo/pipe_flow_notes.md`; numbers=8,1.4e-12; claim: - Flow rate (tight oracle): '/Q − πR⁴f-z/(8ν)//Q < 1e-10' (A; notes report rel.err '1.4e-12', 'pipe-flow-notes.md:67'). B drives mean axial flux to '<1e-8' ('test-invariants.py').
- **A0507 [VERIFIED]:** survey line 2000; `pipe_flow_dns.py:478-490` -> `fn_shenfun/demo/pipe_flow_dns.py`; numbers=1,3,2,3,2; claim: - Family A: 'u-z(r,t)=Re((K/(iρω))[1 − J-0(i(3/2)α-W r/R)/J-0(i(3/2)α-W)] e(iωt))', 'ρ=1', 'i(3/2)=e(i3π/4)' ('pipe-flow-dns.py:478-490').
- **A0508 [VERIFIED]:** survey line 2006; `pipe_flow_notes.md:69` -> `fn_shenfun/demo/pipe_flow_notes.md`; tokens=CNAB2,Womersley,exact,period;numbers=8e-7,6e-6,2,2; claim: Metric & tolerance. 'max/u-z − exact/ < 5e-6' over a full period (A golden; notes report '8e-7', rel '6e-6', 'pipe-flow-notes.md:69'). Temporal error scales '~(ωΔt)²' (CNAB2, 2nd order); halving 'Δt' confirms slope…
- **A0509 [VERIFIED]:** survey line 2023; `pcf_mhd_divfree_notes.md:69-73` -> `fn_shenfun/demo/pcf_mhd_divfree_notes.md`; tokens=divB,divU;numbers=8,8,8,9.41e-17,3.05e-21; claim: / A / PCF MHD (Legendre 'N=(8,8,8)') / 'divU L2' / 'divB L2' / '9.41e-17' / '3.05e-21' / 'pcf-mhd-divfree-notes.md:69-73' /
- **A0510 [VERIFIED]:** survey line 2026; `test_pcf_mhd_mri_shearpy.py:61-62` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=divb_l2,divu_l2;numbers=1e-12; claim: / A / shearpy MRI (Legendre 'N=(8,8,8)') / 'divb-l2', 'divu-l2' / '< 1e-12' (gate) / 'test-pcf-mhd-mri-shearpy.py:61-62' /
- **A0511 [VERIFIED]:** survey line 2029; `test_step_decay.py:29-51` -> `fn_openpipeflow-122/torchchannel/tests/test_step_decay.py`; tokens=Channel,divergence_norm,include_walls;numbers=1e-12; claim: / B / Channel / 'divergence-norm(include-walls)' / '< 1e-12' / 'test-step-decay.py:29-51' /
- **A0512 [VERIFIED]:** survey line 2033; `test_pcf_fluctuations_jax.py:78-95` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_pcf_fluctuations_jax.py`; tokens=divL2; claim: / C / PCF / 'divL2' / '7.18e-17' (gate '<5e-15') / 'test-pcf-fluctuations-jax.py:78-95' /
- **A0513 [VERIFIED]:** survey line 2045; `test_step_decay.py:246-271` -> `fn_openpipeflow-122/torchchannel/tests/test_step_decay.py`; tokens=Assert,Channel,energy,nonlinear,rotational;numbers=50,1e-6,1e-8,1e-6; claim: F3a — Inviscid energy conservation (high-Re limit). Channel, 'Re=1e12', rotational nonlinear form, 'dt=1e-4', 50 steps. Golden (B): 'worst-rel < 1e-6', 'div < 1e-8' ('test-step-decay.py:246-271'). Assert relative en…
- **A0514 [VERIFIED]:** survey line 2048; `pipe_flow_notes.md:71` -> `fn_shenfun/demo/pipe_flow_notes.md`; tokens=Stokes,monotonic;numbers=1,0.178; claim: - A pipe Stokes 'm=1': strictly monotonic, asymptotic rate '0.178', plateau '<5%' ('pipe-flow-notes.md:71').
- **A0515 [VERIFIED]:** survey line 2050; `test_mhd.py:163-180` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=channel,diffusion,energy0,energy1,magnetic;numbers=0; claim: - B channel MHD magnetic diffusion: '0 < energy1 < energy0' ('test-mhd.py:163-180').
- **A0516 [VERIFIED]:** survey line 2073; `test_pcf_fluctuations_jax.py:56-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_pcf_fluctuations_jax.py`; tokens=Assert,after,forced,imag,modes;numbers=0,0,0,1.,2; claim: - C: mean '(0,0)' modes forced real after a step, 'imag < 1e-12' ('test-pcf-fluctuations-jax.py:56-72'). B: 'enforce-mean-mode-cleanup' keeps mean 'v=0', mean 'u,w' real (§1.B.2). Assert symmetry residual '< 1e-12'.
- **A0517 [VERIFIED]:** survey line 2095; `OrrSommerfeld_eigs.py:183-184` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; numbers=8000,1,0.24707506017508621,0.0026644103710965817,1e-12; claim: - A, 'Re=8000, α=1': leading 'c = 0.24707506017508621 + 0.0026644103710965817 i' (positive Im → unstable), tolerance '1e-12' ('OrrSommerfeld-eigs.py:183-184').
- **A0518 [VERIFIED]:** survey line 2096; `test_linstab_poiseuille.py:7-19` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=c_ref;numbers=10000,1,0,0.23752649,0.00373967; claim: - B, 'Re=10000, α=1, β=0': reference 'c-ref = 0.23752649 + 0.00373967 i'; computed '0.23752722198590992 + 0.0037381198835812705 i', abs error '< 1e-4' ('test-linstab-poiseuille.py:7-19').
- **A0519 [VERIFIED]:** survey line 2097; `test_linstab_poiseuille.py:22-39` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=Poiseuille,_crit,change,critical,stable;numbers=1,5742.22,1.02,5802.22; claim: - Published critical Re (Poiseuille): 'Re-crit = 5772.22', 'α-crit ≈ 1.02056', 'c ≈ 0.26400' [Orszag71]. B verifies the sign change: stable at 'Re=5742.22, α=1.02', unstable at 'Re=5802.22' ('test-linstab-poiseuille…
- **A0520 [VERIFIED]:** survey line 2098; `couette_linear_benchmarks.md:84-88` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Romanov;numbers=1973; claim: - Plane Couette: linearly stable for all Re — no unstable OS eigenvalue (Romanov 1973, 'couette-linear-benchmarks.md:84-88').
- **A0521 [VERIFIED]:** survey line 2103; `couette_linear_benchmarks.md:111-118` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=Romanov,Streamwise,analytic,numeric;numbers=0,1,0,1.179054e-01,0; claim: - Couette stability: assert 'max Re(λ) < 0' for all tested '(α,β)'. Golden Romanov rates ('couette-linear-benchmarks.md:111-118'): 'ky=1,kz=0: −1.179054e-01'; 'ky=0,kz=1: −3.467401e-03'; 'ky=2,kz=1: −1.905757e-01'. St…
- **A0522 [VERIFIED]:** survey line 2103; `couette_linear_benchmarks.md:117` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=analytic,numeric;numbers=0,1,0,0,1; claim: - Couette stability: assert 'max Re(λ) < 0' for all tested '(α,β)'. Golden Romanov rates ('couette-linear-benchmarks.md:111-118'): 'ky=1,kz=0: −1.179054e-01'; 'ky=0,kz=1: −3.467401e-03'; 'ky=2,kz=1: −1.905757e-01'. St…
- **A0523 [VERIFIED]:** survey line 2103; `test_couette_linear.py:90-104` -> `fn_shenfun/demo/test_couette_linear.py`; tokens=Couette,Streamwise,analytic,assert;numbers=0,1,0,0,1; claim: - Couette stability: assert 'max Re(λ) < 0' for all tested '(α,β)'. Golden Romanov rates ('couette-linear-benchmarks.md:111-118'): 'ky=1,kz=0: −1.179054e-01'; 'ky=0,kz=1: −3.467401e-03'; 'ky=2,kz=1: −1.905757e-01'. St…
- **A0524 [VERIFIED]:** survey line 2107; `test_couette_linear.py:166` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/test_couette_linear.py`; tokens=FINITE_CAP;numbers=0; claim: Families. All three (channel/Couette). A uses Shen biharmonic Galerkin 'Aφ = c Bφ'; B uses OS/Squire primitive FD blocks with clamped rows; C uses jaxfun assembly. The Squire coupling '−iβU'' must be present for 3D…
- **A0525 [VERIFIED]:** survey line 2148; `couette_linear_benchmarks.md:314-316` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; numbers=24,0.4984075630441907,32,0.49840694616677383,48; claim: Eigenvalue refinement (F5). Increase 'N', confirm the leading eigenvalue converges to the golden monotonically. The Family A MRI pattern ('couette-linear-benchmarks.md:314-316'): 'nx=24 → 0.4984075630441907', 'nx=32…
- **A0526 [VERIFIED]:** survey line 2173; `fd_proto/test_rdt.py:12-26` -> `shearpy-jimenez/shearpy/tests/fd_proto/test_rdt.py`; tokens=coarse_t,error,fine_t,slope,time_order;numbers=3,1.,6,1.,5; claim: The integrators differ by construction (IMEXRK111/222/3/443 + CNAB2 in A/C, §1.A.6/§1.C.5; single-stage θ-method PC at 'implicit=0.51' in B, §1.B.2), so these tests are tailored per family — that tailoring is the poin…
- **A0527 [VERIFIED]:** survey line 2177; `ChannelFlow.py:151` -> `fn_shenfun/demo/ChannelFlow.py`; tokens=grad,linear;numbers=1,1; claim: Description & rationale. The cleanest temporal oracle is one Fourier mode under linear diffusion: '∂u/∂t = ν ∂²u/∂x²' for 'u=û(t)e(ikx)' gives 'dû/dt = −νk² û', exact 'û(t)=û-0 e(−νk²t)'. This isolates the impli…
- **A0528 [VERIFIED]:** survey line 2177; `solver.py:215` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=L_lhs,implicit;numbers=1,1,2; claim: Description & rationale. The cleanest temporal oracle is one Fourier mode under linear diffusion: '∂u/∂t = ν ∂²u/∂x²' for 'u=û(t)e(ikx)' gives 'dû/dt = −νk² û', exact 'û(t)=û-0 e(−νk²t)'. This isolates the impli…
- **A0529 [VERIFIED]:** survey line 2177; `base.py:222` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/base.py`; tokens=implicit;numbers=2; claim: Description & rationale. The cleanest temporal oracle is one Fourier mode under linear diffusion: '∂u/∂t = ν ∂²u/∂x²' for 'u=û(t)e(ikx)' gives 'dû/dt = −νk² û', exact 'û(t)=û-0 e(−νk²t)'. This isolates the impli…
- **A0530 [PARTIAL]:** survey line 2177; `test_theory.py:66-97` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. The cleanest temporal oracle is one Fourier mode under linear diffusion: '∂u/∂t = ν ∂²u/∂x²' for 'u=û(t)e(ikx)' gives 'dû/dt = −νk² û', exact 'û(t)=û-0 e(−νk²t)'. This isolates the impli…
- **A0531 [PARTIAL]:** survey line 2187; `integrators.py:836` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / IMEXRK111 / A, C / 1 / '[0.8, 1.3]' / 'integrators.py:836'; 'imex-rk.py:163' /
- **A0532 [VERIFIED]:** survey line 2187; `imex_rk.py:163` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; numbers=11,1; claim: / IMEXRK111 / A, C / 1 / '[0.8, 1.3]' / 'integrators.py:836'; 'imex-rk.py:163' /
- **A0533 [VERIFIED]:** survey line 2188; `taylor_couette_dns.py:288` -> `fn_shenfun/demo/taylor_couette_dns.py`; numbers=2; claim: / CNAB2 / A (TC/pipe), C (TC) / 2 / '[1.8, 2.3]' / 'taylor-couette-dns.py:288'; 'cnab2.py' /
- **A0534 [VERIFIED]:** survey line 2189; `solver.py:87` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=torch;numbers=1; claim: / torch θ-method PC (θ=0.51) / B / ~1 (see note) / '[0.85, 1.6]' / 'solver.py:87, 594-606' /
- **A0535 [PARTIAL]:** survey line 2190; `integrators.py:858` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / IMEXRK222 / A, C / 2 / '[1.8, 2.3]' / 'integrators.py:858'; 'imex-rk.py' /
- **A0536 [PARTIAL]:** survey line 2191; `integrators.py:665` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / IMEXRK3 (Spalart) / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:665'; 'imex-rk.py:96' /
- **A0537 [VERIFIED]:** survey line 2191; `imex_rk.py:96` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; tokens=IMEXRK3;numbers=3; claim: / IMEXRK3 (Spalart) / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:665'; 'imex-rk.py:96' /
- **A0538 [PARTIAL]:** survey line 2192; `integrators.py:872` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / IMEXRK443 / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:872'; 'imex-rk.py:184' /
- **A0539 [PARTIAL]:** survey line 2192; `imex_rk.py:184` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/integrators/imex_rk.py`; resolved-line-exists-but-claim-numbers-not-found; claim: / IMEXRK443 / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:872'; 'imex-rk.py:184' /
- **A0540 [VERIFIED]:** survey line 2204; `pcf_mhd_mri_shearpy.py:12-15` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=du_x,du_y;numbers=2,2,2,2,2; claim: Description & rationale. T1 exercises only the implicit branch. To verify the explicit branch carrying the rotation/shear source terms, use the epicyclic oscillation. A uniform ('k=0') perturbation with Coriolis…
- **A0541 [VERIFIED]:** survey line 2204; `PLAN_openpipeflow_vs_fnshenfun.md:191-202` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Exact,assertion,epicyclic,first,milestone;numbers=0,2,2,2,2; claim: Description & rationale. T1 exercises only the implicit branch. To verify the explicit branch carrying the rotation/shear source terms, use the epicyclic oscillation. A uniform ('k=0') perturbation with Coriolis…
- **A0542 [PARTIAL]:** survey line 2204; `test_theory.py:100-132` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. T1 exercises only the implicit branch. To verify the explicit branch carrying the rotation/shear source terms, use the epicyclic oscillation. A uniform ('k=0') perturbation with Coriolis…
- **A0543 [PARTIAL]:** survey line 2208; `test_theory.py:100-132` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: > Shearpy reference parametrization. The shearpy theory test ('test-theory.py:100-132') uses 'omega=1.0, shear=1.5' ⇒ 'κ=√(2·1·(2−1.5))=√1=1', amplitude '(shear−2ω)/κ=(1.5−2)/1=−0.5', integrating to 'T=1.0', asserti…
- **A0544 [VERIFIED]:** survey line 2216; `pcf_mhd_mri_shearpy.py:346-348` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=shear,terms;numbers=0.,2,2,1.; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B:…
- **A0545 [VERIFIED]:** survey line 2216; `pcf_mhd_mri_shearpy_jax.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; numbers=0.,2,2,1.; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B:…
- **A0546 [PARTIAL]:** survey line 2216; `mhd.py:71-74` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B:…
- **A0547 [PARTIAL]:** survey line 2216; `mhd.py:71-74` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B:…
- **A0548 [VERIFIED]:** survey line 2216; `PLAN_openpipeflow_vs_fnshenfun.md:125-137` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Coriolis,_base_coupling_terms,metadata,rotation,shear;numbers=0.,2,2,1.; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B:…
- **A0549 [PARTIAL]:** survey line 2220; `mhd.py:115-117` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. Resistive diffusion is treated implicitly in all MHD families (shenfun on the vector potential 'SA = MA − dt·γ·η·LA'; torch magnetic Helmholtz with '1/Rm', 'mhd.py:115-117'). A single magnet…
- **A0550 [PARTIAL]:** survey line 2220; `test_theory.py:66-97` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. Resistive diffusion is treated implicitly in all MHD families (shenfun on the vector potential 'SA = MA − dt·γ·η·LA'; torch magnetic Helmholtz with '1/Rm', 'mhd.py:115-117'). A single magnet…
- **A0551 [VERIFIED]:** survey line 2252; `poisson1D.py:46` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/poisson1D.py`; tokens=error;numbers=1,6,1000,0; claim: The procedure differs by family because the bases differ. Spectral families (A,C) converge exponentially for smooth fields [Boyd01, CHQZ06] (demonstrated in jaxfun's own MMS self-asserts 'poisson1D.py:46 error < ulp…
- **A0552 [VERIFIED]:** survey line 2262; `poisson1D.py:46` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/poisson1D.py`; tokens=Assert,assert,error;numbers=4,10,10,4,1000; claim: Metric & tolerance. Primary (exponential): for consecutive 'N' until the floor, 'E(N+4) < E(N)/10'; equivalently fit 'log E = −c·N + b' on a semilog axis and assert decay constant 'c > 0.5'. Floor handling: stop…
- **A0553 [VERIFIED]:** survey line 2266; `OrrSommerfeld_eigs.py:183` -> `fn_shenfun/demo/OrrSommerfeld_eigs.py`; numbers=8000; claim: Families. A: native ('chebyshev.la.Helmholtz'/'Biharmonic' or 'la.SolverGeneric1ND'); the OS golden 'c=0.24707506017508621+0.0026644103710965817j' at 'Re=8000' to '1e-12' ('OrrSommerfeld-eigs.py:183') is the eigenva…
- **A0554 [VERIFIED]:** survey line 2272; `mesh.py:169` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=point;numbers=9,1; claim: Description & rationale. For torch (B) the wall-normal/radial direction is finite-difference with a Taylor/Vandermonde stencil. The family is labeled "4th-order FD," but the default 'KL=4' 9-point centered stencil i…
- **A0555 [VERIFIED]:** survey line 2272; `test_mesh.py:21-37` -> `fn_openpipeflow-122/torchcouette/tests/test_mesh.py`; tokens=torch;numbers=4,4,9,8,2; claim: Description & rationale. For torch (B) the wall-normal/radial direction is finite-difference with a Taylor/Vandermonde stencil. The family is labeled "4th-order FD," but the default 'KL=4' 9-point centered stencil i…
- **A0556 [PARTIAL]:** survey line 2274; `mesh.py:186-189` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Setup (MMS). Apply the FD derivative matrices 'W-dy1, W-dy2' ('mesh.py:186-189') to a smooth field and measure derivative error vs analytic — the operator-level test mirroring 'test-mesh.py'. Field 'u-exact(y)=cos(3…
- **A0557 [PARTIAL]:** survey line 2274; `mesh.py:80-100` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Setup (MMS). Apply the FD derivative matrices 'W-dy1, W-dy2' ('mesh.py:186-189') to a smooth field and measure derivative error vs analytic — the operator-level test mirroring 'test-mesh.py'. Field 'u-exact(y)=cos(3…
- **A0558 [PARTIAL]:** survey line 2278; `mesh.py:189` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Metric & tolerance. Fit 'log E = −p·log N + b' (log-log); assert 'p ≥ 3.7' for the first-derivative operator (the documented FD floor; the clustered Chebyshev-extrema mesh makes the boundary-limited global order lan…
- **A0559 [VERIFIED]:** survey line 2280; `fd_proto/test_rdt.py:20-26` -> `shearpy-jimenez/shearpy/tests/fd_proto/test_rdt.py`; tokens=order,space_order;numbers=2,4,8,5.5; claim: Tolerance rationale. 'p≥3.7' is the documented FD floor — it rejects a stencil silently fallen to 2nd order (e.g. a boundary-row bug) while accepting the true (≥4, often 8 interior) order. The saturation allowance i…
- **A0560 [VERIFIED]:** survey line 2282; `mesh.py:26-57` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/mesh.py`; tokens=Taylor,weights; claim: Families. B: native — channel 'W-dy1/W-dy2', couette 'W-dr1/W-radlap', pipe banded 'W-dr1' (banded LU but same Taylor weights 'mesh.py:26-57'). For the pipe, the banded and dense builds are verified equal to '1e-12'…
- **A0561 [VERIFIED]:** survey line 2286; `PLAN_openpipeflow_vs_fnshenfun.md:110-111` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Spectral,effectively,eigenvalues,exact,modest; claim: Description & rationale. This is the operational definition of cross-family parity: at torch's converged resolution, a shared physical observable must sit within torch's own truncation band of the spectral (ef…
- **A0562 [VERIFIED]:** survey line 2286; `PLAN_openpipeflow_vs_fnshenfun.md:57-61` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Spectral,_spectral,cross,family,never; claim: Description & rationale. This is the operational definition of cross-family parity: at torch's converged resolution, a shared physical observable must sit within torch's own truncation band of the spectral (ef…
- **A0563 [VERIFIED]:** survey line 2290; `test_linstab_poiseuille.py:7-19` -> `fn_openpipeflow-122/torchchannel/tests/test_linstab_poiseuille.py`; tokens=Poiseuille,Sommerfeld,channel,eigenvalue,leading;numbers=10000,1,0,0.23752649,0.00373967; claim: S3a — Orr–Sommerfeld leading eigenvalue (channel, hydro). Plane Poiseuille, 'Re=10000, α=1, β=0'. Spectral oracle (A/C): the OS leading eigenvalue recomputed at high 'N' ('N=128' Chebyshev, converged ~'1e-10'). Torc…
- **A0564 [VERIFIED]:** survey line 2292; `couette_linear_benchmarks.md:313` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=s_max;numbers=0.75,16,0.7499999944199642,0.9373170323757943,0.75; claim: S3b — Epicyclic frequency κ or MRI growth 's-max' (MHD/rotation, A↔C; B only if rotation wired). Ideal local Keplerian oracle 's-max=0.75Ω', '(k v-A)²=(15/16)Ω²'; the local-MRI computation gives 's-max/Ω = 0.7499999…
- **A0565 [PARTIAL]:** survey line 2296; `mhd.py:71-74` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Metric & tolerance (the truncation band). At B's converged resolution (where its 'N'-refinement has plateaued — confirm '/c-B(201) − c-B(151)/' is below B's truncation estimate), assert '/c-B − c-spectral/ ≤ max(C…
- **A0566 [VERIFIED]:** survey line 2298; `PLAN_openpipeflow_vs_fnshenfun.md:60` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Tolerance,actual,roundoff; claim: Tolerance rationale. The band 'max(C·hp, C·dtq, ε-spectral)' is derived from B's actual grid, not chosen arbitrarily: at 'N=101, KL=4' on the clustered mesh, the FD truncation error in a leading OS eigenvalue is…
- **A0567 [VERIFIED]:** survey line 2300; `PLAN_openpipeflow_vs_fnshenfun.md:120` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=Doppler,_linear_analysis,eigenvalue,frame,index; claim: Families. A, C produce the oracle (spectral, 'N=128'). A↔C can additionally be compared at 'ε-spectral=1e-10' (both spectral Galerkin with matching IMEX tableaux, 'imex-rk.py == integrators.py') — a within-spect…
- **A0568 [PARTIAL]:** survey line 2309; `mhd.py:71-74` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / T2 epicyclic / temporal (explicit) / 'cos(κt)', 'κ²=2Ω(2Ω−S)' / ✓ / gated ('mhd.py:71-74') / ✓ / slope band + '/κ−κ-th//κ-th<1e-2' /
- **A0569 [VERIFIED]:** survey line 2320; `pcf_mhd_mri_shearpy.py:11-15` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; numbers=1; claim: These are the major focus of the suite; each has an analytic oracle. The physical anchor is the shearing-box / shearpy convention ('pcf-mhd-mri-shearpy.py:11-15', 'pcf-mhd-mri-notes.md:37-42'; [BH91, SG10]); the canonic…
- **A0570 [VERIFIED]:** survey line 2320; `pcf_mhd_mri_notes.md:37-42` -> `fn_shenfun/demo/pcf_mhd_mri_notes.md`; numbers=1,0; claim: These are the major focus of the suite; each has an analytic oracle. The physical anchor is the shearing-box / shearpy convention ('pcf-mhd-mri-shearpy.py:11-15', 'pcf-mhd-mri-notes.md:37-42'; [BH91, SG10]); the canonic…
- **A0571 [VERIFIED]:** survey line 2322; `pcf_mhd_mri_shearpy.py:11,102` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; tokens=shear;numbers=2,1,2,2,2; claim: Canonical conventions (apply to ALL SR tests — the §0.2 frame). 'x'=radial/wall-normal (shear gradient), 'y'=azimuthal/streamwise (wall motion), 'z'=vertical/rotation axis, 'Ω=Ω ẑ'. Base flow 'U-b(x)=−S·x·e-y', 'dU-…
- **A0572 [VERIFIED]:** survey line 2322; `pcf_mhd_mri_shearpy_jax.py:70-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=shear;numbers=2,1,2,2,2; claim: Canonical conventions (apply to ALL SR tests — the §0.2 frame). 'x'=radial/wall-normal (shear gradient), 'y'=azimuthal/streamwise (wall motion), 'z'=vertical/rotation axis, 'Ω=Ω ẑ'. Base flow 'U-b(x)=−S·x·e-y', 'dU-…
- **A0573 [VERIFIED]:** survey line 2322; `base_flow.py:38` -> `fn_openpipeflow-122/torchcouette/torchcouette/base_flow.py`; numbers=3,3; claim: Canonical conventions (apply to ALL SR tests — the §0.2 frame). 'x'=radial/wall-normal (shear gradient), 'y'=azimuthal/streamwise (wall motion), 'z'=vertical/rotation axis, 'Ω=Ω ẑ'. Base flow 'U-b(x)=−S·x·e-y', 'dU-…
- **A0574 [VERIFIED]:** survey line 2322; `mhd.py:100-101` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; numbers=1,1,1,1; claim: Canonical conventions (apply to ALL SR tests — the §0.2 frame). 'x'=radial/wall-normal (shear gradient), 'y'=azimuthal/streamwise (wall motion), 'z'=vertical/rotation axis, 'Ω=Ω ẑ'. Base flow 'U-b(x)=−S·x·e-y', 'dU-…
- **A0575 [VERIFIED]:** survey line 2324; `mhd.py:71-74` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; numbers=1,2,3,4,7; claim: Family applicability at a glance. A (shenfun): all SR tests run today (DNS MHD 'pcf-mhd-mri-shearpy.py' + dense-linear '-pcf-linear.py', 'taylor-couette-mri.py'). C (jax): all run today (DNS 'pcf-mhd-mri-shearpy-jax…
- **A0576 [VERIFIED]:** survey line 2324; `PLAN_openpipeflow_vs_fnshenfun.md:191-202` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=first,milestone,needed,rotation,tests;numbers=1,2,3,4,5; claim: Family applicability at a glance. A (shenfun): all SR tests run today (DNS MHD 'pcf-mhd-mri-shearpy.py' + dense-linear '-pcf-linear.py', 'taylor-couette-mri.py'). C (jax): all run today (DNS 'pcf-mhd-mri-shearpy-jax…
- **A0577 [PARTIAL]:** survey line 2328; `test_theory.py:100-132` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. With rotation Ω and shear S, a uniform ('k=0') velocity perturbation is a 2-D harmonic oscillator: Coriolis couples 'u-x ↔ u-y' at the epicyclic frequency κ. The most basic check that the so…
- **A0578 [VERIFIED]:** survey line 2330; `ChannelFlow.py:174-197` -> `fn_shenfun/demo/ChannelFlow.py`; numbers=0,0,0,1,0; claim: Setup. Triply-periodic dynamics (only 'k=0' excited); for wall-bounded families use the '(0,0)'-mode mean equations ('ChannelFlow.py:174-197'). Domain e.g. '((-1,1),(0,2π),(0,2π))', Lorentz/magnetic OFF. Keplerian '…
- **A0579 [PARTIAL]:** survey line 2342; `test_theory.py:131-132` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Metric & tolerance. Sample 'u-x(t), u-y(t)' at ≥20 times over two periods. 'rel-err = max-t/u-x(num)(t) − u-(x0)cos κt//u-(x0)' and likewise for 'u-y' (normalize by '/u-(x0)(S−2Ω)/κ/'). Pass if 'rel-err < 1e-2'…
- **A0580 [VERIFIED]:** survey line 2344; `pcf_mhd_mri_shearpy.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; numbers=2,2,2,1; claim: Families. A, C: run directly (A 'pcf-mhd-mri-shearpy.py:130-134, 346-348'; C 'pcf-mhd-mri-shearpy-jax.py:130-134'). B: acceptance gate — passes only once 'n-x += −2Ω u-y', 'n-y += (S−2Ω)u-x' wiring is added (and…
- **A0581 [VERIFIED]:** survey line 2344; `pcf_mhd_mri_shearpy_jax.py:130-134` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; numbers=2,2,2,1; claim: Families. A, C: run directly (A 'pcf-mhd-mri-shearpy.py:130-134, 346-348'; C 'pcf-mhd-mri-shearpy-jax.py:130-134'). B: acceptance gate — passes only once 'n-x += −2Ω u-y', 'n-y += (S−2Ω)u-x' wiring is added (and…
- **A0582 [VERIFIED]:** survey line 2348; `fd_proto/test_rdt.py:12-26` -> `shearpy-jimenez/shearpy/tests/fd_proto/test_rdt.py`; tokens=slope,space_order,time_order;numbers=0,1,2.8,3.4,5.5; claim: Description & rationale. Isolates the new induction term from the base shear: 'dB-y/dt = −S·B-x' (the Ω-effect — azimuthal field generated from radial field by differential rotation). For a uniform 'k=0' field wit…
- **A0583 [VERIFIED]:** survey line 2356; `pcf_mhd_mri_shearpy.py:16-21` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=induction,shear; claim: Families. A: term arises automatically from 'dA/dt=U×B' with 'U-b=−S x e-y' ('pcf-mhd-mri-shearpy.py:16-21, 366-376'); the linear operator has it explicitly 'L[by,bx]=Uprime=−S' ('-pcf-linear.py:239'). C: same via t…
- **A0584 [VERIFIED]:** survey line 2356; `_pcf_linear.py:239` -> `fn_shenfun/demo/_pcf_linear.py`; tokens=Uprime; claim: Families. A: term arises automatically from 'dA/dt=U×B' with 'U-b=−S x e-y' ('pcf-mhd-mri-shearpy.py:16-21, 366-376'); the linear operator has it explicitly 'L[by,bx]=Uprime=−S' ('-pcf-linear.py:239'). C: same via t…
- **A0585 [VERIFIED]:** survey line 2356; `mhd.py:71-74` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=shear; claim: Families. A: term arises automatically from 'dA/dt=U×B' with 'U-b=−S x e-y' ('pcf-mhd-mri-shearpy.py:16-21, 366-376'); the linear operator has it explicitly 'L[by,bx]=Uprime=−S' ('-pcf-linear.py:239'). C: same via t…
- **A0586 [PARTIAL]:** survey line 2360; `test_theory.py:66-97` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. A single magnetic Fourier mode with no flow decays purely resistively, calibrating the effective numerical resistivity and verifying the magnetic-diffusion operator ('η∇²b' / 'ηΔA') is discr…
- **A0587 [VERIFIED]:** survey line 2368; `pcf_mhd_divfree.py:159-192` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_divfree.py`; tokens=Helmholtz;numbers=1,1,0,3,0; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this te…
- **A0588 [VERIFIED]:** survey line 2368; `pcf_mhd_jax.py:78-85` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_jax.py`; numbers=1,1,0,3,0; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this te…
- **A0589 [VERIFIED]:** survey line 2368; `mhd.py:115-117` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=diffusion,magnetic;numbers=1,1,0,0; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this te…
- **A0590 [VERIFIED]:** survey line 2368; `test_mhd.py:163-180` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=channel,diffusion,magnetic,torch;numbers=1,1,0,3,0; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this te…
- **A0591 [VERIFIED]:** survey line 2368; `mhd.py:89-94` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; numbers=1,1,0,3,0; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this te…
- **A0592 [PARTIAL]:** survey line 2372; `test_theory.py:135-178` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. With a uniform background field 'B0' and no rotation/shear, a transverse perturbation propagates as an Alfvén wave with 'ω=k·v-A', 'v-A=B0/√ρ=B0' (Alfvén units), verifying the ideal inductio…
- **A0593 [PARTIAL]:** survey line 2378; `test_theory.py:135-178` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Metric & tolerance. FFT 'δu-y(t)' (or fit the oscillation); assert '/ω-meas − k v-A//(k v-A) < 1e-2'. Pass if holds. Physics-oracle tolerance: exact dispersion, error = time truncation + tiny resistive damping;…
- **A0594 [VERIFIED]:** survey line 2380; `pcf_mhd_mri_shearpy.py:127-132` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; tokens=_total_b_components,background,background_b;numbers=1,0,1,0,0; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri…
- **A0595 [VERIFIED]:** survey line 2380; `_pcf_linear.py:230-238` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/_pcf_linear.py`; numbers=1,0,1,0,0; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri…
- **A0596 [VERIFIED]:** survey line 2380; `pcf_mhd_mri_shearpy_jax.py:116-118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=background,background_b;numbers=1,1; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri…
- **A0597 [PARTIAL]:** survey line 2380; `mhd.py:207-228` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri…
- **A0598 [PARTIAL]:** survey line 2380; `mhd.py:100-101` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri…
- **A0599 [PARTIAL]:** survey line 2384; `test_theory.py:181-193` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Description & rationale. The geometry-free analytic heart of the MRI: for a Keplerian shearing box with uniform vertical field, the local 4×4 axisymmetric dispersion ('u-x,u-y,b-x,b-y') has maximum growth 's-max=(3/…
- **A0600 [VERIFIED]:** survey line 2386; `taylor_couette_mri.py:91-93` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=dlnr;numbers=4,4,2,2,0; claim: Setup. Run as a dense-linear / algebraic dispersion check for the ideal (inviscid, non-resistive) limit: form the 4×4 matrix and find its leading eigenvalue vs 'k v-A'. The local biquartic ('taylor-couette-mri.p…
- **A0601 [PARTIAL]:** survey line 2386; `test_theory.py:181-193` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: Setup. Run as a dense-linear / algebraic dispersion check for the ideal (inviscid, non-resistive) limit: form the 4×4 matrix and find its leading eigenvalue vs 'k v-A'. The local biquartic ('taylor-couette-mri.p…
- **A0602 [VERIFIED]:** survey line 2397; `couette_linear_benchmarks.md:313` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=_opt,cutoff,s_max;numbers=0.75,0.9375,0,3,0; claim: Metric & tolerance. Scan, find the max real eigenvalue and its argmax. Assert '/s-max/Ω − 0.75/ < 2e-3' and '/(k v-A)²-opt/Ω² − 0.9375/ < 5e-3'; for the cutoff, assert growth '>0' just inside '(k v-A)²=3Ω²' and '≈0'…
- **A0603 [VERIFIED]:** survey line 2399; `taylor_couette_mri.py:104-122` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=dispersion,mri_keplerian_optimum;numbers=4,4; claim: Families. A: 'taylor-couette-mri.py:104-122' 'mri-keplerian-optimum'. C: 'taylor-couette-mri-jax.py:47-62'. B: torch has no MRI operator — to run this as a 4×4 algebraic check, implement the dispersion evaluator…
- **A0604 [VERIFIED]:** survey line 2399; `taylor_couette_mri_jax.py:47-62` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/taylor_couette_mri_jax.py`; tokens=mri_keplerian_optimum;numbers=4,4; claim: Families. A: 'taylor-couette-mri.py:104-122' 'mri-keplerian-optimum'. C: 'taylor-couette-mri-jax.py:47-62'. B: torch has no MRI operator — to run this as a 4×4 algebraic check, implement the dispersion evaluator…
- **A0605 [VERIFIED]:** survey line 2405; `couette_linear_benchmarks.md:329` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=CircularCouette;numbers=1,2,0.5,3,2; claim: Setup. TC annulus, 'R1=1,R2=2', 'η=0.5', quasi-Keplerian ('Ω(r)∝r(−3/2)' analogue; the bench uses 'CircularCouette(1.0, 2.0, 1.0, 0.51.5)', 'couette-linear-benchmarks.md:329'). BCs under test: (a) conducting…
- **A0606 [VERIFIED]:** survey line 2405; `taylor_couette_mri.py:36-40,183-187` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Dirichlet,Neumann,Robin,conducting,function;numbers=1,2,3,2,1.0; claim: Setup. TC annulus, 'R1=1,R2=2', 'η=0.5', quasi-Keplerian ('Ω(r)∝r(−3/2)' analogue; the bench uses 'CircularCouette(1.0, 2.0, 1.0, 0.51.5)', 'couette-linear-benchmarks.md:329'). BCs under test: (a) conducting…
- **A0607 [VERIFIED]:** survey line 2405; `taylor_couette_mri.py:42-47,332-444` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Bessel,Dirichlet,Robin,conducting,function;numbers=1,2,3,2,1.0; claim: Setup. TC annulus, 'R1=1,R2=2', 'η=0.5', quasi-Keplerian ('Ω(r)∝r(−3/2)' analogue; the bench uses 'CircularCouette(1.0, 2.0, 1.0, 0.51.5)', 'couette-linear-benchmarks.md:329'). BCs under test: (a) conducting…
- **A0608 [VERIFIED]:** survey line 2405; `couette_linear_benchmarks.md:330-353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=_mag,conducting,insulating;numbers=1,2,0.5,3,2; claim: Setup. TC annulus, 'R1=1,R2=2', 'η=0.5', quasi-Keplerian ('Ω(r)∝r(−3/2)' analogue; the bench uses 'CircularCouette(1.0, 2.0, 1.0, 0.51.5)', 'couette-linear-benchmarks.md:329'). BCs under test: (a) conducting…
- **A0609 [VERIFIED]:** survey line 2414; `couette_linear_benchmarks.md:352-353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; tokens=conducting,insulating; claim: ('couette-linear-benchmarks.md:352-353'.) Note the sign flip: conducting is (marginally) unstable, insulating (marginally) stable at these BC-specific onset parameters.
- **A0610 [VERIFIED]:** survey line 2416; `PLAN_openpipeflow_vs_fnshenfun.md:120` -> `PLAN_openpipeflow_vs_fnshenfun.md`; tokens=eigenvalue,index;numbers=0; claim: Metric & tolerance. Compute 'max-kz Re(s)' per BC. Within-family (A↔A regression, or C reproducing its own golden): 'rel < 1e-3' against the stored golden (eigenvalue numbers, spectral-exact at modest 'Nr'). C…
- **A0611 [VERIFIED]:** survey line 2416; `test_taylor_couette_mri_jax.py:29-51` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/couette/test_taylor_couette_mri_jax.py`; tokens=insulating;numbers=32,0.25628761535339467,0.25995005500337837,0,9; claim: Metric & tolerance. Compute 'max-kz Re(s)' per BC. Within-family (A↔A regression, or C reproducing its own golden): 'rel < 1e-3' against the stored golden (eigenvalue numbers, spectral-exact at modest 'Nr'). C…
- **A0612 [VERIFIED]:** survey line 2418; `couette_linear_benchmarks.md:314-316` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/couette_linear_benchmarks.md`; numbers=0,0,3,0,0; claim: Families. A: 'taylor-couette-mri.py' (both BCs). C: 'taylor-couette-mri-jax.py' (both BCs, '-assemble-flux-parts' for insulating, 'm=0'). B: torch has conducting/homogeneous 'b=0' only, no insulating, no MRI opera…
- **A0613 [VERIFIED]:** survey line 2424; `taylor_couette_mri.py:553-654` -> `fn_shenfun/demo/taylor_couette_mri.py`; tokens=Fixed,Lundquist,_mag,bisection,critical;numbers=1; claim: Setup. TC, 'η=0.5', quasi-Keplerian, conducting walls (for the like-for-like scan; 'S=4.11'). Fixed Lundquist 'S=4.11'; scan 'Pm ∈ (0.1, 1)' (optionally '0.02'). For each Pm run the critical-Rm bisection 'critical-R…
- **A0614 [VERIFIED]:** survey line 2445; `test_pcf_mhd_mri_shearpy.py:80-84` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=Domain,growth,magnetic,perturbation;numbers=-2,2,0,4,0; claim: Setup. Shearpy net-flux MRI run (A/C). Domain '((-2,2),(0,4),(0,1))', 'Re=Rm=1000', 'S=1, Ω=2/3, b-z=0.025' ('test-pcf-mhd-mri-shearpy.py:80-84'). Conducting magnetic BC (A 'A∈TD³'; C same). Net vertical flux 'B0=b-…
- **A0615 [VERIFIED]:** survey line 2447; `test_pcf_mhd_mri_shearpy.py:130-131` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=assert;numbers=2,0,1.,1,1; claim: Oracle. Two oracles. Linear-growth phase (quantitative): magnetic energy grows, 'E-mag(t-end) > 2·E-mag(0)' (findings expect ~7× over 't=1..3'), strictly monotone increasing ('test-pcf-mhd-mri-shearpy.py:130-131')…
- **A0616 [VERIFIED]:** survey line 2447; `pcf_mhd_mri_shearpy.py:385-413` -> `fn_shenfun/demo/pcf_mhd_mri_shearpy.py`; tokens=Transport,Volume;numbers=2,0,7,1.,1; claim: Oracle. Two oracles. Linear-growth phase (quantitative): magnetic energy grows, 'E-mag(t-end) > 2·E-mag(0)' (findings expect ~7× over 't=1..3'), strictly monotone increasing ('test-pcf-mhd-mri-shearpy.py:130-131')…
- **A0617 [VERIFIED]:** survey line 2449; `test_pcf_mhd_mri_shearpy.py:130-131` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=assert;numbers=2,0,-1,2,0; claim: Metric & tolerance. Energy-balance residual: cross-family / DNS tolerance '< max(C·Δxp, C·Δtq, ε-spectral)' — with 'dt=0.005' (IMEXRK222, 'q=2') and spectral space, bounded by the time-truncation of the energy FD;…
- **A0618 [VERIFIED]:** survey line 2449; `test_pcf_mhd_mri_shearpy.py:133` -> `fn_shenfun/demo/test_pcf_mhd_mri_shearpy.py`; tokens=assert;numbers=0,-1,0,0,0; claim: Metric & tolerance. Energy-balance residual: cross-family / DNS tolerance '< max(C·Δxp, C·Δtq, ε-spectral)' — with 'dt=0.005' (IMEXRK222, 'q=2') and spectral space, bounded by the time-truncation of the energy FD;…
- **A0619 [VERIFIED]:** survey line 2451; `pcf_mhd_mri_shearpy.py:385-413` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/couette/pcf_mhd_mri_shearpy.py`; tokens=Maxwell,Reynolds,alpha,background_b,shear;numbers=1,2,8,2,4; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate…
- **A0620 [VERIFIED]:** survey line 2451; `pcf_mhd_mri_shearpy_jax.py:149-182` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_mhd_mri_shearpy_jax.py`; tokens=False,alpha,diagnostics,shear,total;numbers=1,2,8,2,4; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate…
- **A0621 [VERIFIED]:** survey line 2451; `mhd.py:401-412` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=False,Maxwell,Reynolds,channel,field;numbers=1,2,8,2,4; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate…
- **A0622 [VERIFIED]:** survey line 2451; `mhd.py:407-412` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=False,Maxwell,channel,field,stress;numbers=1,2,8,2,4; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate…
- **A0623 [VERIFIED]:** survey line 2451; `mhd.py:414-415` -> `fn_openpipeflow-122/torchchannel/torchchannel/mhd.py`; tokens=alpha,background_b,denominator;numbers=1,4,1; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate…
- **A0624 [VERIFIED]:** survey line 2466; `couette_linear_benchmarks.md:352-353` -> `fn_shenfun/demo/couette_linear_benchmarks.md`; numbers=+0.00332; claim: The load-bearing fact is the sign flip: '+0.00332' vs '−2.76e-4' ('couette-linear-benchmarks.md:352-353').
- **A0625 [PARTIAL]:** survey line 2476; `mhd.py:71-74` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / SR-1 epicyclic / temporal/physics / 'cos(κt)', 'κ²=2Ω(2Ω−S)' / ✓ / gate ('mhd.py:71-74') / ✓ / 'rel<1e-2'; FFT '/κ−κ-th//κ-th<1e-2' /
- **A0626 [VERIFIED]:** survey line 2519; `torchpipeflow/benchmarks/benchmark_hotspots.py:29,39-41` -> `fn_openpipeflow-122/torchpipeflow/benchmarks/benchmark_hotspots.py`; tokens=device;numbers=3; claim: / GPU / none — CPU/MPI only / yes by construction (device-agnostic tensors; CUDA exercised only in 'torchpipeflow/benchmarks/benchmark-hotspots.py:29,39-41') / yes (JAX/XLA; 'cuda13' extra configured in 'pyproje…
- **A0627 [VERIFIED]:** survey line 2519; `pyproject.toml:19-37` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/pyproject.toml`; tokens=cuda13;numbers=3,0.10,.1; claim: / GPU / none — CPU/MPI only / yes by construction (device-agnostic tensors; CUDA exercised only in 'torchpipeflow/benchmarks/benchmark-hotspots.py:29,39-41') / yes (JAX/XLA; 'cuda13' extra configured in 'pyproje…
- **A0628 [PARTIAL]:** survey line 2520; `docs/couette_fourier_layout.md:11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/docs/couette_fourier_layout.md`; resolved-line-exists-but-semantic-support-not-proven; claim: / TPU / none / none / yes (device-agnostic XLA + 'shard-map'; 'docs/couette-fourier-layout.md:11') /
- **A0629 [VERIFIED]:** survey line 2521; `torchchannel/.../solver.py:87,103` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex128,float,float64;numbers=4,28,4,4,28; claim: / Default float precision / float64 / complex128 / float64 ('torchchannel/.../solver.py:87,103') / float64 / complex128 (x64 forced at import, 'src/jaxfun/--init--.py:8') /
- **A0630 [VERIFIED]:** survey line 2521; `src/jaxfun/__init__.py:8` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; numbers=4,4,4,4; claim: / Default float precision / float64 / complex128 / float64 ('torchchannel/.../solver.py:87,103') / float64 / complex128 (x64 forced at import, 'src/jaxfun/--init--.py:8') /
- **A0631 [PARTIAL]:** survey line 2522; `test_float32.py:21-23` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Lower precision validated? / n/a (always f64) / yes — complex64/float32 path, roundtrip err '<5e-5' ('test-float32.py:21-23') / x64 is unconditional at import; toggle is global /
- **A0632 [PARTIAL]:** survey line 2524; `test_mhd.py:183-201` -> `partial:ambiguous-file`; partial:ambiguous-file; claim: / Autograd / none (NumPy/SciPy) / yes — full, end-to-end incl. magnetic→velocity Lorentz coupling ('test-mhd.py:183-201') / yes — 'value-and-grad'/'grad' ('pcf-minimal-seed-jax.py:118,136') /
- **A0633 [VERIFIED]:** survey line 2524; `pcf_minimal_seed_jax.py:118,136` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_minimal_seed_jax.py`; tokens=grad,value_and_grad; claim: / Autograd / none (NumPy/SciPy) / yes — full, end-to-end incl. magnetic→velocity Lorentz coupling ('test-mhd.py:183-201') / yes — 'value-and-grad'/'grad' ('pcf-minimal-seed-jax.py:118,136') /
- **A0634 [VERIFIED]:** survey line 2525; `sharding.py:24-40` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=vmap; claim: / Mode batching / vectorized NumPy + MPI mode distribution / explicit batched '(H,N,N)'/'(N,H)' tensor ops + 'einsum' (no 'vmap') / 'jax.vmap' (separable per-axis transforms, 'sharding.py:24-40') /
- **A0635 [VERIFIED]:** survey line 2526; `sharding.py:9-11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=Mesh,device,devices; claim: / Multi-device / MPI (slab/pencil) / none (single-device; no distributed primitives in solver) / 'shard-map' SPMD (CPU/GPU/TPU), 'Mesh(jax.devices(), ("k",))' ('sharding.py:9-11') /
- **A0636 [PARTIAL]:** survey line 2527; `la/eig.py:168` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/la/eig.py`; resolved-line-exists-but-semantic-support-not-proven; claim: / Eigensolver / 'scipy.linalg.eig' (host) / 'scipy.linalg.eig' (host) / 'scipy.linalg.eig' (host, 'la/eig.py:168') /
- **A0637 [VERIFIED]:** survey line 2533; `torchpipeflow/benchmarks/benchmark_hotspots.py:29` -> `fn_openpipeflow-122/torchpipeflow/benchmarks/benchmark_hotspots.py`; tokens=cuda,device; claim: B (torch) is device-agnostic by construction. Every tensor is created with an explicit 'device=' taken from the mesh ('self.mesh.y.device' / 'self.mesh.r.device'), and no solver kernel special-cases CUDA. Moving…
- **A0638 [VERIFIED]:** survey line 2533; `:39-41` -> `fn_openpipeflow-122/torchpipeflow/benchmarks/benchmark_hotspots.py`; tokens=cuda,device,synchronize,torch;numbers=39; claim: B (torch) is device-agnostic by construction. Every tensor is created with an explicit 'device=' taken from the mesh ('self.mesh.y.device' / 'self.mesh.r.device'), and no solver kernel special-cases CUDA. Moving…
- **A0639 [VERIFIED]:** survey line 2533; `torchcouette/.../mhd.py:36-50` -> `fn_openpipeflow-122/torchcouette/torchcouette/mhd.py`; tokens=TaylorCouetteMHDState,device,dtype,self,state;numbers=39; claim: B (torch) is device-agnostic by construction. Every tensor is created with an explicit 'device=' taken from the mesh ('self.mesh.y.device' / 'self.mesh.r.device'), and no solver kernel special-cases CUDA. Moving…
- **A0640 [VERIFIED]:** survey line 2535; `pyproject.toml:19-37` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/pyproject.toml`; tokens=cuda13,optional;numbers=.3,0.10,.1,3,3; claim: C (jax) targets CPU, GPU, and TPU through XLA with no device-specific branches. The local '.venv' reports Python 3.12.3 and JAX 0.10.1, and the CUDA build is configured as an optional dependency ('cuda13 = ["jax…
- **A0641 [PARTIAL]:** survey line 2535; `docs/couette_fourier_layout.md:11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/docs/couette_fourier_layout.md`; resolved-line-exists-but-claim-numbers-not-found; claim: C (jax) targets CPU, GPU, and TPU through XLA with no device-specific branches. The local '.venv' reports Python 3.12.3 and JAX 0.10.1, and the CUDA build is configured as an optional dependency ('cuda13 = ["jax…
- **A0642 [VERIFIED]:** survey line 2535; `__init__.py:4` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; tokens=Python,XLA_PYTHON_CLIENT_PREALLOCATE,false,preallocate; claim: C (jax) targets CPU, GPU, and TPU through XLA with no device-specific branches. The local '.venv' reports Python 3.12.3 and JAX 0.10.1, and the CUDA build is configured as an optional dependency ('cuda13 = ["jax…
- **A0643 [VERIFIED]:** survey line 2542; `torchchannel/.../solver.py:87` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex,complex128,dtype,torch;numbers=28,28,28,2; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0644 [VERIFIED]:** survey line 2542; `:100-101` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex,complex128,complex64,dtype,error;numbers=28,4,28,100,4; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0645 [VERIFIED]:** survey line 2542; `:103` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex,complex128,dtype,float32,float64;numbers=28,4,28,4,28; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0646 [VERIFIED]:** survey line 2542; `test_float32.py:21-23` -> `fn_openpipeflow-122/torchchannel/tests/test_float32.py`; tokens=complex,complex64,dtype,float32,torch;numbers=4,4,2,4,4; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0647 [VERIFIED]:** survey line 2542; `test_float32.py:44-50` -> `fn_openpipeflow-122/torchchannel/tests/test_float32.py`; tokens=complex,complex64,divergence,dtype,residual;numbers=4,4,4,4,4; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0648 [VERIFIED]:** survey line 2542; `solver.py:251-271` -> `fn_openpipeflow-122/torchchannel/torchchannel/solver.py`; tokens=complex,dtype,real_dtype,torch;numbers=4,4,2,4,4; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and…
- **A0649 [VERIFIED]:** survey line 2543; `src/jaxfun/__init__.py:8` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/__init__.py`; tokens=JAX_ENABLE_X64,True,config,jax_enable_x64,update;numbers=4,4,4,4,4; claim: - C: float64 is enabled globally and unconditionally at import by 'jax.config.update("jax-enable-x64", True)' ('src/jaxfun/--init--.py:8'). This is asserted by 'tests/test-x64-default.py:7-10' ('jax.config.read(…
- **A0650 [VERIFIED]:** survey line 2543; `tests/test_x64_default.py:7-10` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/tests/test_x64_default.py`; tokens=JAX_ENABLE_X64,True,config,dtype,float64;numbers=4,4,4,1,4; claim: - C: float64 is enabled globally and unconditionally at import by 'jax.config.update("jax-enable-x64", True)' ('src/jaxfun/--init--.py:8'). This is asserted by 'tests/test-x64-default.py:7-10' ('jax.config.read(…
- **A0651 [VERIFIED]:** survey line 2551; `utils/common.py:35-72` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/utils/common.py`; tokens=compile,jit_vmap,static_argnums; claim: C is JIT-pervasive. '@jax.jit' (often with 'static-argnums') decorates space methods and integrator steps; a 'jit-vmap' helper ('utils/common.py:35-72') jits and conditionally vmaps based on input rank; matrices and…
- **A0652 [VERIFIED]:** survey line 2557; `torchpipeflow/field_io.py:161` -> `fn_openpipeflow-122/torchpipeflow/torchpipeflow/field_io.py`; tokens=grad,no_grad,torch; claim: B is fully differentiable end-to-end, including through the divergence-free projection, the influence/boundary correction, the FFTs, and — critically — the MHD coupling. There are no '.detach()' calls in the step pa…
- **A0653 [VERIFIED]:** survey line 2557; `torchchannel/tests/test_mhd.py:183-201` -> `fn_openpipeflow-122/torchchannel/tests/test_mhd.py`; tokens=Lorentz,autograd,detach,finite,grad;numbers=0,2.0; claim: B is fully differentiable end-to-end, including through the divergence-free projection, the influence/boundary correction, the FFTs, and — critically — the MHD coupling. There are no '.detach()' calls in the step pa…
- **A0654 [VERIFIED]:** survey line 2557; `torchcouette/tests/test_mhd.py:105-130` -> `fn_openpipeflow-122/torchcouette/tests/test_mhd.py`; tokens=Lorentz,autograd,couette,coupling,detach;numbers=0,2.0; claim: B is fully differentiable end-to-end, including through the divergence-free projection, the influence/boundary correction, the FFTs, and — critically — the MHD coupling. There are no '.detach()' calls in the step pa…
- **A0655 [VERIFIED]:** survey line 2559; `:118` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_minimal_seed_jax.py`; tokens=energy,grad,perturbation,perturbation_energy,solver;numbers=118; claim: C provides JAX autograd with a minimal-seed adjoint. 'examples/pcf-minimal-seed-jax.py' builds the plane-Couette minimal-seed optimization loop: 'jax.grad(solver.perturbation-energy)(state)' for the energy gradient…
- **A0656 [VERIFIED]:** survey line 2559; `:136` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pcf_minimal_seed_jax.py`; tokens=gain,grad,gradient,objective,state;numbers=136; claim: C provides JAX autograd with a minimal-seed adjoint. 'examples/pcf-minimal-seed-jax.py' builds the plane-Couette minimal-seed optimization loop: 'jax.grad(solver.perturbation-energy)(state)' for the energy gradient…
- **A0657 [VERIFIED]:** survey line 2559; `utils/common.py:100-103` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/utils/common.py`; tokens=jacfwd,jacn; claim: C provides JAX autograd with a minimal-seed adjoint. 'examples/pcf-minimal-seed-jax.py' builds the plane-Couette minimal-seed optimization loop: 'jax.grad(solver.perturbation-energy)(state)' for the energy gradient…
- **A0658 [VERIFIED]:** survey line 2576; `sharding.py:30-40` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=vmap;numbers=1,2,3; claim: - C: 'jax.vmap' is the batching primitive. '-build-local-apply-fn(dim, ax, fn)' returns a 'jax.jit(jax.vmap(...))' that applies a 1-D transform along one axis of a local shard (2-D: single vmap; 3-D: nested vmap…
- **A0659 [VERIFIED]:** survey line 2582; `sharding.py:9` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=Mesh,device,devices,spmd,spmd_mesh; claim: - C: 'shard-map' SPMD. A global 1-D device mesh 'spmd-mesh = Mesh(jax.devices(), ("k",))' ('sharding.py:9') defines a spectral sharding 'P("k")' (axis 0 sharded) and physical sharding 'P(None, "k")' (axis 1 shar…
- **A0660 [VERIFIED]:** survey line 2582; `sharding.py:10-11` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=Mesh,None,physical,sharding,spectral;numbers=1,0,1,1; claim: - C: 'shard-map' SPMD. A global 1-D device mesh 'spmd-mesh = Mesh(jax.devices(), ("k",))' ('sharding.py:9') defines a spectral sharding 'P("k")' (axis 0 sharded) and physical sharding 'P(None, "k")' (axis 1 shar…
- **A0661 [VERIFIED]:** survey line 2582; `:14-21` -> `shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/sharding.py`; tokens=get_transposed_sharding,physical,sharded,sharding,spectral;numbers=1,0,1,14,1; claim: - C: 'shard-map' SPMD. A global 1-D device mesh 'spmd-mesh = Mesh(jax.devices(), ("k",))' ('sharding.py:9') defines a spectral sharding 'P("k")' (axis 0 sharded) and physical sharding 'P(None, "k")' (axis 1 shar…

## A.5 Complete Benchmark-Claim Ledger

- **B0001 [PARTIAL]:** survey line 12; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2026, -06, -07`; claim: Provenance. Generated 2026-06-07 from (i) a direct reading of the three solver codebases in the 'cfd' repository and (ii) the classic literature. Load-bearing implementation claims carry 'file:line' anchors into the source where availa…
- **B0002 [PARTIAL]:** survey line 37; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `122`; claim: - Missing test coverage now: no target-tree Taylor-Green/TGV harness was found in 'fn-shenfun/demo', 'fn-openpipeflow-122/torch', or 'fork-jaxfun/(examples,tests)'. F6 remains a required foundational test to add, not evidence of curre…
- **B0003 [PARTIAL]:** survey line 55; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4., 10, -16, 10, -12`; claim: - A = shenfun — spectral Galerkin (composite/Shen bases that bake boundary conditions into the trial space), Python on CPU with MPI ('mpi4py'/'mpi4py-fft'), float64. This is the spectral oracle: exponential convergence, divergence…
- **B0004 [PARTIAL]:** survey line 59; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 8`; claim: The families are intentionally not identical discretizations. Family A is the reference against which B (4th-order-family FD, ~8th-order interior) and C (spectral, JAX-native) are validated. Parity means agreement on formulation-indep…
- **B0005 [PARTIAL]:** survey line 73; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 28, 2, 4`; claim: / Precision floor / float64 / complex128 (float32 validated) / x64-by-default /
- **B0006 [PARTIAL]:** survey line 75; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4.2, .2, 3.12, 0.10, .1, 3, 2`; claim: Run environments are disjoint (no live cross-import): A → '/home/nauman/miniconda3/envs/shenfun/bin/python' (shenfun 4.2.2, no torch); B → '/home/nauman/miniconda3/envs/huggingface/bin/python' (torch, no shenfun); C → '/home/nauman/cfd…
- **B0007 [PARTIAL]:** survey line 98; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1, 1, 0.2, .3`; claim: - Canonical Lorentz force. +,boldsymbol(J)timesboldsymbol(B) with prefactor 1, in Alfvén / Lorentz–Heaviside units rho = mu-0 = 1 (so boldsymbol(B) is in velocity units and the Alfvén speed is v-A = /boldsymbol(B)/)…
- **B0008 [VERIFIED]:** survey line 115; anchors: A0022, A0023, A0024, A0025; local source anchor(s) verified; numbers: `1, 2, 2, 1, 1`; claim: / Lorentz prefactor / 1 ('pcf-mhd-divfree.py:325-333') / Ha2/(Re,Rm) channel ('mhd.py:100-101'); Ha2/Pm couette ('mhd.py:79') → set explicit override = 1 for oracle tests / 1 ('pcf-mhd-jax.py:175-176') /
- **B0009 [PARTIAL]:** survey line 131; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3., 1, 1, 2, 2`; claim: 3. Override the Lorentz prefactor to 1 for any oracle/parity comparison: set B's 'lorentz-prefactor = 1' instead of Ha2/(Re,Rm) (channel) / Ha2/Pm (couette).
- **B0010 [PARTIAL]:** survey line 133; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 2`; claim: All cross-family comparisons should operate on canonical-frame observables (growth rates, energies, stresses), which are frame-invariant once the adapter is applied. This should be implemented as 'parity/conventions.py::to-canonical()'…
- **B0011 [VERIFIED]:** survey line 137; anchors: A0034; local source anchor(s) verified; numbers: `4, 4, 4, 9, 8, 8, 7, 8, 1e-7, 0, 8, 4`; claim: - B's "4th-order FD" label vs. reality. B is labeled a 4th-order FD family, but the default half-bandwidth 'KL=4' gives a 9-point centered stencil that is formally 8th-order interior (first/second derivatives accurate to 8th/7th or…
- **B0012 [PARTIAL]:** survey line 167; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0.2, .3`; claim: For oracle comparisons against A/C, B's Lorentz prefactor is overridden to 1 (§0.2.3).
- **B0013 [PARTIAL]:** survey line 177; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: The implemented shearing-box source terms (A and C; the acceptance target for B, §III.1) are, in canonical axes:
- **B0014 [VERIFIED]:** survey line 193; anchors: A0056; local source anchor(s) verified; numbers: `8000, 1, 0.24707506017508621, 0.0026644103710965817, 10, -12`; claim: / Orr–Sommerfeld leading eigenvalue, Re=8000, alpha=1 (A) / — / c = 0.24707506017508621 + 0.0026644103710965817,i, tol 10(-12) ('OrrSommerfeld-eigs.py:183-184') /
- **B0015 [VERIFIED]:** survey line 194; anchors: A0057; local source anchor(s) verified; numbers: `10000, 1, 0.23752649, 0.00373967, 10, -4`; claim: / Orr–Sommerfeld leading eigenvalue, Re=10000, alpha=1 (B) / c-(text(ref)) = 0.23752649 + 0.00373967,i / tol 10(-4) ('torchchannel/tests/test-linstab-poiseuille.py:7-19') /
- **B0016 [PARTIAL]:** survey line 195; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `5772.22, 1, 10, -2`; claim: / OS critical Reynolds (cross-family) / Re-(text(crit)) = 5772.22 [Orszag71] / rel <10(-2) /
- **B0017 [VERIFIED]:** survey line 196; anchors: A0058; local source anchor(s) verified; numbers: `0.75, 2, 2, 15, 16, 0.7499999944199642, 0.9373170323757943`; claim: / Ideal local Keplerian MRI / s-(max)/Omega = 0.75, (k v-A)2/Omega2 = 15/16 / A/C: 0.7499999944199642, 0.9373170323757943 ('couette-linear-benchmarks.md:313') /
- **B0018 [VERIFIED]:** survey line 197; anchors: A0059; local source anchor(s) verified; numbers: `0.5, 24.7, 4.11, 0, +0.003322863594034156, 1.75`; claim: / TC MRI conducting walls (eta=0.5 quasi-Kep, Rm=24.7, S=4.11) / growth > 0 / +0.003322863594034156 at best k-z=1.75 ('couette-linear-benchmarks.md:352') /
- **B0019 [VERIFIED]:** survey line 198; anchors: A0060; local source anchor(s) verified; numbers: `16.5, 5.21, 0, -0.00027582037141390655, 1.25`; claim: / TC MRI insulating walls (Rm=16.5, S=5.21) / growth < 0 (sign flip) / -0.00027582037141390655 at best k-z=1.25 ('couette-linear-benchmarks.md:353') /
- **B0020 [PARTIAL]:** survey line 200; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `+0.00332, -2.76, 0, -4, 6, 9`; claim: The conducting/insulating sign flip (+0.00332 vs -2.76times10(-4)) is reproduced by A and C only (B has no insulating walls); pin 'magnetic-bc' identically on both sides before comparing (§Part IV/SR-6, SR-9).
- **B0021 [PARTIAL]:** survey line 234; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0`; claim: / s / complex growth rate / eigenvalue / propto e(st); text(Re)(s)>0 unstable /
- **B0022 [PARTIAL]:** survey line 256; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0`; claim: Family A contains two complementary stacks throughout: a nonlinear pseudo-spectral DNS and a dense linear-stability / non-modal layer (collocation and Galerkin) that shares Butcher tableaux with the DNS time-steppers. Family A…
- **B0023 [VERIFIED]:** survey line 371; anchors: A0092; local source anchor(s) verified; numbers: `0, 0, 0, 0, 4, 1.0, 8000, 80, 1, 2, 1, 2`; claim: 'OrrSommerfeld-eigs.py' solves Avarphi=cBvarphi in Shen's biharmonic Chebyshev basis ('bc=(0,0,0,0)', dim N-4); default 'alfa=1.0, Re=8000, N=80'. Operators ('OrrSommerfeld-eigs.py:84-99'): with weighted inner products K=(u'',v)-w,…
- **B0024 [PARTIAL]:** survey line 373; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 2, 4, 2, 2`; claim: B=-mathrm(Re),alpha,mathrm(i),(K-alpha2 M), quad A = Q - 2alpha2 K + (alpha4 - 2alpha,mathrm(Re),mathrm(i))M - mathrm(i)alphamathrm(Re),(K2-alpha2 K1).
- **B0025 [VERIFIED]:** survey line 375; anchors: A0093; local source anchor(s) verified; numbers: `8000, 1, 80`; claim: Golden eigenvalue (self-asserted, 'OrrSommerfeld-eigs.py:183-184'): at mathrm(Re)=8000, alpha=1, N>80,
- **B0026 [PARTIAL]:** survey line 377; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.24707506017508621, 0.0026644103710965817, 10, -12`; claim: c = 0.24707506017508621 + 0.0026644103710965817,mathrm(i), qquad text(tol ) 10(-12).
- **B0027 [VERIFIED]:** survey line 379; anchors: A0094, A0095; local source anchor(s) verified; numbers: `10, -7, 1, 2, 2, 1000, 139, 1165.2, 1165.93, 139, 3`; claim: The DNS OS path ('OrrSommerfeld.py'/'OrrSommerfeld2D.py') seeds the eigenmode at amplitude 10(-7) on the Poiseuille base 1-x2 and checks energy proptoexp(2,mathrm(Im)(c),t) ('OrrSommerfeld.py:47-56'). PCF is linearly stable fo…
- **B0028 [PARTIAL]:** survey line 383; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 0`; claim: Two operators target the generalized eigenproblem Lq=sMq for perturbations q(x)exp(s t+mathrm(i)k-y y+mathrm(i)k-z z) (k-y2+k-z2>0), base flow U(x)=U-(text(off))+U'x along mathbf(e)-y:
- **B0029 [VERIFIED]:** survey line 388; anchors: A0098, A0099; local source anchor(s) verified; numbers: `0, 1e8, 0, 4, 6, 0, -8`; claim: Both treat pressure as a Lagrange multiplier enforcing mathrm(div)(u)=0 (saddle-point), with -nabla p columns and mathrm(div),u rows ('-pcf-linear.py:220-228'); the pressure block has zero mass so M is singular and the…
- **B0030 [PARTIAL]:** survey line 419; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 2, 2, 0, 2, 2, 1, 0, 3, 2, 0`; claim: Velocity uses the Dirichlet composite 'bc=(0,0)' (dim N-2); pressure the orthogonal space sliced to N-2 modes ('SP.slice = lambda: slice(0,N-2)'), giving the inf-sup-stable P-N/P-(N-2) pair with 'assert SP.dim()==SD.dim()'. Family…
- **B0031 [VERIFIED]:** survey line 482; anchors: A0115, A0116; local source anchor(s) verified; numbers: `2.4, 1, 2, 2`; claim: Same CNAB2 / coupled saddle-point per (m,k-z) as §I.A.2.4. For a time-dependent body force (Womersley) the force is evaluated at the midpoint t(n+1/2) for 2nd-order accuracy ('pipe-flow-dns.py:299-309'). Exact reference solutions…
- **B0032 [PARTIAL]:** survey line 484; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 2, 2, 4, 8`; claim: - Hagen–Poiseuille: u-z(r)=dfrac(f-z)(4nu)(R2-r2), flow rate Q=dfrac(pi R4 f-z)(8nu).
- **B0033 [PARTIAL]:** survey line 485; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `1, 3, 2, 3, 4, 1, 3, 2, 3, 2, 5`; claim: - Womersley (-partial-z p=Kcosomega t, rho=1, alpha=Rsqrt(omega/nu), mathrm(i)(3/2)=e(mathrm(i)3pi/4)): displaystyle u-z(r,t)=mathrm(Re)!left(frac(K)(mathrm(i)omega)!left[1-frac(J-0(mathrm(i)(3/2)alp…
- **B0034 [PARTIAL]:** survey line 486; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0, 2, 2`; claim: - Bessel viscous decay: u-z(r,0)=J-0(j-(0,n)r/R), decaying as exp(-nu j-(0,n)2 t/R2).
- **B0035 [VERIFIED]:** survey line 488; anchors: A0117; local source anchor(s) verified; numbers: `10, -6, 10, -10, 1.4, 0, -12, 10, -10, 0, 1, 2`; claim: Golden tolerances ('test-pipe-flow-dns.py', 'pipe-flow-notes.md:67-71'): Hagen–Poiseuille max/u-z-text(exact)/<10(-6), /Q-Q-(text(exact))//Q-(text(exact))<10(-10) (measured 1.4times10(-12)), mathrm(div)-(infty)<10(-10);…
- **B0036 [VERIFIED]:** survey line 512; anchors: A0120; local source anchor(s) verified; numbers: `8, 8, 8, 400, 0.003, 9.41, 0, -17, 3.05, 0, -21, 16`; claim: Golden div-control numbers ('pcf-mhd-divfree-notes.md:69-96'): Legendre N=(8,8,8), mathrm(Re)=mathrm(Rm)=400, t=0.003 → mathrm(div),U-(L-2)=9.41times10(-17), mathrm(div),B-(L-2)=3.05times10(-21); Chebyshev N=(16,16,16)…
- **B0037 [VERIFIED]:** survey line 516; anchors: A0121, A0122, A0123; local source anchor(s) verified; numbers: `0, 0, 230, -245, 0, 0, 0, 0, 9, 50, 3.0741038, 0`; claim: Collocation primitive variables add (b-x,b-y,b-z,phi), where phi is a magnetic-pressure Lagrange multiplier enforcing mathrm(div)(b)=0 ('-pcf-linear.py:240-245'). Induction couplings with imposed uniform field B-0=(0,b-y,b-z) ('…
- **B0038 [PARTIAL]:** survey line 530; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 2, 1, 2, 2, 2, 2`; claim: with scalar Laplacian L-p=partial-(rr)+tfrac1rpartial-r-(m2/r2+k-z2), vector diagonal L-v=L-p-1/r2, cross terms pm2mathrm(i)m/r2, and 2a=2Omega+rOmega' (const). The radial induction has no shear source (only advecti…
- **B0039 [VERIFIED]:** survey line 540; anchors: A0126; local source anchor(s) verified; numbers: `0.5, 7, 2023, 24.7, 4.11, 1.75, +0.003322863594034156, 16.5, 5.21, 1.25, -0.00027582037141390655, 1`; claim: Sign-distinguishing golden numbers (eta=0.5 quasi-Keplerian, mathrm(Pm)to0, [LL07]/Rüdiger 2023; 'couette-linear-benchmarks.md:352-353'): conducting target mathrm(Rm)=24.7, S=4.11, best k-z=1.75 → text(growth)=+0.003322…
- **B0040 [VERIFIED]:** survey line 564; anchors: A0131; local source anchor(s) verified; numbers: `2, 3, 0.025, 25.81988897471611, .498406, 0.5, -1, 2, 0, 10, -10`; claim: Validation ('couette-linear-benchmarks.md:313-317'): PCF rotating-shear MRI analogue at Omega=2/3, b-z=0.025, k-z=25.81988897471611 → leading eigenvalue sapprox0.498406 (theory s=0.5); DNS netflux case grows E-(text(mag)) mo…
- **B0041 [PARTIAL]:** survey line 630; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `12., 1e8`; claim: 12. Linear operators have singular mass M (zero pressure/phi mass) → filter infinite eigenvalues with 'FINITE-CAP=1e8'.
- **B0042 [PARTIAL]:** survey line 639; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.2, 0.2, 4, 4, 9, 8, 1, 6, 1, 2.`; claim: Two contradictions from §0.2 are honored throughout: (i) family B's axis labels are swapped relative to the canonical frame (B uses streamwise='x', wall-normal='y'; the canonical frame uses wall-normal='x', streamwise='y') — apply the pl…
- **B0043 [PARTIAL]:** survey line 684; anchors: A0150, A0151, A0152; one or more local source anchors only partially verified; numbers: `0, 1, 1`; claim: The FD weights mirror OpenPipeFlow's 'mes-weights' ('mesh.py:12-45'). For target x-0 and stencil (x-j), build A-(:,0)=1, A-(:,j)=A-(:,j-1)cdot(x-x-0)/j so that A-(j)=(x-x-0)j/j!, then solve the transposed system A(math…
- **B0044 [PARTIAL]:** survey line 686; anchors: A0153, A0154, A0155; one or more local source anchors only partially verified; numbers: `2, 1, 4, 9, 9, 8, 8, 4, 0., .8, 10, -7`; claim: Stencil width and order. 'KL' is the half-bandwidth; the interior stencil is min(2,KL+1,,N) centered points ('mesh.py:109-117'), default 'KL=4' Rightarrow 9-point centered stencil. With 9 points the interior FD is polynomia…
- **B0045 [VERIFIED]:** survey line 714; anchors: A0169, A0170; local source anchor(s) verified; numbers: `2, 2, 0, 0, 2`; claim: Channel (Cartesian): partial-x f=i k-alpha f, partial-z f=i m-beta f, wall-normal partial-y,partial-(yy) via 'einsum("ij,...jkm->...ikm", W, f)'; nabla2 f=partial-(yy)f-k2 f; nabla!cdot!mathbf u=partial-x u+partia…
- **B0046 [VERIFIED]:** survey line 724; anchors: A0173; local source anchor(s) verified; numbers: `0.51, 0, 0.51, 2, 0.51, 0.5`; claim: The scheme is a one-stage θ-method (Crank–Nicolson-like) with diffusion split implicit/explicit and nonlinear + base-flow advection explicit, wrapped in a fixed-point predictor/corrector. The implicit fraction is theta=textt…
- **B0047 [VERIFIED]:** survey line 756; anchors: A0183, A0184, A0185, A0186, A0187; local source anchor(s) verified; numbers: `3, 1, 10, -10, 1, 1, 1, 2`; claim: Couette uses an iterated/adaptive corrector instead of a fixed count ('step-with-info', 'solver.py:553-594'): it loops up to 'max-corrector-iters' (default 3), blends N=theta N-1+(1-theta)N-0 ('solver.py:574-577'), measures text(…
- **B0048 [VERIFIED]:** survey line 777; anchors: A0195, A0196; local source anchor(s) verified; numbers: `0, 0`; claim: Eigenvalues are temporal rates lambda for exp(ialpha x+ibeta z+lambda t); phase speed c=ilambda/alpha ('linstab.py:124-125'). OS BC rows clamp v=Dv=0 at both walls; Squire rows clamp eta=0 ('linstab.py:42-58').
- **B0049 [VERIFIED]:** survey line 860; anchors: A0247, A0248, A0249; local source anchor(s) verified; numbers: `1, 1, 1, 3, 1e-10, 0.5`; claim: The time integrator is the same family θ-method predictor/corrector ('-step-with-history', 'solver.py:1028-1114'): RHS text(rhs)=N+tfrac(1)(Delta t)b+(1-theta)nu[text(radlap)(b)+d,b] ('-rhs-meshmult', 'solver.py:678-693'); predict…
- **B0050 [VERIFIED]:** survey line 862; anchors: A0250; local source anchor(s) verified; numbers: `3.0132822082797048, 0, -7, 5, 0, -5, 16, 4, 4, 4000, 0.75, 10`; claim: Pipe golden regression: against the Fortran reference, energy =3.0132822082797048times10(-7) within rel 5times10(-5) for N(=)16,K(=)4,M(=)4,Re(=)4000,alpha(=)0.75,dt(=)10(-3),theta(=)0.5 ('test-fortran-regression.py:15-16').
- **B0051 [PARTIAL]:** survey line 888; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0.2, 1`; claim: For oracle cross-checks against the canonical +mathbf Jtimesmathbf B with prefactor 1 (§0.2), pass 'lorentz-prefactor=1' explicitly.
- **B0052 [VERIFIED]:** survey line 890; anchors: A0256, A0257, A0258; local source anchor(s) verified; numbers: `0.0, 0, 10, -10`; claim: Induction partial-tmathbf bbig/-(rm expl)=operatorname(curl)(mathbf u-(rm tot)timesmathbf B-(rm tot)) computed pseudospectrally, where mathbf u-(rm tot) includes base flow (perturbation form) and mathbf B-(rm tot)=m…
- **B0053 [VERIFIED]:** survey line 910; anchors: A0266; local source anchor(s) verified; numbers: `1, 1, 1`; claim: Pm / magnetic diffusion ('-build-magnetic-operators', 'mhd.py:153-163'): c-1=1/Delta t, LHS c-2=-theta/Pm, RHS (1-theta)/Pm, i.e. magnetic diffusivity =1/Pm relative to unit-viscosity momentum (so Pm=nu/eta; larger Pm…
- **B0054 [VERIFIED]:** survey line 914; anchors: A0268, A0269, A0270, A0271, A0272; local source anchor(s) verified; numbers: `0, 8, 8, 3, 1e-10`; claim: Induction =operatorname(curl)(mathbf utimesmathbf B-(rm tot)) ('mhd.py:316-337'); Lorentz =C-L,mathbf Jtimesmathbf B-(rm tot), mathbf J=operatorname(curl)(text(induced )mathbf b) ('mhd.py:339-370'). Magnetic w…
- **B0055 [PARTIAL]:** survey line 948; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3`; claim: - Backend: pure PyTorch (FFT + dense/banded linear solves), CPU default everywhere, device-agnostic ('device' param). CUDA is supported by construction (all ops are tensor ops with explicit 'device='); it is only special-cased in t…
- **B0056 [PARTIAL]:** survey line 949; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `28, 4, 4, 2, 2., 5, 0, -5, 4, 10, -4, 10`; claim: - Precision: 'complex128' default (real 'float64') in all three solvers; 'complex64'/'float32' supported and validated (channel 'test-float32.py': roundtrip error <5times10(-5); solver step preserves 'complex64', div <10(-4)…
- **B0057 [VERIFIED]:** survey line 950; anchors: A0283, A0284; local source anchor(s) verified; numbers: `0, 0, 2.0`; claim: - Autograd: the solvers are fully differentiable end-to-end through projection, BC correction, FFTs, and the MHD Lorentz coupling — no '.detach()' in step paths ('torch.no-grad()' only in IO). The banded 'lu-solve' is autograd-awar…
- **B0058 [VERIFIED]:** survey line 956; anchors: A0285; local source anchor(s) verified; numbers: `0.23752649, +0.00373967, 10000, 1, 0, 101, 4, 10, -4`; claim: / Poiseuille OS leading c (ref) / 0.23752649+0.00373967,i / Re(=)10000,alpha(=)1,beta(=)0,N(=)101,KL(=)4; 'tests/test-linstab-poiseuille.py:7-19', asserts /c-c-(rm ref)/<10(-4) /
- **B0059 [VERIFIED]:** survey line 957; anchors: A0286; local source anchor(s) verified; numbers: `0.23752722198590992, +0.0037381198835812705`; claim: / Poiseuille OS leading c (computed) / 0.23752722198590992+0.0037381198835812705,i / 'VALIDATION.md:83-96' /
- **B0060 [VERIFIED]:** survey line 958; anchors: A0287; local source anchor(s) verified; numbers: `5742.22, 5802.22, 1.02, 96`; claim: / Critical-Re sign change / stable Re(=)5742.22, unstable Re(=)5802.22 / alpha(=)1.02,N(=)96; 'test-linstab-poiseuille.py:22-39' /
- **B0061 [VERIFIED]:** survey line 959; anchors: A0288; local source anchor(s) verified; numbers: `2000, 10, -20, 10, -12, 10, -14, 9, 500, 0.01`; claim: / Laminar Couette 2000-step decay / E-(rm pert)<10(-20), div <10(-12), max/u/<10(-14) / N(=)9,Re(=)500,dt(=)0.01; 'test-step-decay.py:29-51' /
- **B0062 [PARTIAL]:** survey line 960; anchors: A0289; one or more local source anchors only partially verified; numbers: `10, -7, 0., .8, 2, 0, 2, 2, 3`; claim: / Mesh polynomial exactness / err <10(-7), deg 0..8; int1(=)2,int y(=)0,int y2(=)2/3 / 'test-mesh.py:21-48' /
- **B0063 [VERIFIED]:** survey line 961; anchors: A0290; local source anchor(s) verified; numbers: `10, -7, 100, 5, 100`; claim: / Channel MHD div-free / div, divB<10(-7); Rm(=)100,Pm(=)5Rightarrow Pm(=)100/Re / 'test-mhd.py:54-180' /
- **B0064 [PARTIAL]:** survey line 962; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `32, 16, 16, 6.0, 2.6179938779914944, 0.868, 200, -200`; claim: / Couette nsCouette reference / m-r(=)32,m-theta(=)16,m-(z0)(=)16,k-(theta0)(=)6.0,k-(z0)(=)2.6179938779914944,eta(=)0.868,Re-i(=)200,Re-o(=)-200 / 'test-fortran-reference-config.py' /
- **B0065 [PARTIAL]:** survey line 963; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1000, 10, -8, 10, -12, 10, -8`; claim: / Couette laminar 1000-step / div <10(-8), E-(rm pert)<10(-12), Nu-i,Nu-oapprox1 (atol 10(-8)) / 'test-integration-laminar.py' /
- **B0066 [PARTIAL]:** survey line 964; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10, -8, 10, -8, 0, 10, -12, 8, 0.868, 20, -20, 2`; claim: / Couette MHD / div <10(-8), divB-(Linfty)<10(-8), walls mathbf b(=)0 (atol 10(-12)) / 'test-mhd.py' (N(=)8,eta(=)0.868,Re-i(=)20,Re-o(=)-20,Pm(=)2,Ha(=)1) /
- **B0067 [VERIFIED]:** survey line 965; anchors: A0291; local source anchor(s) verified; numbers: `3.0132822082797048, 0, -7, 5, 0, -5`; claim: / Pipe Fortran regression energy / 3.0132822082797048times10(-7) (rel <5times10(-5)) / 'test-fortran-regression.py:15-16' /
- **B0068 [VERIFIED]:** survey line 966; anchors: A0292; local source anchor(s) verified; numbers: `10, -10, 10, -12`; claim: / Pipe banded↔dense step / rtol 10(-10)/atol 10(-12) / 'test-banded.py:79-100' /
- **B0069 [PARTIAL]:** survey line 968; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `5772.22, 1, 10, -2, 0, 1, 2, 5`; claim: Cross-family OS comparisons use the published Re-(rm crit)=5772.22 [Orszag71] at rel <10(-2) (the within-family golden c above is family-specific and not portable; see the §0 hand-off note). For all MRI/rotation acceptance (SR-1, S…
- **B0070 [PARTIAL]:** survey line 976; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5., 1, 1e-10, 3`; claim: 5. Corrector differs: channel fixed ('corrector-iterations', default 1); couette/pipe iterated to 'tol=1e-10' ('max-corrector-iters=3') with 'StepInfo'.
- **B0071 [PARTIAL]:** survey line 1006; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 5, 6`; claim: integrators; §I.C.4 the JAX-specific compute capabilities; §I.C.5 golden numbers; §I.C.6 the explicit pipe gap and
- **B0072 [PARTIAL]:** survey line 1129; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e6`; claim: 'generalized-eig(L,M)' via host 'scipy.linalg.eig' with finite-eigenvalue caps 'MODAL-FINITE-CAP = 1e6',
- **B0073 [PARTIAL]:** survey line 1425; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `2`; claim: adjoint/minimal-seed loops [PWK12], validated against finite differences ('test-differentiability-jax.py').
- **B0074 [PARTIAL]:** survey line 1453; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: These are the within-family acceptance constants (tolerance ladder per §1 hand-off notes). Use Family-C goldens
- **B0075 [PARTIAL]:** survey line 1454; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2`; claim: only against C; cross-family physics oracles use published values at rel < 1e-2.
- **B0076 [VERIFIED]:** survey line 1456; anchors: A0406; local source anchor(s) verified; numbers: `9, 8, 8, 10, -3`; claim: PCF fluctuation diagnostics ('test-pcf-fluctuations-jax.py:78-95'; N=(9,8,8), Legendre, dt=10(-3),
- **B0077 [PARTIAL]:** survey line 1457; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.05, 4, 1e-10`; claim: amp 0.05, one step, x64, 'rtol=1e-10'):
- **B0078 [PARTIAL]:** survey line 1459; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.21836099019180652`; claim: Epert = 0.21836099019180652
- **B0079 [PARTIAL]:** survey line 1460; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `52.85625108205688`; claim: Etot = 52.85625108205688
- **B0080 [PARTIAL]:** survey line 1461; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `7.183953559387109e-17, 5e-15`; claim: divL2 = 7.183953559387109e-17 (atol 5e-15)
- **B0081 [PARTIAL]:** survey line 1462; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.968160239435768`; claim: u-top = 0.968160239435768
- **B0082 [PARTIAL]:** survey line 1463; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-0.9681602394357679`; claim: u-bot = -0.9681602394357679
- **B0083 [PARTIAL]:** survey line 1464; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1.0000000004699001`; claim: mean-shear = 1.0000000004699001
- **B0084 [PARTIAL]:** survey line 1467; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 1e-4, 1e-5, 4`; claim: PCF MHD ('test-pcf-mhd-jax.py'): 'Epert>0', 'Emag>0', 'divL2<1e-4', 'divB-L2<1e-5'; float64-invariant
- **B0085 [PARTIAL]:** survey line 1468; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-12`; claim: 'magnetic-divergence-l2 < 1e-12'.
- **B0086 [VERIFIED]:** survey line 1470; anchors: A0407; local source anchor(s) verified; numbers: `1e-4, 1e-5`; claim: PCF MRI shearpy ('test-pcf-mhd-mri-shearpy-jax.py:6-23'): 'divL2<1e-4', 'divB-L2<1e-5', finite
- **B0087 [PARTIAL]:** survey line 1471; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1.0, 1`; claim: 'alpha/reynolds-xy/maxwell-xy', 'q-shear == 1.0' (at Omega=S=1).
- **B0088 [PARTIAL]:** survey line 1474; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 1, 0, 0.002, 12, 0, 3, 1e-11`; claim: R-1=1,R-2=2,Omega-1=1,Omega-2=0, nu=0.002, N=12, Legendre, m=0, k-z=3, 'rtol=1e-11'): leading
- **B0089 [PARTIAL]:** survey line 1475; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.36073352898670064, 4.8, 0, -22, 5`; claim: eigenvalue 0.36073352898670064 + 4.8times10(-22)i (5 more in source).
- **B0090 [PARTIAL]:** survey line 1478; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 1, 0.5, 1.5, 0.1, 0.001, 12, 0, 3, 1e-11`; claim: 'CircularCouette(1,2,1,0.51.5)', B-0=0.1, nu=eta=0.001, N=12, Legendre, m=0, k-z=3, 'rtol=1e-11'):
- **B0091 [PARTIAL]:** survey line 1479; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.25628761535339467, 1.6, 0, -16`; claim: - conducting leading: 0.25628761535339467 + 1.6times10(-16)i;
- **B0092 [PARTIAL]:** survey line 1480; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.25995005500337837, 5.2, 0, -17`; claim: - insulating leading: 0.25995005500337837 + 5.2times10(-17)i;
- **B0093 [PARTIAL]:** survey line 1481; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `.75, 1e-3, 2, 2, 5, 16, 2e-3`; claim: - local Keplerian-MRI optimum: s-(max)/Omegaapprox0.75 (rel 1e-3), (kv-A)2/Omega2approx15/16 (rel 2e-3).
- **B0094 [PARTIAL]:** survey line 1484; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 1e-7, 1e-7, 3`; claim: ('/p[0,0]/<1e-7'); eigenmode growth-rate matches the linear solver (axisym hydro 'rtol=1e-7'; 3-D hydro/MRI
- **B0095 [PARTIAL]:** survey line 1485; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-6, 100, 4, 1e-11, 1e-18, 4`; claim: 'rtol=1e-6'; 100 steps, x64); pinned-saddle LU residual '<1e-11'; continuity residual '<1e-18' (x64);
- **B0096 [PARTIAL]:** survey line 1486; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-7`; claim: nablacdot u, nablacdot b < 1e-7.
- **B0097 [PARTIAL]:** survey line 1489; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2e-3, 2e-3, 5e-3`; claim: single-step amplitude 'rtol=2e-3', full-state directional 'rtol=2e-3', multi-step finite-amplitude 'rtol=5e-3';
- **B0098 [PARTIAL]:** survey line 1490; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-10`; claim: energy-tangent projection orthogonality 'atol=1e-10'.
- **B0099 [PARTIAL]:** survey line 1493; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `1000, 0, 1.66, 1165.2, 1165.93, 3`; claim: PCF hydro transient growth Re=1000,alpha=0,beta=1.66: literature G=1165.2 vs computed G=1165.93 [RH93];
- **B0100 [PARTIAL]:** survey line 1494; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `0.5, 68.186, 3.167, 3.167, 3`; claim: TC hydro onset eta=0.5 outer-stationary Re-c=68.186 (a-c=3.167, k-(z,c)=3.167) [Taylor23]; ideal local
- **B0101 [PARTIAL]:** survey line 1495; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `0.7500, 2, 2, 0.9373, 0.75, 15, 16, 1`; claim: Keplerian MRI s-(max)/Omega=0.7500, (kv-A)2/Omega2=0.9373 (theory 0.75, 15/16) [BH91]; PCF linearly stable
- **B0102 [PARTIAL]:** survey line 1496; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-2.758, 0, -4`; claim: for all Re (Romanov); insulating MRI scan best growth -2.758times10(-4).
- **B0103 [PARTIAL]:** survey line 1504; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3`; claim: skip in C (F1d/F1e, S2-pipe; §3 gap matrix, §IV hand-off notes). This is the single geometry gap of C relative
- **B0104 [PARTIAL]:** survey line 1538; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1, 6.`; claim: - Family A (the other spectral oracle, same tableaux/Lorentz=1): §I.A.1–I.A.6.
- **B0105 [PARTIAL]:** survey line 1570; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `182, -184, 11`; claim: / Pipe MHD / A (deferred, low parity value; 'PLAN…:182-184') / A — pipe is hydro-only (B-PIPE §11) / A (no pipe) /
- **B0106 [PARTIAL]:** survey line 1576; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10, 6.1, 3`; claim: / GPU / A — CPU/MPI only (A-PCF §10) / P — device-agnostic torch (CUDA in benchmarks; B-MHD §6.1) / P — JAX/XLA; 'cuda13' extra configured ('pyproject.toml') /
- **B0107 [PARTIAL]:** survey line 1580; anchors: A0437, A0438; one or more local source anchors only partially verified; numbers: `4, 28, 2, 4`; claim: / Double precision / P — float64 default / P — complex128 default; float32 validated ('solver.py:87') / P — x64-by-default ('--init--.py:8') /
- **B0108 [PARTIAL]:** survey line 1582; anchors: A0442, A0443, A0444; one or more local source anchors only partially verified; numbers: `1e-12, 1e-4, 8`; claim: / Convergence-order tests / P — golden eig 1e-12, MMS ('OrrSommerfeld-eigs.py:183') / P — OS golden 1e-4, mesh poly-exact deg 8 ('test-mesh.py:21-37') / P — MMS self-asserts ('poisson1D.py:46') /
- **B0109 [PARTIAL]:** survey line 1615; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `+0.00332, 24.7, 4.11`; claim: fixed parameters: conducting gives growth +0.00332 (at mathrm(Rm)=24.7, S=4.11,
- **B0110 [PARTIAL]:** survey line 1639; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4`; claim: (4) Compute/autograd/JIT asymmetry. A is the spectral oracle but has no GPU, no
- **B0111 [PARTIAL]:** survey line 1669; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `122`; claim: Target files (new): 'fn-openpipeflow-122/parity/conventions.md',
- **B0112 [PARTIAL]:** survey line 1674; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `67, -71, 120`; claim: not reinvent eigenvalue matching ('PLAN…:67-71,120'). Gate: the adapter round-trips
- **B0113 [PARTIAL]:** survey line 1709; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `122`; claim: Target files. 'fn-openpipeflow-122/torchchannel/torchchannel/solver.py'
- **B0114 [PARTIAL]:** survey line 1716; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1`; claim: seed a uniform (k=0) mode and assert the epicyclic oscillation SR-1:
- **B0115 [PARTIAL]:** survey line 1723; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2, 5e-4`; claim: to 'rel<1e-2, abs<5e-4' — byte-for-byte the shearpy assertion
- **B0116 [PARTIAL]:** survey line 1735; anchors: A0464; one or more local source anchors only partially verified; numbers: `1e-2`; claim: ('test-theory.py:135-178'). Gate: rate / frequency match to 'rel<1e-2' with an explicit
- **B0117 [PARTIAL]:** survey line 1738; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0.2`; claim: set the override to 1 to match the A/C Alfvén-unit oracle, §0.2).
- **B0118 [PARTIAL]:** survey line 1768; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `none`; claim: E-mathrm(mag) grows monotonically and 'div(B)' stays at roundoff
- **B0119 [PARTIAL]:** survey line 1780; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 0, -3`; claim: DNS growth matches the linear eigenvalue to 2times10(-3)cdot/s/ — the TC precedent
- **B0120 [VERIFIED]:** survey line 1781; anchors: A0475; local source anchor(s) verified; numbers: `4e-8, 4e-7, 2e-6`; claim: ('taylor-couette-notes.md:401-406'; per-m rel-errs 4e-8/4e-7/2e-6 reused as the WS-A
- **B0121 [PARTIAL]:** survey line 1782; anchors: A0476; one or more local source anchors only partially verified; numbers: `none`; claim: target, 'mhd-parity-plan.md:46'); 'div(B)', 'div(u)' at roundoff throughout. Tighten to
- **B0122 [PARTIAL]:** survey line 1783; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, -6`; claim: sim10(-6) after the imposed-field Alfvén coupling is made implicit (WS-G,
- **B0123 [PARTIAL]:** survey line 1801; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.2, 8, 1`; claim: oracle tests (§0.2; B-MHD §8 item 1). (iii) Write 'conventions.md' documenting every
- **B0124 [PARTIAL]:** survey line 1804; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `122`; claim: Target files (new). 'fn-openpipeflow-122/parity/conventions.py',
- **B0125 [PARTIAL]:** survey line 1827; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 8, 1e-10, 1.4e-12`; claim: Q=pi R4 f-z/(8nu) to 'rel<1e-10' (the A precedent: Q rel-err 1.4e-12,
- **B0126 [PARTIAL]:** survey line 1847; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 28, 4`; claim: three already default to float64/complex128/x64; assert this in the harness so a regression
- **B0127 [PARTIAL]:** survey line 1848; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2`; claim: (e.g. an accidental float32 path) is caught — B validates float32 separately with err
- **B0128 [PARTIAL]:** survey line 1864; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 1, 1e-2, 2`; claim: / III.1.a / torch MRI wiring / MRI row S→P (B) / SR-1 epicyclic 'rel<1e-2'; SR-2 shear-winding /
- **B0129 [PARTIAL]:** survey line 1865; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 3, 4, 1e-2, 1`; claim: / III.1.b / torch MHD regressions / (validation) / SR-3 Ohmic, SR-4 Alfvén 'rel<1e-2', prefactor=1 /
- **B0130 [PARTIAL]:** survey line 1867; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 2e-3`; claim: / III.1.d / jax MRI fidelity + WS-A / (validation, A/C) / PCF DNS growth vs linear '2e-3·/s/' /
- **B0131 [PARTIAL]:** survey line 1868; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2`; claim: / III.2 / coordinate/sign unification / cross-family parity (B) / canonical round-trip; 'match-eigenvalues' set-match /
- **B0132 [PARTIAL]:** survey line 1869; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 1e-10, 5e-6`; claim: / III.3 / geometry coverage / pipe hydro A→P (C) / Hagen–Poiseuille 'rel<1e-10'; Womersley '<5e-6' /
- **B0133 [PARTIAL]:** survey line 1883; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0.1, 1`; claim: This part specifies the executable acceptance suite for all three families (A=shenfun, B=torch, C=jax) defined in §1 and §0.1. It is organized into four subsections that mirror the §1 section map:
- **B0134 [PARTIAL]:** survey line 1885; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2`; claim: - IV.1 Foundational tests ('F1'–'F8') — the common floor every family must satisfy: laminar base-flow profiles, the divergence-free / 'div(B)' identities, conservation / energy-balance closure, discrete symmetries, the Orr–Sommerfeld…
- **B0135 [PARTIAL]:** survey line 1886; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2`; claim: - IV.2 Temporal discretization-order tests ('T1'–'T4') — per-integrator 'Δt'-halving slope extraction plus the IMEX splitting-error probe.
- **B0136 [PARTIAL]:** survey line 1887; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3`; claim: - IV.3 Spatial discretization-order tests ('S1'–'S3') — MMS recipe; spectral exponential (A,C) vs FD ~algebraic (B); the cross-family floor-meeting parity definition.
- **B0137 [PARTIAL]:** survey line 1888; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 1, 9`; claim: - IV.4 Shear / rotation / MHD regime tests ('SR-1'…'SR-9') — the headline physics: epicyclic frequency, RDT shear-winding, Ohmic decay, Alfvén wave, ideal MRI dispersion, wall-bounded MRI conducting vs insulating, the 'Pm'-scan, the ma…
- **B0138 [PARTIAL]:** survey line 1890; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 1., 4, 3`; claim: Which tests form the "foundational floor." The floor that MUST exist in all three families after closure is: 'F1a' (Couette 'U=y'), 'F1b' (Poiseuille '1−x²'), 'F1c' (Taylor–Couette 'Ar+B/r'), 'F2' (div-free, and 'div(B)' where MHD…
- **B0139 [PARTIAL]:** survey line 1896; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-10, 1e-21, 1e-7`; claim: / Within-family operator identity / div-free, 'div(B)', symmetry residual, Robin-BC satisfaction / roundoff: '1e-10…1e-21' (spectral A/C), '1e-7' (B pinv cleanup) /
- **B0140 [PARTIAL]:** survey line 1897; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `none`; claim: / Cross-family physical observable / any A↔B↔C comparison of growth rate / energy / eigenvalue / truncation band 'max(C·Δxp, C·Δtq, ε-(spectral))' — never roundoff /
- **B0141 [PARTIAL]:** survey line 1898; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.75, 1e-2, 1e-3, 1e-4`; claim: / Closed-form physics oracle / epicyclic κ, Alfvén ω, Ohmic rate, ideal MRI '0.75Ω' / 'rel<1e-2' (tighten to '1e-3/1e-4' within-family) /
- **B0142 [VERIFIED]:** survey line 1900; anchors: A0484; local source anchor(s) verified; numbers: `0.51, 1e-2`; claim: The cross-family band derivation is fixed by 'PLAN-openpipeflow-vs-fnshenfun.md:57-61': "Tolerances for any cross-family comparison are derived as 'max(C·Δx-FD⁴, C·Δt-FD², ε-spectral)' from the actual grid — never roundoff." Because Fami…
- **B0143 [PARTIAL]:** survey line 1906; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: p = polyfit(log(dt-or-h), log(error), 1).slope
- **B0144 [PARTIAL]:** survey line 1909; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `2, 2, 4, 2`; claim: p = log( (u[dt] - u[dt/2]) / (u[dt/2] - u[dt/4]) ) / log(2)
- **B0145 [VERIFIED]:** survey line 1912; anchors: A0485; local source anchor(s) verified; numbers: `1, 2, 4, 2`; claim: Report both the least-squares slope over the full ladder and the successive pairwise orders 'p-i = log(E-i/E-(i+1))/log 2'; a non-monotone pairwise sequence flags either the roundoff floor (drop coarse points) or a pre-asymptotic regim…
- **B0146 [VERIFIED]:** survey line 1918; anchors: A0486; local source anchor(s) verified; numbers: `0.2, 1e-10, 1e-12, 4, 9, 4, 8, 0.2, 2, 0.2`; claim: Notation: 'x' = wall-normal/radial (Dirichlet walls), 'y' = streamwise/azimuthal, 'z' = spanwise/axial (rotation axis), per the §0.2 canonical frame. 'Δx' = wall-normal/radial grid spacing; 'p' = spatial order (spectral ⇒ 'ε-spectral ≈ 1e-…
- **B0147 [PARTIAL]:** survey line 1926; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-1, 1, 1, 500, 1, 0, 9, 8, 8, 9, 1, 1`; claim: Setup. Channel 'x∈[-1,1]' wall-normal, periodic 'y,z'. No-slip moving walls 'u(±1)=±U-wall·e-y'. 'Re=500', 'U-wall=1'. IC 'u'=0'. Resolution A/C 'N=(9,8,8)', B 'N=9,K=1,M=1'. 'dt=0.01'.
- **B0148 [VERIFIED]:** survey line 1927; anchors: A0487; local source anchor(s) verified; numbers: `1`; claim: - Family A: 'U-b = +U-wall·x·e-y', 'dU-b/dx = U-wall' ('pcf-fluctuations-corrected.py:130-135'); streamwise = axis 1 ('y').
- **B0149 [VERIFIED]:** survey line 1928; anchors: A0488; local source anchor(s) verified; numbers: `-1, 1`; claim: - Family B: 'U=y.clone(), Up=ones, Upp=zeros, walls=(-1,1)' ('base-flow.py:37-41'); streamwise = 'x' (swapped — apply adapter).
- **B0150 [VERIFIED]:** survey line 1931; anchors: A0490; local source anchor(s) verified; numbers: `1, 0, 1, 0, 1973, 440`; claim: Oracle. 'U(x)=x' exactly ('U'=1', 'U''=0'); wall values '±1'. Plane Couette is the canonical linearly-stable base flow for all 'Re' [Nagata90]; Romanov (1973) ('couette-linear-benchmarks.md:84-88, 440').
- **B0151 [PARTIAL]:** survey line 1934; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-13, 1`; claim: - Base-flow identity (roundoff): 'max-j/U(x-j) − x-j/ < 1e-13' for A/C (degree-1 polynomial ∈ trial space exactly; B stores the mesh 'y' exactly). Roundoff is justified because 'U=σx' lies in every family's trial space exactly.
- **B0152 [VERIFIED]:** survey line 1935; anchors: A0491, A0492; local source anchor(s) verified; numbers: `1e-20, 0, 0, 1e-20, 2000`; claim: - Fixed-point check: 'E-pert < 1e-20' after one step from 'u'=0' (C reports 'E-pert=0' exactly, 'test-pcf-fluctuations-jax.py:48-53'; B 'perturbation-energy < 1e-20' after 2000 steps, 'test-step-decay.py:29-51').
- **B0153 [VERIFIED]:** survey line 1936; anchors: A0493; local source anchor(s) verified; numbers: `1.0000000004699001, 1e-8, 1e-2, 1, 1.0, 1e-10`; claim: - mean-shear cross-check: 'mean-shear ≈ σ'. C golden 'mean-shear = 1.0000000004699001' after one step; within-family 'rel<1e-8', cross-family 'rel<1e-2'. Sign note: in the MRI/shearpy convention 'σ = −S', so the shearpy diagnostic re…
- **B0154 [PARTIAL]:** survey line 1944; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-1, 1, 1, 0, 2, 1, 128, 32, 4, 33`; claim: Setup. Channel 'x∈[-1,1]', stationary no-slip walls 'u(±1)=0'. Mean gradient 'dp/dy = −2/Re' (A) imposing the parabola. IC 'u=1−x²' (full) or zero fluctuation. 'N≈(128,32,4)' for the OS validation case, 'N=(33,…)' for the profile check.
- **B0155 [VERIFIED]:** survey line 1945; anchors: A0494; local source anchor(s) verified; numbers: `1, 2`; claim: - Family A: 'U-b=(1−x²)e-y', 'dpdy=−2/Re' ('OrrSommerfeld.py:14,30').
- **B0156 [PARTIAL]:** survey line 1946; anchors: A0495, A0496; one or more local source anchors only partially verified; numbers: `1, 2, 2, 0, 0, 4, 3`; claim: - Family B: 'U=1−y², Up=−2y, Upp=−2, walls=(0,0)', const-flux target '4/3' ('base-flow.py:42-46'; '-flux-target' 'solver.py:621-626').
- **B0157 [PARTIAL]:** survey line 1949; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `1, 0, 1, 1, 0, 2, -1, 1, 1, 4, 3, 1`; claim: Oracle. 'U(x)=1−x²', centerline 'U(0)=1', walls 'U(±1)=0', 'U''=−2', flux '∫-(-1)(1)(1−x²)dx = 4/3' [Orszag71].
- **B0158 [PARTIAL]:** survey line 1952; anchors: A0497; one or more local source anchors only partially verified; numbers: `1, 1e-12, 2, 1e-7, 9, 8`; claim: - Profile: 'max-j/U(x-j) − (1−x-j²)/ < 1e-12' for A/C (degree-2 ∈ trial space); '< 1e-7' for B (9-point FD polynomial-exact through degree 8, 'test-mesh.py:21-37').
- **B0159 [VERIFIED]:** survey line 1953; anchors: A0498; local source anchor(s) verified; numbers: `4, 3, 1e-12`; claim: - Constant-flux oracle (B): '/flux − 4/3/ < 1e-12' ('test-step-decay.py:108-126').
- **B0160 [PARTIAL]:** survey line 1954; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 1e-10, 1e-7`; claim: - Curvature: 'max-j/U''(x-j) − (−2)/ < 1e-10' (A/C), '< 1e-7' (B).
- **B0161 [PARTIAL]:** survey line 1956; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-7`; claim: Families. All three; cross-family compare centerline value and flux (formulation-independent), band ≈ '1e-7' (B FD-limited).
- **B0162 [PARTIAL]:** survey line 1960; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: Description & rationale. Circular-Couette azimuthal profile is the exact steady annular solution; validates the cylindrical '1/r' metric and the wall-rotation BCs.
- **B0163 [PARTIAL]:** survey line 1962; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 1, 2, 1, 1, 2, 0, 0.5, 0.868, 200, 200`; claim: Setup. Annulus 'r∈[R1,R2]', no-slip 'U-θ(R1)=Ω1 R1', 'U-θ(R2)=Ω2 R2'. Canonical A case 'R1=1,R2=2,Ω1=1,Ω2=0' ('η=0.5'); B default 'η=0.868, Re-i=200, Re-o=−200'. IC zero fluctuation. 'Nr=12…48'.
- **B0164 [VERIFIED]:** survey line 1963; anchors: A0499; local source anchor(s) verified; numbers: `2, 1, 1, 2`; claim: - Family A: 'V(r)=ar+b/r', 'a=(Ω2 R2²−Ω1 R1²)/(R2²−R1²)', 'b=(Ω1−Ω2)R1²R2²/(R2²−R1²)' ('taylor-couette-linear.py:89-91').
- **B0165 [VERIFIED]:** survey line 1964; anchors: A0500; local source anchor(s) verified; numbers: `1, 1, 1`; claim: - Family B: 'u-θ=ar+b/r', 'a=(Re-o−η Re-i)/(1+η)', 'b=η(Re-i−η Re-o)/((1−η)(1−η²))' ('base-flow.py:18-30').
- **B0166 [VERIFIED]:** survey line 1967; anchors: A0502; local source anchor(s) verified; numbers: `1, 3, 4, 3, 3, 4, 3, 1, 1, 1, 2, 0`; claim: Oracle. For the canonical A case 'a=−1/3', 'b=4/3' ⇒ 'V(r)=−r/3 + 4/(3r)'; check 'V(1)=1=Ω1 R1', 'V(2)=0=Ω2 R2'. The hydro onset for this case is 'Re-c=68.18635', 'kz-c=3.1667' ('couette-linear-benchmarks.md:29,227'), available as a st…
- **B0167 [PARTIAL]:** survey line 1970; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1e-13, 2, 1e-13, 1e-13`; claim: - Wall values (roundoff): '/V(R1)−Ω1 R1/ < 1e-13', '/V(R2)−Ω2 R2/ < 1e-13' ('atol=1e-13'-class assertion, 'test-taylor-couette.py').
- **B0168 [PARTIAL]:** survey line 1971; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 2, 2, 1e-12`; claim: - Constant-shear identity: '2Ω(r) + r Ω'(r) = 2a' (constant); 'max-r/2Ω + rΩ' − 2a/ < 1e-12'.
- **B0169 [PARTIAL]:** survey line 1972; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-4, 0.0`; claim: - Laminar fixed point: zero perturbation stays zero, 'div-linf < 1e-4' (B); 'energy()==0.0' (A).
- **B0170 [PARTIAL]:** survey line 1973; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 1.0, 1e-8`; claim: - Torque/Nusselt oracle (B): laminar 'τ-lam = −2b/r²', 'Nu-i=Nu-o≈1.0' within 'atol=1e-8'.
- **B0171 [PARTIAL]:** survey line 1975; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1., 2`; claim: Families. All three. A/C use explicit-'1/r' plain-measure (NOT curvilinear shenfun for TC, §1.A.2); B uses cylindrical FD. Base sign 'U-base = +V(r)e-θ'. Cross-family sample 'U-θ(r)' at common radii, band 'max(C·Δx⁴, ε-spectral)'.
- **B0172 [PARTIAL]:** survey line 1979; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0, 1., 3`; claim: Description & rationale. Steady axisymmetric pipe flow under uniform axial forcing — the exact parabola with maximum on the axis, which critically validates axis regularity at 'r=0' (a naive 'u-z(0)=0' BC is fatal, §1.A.3).
- **B0173 [PARTIAL]:** survey line 1981; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1, 0, 32, 8, 8, 64, 18, 32`; claim: Setup. Pipe 'r∈[0,R]', 'R=1', periodic 'z', no-slip 'u-z(R)=0', regular axis. 'Nr=32, Nθ=8, Nz=8' (A); 'N=64,K=18,M=32' (B).
- **B0174 [VERIFIED]:** survey line 1982; anchors: A0503; local source anchor(s) verified; numbers: `4, 8, 0`; claim: - Family A: 'u-z(r)=(f-z/(4ν))(R²−r²)', 'Q=πR⁴f-z/(8ν)' ('pipe-flow-dns.py:473-475'); axis via unified 'bc=(None,0)'.
- **B0175 [VERIFIED]:** survey line 1983; anchors: A0504; local source anchor(s) verified; numbers: `1, 2, 2`; claim: - Family B: 'U(r)=1−r²', 'U'=−2r', '-b-hpf=2r' ('solver.py:200-203'); axis via parity folding (negative-radius ghosts).
- **B0176 [PARTIAL]:** survey line 1984; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 4`; claim: - Family C: N/A — no pipe (§1.C.4); record as 'skip'.
- **B0177 [PARTIAL]:** survey line 1986; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `8, 64, 7`; claim: Oracle. 'u-z(r) ∝ R²−r²'; flow rate 'Q=πR⁴f-z/(8ν)'; laminar Darcy friction 'f=64/Re'; linearly stable for all Re [EBHW07].
- **B0178 [VERIFIED]:** survey line 1989; anchors: A0505; local source anchor(s) verified; numbers: `1e-6`; claim: - Profile: 'max/u-z − exact/ < 1e-6' (A golden, 'test-pipe-flow-dns.py:33-37').
- **B0179 [VERIFIED]:** survey line 1990; anchors: A0506; local source anchor(s) verified; numbers: `8, 1e-10, 1.4e-12, 1e-8`; claim: - Flow rate (tight oracle): '/Q − πR⁴f-z/(8ν)//Q < 1e-10' (A; notes report rel.err '1.4e-12', 'pipe-flow-notes.md:67'). B drives mean axial flux to '<1e-8' ('test-invariants.py').
- **B0180 [PARTIAL]:** survey line 1991; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-10, 1e-8`; claim: - Divergence: 'div-l2 < 1e-10' (A), '< 1e-8' (B).
- **B0181 [PARTIAL]:** survey line 1993; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 3`; claim: Families. A & B only. A uses curvilinear '√g=r' weighting (do NOT re-multiply by 'r', §1.A.3); B uses parity folding. Cross-family compare centerline velocity and 'Q'.
- **B0182 [PARTIAL]:** survey line 1999; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1, 3, 9, 2, 9, 0.698, 32`; claim: Setup. Pipe 'r∈[0,1]', forcing '−dp/dz = K cos(ωt)'. Womersley number 'α-W=R√(ω/ν)'; A's test uses 'α-W=3', 'ω=9', period '2π/9≈0.698'. 'Nr≈32', 'dt' small so '(ωΔt)²' is below tolerance.
- **B0183 [VERIFIED]:** survey line 2000; anchors: A0507; local source anchor(s) verified; numbers: `1, 3, 2, 3, 2, 1, 3, 2, 4`; claim: - Family A: 'u-z(r,t)=Re((K/(iρω))[1 − J-0(i(3/2)α-W r/R)/J-0(i(3/2)α-W)] e(iωt))', 'ρ=1', 'i(3/2)=e(i3π/4)' ('pipe-flow-dns.py:478-490').
- **B0184 [PARTIAL]:** survey line 2004; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `5`; claim: Oracle. The Womersley Bessel solution above [Womersley55].
- **B0185 [VERIFIED]:** survey line 2006; anchors: A0508; local source anchor(s) verified; numbers: `5e-6, 8e-7, 6e-6, 2, 2`; claim: Metric & tolerance. 'max/u-z − exact/ < 5e-6' over a full period (A golden; notes report '8e-7', rel '6e-6', 'pipe-flow-notes.md:69'). Temporal error scales '~(ωΔt)²' (CNAB2, 2nd order); halving 'Δt' confirms slope ≈ 2 (see F8). This i…
- **B0186 [PARTIAL]:** survey line 2010; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0`; claim: Description & rationale. Incompressibility '∇·u=0' (and solenoidality '∇·B=0') must hold to within-family roundoff. This is a pure operator/projection identity, so the tolerance is roundoff — the single most diagnostic foundational t…
- **B0187 [PARTIAL]:** survey line 2013; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1., 1, 0, 1., 4, 0`; claim: - Family A: KMM eliminates pressure exactly, recovers 'v,w' enforcing 'div(u)=0' (§1.A.1); MHD 'B=curl(A)' ⇒ 'div(B)=div(curl A)=0' by the discrete identity (§1.A.4); TC/pipe saddle-point enforces 'div(u)=0'.
- **B0188 [PARTIAL]:** survey line 2014; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 3, 0, 0`; claim: - Family B: influence-matrix + projection + dense 'enforce-constraints' pinv cleanup (§1.B.3); MHD induced 'b=0' walls + same pinv for 'div(b)=0'.
- **B0189 [PARTIAL]:** survey line 2017; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4.`; claim: Oracle. Exactly zero (operator identity); achievable floor = roundoff at float64.
- **B0190 [PARTIAL]:** survey line 2019; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `none`; claim: Metric & tolerance (per-family roundoff — verbatim goldens).
- **B0191 [VERIFIED]:** survey line 2023; anchors: A0509; local source anchor(s) verified; numbers: `8, 8, 8, 9.41e-17, 3.05e-21`; claim: / A / PCF MHD (Legendre 'N=(8,8,8)') / 'divU L2' / 'divB L2' / '9.41e-17' / '3.05e-21' / 'pcf-mhd-divfree-notes.md:69-73' /
- **B0192 [PARTIAL]:** survey line 2024; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `16, 16, 16, 9.03e-17, 4.71e-21, 1., 4`; claim: / A / PCF MHD (Cheb 'N=(16,16,16)') / 'divU L2' / 'divB L2' / '9.03e-17' / '4.71e-21' / §1.A.4 /
- **B0193 [PARTIAL]:** survey line 2025; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2.84e-16, 8.32e-16, 1., 4`; claim: / A / PCF MHD near-transition / 'divU L2' / 'divB rel RMS' / '2.84e-16' / '8.32e-16' / §1.A.4 /
- **B0194 [VERIFIED]:** survey line 2026; anchors: A0510; local source anchor(s) verified; numbers: `8, 8, 8, 1e-12`; claim: / A / shearpy MRI (Legendre 'N=(8,8,8)') / 'divb-l2', 'divu-l2' / '< 1e-12' (gate) / 'test-pcf-mhd-mri-shearpy.py:61-62' /
- **B0195 [PARTIAL]:** survey line 2027; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-9`; claim: / A / TC DNS / 'div-linf' / '< 1e-9·umax' / 'test-taylor-couette-dns.py' /
- **B0196 [PARTIAL]:** survey line 2028; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-13, 1e-11, 1e-9, 1., 3`; claim: / A / Pipe / 'divergence-l2' / '1e-13…1e-11'; gate '<1e-9' / §1.A.3 /
- **B0197 [VERIFIED]:** survey line 2029; anchors: A0511; local source anchor(s) verified; numbers: `1e-12`; claim: / B / Channel / 'divergence-norm(include-walls)' / '< 1e-12' / 'test-step-decay.py:29-51' /
- **B0198 [PARTIAL]:** survey line 2030; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-7`; claim: / B / Channel MHD / 'divLinf, divB-Linf, divB-L2' / '< 1e-7' / 'test-mhd.py' /
- **B0199 [PARTIAL]:** survey line 2031; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-8, 1., 5`; claim: / B / Couette MHD / 'div(velocity), divB-Linf' / '< 1e-8' / §1.B.5 /
- **B0200 [PARTIAL]:** survey line 2032; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-6`; claim: / B / Pipe / 'div-linf' / '< 1e-6' / 'test-invariants.py' /
- **B0201 [VERIFIED]:** survey line 2033; anchors: A0512; local source anchor(s) verified; numbers: `7.18e-17, 5e-15`; claim: / C / PCF / 'divL2' / '7.18e-17' (gate '<5e-15') / 'test-pcf-fluctuations-jax.py:78-95' /
- **B0202 [PARTIAL]:** survey line 2034; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-4, 1e-5, 4, 1e-12, 1., 3`; claim: / C / PCF MHD / 'divL2'/'divB-L2' / '<1e-4'/'<1e-5'; x64 'divB<1e-12' / §1.C.3 /
- **B0203 [PARTIAL]:** survey line 2035; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-18, 4, 1., 2`; claim: / C / TC DNS / continuity residual / '< 1e-18' (x64) / §1.C.2 /
- **B0204 [PARTIAL]:** survey line 2037; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-16, 1e-21, 1e-21, 1e-7, 1e-12, 1., 2`; claim: Tolerance rationale. This is a within-family operator identity → roundoff. Spectral families (A,C) reach '1e-16…1e-21' because the divergence operator and the compatible-space chain are exact; the 'div(B)' floor for A is spectacular…
- **B0205 [PARTIAL]:** survey line 2041; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0`; claim: Description & rationale. The kinetic-energy budget 'dE/dt = P − D' (production minus dissipation) must close to ≈0, and decaying cases must decay monotonically. Validates that the discrete nonlinear term conserves energy (no spurious p…
- **B0206 [VERIFIED]:** survey line 2045; anchors: A0513; local source anchor(s) verified; numbers: `1e12, 1e-4, 50, 1e-6, 1e-8, 1e-6`; claim: F3a — Inviscid energy conservation (high-Re limit). Channel, 'Re=1e12', rotational nonlinear form, 'dt=1e-4', 50 steps. Golden (B): 'worst-rel < 1e-6', 'div < 1e-8' ('test-step-decay.py:246-271'). Assert relative energy drift '< 1e-6'.
- **B0207 [VERIFIED]:** survey line 2048; anchors: A0514; local source anchor(s) verified; numbers: `1, 0.178, 5`; claim: - A pipe Stokes 'm=1': strictly monotonic, asymptotic rate '0.178', plateau '<5%' ('pipe-flow-notes.md:71').
- **B0208 [PARTIAL]:** survey line 2049; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 01, 5.78319, 5.78320, 2.8e-6, 1., 3, 01, 01, 1e-4`; claim: - A pipe Bessel mode: 'E~e(−2λt)', 'rate=−log(E1/E0)/(2(t1−t0))', oracle 'λ = ν j-(01)²/R² = 5.78319', measured '5.78320', rel.err '2.8e-6' (§1.A.3). Assert '/rate − ν j-(01)²/R²//(ν j-(01)²/R²) < 1e-4'.
- **B0209 [VERIFIED]:** survey line 2050; anchors: A0515; local source anchor(s) verified; numbers: `0`; claim: - B channel MHD magnetic diffusion: '0 < energy1 < energy0' ('test-mhd.py:163-180').
- **B0210 [PARTIAL]:** survey line 2051; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 2`; claim: - C: TC/PCF decaying-mode energy decreases (§1.C.2).
- **B0211 [PARTIAL]:** survey line 2055; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 2, 01`; claim: Oracle. F3a 'dE/dt=0' at 'ν→0'; F3b 'E(t)=E-0 e(−2λt)' ('λ=ν j-(01)²/R²' Bessel); F3c the Reynolds–Orr equation 'dE/dt = P − D'.
- **B0212 [PARTIAL]:** survey line 2058; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 1e-6`; claim: - F3a: '/E(t)−E(0)//E(0) < 1e-6' (within-family energy-conservation property).
- **B0213 [PARTIAL]:** survey line 2059; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1e-4, 2.8e-6`; claim: - F3b: monotonicity is a hard boolean (every step 'E-(n+1)<E-n'); rate match is an analytic oracle → 'rel < 1e-4' (A Bessel golden '2.8e-6').
- **B0214 [PARTIAL]:** survey line 2060; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-8, 1e-2, 1`; claim: - F3c: closure '/dE/dt − (P−D)//E <' truncation band 'max(C·Δxp, C·Δtq, ε-spectral)' — '~1e-8' for spectral A/C; '~1e-2' for B (θ-method, 'q≈1').
- **B0215 [PARTIAL]:** survey line 2062; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-4, 1, 2`; claim: Tolerance rationale. F3a/F3b monotonicity = structural property → tight; F3b rate = analytic oracle → 'rel<1e-4'; F3c carries a temporal component (the FD of 'dE/dt') and so sits in the truncation band, dominated by 'Δtq' ('q=1' for…
- **B0216 [PARTIAL]:** survey line 2064; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e12`; claim: Families. All three. F3a best demonstrated in B ('Re=1e12'); F3b uses pipe Bessel (A/B) or Stokes mode (all). Energy norm: A/C use the spectral mass-matrix / quadrature inner product (curvilinear 'r'-weight for TC/pipe); B uses 'intrdr…
- **B0217 [VERIFIED]:** survey line 2073; anchors: A0516; local source anchor(s) verified; numbers: `0, 0, 1e-12, 0, 1., 2, 1e-12`; claim: - C: mean '(0,0)' modes forced real after a step, 'imag < 1e-12' ('test-pcf-fluctuations-jax.py:56-72'). B: 'enforce-mean-mode-cleanup' keeps mean 'v=0', mean 'u,w' real (§1.B.2). Assert symmetry residual '< 1e-12'.
- **B0218 [PARTIAL]:** survey line 2076; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-9, 1e-9, 1., 2`; claim: - A golden: 'test-hydro-nonaxisymmetric-mirror-symmetry' to 'atol=1e-9'; assert 'max/q(−m) − conj(q(m))/ < 1e-9'. C evaluates the complex eigenvector real/imag separately to preserve this (§1.C.2).
- **B0219 [PARTIAL]:** survey line 2078; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0, 0, 0`; claim: F4c — Pipe: Hermitian/parity & rotational symmetry. Real field ⇒ 'f(−k,0)=conj(f(k,0))', 'f(0,0)' real; azimuthal parity folding at 'r=0'; optional 'm-p'-fold rotation.
- **B0220 [PARTIAL]:** survey line 2079; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0, 0, 1., 4, 1e-12`; claim: - B golden: 'enforce-m0-reality' keeps 'f(−k,0)=conj(f(k,0))', 'f(0,0)' real, 'm0-hermitian-residual < ' roundoff; the discrete symmetries 'mirror-z', 'shift-reflect', 'shift-rotate' (keeps 'k+m' even) are ported verbatim from OpenPipeFlow…
- **B0221 [PARTIAL]:** survey line 2081; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-9, 1e-12`; claim: Metric & tolerance. Within-family roundoff (symmetry is an exact equivariance of the discrete operators when correctly implemented): '1e-9' for TC 'm↔−m' (accumulated arithmetic in dense per-mode blocks), '1e-12' for the FFT-based real…
- **B0222 [PARTIAL]:** survey line 2089; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: Setup. Channel, Poiseuille base 'U=1−x²'. Operating points:
- **B0223 [PARTIAL]:** survey line 2090; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10000, 1, 0, 8000, 1, 0, 0`; claim: - Orszag point: 'Re=10000, α=1, β=0' (or 'Re=8000, α=1' for A's golden). Clamped OS BC 'v=v'=0' at both walls; Squire 'η=0'.
- **B0224 [PARTIAL]:** survey line 2091; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `80, 101, 4`; claim: - A 'N>80' (Chebyshev biharmonic, 'quad='GC''); B 'N=101, KL=4'; C analogous.
- **B0225 [VERIFIED]:** survey line 2095; anchors: A0517; local source anchor(s) verified; numbers: `8000, 1, 0.24707506017508621, 0.0026644103710965817, 1e-12`; claim: - A, 'Re=8000, α=1': leading 'c = 0.24707506017508621 + 0.0026644103710965817 i' (positive Im → unstable), tolerance '1e-12' ('OrrSommerfeld-eigs.py:183-184').
- **B0226 [VERIFIED]:** survey line 2096; anchors: A0518; local source anchor(s) verified; numbers: `10000, 1, 0, 0.23752649, 0.00373967, 0.23752722198590992, 0.0037381198835812705, 1e-4`; claim: - B, 'Re=10000, α=1, β=0': reference 'c-ref = 0.23752649 + 0.00373967 i'; computed '0.23752722198590992 + 0.0037381198835812705 i', abs error '< 1e-4' ('test-linstab-poiseuille.py:7-19').
- **B0227 [VERIFIED]:** survey line 2097; anchors: A0519; local source anchor(s) verified; numbers: `5772.22, 1.02056, 0.26400, 1, 5742.22, 1.02, 5802.22`; claim: - Published critical Re (Poiseuille): 'Re-crit = 5772.22', 'α-crit ≈ 1.02056', 'c ≈ 0.26400' [Orszag71]. B verifies the sign change: stable at 'Re=5742.22, α=1.02', unstable at 'Re=5802.22' ('test-linstab-poiseuille.py:22-39').
- **B0228 [VERIFIED]:** survey line 2098; anchors: A0520; local source anchor(s) verified; numbers: `1973`; claim: - Plane Couette: linearly stable for all Re — no unstable OS eigenvalue (Romanov 1973, 'couette-linear-benchmarks.md:84-88').
- **B0229 [PARTIAL]:** survey line 2101; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-12, 1e-4`; claim: - Within-family golden: A '/c − golden/ < 1e-12'; B '/c − c-ref/ < 1e-4'. Family-specific because the published reference itself was computed by one method.
- **B0230 [PARTIAL]:** survey line 2102; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 5772.22, 30, 5e-3, 5742, 5802, 30.`; claim: - Cross-family / published oracle: assert each family's neutral curve crosses 'Re(λ)=0' at 'Re=5772.22 ± 30' ('rel<5e-3'); B's flanking points (5742/5802) bracket it within ±30.
- **B0231 [VERIFIED]:** survey line 2103; anchors: A0521, A0522, A0523; local source anchor(s) verified; numbers: `0, 1, 0, 1.179054e-01, 0, 1, 3.467401e-03, 2, 1, 1.905757e-01, 2, 1.0`; claim: - Couette stability: assert 'max Re(λ) < 0' for all tested '(α,β)'. Golden Romanov rates ('couette-linear-benchmarks.md:111-118'): 'ky=1,kz=0: −1.179054e-01'; 'ky=0,kz=1: −3.467401e-03'; 'ky=2,kz=1: −1.905757e-01'. Streamwise-roll analyt…
- **B0232 [PARTIAL]:** survey line 2105; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-12, 1e-4, 9, 5772.22, 1e-2, 2, 1e-9`; claim: Tolerance rationale. The within-family golden is asserted at the precision the family achieves (A spectral '1e-12'; B FD '1e-4', limited by the 9-point stencil truncation). The cross-family 'Re-crit=5772.22' uses 'rel<1e-2' because the…
- **B0233 [VERIFIED]:** survey line 2107; anchors: A0524; local source anchor(s) verified; numbers: `3, 0, 1e8, 1e6`; claim: Families. All three (channel/Couette). A uses Shen biharmonic Galerkin 'Aφ = c Bφ'; B uses OS/Squire primitive FD blocks with clamped rows; C uses jaxfun assembly. The Squire coupling '−iβU'' must be present for 3D ('β≠0'). Rank eigenv…
- **B0234 [PARTIAL]:** survey line 2111; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2`; claim: Description & rationale. The 2D Taylor–Green vortex is an exact unsteady solution of incompressible NS on a doubly-periodic box: pure exponential viscous decay with no change in spatial structure. Because the spatial modes are exactl…
- **B0235 [PARTIAL]:** survey line 2113; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 2, 1, 0, 16, 16, 2, 4`; claim: Setup. Doubly-periodic '[0,2π]²' (or '[−π,π]²'), 'ν=1/Re'. IC = exact TG field at 't=0', integrate to 't=T', compare. Resolution modest ('16×16' Fourier — TG modes exactly resolved). 'dt' ladder '(Δt, Δt/2, Δt/4)' for order extraction.
- **B0236 [PARTIAL]:** survey line 2115; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `7`; claim: Oracle (verbatim formula, [TaylorGreen37]).
- **B0237 [PARTIAL]:** survey line 2120; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 2, 2`; claim: p(x,y,t) = (ρ/4)·(cos(2x) + cos(2y)) · F(t)²
- **B0238 [PARTIAL]:** survey line 2121; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-2`; claim: F(t) = exp(-2 ν t) # velocity decay factor
- **B0239 [PARTIAL]:** survey line 2124; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 4, 0.01, 1, 0.02, 10, 0.2, 0.818731, 10, 0, 0.4, 0.670320`; claim: Kinetic energy decays as 'E(t) = E(0)·exp(−4νt)' (twice the velocity rate, 'E ∝ /u/²'). Concrete: 'ν=0.01, k=1': 'F(t)=e(−0.02t)'; at 't=10', 'F=e(−0.2)=0.818731', energy ratio 'E(10)/E(0)=e(−0.4)=0.670320'.
- **B0240 [PARTIAL]:** survey line 2128; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 4, 4, 1e-2`; claim: - Energy-decay oracle: '/E(T)/E(0) − e(−4νT)//e(−4νT) <' the same band (coarse 'dt' → 'rel<1e-2'; fine 'dt' → roundoff).
- **B0241 [PARTIAL]:** survey line 2129; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `none`; claim: - Divergence: '‖∇·u‖ <' roundoff throughout (F2 floor).
- **B0242 [PARTIAL]:** survey line 2130; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 4, 2, 22, 1, 0.51, 1, 1., 2, 0.15`; claim: - Temporal-order slope (headline deliverable): run '(Δt, Δt/2, Δt/4)', fit 'log‖error‖' vs 'log Δt'; assert slope 'q ≈' the scheme's formal order: q=2 for CNAB2 (A-TC/B-couette/C-TC) and IMEXRK222 (A-PCF/C-PCF); q=1 for B's θ-m…
- **B0243 [PARTIAL]:** survey line 2132; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2`; claim: Tolerance rationale. The TG field error has NO spatial truncation contribution for Fourier families → the band collapses to 'max(C·Δtq, ε-spectral)', making it a pure temporal-accuracy probe. The energy rate is an analytic oracle ('re…
- **B0244 [PARTIAL]:** survey line 2134; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `1., 5, 2, 22, 2, 2, 1, 2, 3, 1600, 3`; claim: Families. All three should add this as a periodic-box verification. Adaptation for wall-bounded codes: run TG in the two periodic directions with a trivial wall-normal dependence, or preferably as a standalone Fourier-only verifica…
- **B0245 [PARTIAL]:** survey line 2138; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 3`; claim: Not a standalone test but a mandatory invariant harness running inside 'F1'–'F6': (1) divergence-free (F2) always; (2) energy/enstrophy budget (F3) — inviscid → roundoff, viscous → matches resolved dissipation; (3) symmetry residual (F4) —…
- **B0246 [PARTIAL]:** survey line 2142; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 3.`; claim: Used by F1e (temporal), F3b (decay rate), F5 (eigenvalue refinement), F6 (temporal slope), and reused by IV.2/IV.3.
- **B0247 [PARTIAL]:** survey line 2144; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-10, 2, 9, 8, 8, 4, 4`; claim: Spatial order. Refine 'N'. Spectral (A,C): straight line on semilog ('error ~ e(−cN)'); assert 'error < 1e-10' at modest 'N' and that it drops '≥2' decades per resolution doubling until the floor. FD (B): straight line on lo…
- **B0248 [PARTIAL]:** survey line 2146; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 4, 2, 2, 4, 2, 0.15`; claim: Temporal order. Fix 'N' fine (spatial error ≪ temporal), run '(Δt, Δt/2, Δt/4)'. Observed order 'q = log[(u-(Δt)−u-(Δt/2))/(u-(Δt/2)−u-(Δt/4))]/log 2' (three-level Richardson, no oracle) OR the 'log‖u−u-exact‖' vs 'log Δt' slope when a…
- **B0249 [VERIFIED]:** survey line 2148; anchors: A0525; local source anchor(s) verified; numbers: `24, 0.4984075630441907, 32, 0.49840694616677383, 48, 0.49840620435392047, 0.5, 2, 3`; claim: Eigenvalue refinement (F5). Increase 'N', confirm the leading eigenvalue converges to the golden monotonically. The Family A MRI pattern ('couette-linear-benchmarks.md:314-316'): 'nx=24 → 0.4984075630441907', 'nx=32 → 0.498406946166773…
- **B0250 [PARTIAL]:** survey line 2150; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `500, 33, 0.01, 1, 1e-2, 8, 1e-2`; claim: Tolerance-band assembly (cross-family). For any cross-family comparison the band is 'max(C·Δxp, C·Δtq, ε-spectral)' evaluated on the actual grid/'dt' of the coarsest family (almost always B, FD + θ-method): at 'Re=500, N~33, dt~0.0…
- **B0251 [PARTIAL]:** survey line 2157; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 4, 3`; claim: / F1b Poiseuille '1−x²' / ✓ / ✓ / ✓ / B uses const-flux '4/3' /
- **B0252 [PARTIAL]:** survey line 2158; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: / F1c TC 'Ar+B/r' / ✓ / ✓ / ✓ / explicit-'1/r' (A/C) vs cyl-FD (B) /
- **B0253 [PARTIAL]:** survey line 2161; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `none`; claim: / F2 div(u), div(B) / ✓ / ✓ / ✓ / per-family roundoff; MHD where present /
- **B0254 [PARTIAL]:** survey line 2162; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e12`; claim: / F3 energy balance / ✓ / ✓ / ✓ / F3a best in B ('Re=1e12') /
- **B0255 [PARTIAL]:** survey line 2164; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-12, 1e-4, 5772.22`; claim: / F5 OS/Squire eig / ✓ / ✓ / ✓ / A golden '1e-12'; B golden '1e-4'; 'Re-c=5772.22' /
- **B0256 [PARTIAL]:** survey line 2165; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2`; claim: / F6 2D Taylor–Green / add / add / add / required/proposed; no target-tree harness found /
- **B0257 [VERIFIED]:** survey line 2173; anchors: A0526; local source anchor(s) verified; numbers: `11, 222, 3, 443, 1., 6, 1., 5, 0.51, 1., 2, 4`; claim: The integrators differ by construction (IMEXRK111/222/3/443 + CNAB2 in A/C, §1.A.6/§1.C.5; single-stage θ-method PC at 'implicit=0.51' in B, §1.B.2), so these tests are tailored per family — that tailoring is the point. The shared proced…
- **B0258 [PARTIAL]:** survey line 2177; anchors: A0527, A0528, A0529, A0530; one or more local source anchors only partially verified; numbers: `1, 1, 2, 1e-5, 1e-8`; claim: Description & rationale. The cleanest temporal oracle is one Fourier mode under linear diffusion: '∂u/∂t = ν ∂²u/∂x²' for 'u=û(t)e(ikx)' gives 'dû/dt = −νk² û', exact 'û(t)=û-0 e(−νk²t)'. This isolates the implicit branch of ever…
- **B0259 [PARTIAL]:** survey line 2179; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 3, 1, 0.01, 0.01, 0, 1, 1, 1.0, 16, 32, 0`; claim: Setup. Periodic 1-D (or a periodic direction of the 3-D box with all other modes zeroed). 'k=1' (mildest stiffness), 'ν=0.01' ('λ=−0.01'). IC 'û(0)=1' in mode 'k=1'. Integrate to 'T=1.0'. 'N-x=16' Fourier; for families needing a wall-n…
- **B0260 [PARTIAL]:** survey line 2181; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.01, 0.990049833749168, 0.01`; claim: Oracle. 'û(T) = e(−νk²T) = e(−0.01) = 0.990049833749168…'; 'E(dt) = /û-num(T) − e(−0.01)/'.
- **B0261 [PARTIAL]:** survey line 2183; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5`; claim: Metric & tolerance (per integrator). Fit 'log E = p·log dt + c' (least squares, 5 points); assert 'p' in the per-integrator band:
- **B0262 [PARTIAL]:** survey line 2187; anchors: A0531, A0532; one or more local source anchors only partially verified; numbers: `11, 1, 0.8, 1.3`; claim: / IMEXRK111 / A, C / 1 / '[0.8, 1.3]' / 'integrators.py:836'; 'imex-rk.py:163' /
- **B0263 [VERIFIED]:** survey line 2188; anchors: A0533; local source anchor(s) verified; numbers: `2, 1.8, 2.3`; claim: / CNAB2 / A (TC/pipe), C (TC) / 2 / '[1.8, 2.3]' / 'taylor-couette-dns.py:288'; 'cnab2.py' /
- **B0264 [VERIFIED]:** survey line 2189; anchors: A0534; local source anchor(s) verified; numbers: `0.51, 1, 0.85, 1.6, 594, -606`; claim: / torch θ-method PC (θ=0.51) / B / ~1 (see note) / '[0.85, 1.6]' / 'solver.py:87, 594-606' /
- **B0265 [PARTIAL]:** survey line 2190; anchors: A0535; one or more local source anchors only partially verified; numbers: `22, 2, 1.8, 2.3`; claim: / IMEXRK222 / A, C / 2 / '[1.8, 2.3]' / 'integrators.py:858'; 'imex-rk.py' /
- **B0266 [PARTIAL]:** survey line 2191; anchors: A0536, A0537; one or more local source anchors only partially verified; numbers: `3, 2.7, 3.3`; claim: / IMEXRK3 (Spalart) / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:665'; 'imex-rk.py:96' /
- **B0267 [PARTIAL]:** survey line 2192; anchors: A0538, A0539; one or more local source anchors only partially verified; numbers: `43, 3, 2.7, 3.3`; claim: / IMEXRK443 / A, C / 3 / '[2.7, 3.3]' / 'integrators.py:872'; 'imex-rk.py:184' /
- **B0268 [PARTIAL]:** survey line 2194; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5, 0.15, 0.3, 2, 3, +0.3`; claim: Tolerance rationale. These are within-family order-verification bands, not roundoff: a 5-point log-log fit on a clean linear ODE recovers the asymptotic slope to ≈±0.15 once in the asymptotic regime, so a ±0.3 half-width accepts the tr…
- **B0269 [PARTIAL]:** survey line 2196; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1, 1, 1, 2, 0.5, 0.51, 0.5, 2, 0.85, 1.6, 0.5`; claim: θ-method note. For a scalar linear ODE the θ-method 'û-(n+1) = û-n (1+(1−θ)λdt)/(1−θλdt)' is exactly 2nd-order only at θ=0.5; at the shipped 'θ=0.51' it carries an 'O((θ−0.5)·λ·dt)' first-order term, small over this window so the e…
- **B0270 [PARTIAL]:** survey line 2198; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `11, 222, 3, 443, 1, 0, 0, 2, 2, 2, 1, 1`; claim: Families. A: 'KMM'/PCF subclass, '--timestepper ∈ (IMEXRK111,222,3,443)', seed mode 'k=1' (the '(0,0)' mean path is untouched). C: same tableaux (verified 'imex-rk.py' γ=(2−√2)/2, δ=1−1/(2γ) match A), 'channelflow-kmm' with nonlinear→0…
- **B0271 [PARTIAL]:** survey line 2200; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2`; claim: Extraction. Least-squares slope + successive pairwise orders 'p-i = log(E-i/E-(i+1))/log 2' to confirm monotone approach to the asymptote.
- **B0272 [PARTIAL]:** survey line 2204; anchors: A0540, A0541, A0542; one or more local source anchors only partially verified; numbers: `0, 2, 2, 2, 2, 2`; claim: Description & rationale. T1 exercises only the implicit branch. To verify the explicit branch carrying the rotation/shear source terms, use the epicyclic oscillation. A uniform ('k=0') perturbation with Coriolis + base-shear obeys…
- **B0273 [PARTIAL]:** survey line 2206; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 3, 1, 3, 2, 2, 2, 3, 1, 3, 4, 9`; claim: Setup. Shearing box. Keplerian 'Ω=2/3, S=1' ⇒ 'q=3/2', 'κ²=2·(2/3)·(1/3)=4/9' ⇒ 'κ=2/3' (= Ω, the Keplerian result). Also a non-Keplerian case 'Ω=1, S=1' ⇒ 'κ²=2', 'κ=√2'. Seed only the 'k=0' mode: 'u-(x0)=1e-3', 'u-y=u-z=0'. Lorentz…
- **B0274 [PARTIAL]:** survey line 2208; anchors: A0543; one or more local source anchors only partially verified; numbers: `1.0, 1.5, 2, 1, 2, 1.5, 1, 1, 2, 1.5, 2, 1`; claim: > Shearpy reference parametrization. The shearpy theory test ('test-theory.py:100-132') uses 'omega=1.0, shear=1.5' ⇒ 'κ=√(2·1·(2−1.5))=√1=1', amplitude '(shear−2ω)/κ=(1.5−2)/1=−0.5', integrating to 'T=1.0', asserting 'vx-mean=cos(κT)'…
- **B0275 [PARTIAL]:** survey line 2210; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6, 2`; claim: Oracle. 'u-x(T)=u-(x0)cos(κT)'. For Keplerian, 'κT=6π' ⇒ 'u-x(T)=u-(x0)' exactly. Use the whole-trajectory L2 phase error (more sensitive than the endpoint, which sits at a node): 'E(dt)=√(Σ-n[(u-x(num)(t-n)−u-(x0)cos κt-n)² + (u-y(n…
- **B0276 [PARTIAL]:** survey line 2212; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `22, 1.8, 2.3, 443, 2.7, 3.3, 11, 0.8, 1.3, 1e-2, 1e-2, 5e-4`; claim: Metric & tolerance. Slope of 'log E' vs 'log dt': IMEXRK222 'p ∈ [1.8, 2.3]'; IMEXRK3/443 '[2.7, 3.3]'; IMEXRK111 '[0.8, 1.3]'. Plus a physics-oracle check at the finest 'dt': '/κ-measured − κ-theory//κ-theory < 1e-2', 'κ-measured' fro…
- **B0277 [PARTIAL]:** survey line 2214; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2, 0.3, 5`; claim: Tolerance rationale. The explicit branch carries the source terms, so this catches a mis-wired Coriolis sign or an order-reduced explicit tableau that T1 cannot see. The κ 'rel<1e-2' is a physics tolerance asserting the integrator repr…
- **B0278 [PARTIAL]:** survey line 2216; anchors: A0544, A0545, A0546, A0547, A0548; one or more local source anchors only partially verified; numbers: `0., 2, 2, 1.`; claim: Families. A: native — 'pcf-mhd-mri-shearpy.py:346-348' Coriolis, base shear 'dUb-dx=−S'; set magnetic amplitude 0. C: native — 'pcf-mhd-mri-shearpy-jax.py:130-134' adds 'n-0 −= 2Ω u-1; n-1 += 2Ω u-0'; same κ². B: CONDITIONAL / GATED…
- **B0279 [PARTIAL]:** survey line 2220; anchors: A0549, A0550; one or more local source anchors only partially verified; numbers: `1, 2`; claim: Description & rationale. Resistive diffusion is treated implicitly in all MHD families (shenfun on the vector potential 'SA = MA − dt·γ·η·LA'; torch magnetic Helmholtz with '1/Rm', 'mhd.py:115-117'). A single magnetic Fourier mode with…
- **B0280 [PARTIAL]:** survey line 2222; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0, 1, 1, 100, 0.01, 0.01, 1.0, 1, 20, 1, 320`; claim: Setup. Single magnetic mode 'k=1' in a periodic direction, 'b̂(0)=1'; velocity held zero (no EMF — use the fluctuation form so the base flow contributes none, or set base flow off). 'η=1/Rm', 'Rm=100' ⇒ 'η=0.01', 'λ-mag=−0.01'. 'T=1.0'…
- **B0281 [PARTIAL]:** survey line 2224; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 2, 0.02, 0.980198, 0.01, 0.990049833, 0.01`; claim: Oracle. 'E-mag(T)/E-mag(0)=e(−2ηk²T)=e(−0.02)=0.980198…'; amplitude 'b̂(T)=e(−0.01)=0.990049833…'; 'E(dt)=/b̂-num(T) − e(−0.01)/'.
- **B0282 [PARTIAL]:** survey line 2226; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `22, 1.8, 2.3, 1.8, 2.3, 443, 2.7, 3.3, 0.85, 1.6, 0.51, 1.8`; claim: Metric & tolerance. Slope 'p': IMEXRK222 '[1.8, 2.3]'; CNAB2 (TC MHD A/C) '[1.8, 2.3]'; IMEXRK3/443 '[2.7, 3.3]'; torch θ-method '[0.85, 1.6]' (θ=0.51), '[1.8, 2.3]' (θ=0.5 override). Plus physics check '/η-measured − 0.01//0.01 < 1e-2…
- **B0283 [PARTIAL]:** survey line 2230; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1., 0, 0, 0, 3.`; claim: Families. A: 'pcf-mhd-divfree.py' (vector potential, η implicit). C: 'pcf-mhd-jax.py' analog. Both Alfvén-unit, Lorentz prefactor 1. B: runs (induction + magnetic diffusion implemented), but (i) Lorentz prefactor 'Ha²/(Re·Rm)' channel…
- **B0284 [PARTIAL]:** survey line 2234; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `7`; claim: Description & rationale. When implicit (diffusion) and explicit (advection/source) terms are both active and stiff, an IMEX scheme can show observed order < formal order — the classic IMEX order-reduction pathology [ARS97]. T1–T3 e…
- **B0285 [PARTIAL]:** survey line 2236; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 1, 0.1, 0.05, 1.0, 32, 1.0, 1, 40, 1, 640`; claim: Setup (MMS). Advection–diffusion of a single mode with forcing: '∂u/∂t = ν ∂²u/∂x² − c ∂u/∂x + Q(x,t)', target 'u-exact = sin(kx − ωt)·e(−αt)'. Compute 'Q = ∂-t u-exact − ν∂-(xx)u-exact + c∂-x u-exact' symbolically (sympy), inject as…
- **B0286 [PARTIAL]:** survey line 2240; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.5, 22, 1.5, 2.5, 0.5, 1, 0.5`; claim: Metric & tolerance. Slope 'p'. Order-reduction assertion: 'p ≥ formal-order − 0.5' (IMEXRK222 ⇒ 'p≥1.5'; IMEXRK3 ⇒ 'p≥2.5'). The '−0.5' margin tolerates mild splitting-induced reduction while rejecting a full integer drop. Report m…
- **B0287 [PARTIAL]:** survey line 2242; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.3, 0.5`; claim: Tolerance rationale. Unlike T1–T3 (clean single-branch ⇒ tight ±0.3), T4 must budget for the known reduction mechanism, hence the one-sided '≥ formal−0.5' — the operationally honest tolerance.
- **B0288 [PARTIAL]:** survey line 2246; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0`; claim: Extraction. Log-log slope + per-pair orders to distinguish asymptotic reduction (constant 'p<formal') from pre-asymptotic behaviour ('p' rising toward formal as 'dt→0' — refine further, not a true reduction).
- **B0289 [VERIFIED]:** survey line 2252; anchors: A0551; local source anchor(s) verified; numbers: `1, 6, 1000, 0`; claim: The procedure differs by family because the bases differ. Spectral families (A,C) converge exponentially for smooth fields [Boyd01, CHQZ06] (demonstrated in jaxfun's own MMS self-asserts 'poisson1D.py:46 error < ulp(1000)'); the FD fam…
- **B0290 [PARTIAL]:** survey line 2256; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10, 8, 4, 5`; claim: Description & rationale. For shenfun (A) and jaxfun (C) a smooth manufactured solution must converge faster than any algebraic rate — error dropping '>10×' per fixed-'N' increment until the roundoff floor, the defining property of spec…
- **B0291 [PARTIAL]:** survey line 2258; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0`; claim: Setup (MMS). Solve the wall-normal Helmholtz the channel solvers invert: 'α u − ν u'' = f' on 'x∈[−1,1]' with the solver's own bases. Manufactured 'u-exact(x)=(1−x²)cos(3x)' — smooth, satisfies the Dirichlet wall BC 'u(±1)=0' exactly (…
- **B0292 [VERIFIED]:** survey line 2262; anchors: A0552; local source anchor(s) verified; numbers: `4, 10, 0.5, 10, 1e-11, 4, 2.2e-16, 1000, 32`; claim: Metric & tolerance. Primary (exponential): for consecutive 'N' until the floor, 'E(N+4) < E(N)/10'; equivalently fit 'log E = −c·N + b' on a semilog axis and assert decay constant 'c > 0.5'. Floor handling: stop the '>10×' chain on…
- **B0293 [PARTIAL]:** survey line 2264; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10, +4, 4, 1.78, +4, 10, 1e-11, 4, 5`; claim: Tolerance rationale. The '>10×/+4N' criterion is the operational definition of "spectral" — unattainable by any fixed-order FD scheme (a 4th-order scheme would need a '1.78×' 'N' increase, not '+4', to drop '10×'). 'ε-spectral=1e-11' i…
- **B0294 [VERIFIED]:** survey line 2266; anchors: A0553; local source anchor(s) verified; numbers: `0.24707506017508621, +0.0026644103710965817, 8000, 1e-12`; claim: Families. A: native ('chebyshev.la.Helmholtz'/'Biharmonic' or 'la.SolverGeneric1ND'); the OS golden 'c=0.24707506017508621+0.0026644103710965817j' at 'Re=8000' to '1e-12' ('OrrSommerfeld-eigs.py:183') is the eigenvalue analog. C: nativ…
- **B0295 [PARTIAL]:** survey line 2268; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 10`; claim: Extraction. Decay constant 'c' from the semilog fit; per-increment ratios 'E(N)/E(N+4)' to confirm '>10×' until the floor.
- **B0296 [VERIFIED]:** survey line 2272; anchors: A0554, A0555; local source anchor(s) verified; numbers: `4, 4, 9, 8, 0.2, 2, 8, 1e-7, 0., .8, 1, 2`; claim: Description & rationale. For torch (B) the wall-normal/radial direction is finite-difference with a Taylor/Vandermonde stencil. The family is labeled "4th-order FD," but the default 'KL=4' 9-point centered stencil is formally 8th-ord…
- **B0297 [PARTIAL]:** survey line 2274; anchors: A0556, A0557; one or more local source anchors only partially verified; numbers: `3, 0.5, 1, 1, 8, 3, 3, 0.5, 3, 0.5, 17, 33`; claim: Setup (MMS). Apply the FD derivative matrices 'W-dy1, W-dy2' ('mesh.py:186-189') to a smooth field and measure derivative error vs analytic — the operator-level test mirroring 'test-mesh.py'. Field 'u-exact(y)=cos(3y)·e(0.5y)' on 'y∈[…
- **B0298 [PARTIAL]:** survey line 2278; anchors: A0558; one or more local source anchors only partially verified; numbers: `3.7, 4, 8, 4, 6, 1, 17, 33, 65, 3.7, 257, 129`; claim: Metric & tolerance. Fit 'log E = −p·log N + b' (log-log); assert 'p ≥ 3.7' for the first-derivative operator (the documented FD floor; the clustered Chebyshev-extrema mesh makes the boundary-limited global order land between 4 and the…
- **B0299 [VERIFIED]:** survey line 2280; anchors: A0559; local source anchor(s) verified; numbers: `3.7, 2, 4, 8, 5.5`; claim: Tolerance rationale. 'p≥3.7' is the documented FD floor — it rejects a stencil silently fallen to 2nd order (e.g. a boundary-row bug) while accepting the true (≥4, often 8 interior) order. The saturation allowance is physically necessa…
- **B0300 [VERIFIED]:** survey line 2282; anchors: A0560; local source anchor(s) verified; numbers: `1e-12`; claim: Families. B: native — channel 'W-dy1/W-dy2', couette 'W-dr1/W-radlap', pipe banded 'W-dr1' (banded LU but same Taylor weights 'mesh.py:26-57'). For the pipe, the banded and dense builds are verified equal to '1e-12' ('test-banded.py'),…
- **B0301 [VERIFIED]:** survey line 2286; anchors: A0561, A0562; local source anchor(s) verified; numbers: `none`; claim: Description & rationale. This is the operational definition of cross-family parity: at torch's converged resolution, a shared physical observable must sit within torch's own truncation band of the spectral (effectively exact) val…
- **B0302 [VERIFIED]:** survey line 2290; anchors: A0563; local source anchor(s) verified; numbers: `10000, 1, 0, 128, 1e-10, 0.23752649, 0.00373967, 101, 4, 65, 101, 151`; claim: S3a — Orr–Sommerfeld leading eigenvalue (channel, hydro). Plane Poiseuille, 'Re=10000, α=1, β=0'. Spectral oracle (A/C): the OS leading eigenvalue recomputed at high 'N' ('N=128' Chebyshev, converged ~'1e-10'). Torch (B) computes it vi…
- **B0303 [VERIFIED]:** survey line 2292; anchors: A0564; local source anchor(s) verified; numbers: `0.75, 15, 16, 0.7499999944199642, 0.9373170323757943, 1e-3, 0.75, 2e-3`; claim: S3b — Epicyclic frequency κ or MRI growth 's-max' (MHD/rotation, A↔C; B only if rotation wired). Ideal local Keplerian oracle 's-max=0.75Ω', '(k v-A)²=(15/16)Ω²'; the local-MRI computation gives 's-max/Ω = 0.7499999944199642', '(k v-A)…
- **B0304 [PARTIAL]:** survey line 2294; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `128, 0.75`; claim: Oracle. S3a: 'c-spectral' from A/C at 'N=128'. S3b: 's-max/Ω = 0.75' analytic + A/C spectral confirmation.
- **B0305 [PARTIAL]:** survey line 2296; anchors: A0565; one or more local source anchors only partially verified; numbers: `201, 151, 4, 1e-3, 1e-4, 1e-3, 1e-3, 1e-12, 4, 8, 0.75, 1e-2`; claim: Metric & tolerance (the truncation band). At B's converged resolution (where its 'N'-refinement has plateaued — confirm '/c-B(201) − c-B(151)/' is below B's truncation estimate), assert '/c-B − c-spectral/ ≤ max(C-x·h-FDp, C-t·dtq,…
- **B0306 [VERIFIED]:** survey line 2298; anchors: A0566; local source anchor(s) verified; numbers: `101, 4, 1e-3, 1e-4, 1e-3`; claim: Tolerance rationale. The band 'max(C·hp, C·dtq, ε-spectral)' is derived from B's actual grid, not chosen arbitrarily: at 'N=101, KL=4' on the clustered mesh, the FD truncation error in a leading OS eigenvalue is empirically '~1e-3……
- **B0307 [VERIFIED]:** survey line 2300; anchors: A0567; local source anchor(s) verified; numbers: `128, 1e-10`; claim: Families. A, C produce the oracle (spectral, 'N=128'). A↔C can additionally be compared at 'ε-spectral=1e-10' (both spectral Galerkin with matching IMEX tableaux, 'imex-rk.py == integrators.py') — a within-spectral-class parity sub…
- **B0308 [PARTIAL]:** survey line 2302; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3.7`; claim: Extraction. Confirm B's eigenvalue converges to the spectral value at FD rate: fit '/c-B(N) − c-spectral/' vs 'N' on log-log; slope ≈ S2's 'p' (≥3.7), demonstrating B approaches the spectral floor at its own order — the cleanest poss…
- **B0309 [PARTIAL]:** survey line 2308; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1.8, 2.3, 2`; claim: / T1 decaying mode / temporal / 'e(−νk²T)' / ✓ / ✓ (θ band) / ✓ / per-integrator slope band (e.g. '[1.8,2.3]' 2nd-order) /
- **B0310 [PARTIAL]:** survey line 2309; anchors: A0568; one or more local source anchors only partially verified; numbers: `2, 2, 1e-2`; claim: / T2 epicyclic / temporal (explicit) / 'cos(κt)', 'κ²=2Ω(2Ω−S)' / ✓ / gated ('mhd.py:71-74') / ✓ / slope band + '/κ−κ-th//κ-th<1e-2' /
- **B0311 [PARTIAL]:** survey line 2310; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 0, 0.01, 0.01, 1e-2`; claim: / T3 Ohmic decay / temporal (mag. implicit) / 'e(−2ηk²t)' / ✓ / ✓ ('Ha=0') / ✓ / slope band + '/η−0.01//0.01<1e-2' /
- **B0312 [PARTIAL]:** survey line 2311; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.5`; claim: / T4 IMEX splitting / temporal (both) / MMS adv-diff / ✓ / N/A (single-stage) / ✓ / 'p ≥ formal−0.5' /
- **B0313 [PARTIAL]:** survey line 2312; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `10, +4, 1e-11`; claim: / S1 spectral exp. / spatial / smooth MMS Helmholtz / ✓ / N/A / ✓ / '>10×'/'+4N' until 'ε-spectral=1e-11' /
- **B0314 [PARTIAL]:** survey line 2313; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3.7`; claim: / S2 FD algebraic / spatial / MMS derivative / N/A / ✓ / N/A / 'p≥3.7' + saturation ceiling allowed /
- **B0315 [PARTIAL]:** survey line 2314; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-3, 1e-10`; claim: / S3 floor-meeting / cross-family / spectral value (A/C) / oracle / FD party / oracle / 'max(C·h-FDp, ε-spectral)≈1e-3'; A↔C at '1e-10' /
- **B0316 [VERIFIED]:** survey line 2320; anchors: A0569, A0570; local source anchor(s) verified; numbers: `1, 0`; claim: These are the major focus of the suite; each has an analytic oracle. The physical anchor is the shearing-box / shearpy convention ('pcf-mhd-mri-shearpy.py:11-15', 'pcf-mhd-mri-notes.md:37-42'; [BH91, SG10]); the canonical reference for the…
- **B0317 [VERIFIED]:** survey line 2322; anchors: A0571, A0572, A0573, A0574; local source anchor(s) verified; numbers: `0.2, 2, 3, 1, 3, 2, 0.025, 1000, 1e6, 2, 2, 2`; claim: Canonical conventions (apply to ALL SR tests — the §0.2 frame). 'x'=radial/wall-normal (shear gradient), 'y'=azimuthal/streamwise (wall motion), 'z'=vertical/rotation axis, 'Ω=Ω ẑ'. Base flow 'U-b(x)=−S·x·e-y', 'dU-b/dx=−S' (A 'pcf-mhd…
- **B0318 [VERIFIED]:** survey line 2324; anchors: A0575, A0576; local source anchor(s) verified; numbers: `1, 2, 3, 4, 5, 6, 7, 9`; claim: Family applicability at a glance. A (shenfun): all SR tests run today (DNS MHD 'pcf-mhd-mri-shearpy.py' + dense-linear '-pcf-linear.py', 'taylor-couette-mri.py'). C (jax): all run today (DNS 'pcf-mhd-mri-shearpy-jax.py', 'taylor-couett…
- **B0319 [PARTIAL]:** survey line 2328; anchors: A0577; one or more local source anchors only partially verified; numbers: `0, 2, +2, 2, 0`; claim: Description & rationale. With rotation Ω and shear S, a uniform ('k=0') velocity perturbation is a 2-D harmonic oscillator: Coriolis couples 'u-x ↔ u-y' at the epicyclic frequency κ. The most basic check that the source terms '+2Ω u-y'…
- **B0320 [VERIFIED]:** survey line 2330; anchors: A0578; local source anchor(s) verified; numbers: `0, 0, 0, -1, 1, 0, 2, 0, 2, 2, 3, 1`; claim: Setup. Triply-periodic dynamics (only 'k=0' excited); for wall-bounded families use the '(0,0)'-mode mean equations ('ChannelFlow.py:174-197'). Domain e.g. '((-1,1),(0,2π),(0,2π))', Lorentz/magnetic OFF. Keplerian 'Ω=2/3, S=1' ('κ=2/3'…
- **B0321 [PARTIAL]:** survey line 2336; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2`; claim: u-y(t) = u-x0 · ((S − 2Ω)/κ) · sin(κ t)
- **B0322 [PARTIAL]:** survey line 2337; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2`; claim: κ = sqrt(2Ω(2Ω − S))
- **B0323 [PARTIAL]:** survey line 2340; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 3, 1, 0.6666666666666665, 2, 1, 4, 3, 2, 3, 0.5, 2`; claim: Keplerian ('Ω=2/3,S=1'): 'κ=0.6666666666666665', amplitude ratio '(S−2Ω)/κ=(1−4/3)/(2/3)=−0.5', period '2π/κ=9.424777960769381'. Non-Keplerian ('Ω=1,S=1'): 'κ=√2=1.41421356', ratio '(1−2)/√2=−0.70710678'.
- **B0324 [PARTIAL]:** survey line 2342; anchors: A0579; one or more local source anchors only partially verified; numbers: `20, 2, 1e-2, 1e-2, 22, 1e-2, 5e-4, 1e-2`; claim: Metric & tolerance. Sample 'u-x(t), u-y(t)' at ≥20 times over two periods. 'rel-err = max-t/u-x(num)(t) − u-(x0)cos κt//u-(x0)' and likewise for 'u-y' (normalize by '/u-(x0)(S−2Ω)/κ/'). Pass if 'rel-err < 1e-2' — a physics-oracle…
- **B0325 [VERIFIED]:** survey line 2344; anchors: A0580, A0581; local source anchor(s) verified; numbers: `346, -348, 2, 2, 2, 22, 1, 0.51`; claim: Families. A, C: run directly (A 'pcf-mhd-mri-shearpy.py:130-134, 346-348'; C 'pcf-mhd-mri-shearpy-jax.py:130-134'). B: acceptance gate — passes only once 'n-x += −2Ω u-y', 'n-y += (S−2Ω)u-x' wiring is added (and apply the canonical…
- **B0326 [VERIFIED]:** survey line 2348; anchors: A0582; local source anchor(s) verified; numbers: `0, 1, 2.8, 3.4, 5.5`; claim: Description & rationale. Isolates the new induction term from the base shear: 'dB-y/dt = −S·B-x' (the Ω-effect — azimuthal field generated from radial field by differential rotation). For a uniform 'k=0' field with no flow and no res…
- **B0327 [PARTIAL]:** survey line 2350; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 0, 2, 3, 1, 0, 1e12, 0, 0.025, 0, 0, 0`; claim: Setup. As SR-1, 'k=0' magnetic mode only. 'Ω=2/3, S=1'; resistivity OFF ('η=0' or 'Rm=1e12'). Velocity perturbation suppressed / 'B-x' small so 'J×B' back-reaction is negligible over a short integration. IC uniform 'B-x(0)=B-(x0)=0…
- **B0328 [PARTIAL]:** survey line 2352; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0.025, 1, 0.025, 5, 0.125, 0.025`; claim: Oracle. 'B-y(t) = B-(y0) − S·B-(x0)·t' (with 'B-x(t)=B-(x0)' const for 'k=0', no decay). For 'B-(y0)=0, B-(x0)=0.025, S=1': 'B-y(t)=−0.025 t', 'B-y(5)=−0.125'; slope 'dB-y/dt=−0.025'.
- **B0329 [PARTIAL]:** survey line 2354; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2, 1e-3, 1, 1e12, 22, 1e-6, 1e-4`; claim: Metric & tolerance. Linear fit of 'B-y(t)' vs 't'; assert '/slope-fit − (−S·B-(x0))///S·B-(x0)/ < 1e-2' and 'B-x' constant '/B-x(t)−B-(x0)//B-(x0) < 1e-3'. Pass if both. Physics-oracle tolerance: exact in the continuum, error = tim…
- **B0330 [VERIFIED]:** survey line 2356; anchors: A0583, A0584, A0585; local source anchor(s) verified; numbers: `366, -376`; claim: Families. A: term arises automatically from 'dA/dt=U×B' with 'U-b=−S x e-y' ('pcf-mhd-mri-shearpy.py:16-21, 366-376'); the linear operator has it explicitly 'L[by,bx]=Uprime=−S' ('-pcf-linear.py:239'). C: same via the EMF with 'U-b=−S…
- **B0331 [PARTIAL]:** survey line 2360; anchors: A0586; one or more local source anchors only partially verified; numbers: `2, 1e-5, 1e-8`; claim: Description & rationale. A single magnetic Fourier mode with no flow decays purely resistively, calibrating the effective numerical resistivity and verifying the magnetic-diffusion operator ('η∇²b' / 'ηΔA') is discretized and time-spli…
- **B0332 [PARTIAL]:** survey line 2362; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 100, 0.01, 1, 0, 2, 1, 0.01, 16, 0.01, 50`; claim: Setup. Single mode 'b ∝ sin(kz)' in a periodic axis (eigenfunctions in the wall-normal direction are not pure Fourier — use 'z'). Flow OFF ('u=0', freeze the momentum solve). 'Ω, S' irrelevant (set 'S=0' to avoid winding contaminating…
- **B0333 [PARTIAL]:** survey line 2364; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 0.01, 1, 0.01, 0.02, 50, 1, 0.3678794`; claim: Oracle. 'E(t)=E-0 e(−2ηk²t)', 'b(t)=b-0 e(−ηk²t)'. For 'η=0.01, k=1': amplitude rate '0.01', energy rate '0.02', 'E(50)/E-0=e(−1)=0.3678794'.
- **B0334 [PARTIAL]:** survey line 2366; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2, 1e-2, 1e-4, 1e-3, 1e-6`; claim: Metric & tolerance. Fit 'log E(t)' vs 't'; assert '/rate-fit − 2ηk²//(2ηk²) < 1e-2'. Pass if holds. Physics-oracle tolerance: exact decay law, error from time truncation only (implicit diffusion, very accurate); within-family A/C e…
- **B0335 [VERIFIED]:** survey line 2368; anchors: A0587, A0588, A0589, A0590, A0591; local source anchor(s) verified; numbers: `1, 1, 0, 3, 0`; claim: Families. A: resistive Helmholtz on A ('pcf-mhd-divfree.py:159-192', 'η=U/Rm'). C: same ('pcf-mhd-jax.py:78-85'). B: HAS magnetic diffusion ('mhd.py:115-117', diffusivity '1/Rm' channel / '1/Pm' couette) — this test CAN run on torch…
- **B0336 [PARTIAL]:** survey line 2374; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1e6, 0.1, 0.025, 1, 2, 0, 0, 0, 1e-3, 16, 0.05`; claim: Setup. Periodic propagation direction. Uniform background 'B0=B0 ẑ', transverse perturbation. Rotation/shear OFF ('Ω=S=0'); resistivity OFF or tiny ('Rm=1e6', damping rate '½ηk²' negligible). 'B0=v-A=0.1' (or '0.025'), 'k=1' along 'z'…
- **B0337 [PARTIAL]:** survey line 2376; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.1, 1, 0.1, 62.832, 0.025, 0.025`; claim: Oracle. 'ω=k·v-A', 'v-A=B0'; 'δu-y(t)=ε v-A cos(ωt)'. For 'B0=0.1, k=1': 'ω=0.1', 'T=62.832'. For 'B0=0.025': 'ω=0.025'.
- **B0338 [PARTIAL]:** survey line 2378; anchors: A0593; one or more local source anchors only partially verified; numbers: `1e-2, 1e-4, 1e-2, 1.2e-2`; claim: Metric & tolerance. FFT 'δu-y(t)' (or fit the oscillation); assert '/ω-meas − k v-A//(k v-A) < 1e-2'. Pass if holds. Physics-oracle tolerance: exact dispersion, error = time truncation + tiny resistive damping; within-family A/C '~…
- **B0339 [PARTIAL]:** survey line 2380; anchors: A0594, A0595, A0596, A0597, A0598; one or more local source anchors only partially verified; numbers: `350, -376, 138, -146, 1, 0, 1, 0, 0, 0`; claim: Families. A: imposed field via '-total-b-components' + EMF/Lorentz with 'B-total' ('pcf-mhd-mri-shearpy.py:127-132, 350-376'); the linear operator has 'ikB' coupling ('-pcf-linear.py:230-238'). C: same ('pcf-mhd-mri-shearpy-jax.py:116-…
- **B0340 [PARTIAL]:** survey line 2384; anchors: A0599; one or more local source anchors only partially verified; numbers: `4, 4, 3, 4, 15, 16, 3, 1, 4, 4`; claim: Description & rationale. The geometry-free analytic heart of the MRI: for a Keplerian shearing box with uniform vertical field, the local 4×4 axisymmetric dispersion ('u-x,u-y,b-x,b-y') has maximum growth 's-max=(3/4)Ω' at '(k v-A)²=(1…
- **B0341 [PARTIAL]:** survey line 2386; anchors: A0600, A0601; one or more local source anchors only partially verified; numbers: `4, 4, 2, 2, 0, 3, 2, 3, 0.5, 0, 3.2, 4`; claim: Setup. Run as a dense-linear / algebraic dispersion check for the ideal (inviscid, non-resistive) limit: form the 4×4 matrix and find its leading eigenvalue vs 'k v-A'. The local biquartic ('taylor-couette-mri.py:91-93'): 's⁴ + 2s²…
- **B0342 [PARTIAL]:** survey line 2389; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 2, 0`; claim: [[ 0, 2Ω, i k b0, 0 ],
- **B0343 [PARTIAL]:** survey line 2390; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 0, 0`; claim: [ S−2Ω, 0, 0, i k b0],
- **B0344 [PARTIAL]:** survey line 2391; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0, 0`; claim: [ i k b0, 0, 0, 0 ],
- **B0345 [PARTIAL]:** survey line 2392; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 0`; claim: [ 0, i k b0, −S, 0 ]]
- **B0346 [PARTIAL]:** survey line 2395; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 4, 2, 3, 0.5, 0.75, 15, 16, 0.9375, 3.0`; claim: Oracle. 's-max=(3/4)Ω' ('Ω=2/3' ⇒ '0.5'), 's-max/Ω=0.75', argmax at '(k v-A)²/Ω²=15/16=0.9375', marginal cutoff '(k v-A)²/Ω²=3.0'.
- **B0347 [VERIFIED]:** survey line 2397; anchors: A0602; local source anchor(s) verified; numbers: `0.75, 2e-3, 0.9375, 5e-3, 0, 3, 0, 1e-12, 2e-3, 5e-3, 0.7499999944199642, 0.9373170323757943`; claim: Metric & tolerance. Scan, find the max real eigenvalue and its argmax. Assert '/s-max/Ω − 0.75/ < 2e-3' and '/(k v-A)²-opt/Ω² − 0.9375/ < 5e-3'; for the cutoff, assert growth '>0' just inside '(k v-A)²=3Ω²' and '≈0' ('≤1e-12') just out…
- **B0348 [VERIFIED]:** survey line 2399; anchors: A0603, A0604; local source anchor(s) verified; numbers: `4, 4`; claim: Families. A: 'taylor-couette-mri.py:104-122' 'mri-keplerian-optimum'. C: 'taylor-couette-mri-jax.py:47-62'. B: torch has no MRI operator — to run this as a 4×4 algebraic check, implement the dispersion evaluator (pure linear algebr…
- **B0349 [PARTIAL]:** survey line 2403; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.75, 5`; claim: Description & rationale. In a wall-bounded geometry the MRI growth 'γ(k-z)' depends on the magnetic BC: conducting vs insulating give different — even sign-flipped — growth at the same hydro/resistive parameters. Verifies the BC machin…
- **B0350 [VERIFIED]:** survey line 2405; anchors: A0605, A0606, A0607, A0608; local source anchor(s) verified; numbers: `1, 2, 0.5, 3, 2, 1.0, 2.0, 1.0, 0.5, 1.5, 0, 0`; claim: Setup. TC annulus, 'R1=1,R2=2', 'η=0.5', quasi-Keplerian ('Ω(r)∝r(−3/2)' analogue; the bench uses 'CircularCouette(1.0, 2.0, 1.0, 0.51.5)', 'couette-linear-benchmarks.md:329'). BCs under test: (a) conducting — 'b-r=0' (Dirichlet…
- **B0351 [PARTIAL]:** survey line 2410; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `+0.003322863594034156, 1.75, +0.00332`; claim: Conducting: max-kz Re(s) = +0.003322863594034156 at kz ≈ 1.75 (≈ +0.00332)
- **B0352 [PARTIAL]:** survey line 2411; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-0.00027582037141390655, 1.25, -2.76e-4`; claim: Insulating: max-kz Re(s) = -0.00027582037141390655 at kz ≈ 1.25 (≈ -2.76e-4)
- **B0353 [VERIFIED]:** survey line 2416; anchors: A0610, A0611; local source anchor(s) verified; numbers: `1e-3, 1e-6, 1e-6, 32, 0.25628761535339467, 0.25995005500337837, 0, 9`; claim: Metric & tolerance. Compute 'max-kz Re(s)' per BC. Within-family (A↔A regression, or C reproducing its own golden): 'rel < 1e-3' against the stored golden (eigenvalue numbers, spectral-exact at modest 'Nr'). Cross-family A↔C (b…
- **B0354 [VERIFIED]:** survey line 2418; anchors: A0612; local source anchor(s) verified; numbers: `0, 0, 3, 0, 0, 24, 32, 48, 2, 32`; claim: Families. A: 'taylor-couette-mri.py' (both BCs). C: 'taylor-couette-mri-jax.py' (both BCs, '-assemble-flux-parts' for insulating, 'm=0'). B: torch has conducting/homogeneous 'b=0' only, no insulating, no MRI operator — cannot run…
- **B0355 [PARTIAL]:** survey line 2422; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `7, 0.25, 0.5, 24.7, 16.5, 0`; claim: Description & rationale. The MRI onset Rm depends on the magnetic Prandtl number 'Pm=ν/η'. Scanning 'Pm' and checking critical 'Rm-onset' against the TC-MRI conducting goldens validates the relative viscous/resistive scaling and the cr…
- **B0356 [VERIFIED]:** survey line 2424; anchors: A0613; local source anchor(s) verified; numbers: `0.5, 4.11, 4.11, 0.1, 1, 0.02, 32`; claim: Setup. TC, 'η=0.5', quasi-Keplerian, conducting walls (for the like-for-like scan; 'S=4.11'). Fixed Lundquist 'S=4.11'; scan 'Pm ∈ (0.1, 1)' (optionally '0.02'). For each Pm run the critical-Rm bisection 'critical-Rm(Pm,S)' ('taylor-co…
- **B0357 [PARTIAL]:** survey line 2426; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.5, 4.11`; claim: Oracle (golden numbers, conducting, 'η=0.5' quasi-Kep, 'S=4.11').
- **B0358 [PARTIAL]:** survey line 2429; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 95.3`; claim: Pm = 1 → Rm-onset = 95.3
- **B0359 [PARTIAL]:** survey line 2430; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.1, 32.9`; claim: Pm = 0.1 → Rm-onset = 32.9
- **B0360 [PARTIAL]:** survey line 2431; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.02, 26.7`; claim: Pm = 0.02 → Rm-onset = 26.7
- **B0361 [PARTIAL]:** survey line 2432; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `24.7, 0, 2023`; claim: (→ Rm-min ≈ 24.7 as Pm → 0; Rüdiger 2023)
- **B0362 [PARTIAL]:** survey line 2435; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5.21, 0.1, 28.2, 16.5`; claim: Insulating analogue ('S=5.21'): 'Pm=0.1 → Rm=28.2', 'Rm-min=16.5'.
- **B0363 [PARTIAL]:** survey line 2437; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1e-2, 3, 1e-2, 3e-3, 1e-6, 1, 0.1, 0.02`; claim: Metric & tolerance. Per Pm, assert '/Rm-onset − golden//golden < 1e-2' (goldens quoted to 3 sig figs ⇒ '1e-2' covers quoting + 'Nr' convergence); within-family A↔A '< 3e-3'. Cross-family A↔C: set-matched leading eigenvalue at the same…
- **B0364 [PARTIAL]:** survey line 2439; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 3, 4`; claim: Families. A: 'critical-Rm(Pm,S)' ('taylor-couette-mri.py'); test 'test-critical-rm-uses-fixed-pm-and-lundquist-controls' ('test-taylor-couette.py'). C: 'taylor-couette-mri-jax.py' critical-parameter path. B: cannot run (no MRI oper…
- **B0365 [PARTIAL]:** survey line 2443; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `5`; claim: Description & rationale. In a nonlinear MRI DNS total energy obeys 'dE/dt = Production − Dissipation', production from Maxwell + Reynolds stresses working against the background shear, dissipation viscous + Ohmic. Closing the budget to…
- **B0366 [VERIFIED]:** survey line 2445; anchors: A0614; local source anchor(s) verified; numbers: `-2, 2, 0, 4, 0, 1, 1000, 1, 2, 3, 0.025, 1e-3`; claim: Setup. Shearpy net-flux MRI run (A/C). Domain '((-2,2),(0,4),(0,1))', 'Re=Rm=1000', 'S=1, Ω=2/3, b-z=0.025' ('test-pcf-mhd-mri-shearpy.py:80-84'). Conducting magnetic BC (A 'A∈TD³'; C same). Net vertical flux 'B0=b-z'. IC small perturb…
- **B0367 [VERIFIED]:** survey line 2447; anchors: A0615, A0616; local source anchor(s) verified; numbers: `2, 0, 7, 1., .3, 1, 1`; claim: Oracle. Two oracles. Linear-growth phase (quantitative): magnetic energy grows, 'E-mag(t-end) > 2·E-mag(0)' (findings expect ~7× over 't=1..3'), strictly monotone increasing ('test-pcf-mhd-mri-shearpy.py:130-131'). Energy-balance cl…
- **B0368 [VERIFIED]:** survey line 2449; anchors: A0617, A0618; local source anchor(s) verified; numbers: `0.005, 22, 2, 1e-2, 1e-3, 0, -1, 2, 0, 7, 0, 0`; claim: Metric & tolerance. Energy-balance residual: cross-family / DNS tolerance '< max(C·Δxp, C·Δtq, ε-spectral)' — with 'dt=0.005' (IMEXRK222, 'q=2') and spectral space, bounded by the time-truncation of the energy FD; use '< 1e-2' (relat…
- **B0369 [VERIFIED]:** survey line 2451; anchors: A0619, A0620, A0621, A0622, A0623; local source anchor(s) verified; numbers: `1, 2, 8, 2, 4, 3, 2e-3, 4e-8, 4e-7, 2e-6, 0, 1`; claim: Families. A: 'pcf-mhd-mri-shearpy.py:385-413' computes Reynolds/Maxwell/α; 'test-netflux-mri-magnetic-energy-grows' is the seed. C: 'pcf-mhd-mri-shearpy-jax.py:149-182' computes the same + α. B: acceptance gate — torch computes Max…
- **B0370 [PARTIAL]:** survey line 2455; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6`; claim: Description & rationale. A focused, fast regression that the magnetic BC changes the sign of the marginal MRI growth at fixed-but-BC-specific parameters. Guards against the most insidious BC bug (the wrong Robin Jacobian 'c=r-wall/J'…
- **B0371 [PARTIAL]:** survey line 2457; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6, 0.5, 32, 24.7, 4.11, 1.75, 16.5, 5.21, 1.25`; claim: Setup. Identical to SR-6 reduced to a single assertion. TC, 'η=0.5', quasi-Keplerian, 'Nr=32'. Two runs: conducting 'Rm=24.7, S=4.11, kz=1.75'; insulating 'Rm=16.5, S=5.21, kz=1.25'.
- **B0372 [PARTIAL]:** survey line 2462; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `+0.003322863594034156, 0`; claim: conducting max growth = +0.003322863594034156 (> 0, unstable)
- **B0373 [PARTIAL]:** survey line 2463; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `-0.00027582037141390655, 0`; claim: insulating max growth = -0.00027582037141390655 (< 0, stable)
- **B0374 [VERIFIED]:** survey line 2466; anchors: A0624; local source anchor(s) verified; numbers: `+0.00332, 2.76e-4`; claim: The load-bearing fact is the sign flip: '+0.00332' vs '−2.76e-4' ('couette-linear-benchmarks.md:352-353').
- **B0375 [PARTIAL]:** survey line 2468; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 1e-2, 3e-3, 1e-10, 1e-10`; claim: Metric & tolerance. Two-part: (a) sign assertion 'conducting > 0 > insulating' (binary, no tolerance needed); (b) value regression '/computed − golden///golden/ < 1e-2' per BC (within-family '< 3e-3'). Pass if both. Additionall…
- **B0376 [PARTIAL]:** survey line 2470; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6.`; claim: Families. A: 'taylor-couette-mri.py' (both BCs + the BC-satisfaction test). C: 'taylor-couette-mri-jax.py' (both BCs). B: cannot run (no insulating walls, no MRI operator) — explicit documentation of torch's gap. This test is A/C o…
- **B0377 [PARTIAL]:** survey line 2476; anchors: A0625; one or more local source anchors only partially verified; numbers: `1, 2, 2, 1e-2, 1e-2`; claim: / SR-1 epicyclic / temporal/physics / 'cos(κt)', 'κ²=2Ω(2Ω−S)' / ✓ / gate ('mhd.py:71-74') / ✓ / 'rel<1e-2'; FFT '/κ−κ-th//κ-th<1e-2' /
- **B0378 [PARTIAL]:** survey line 2477; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 1e-2, 1e-4`; claim: / SR-2 shear-winding / physics / 'B-y=−S B-x t' / ✓ / gate / ✓ / slope 'rel<1e-2' ('1e-4' within A/C) /
- **B0379 [PARTIAL]:** survey line 2478; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 2, 0, 2, 2, 1e-2`; claim: / SR-3 Ohmic decay / physics / 'e(−2ηk²t)' / ✓ / ✓ ('Ha=0') / ✓ / '/rate−2ηk²//(2ηk²)<1e-2' /
- **B0380 [PARTIAL]:** survey line 2479; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 1, 1e-2`; claim: / SR-4 Alfvén wave / physics / 'ω=k v-A' / ✓ / ✓ ('lorentz-prefactor=1') / ✓ / '/ω−k v-A//(k v-A)<1e-2' /
- **B0381 [PARTIAL]:** survey line 2480; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5, 4, 4, 0.75, 15, 16, 4, 4, 2e-3, 5e-3`; claim: / SR-5 ideal MRI 4×4 / algebraic / 's-max/Ω=0.75', '(k v-A)²/Ω²=15/16' / ✓ / (4×4 evaluator only) / ✓ / '2e-3' / '5e-3' /
- **B0382 [PARTIAL]:** survey line 2481; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6, +0.00332, 2.76e-4, 1e-3, 1e-6`; claim: / SR-6 wall-bounded MRI BC / eigenvalue / golden '+0.00332' / '−2.76e-4' / ✓ / A (no insul./MRI) / ✓ / within '1e-3'; A↔C 'max(ε-spec,1e-6)' /
- **B0383 [PARTIAL]:** survey line 2482; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `7, 1e-2, 3e-3`; claim: / SR-7 Pm-scan / eigenvalue / 'Rm-onset' table / ✓ / A / ✓ / '1e-2' (within '3e-3') + monotonicity /
- **B0384 [PARTIAL]:** survey line 2483; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `8, 2, 1e-2`; claim: / SR-8 energy/stress budget / DNS integral / 'dE/dt=P−D'; 'E-mag>2×' / ✓ / gate (stresses exist) / ✓ / residual '<1e-2'; growth binary /
- **B0385 [PARTIAL]:** survey line 2484; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `9, +0.00332, 2.76e-4, 1e-2`; claim: / SR-9 BC sign-flip / eigenvalue / '+0.00332' vs '−2.76e-4' / ✓ / A / ✓ / sign binary + '1e-2' value /
- **B0386 [PARTIAL]:** survey line 2486; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 8`; claim: Legend: ✓ runs today; gate = the test defines the torch-wiring acceptance milestone (SR-1/SR-2 first, then SR-8); A = absent capability in torch (no insulating wall and/or no MRI operator), record as 'skip' with the cited gap.
- **B0387 [PARTIAL]:** survey line 2492; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `6, 7`; claim: Per the §"Disjoint run environments" hand-off note, no live in-process cross-family import is permitted — cross-family parity (S3, the A↔C sub-assertions of SR-6/SR-7) reads committed JSON/HDF5 goldens written by each family in its own env…
- **B0388 [PARTIAL]:** survey line 2496; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3.12, .3, 0.10, .1, 3`; claim: - C → '/home/nauman/cfd/shenfun-jaxfun-spectralDNS/fork-jaxfun/.venv' (uv, system Python 3.12.3, local JAX 0.10.1; 'cuda13' optional extra configured).
- **B0389 [PARTIAL]:** survey line 2498; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 2, 3, 4, 6, 7, 9`; claim: Per-family "skip not fail" gaps to encode (so a missing capability never reads as a regression): C has no pipe ('F1d', 'F1e', any 'S2'-pipe variant → 'skip'); B has no Womersley ('F1e' → 'skip'), no pipe MHD, and MRI metadata-only ('SR…
- **B0390 [PARTIAL]:** survey line 2505; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0.1, 2, 3`; claim: This part is the cross-family reference layer for everything the three solver families (A = shenfun, B = torch, C = jax; see §0.1) share or diverge on at the infrastructure level — hardware targets, floating precision, just-i…
- **B0391 [PARTIAL]:** survey line 2507; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 1, 4, 2`; claim: The single recurring theme: A is the spectral oracle but the compute laggard (CPU/MPI, float64, no JIT, no autograd); B is the differentiable GPU workhorse (PyTorch, CUDA-by-construction, full autograd including the Lorentz couplin…
- **B0392 [VERIFIED]:** survey line 2519; anchors: A0626, A0627; local source anchor(s) verified; numbers: `3, 0.10, .1`; claim: / GPU / none — CPU/MPI only / yes by construction (device-agnostic tensors; CUDA exercised only in 'torchpipeflow/benchmarks/benchmark-hotspots.py:29,39-41') / yes (JAX/XLA; 'cuda13' extra configured in 'pyproject.toml:19-37'; loca…
- **B0393 [PARTIAL]:** survey line 2522; anchors: A0631; one or more local source anchors only partially verified; numbers: `4, 4, 2, 5e-5, 4`; claim: / Lower precision validated? / n/a (always f64) / yes — complex64/float32 path, roundtrip err '<5e-5' ('test-float32.py:21-23') / x64 is unconditional at import; toggle is global /
- **B0394 [PARTIAL]:** survey line 2531; anchors: no same-line source anchor; literature-cited benchmark not independently checked in this run; numbers: `8`; claim: A (shenfun) is CPU-only and scales via MPI: 'mpi4py-fft' provides automatic slab and pencil domain decompositions and a global-array redistribution algorithm, and has been run on thousands of cores on supercomputers [Morten…
- **B0395 [VERIFIED]:** survey line 2533; anchors: A0637, A0638, A0639; local source anchor(s) verified; numbers: `39, -41`; claim: B (torch) is device-agnostic by construction. Every tensor is created with an explicit 'device=' taken from the mesh ('self.mesh.y.device' / 'self.mesh.r.device'), and no solver kernel special-cases CUDA. Moving a problem to GPU is…
- **B0396 [VERIFIED]:** survey line 2542; anchors: A0643, A0644, A0645, A0646, A0647, A0648; local source anchor(s) verified; numbers: `28, 4, 28, 100, -101, 4, 28, 2, 4, 103, 4, 4`; claim: - B: the default solver 'dtype' is 'torch.complex128' ('torchchannel/.../solver.py:87'), validated to 'complex64'/'complex128' only (':100-101'); the derived real working dtype is float64 for complex128 and float32 for complex6…
- **B0397 [PARTIAL]:** survey line 2545; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 4, 1e-10, 1e-21, 1e-7, 1, 2`; claim: Implication for cross-family comparisons (ties to the §3 tolerance ladder): all three families can meet a float64 floor, so within-family operator identities may be asserted at roundoff (spectral families 1e-10…1e-21; B's pinv cleanup…
- **B0398 [PARTIAL]:** survey line 2586; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5`; claim: - B and C have full autograd; A has none. Any "differentiate the solver" parity test (e.g. confirming ∂growth/∂Re agrees) can only be a B↔C comparison; A participates only through forward observables. This is captured in the closur…
- **B0399 [PARTIAL]:** survey line 2592; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3.12, .3, 0.10, .1, 3`; claim: The three families live in three mutually incompatible Python environments and cannot be co-imported in one process. This is a hard constraint, not a preference: A is a conda env built around 'shenfun'/'mpi4py-fft'; B is a separate con…
- **B0400 [PARTIAL]:** survey line 2598; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3.12, .3, 0.10, .1, 3`; claim: / C = jax / 'uv' venv (system Python 3.12.3, JAX 0.10.1 locally verified; 'cuda13' extra configured) / '/home/nauman/cfd/shenfun-jaxfun-spectralDNS/fork-jaxfun/.venv' /
- **B0401 [PARTIAL]:** survey line 2602; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1.`; claim: 1. Running each family's solver in its own environment as a subprocess (or in CI as a separate job), invoked through the per-family interpreter above.
- **B0402 [PARTIAL]:** survey line 2603; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2., 0.2`; claim: 2. Writing the canonical-frame observables — growth rates, energies, Reynolds/Maxwell stresses, transport 'α', divergence norms — to a file (a "golden") after applying the planned 'to-canonical()' adapter (§0.2) so that B's swapped axe…
- **B0403 [PARTIAL]:** survey line 2604; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3., 3`; claim: 3. Reading the goldens back in a neutral comparison harness and asserting agreement against the cross-family tolerance band 'max(C·Δxp, C·Δtq, ε-spectral)', never roundoff (§3 tolerance ladder).
- **B0404 [PARTIAL]:** survey line 2606; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0`; claim: This is the intended mechanism behind the Phase-0 prerequisite of the closure roadmap ('parity/conventions.py', 'parity/observables.py'): the adapter and the golden writer should be the only shared code, and they should operate on plain ar…
- **B0405 [PARTIAL]:** survey line 2608; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1`; claim: Skip-not-fail conventions encoded in the golden harness (so a missing capability is recorded as a skip, not a spurious failure): C has no pipe (pipe goldens skip for C); B has no Womersley oracle and its MRI is metadata-only (the MRI s…
- **B0406 [PARTIAL]:** survey line 2619; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1971, 50, 4, 689, 703., 10.1017, 022112071002842., 5772.22`; claim: - Orszag71 — Orszag, S. A. (1971). Accurate solution of the Orr–Sommerfeld stability equation. J. Fluid Mech. 50(4), 689–703. DOI:10.1017/S0022112071002842. — plane-Poiseuille linear critical Re-crit = 5772.22 (Chebyshev-tau + QR…
- **B0407 [PARTIAL]:** survey line 2620; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 1993, 252, 209, 238., 10.1017, 022112093003738., 1000`; claim: - RH93 — Reddy, S. C. & Henningson, D. S. (1993). Energy growth in viscous channel flows. J. Fluid Mech. 252, 209–238. DOI:10.1017/S0022112093003738. — transient growth O(R²), magnitude O(1000), from operator non-normality.
- **B0408 [PARTIAL]:** survey line 2632; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `5, 1955, 127, 3, 553, 563., 10.1113, 1955., 05276., 10.1113, 1955., 05276`; claim: - Womersley55 — Womersley, J. R. (1955). Method for the calculation of velocity, rate of flow and viscous drag in arteries when the pressure gradient is known. J. Physiol. 127(3), 553–563. DOI:10.1113/jphysiol.1955.sp005276. https://…
- **B0409 [PARTIAL]:** survey line 2636; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2012, 702, 415, 443., 1.5`; claim: - PWK12 — Pringle, C. C. T., Willis, A. P. & Kerswell, R. R. (2012). Minimal seeds for shear flow turbulence: using nonlinear transient growth to touch the edge of chaos. J. Fluid Mech. 702, 415–443. — the minimal-seed adjoint-optimi…
- **B0410 [PARTIAL]:** survey line 2639; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 1994, 268, 175, 209., 10.1017, 02211209400131, 5300`; claim: - Eggels94 — Eggels, J. G. M. et al. (1994). Fully developed turbulent pipe flow: DNS vs experiment. J. Fluid Mech. 268, 175–209. DOI:10.1017/S002211209400131X. — Re=5300 turbulent-statistics benchmark.
- **B0411 [PARTIAL]:** survey line 2645; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `4, 1984, 1., 146, 45, 64., 10.1017, 022112084001762.`; claim: - Marcus84 — Marcus, P. S. (1984). Simulation of Taylor-Couette flow. Part 1. J. Fluid Mech. 146, 45–64. DOI:10.1017/S0022112084001762. — pseudospectral TC with Green-function/capacitance BC enforcement; growth rates/wave speeds vs l…
- **B0412 [PARTIAL]:** survey line 2657; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `2, 2002, 176, 2, 430, 455., 10.1006, 2002.6995`; claim: - CM02 — Cox, S. M. & Matthews, P. C. (2002). Exponential time differencing for stiff systems. J. Comput. Phys. 176(2), 430–455. DOI:10.1006/jcph.2002.6995. — ETD/ETDRK4 (C's 'ETDRK4' menu option); small-eigenvalue Taylor cutoff.
- **B0413 [PARTIAL]:** survey line 2668; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `0, 2000, 000, -1444, 10.2172, 759450., 759450`; claim: - SalariKnupp00 — Salari, K. & Knupp, P. (2000). Code Verification by the Method of Manufactured Solutions. SAND2000-1444, Sandia. DOI:10.2172/759450. https://www.osti.gov/biblio/759450 — MMS detects any order-of-accuracy coding error.
- **B0414 [PARTIAL]:** survey line 2671; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `7, 1937, 158, 895, 499, 521., 10.1098, 1937.0036, 2, 2, 4`; claim: - TaylorGreen37 — Taylor, G. I. & Green, A. E. (1937). Mechanism of the production of small eddies from large ones. Proc. R. Soc. A 158(895), 499–521. DOI:10.1098/rspa.1937.0036. — 2D TGV exact solution 'u=sin x cos y·e(−2νt)', KE∝e…
- **B0415 [PARTIAL]:** survey line 2672; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `3, 1983, 130, 411, 452., 10.1017, 022112083001159., 3, 256, 3000`; claim: - BMO83 — Brachet, M. E. et al. (1983). Small-scale structure of the Taylor–Green vortex. J. Fluid Mech. 130, 411–452. DOI:10.1017/S0022112083001159. — 3D TGV transition benchmark (≤256³ modes, Re up to 3000+).
- **B0416 [PARTIAL]:** survey line 2676; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `1, 1991, 376, 214, 222., 10.1086, 170270., 0.`; claim: - BH91 — Balbus, S. A. & Hawley, J. F. (1991). A powerful local shear instability in weakly magnetized disks. I. Linear analysis. ApJ 376, 214–222. DOI:10.1086/170270. — MRI linear analysis; max growth ~Ω, field-strength-independent;…
- **B0417 [PARTIAL]:** survey line 2680; anchors: no same-line source anchor; benchmark-style numeric claim without same-line source anchor; numbers: `7, 2007, 378, 4, 1471, 1480., 10.1111, 1365, -2966.2007, .11888, 0704.2943, 0.25`; claim: - LL07 — Lesur, G. & Longaretti, P.-Y. (2007). Impact of dimensionless numbers on the efficiency of MRI-induced turbulent transport. MNRAS 378(4), 1471–1480. DOI:10.1111/j.1365-2966.2007.11888.x. arXiv:0704.2943. — α∝Pmδ, δ∈[0.25,…

<!-- SOLVER_SURVEY_AUDIT_APPENDIX_END -->
