# Couette Demo Guide

This directory contains the Shenfun reference examples for plane-Couette and
Taylor-Couette flow in hydrodynamic and MHD regimes.  The linear paths are small
dense generalized-eigenvalue tools for analysis and validation; the DNS paths
are the time-dependent Shenfun solvers; and `thin_gap_compare.py` puts the two
geometries side by side
(apples-to-apples).  **Several tasks have more than one implementation** (a
shenfun-Galerkin operator *and* a dense Chebyshev-collocation operator; a CNAB2
*and* an IMEXRK DNS); see [Which approach to use](#which-approach-to-use) for a
decision table, and [Porting to another shenfun
codebase](#porting-to-another-shenfun-codebase) for the minimal file sets.

> **Where to run from.**  Run the scripts from inside `couette/` (the modules
> import their siblings by bare name), in the shenfun conda environment.
> Examples below are written `python <script>.py ...`; from the repo root prefix
> them with `couette/`.

## Shared Helpers

- `_linear_analysis.py`: shared dense generalized-eigenvalue utilities,
  finite-spectrum filtering, energy-norm transient-growth from modal
  propagators, the one-to-one eigenvalue set-matcher `match_eigenvalues`, and the
  shared IMEXRK Butcher tableaux / descriptor-system step core (`imex_tableau`,
  `imexrk_step`) used by both IMEXRK linear steppers.  Exposes the single shared
  modal-basis magnitude cap `FINITE_CAP = 1e8` used by every `nonmodal_growth`
  (it must sit above the most strongly damped physical mode yet below the
  spurious constraint modes at infinity; verified insensitive between `1e6` and
  `1e12`).  Pure NumPy/SciPy -- no shenfun.
- `_pcf_linear.py`: dense primitive-variable **Chebyshev-collocation** PCF
  operator (hydro, MHD, and the rotating-shear/MRI analogue).  Pure NumPy/SciPy.
  This is the operator the PCF DNS scripts wrap when `--linear` is `eigs` or
  `nonmodal`; `pcf_galerkin_linear.py` is the shenfun-Galerkin alternative (same
  primitive variables, different discretisation).  Also provides `cheb_lobatto`,
  reused by `taylor_couette_collocation.py`.
- `thin_gap_common.py`: scaling/annulus-map helpers (`ShearScales`,
  `annulus_for_plane_couette_limit`, `annulus_for_shearing_box_limit`) that
  reconcile the differing nondimensionalisations (PCF half-gap `h`/`U_wall` vs TC
  `Re=Omega1 R1 d/nu` vs local shearing-box `Omega0`, `S=q Omega0`).  No shenfun
  dependency, so it is importable by lightweight tests.
- `_demo_utils.py`: `default_thread_cap` caps OMP/OpenBLAS/MKL/NUMEXPR threads
  (default 2) to keep CLI runs laptop-friendly; override with
  `SHENFUN_DEMO_THREADS` (set empty to disable).

## Which approach to use

Many tasks can be done more than one way; the operators agree where they
overlap (that is what `thin_gap_compare.py` checks).  Pick by what you need:

| Task | Approaches (file / class) | When to prefer which |
|------|---------------------------|----------------------|
| PCF linear eigenvalues / non-modal | dense collocation `_pcf_linear.PlaneCouetteLinear`; Galerkin `pcf_galerkin_linear.PlaneCouetteGalerkinLinear`; or the DNS scripts' `--linear {eigs,nonmodal}` flag (wraps `_pcf_linear`) | collocation = lightweight, no shenfun, and the collocation↔collocation non-modal partner; Galerkin = same modal Gram norm as the TC Galerkin solvers; `--linear` = sanity-check from inside a DNS run |
| TC linear eigenvalues / non-modal | Galerkin `taylor_couette_linear.TaylorCouetteLinear` (+ `taylor_couette_mri.TaylorCouetteMRI` for MHD); dense collocation `taylor_couette_collocation.TaylorCouetteCollocationLinear`; or `taylor_couette_dns.py --linear-analysis {eigs,nonmodal}` (wraps the TC Galerkin operators) | Galerkin = production spectra, critical-Re/Rm, default; collocation = thin-gap cross-checks and the **only reliable MHD transient-growth partner** (it carries a magnetic-pressure projection) |
| Non-modal transient growth (matching) | must compare **within one discretisation** | Galerkin↔Galerkin (modal Gram norm) or collocation↔collocation (nodal quadrature norm); mixing injects ~10% norm bias.  For MHD magnetic/total energy use the collocation pairing; for the Galerkin pairing use `energy=kinetic` |
| DNS-style linear time stepping | `pcf_imexrk_linear.PlaneCouetteIMEXRKLinearStepper`, `taylor_couette_imexrk.TaylorCouetteIMEXRKLinearStepper` | the only path that exercises the time integrator itself apples-to-apples (shared IMEXRK via `_linear_analysis`); driven jointly by `thin_gap_compare.py --dns` |
| TC nonlinear DNS time stepper | CNAB2 `taylor_couette_dns.py` (axisym + 3D, hydro + MHD, **conducting walls only**); IMEXRK `taylor_couette_imexrk_dns.py` (axisymmetric, conducting MHD only) | CNAB2 for 3D / `m!=0`; IMEXRK companion to cross-check CNAB2 or match the PCF IMEXRK integrator.  Insulating walls are linear-analysis only (the operators + `--linear-analysis`), not the nonlinear DNS |
| PCF MHD field representation | primitive `b` + magnetic-pressure multiplier (linear operators); vector potential `A`, `B=curl(A)` (nonlinear DNS `pcf_mhd_divfree.py`) | multiplier/flux-function enforces `div(b)=0` for the linear eig/non-modal layer; vector potential gives `div(B)~1e-16` over long DNS integration |
| MHD wall BC | PCF: `conducting` or `dirichlet` (non-physical diagnostic); TC: `conducting` (any `m`) or `insulating` (vacuum match, flux-function, `m=0` only) | `conducting` is the physical default everywhere; PCF `dirichlet` = operator checks only; TC `insulating` = vacuum-matched walls (lower MRI `Rm_min`) |

### Energy norm for MHD non-modal growth

For MHD operators the transient-growth gain is measured in the **total**
(kinetic + magnetic, Alfven units) perturbation energy by default.  Select the
norm with `--linear-energy {total,kinetic,magnetic}` (PCF scripts) /
`--energy {total,kinetic,magnetic}` (Taylor-Couette scripts), or
`nonmodal_growth(..., energy=...)` in the API.  Because differential
rotation / shear stretches `b_r` into `b_theta` (the Omega-effect), the magnetic
field has its own transient growth, so the **total**-energy gain does *not*
reduce to the hydrodynamic value as the imposed field `B0 -> 0` -- it returns
the larger of the (then decoupled) kinetic and magnetic gains.  Use
`energy=kinetic` to recover the velocity-only gain, which *does* match the hydro
result at `B0=0` (see `couette_linear_benchmarks.md`).  A test pinning all
magnetic components (`magnetic_bc=dirichlet` in `_pcf_linear`) is a non-physical
diagnostic BC, not a conductor.

## Plane Couette Flow

### Standalone linear operators

The PCF linear-stability operators can be used directly, in addition to the
`--linear` flag on the DNS scripts below:

- `pcf_galerkin_linear.py` (`PlaneCouetteGalerkinLinear`): shenfun-Galerkin
  primitive-variable PCF operator (hydro + MHD), the Galerkin counterpart to the
  dense collocation `_pcf_linear.py`.  CLI: `--N --family {C,L} --Re --Rm --ky
  --kz --mhd --by --bz --omega --Uprime --magnetic-bc {conducting,dirichlet}
  --nonmodal --times --energy {total,kinetic,magnetic}`.
- `pcf_imexrk_linear.py` (`PlaneCouetteIMEXRKLinearStepper`): IMEXRK
  (111/222/443) linear time-stepper on the Galerkin operator -- the DNS-style
  path used by `thin_gap_compare.py --dns`.  CLI: `--mhd --N --family --Re --Rm
  --Uprime --omega --by --bz --ky --kz --dt --end-time --scheme
  {IMEXRK111,IMEXRK222,IMEXRK443} --split {diffusion,full} --energy`.

```bash
python pcf_galerkin_linear.py --mhd --ky 1 --kz 1 --Re 500 --bz 0.1 --nonmodal --times 1,5
python pcf_imexrk_linear.py --ky 0 --kz 1 --end-time 0.05   # measured vs eigenvalue growth
```

### Hydrodynamic

- `pcf_fluctuations_corrected.py`
  - Nonlinear/DNS: default mode.  Evolves hydrodynamic plane-Couette
    fluctuations with diagnostics, spectra, and plotting.
  - Linear eigenvalues: `--linear eigs`.
  - Linear non-modal growth: `--linear nonmodal`.

Examples:

```bash
python couette/pcf_fluctuations_corrected.py --linear eigs --linear-nx 48 --ky 1 --kz 1
python couette/pcf_fluctuations_corrected.py --linear nonmodal --linear-nx 80 --ky 0 --kz 1.66 --linear-times 139
python couette/pcf_fluctuations_corrected.py --no-save-plots --no-save-analysis --no-save-spectra
```

The hydrodynamic PCF DNS end time is currently fixed inside the script
(`end_time = 50.0`).

### Magnetohydrodynamic

- `pcf_mhd_divfree.py`
  - Nonlinear/DNS: default mode.  Evolves incompressible MHD plane-Couette flow
    with a vector-potential magnetic representation and divergence diagnostics.
  - Linear eigenvalues: `--linear eigs`.
  - Linear non-modal growth: `--linear nonmodal`.
  - Linear imposed fields use `--linear-by` and `--linear-bz`.
  - Energy norm: `--linear-energy {total,kinetic,magnetic}` (default `total`).
  - Magnetic wall BC: `--linear-magnetic-bc {conducting,dirichlet}`.

Examples:

```bash
python couette/pcf_mhd_divfree.py --linear eigs --linear-nx 48 --ky 1 --kz 1 --linear-bz 0.1
python couette/pcf_mhd_divfree.py --linear nonmodal --linear-nx 48 --ky 1 --kz 1 --linear-bz 0.1 --linear-times 0,1,5
python couette/pcf_mhd_divfree.py --linear nonmodal --linear-nx 48 --ky 1 --kz 1 --linear-energy kinetic
python couette/pcf_mhd_divfree.py --end-time 0.1 --moderror 10
```

- `pcf_mhd_mri_shearpy.py`
  - Nonlinear/DNS: default mode.  Plane-Couette MHD analogue of the shearing-box
    MRI setup, with rotation, imposed shear, and net imposed magnetic field.
  - Linear eigenvalues: `--linear eigs`.
  - Linear non-modal growth: `--linear nonmodal`.
  - Energy norm: `--linear-energy {total,kinetic,magnetic}` (default `total`).
  - Use this script for the PCF rotating-shear/MRI linear check.

Examples:

```bash
python couette/pcf_mhd_mri_shearpy.py --linear eigs --linear-nx 48 --ky 0 --kz 25.81988897471611 --Re 1000000 --Rm 1000000
python couette/pcf_mhd_mri_shearpy.py --linear nonmodal --linear-nx 32 --ky 0 --kz 6.283185307179586 --linear-times 0,1,5
python couette/pcf_mhd_mri_shearpy.py --linear nonmodal --linear-nx 32 --ky 0 --kz 6.283185307179586 --linear-energy magnetic
python couette/pcf_mhd_mri_shearpy.py --end-time 0.01 --store-history
```

## Taylor-Couette Flow

### Standalone linear operators

In addition to the Galerkin `taylor_couette_linear.py` / `taylor_couette_mri.py`
(documented below) and the `taylor_couette_dns.py --linear-analysis` flag:

- `taylor_couette_collocation.py` (`TaylorCouetteCollocationLinear`): dense
  Chebyshev-collocation TC operator (hydro + MHD), the collocation counterpart to
  the Galerkin solvers; supports `conducting` + `insulating` walls.  It is the
  collocation↔collocation partner in `thin_gap_compare.py` and the reliable MHD
  transient-growth reference (carries a magnetic-pressure projection).  CLI: `--N
  --R1 --R2 --Omega1 --Omega2 --nu --eta-mag --B0 --m --kz --mhd --magnetic-bc
  {conducting,insulating} --nonmodal --times --energy`.
- `taylor_couette_imexrk.py` (`TaylorCouetteIMEXRKLinearStepper`): IMEXRK linear
  time-stepper on the TC Galerkin operators -- the TC side of `--dns`.  CLI:
  `--mhd --magnetic-bc {conducting,insulating} --R1 --R2 --Omega1 --Omega2 --N
  --family --nu --eta-mag --B0 --m --kz --dt --end-time --scheme --split
  --energy`.

### Hydrodynamic

- `taylor_couette_linear.py`
  - Linear eigenvalues: default mode with either a fixed `--kz` or a scan over
    `--kz-min`, `--kz-max`, and `--kz-num`.
  - Linear non-modal growth: `--nonmodal`.
  - Includes critical-Reynolds utilities on `TaylorCouetteLinear`.

Examples:

```bash
python couette/taylor_couette_linear.py --N 48 --family C --kz 3.16
python couette/taylor_couette_linear.py --N 80 --family C --nonmodal --kz 1.5707963267948966 --times 5,10,20
```

- `taylor_couette_dns.py`
  - Nonlinear/DNS: default hydrodynamic mode.
  - Linear hydrodynamic eigenvalues: `--linear-analysis eigs`.
  - Linear hydrodynamic non-modal growth: `--linear-analysis nonmodal`.
  - This script can be used as the common DNS entry point when comparing linear
    predictions to nonlinear runs.

Examples:

```bash
python couette/taylor_couette_dns.py --linear-analysis eigs --Nr 48 --family C --kz 3.16
python couette/taylor_couette_dns.py --linear-analysis nonmodal --Nr 80 --family C --kz 1.5707963267948966 --times 5,10,20
python couette/taylor_couette_dns.py --Nr 32 --Nz 64 --end-time 1.0
```

### Magnetohydrodynamic

- `taylor_couette_mri.py`
  - Linear eigenvalues: default mode with a fixed `--kz` or an axial-wavenumber
    scan.
  - Linear non-modal growth: `--nonmodal`.
  - Energy norm: `--energy {total,kinetic,magnetic}` (default `total`).
  - MHD walls: `--magnetic-bc conducting` or `--magnetic-bc insulating`
    (insulating uses the flux-function formulation and is `m=0` only).
  - Local analytic MRI check: `--local-check`.

Examples:

```bash
python couette/taylor_couette_mri.py --local-check
python couette/taylor_couette_mri.py --N 32 --family C --magnetic-bc conducting --kz 1.75 --B0 0.16639676113360324 --eta-mag 0.04048582995951417 --nu 4.048582995951417e-08
python couette/taylor_couette_mri.py --N 32 --family C --magnetic-bc insulating --nonmodal --kz 1.25 --times 0,1,5
python couette/taylor_couette_mri.py --N 32 --family C --nonmodal --kz 1.5 --times 5 --energy kinetic
```

- `taylor_couette_dns.py --mhd`
  - Nonlinear/DNS: axisymmetric or full 3D MHD Taylor-Couette DNS depending on
    `--Ntheta`.
  - Linear MHD/MRI eigenvalues: `--mhd --linear-analysis eigs`.
  - Linear MHD/MRI non-modal growth: `--mhd --linear-analysis nonmodal`.
  - Magnetic wall BC: `--magnetic-bc {conducting,insulating}` (insulating is
    `m=0` only); energy norm: `--energy {total,kinetic,magnetic}`.

Examples:

```bash
python couette/taylor_couette_dns.py --mhd --linear-analysis eigs --Nr 32 --family C --kz 1.75 --B0 0.16639676113360324 --eta-mag 0.04048582995951417 --nu 4.048582995951417e-08
python couette/taylor_couette_dns.py --mhd --linear-analysis nonmodal --Nr 32 --family C --kz 1.25 --times 0,1,5
python couette/taylor_couette_dns.py --mhd --linear-analysis nonmodal --Nr 32 --family C --kz 1.25 --magnetic-bc insulating --energy kinetic
python couette/taylor_couette_dns.py --mhd --Nr 32 --Nz 64 --end-time 0.1 --B0 0.1
```

- `taylor_couette_imexrk_dns.py`
  - Axisymmetric IMEXRK companion to `taylor_couette_dns.py` (same axisymmetric
    hydro / conducting-wall MHD formulation), using shenfun-style IMEXRK
    (`IMEXRK111/222/443`) stage solves instead of CNAB2.  **Axisymmetric only**
    (no `--Ntheta`); MHD is conducting-wall only.  Use it to cross-check the
    CNAB2 result or to match the PCF IMEXRK integrator.
  - CLI: `--mhd --Nr --Nz --Lz --family --nu --eta-mag --B0 --dt --end-time
    --timestepper {IMEXRK111,IMEXRK222,IMEXRK443} --seed-linear --kz-mode --amp`.

```bash
python couette/taylor_couette_imexrk_dns.py --Nr 32 --Nz 16 --nu 1e-2 --dt 1e-3 --end-time 0.1 --seed-linear --kz-mode 1
python couette/taylor_couette_imexrk_dns.py --mhd --B0 0.1 --eta-mag 1e-3 --timestepper IMEXRK222 --seed-linear
```

## Apples-to-Apples Thin-Gap Comparison

`thin_gap_compare.py` puts the plane-Couette and Taylor-Couette operators on a
common footing and reports the comparison at **three** levels, all in the local
frame rotating at the mid-gap `Omega0` (via the analytic Doppler shift).  Note
`--limit` defaults to **`plane`** (the *singular* non-rotating case -- see the
warning below); pass `--limit shearing` for the recommended apples-to-apples
test.  The default `--curvature` is `0.02` (fine for the shearing limit); reduce
it to `1e-4` or below for the plane limit.

```bash
# eigenvalues only -- shearing (rotating) limit is the recommended test
python couette/thin_gap_compare.py --limit shearing --Omega 0.6666666667 --S 1 --ky 0 --kz 1
# add non-modal transient growth and the DNS-style time-stepped check
python couette/thin_gap_compare.py --limit shearing --mhd --B0 0.05 --nonmodal --dns
```

1. **Linear eigenvalues.**  Reported as a *set-matched* full-complex residual:
   the leading PCF eigenvalue is matched to its nearest neighbour in each TC
   spectrum (`_linear_analysis.match_eigenvalues`).  This is robust to the
   arbitrary `+/-`-frequency tie-break of a (near-)conjugate-symmetric spectrum
   and to the orientation convention (PCF `+ky` vs TC `+m`); comparing the raw
   "leading" eigenvalue could otherwise show a spurious frequency-sign flip.

2. **Linear non-modal growth** (`--nonmodal`).  Optimal transient energy gain in
   a shared norm (frame-independent, so no Doppler conversion).  Transient growth
   is norm-discretisation sensitive, so it is only apples-to-apples *within a
   consistent discretisation*: **Galerkin↔Galerkin** (modal Gram norm) and
   **collocation↔collocation** (nodal quadrature norm) are reported separately.
   For MHD, the magnetic/total transient growth is reliable only in the
   collocation↔collocation pairing -- both carry a magnetic-pressure solenoidal
   projection, whereas the TC-Galerkin conducting-wall MRI operator has none and
   overstates the magnetic transient growth.  Use `--energy kinetic` for the
   Galerkin pairing.

3. **DNS-style** (`--dns`).  Both geometries are advanced by the **same** IMEXRK
   integrator (`pcf_imexrk_linear.PlaneCouetteIMEXRKLinearStepper` and
   `taylor_couette_imexrk.TaylorCouetteIMEXRKLinearStepper`, sharing
   `_linear_analysis.imex_tableau` / `imexrk_step`), seeded with the leading
   eigenmode; the measured growth rates are compared.  This is the apples-to-apples
   *time-stepping* path the eigenvalue/non-modal checks cannot exercise.

**Rotating vs non-rotating limit.**  `--limit shearing` (mid-gap `Omega0 != 0`)
is Rayleigh-stable and is the natural apples-to-apples test; the operators agree
to `~1e-6` (eigenvalues, DNS growth) at `curvature=0.01`.  `--limit plane`
(`Omega0 = 0`) is a **singular** limit: the non-rotating mid-gap forces the inner
region to rotate retrograde, so `kappa^2 = 4 a Omega(r) < 0` near `R1` and the
annulus is centrifugally (Taylor) unstable for *any* finite curvature.  The tool
prints the Rayleigh diagnostic and warns; the residual is `O(curvature)`, so use
a small `--curvature` (`1e-4` or below) for the plane limit to converge.

**Full nonlinear DNS.**  The nonlinear DNS solvers are *not* yet a single shared
code path -- PCF uses the Kim-Moin-Moser velocity-vorticity formulation
(`ChannelFlow.KMM`) with a magnetic **vector potential** `A`, while the TC DNS
uses **primitive** `(u_r,u_theta,u_z,p)` / `(b_r,b_theta,b_z)`.  Both, however,
can be advanced with **IMEXRK222** (PCF by default; TC via
`taylor_couette_imexrk_dns.py`), and both reproduce their own linear eigenvalue
when seeded at small amplitude (the `--seed-linear` smoke tests).  The validated
linear bridge (level 1 above) then closes the loop between the two geometries.

## Porting to another shenfun codebase

The Couette files form layered clusters; copy the smallest set that gives the
capability you need (verified from the import graph).  External-dependency
legend: `np`=NumPy, `sl`=`scipy.linalg`, `so`=`scipy.optimize`,
`ss`=`scipy.special`, `sp`=SymPy, `shenfun`, `ChannelFlow.py` (a local file that
does `from shenfun import *`).

| Cluster | Files | Intra-repo deps | External | Gives |
|---------|-------|-----------------|----------|-------|
| **1. Pure-NumPy linear core** | `_linear_analysis.py`, `_pcf_linear.py` | (self-contained) | np, sl, so | PCF dense-collocation eigs/non-modal (hydro+MHD), `match_eigenvalues`, `imex_tableau`, `imexrk_step`, transient growth.  **No shenfun.**  Most portable unit; `cheb_lobatto` here is reused by cluster 3. |
| **2. shenfun PCF linear** | `pcf_galerkin_linear.py`, `pcf_imexrk_linear.py` | cluster 1 | np, sp, shenfun, sl | PCF Galerkin eigs/non-modal + `assemble_parts` + DNS-style IMEXRK stepper |
| **3. shenfun TC linear** | `taylor_couette_linear.py`, `taylor_couette_mri.py`, `taylor_couette_collocation.py`, `taylor_couette_imexrk.py` | cluster 1 (`_pcf_linear.cheb_lobatto` too), `_demo_utils.py` | np, sp, shenfun, sl, ss | TC hydro + MHD/MRI Galerkin (conducting any-`m` / insulating `m=0`), TC collocation, TC IMEXRK stepper, critical-Re/Rm.  Override `family` to `'C'` (Legendre DLT needs the compiled `Leg2Cheb`). |
| **4. PCF nonlinear DNS** | `ChannelFlow.py`, `pcf_fluctuations_corrected.py`, `pcf_fluctuations_divV.py`, `pcf_mhd_divfree.py`, `pcf_mhd_mri_shearpy.py` | each `pcf_*` → `ChannelFlow.KMM`; `mri_shearpy` → `pcf_mhd_divfree`; `--linear` paths lazily pull cluster 1 | np, shenfun, `shenfun.optimization.numba`, matplotlib (corrected) | KMM velocity-vorticity hydro DNS, `div(u)` diagnostics, MHD vector-potential DNS, MRI shearing-box DNS.  `ChannelFlow.py` is the shared base. |
| **5. TC nonlinear DNS** | `taylor_couette_dns.py`, `taylor_couette_imexrk_dns.py` | `taylor_couette_dns.py` → `taylor_couette_linear.CircularCouette` eager + cluster 3 lazily (seeding / `--linear-analysis`); `taylor_couette_imexrk_dns.py` → `taylor_couette_imexrk` **eager**, which pulls `taylor_couette_linear` + `taylor_couette_mri` + `_linear_analysis` eagerly (i.e. cluster 3 **minus** `taylor_couette_collocation`); `_demo_utils.py` | np, shenfun; **sp + ss become eager** once `taylor_couette_imexrk_dns.py` is included (via `taylor_couette_mri`), else needed only when seeding | CNAB2 axisym+3D hydro & MHD/MRI (conducting walls only); axisymmetric IMEXRK companion (hydro + conducting MHD).  Insulating walls are linear-analysis only.  **Does not use `ChannelFlow.py`.** |
| **6. Comparison driver** | `thin_gap_common.py`, `thin_gap_compare.py` | clusters 1+2+3 | np, sp, shenfun, sl, so, ss | apples-to-apples eigenvalue / `--nonmodal` / `--dns` driver.  `thin_gap_common.py` alone (no shenfun) is a trivially portable scalings/annulus-map unit. |

Notes for porters:
- Clusters 2, 3, 6 all sit on cluster 1; cluster 3's collocation file uniquely
  pulls `_pcf_linear` back in (for `cheb_lobatto`), so "TC linear" needs both
  `_linear_analysis.py` and `_pcf_linear.py`.
- `ChannelFlow.py` is exclusive to the PCF DNS (cluster 4); the TC DNS has its
  own four classes and does **not** subclass `KMM`.
- Cluster 5's IMEXRK companion makes cluster 3 an **eager** dependency, not a
  lazy one: `taylor_couette_imexrk_dns.py` imports `imex_tableau` from
  `taylor_couette_imexrk.py` at module top, which itself imports
  `taylor_couette_linear` and `taylor_couette_mri` at module top.  So copying
  cluster 5 with the IMEXRK companion requires `taylor_couette_imexrk`,
  `taylor_couette_linear`, `taylor_couette_mri` and `_linear_analysis` (cluster 3
  minus `taylor_couette_collocation`) up front — even before any seeding or
  `--linear-analysis` call.  Copying only `taylor_couette_dns.py` (CNAB2) keeps
  cluster 3 lazy.
- The linear clusters (1-3, 6) are independent of the DNS clusters (4-5).
- Tests: `test_couette_linear.py`, `test_thin_gap_comparison.py`,
  `test_fastgl_fallback.py` (linear/comparison); `test_taylor_couette.py`,
  `test_taylor_couette_dns.py`, `test_pcf_mhd_divfree.py`,
  `test_pcf_mhd_mri_shearpy.py` (solvers).

## Benchmark Notes

See `couette_linear_benchmarks.md` for the literature references, expected
growth rates, and reproducible benchmark commands used to validate the linear
and non-modal additions.

> **Environment note.**  The Taylor-Couette modal solvers default to the Legendre
> family (`family="L"`), which relies on shenfun's compiled extensions.
> `shenfun.legendre.fastgl` now falls back to a NumPy Gauss-Legendre rule when the
> compiled `fastgl_wrap` is missing (it catches the bare `ImportError: cannot
> import name 'fastgl_wrap' ... circular import`, not only `ModuleNotFoundError`),
> which removes that crash and the `getGLPair`-is-`None` trap.  Note this is only a
> *partial* source-only Legendre fix: the Legendre discrete transform (`DLT`) also
> needs the compiled `Leg2Cheb` from `shenfun.optimization.cython`, so a checkout
> without built extensions still cannot construct a `family="L"` space (it fails
> at `Leg2Cheb`).  Use a built/installed shenfun for Legendre, or `family="C"`
> (Chebyshev), which needs no compiled extension.  The dense linear/non-modal
> comparison layer and `thin_gap_compare.py` pin `family="C"` so the PCF and TC
> operators use an identical radial basis; the three PCF DNS scripts also default
> to `"C"`, while the TC DNS keeps its natural Galerkin `"L"` basis (pass
> `--family` to match when cross-comparing).
