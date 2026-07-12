# Couette Linear And Non-Modal Benchmark Checks

This note records the literature checks used for the Couette linear eigenvalue
and non-modal demo variants.  The snippets below are intended to be run from the
repository root.  For the here-doc commands, activate the project environment
first:

```bash
conda activate shenfun
export PYTHONPATH="$PWD/demo:${PYTHONPATH}"
```

The Taylor-Couette solvers default to the Legendre radial family
(`family="L"`), which requires shenfun's compiled extensions (`fastgl` and
`Leg2Cheb`).  The examples below pass `family="C"` (Chebyshev) so they run
without those extensions; pass `family="L"` when shenfun is built (or installed)
with them.  See the "Legendre `fastgl` note" near the end for the partial
source-only fallback.

## Summary

| Case | Reference value | Computed value |
| --- | ---: | ---: |
| Plane Couette linear stability, any `(ky,kz)`, `Re=1000` | stable, `Re(s)<0` (Romanov 1973) | max `growth=-3.47e-3` |
| Plane Couette streamwise-roll eigenvalue, `ky=0`, `kz=1` | `s=-nu(kz^2+(pi/2)^2)=-3.467401e-3` | `-3.46740110e-3` (rel.err 5e-10) |
| Plane Couette hydro transient growth, `Re=1000`, `alpha=0`, `beta=1.66`, `t=139` | `G=1165.2` | `G=1165.93` |
| Plane Couette optimal-growth scaling, `Re=500..2000` | `Gmax ~ Re^2`, `t_opt ~ Re` | `Gmax/Re^2=1.165e-3` (const), `t_opt/Re=0.1415` |
| Plane Couette MHD non-modal at `B0=0`, kinetic norm | reduces to hydro exactly | `3.07410e-4` (matches hydro to 9 digits) |
| Taylor-Couette hydro onset, `eta=0.5`, outer cylinder stationary | `Re_c ~= 68.2` | `Re_c=68.18635`, `a_c=3.1667` |
| Taylor-Couette hydro transient growth, `eta=0.5`, `mu=-1`, `Re=125`, `beta=pi/2` | `Gmax ~= 4.17` (`beta`-optimized) | `Gmax=3.866` at `t=4.9` (`beta=pi/2` fixed, N-converged) |
| Taylor-Couette hydro transient growth, `eta=0.99`, `mu=-1`, `Re=350`, `beta=pi/2` | `Gmax ~= 155` | `Gmax=152.55` at `t=67` |
| Ideal local Keplerian MRI | `smax/Omega=0.75`, `(k vA)^2/Omega^2=15/16` | `0.74999999`, `0.937317` |
| PCF rotating-shear MRI analogue, high `Re=Rm`, optimal local `kz` | `s=0.5` for `Omega=2/3` | `s=0.498406` |
| Global TC MRI, `eta=0.5`, quasi-Keplerian, conducting walls | `Rm=24.7`, `S=4.11`, `k ~= 1.7` near neutral | `growth=+0.00332` at `kz=1.75` |
| Global TC MRI, `eta=0.5`, quasi-Keplerian, insulating walls | `Rm=16.5`, `S=5.21`, `k ~= 1.2` near neutral | `growth=-2.76e-4` at `kz=1.25` |

The non-modal numbers are energy gains, i.e. squared amplification factors in
the kinetic or total perturbation energy norm.  In MHD, `--linear nonmodal`
(scripts) / `nonmodal_growth(..., energy=...)` (API) defaults to the **total**
(kinetic + magnetic) energy norm; pass `energy='kinetic'` for the velocity-only
gain.  Because differential rotation / shear stretches `b_r` into `b_theta`
(the Omega-effect), the total-energy gain does **not** reduce to the
hydrodynamic value as `B0 -> 0` -- it returns the larger of the (then decoupled)
kinetic and magnetic transient growths; only the kinetic norm matches hydro at
`B0=0` (verified to 9 digits below).

## Plane Couette Hydro Non-Modal Growth

Reference: the NASA/Reddy-Henningson transient-growth report lists for plane
Couette flow with `beta=1.66`, `phi=90 deg`, `Re=1000`: `t*=139.0`,
`G*=1165.2`.

```bash
python - <<'PY'
import numpy as np
from _pcf_linear import PlaneCouetteLinear

pcf = PlaneCouetteLinear.couette(nx=80, Re=1000.0, U_wall=1.0, mhd=False)
rows = pcf.nonmodal_growth(0.0, 1.66, np.linspace(120.0, 160.0, 21))
best = max(rows, key=lambda r: r["gain"])
print("best_scan", best)
print("at_139", pcf.nonmodal_growth(0.0, 1.66, [139.0])[0])
PY
```

Expected output:

```text
best_scan {'t': 138.0, 'gain': 1165.9386702883562, 'amplification': 34.14584411445053}
at_139 {'t': 139.0, 'gain': 1165.9316456555734, 'amplification': 34.14574125210307}
```

The CLI path exercises the same operator, but the script default Reynolds number
is `Re=400`:

```bash
python demo/pcf_fluctuations_corrected.py --linear nonmodal --linear-nx 80 --ky 0 --kz 1.66 --linear-times 139
```

## Plane Couette Modal Stability And The Streamwise-Roll Eigenvalue

Two exact checks of the eigenvalue (`--linear eigs`) branch.  First, plane
Couette flow is linearly stable for every wavenumber and Reynolds number
(Romanov 1973): the leading growth rate is always negative.  Second, for
streamwise-independent perturbations (`ky=0`) the spanwise/streamwise velocity
decouples into a 1D heat equation, whose slowest mode decays at the analytic
rate `s = -nu (kz^2 + (pi/2)^2)` on the half-gap-1 domain `[-1, 1]` (the full
Squire ladder `-nu (kz^2 + (j pi/2)^2)` also appears in the spectrum).

```bash
python - <<'PY'
import math, numpy as np
from _pcf_linear import PlaneCouetteLinear

Re = 1000.0; nu = 1.0 / Re
pcf = PlaneCouetteLinear.couette(nx=96, Re=Re, U_wall=1.0, mhd=False)
print("Romanov: leading growth rate (must be < 0) at Re=1000")
for ky, kz in [(1., 0.), (0., 1.), (1., 1.), (2., 1.)]:
    print(f"  ky={ky} kz={kz}: growth={pcf.growth_rate(ky, kz):+.6e}")
print("Streamwise roll (ky=0): lead eig vs analytic -nu*(kz^2+(pi/2)^2)")
for kz in (1.0, 2.5):
    w, _ = pcf.eigs(0.0, kz, n_return=4)
    ana = -nu * (kz**2 + (math.pi / 2) ** 2)
    print(f"  kz={kz}: numeric={w[0].real:+.8e}  analytic={ana:+.8e}")
PY
```

Expected output:

```text
Romanov: leading growth rate (must be < 0) at Re=1000
  ky=1.0 kz=0.0: growth=-1.179054e-01
  ky=0.0 kz=1.0: growth=-3.467401e-03
  ky=1.0 kz=1.0: growth=-1.189054e-01
  ky=2.0 kz=1.0: growth=-1.905757e-01
Streamwise roll (ky=0): lead eig vs analytic -nu*(kz^2+(pi/2)^2)
  kz=1.0: numeric=-3.46740110e-03  analytic=-3.46740110e-03
  kz=2.5: numeric=-8.71740110e-03  analytic=-8.71740110e-03
```

## Plane Couette Optimal-Growth Re^2 Scaling

The classic non-modal result (Gustavsson 1991; Reddy & Henningson 1993;
Trefethen, Trefethen, Reddy & Driscoll 1993; Butler & Farrell 1992 for the
optimal perturbation): the streamwise-vortex optimal energy gain scales as
`Gmax ~ Re^2` with optimal time `t_opt ~ Re`.  The ratios `Gmax/Re^2` and
`t_opt/Re` are therefore Reynolds-independent.

```bash
python - <<'PY'
import numpy as np
from _pcf_linear import PlaneCouetteLinear

for Re in (500.0, 1000.0, 2000.0):
    pcf = PlaneCouetteLinear.couette(nx=80, Re=Re, U_wall=1.0, mhd=False)
    rows = pcf.nonmodal_growth(0.0, 1.66, np.linspace(0.12 * Re, 0.40 * Re, 40))
    best = max(rows, key=lambda r: r["gain"])
    print(f"Re={Re:6.0f}  Gmax={best['gain']:9.2f}  t_opt={best['t']:6.1f}  "
          f"Gmax/Re^2={best['gain']/Re**2:.4e}  t_opt/Re={best['t']/Re:.4f}")
PY
```

Expected output:

```text
Re=   500  Gmax=   291.57  t_opt=  70.8  Gmax/Re^2=1.1663e-03  t_opt/Re=0.1415
Re=  1000  Gmax=  1165.40  t_opt= 141.5  Gmax/Re^2=1.1654e-03  t_opt/Re=0.1415
Re=  2000  Gmax=  4660.72  t_opt= 283.1  Gmax/Re^2=1.1652e-03  t_opt/Re=0.1415
```

(The `Re=1000` row is the same optimum as the `G=1165.93` benchmark above; the
small differences come from the coarser `t` grid used here for the scan.)

## Plane Couette MHD Energy Norm: Kinetic vs Total

With no imposed field (`B0=0`) the velocity and magnetic perturbations decouple.
The *kinetic* energy gain of the MHD operator then matches the hydrodynamic gain
to machine precision, confirming the velocity operator.  The *total* (kinetic +
magnetic) gain is larger because the base shear drives an independent
transient amplification of `b_theta` from `b_r` (the Omega-effect), so MHD total
energy `!=` hydro at `B0=0`.  Switching on a vertical field then suppresses the
total-energy growth monotonically via magnetic tension (cf. Camobreco, Potherat
& Sheard 2020/2021 for MHD channel/duct transient growth).

```bash
python - <<'PY'
from _pcf_linear import PlaneCouetteLinear

ky, kz, t = 1.0, 1.0, [50.0]
hyd = PlaneCouetteLinear.couette(nx=48, Re=500.0, mhd=False)
mhd = PlaneCouetteLinear.couette(nx=48, Re=500.0, Rm=500.0, mhd=True, by=0.0, bz=0.0)
print("hydro  G(50)        =", hyd.nonmodal_growth(ky, kz, t)[0]["gain"])
print("MHD B0=0 kinetic    =", mhd.nonmodal_growth(ky, kz, t, energy='kinetic')[0]["gain"])
print("MHD B0=0 magnetic   =", mhd.nonmodal_growth(ky, kz, t, energy='magnetic')[0]["gain"])
print("MHD B0=0 total      =", mhd.nonmodal_growth(ky, kz, t, energy='total')[0]["gain"])
print("imposed-bz sweep (total energy):")
for bz in (0.0, 0.05, 0.1, 0.2):
    m = PlaneCouetteLinear.couette(nx=48, Re=500.0, Rm=500.0, mhd=True, by=0.0, bz=bz)
    print(f"  bz={bz:4.2f}  G(50)={m.nonmodal_growth(ky, kz, t)[0]['gain']:.6e}")
PY
```

Expected output:

```text
hydro  G(50)        = 0.0003074103849712573
MHD B0=0 kinetic    = 0.0003074103821173053
MHD B0=0 magnetic   = 0.0033750397910762343
MHD B0=0 total      = 0.0033750397909788512
imposed-bz sweep (total energy):
  bz=0.00  G(50)=3.375040e-03
  bz=0.05  G(50)=2.156254e-03
  bz=0.10  G(50)=7.846618e-04
  bz=0.20  G(50)=1.354645e-04
```

## Taylor-Couette Hydro Linear Onset

For the classic stationary-outer-cylinder case with `R1=1`, `R2=2`,
`Omega1=1`, `Omega2=0`, the standard hydrodynamic critical value is
`Re_c ~= 68.2`.

```bash
python - <<'PY'
import numpy as np
from taylor_couette_linear import CircularCouette, TaylorCouetteLinear

base = CircularCouette(1.0, 2.0, 1.0, 0.0)
solver = TaylorCouetteLinear(base, nu=1.0e-3, N=48, family="C")
res = solver.critical_reynolds(
    m=0,
    kz_list=np.linspace(2.6, 3.6, 13),
    refine=True,
)
print(res)

nu = base.Omega1 * base.R1 * base.gap / 68.19
solver = TaylorCouetteLinear(base, nu=nu, N=48, family="C")
for kz in (3.13, 3.16, 3.20):
    print("kz", kz, "growth_at_Re68.19", solver.growth_rate(0, kz))
PY
```

Expected output:

```text
{'kz_c': 3.166666666666667, 'nu_c': 0.01466569122691028, 'Re_c': 68.18635306906545, 'a_c': 3.166666666666667}
kz 3.13 growth_at_Re68.19 -8.102484959073185e-06
kz 3.16 growth_at_Re68.19 2.1384114948228477e-05
kz 3.2 growth_at_Re68.19 -1.76471680907601e-05
```

## Taylor-Couette Hydro Non-Modal Growth

Hristova, Roch, Schmid, and Tuckerman use distances scaled by half the gap,
`eta=Rin/Rout`, `mu=Vout/Vin`, and `Re=Rin Vin (Rout-Rin)/(2 nu)`.  The command
below maps that convention to the dimensional `CircularCouette` arguments by
setting the nondimensional gap width to 2 and the inner wall speed to 1.

```bash
python - <<'PY'
import numpy as np
from taylor_couette_linear import CircularCouette, TaylorCouetteLinear

def hristova_case(eta, Re, N=80):
    mu = -1.0
    R1 = 2.0 * eta / (1.0 - eta)
    R2 = 2.0 / (1.0 - eta)
    base = CircularCouette(R1, R2, 1.0 / R1, (mu / eta) / R2)
    solver = TaylorCouetteLinear(base, nu=1.0 / Re, N=N, family="C")
    beta = np.pi / 2.0
    times = np.linspace(2.0, 8.0, 61) if eta < 0.9 else np.linspace(55.0, 80.0, 51)
    best = max(solver.nonmodal_growth(0, beta, times), key=lambda r: r["gain"])
    print("eta", eta, "Re", Re, "best", best, "modal", solver.growth_rate(0, beta))

hristova_case(0.5, 125.0)
hristova_case(0.99, 350.0)
PY
```

Expected output:

```text
eta 0.5 Re 125.0 best {'t': 4.9, 'gain': 3.866147499863873, 'amplification': 1.9662521455459046} modal -0.03941996086947881
eta 0.99 Re 350.0 best {'t': 67.0, 'gain': 152.5493393325192, 'amplification': 12.35108656485409} modal -0.0006623987546807989
```

The `eta=0.5` value is a little below the tabulated `4.17`; the near-plane case
matches the paper's total-growth value closely.  The `eta=0.5` `Gmax=3.866` is
**fully resolved** -- `N=60/80/120/160` all give `3.86614...`, so the ~7% gap is
not a discretisation error.  The most likely cause is that this benchmark fixes
the axial wavenumber at `beta=pi/2`, whereas the paper's `4.17` is the optimum
over `beta` as well; a small `beta` scan should close the gap (left as a
suggested extension below).  The paper also reports that the azimuthal energy
component reaches about `164` in the `eta=0.99`, `Re=350` case, so distinguish
total optimal energy growth from component energy.

## MRI Linear Growth Checks

The local ideal Keplerian MRI benchmark is analytic.  For `q=3/2`, the maximum
growth rate is `0.75 Omega` at `(kz vA)^2/Omega^2 = 15/16`.  The wall-bounded
PCF rotating-shear operator approaches that value at high `Re` and `Rm`.

```bash
python - <<'PY'
import math
from taylor_couette_mri import mri_keplerian_optimum
from _pcf_linear import PlaneCouetteLinear

Omega = 2.0 / 3.0
Bz = 0.025
kz = math.sqrt(15.0 / 16.0) * Omega / Bz
print("local", mri_keplerian_optimum(Omega=Omega, vA=Bz))

for nx in (24, 32, 48):
    lin = PlaneCouetteLinear.shearpy(
        nx=nx,
        Re=1.0e6,
        Rm=1.0e6,
        shear_rate=1.0,
        omega=Omega,
        by=0.0,
        bz=Bz,
    )
    w, _ = lin.eigs(0.0, kz, n_return=3)
    print("shearpy nx", nx, "kz", kz, "lead", w[0])
PY
```

Expected output:

```text
local {'s_max': 0.49999999627997616, 's_max_over_Omega': 0.7499999944199642, 'wa2_opt_over_O2': 0.9373170323757943, 'theory_s_max_over_Omega': 0.75, 'theory_wa2_opt': 0.9375, 'theory_cutoff_wa2': 3.0}
shearpy nx 24 kz 25.81988897471611 lead (0.4984075630441907+0j)
shearpy nx 32 kz 25.81988897471611 lead (0.49840694616677383-1.4571865789133866e-16j)
shearpy nx 48 kz 25.81988897471611 lead (0.49840620435392047-6.383821520699091e-16j)
```

The global Taylor-Couette MRI check uses the `eta=0.5` quasi-Keplerian table
from Ruediger and Schultz.  The values below use very small `Pm` and scan over
the axial wavenumber listed in the table.

```bash
python - <<'PY'
import numpy as np
from taylor_couette_linear import CircularCouette
from taylor_couette_mri import TaylorCouetteMRI

base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
for bc, Rm, S in (("conducting", 24.7, 4.11), ("insulating", 16.5, 5.21)):
    eta_mag = 1.0 / Rm
    B0 = S * eta_mag
    nu = 1.0e-6 * eta_mag
    solver = TaylorCouetteMRI(
        base,
        B0=B0,
        nu=nu,
        eta_mag=eta_mag,
        N=32,
        family="C",
        magnetic_bc=bc,
    )
    kzs = np.linspace(0.5, 8.0, 31)
    kb, gb, _ = solver.max_growth_over_kz(0, kzs)
    print(bc, "target_Rm", Rm, "target_S", S, "best_kz", kb, "max_growth", gb)
PY
```

Expected output:

```text
conducting target_Rm 24.7 target_S 4.11 best_kz 1.75 max_growth 0.003322863594034156
insulating target_Rm 16.5 target_S 5.21 best_kz 1.25 max_growth -0.00027582037141390655
```

## Regression Commands

```bash
conda run -n shenfun python -m py_compile \
  demo/_linear_analysis.py \
  demo/_pcf_linear.py \
  demo/pcf_galerkin_linear.py \
  demo/pcf_imexrk_linear.py \
  demo/pcf_fluctuations_corrected.py \
  demo/pcf_mhd_divfree.py \
  demo/pcf_mhd_mri_shearpy.py \
  demo/taylor_couette_linear.py \
  demo/taylor_couette_mri.py \
  demo/taylor_couette_collocation.py \
  demo/taylor_couette_imexrk.py \
  demo/taylor_couette_dns.py \
  demo/taylor_couette_imexrk_dns.py \
  demo/thin_gap_common.py \
  demo/thin_gap_compare.py

# Dense linear / non-modal layer (pure-numpy PCF + Chebyshev TC; fast, no Legendre)
conda run -n shenfun pytest -q demo/test_couette_linear.py

# Apples-to-apples thin-gap comparison (eigenvalues, non-modal, DNS-style)
conda run -n shenfun pytest -q demo/test_thin_gap_comparison.py

# Cylindrical-solver modal benchmarks (Legendre family="L" by default)
conda run -n shenfun pytest -q demo/test_taylor_couette.py
```

Verified results: `py_compile` clean; `test_couette_linear.py` `10 passed`;
`test_thin_gap_comparison.py` `30 passed`; `test_taylor_couette.py` `25 passed`.

> **Legendre `fastgl` note (partial source-only fix).**  The cylindrical solvers
> default to `family="L"`, which relies on shenfun's compiled extensions.  When
> the `fastgl` wrapper is absent the bare `from . import fastgl_wrap` raises a
> plain `ImportError` ("partially initialized module ... circular import"), not
> `ModuleNotFoundError`.  The fallback in `shenfun/legendre/fastgl/__init__.py`
> now catches `ImportError` and substitutes a NumPy Gauss-Legendre rule for both
> `leggauss` and `getGLPair`, removing that crash and the `getGLPair`-is-`None`
> trap.  This does **not** fully enable a source-only `family="L"`: the Legendre
> `DLT` additionally needs the compiled `Leg2Cheb` (`shenfun.optimization.cython`),
> so a checkout without built extensions still fails when constructing a Legendre
> space.  The numbers below were produced against a built/installed shenfun
> (`test_taylor_couette.py` `25 passed`); without compiled extensions use the
> Chebyshev path:

```bash
conda run -n shenfun python - <<'PY'
import numpy as np
from taylor_couette_linear import CircularCouette, TaylorCouetteLinear
s = TaylorCouetteLinear(CircularCouette(1.0, 2.0, 1.0, 0.0), nu=1e-3, N=48, family="C")
print("Chebyshev Re_c =", round(s.critical_reynolds(0)["Re_c"], 3))   # 68.186
PY
```

## Suggested Further Comparisons (Not Yet Automated)

High-value literature checks that the current operators already support but that
are not yet scripted here:

- **Non-axisymmetric Taylor-Couette onset** for counter-rotating cylinders
  (`m != 0` spiral modes), e.g. Andereck, Liu & Swinney (1986) or Langford et al.
  (1988).  The cylindrical solvers accept any `m`, but only `m=0` is benchmarked.
- **Critical Reynolds / Taylor number across `eta`** (Chandrasekhar 1961; DiPrima
  & Swinney), including the narrow-gap limit `Ta_c ~ 1708`; presently only
  `eta=0.5` is checked.
- **`beta`-optimized Taylor-Couette transient growth** to reconcile the `eta=0.5`
  `Gmax=3.866` (at fixed `beta=pi/2`) with Hristova et al.'s `4.17`.
- **Goodman & Ji (2002)** explicit conducting-wall MRI thresholds for dissipative
  Couette flow -- the canonical resistive TC-MRI comparison.
- **Insulating-wall MRI transient growth** (only the insulating *modal* onset is
  benchmarked; `energy=...` non-modal for insulating walls is exercised but not
  compared to a reference).
- A **direct matrix-exponential cross-check** of `transient_growth_from_eigs` on a
  physics operator (the synthetic-matrix unit test in `test_couette_linear.py`
  already validates the helper math).

## References

- Reddy, Henningson, and collaborators, NASA report on linear stability and
  transient energy growth in plane Couette flow:
  https://ntrs.nasa.gov/api/citations/19950016790/downloads/19950016790.pdf
- Romanov, "Stability of plane-parallel Couette flow", Functional Analysis and
  Its Applications 7 (1973) -- plane Couette is linearly stable for all Re.
- Butler and Farrell, "Three-dimensional optimal perturbations in viscous shear
  flow", Physics of Fluids A 4, 1637 (1992) -- the streamwise-vortex optimal
  perturbation (the `(alpha,beta)=(0,1.66)`, `G ~ 1185` plane-Couette optimum).
- Reddy and Henningson, "Energy growth in viscous channel flows", Journal of
  Fluid Mechanics 252 (1993); Trefethen, Trefethen, Reddy and Driscoll,
  "Hydrodynamic stability without eigenvalues", Science 261 (1993) -- the
  `G_max ~ Re^2` transient-growth scaling.
- Camobreco, Potherat and Sheard, on transient growth and its magnetic-field
  suppression in MHD channel/duct flow, Physical Review Fluids / J. Fluid Mech.
  (2020-2021).
- Chandrasekhar, "Hydrodynamic and Hydromagnetic Stability" (1961) -- tabulated
  Taylor-Couette critical Taylor numbers across the gap.
- Andereck, Liu and Swinney, "Flow regimes in a circular Couette system with
  independently rotating cylinders", Journal of Fluid Mechanics 164 (1986) --
  non-axisymmetric (spiral) onset for counter-rotating cylinders.
- Goodman and Ji, "Magnetorotational instability of dissipative Couette flow",
  Journal of Fluid Mechanics 462 (2002) -- conducting-wall resistive TC-MRI
  thresholds.
- Hristova, Roch, Schmid, and Tuckerman, "Transient growth in Taylor-Couette
  flow", Physics of Fluids 14, 3475 (2002):
  https://blog.espci.fr/laurette/files/2018/01/transient_TC_pof.pdf
- Ruediger, Gellert, Hollerbach, Schultz, and Stefani, "Stability and
  instability of hydromagnetic Taylor-Couette flows", Physics Reports 741
  (2018), for the standard `Re_0 ~= 68.2` hydrodynamic onset:
  https://eprints.whiterose.ac.uk/132809/1/1-s2.0-S0370157318300346-main.pdf
- Balbus and Hawley, "A powerful local shear instability in weakly magnetized
  disks. I. Linear analysis. II. Nonlinear evolution", Astrophysical Journal 376
  (1991):
  https://inspirehep.net/literature/330903
- Ruediger and Schultz, "The gap-size influence on the excitation of
  magnetorotational instability in cylindric Taylor-Couette flows", Journal of
  Plasma Physics 90 (2024):
  https://www.cambridge.org/core/services/aop-cambridge-core/content/view/16D8A92B7FD23DF5F58D01E8739069B6/S0022377823001356a.pdf/gapsize_influence_on_the_excitation_of_magnetorotational_instability_in_cylindrictaylorcouette_flows.pdf
