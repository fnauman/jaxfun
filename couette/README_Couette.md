# Couette Demo Guide

This directory contains plane-Couette and Taylor-Couette demos in hydrodynamic
and MHD regimes.  The newer linear paths are small dense eigenvalue tools for
analysis and validation; the DNS paths are the time-dependent Shenfun solvers.

## Shared Helpers

- `_linear_analysis.py`: shared dense generalized-eigenvalue utilities,
  finite-spectrum filtering, and energy-norm transient-growth calculation from
  modal propagators.  Exposes the single shared modal-basis magnitude cap
  `FINITE_CAP = 1e8` used by every `nonmodal_growth` (it must sit above the most
  strongly damped physical mode yet below the spurious constraint modes at
  infinity; verified insensitive between `1e6` and `1e12`).
- `_pcf_linear.py`: dense primitive-variable Chebyshev collocation operator for
  plane Couette hydro, plane Couette MHD, and the rotating-shear/MRI PCF
  analogue.  This is used by the PCF demo scripts when `--linear` is not `dns`.

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

### Hydrodynamic

- `pcf_fluctuations_corrected.py`
  - Nonlinear/DNS: default mode.  Evolves hydrodynamic plane-Couette
    fluctuations with diagnostics, spectra, and plotting.
  - Linear eigenvalues: `--linear eigs`.
  - Linear non-modal growth: `--linear nonmodal`.

Examples:

```bash
python demo/pcf_fluctuations_corrected.py --linear eigs --linear-nx 48 --ky 1 --kz 1
python demo/pcf_fluctuations_corrected.py --linear nonmodal --linear-nx 80 --ky 0 --kz 1.66 --linear-times 139
python demo/pcf_fluctuations_corrected.py --no-save-plots --no-save-analysis --no-save-spectra
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
python demo/pcf_mhd_divfree.py --linear eigs --linear-nx 48 --ky 1 --kz 1 --linear-bz 0.1
python demo/pcf_mhd_divfree.py --linear nonmodal --linear-nx 48 --ky 1 --kz 1 --linear-bz 0.1 --linear-times 0,1,5
python demo/pcf_mhd_divfree.py --linear nonmodal --linear-nx 48 --ky 1 --kz 1 --linear-energy kinetic
python demo/pcf_mhd_divfree.py --end-time 0.1 --moderror 10
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
python demo/pcf_mhd_mri_shearpy.py --linear eigs --linear-nx 48 --ky 0 --kz 25.81988897471611 --Re 1000000 --Rm 1000000
python demo/pcf_mhd_mri_shearpy.py --linear nonmodal --linear-nx 32 --ky 0 --kz 6.283185307179586 --linear-times 0,1,5
python demo/pcf_mhd_mri_shearpy.py --linear nonmodal --linear-nx 32 --ky 0 --kz 6.283185307179586 --linear-energy magnetic
python demo/pcf_mhd_mri_shearpy.py --end-time 0.01 --store-history
```

## Taylor-Couette Flow

### Hydrodynamic

- `taylor_couette_linear.py`
  - Linear eigenvalues: default mode with either a fixed `--kz` or a scan over
    `--kz-min`, `--kz-max`, and `--kz-num`.
  - Linear non-modal growth: `--nonmodal`.
  - Includes critical-Reynolds utilities on `TaylorCouetteLinear`.

Examples:

```bash
python demo/taylor_couette_linear.py --N 48 --family C --kz 3.16
python demo/taylor_couette_linear.py --N 80 --family C --nonmodal --kz 1.5707963267948966 --times 5,10,20
```

- `taylor_couette_dns.py`
  - Nonlinear/DNS: default hydrodynamic mode.
  - Linear hydrodynamic eigenvalues: `--linear-analysis eigs`.
  - Linear hydrodynamic non-modal growth: `--linear-analysis nonmodal`.
  - This script can be used as the common DNS entry point when comparing linear
    predictions to nonlinear runs.

Examples:

```bash
python demo/taylor_couette_dns.py --linear-analysis eigs --Nr 48 --family C --kz 3.16
python demo/taylor_couette_dns.py --linear-analysis nonmodal --Nr 80 --family C --kz 1.5707963267948966 --times 5,10,20
python demo/taylor_couette_dns.py --Nr 32 --Nz 64 --end-time 1.0
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
python demo/taylor_couette_mri.py --local-check
python demo/taylor_couette_mri.py --N 32 --family C --magnetic-bc conducting --kz 1.75 --B0 0.16639676113360324 --eta-mag 0.04048582995951417 --nu 4.048582995951417e-08
python demo/taylor_couette_mri.py --N 32 --family C --magnetic-bc insulating --nonmodal --kz 1.25 --times 0,1,5
python demo/taylor_couette_mri.py --N 32 --family C --nonmodal --kz 1.5 --times 5 --energy kinetic
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
python demo/taylor_couette_dns.py --mhd --linear-analysis eigs --Nr 32 --family C --kz 1.75 --B0 0.16639676113360324 --eta-mag 0.04048582995951417 --nu 4.048582995951417e-08
python demo/taylor_couette_dns.py --mhd --linear-analysis nonmodal --Nr 32 --family C --kz 1.25 --times 0,1,5
python demo/taylor_couette_dns.py --mhd --linear-analysis nonmodal --Nr 32 --family C --kz 1.25 --magnetic-bc insulating --energy kinetic
python demo/taylor_couette_dns.py --mhd --Nr 32 --Nz 64 --end-time 0.1 --B0 0.1
```

## Benchmark Notes

See `couette_linear_benchmarks.md` for the literature references, expected
growth rates, and reproducible benchmark commands used to validate the linear
and non-modal additions.

> **Environment note.**  The Taylor-Couette modal tests (`test_taylor_couette.py`)
> default to the Legendre family (`family="L"`), which requires shenfun's optional
> fast-Gauss-Legendre extension.  If that compiled wrapper is missing the import
> can raise `ImportError: cannot import name 'fastgl_wrap' ... circular import`;
> use `--family C` (Chebyshev) for the demos and `test_couette_linear.py` for the
> dense linear/non-modal layer (pure NumPy PCF + Chebyshev TC), which is
> Legendre-independent.
