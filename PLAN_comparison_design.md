# Plane Couette / Taylor-Couette / shearing-box comparison design

**Status:** design contract; claim-tier DNS is blocked by the gates below
**Local implementation:** `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun`
**Shearing-box implementation:** `/home/nauman/cfd/shearpy-jimenez`

## Scientific question

Measure how walls and curvature change rotating-MHD/MRI onset, saturation,
transport, and zero-net-flux survival. The three legs are:

1. rotating plane Couette flow in this repository;
2. Taylor-Couette flow in this repository; and
3. the shear-periodic spectral box in `shearpy-jimenez`.

The comparison is statistical and convention-matched. It is not a pointwise
trajectory comparison between different geometries.

## Canonical controls

Use a reference shear magnitude `S_ref`, rotation `Omega_ref`, length `h`, and
Alfven-speed magnetic units. Record, rather than infer, all of

```text
q_ref = S_ref / Omega_ref
Re_h  = |S_ref| h^2 / nu
Rm_h  = |S_ref| h^2 / eta
Pm    = Rm_h / Re_h
B_ref / (|S_ref| h)
Ly/h, Lz/h, magnetic wall/flux family, precision, dt |S_ref|
```

The first Keplerian comparison uses `S_ref=1`, `Omega_ref=2/3`, and
`q_ref=3/2`. Plane Couette uses half-gap `h=1` and `U0=-S_ref x e_y`.

For Taylor-Couette, define the comparison point `r0`, local
`Omega_ref=Omega(r0)`, and `S_ref=-r0 dOmega/dr|r0`. Report curvature
`epsilon=h/r0` as an additional control and use a curvature ladder; do not
present one finite-curvature point as identical to a Cartesian box.

Shearpy's config convention is `nu=1/Re` when `S=h=1`. Under a uniform box
rescaling `L'=lambda L` at fixed `S` and `Omega`, use

```text
u'=lambda u, B'=lambda B,
nu'=lambda^2 nu, eta'=lambda^2 eta,
Re'=Re/lambda^2, Rm'=Rm/lambda^2.
```

Thus the shearpy calibration `(4,4,1), Re=100, Rm=105, Bz=0.05` maps to
`(1,1,0.25), Re=1600, Rm=1680, Bz=0.0125`; the inverted Re/Rm mapping is
forbidden.

## Common outputs

Every leg must archive the same physical, volume-mean quantities and their raw
cadence series:

- fluctuating kinetic energy and total/mean/fluctuating magnetic energy;
- Reynolds, Maxwell, and total stress with the same sign convention;
- mean magnetic flux and drift;
- viscous/Ohmic dissipation, shear injection, and budget residual;
- fluctuation-energy slope with window and uncertainty;
- stress correlation time and effective independent-sample count;
- CFL decomposition, per-axis spectral tails, and mode occupancy;
- operational status, scientific class, resolution, timestep, precision, cost,
  commit/tag, and parent-checkpoint provenance.

The curl PCF family's `integral_abs2` energies must be multiplied by `0.5/V`;
the primitive PCF family's `half_integral_abs2` energies by `1/V`. Taylor-
Couette must be moved onto this same diagnostic contract before comparison.

## Campaign ladder

1. Match linear net-flux growth/onset at declared admissible modes.
2. Repeat one net-flux nonlinear saturated point at two resolutions and half
   timestep in all three legs.
3. Build independently sampled plateau parent banks.
4. Quench Rm using several qualified parent phases.
5. Compare ZNF survival probability and stress distributions at a common
   horizon; never compare a single surviving trajectory as a threshold.
6. Repeat boundary-condition and Taylor-Couette curvature ladders before a
   geometry-robust claim.

## Blocking gates

No claim-tier comparison may start until:

- this repository closes the applicable entries in `production/KNOWN_ISSUES.md`,
  especially vector-potential BC coverage, GPU cost, precision/timestep
  calibration, wall PDE residuals, and Taylor-Couette diagnostic alignment;
- shearpy's production requirements report `production_ready=true`, including
  measured net-flux saturated stress and a ZNF survival ensemble;
- both repositories have clean, pushed immutable release tags (or accepted
  commits merged to their protected main branches) with release test artifacts;
- every selected quench parent is plateau-qualified and every `sustained`
  classification meets the independent-sample floor; and
- underresolved/CFL-aborted runs are excluded from frontier inference.

Until then, bounded smoke, linear calibration, and discovery runs are allowed
only with their non-claim scope recorded.
