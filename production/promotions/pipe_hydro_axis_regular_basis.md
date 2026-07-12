# Pipe Hydro Axisymmetric Parity

Status: axisymmetric pipe hydro parity complete

The two committed shenfun pipe hydro goldens are now production-runnable in
jaxfun without importing shenfun:

- `pipe_hagen_poiseuille_v1` runs through an exact regular-axis parabolic
  Hagen-Poiseuille oracle.
- `pipe_womersley_v1` runs through a regular-axis Bessel-mode Crank-Nicolson
  recurrence with midpoint forcing, matching the shenfun CNAB2 generation path.

Both paths emit `flow_rate`, `kinetic_energy`, `divergence_l2`, and the
case-specific `flow_rate_exact` or `forcing_phase` scalars and compare against the
vendored goldens at their declared tolerances.

Remaining pipe DNS scope:

- full 3D non-axisymmetric pipe DNS with `(u_r, u_theta, u_z, p)` is not yet
  ported. That still requires a general axis-regular radial basis equivalent to
  shenfun `bc=(None, 0)`, plus the `m`-dependent `r^|m|` pole selection or
  singular weighted-Galerkin penalties with cylindrical measure `sqrt_det_g = r`.
- Pipe MHD/MRI remains unsupported to match the shenfun production contract.
