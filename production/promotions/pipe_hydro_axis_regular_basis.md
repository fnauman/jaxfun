# Pipe Hydro Promotion Gap

Status: parity_pending

Pipe hydro remains intentionally rejected in jaxfun until the axis-regular radial
basis lands. GitHub issues are disabled for this repository, so this file is the
local promotion record required by the production-readiness plan.

Required implementation work:

- Add an axis-regular radial basis equivalent to shenfun one-sided-free
  `bc=(None, 0)` for a free/regular pipe axis and no-slip wall.
- Add the `m`-dependent `r^|m|` pole selection or equivalent
  singular weighted-Galerkin penalties with cylindrical measure `sqrt_det_g = r`.
- Implement `examples/pipe_flow_dns_jax.py` for hydro pipe DNS with
  `(u_r, u_theta, u_z, p)` in Fourier(theta) x Fourier(z) x radial spaces.
- Emit pipe hydro observables: `flow_rate`, `flow_rate_exact`,
  `kinetic_energy`, `divergence_l2`, and forcing phase for Womersley.
- Promote pipe hydro only after comparing against `pipe_hagen_poiseuille_v1`
  and `pipe_womersley_v1` at their declared tolerances.

Current required behavior:

- `production/problem_spec.py` rejects pipe hydro before solver allocation with a
  message naming the missing axis-regularity work and both pipe golden ids.
- Pipe MHD/MRI remains unsupported to match the shenfun production contract.
