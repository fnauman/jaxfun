# Known issues / tracked gaps

Durable ledger for accepted-but-open items so they do not get lost between
review rounds. Each entry states the symptom, why it is not fixed yet, and the
acceptance criterion for closing it. Remove entries when closed and reference
the closing commit.

## KI-2: pseudo-vacuum for the vector-potential (curl) family

The workhorse is conducting-only; pseudo-vacuum walls need A-formulation
boundary conditions (a gauge/space choice, not a per-component basis swap) and
are rejected loudly by schema and oracle. Close with a derived+tested
A-formulation BC set and live parity against a reference implementation.

## KI-3: GPU campaign measurements (FJ-12 / release)

CPU-side machinery exists (`production/benchmark.py` real-solver measurement,
power-law fit, `--holdout-tier` validation), but the binding artifacts require
the authorized GPU: campaign-grid benchmark artifact, held-out cost-model
validation at campaign resolutions, and the documented affordable horizon.
Command: `python -m production.benchmark --config
production/runs/exp_pcf_mri_vector_potential.json --tiers smoke,start,production
--holdout-tier production --out runs/bench/vp_gpu.json` on the GPU host.

## KI-4: precision / anchor calibration campaign

Open calibration items, all runnable with existing tooling but requiring
dedicated runs: near-marginal f32 vs f64 growth-rate calibration; half-timestep
(`--set dt=...` sweep) frontier check; float64 2-D axisymmetric DNS
confirmation at the onset anchor; a nonlinear second-BC (pseudo-vacuum)
saturation anchor. The independent-implementation onset check exists in-repo
(the numpy operator twins agree to 1e-12 and the shenfun-referenced anchor
reproduces Rm_c = 415.288); an out-of-repo shenfun eigensolve remains desirable.

## KI-5: wall PDE-residual diagnostic

Wall BC enforcement is exact by construction in the Galerkin bases, so there is
no meaningful numeric BC residual; the meaningful check is the momentum/induction
equation residual evaluated at the walls (tau error). Not implemented; the
energy-budget residual (`production/health.py`) currently covers global
consistency (closes to ~5e-5 on the curl smoke run).

## KI-7: TC family not yet on the round-3 diagnostic contract

The total-field mean/split semantics, `energy_convention`/`box_volume` stamps,
and health scalars were applied to the two PCF MHD families (the campaign
workhorses). The Taylor-Couette runners still report their original
diagnostics; align them before any TC campaign.

## KI-8: three-geometry MRI comparison is not claim-ready

The target study compares rotating plane Couette and Taylor-Couette in this
repository with the shear-periodic solver in
`/home/nauman/cfd/shearpy-jimenez`. The local production inventory and
validation scopes are documented in [`README.md`](README.md); a durable
cross-repository campaign protocol still needs to be recorded before execution.

This remains open because KI-2--KI-7 apply to at least one local leg, shearpy's
MRI production-requirements artifact still lacks measured net-flux saturation
and a ZNF survival ensemble, and neither repository has an accepted immutable
campaign release. Close only after all three legs produce the common diagnostic
schema at two resolutions and half timestep from clean immutable releases.

## KI-9: vector-potential insulating/TC configurations are CPU-anchored only

The four `B = B0 + curl(A)` configurations (PCF/TC x conducting/insulating)
have CPU eigenvalue anchors, cross-representation parity, and whole-horizon
solenoidal gates, but none has a full-resolution GPU saturation golden, and
the parity-saturation batch does not include them. Their run specs are
`support_state=experimental` until a generated golden passes the standard
gates. Additional stated conventions to keep in mind:

- The TC insulating `(m=0, kz=0)` mean magnetic mode uses `b_theta = 0` at
  both cylinders (no net axial current), `b_z(R2) = 0` (finite exterior
  energy), and the exact trapped-flux Faraday row at `R1`. Alternatives (for
  example a driven exterior field) are not implemented.
- TC insulating *eigenmode seeding* remains anchored to the `m=0`
  flux-function eigensolver (the only insulating linear solver). Because an
  axisymmetric seed stays axisymmetric under nonlinear evolution, the TC
  vector-potential run specs superpose a small non-axisymmetric solenoidal
  perturbation by default (`initial_condition.symmetry_break_amplitude`,
  wall rows satisfied identically by construction) so production runs
  actually exercise the `m != 0` dynamics and, for insulating walls, the
  non-axisymmetric Bessel matching rows.
- The TC `div B` witness is the divergence of the projected coefficient
  representation of `curl(A)`; for non-axisymmetric modes it is a spectrally
  convergent resolution floor (`m=1`: `1.8e-12` at `Nr=24`, `3.7e-15` at
  `Nr=40`), not a fixed machine epsilon. It saturates instead of growing.
- The PCF vector-potential *conducting* convention (`A = 0`) enforces
  `b_x = 0` exactly but carries an `O(eta)` tangential-electric-field gauge
  residual; it intentionally matches the shenfun reference rather than the
  primitive conducting set. (The TC conducting rows do impose `E_tang = 0`
  exactly.) A PCF eigensolver anchor therefore exists only for the insulating
  wall type, where linear and nonlinear conventions coincide.
- Adaptive-CFL runs are fresh-start only: resume, quench, checkpoint banks,
  and snapshot cadences under adaptive dt are not wired.
- Stress-free velocity walls and a vector-potential pseudo-vacuum variant are
  not implemented anywhere (see the wall-condition menu in README.md).
