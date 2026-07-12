# Production-sweep remediation plan for `fork_jaxfun`

**Status:** implementation plan; production sweeps are not yet authorized
**Audit basis:** `fork_jaxfun` at commit `2506864` on the unmerged branch `review-validation-hardening`, reviewed 2026-07-10
**Target repository:** `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun`
**Companion documents:** `PLAN_fixes_shearpy_jimenez.md`, `PLAN_comparison_design.md`

## Executive decision

The equation and operator core is strong enough to preserve. Focused tests and live
Shenfun/JAX parity checks found no sign or operator error, and the primitive MRI path is
second order in time. The repository is nevertheless **not sweep-ready**. The blockers
are experiment plumbing and nonlinear-state integrity:

1. sweep parameters are duplicated and can disagree with the coefficients actually used;
2. production dealiasing factors are supplied in the wrong axis order;
3. the nominally 3-D MRI seed remains in the exactly axisymmetric subspace;
4. the promoted nonlinear golden contains guard-violating velocity and magnetic divergence;
5. the ZNF path lacks flux-safe diagnostics and a validated solenoidal production route;
6. restarts reject the parameter changes required by a quench protocol;
7. W&B and a parameter-override sweep driver do not exist;
8. nonlinear PCF DNS supports only one magnetic wall condition;
9. the current primitive 3-D cost can make a formally valid campaign uneconomical; and
10. the audited capabilities live on an unmerged branch rather than a production release.

Net-flux **linear** pre-scans may proceed while these items are fixed. Production 3-D,
saturated, and zero-net-flux sweeps may begin only after the launch gates at the end of
this document pass.

## What is already trusted

- The primitive PCF-MRI DNS advances diffusion, rotation, shear, and imposed-field
  coupling with Crank--Nicolson and nonlinear terms with AB2. It is CNAB2, not a
  first-order-diffusion/third-order-advection mixture.
- A coupled linear MRI step-doubling check gives the expected factor of approximately
  four for a second-order method.
- The KMM/vector-potential family offers IMEX-RK222 by default and IMEX-RK443 as an
  available third-order method. Its magnetic field is represented as a curl and is
  solenoidal by construction.
- The focused audit passed 163 tests; the complete live Shenfun parity module passed
  24/24 again during this amendment review. Both the primitive and vector-potential
  PCF families have coefficient-level finite-amplitude evidence. The primitive 3-D
  case is nevertheless short and uses `dealias=1.0`, so it does not validate the
  corrected production dealiasing contract by itself.
- The 2-D DNS growth-rate benchmark agrees with an independent linear eigensolver to
  about `6.5e-8` relative error in float64.
- Local JSONL diagnostics, metadata, HDF5 checkpoints, and cadence callbacks provide a
  good base for tracking. W&B should remain a mirror of these local records, not replace
  them.

## Reconciled findings that supersede review shorthand

### The production primitive integrator is CNAB2

Some production JSON describes `IMEXRK222`, but the primitive solver used by the wired
MRI DNS specifications actually runs hard-coded CNAB2. Correct the specification and
metadata rather than reporting the requested-but-unused method. An integrator field on
a dense eigensolve specification is inert and should be rejected or omitted.

### Magnetic divergence is not the only suspect quantity

The existing shear-box-like golden ends near
`div_b_l2 = 2.67e-2` and `div_u_l2 = 2.30e-2`, above the current `1e-2` guard. This may
combine a real constraint failure with a normalization/derivative inconsistency, but it
cannot be accepted as a physical nonlinear reference. Diagnose both fields and their
wall-condition residuals before choosing a repair.

### Re/Rm scaling must not use the inverted mapping from the supplied fork review

For shearpy's `nu = 1/Re` convention, and for canonical PCF units `S=h=1`, a
uniform box rescaling `L' = lambda L` at fixed `S` and `Omega` requires

```text
u' = lambda u,  B' = lambda B,
nu' = lambda^2 nu,  eta' = lambda^2 eta,
Re' = Re/lambda^2,   Rm' = Rm/lambda^2.
```

Thus `(4,4,1), Re=100, Rm=105, Bz=0.05` maps to
`(1,1,0.25), Re=1600, Rm=1680, Bz=0.0125`, not to smaller Re/Rm. The
cross-geometry convention and full worked example are fixed in
`PLAN_comparison_design.md`.

## Required work packages

Priorities mean:

- **P0:** can silently change or corrupt the scientific result;
- **P1:** required for an efficient, auditable production campaign;
- **P2:** expands scientific scope or removes an important limitation;
- **P3:** useful follow-on, not a first-campaign gate.

### FJ-00 — Freeze one campaign parameter contract (`P0`)

**Problem.** Solver coefficients (`nu`, `eta_mag`, imposed field) and reported
`Re`, `Rm`, `Pm`, and `B0` have independent sources. The current schema checks
`Pm = Rm/Re` but does not guarantee that changing Re/Rm changes the coefficients used by
the solver. A sweep could therefore relabel identical physics.

**Implementation.**

1. Introduce a single resolved-physics object consumed by every solver and diagnostic.
2. Make the dimensional/code coefficients canonical:
   `h`, `S`, `Omega`, `nu`, `eta`, `B0`, `Ly`, `Lz`, wall conditions, and precision.
3. Permit user-facing `Re_h` and `Rm_h` overrides, but resolve them exactly once via
   `Re_h = |S| h^2/nu` and `Rm_h = |S| h^2/eta`. Derive `Pm = nu/eta = Rm_h/Re_h`.
4. Reject over-specified inconsistent inputs; do not silently choose one source.
5. Record both the raw inputs and the fully resolved object in metadata and in the
   checkpoint hash.
6. Use one imposed-field value everywhere: linear operators, nonlinear RHS, initial
   condition, energy decomposition, CFL estimate, and W&B configuration.
7. Name the Reynolds convention in every output. If a solver uses
   `U_wall h/nu`, also store `U_wall` and verify `U_wall = |S|h` for the comparison
   campaign.

**Acceptance tests.**

- Changing only `Re_h` changes `nu` in the assembled operator and the one-step decay
  factor by the analytically expected amount.
- Inconsistent `{Re_h, nu}` or `{Rm_h, eta}` fails before JAX compilation.
- Primitive, vector-potential, linear, and diagnostic paths receive the same resolved
  values.
- Metadata round-trips the resolved object without loss, and the run ID changes when a
  physics coefficient changes.

### FJ-01 — Make axis order explicit and repair dealiasing (`P0`)

**Problem.** The primitive arrays are ordered `(y,z,x)`. Their correct 3-D padding is
`(1.5,1.5,1.0)`, but production specifications supply `(1.0,1.5,1.5)` as though arrays
were `(x,y,z)`. The streamwise nonlinearity is therefore undealiased while the
wall-normal direction is padded. The 2-D production path also forces padding `1.0`,
which is acceptable for linear onset but not finite-amplitude work.

**Implementation.**

1. Replace positional user-facing padding tuples with semantic fields such as
   `{x: 1.0, y: 1.5, z: 1.5}`.
2. Convert to native array order in one internal adapter and log both forms.
3. Reject anonymous three-tuples in new production specifications. Retain a temporary
   compatibility parser only if it emits a loud deprecation warning.
4. Enable periodic-direction dealiasing in nonlinear 2-D runs; allow undealiased mode
   only for an explicitly linear run.
5. Audit all padding, shape, FFT-axis, diagnostic-gradient, and checkpoint code against
   the same named-axis contract.
6. Increment a first-class `numerics_contract_version`. Mark every checkpoint, golden,
   and parent-bank entry created before this fix as pre-FJ-01 and forbid it from seeding
   post-fix production. It may be retained only as quarantined regression evidence.

**Acceptance tests.**

- A manufactured triad that aliases without padding is removed in every periodic
  direction.
- Permuting the semantic input order cannot change the native padding.
- Production 2-D and 3-D specifications report their physical and native axis order.
- A short nonlinear parity run is repeated with the corrected production tuple.
- A pre-fix checkpoint/golden/bank is rejected by a post-fix production continuation.

### FJ-02 — Make 3-D runs genuinely non-axisymmetric (`P0`)

**Problem.** The current MRI production seed excites only `k_y=0`. Exact arithmetic
keeps such a run in the axisymmetric invariant subspace, so changing `Ly` cannot test
parasites, 3-D saturation, or a streamwise aspect-ratio effect.

**Implementation.**

1. Split initial conditions into explicitly named modes:
   `axisymmetric_eigenmode`, `net_flux_3d_perturbed`, `znf_profile_3d_perturbed`, and
   `znf_random_3d`.
2. For 3-D nonlinear work, seed at least one divergence-free `k_y != 0` perturbation
   with a recorded amplitude and seed.
3. Preserve the eigenmode-only IC for linear growth validation, where it is desirable.
4. Add `E_nonaxisymmetric/E_total` and selected parasite-mode amplitudes to diagnostics.
5. Fail a production 3-D specification if its requested purpose is nonlinear but its
   initial non-axisymmetric energy is exactly zero.

**Acceptance tests.**

- Axisymmetric validation remains invariant to roundoff.
- The 3-D IC has the requested nonzero `k_y` energy, is solenoidal, and satisfies wall
  conditions.
- `Ly` changes the admissible mode lattice and is visible in a non-axisymmetric-mode
  diagnostic.

### FJ-03 — Resolve the solenoidal-production path and retire the invalid golden (`P0`)

**Problem.** The specifications named as divergence-free dispatch to the primitive
solver, not the curl/vector-potential solver. The existing nonlinear golden violates
current divergence guards. Long saturated and ZNF runs cannot proceed on this basis.

**Implementation sequence.**

1. Recompute divergence using independent physical- and coefficient-space kernels;
   report absolute, relative, per-component, and maximum norms.
2. Report no-slip and magnetic wall-condition residuals separately from divergence.
3. Determine whether the observed `div_u`/`div_b` values are field errors, derivative
   axis errors, normalization errors, or all three. FJ-01 must land first.
4. Wire `PlaneCouetteMRIShearpyJax` (the curl/vector-potential family) into a production
   oracle as a candidate primary ZNF path.
5. Cross-check it against the primitive path at linear growth and finite amplitude.
   Confirm that its gauge and basis imply the intended *physical* magnetic wall
   conditions; curl representation alone is not a boundary-condition proof.
6. If the primitive path remains supported, implement a wall-compatible Helmholtz
   projection or equivalent constrained solve. A periodic-only projection is
   insufficient for a wall-normal Chebyshev direction.
7. Remove or quarantine the old golden. Generate a new float64 nonlinear reference
   only after the current guards pass. Change validation scripts so comparison is the
   default and golden replacement requires an explicit acceptance action.
8. Make the live parity harness convention-explicit and rerun the full module on the
   eventual release commit. Record coordinate order, signed shear, half/full-gap
   convention, reference interpreter, and Shenfun commit in the result artifact.

**Acceptance gates.**

- `div_u` and `div_b` remain at the method's verified tolerance for a nonlinear run
  long enough to exercise multiple checkpoint/diagnostic cadences.
- Constraint and wall residuals do not grow secularly.
- The kinetic-plus-magnetic energy budget closes to the measured temporal/spatial
  discretization error.
- Primitive and vector-potential growth rates agree with the independent eigensolver;
  finite-amplitude differences are quantified before selecting a campaign solver.
- Solver names, specification names, and actual dispatch agree.
- The release candidate reproduces the currently green 24-test live Shenfun parity tier;
  skipped or convention-ambiguous reference tests do not count as a pass.

### FJ-04 — Add ZNF-safe flux and transport semantics (`P0`)

**Problem.** There is no first-class mean magnetic-flux monitor or repair policy.
`transport_alpha` divides by `B0^2`, so a correct `B0=0` run produces NaN and can fail
the finiteness gate.

**Implementation.**

1. Log all three volume-mean magnetic components and their drift from the initial value.
2. Define separate energy fields:
   `energy.mag_total`, `energy.mag_mean`, and `energy.mag_fluct`.
3. Define dimensional/code stress components independently of any normalization:
   Reynolds stress, Maxwell stress, and their sum.
4. Emit a net-flux-normalized `alpha_B0` only when `B0 != 0`. For ZNF use a clearly
   named alternative, such as stress normalized by `(S h)^2`, and never substitute an
   arbitrary denominator.
5. If a mean-flux projector is added, log the pre-projection drift and correction. Do
   not let a projector hide a conservation defect.
6. Provide distinct, wall-compatible ZNF profile ICs for conducting and pseudo-vacuum
   walls, plus a random-field ZNF IC. Record rms field, radial wavenumber, parity, and
   exact mean.

**Acceptance tests.**

- A ZNF run has finite diagnostics and preserves mean flux to the verified tolerance.
- A net-flux run cleanly separates imposed/mean and fluctuating magnetic energy.
- The same stress kernel feeds JSONL, summaries, W&B, and comparison reports.

### FJ-05 — Build restart banks and an explicit quench continuation mode (`P0`)

**Problem.** `--resume` requires an identical specification hash and primarily exposes
the latest state. It cannot lower Rm, select several stationary plateau states, or
perform a controlled net-flux-to-ZNF transformation.

**Implementation.**

1. Preserve strict same-spec resume as `resume-exact`.
2. Add a separate, explicit `continue-from`/`quench` operation with an allowlist of
   mutable fields. Initially allow `nu`/`eta` (or resolved Re/Rm) only.
3. Keep geometry, basis, resolution, state layout, wall conditions, coordinate order,
   representation, and `numerics_contract_version` immutable unless a tested
   interpolation/transform operation is selected. Pre-FJ-01 states are never eligible
   for post-FJ-01 production continuation.
4. Treat changing `B0` as a field transformation, not a simple config override. Define
   whether the checkpoint stores total or fluctuating field, transform it explicitly,
   and revalidate flux and energy.
5. Write immutable, time-indexed checkpoints and a checkpoint-bank manifest containing
   parent run ID, state time, plateau-window statistics, hash, solver representation,
   and compatibility fields.
6. Reset or quarantine inherited growth/classification history during a configurable
   post-quench burn-in.
7. Make horizons unambiguous: record both absolute final time and requested additional
   shear times.

**Acceptance tests.**

- Same-spec resume remains bit-exact.
- A changed-Rm quench starts from identical fields, changes only resistive coefficients,
  and records parent/child provenance.
- An incompatible geometry, basis, BC, or state representation is rejected.
- Two selected plateau checkpoints create distinct child trajectories; merely changing
  a seed after loading the same deterministic state is not mislabeled as an ensemble.

### FJ-06 — Complete the diagnostic and health contract (`P1`)

**Problem.** Cadenced production output keeps energies and divergence but discards the
already-computed stresses. There is no CFL, spectral-tail, budget, flux, or
non-axisymmetric health signal. Guards run only at cadence boundaries.

**Required cadence metrics.**

- kinetic, total magnetic, mean magnetic, and fluctuating magnetic energy;
- Reynolds, Maxwell, and total stress with a documented sign convention;
- viscous and Ohmic dissipation, mean-shear/wall injection, and budget residual;
- absolute and relative `div_u`, `div_b`, and wall-condition residuals;
- all mean magnetic components and flux drift;
- CFL contributions from advection, shear, rotation, imposed/total Alfvén speed, and
  diffusion where relevant;
- per-direction spectral-tail ratios and maximum retained-mode occupancy;
- non-axisymmetric energy fraction and selected channel/parasite modes;
- wallclock per step, compilation time, device, precision, and effective cadence.

**Health policy.**

1. Separate an **operational status** (`completed`, `early_stop`, `nan_inf`,
   `blew_up`, `walltime`, `failed`, `underresolved`) from a **scientific class**
   (`growing`, `sustained`, `marginal`, `decayed`, `inconclusive`).
2. Add a finite runaway-energy ceiling as well as non-finite checks.
3. Evaluate cheap health checks every compiled block. Choose commensurate diagnostic
   and checkpoint cadences to avoid avoidable scan recompilation.
4. Fit late-window log-slopes only to positive fluctuation energy above a declared noise
   floor. Store fit window, uncertainty, sample count, and goodness of fit.
5. Require persistent stress and a stationarity/correlation-time criterion for
   `sustained`; “alive at final time” is not sufficient.
6. Quarantine underresolved results from threshold inference automatically.

### FJ-07 — Add an optional W&B sink and a sweep-safe CLI (`P1`)

**Problem.** There is currently no W&B code or installed dependency. The runner accepts
a JSON specification but has no safe physics override interface.

**Implementation.**

1. Keep `diagnostics.jsonl`, `metadata.json`, summaries, and checkpoints as the source of
   truth. Make W&B an optional extra and permit fully offline runs.
2. Add W&B lifecycle handling around the existing host-side cadence callback; no W&B
   call should occur inside JAX tracing.
3. Log the complete canonical cadence dictionary, not a reduced hand-maintained subset.
4. Populate run summary fields with operational status, scientific class, trailing
   growth rates and uncertainty, late-window stresses, stationarity, maximum constraint
   errors, spectral-tail verdict, final time, and cost.
5. Add a base-spec plus validated override CLI, for example semantic overrides for
   `Re_h`, `Rm_h`, `B0`, `Ly`, `Lz`, resolution, BC, seed, horizon, and precision.
   Materialize and archive the resolved per-run specification before launch.
6. Generate collision-resistant run IDs and group by geometry/aspect/BC/flux family.
7. Support `WANDB_MODE=offline` followed by `wandb sync`. Upload selected plateau
   checkpoints as artifacts only by policy; otherwise log their hash and durable URI to
   avoid uncontrolled storage use.
8. Provide both a simple Cartesian executor and an adaptive bisection/continuation
   driver. W&B native sweeps may dispatch independent cells, but they should not replace
   threshold-aware orchestration.
9. Never run off-golden parameter points with `--compare-golden`; use experimental
   support status so a physically decaying point is not treated as a software failure.

**Remote use.** Runs on a GPU server and analysis on this machine can share an
entity/project. Authentication must be smoke-tested on each host rather than inferred
from a credential file. API polling can summarize runs when requested; continuous
monitoring across sessions requires a scheduled watcher/service.

**Acceptance tests.**

- Online-disabled and W&B-uninstalled tests pass.
- Offline smoke, online smoke, resume, exception, and early-stop paths each finish one
  and only one W&B run and preserve local output.
- Every sweep override changes the resolved spec or fails validation.
- The API can filter a project to a compact frontier table without downloading all
  checkpoints.

### FJ-08 — Validate precision and timestep policies (`P1`)

**Problem.** The production CLI defaults to float32, while the strongest marginal-growth
validation is float64. The timestep is fixed, baked into cached factorizations, and has
no adaptive CFL control.

**Implementation.**

1. Make float64 mandatory for linear thresholds, final brackets, and all claim-tier
   results.
2. Permit float32 only as a discovery-tier scout after one near-marginal f32/f64
   distribution-level calibration.
3. Add formal step-doubling tests for the complete coupled linear block and at least one
   nonlinear manufactured/parity problem.
4. Record the actual integrator, its formal order, `dt`, CFL diagnostics, and factorized
   operator hash.
5. At each frontier, repeat at half `dt`; for fixed-step runs size `dt` from a short
   preflight and abort on a declared CFL ceiling.

**Acceptance gates.**

- CNAB2 remains second order with viscosity, resistivity, rotation, shear, and imposed
  field all active.
- The KMM method reports the method actually selected and passes its corresponding order
  test.
- A near-marginal f32 scout and f64 reference yield the same classification/bracket
  within the declared scouting tolerance; trajectories are not expected to coincide.

### FJ-09 — Add a second magnetic wall-condition family (`P2`)

**Problem.** Nonlinear PCF DNS is restricted to no-slip velocity plus perfect-conductor
magnetic walls. Linear tools expose more choices, but some labels conflate
perfect-conductor and pseudo-vacuum conditions.

**Implementation order.**

1. Replace generic/ambiguous labels with physical enums and equations:
   - perfect conductor: `b_x=0`, `d_x b_y=d_x b_z=0`;
   - pseudo-vacuum/vertical field: `b_y=b_z=0` with the compatible normal condition;
   - true vacuum/insulating: exterior matching, not merely Dirichlet data.
2. Implement pseudo-vacuum as the second nonlinear DNS family in both supported
   formulations.
3. Add basis, solenoidality, energy-budget, and live-parity tests for each BC.
4. Map net-flux linear onset for all already-supported BCs before spending nonlinear
   compute.
5. Defer true-vacuum mode-dependent matching to a third family unless the linear
   sensitivity map shows it is essential.

No-slip velocity walls define plane Couette flow and need not be varied in the initial
campaign. Magnetic-wall sensitivity is the immediate boundary-condition concern.

### FJ-10 — Commit a PCF-MRI onset anchor and threshold driver (`P1`)

**Problem.** The repository has strong individual linear components but no automated
PCF critical-Rm campaign or committed PCF-MRI onset anchor.

**Implementation.**

1. Add a discrete-mode lattice driver over `k_y=2*pi*n/Ly` and
   `k_z=2*pi*m/Lz`.
2. For fixed `(Re_h, B0, BC, aspect)`, bracket and bisect Rm on the maximum real
   eigenvalue. Store the winning mode and the next-nearest competitors.
3. Cross-check selected points among the atlas `PlaneCouetteOperator`, the independent
   PCF MRI eigensolver in this knowledge base, and `code/shenfun/_pcf_linear.py`. The
   first two wall implementations share a Chebyshev routine, so the live Shenfun result
   is the genuinely independent wall cross-check.
4. Encode the convention traps in the driver and its tests: shearing-box parameters use
   `S=+1` for `U0=-Sx`, PCF operators use signed `S0=-1`; atlas `a` is a half-gap while
   the wall eigensolver's `Lx` is a full width; and `from_atlas()` hardwires `a=0.5`, so
   the `h=1` operator must be constructed directly.
5. Confirm selected marginal points with float64 2-D DNS seeded by the eigenmode.
6. Commit the verified conducting anchor and at least one second-BC anchor with explicit
   tolerances and the winning/competing modes.

**Verified conducting calibration anchor.** Under `h=1`, `S=1`, `Omega=2/3`, full box
`(2,2,0.5)`, `Re_h=400`, and `B0=0.025`, independent recomputation gives
`Rm_h,c=415.288` for `k_y=0`, vertical mode `n=1`, and
`gamma(Rm_h=420)=3.17887e-3`. The root is unchanged at wall resolutions 48, 64, and 96;
`n=2` onsets only near `Rm_h=3172`, while tested `k_y=pi,2*pi` modes remain stable to
`Rm_h=8000`. FJ-10 must turn these verified numbers into an executable regression; it
does not need to rediscover them before they can guide the Phase-2 confirmation pair.

### FJ-11 — Clean names, docs, and dead configuration (`P1`)

- Rename specifications so `linear` versus `dns`, `primitive` versus
  `vector_potential`, and `conducting` versus `pseudo_vacuum` are truthful.
- Remove or reject inactive integrator fields on eigensolve specifications.
- Correct comments that advertise an unassembled magnetic saddle constraint.
- Replace the current “butterfly” global mean with an actual wall-normal--time profile,
  or rename it so it does not imply a butterfly diagram.
- Assert the JAX x64 state after production imports in custom drivers.
- Correct the ideal MRI cutoff formula in the PCF MHD/MRI note: for the standard
  vertical-field axisymmetric problem the ideal cutoff is tied to
  `k^2 v_A^2 < 2 q Omega^2`, not `4 Omega^2(q-1)`.

### FJ-12 — Establish a 3-D performance contract and choose the workhorse (`P1`)

**Problem.** Correctness gates alone do not make the nonlinear campaign affordable.
The audited primitive path measured roughly `2.2 s/step` for a dealiased `32^3`
float64 case, corresponding to order `36 GPU-h` for 300 shear times under the measured
timestep. The vector-potential family has shown roughly order-of-magnitude better step
throughput, but it must be benchmarked with matched physics, diagnostics, BCs, and the
corrected dealiasing before being declared the workhorse.

**Implementation.**

1. Benchmark both formulations at the actual short, wide, and tall campaign grids in
   float32 and float64 after FJ-01. Match physical coefficients, timestep, output
   cadence, dealiasing, diagnostics, and device.
2. Separate compilation/factorization time, warm-cache steady-state time per step,
   checkpoint/diagnostic I/O, and peak memory. Report cost per simulated shear time, not
   only cost per step.
3. Profile the primitive per-mode LU/factorization and transform path; verify that cache
   keys are stable across blocks and that cadence choices do not trigger recompilation.
4. Measure scaling with `Nx`, `Ny`, `Nz`, precision, and periodic padding; fit a simple
   cost model and validate it on at least one held-out grid.
5. Select one 3-D production workhorse only after FJ-03 shows matched growth,
   constraints, wall semantics, and finite-amplitude behavior. The expected choice is
   the vector-potential family; retain the primitive path as an independent spot-check,
   not as the default merely because existing specs dispatch to it.
6. Attach predicted and actual GPU-hours to every planned campaign rung. Refuse a batch
   that exceeds its declared budget without explicit approval.

**Acceptance gates.**

- The cost model predicts a held-out run within a declared tolerance (initially 20%).
- A second invocation demonstrates cache reuse and reports no unexplained compilation.
- The selected workhorse passes the same physics/constraint/BC gates as its reference
  path and has a documented maximum affordable grid and horizon.
- Phase 3--5 PCF budgets are based on measured cost per shear time, including output,
  rather than an optimistic kernel-only timing.

### FJ-13 — Create a reproducible production release (`P0`)

**Problem.** All audited hardening capabilities live at `2506864` on
`review-validation-hardening`, 96 commits ahead of `main` at the audit. A branch name is
movable and a local unmerged commit may later become unreachable. Production provenance
cannot depend on remembering which checkout happened to be active.

**Implementation.**

1. Choose one policy before the first DNS confirmation: merge the accepted fixes to
   `main`, or create and push an immutable campaign release tag containing them.
2. Run only from a clean worktree. Record commit, branch (informational), release tag,
   remote URL, dirty-state flag, dependency lock hash, JAX/CUDA versions, and container
   or environment hash in every manifest.
3. If an emergency dirty-tree run is ever authorized, archive the exact diff and its
   SHA256 with the run; it remains discovery-only.
4. Run the full unit/integration selection, the 24-test live Shenfun parity tier, and the
   corrected-dealias regression on the exact release commit. Publish the test artifact
   beside the release.
5. Preserve the release ref on the remote used by the GPU server and verify that a clean
   clone can reproduce the Phase-2 anchor command.

**Acceptance gate.** No production run may start from an untagged, unpushed, dirty, or
untested commit. The immutable commit is authoritative; the recorded branch is useful
context but is not sufficient provenance.

## Recommended implementation order

```text
FJ-13 release/integration policy + FJ-01 stop-the-bleeding dealias fix
   -> FJ-00 parameter contract
   -> FJ-02 true 3-D ICs
   -> FJ-03 constraint diagnosis + production solver decision
   -> FJ-12 measured performance + workhorse decision
   -> FJ-04 ZNF-safe semantics
   -> FJ-05 quench/restart banks
   -> FJ-06 diagnostics and classifier
   -> FJ-07 W&B + sweep/threshold drivers
   -> FJ-08 precision/timestep gates
   -> FJ-10 linear anchor
   -> FJ-09 second magnetic BC
   -> FJ-11 naming/documentation cleanup throughout
   -> FJ-13 immutable release tag + release test artifact
```

FJ-09 can proceed alongside early conducting-wall linear work, but no cross-BC nonlinear
claim should be made until it passes the same gates as the conducting path.

## Minimum validation matrix

| Test | Primitive | Vector potential | Linear stack | Required precision |
|---|---:|---:|---:|---:|
| parameter-resolution contract | yes | yes | yes | exact/schema |
| coupled temporal order | yes | yes | n/a | float64 |
| corrected dealias triad | yes | yes | n/a | float64 |
| linear growth vs eigensolver | yes | yes | cross-stack | float64 |
| finite-amplitude Shenfun parity | yes | yes | n/a | float64 |
| `div_u`, `div_b`, wall residual | yes | yes | eigenfunction | float64 |
| full live-parity tier on release commit | yes | yes | cross-stack | float64 |
| energy-budget closure | yes | yes | n/a | float64 |
| exact/same-spec restart | yes | yes | n/a | float64 |
| changed-Rm quench provenance | campaign solver | campaign solver | n/a | float64 |
| net-flux and ZNF diagnostic finiteness | campaign solver | campaign solver | n/a | both |
| conducting + pseudo-vacuum BC | supported paths | supported paths | yes | float64 |
| f32 scouting equivalence | campaign solver | campaign solver | n/a | f32 vs f64 |
| cost per shear time + cache reuse | benchmark | benchmark | n/a | both |

## Production launch gates

### Net-flux linear pre-pass

The already-verified atlas/theory pre-pass may start immediately if its own commit and
conventions are recorded. A fork-integrated scan requires FJ-00 and the linear part of
FJ-10. Neither depends on W&B.

### Net-flux DNS confirmation

Requires FJ-00, FJ-01, FJ-03 constraint diagnostics/live parity, FJ-06, FJ-08,
FJ-10, and the FJ-13 release gate. Axisymmetric confirmation may use an eigenmode-only
seed; 3-D saturation also requires FJ-02 and the FJ-12 cost/workhorse decision.

### ZNF scouting

Requires all of the following:

- one selected solenoidal production formulation from FJ-03;
- corrected dealiasing and true 3-D perturbations;
- exact flux, ZNF-safe energy/stress semantics, and finite diagnostics;
- checkpoint banks plus changed-Rm continuation;
- spectral-tail/CFL/budget guards;
- rejection of every pre-FJ-01 checkpoint, golden, and parent bank;
- float32-vs-float64 scout calibration.

### Production threshold or survival claims

Additionally require two-resolution and half-timestep invariance, float64 verification,
multiple parent phases/seeds where appropriate, a second magnetic BC for any BC-robust
claim, and local ledgers that reproduce every W&B summary value.

## Definition of done

This plan is complete when a dry-run campaign can:

1. materialize unique, self-consistent net-flux and ZNF specifications from semantic
   CLI overrides;
2. reproduce a committed PCF-MRI marginal point in three linear implementations and in
   float64 DNS;
3. run a genuinely 3-D nonlinear case with bounded constraint, wall, CFL, budget, and
   spectral-tail errors;
4. select several stationary checkpoints and continue them at lower Rm with explicit
   parent/child provenance;
5. classify runs into a compact bracket/frontier without inspecting a mammoth run list;
6. mirror the same complete diagnostics to W&B online or offline without compromising
   the local source of truth;
7. predict and enforce the GPU-hour cost of every 3-D rung using a measured workhorse;
8. reproduce the run from an immutable, clean, remotely preserved release commit; and
9. repeat one scientifically interesting point at higher resolution, half timestep, and
   a second magnetic wall condition.
