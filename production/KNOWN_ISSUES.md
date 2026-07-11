# Known issues / tracked gaps

Durable ledger for accepted-but-open items so they do not get lost between
review rounds. Each entry states the symptom, why it is not fixed yet, and the
acceptance criterion for closing it. Remove entries when closed and reference
the closing commit.

## KI-1: smoke-tier eigenmode seeding can select spurious collocation modes

**Symptom.** At under-resolved smoke tiers (`Nx <= ~12`), the collocation
linear operator (`examples/pcf_linear_jax.py` / `couette/_pcf_linear.py`) can
return a spurious leading eigenvalue (observed: `growth_rate_linear ~ 4e5` at
`Nx=8-12` for the pseudo-vacuum shearbox), so `seed_linear_eigenmode(which=0)`
seeds a numerically spurious mode and labels `growth_rate_linear` with it. The
run itself stays finite and the health guards pass; the label and the seed
shape are the artifact.

**Why open.** Spurious modes are a known hazard of row-replacement collocation
eigenproblems. Filtering them (e.g. a `|Re(lambda)|` cap or a resolution-pair
convergence check inside `eigs`) must be applied identically to both operator
twins and would touch the anchored onset scan (Rm_c = 415.288) and the parity
eigenvalue tests, so it needs its own carefully-tested pass, not a drive-by.

**Workaround.** Trust eigen-seeded `growth_rate_linear` only at `start`/
`production` tiers (converged by `Nx ~ 16-48`); smoke tiers validate
finiteness/divergence only (already the recorded `validation_scope`).

**Close when.** `eigs` filters spurious modes symmetrically in both twins with
the onset anchor and eigenvalue parity tests still passing, plus a regression
test that the `Nx=12` pseudo-vacuum leading mode is physical.

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

## KI-6: adaptive sweep frontier

`production/sweep.py` executes Cartesian grids with per-point status, resume,
and skip-completed (a widened re-invocation is the manual frontier workflow).
Choosing the next points automatically from prior results (bisection toward
onset, refinement near classification boundaries) is not implemented.

## KI-7: TC family not yet on the round-3 diagnostic contract

The total-field mean/split semantics, `energy_convention`/`box_volume` stamps,
and health scalars were applied to the two PCF MHD families (the campaign
workhorses). The Taylor-Couette runners still report their original
diagnostics; align them before any TC campaign.
