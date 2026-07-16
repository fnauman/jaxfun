# jaxfun production commands

## PCF zero-net-flux MRI scout

The fixed-rotation scout uses conducting walls, `S=1`, `Omega=2/3`
(`q=1.5`, `R_Omega=-4/3`), `Re=2000`, `Rm=6000`, and no imposed net
field. Its seed is `B_z=0.1 sin(pi x/h)`, represented as `curl(A)`.

Run the bounded operational check before moving to the GPU server:

```bash
.venv/bin/python -m production.run_problem \
  --config production/runs/pcf_mri_znf_scout_v1.json \
  --out runs/pcf_mri_znf_scout_v1/smoke \
  --device cpu --resolution-tier smoke --steps 2 \
  --profiles-every 1 --diagnostics-every 1
```

For an H100 start-tier scout, retain restart, snapshot, scalar, and multiplane
artifacts explicitly:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python -m production.run_problem \
  --config production/runs/pcf_mri_znf_scout_v1.json \
  --out runs/pcf_mri_znf_scout_v1/start \
  --device gpu --resolution-tier start \
  --checkpoint-every 1000 --snapshot-every 1000 \
  --profiles-every 1000 --diagnostics-every 100
```

The same cadences pass through sweep execution. For example, a one-point
operational launch is:

```bash
.venv/bin/python -m production.sweep \
  --base production/runs/pcf_mri_znf_scout_v1.json \
  --out runs/pcf_mri_znf_scout_v1/sweep \
  --set Rm_h=6000 --execute --resolution-tier start \
  --checkpoint-every 1000 --snapshot-every 1000 \
  --profiles-every 1000 --diagnostics-every 100
```

The PCF `multiplane_v2` file is written to
`profiles/multiplane_v2.h5`. It preserves shearpy's 33 channel names and its
`z_profile`, `xy`, `xz`, and `yz` products. PCF wall-normal means use Galerkin
quadrature; periodic means are arithmetic. Velocity channels are perturbations
about `U_y=-S x`, while magnetic channels contain the represented total field.
EMF channels use total velocity crossed with total represented magnetic field,
matching the induction equation and retaining the base-shear contribution.

## Contract and comparator smoke

```bash
.venv/bin/python -m pytest -q tests/production
```

## Cross-repository comparison manifest

The left input for `shearbox_to_pcf` is the JSON emitted by shearpy's
`build_run_manifest`; the right input is a JAXfun PCF problem spec. Provenance
requires full Git object IDs, not branches or abbreviated hashes.

```bash
.venv/bin/python -m production.comparison_manifest \
  --relation shearbox_to_pcf \
  --left artifacts/shearpy-run-manifest.json \
  --right production/runs/exp_pcf_mri_vector_potential.json \
  --left-repository https://github.com/fnauman/shearpy-jimenez \
  --left-commit 0123456789abcdef0123456789abcdef01234567 \
  --right-repository https://github.com/fnauman/jaxfun \
  --right-commit 89abcdef0123456789abcdef0123456789abcdef \
  --out artifacts/shearbox-pcf-comparison.json
```

For the local PCF-to-Taylor-Couette control map, both inputs are JAXfun specs:

```bash
.venv/bin/python -m production.comparison_manifest \
  --relation local_pcf_to_taylor_couette \
  --left production/runs/exp_pcf_mri_vector_potential.json \
  --right production/runs/exp_tc_mri_vector_potential.json \
  --left-repository https://github.com/fnauman/jaxfun \
  --left-commit 89abcdef0123456789abcdef0123456789abcdef \
  --right-repository https://github.com/fnauman/jaxfun \
  --right-commit 89abcdef0123456789abcdef0123456789abcdef \
  --out artifacts/pcf-tc-comparison.json
```

The output is canonical JSON. `comparison_id` identifies the versioned mapping
and normalized-observable contract; `pair_id` also binds the ordered input
hashes, repositories, and commits. Rebuilding with identical inputs is
byte-for-byte deterministic.

## Immutable production promotion

Promotion is a separate, stricter step than writing a golden. The checkout and
the production run must use the same exact tag, and that tag must already resolve
to the same commit on the selected remote. A branch ref, including `main`, is
not a release identity.

When the release tag is hosted somewhere other than `origin`, launch the full
run with `run_problem --release-remote NAME` and pass the same
`production.promotion --remote NAME`; the remote name is part of the identity.

```bash
.venv/bin/python -m production.promotion \
  --run runs/pcf_mri_vp/full \
  --test-summary artifacts/release-test-summary.json \
  --test-artifact artifacts/junit-fast.xml \
  --test-artifact artifacts/junit-live-shenfun.xml \
  --out releases/pcf-mri-vp-production-v1
```

The JSON test summary must contain integer `failed` and `errors` totals plus
`live_shenfun.passed > 0` and `live_shenfun.skipped == 0`. The referenced
JUnit/log/coverage files are copied into the bundle rather than represented by
manually entered counts.

The command refuses promotion unless all of these are true:

- `uv.lock` exists, registry packages have locked versions, and every remote
  git source is pinned to a full commit;
- the run completed at full production scope under the strict remote-tag gate;
- every diagnostics row is finite, time is strictly increasing across the full
  requested horizon, and every row carries passing constraint evidence;
- the recorded whole-horizon CFL, spectral-tail, and occupancy maxima pass the
  health contract, the energy budget closes, and saturation is stationary;
- classification is `sustained`, resolved, persistently stressed, and based on
  enough independent samples; and
- the golden, metadata, spec, current checkout, dependency hashes, and remote tag
  all identify the same release.

The output directory is created atomically and cannot already exist. Its
`release.json` hashes every archived dependency, run, golden, and test file.

## Device capture

```bash
.venv/bin/python -m production.device --write production/run_env.json
```

Local production runner processes use `JAXFUN_PRODUCTION_DTYPE=float32` when a
spec does not declare `precision`, and set `JAXFUN_ENABLE_X64=0`/
`JAX_ENABLE_X64=0` before importing JAX. A declared precision takes precedence
when the environment does not explicitly override it; both PCF vector-potential
campaign specs declare `float64` for their strict `1e-12` divergence guard.
Normal `import jaxfun` still defaults to x64 unless those env vars override it,
so parity checks that need x64 keep their default behavior.

## Golden comparison

```bash
.venv/bin/python -m production.compare_goldens \
  --problem-id pcf_hydro_laminar_v1 \
  --actual path/to/scalars.json
```

The comparator resolves `production/goldens/<problem_id>/golden/golden.json`
first. If that vendored root is absent it uses `$SHENFUN_GOLDENS_ROOT`, then the
sibling shenfun checkout path.

## Nine-run cheap parity batch

```bash
make -C production parity-cheap
```

This runs the nine cheap goldens, including `pipe_hagen_poiseuille_v1` and
`pipe_womersley_v1`, and writes `runs/_report/results.json`.

## Linear-window DNS parity

```bash
make -C production parity-dns
```

This runs the four committed non-pipe linear-window DNS goldens with the
production runner and a 30-minute timeout per run. Use `parity-dns-pcf` or
`parity-dns-tc` for the geometry-specific subsets.

## Generated-saturation parity

```bash
make -C production parity-saturation
```

This compares current code against the two retained non-quarantined saturation
goldens (`tc_supercritical_saturation`, `tc_mri_nonlinear_saturation`). The
primitive MRI artifact and historical Legendre `pcf_fluct_re400` artifact are
excluded so the batch does not error on the quarantine guard. These
remain qualified-candidate / finite-divergence legacy goldens (see
`production/README.md`), so a green run is a regression pass, not a
campaign-release certification. It uses the full checked-in saturation specs and
can be long; it deliberately does not apply the bounded smoke defaults used by
`validate-all`.

## Validation script parity modes

```bash
make -C production validate-cheap
make -C production validate-dns
make -C production validate-dns-pcf
make -C production validate-dns-tc
```

These use `production/validate_gpu.sh` to run the same wired golden comparisons
as the parity make targets and write `runs/_report/results.json`. Each executed
run is wrapped by `JAXFUN_VALIDATE_TIMEOUT_SECONDS`, defaulting to 1800 seconds,
and writes `logs/<problem_id>.log` with command, exit status, and elapsed seconds
unless `JAXFUN_VALIDATE_LOGS_DIR` overrides the log directory. If an executed run
fails, the script still writes `runs/_report/results.json` with the failed row and
comparison details before exiting nonzero. Strict cheap/DNS parity subprocesses
default to `JAXFUN_VALIDATE_PARITY_DTYPE=float64` because the committed shenfun
goldens carry `1e-10` scalar tolerances; heavy smoke/full runs still default to
float32.

```bash
make -C production validate-all
make -C production validate-tc_supercritical_saturation
```

`validate-all` and direct production run IDs now execute bounded start-tier
smoke by default (`--resolution-tier start --steps 2`). Pass `--smoke` to use
the lighter checked-in `smoke` resolution tier for local CPU/consumer-GPU
development. Non-validate heavy runs also write `golden/golden.json` and
`checkpoints/checkpoints.h5` by default. Reduced or step-limited saturation
runs are reported as `validation_scope=bounded_saturation_smoke`; their generated
artifacts are smoke diagnostics, not full production saturation goldens. The
script prints a `SMOKE ONLY` banner for these bounded heavy/default modes. Pass
`--full` to run the checked-in production spec without smoke defaults, or pass
`--validate-only` for metadata-only validation. `JAXFUN_VALIDATE_RESOLUTION_TIER`,
`JAXFUN_VALIDATE_SMOKE_RESOLUTION_TIER`, `JAXFUN_VALIDATE_HEAVY_STEPS`, and
`JAXFUN_VALIDATE_CHECKPOINT_EVERY` can change the smoke defaults.

## Taylor-Couette MHD cheap parity

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/taylor_couette_mhd_conducting_v1.json \
  --out runs/taylor_couette_mhd_conducting_v1/smoke \
  --compare-golden

.venv/bin/python production/run_problem.py \
  --config production/examples/taylor_couette_mhd_insulating_v1.json \
  --out runs/taylor_couette_mhd_insulating_v1/smoke \
  --compare-golden
```

## PCF MHD/MRI cheap parity

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/pcf_mhd_conducting_v1.json \
  --out runs/pcf_mhd_conducting_v1/smoke \
  --compare-golden

.venv/bin/python production/run_problem.py \
  --config production/examples/pcf_mri_shearbox_v1.json \
  --out runs/pcf_mri_shearbox_v1/smoke \
  --compare-golden
```

## Taylor-Couette hydro cheap parity

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/taylor_couette_hydro_v1.json \
  --out runs/taylor_couette_hydro_v1/smoke \
  --compare-golden
```

## PCF hydro cheap parity

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/pcf_hydro_laminar_v1.json \
  --out runs/pcf_hydro_laminar_v1/smoke \
  --compare-golden
```

## Channel analytic parity

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/channel_poiseuille_hydro_v1.json \
  --out runs/channel_poiseuille_hydro_v1/smoke \
  --compare-golden
```

## CPU/GPU scalar comparison

```bash
.venv/bin/python -m production.compare_devices \
  --config production/examples/channel_poiseuille_hydro_v1.json \
  --out runs/device_compare_channel \
  --device-a cpu \
  --device-b auto \
  --timeout-seconds 1800

.venv/bin/python -m production.compare_devices \
  --config production/runs/pcf_mhd_divfree.json \
  --out runs/device_compare_pcf_mhd_smoke \
  --device-a cpu \
  --device-b auto \
  --resolution-tier smoke \
  --steps 2 \
  --timeout-seconds 1800
```

This launches separate runner subprocesses for each device so JAX backend
selection is process-local, then compares the final numeric diagnostics and writes
`device_comparison.json` with per-side wall times and a left/right speedup ratio. Same-backend runs fail by default so CPU/CPU does not masquerade as CPU/GPU evidence; pass `--allow-same-backend` only for an intentional determinism smoke. Use `--device-b gpu` to require CUDA explicitly. For
Phase J5 production-run specs, pass `--resolution-tier smoke` or
`--resolution-tier start` with `--steps` to get bounded CPU/GPU agreement evidence
without running the full saturation case. The report records the selected
`run_options`.

## Autograd objective smoke

```bash
.venv/bin/python -m pytest -q tests/production/test_objectives.py -n 0
```

This checks the production objective wrappers with finite-difference gradients on
a reduced Plane Couette solver.

## Runner checkpoint, resume, and snapshot smoke

```bash
.venv/bin/python production/run_problem.py \
  --config production/examples/taylor_couette_hydro_dns_v1.json \
  --out runs/taylor_couette_hydro_dns_v1/checkpoint_smoke \
  --steps 4 \
  --checkpoint-every 2 \
  --snapshot-every 2 \
  --diagnostics-every 1

.venv/bin/python production/run_problem.py \
  --resume runs/taylor_couette_hydro_dns_v1/checkpoint_smoke \
  --steps 6 \
  --checkpoint-every 2
```

This writes `checkpoints/checkpoints.h5` with coefficient-space state payloads
readable by `jaxfun.io.read_checkpoint`. With `--snapshot-every`, it writes
atomic per-step HDF5 shards under `snapshots/steps/`, updates
`snapshots/snapshots.h5` as an external-link index without reopening prior
shards, and writes `snapshots/snapshots.xdmf` plus `snapshots/manifest.json`. Production checkpoint attrs include the spec
hash, schema versions, dtype/shape metadata, device metadata, and the diagnostics
path. `--resume` defaults `--config` and `--out` to the resumed run directory when
omitted, validates the checkpoint spec hash/dtype metadata, continues `tstep`,
and appends diagnostics without duplicating the first resumed row.

Before each fixed-step snapshot is persisted, the runner applies the divergence
guard to the candidate state. The configured limit is an inclusive ceiling:
equality is accepted, while a value above the limit aborts without creating the
rejected step's shard or adding it to the snapshot index.

## PCF fluctuation smoke

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/pcf_fluct_re400.json \
  --out runs/pcf_fluct_re400/smoke \
  --resolution-tier start \
  --steps 2
```

This executes the Phase J5 plane-Couette fluctuation runner path at the
checked-in spec's `start` resolution for local smoke coverage and records
`validation_scope=bounded_saturation_smoke`. Full KMM production uses the spec
final time and production resolution and remains a long GPU run.

The full `pcf_fluct_re400` run has a qualified-candidate legacy generated
saturation golden in `production/goldens/pcf_fluct_re400` (regenerate under the
current contract before release) from:

```bash
production/validate_gpu.sh pcf_fluct_re400 --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
finite-amplitude energy-growth check. The generated 807 MB HDF5 payload is
omitted from git; the comparator validates `golden/golden.json` against
`spec.json`.

## PCF MHD/MRI saturation smoke

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/pcf_mhd_divfree.json \
  --out runs/pcf_mhd_divfree/smoke \
  --resolution-tier start \
  --steps 2

.venv/bin/python production/run_problem.py \
  --config production/runs/exp_pcf_mri_shearbox_growth.json \
  --out runs/exp_pcf_mri_shearbox_growth/smoke \
  --resolution-tier start \
  --steps 2
```

These execute the Phase J5 primitive-`b` PCF MHD/MRI runner paths at each
checked-in spec `start` resolution for local smoke coverage and record
`validation_scope=bounded_saturation_smoke`. Full saturation uses each spec final
time and production resolution and remains a long GPU run; `pcf_mhd_divfree`
uses the Phase J5 pinned `N=(32,64,32)` production grid, with its `start` tier
kept lower for bounded smoke.

The full `pcf_mhd_divfree` run has a retained failed generated-saturation
candidate in `production/goldens/pcf_mhd_divfree`; it is not promoted because the
recorded magnetic energy decays below the required 2x threshold. A follow-up
production-resolution float64 GPU run also completed the full 1000-step horizon
(`runs/pcf_mhd_divfree/20260705T205917Z`, 4374.9 seconds) and failed the gate
for the same physics reason: `magnetic_energy_growth_factor=0.35249`,
`stationarity_check_passed=false`, and `saturation_check_passed=false`.

This is not the no-field case: `pcf_mhd_divfree` includes `B0=0.05`, but it
omits `Omega`, so the runner defaults `Omega=0.0` and models stable plain
PCF-MHD rather than rotating MRI/shearbox dynamics. The seeded linear mode has
`growth_rate_linear=-0.0526771745766`, consistent with decay. Use
`exp_pcf_mri_shearbox_growth` for the rotating PCF MRI/shearbox production
comparison. To reproduce the retained failed generated-saturation result, run:

```bash
JAXFUN_PRODUCTION_DTYPE=float64 \
JAXFUN_ENABLE_X64=1 \
JAX_ENABLE_X64=1 \
JAXFUN_VALIDATE_TIMEOUT_SECONDS=10800 \
production/validate_gpu.sh pcf_mhd_divfree --full
```

A passing full run would have to record `validation_scope=generated_saturated_golden`
and `magnetic_energy_growth_factor > 2.0`; the retained candidates record
`saturation_check_passed=false`. The generated 64 MB HDF5 checkpoint is
intentionally not committed; the comparator validates `golden/golden.json`
against `spec.json`.

The full `exp_pcf_mri_shearbox_growth` run produced the now-**quarantined**
primitive-`b` golden in `production/goldens/exp_pcf_mri_shearbox_growth` from:

```bash
production/validate_gpu.sh exp_pcf_mri_shearbox_growth --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
growth gate, but the primitive-`b` field is not solenoidal at finite MRI
amplitude (`div B` grew to `2.67e-2`, past the `1e-2` guard), so the golden is
quarantined and forbidden from production seeding -- it is excluded from
`parity-saturation` and covered only by the quarantine regression in
`tests/production/test_compare_goldens.py`. Use the vector-potential workhorse
(`exp_pcf_mri_vector_potential`) for rotating PCF MRI production work. The
generated 385 MB HDF5 checkpoint is intentionally not committed; the comparator
validates `golden/golden.json` against `spec.json`.

## Vector-potential MHD/MRI runs (div B = 0 preserving, both geometries)

The solenoidal-preserving `B = B0 + curl(A)` configurations for both
geometries and both wall types:

```bash
# PCF, conducting (the selected workhorse)
.venv/bin/python production/run_problem.py \
  --config production/runs/exp_pcf_mri_vector_potential.json \
  --out runs/exp_pcf_mri_vector_potential/smoke \
  --resolution-tier start --steps 2

# PCF, true insulating (vacuum-matched) walls
.venv/bin/python production/run_problem.py \
  --config production/runs/exp_pcf_mri_vp_insulating.json \
  --out runs/exp_pcf_mri_vp_insulating/smoke \
  --resolution-tier start --steps 2

# Taylor-Couette, full 3D, conducting cylinders (E_tang = 0 exact)
.venv/bin/python production/run_problem.py \
  --config production/runs/exp_tc_mri_vector_potential.json \
  --out runs/exp_tc_mri_vector_potential/smoke \
  --resolution-tier start --steps 2

# Taylor-Couette, full 3D, insulating cylinders (per-mode Bessel matching)
.venv/bin/python production/run_problem.py \
  --config production/runs/exp_tc_mri_vp_insulating.json \
  --out runs/exp_tc_mri_vp_insulating/smoke \
  --resolution-tier start --steps 2
```

All four seed the matching linear eigensolver and stamp
`representation=vector_potential`; the insulating runs additionally emit
`insulating_bc_residual` (the wall vacuum-matching witness). CPU anchor
evidence and measured `div B` floors are recorded in
[`README.md`](README.md#magnetic-representation-and-divergence-evidence);
none of the four has a committed full-resolution GPU golden yet.

To run with adaptive-CFL stepping, add an
`adaptive_cfl` block to the spec's `time` section, e.g.
`"adaptive_cfl": {"target": 0.4, "check_every": 25, "dt_min": 1e-6,
"dt_max": 0.01}`; the run then records `n_dt_changes`, `dt_final`,
`dt_min_used`/`dt_max_used`, `cfl_total_max_observed`, and per-row `dt` and
`cfl_total`. Adaptive runs support resume-exact, checkpoint banks, snapshots,
and PCF profiles; controller `dt`, CFL-check phase, and any in-progress
exact-endpoint redistribution schedule are checkpointed.

## Taylor-Couette saturation smoke

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/tc_supercritical_saturation.json \
  --out runs/tc_supercritical_saturation/smoke \
  --resolution-tier start \
  --steps 2

.venv/bin/python production/run_problem.py \
  --config production/runs/tc_mri_nonlinear_saturation.json \
  --out runs/tc_mri_nonlinear_saturation/smoke \
  --resolution-tier start \
  --steps 2
```

These execute the Phase J5 Taylor-Couette hydro and MHD/MRI saturation runner
paths at each checked-in spec's `start` resolution for local smoke coverage and
record `validation_scope=bounded_saturation_smoke`. Full saturation uses each
spec final time and production resolution and remains a long GPU run; omit
`--resolution-tier start` or pass `--resolution-tier production` for that path.
Full non-bounded saturation runs fail if their emitted `saturation_check_passed`
diagnostic is missing or false. Bounded smoke rows are reported as smoke/skipped
validation evidence rather than full production passes.

The full `tc_supercritical_saturation` run has a qualified-candidate legacy
generated saturation golden in `production/goldens/tc_supercritical_saturation`
(current TC diagnostic-contract and release gates remain open) from:

```bash
production/validate_gpu.sh tc_supercritical_saturation --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
hydro saturation check. The generated 25 MB HDF5 checkpoint is intentionally not
committed; the comparator validates `golden/golden.json` against `spec.json`.

The full `tc_mri_nonlinear_saturation` run has a finite-divergence-only legacy
generated saturation golden in `production/goldens/tc_mri_nonlinear_saturation`
from:

```bash
production/validate_gpu.sh tc_mri_nonlinear_saturation --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
growth gate (`magnetic_energy_growth_factor=8.2e6`), but the primitive/direct-`b`
field ended at `div B=7.96e-4` -- below the coarse `1e-2` health ceiling yet not
roundoff-solenoidal, so it is a regression reference, not a campaign-release
golden. The generated 52 MB HDF5 checkpoint is intentionally not committed; the
comparator validates `golden/golden.json` against `spec.json`.

## Runner metadata validation

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/tc_supercritical_saturation.json \
  --out runs/tc_supercritical_saturation/validate_only \
  --validate-only
```

Non-`--validate-only` execution now runs wired DNS/heavy specs; unsupported or
intentionally unwired specs still exit with status 1/2 before solver allocation.
Runner metadata also records the persistent JAX compilation-cache path and DNS
timing fields (`solver_steps`, `ms_per_step`, `steps_per_second`) so throughput
regressions are visible in `metadata.json` and reports.

## Curl-workhorse continuation (checkpoint / resume / quench)

The vector-potential (curl) PCF-MRI family serializes its MHDState (KMM flow
block + A coefficients, `state_kind=pcf_vector_potential_mhd_saturation`):

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/exp_pcf_mri_vector_potential.json \
  --out runs/exp_pcf_mri_vector_potential/parent \
  --resolution-tier smoke --steps 4 --checkpoint-every 2 --checkpoint-bank \
  --diagnostics-every 2

# resume-exact continuation (same spec_hash enforced)
.venv/bin/python production/run_problem.py \
  --resume runs/exp_pcf_mri_vector_potential/parent

# FJ-05 quench: child spec may change only nu/eta (Re/Rm); baselines are taken
# from the loaded parent state. The child runs 20 physical-time units beyond it.
.venv/bin/python production/run_problem.py \
  --config <child-spec>.json --out runs/.../quench \
  --quench runs/exp_pcf_mri_vector_potential/parent --additional-time 20
```

A quench must provide exactly one explicit horizon: `--additional-time T` or
`--additional-steps N`. Fixed stepping requires `T` to be an integer multiple
of the child `dt`. For adaptive stepping, time is authoritative;
`--additional-steps N` is a nominal-time shorthand for `N * initial_dt`, and
the attained adaptive step count is recorded at completion. `--steps` remains
an absolute target for fresh runs and resume-exact only; it cannot be combined
with `--quench`. The immutable child spec's `time.final_time` does not define
the quench horizon.

### Parent bank: multiple plateau times

Add `--checkpoint-bank` to the parent so every `--checkpoint-every` interval is
retained immutably under `checkpoints/bank/` with a provenance manifest
(`index.json`: tstep, state time, spec hash, sha256). A quench then selects any
plateau via `--quench-step <tstep>`, and `--burn-in-steps N` excludes the first
N child steps from the fitted history (stationarity, classification,
correlation time, budget), not only from metadata.

```bash
.venv/bin/python production/run_problem.py --config <child>.json --out runs/.../q415 \
  --quench runs/.../parent --quench-step 24000 --additional-steps 12000 \
  --burn-in-steps 2000
```

`metadata.json` records the selected parent time/step, requested additional
duration, resolved absolute target, and attained final time/step under the
versioned `quench.duration` block. Quench children use
`validation_scope=quench_continuation`: finite/divergence health remains a hard
gate, while growth, decay, and saturation are recorded as scientific outcomes
rather than launch-success requirements. Burn-in must satisfy
`0 <= --burn-in-steps < additional steps`. For adaptive quenches, preflight
uses the conservative minimum step count implied by `dt_max`, then validates
the attained step count again at completion. Golden comparison and promotion
are not quench workflows and are rejected. A failed or interrupted solve leaves the
certified `attained` fields unset and records only a conservative
`last_observed` cadence lower bound when one is available.

### First-passage survival ensembles

Build transition-regime survival curves from completed and interrupted quench
children with explicit, reviewable decay controls:

```bash
.venv/bin/python -m production.survival \
  runs/quench/q1 runs/quench/q2 runs/quench/q3 \
  --energy-key mag_energy_fluct --threshold 1e-8 --dwell-time 5 \
  --cluster-bootstrap 2000 --seed 0 --out runs/quench/survival.json
```

Every diagnostic time is converted to age since the authenticated parent
checkpoint. Burn-in samples are excluded using
`classification_valid_after_tstep`. A decay event is recorded at the first
below-threshold age only when subsequent samples remain below the threshold for
the requested dwell time; the later qualification age is stored separately.
Unqualified, interrupted, or wall-time-limited histories are right-censored at
their last canonical diagnostic sample.

The output separates ensembles by the complete numerical, physical, provenance,
energy, and decay-policy key set recorded in `grouping_keys`. It reports
Kaplan-Meier risk sets, pointwise log-log Greenwood intervals, and percentile
intervals from resampling whole parent clusters. Quenches from different
checkpoints of the same parent run therefore remain dependent during bootstrap
uncertainty estimation.

### Cartesian sweeps

```bash
.venv/bin/python -m production.sweep \
  --base production/runs/exp_pcf_mri_vector_potential.json \
  --out runs/sweeps/rm_scan \
  --grid '{"Rm_h": [400, 600, 800], "B0": [0.025]}' \
  --execute --resolution-tier smoke --steps 200
```

Materializes every combination (validated, physics-resolved, archived per run
id), executes serially, and records per-point status plus canonical
`operational_status`, `scientific_class`, fit uncertainty, and eligibility in
`sweep_index.json`. Re-invocation (including with a widened grid) skips
completed points.

A one-axis transition frontier can refine those results automatically:

```bash
.venv/bin/python -m production.sweep \
  --base production/runs/exp_pcf_mri_vector_potential.json \
  --out runs/frontiers/rm_onset \
  --frontier "{\"axis\":\"Rm_h\",\"bounds\":[400,800],\"fixed\":{\"B0\":0.025},\"abs_tolerance\":25,\"confidence_z\":1.96,\"max_refinements\":8}" \
  --execute --resolution-tier smoke --steps 200
```

Each decision is appended to `frontier_index.json` with endpoint run IDs and
spec hashes plus a SHA-256 parent link. Growing/decayed endpoints must exclude
zero at the requested slope-confidence width; a `sustained` endpoint has
already passed the classifier stationarity, stress, and independent-sample
gates. Marginal, inconclusive, nonmonotonic, or uncertain evidence stops without
claiming convergence. Re-invocation verifies the full hash chain and resumes the
pending midpoint.

For Taylor-Couette, `Re_h` and `Rm_h` are midpoint-local controls,
`|S_mid| h^2/nu` and `|S_mid| h^2/eta`. Materialized specs and run metadata
also record the native inner-cylinder values `Re_TC` and `Rm_TC`; legacy `Re`
and `Rm` remain aliases for those native values. `R1`, `R2`, `Omega1`, and
`Omega2` are sweep controls; radius changes update the radial domain and radius
ratio, while curvature and both Reynolds-number conventions are re-derived.
The tiered `resolution` control exposes the modal axes consumed by each solver,
including `Nr`, `Ntheta`, and `Nz` for the 3D vector-potential family. TC
production solvers use the full `2*pi` annulus, so Cartesian `Ly` and
azimuthal-wedge overrides are rejected; `Lz` remains available for nonlinear TC
runs.

Override availability follows the selected oracle. Static validation oracles
reject inert box, time, coefficient, and resolution axes. Magnetic wall-family
changes are not sweep axes: use the separate conducting/insulating base spec so
the `problem_id`, expected oracle, and golden artifact remain consistent.

### Health contract

Saturation runs through the PCF families emit the CFL decomposition
(advective/Alfven per direction + the implicit-diffusion number), per-axis
spectral tail fractions, retained-mode occupancy, the correlation time of
`total_stress`, and (curl family) the shearing-box `energy_budget_residual`
(closes to ~5e-5 on the smoke anchor). Runs beyond the thresholds in
`production/health.py` are classified `inconclusive` (`underresolved`) instead
of trusted. Cadence rows also stream to `diagnostics.partial.jsonl` during the
solve, so a crash leaves the same history locally that any mirror received.

## Pseudo-vacuum magnetic walls (FJ-09, primitive family)

`production/runs/exp_pcf_mri_pseudo_vacuum.json` runs the primitive-b solver
with `d_x b_x = 0`, `b_y = b_z = 0` walls; the eigenmode seed and
`growth_rate_linear` come from the pseudo-vacuum linear operator. Live parity
covers both walls (`test_pcf_primitive_3d_finite_amplitude_matches_live_shenfun`).
The vector-potential form rejects pseudo-vacuum until A-formulation BCs exist.

## Performance measurement (FJ-12)

```bash
.venv/bin/python -m production.benchmark \
  --config production/runs/exp_pcf_mri_vector_potential.json \
  --tiers smoke,start --timed-steps 10 --out runs/bench/vp_cpu.json
```

Times the *real* production solver per materialized tier (compile vs warm step),
fits the power-law cost model over >= 2 tiers, and predicts hours for the spec's
horizon. The workhorse decision still requires this CLI on the authorized GPU.

## Release gate defaults (FJ-13)

`--write-golden` and production-scale runs (no `--steps`, no smoke/start tier)
of `support_state: production` DNS specs refuse a dirty/unpushed/mutable-ref
worktree by default; `--allow-dirty` archives the diff (tracked + untracked)
for an explicit discovery run. `--wandb` streams cadence rows live during the
solve and errors out if `wandb` is not installed (`pip install .[wandb]`).
