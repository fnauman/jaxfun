# jaxfun production commands

## Contract and comparator smoke

```bash
.venv/bin/python -m pytest -q tests/production
```

## Device capture

```bash
.venv/bin/python -m production.device --write production/run_env.json
```

Local production runner processes use `JAXFUN_PRODUCTION_DTYPE=float32` by
default and set `JAXFUN_ENABLE_X64=0`/`JAX_ENABLE_X64=0` before importing JAX.
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

This compares current code against the promoted generated saturation goldens. It
uses the full checked-in saturation specs and can be long; it deliberately does
not apply the bounded smoke defaults used by `validate-all`.

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
readable by `jaxfun.io.read_checkpoint`, plus `snapshots/snapshots.h5`,
`snapshots/snapshots.xdmf`, and `snapshots/manifest.json` when
`--snapshot-every` is provided. Production checkpoint attrs include the spec
hash, schema versions, dtype/shape metadata, device metadata, and the diagnostics
path. `--resume` defaults `--config` and `--out` to the resumed run directory when
omitted, validates the checkpoint spec hash/dtype metadata, continues `tstep`,
and appends diagnostics without duplicating the first resumed row.

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

The full `pcf_fluct_re400` run has a promoted generated saturated golden in
`production/goldens/pcf_fluct_re400` from:

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

The full `pcf_mhd_divfree` run currently has a retained failed generated-saturation
candidate in `production/goldens/pcf_mhd_divfree`; it is not promoted because the
recorded magnetic energy decays below the required 2x threshold. Re-run with:

```bash
production/validate_gpu.sh pcf_mhd_divfree --full
```

A passing full run must record `validation_scope=generated_saturated_golden`
and `magnetic_energy_growth_factor > 2.0`; the retained candidate records
`saturation_check_passed=false`. The generated 64 MB HDF5 checkpoint is
intentionally not committed; the comparator validates `golden/golden.json`
against `spec.json`.

The full `exp_pcf_mri_shearbox_growth` run has a promoted generated saturated
golden in `production/goldens/exp_pcf_mri_shearbox_growth` from:

```bash
production/validate_gpu.sh exp_pcf_mri_shearbox_growth --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
MRI saturation check. The generated 385 MB HDF5 checkpoint is intentionally not
committed; the comparator validates `golden/golden.json` against `spec.json`.

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

The full `tc_supercritical_saturation` run has a promoted generated saturated
golden in `production/goldens/tc_supercritical_saturation` from:

```bash
production/validate_gpu.sh tc_supercritical_saturation --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
hydro saturation check. The generated 25 MB HDF5 checkpoint is intentionally not
committed; the comparator validates `golden/golden.json` against `spec.json`.

The full `tc_mri_nonlinear_saturation` run has a promoted generated saturated
golden in `production/goldens/tc_mri_nonlinear_saturation` from:

```bash
production/validate_gpu.sh tc_mri_nonlinear_saturation --full
```

That run records `validation_scope=generated_saturated_golden` and passed the
MRI saturation check. The generated 52 MB HDF5 checkpoint is intentionally not
committed; the comparator validates `golden/golden.json` against `spec.json`.

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
