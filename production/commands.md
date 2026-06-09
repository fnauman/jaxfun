# jaxfun production commands

## Contract and comparator smoke

```bash
.venv/bin/python -m pytest -q tests/production
```

## Device capture

```bash
.venv/bin/python -m production.device --write production/run_env.json
```

Local development runs use `JAXFUN_PRODUCTION_DTYPE=float32` by default through
the production device metadata. x64 remains enabled after `import jaxfun` for
parity checks that need it.

## Golden comparison

```bash
.venv/bin/python -m production.compare_goldens \
  --problem-id pcf_hydro_laminar_v1 \
  --actual path/to/scalars.json
```

The comparator resolves `production/goldens/<problem_id>/golden/golden.json`
first. If that vendored root is absent it uses `$SHENFUN_GOLDENS_ROOT`, then the
sibling shenfun checkout path.

## Seven-run cheap parity batch

```bash
make -C production parity-cheap
```

This runs the seven non-pipe cheap goldens and writes `runs/_report/results.json`.
Pipe hydro remains skipped until the axis-regular radial basis lands.

## Taylor-Couette DNS parity

```bash
make -C production parity-dns-tc
```

This runs the two committed Taylor-Couette linear-window DNS goldens with the
production runner and a 30-minute timeout per run.

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

## Runner metadata validation

```bash
.venv/bin/python production/run_problem.py \
  --config production/runs/tc_supercritical_saturation.json \
  --out runs/tc_supercritical_saturation/validate_only \
  --validate-only
```

Non-`--validate-only` execution currently exits with status 2 for DNS/heavy specs
that are not yet wired into the production runner.
