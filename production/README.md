# jaxfun production layer

This package is the jaxfun side of the shared shenfun production contract. It
vendors the neutral problem-spec schema, example specs, 13 committed shenfun
goldens, and five generated jaxfun saturation artifacts so parity checks do not
import or require a live shenfun process. None of the five committed saturation
artifacts is release-ready as-is: one is quarantined (non-solenoidal primitive-`b`
MRI), one is a failed candidate (decayed below the growth gate), two are
qualified candidates pending regeneration under the current contract, and one is
finite-divergence only (saturated but not roundoff-solenoidal). The per-artifact
status is in the inventory below.

The intended rotating-MHD campaign compares the Jaxfun plane-Couette and
Taylor-Couette runners here with the shearing box in
`/home/nauman/cfd/shearpy-jimenez`. The production inventory and validation
scopes are documented below; open implementation and measurement gates remain
authoritative in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Production-run inventory

| problem_id | geometry | physics | solver file | status | fallback rung |
|---|---|---|---|---|---|
| `pcf_fluct_re400` | pcf | hydro | `examples/pcf_fluctuations_jax.py` | qualified candidate: runner, smoke coverage, and a legacy full-resolution GPU saturation artifact exist; regenerate under the current contract before release | rung 3 |
| `pcf_mhd_divfree` | pcf | mhd | `examples/pcf_mri_primitive_jax.py` | failed candidate: the direct/primitive-`b` full `N=(32,64,32)` run completed with small finite `div B`, but decayed and failed the saturation growth gate (not a divergence failure) | rung 3 |
| `exp_pcf_mri_shearbox_growth` | pcf | mri | `examples/pcf_mri_primitive_jax.py` | quarantined: direct/primitive-`b` MRI reached `div B=1.34e-2` by `t=30`; its old saturated artifact is regression-only and forbidden from production seeding | rung 1/2/3 |
| `exp_pcf_mri_vector_potential` | pcf | mri | `examples/pcf_mhd_mri_shearpy_jax.py` | selected workhorse pending full run: `B=B0+curl(A)`, 300-step CPU qualification and start-tier GPU smoke pass with roundoff `div B`; full-resolution GPU golden still required | rung 1/2/3 |
| `tc_supercritical_saturation` | taylor_couette | hydro | `examples/taylor_couette_dns_jax.py` | qualified candidate: runner, smoke coverage, and a legacy full-resolution GPU saturation artifact exist; current TC diagnostic-contract and release gates remain | rung 2/3 |
| `tc_mri_nonlinear_saturation` | taylor_couette | mri | `examples/taylor_couette_dns_jax.py` | finite-divergence only: the direct-`b` legacy full run saturated but ended at `div B=7.96e-4`; below the generic health ceiling, not roundoff-solenoidal or campaign-ready | rung 1/2/3 |
| `stab_PCF_MRI_stability` | pcf | mri | `examples/pcf_mhd_mri_shearpy_jax.py` | config-undetermined placeholder | not executable |

`pcf_mhd_divfree` is not a no-field run: the spec includes `B0=0.05`, but it
does not declare rotation, so production defaults `Omega=0.0`. The follow-up
float64 full run therefore tested stable plain PCF-MHD; its leading seeded linear
eigenvalue was negative (`growth_rate_linear=-0.0526771745766`) and the nonlinear
run decayed (`magnetic_energy_growth_factor=0.35249`). For rotating PCF
MRI/shearbox production work, use the vector-potential configuration
`exp_pcf_mri_vector_potential`; the primitive-`b`
`exp_pcf_mri_shearbox_growth` artifact is quarantined.

## Readiness terminology

Readiness is implementation- and configuration-specific, not a blanket status
for a geometry:

- `production_ready_limited_scope`: fully wired and parity-tested for the narrow
  analytic or axisymmetric scope stated in the table. It does not imply a full
  3-D nonlinear campaign solver.
- `qualified_candidate`: the nonlinear runner and legacy full-resolution
  evidence exist, but current-contract regeneration, open diagnostics, or
  release gates remain.
- `selected_workhorse_pending_full_run`: the implementation selected for the
  campaign passes current nonlinear qualification and GPU smoke, but lacks its
  full-resolution production golden.
- `finite_divergence_only`: the path emits finite `div B` above roundoff but
  below the generic `1e-2` health ceiling. This is a divergence-quality tag
  only -- it says the primitive-`b` discretization is not roundoff-solenoidal,
  independent of whether a given run saturated (e.g. `tc_mri_nonlinear_saturation`)
  or decayed. The `1e-2` ceiling catches catastrophic constraint failure; it does
  **not** establish `div B = 0` to discretization error or roundoff. A
  decayed-and-failed artifact is a `failed candidate` (growth), not this tag.
- `quarantined`: retained only for regression/provenance and rejected as a
  production seed or comparison reference.
- `linear_only` and `unsupported`: no qualified nonlinear DNS path exists for
  that case.

No nonlinear magnetic Couette implementation currently has an immutable
campaign release. The conducting PCF vector-potential/curl family is the
selected magnetic workhorse, while the direct/primitive-`b` PCF and TC cases
below are explicitly distinguished as finite-divergence or quarantined.

### Magnetic representation and divergence evidence

The single discriminator is the magnetic representation. Only the
vector-potential form `B = B0 + curl(A)` (`examples/pcf_mhd_mri_shearpy_jax.py`)
makes `div B = div(curl A) = 0` an identity, so `div B` stays at roundoff for the
whole horizon and does not grow. Every primitive/direct-`b` path
(`examples/pcf_mri_primitive_jax.py`, `examples/taylor_couette_dns_jax.py`) evolves
the field components directly and does **not** enforce the solenoidal constraint;
its `div B` stays small only while amplitudes are low and grows into the finite
regime at finite MRI amplitude.

| Case and evidence | Magnetic representation | Preserves `div B=0`? | `div u` (L2) | `div B` (L2) | Qualification |
|---|---|---|---:|---:|---|
| PCF plain MHD, decaying GPU smoke (`pcf_mhd_divfree`) | primitive/direct `b` | no (small while decaying) | `1.05e-8` | `7.15e-6` | failed candidate; decayed, not a saturation reference |
| PCF growing MRI, GPU smoke at `t=30` (`exp_pcf_mri_shearbox_growth`) | primitive/direct `b` | no — grows past the `1e-2` guard | not recorded at guard exit | `1.34e-2` | guard failure; primitive MRI is not production-ready |
| PCF growing MRI, older full artifact (`exp_pcf_mri_shearbox_growth`) | primitive/direct `b` | no — grew to `2.67e-2` | `2.30e-2` | `2.67e-2` | quarantined and forbidden from production seeding |
| PCF MRI, 300-step CPU qualification + GPU smoke (`exp_pcf_mri_vector_potential`) | vector potential, `B=B0+curl(A)` | yes — by construction, non-growing | `~1e-16` (CPU) | GPU `~1e-18`; CPU `~2.5e-16` (max over 300 steps) | selected conducting-wall workhorse; gated `< 1e-12` over the whole horizon; full-resolution GPU golden pending |
| Taylor-Couette conducting MHD/MRI, legacy full artifact (`tc_mri_nonlinear_saturation`) | primitive/direct `b` | no — grew to `7.96e-4` | `2.26e-5` | `7.96e-4` | finite-divergence only; saturated but not roundoff-solenoidal |

Every `div u`/`div B` cell above is a **measured** emitted L2 diagnostic (not a
gate) from different resolutions, horizons, and solver families; the gate
thresholds live in the Qualification column and in the tests
(`tests/production/test_vector_potential_oracle.py`,
`tests/production/test_workhorse_qualification.py` gate the curl family at
`div B < 1e-12` across the whole horizon). They are qualification flags, not a
cross-geometry convergence comparison. The PCF GPU evidence is recorded in
[`promotions/fj03_gpu_smoke_findings.md`](promotions/fj03_gpu_smoke_findings.md);
the full-artifact values come from the committed golden diagnostics.

## Support matrix

| Geometry | Physics path | Support state | Internal formulation | jaxfun source files | Axis mapping | Boundary/sign conventions | Golden / fallback | Divergence keys | Tests |
|---|---|---|---|---|---|---|---|---|---|
| pcf | hydro | qualified_candidate | cheap and linear-window DNS parity, wired nonlinear runner, smoke coverage, and a legacy full GPU saturation artifact; regenerate under the current contract before release | `examples/pcf_fluctuations_jax.py`, `examples/pcf_linear_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip moving walls, `U_b=U_wall*x e_y` | `pcf_hydro_laminar_v1`; DNS `pcf_hydro_primitive_dns_v1`; heavy `pcf_fluct_re400` | `divergence_l2`; DNS `divergence_u` | cheap/DNS parity, smoke, and legacy saturation regressions |
| channel | hydro | production_ready_limited_scope | driven KMM pressure-gradient steady state with golden-normalized Poiseuille observables | `examples/channelflow_kmm.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip walls, pressure-gradient drive | `channel_poiseuille_hydro_v1` | `divergence_l2` | KMM steady-profile regression and CLI golden comparison |
| pcf | plain mhd, primitive `b` | finite_divergence_only | direct magnetic components; cheap linear parity and low-amplitude/decaying nonlinear execution work, but the full candidate decayed and was not promoted | `examples/pcf_mhd_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting walls; `B0=0.05`; `Omega=0` | `pcf_mhd_conducting_v1`; failed `pcf_mhd_divfree` candidate | `divergence_u_l2`, `divergence_b_l2` | finite-divergence smoke and failed-candidate regression |
| pcf | mri, primitive `b` | quarantined | direct magnetic components do not preserve the solenoidal constraint at finite MRI amplitude; linear and short DNS parity remain diagnostic-only | `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting or pseudo-vacuum walls, Coriolis/shear, imposed `B0 e_z` | cheap `pcf_mri_shearbox_v1`; DNS `pcf_mri_primitive_dns_v1`; quarantined `exp_pcf_mri_shearbox_growth` | `divergence_u_l2`, `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | quarantine and divergence-guard regressions |
| pcf | mri, vector potential | selected_workhorse_pending_full_run | nonlinear `B=B0+curl(A)` workhorse; current 300-step CPU qualification and GPU start-tier smoke pass, with checkpoint/resume/quench support; full production golden pending | `examples/pcf_mhd_mri_shearpy_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting walls only, Coriolis/shear, imposed `B0 e_z` | fallback rungs 1/2/3; full curl golden not yet committed | `divergence_u_l2`, `divergence_b_l2` | vector-potential oracle and long nonlinear workhorse qualification |
| taylor_couette | hydro | qualified_candidate | cheap and linear-window DNS parity, wired nonlinear runner, smoke coverage, and a legacy full GPU saturation artifact; current TC diagnostic contract remains open | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_linear_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | rotating no-slip cylinders, `V=A*r+B/r` | `taylor_couette_hydro_v1`; DNS `taylor_couette_hydro_dns_v1`; heavy `tc_supercritical_saturation` | `divergence_l2`; DNS `divergence_linf` | cheap/DNS parity, smoke, and legacy saturation regressions |
| taylor_couette | mhd/mri conducting, primitive `b` | finite_divergence_only | direct magnetic components; cheap and linear-window DNS parity plus a legacy full GPU saturation artifact, but not roundoff-solenoidal and not on the current TC diagnostic contract | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | conducting walls, Alfven units, `Pm=Rm/Re` | `taylor_couette_mhd_conducting_v1`; DNS `taylor_couette_mhd_dns_v1`; legacy `tc_mri_nonlinear_saturation` | `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | cheap/DNS parity, smoke, and legacy saturation regressions |
| taylor_couette | mhd insulating | linear_only | cheap insulating linear parity only; no qualified nonlinear insulating DNS path | `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | insulating only for `m=0`, `kz!=0` | `taylor_couette_mhd_insulating_v1` | `divergence_b_l2` | linear golden and loader rejection for `m!=0` |
| pipe | hydro | production_ready_limited_scope | axisymmetric regular-axis Hagen-Poiseuille and Womersley oracle; full 3-D non-axisymmetric pipe DNS remains out of scope | `examples/pipe_flow_dns_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | no-slip wall, regular axis at `r=0` | `pipe_hagen_poiseuille_v1`, `pipe_womersley_v1` | `divergence_l2` | golden comparisons wired for both pipe hydro goldens |
| pipe | mhd/mri | unsupported | no shenfun production formulation | none | `axis_0=r`, `axis_1=theta`, `axis_2=z` | unsupported | rejected to match shenfun | n/a | loader rejection test |

## Validation scripts

Current implemented entry points:

- `production/device.py` captures backend/device/dtype/golden-policy metadata.
- `production/compare_goldens.py` validates `schema_version=1` goldens and compares
  tolerance-declared scalars.
- `production/run_problem.py --validate-only` validates a spec and writes metadata;
  unsupported or intentionally unwired specs still fail explicitly before solver allocation.
- `production/run_problem.py --resume RUN_DIR` resumes from the latest production
  checkpoint after validating spec hash and dtype metadata; diagnostics append from
  the resumed step.
- `make -C production parity-cheap` runs the nine cheap golden comparisons, including the two pipe hydro goldens, and writes `runs/_report/results.json`.
- `make -C production parity-dns` runs the four committed non-pipe linear-window DNS golden comparisons and writes `runs/_report/results.json`; `parity-dns-pcf` and `parity-dns-tc` run geometry-specific subsets.
- `make -C production parity-saturation` runs the three retained non-quarantined
  saturation regressions (`pcf_fluct_re400`, `tc_supercritical_saturation`,
  `tc_mri_nonlinear_saturation`) against their committed goldens. The quarantined
  primitive-`b` PCF MRI artifact (`exp_pcf_mri_shearbox_growth`) is excluded so
  the batch does not error on the quarantine guard. These three remain
  qualified-candidate / finite-divergence legacy goldens (not roundoff-solenoidal,
  pending regeneration under the current contract), so a green run is a regression
  pass, not a campaign-release certification; the curl workhorse golden is still
  required for a solenoidal MRI release.
- `production/validate_gpu.sh cheap|dns|dns-pcf|dns-tc` runs the same wired parity groups with a 30-minute timeout per run, writes `logs/<problem_id>.log` with command/status/duration, and writes `runs/_report/results.json`; strict parity subprocesses default to `JAXFUN_VALIDATE_PARITY_DTYPE=float64` for the committed `1e-10` goldens. `all`, `heavy`, and direct production run IDs execute bounded start-tier float32 smoke by default (`--resolution-tier start --steps 2`) unless `--full`, `--validate-only`, `--smoke`, or explicit run args are supplied; `--smoke` selects the lighter checked-in `smoke` resolution tier. Non-validate heavy runs also write `golden/golden.json` and `checkpoints/checkpoints.h5` by default, but reduced/step-limited saturation artifacts are labeled `bounded_saturation_smoke`, not full production saturation goldens.
- `production/run_problem.py --checkpoint-every K` writes HDF5 coefficient checkpoints for wired DNS paths through `production/checkpoint.py`, including spec hash, dtype/shape metadata, device metadata, and diagnostics pointer attrs.
- `production/run_problem.py --snapshot-every K` writes uniform HDF5 snapshots as atomic per-step HDF5 shards,
  updates a `snapshots.h5` external-link index without reopening prior shards, and emits an XDMF sidecar plus
  a snapshot manifest. `--diagnostics-every K` controls
  host-side diagnostic rows and mid-run finite/divergence monitoring cadence.
- Runner metadata records `compilation_cache` plus timing fields such as
  `solver_steps`, `ms_per_step`, and `steps_per_second` for DNS paths.
- `production/objectives.py` exposes differentiable final-energy, integrated-energy, stress/alpha, growth-proxy, and PCF minimal-seed objectives with finite-difference tests.
- `production/compare_devices.py` runs the same config in separate device-specific subprocesses, compares final numeric diagnostics for CPU/GPU agreement checks, and records left/right wall times plus speedup; production run specs can pass `--resolution-tier smoke|start|production` plus `--steps` for bounded agreement evidence. Same-backend comparisons fail by default; pass `--allow-same-backend` only for intentional CPU/CPU smoke checks.
- `production/report.py` builds machine-readable summaries from run metadata, including `validation_scope`, `checked_observables`, fallback rung fields, and failed comparison details. `production/validate_gpu.sh` writes this report before exiting nonzero when an executed run fails.

## Validation scopes

`run_problem.py` writes `metadata.json.validation_scope`, and `report.py` carries the same field into `runs/_report/results.{json,md}`:

- `golden_comparison`: diagnostics were compared against a resolved committed shenfun golden.
- `cpu_smoke_fallback_oracle`: CPU smoke for a saturation run with rung-1 or rung-2 fallback checks available.
- `cpu_smoke_finiteness_divergence_only`: CPU smoke for rung-3-only saturation specs such as `pcf_fluct_re400` and `pcf_mhd_divfree`; this proves solver completion, finite diagnostics, and emitted divergence diagnostics, not production parity.
- `bounded_saturation_smoke`: GPU or CPU saturation execution with `--steps`, `--resolution-tier start`, or `--resolution-tier smoke`; generated golden/checkpoint files are smoke artifacts, not full production saturation goldens, and report rows are skipped rather than counted as production passes.
- `generated_saturated_golden`: full saturation execution without bounded smoke overrides; if `saturation_check_passed` is missing or false (including non-boolean values), stationarity fails, diagnostics are non-finite/missing divergence, or final divergence exceeds the generic `1e-2` ceiling, the runner marks metadata failed and exits nonzero. Passing this coarse floor alone is not proof of a solenoidal magnetic discretization.
- `oracle_execution`: analytic, linear, or DNS oracle execution without a committed-golden comparison.

Long-run Phase J5/J6 entry point:

- `production/validate_gpu.sh --full` long-form heavy-run execution mode for saturation specs.
