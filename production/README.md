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
| `exp_pcf_mri_vp_insulating` | pcf | mri | `examples/pcf_mhd_mri_shearpy_jax.py` | CPU-anchored candidate: `B=B0+curl(A)` with exact per-mode vacuum matching (true insulating walls); DNS growth matches the insulating linear eigensolver to `3.3e-8` and `div B` holds roundoff for the whole horizon; no GPU artifact yet | rung 1/2/3 |
| `exp_tc_mri_vector_potential` | taylor_couette | mri | `examples/taylor_couette_vp_jax.py` | CPU-anchored candidate: full 3D `(theta,z,r)` `B=B0 e_z+curl(A)` with exact resistive-conducting cylinders; DNS growth matches the conducting eigensolver at `m=0` (`1.5e-9`) and `m=1` (`9.7e-8`), nonlinear trajectories match the primitive solver to `~1e-10`, and `div B` holds its resolution floor; no GPU artifact yet | rung 1/2/3 |
| `exp_tc_mri_vp_insulating` | taylor_couette | mri | `examples/taylor_couette_vp_jax.py` | CPU-anchored candidate: full 3D insulating cylinders via per-mode Bessel-ratio vacuum matching (all `(m,kz)`, not just axisymmetric); `m=0` growth matches the flux-function eigensolver to `1.5e-7`; no GPU artifact yet | rung 1/2/3 |
| `tc_supercritical_saturation` | taylor_couette | hydro | `examples/taylor_couette_dns_jax.py` | qualified candidate: runner, smoke coverage, and a legacy full-resolution GPU saturation artifact exist; current TC diagnostic-contract and release gates remain | rung 2/3 |
| `tc_mri_nonlinear_saturation` | taylor_couette | mri | `examples/taylor_couette_dns_jax.py` | finite-divergence only: the direct-`b` legacy full run saturated but ended at `div B=7.96e-4`; below the generic health ceiling, not roundoff-solenoidal or campaign-ready. Note this runner is the **axisymmetric** primitive class; the full-3D primitive class exists but is not production-wired | rung 1/2/3 |
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
vector-potential form `B = B0 + curl(A)` makes `div B = div(curl A) = 0` an
identity, so `div B` stays at its roundoff/resolution floor for the whole
horizon and does not grow. This now covers **both geometries and both wall
types**: plane Couette (`examples/pcf_mhd_mri_shearpy_jax.py`, conducting and
insulating walls) and Taylor-Couette (`examples/taylor_couette_vp_jax.py`,
conducting and insulating cylinders, full 3D `(theta,z,r)`). Every
primitive/direct-`b` path (`examples/pcf_mri_primitive_jax.py`,
`examples/taylor_couette_dns_jax.py`) evolves the field components directly and
does **not** enforce the solenoidal constraint; its `div B` stays small only
while amplitudes are low and grows into the finite regime at finite MRI
amplitude.

Wall-condition conventions of the vector-potential family:

- PCF conducting: `A = 0` on the walls (`b_x = 0` exact; the tangential
  electric field carries an `O(eta)` gauge residual — the convention of the
  shenfun reference `couette/pcf_mhd_divfree.py`).
- PCF insulating: exact per-mode vacuum matching
  `b_x' = -+k b_x`, `b_y = -+(i ky/k) b_x`, `b_z = -+(i kz/k) b_x` imposed as
  per-mode tau rows (Neumann/Robin split of the tangential potentials).
- TC conducting: `A_theta = A_z = 0` plus `(r A_r)' = 0`, i.e. `div A = 0` at
  the walls, which makes `E_tang = 0` **exact** — on-shell equivalent to the
  primitive set `{b_r=0, (r b_theta)'=0, b_z'=0}`.
- TC insulating: per-mode matching to `I_m(|kz| r)` / `K_m(|kz| r)` exterior
  potentials (`r^{+-|m|}` at `kz=0`); the `(0,0)` mean mode uses
  `b_theta = 0` at both walls, `b_z(R2) = 0`, and the exact trapped-flux
  Faraday row `(R1/2) db_z(R1)/dt = eta db_z/dr(R1)`.

| Case and evidence | Magnetic representation | Preserves `div B=0`? | `div u` (L2) | `div B` (L2) | Qualification |
|---|---|---|---:|---:|---|
| PCF plain MHD, decaying GPU smoke (`pcf_mhd_divfree`) | primitive/direct `b` | no (small while decaying) | `1.05e-8` | `7.15e-6` | failed candidate; decayed, not a saturation reference |
| PCF growing MRI, GPU smoke at `t=30` (`exp_pcf_mri_shearbox_growth`) | primitive/direct `b` | no — grows past the `1e-2` guard | not recorded at guard exit | `1.34e-2` | guard failure; primitive MRI is not production-ready |
| PCF growing MRI, older full artifact (`exp_pcf_mri_shearbox_growth`) | primitive/direct `b` | no — grew to `2.67e-2` | `2.30e-2` | `2.67e-2` | quarantined and forbidden from production seeding |
| PCF MRI, 300-step CPU qualification + GPU smoke (`exp_pcf_mri_vector_potential`) | vector potential, `B=B0+curl(A)`, conducting | yes — by construction, non-growing | `~1e-16` (CPU) | GPU `~1e-18`; CPU `~2.5e-16` (max over 300 steps) | selected conducting-wall workhorse; gated `< 1e-12` over the whole horizon; full-resolution GPU golden pending |
| PCF MRI insulating, 300-step finite-amplitude CPU run + eigenmode anchor (`exp_pcf_mri_vp_insulating`) | vector potential, `B=B0+curl(A)`, vacuum-matched | yes — by construction, non-growing | `~1e-16` | max `9.9e-17` over 300 steps; matching-row residual `<=1.7e-16`; DNS growth matches the insulating eigensolver to `3.3e-8` | CPU-anchored candidate; gated `< 1e-12` whole-horizon; no GPU artifact yet |
| TC MRI conducting, CPU anchors + finite-amplitude horizon (`exp_tc_mri_vector_potential`) | vector potential, `B=B0 e_z+curl(A)`, `E_tang=0` exact | yes — by construction, non-growing | projected witness: smoke `<1e-9`; start/production `<1e-12`, enforced every block | `m=0`: max `~1e-19`; `m=1` (3D): `~4e-15` at `Nr=40` (spectrally convergent projection floor); finite amplitude (`3e-2`): `~1.5e-15` while the primitive solver is already at `~7e-8`; the start-tier `m=1` eigenmode scaled to `1e-1` measures `7.35e-13` | CPU-anchored candidate; growth matches the linear eigensolver at `m=0` (`1.5e-9`) and `m=1` (`9.7e-8`); nonlinear parity vs primitive `~1e-10`; no GPU artifact yet |
| TC MRI insulating, CPU anchors (`exp_tc_mri_vp_insulating`) | vector potential, Bessel vacuum matching | yes — by construction, non-growing | projected witness: smoke `<1e-9`; start/production `<1e-12`, enforced every block | max `1.2e-19` (`m=0`, 400 steps); matching-row residual `<=4.5e-22`; growth matches the flux eigensolver to `1.5e-7` | CPU-anchored candidate; no GPU artifact yet |
| Taylor-Couette conducting MHD/MRI, legacy full artifact (`tc_mri_nonlinear_saturation`) | primitive/direct `b` | no — grew to `7.96e-4` | `2.26e-5` | `7.96e-4` | finite-divergence only; saturated but not roundoff-solenoidal |

Every `div u`/`div B` cell above is a **measured** emitted L2 diagnostic (not a
gate) from different resolutions, horizons, and solver families; the gate
thresholds live in the Qualification column and in the tests
(`tests/production/test_vector_potential_oracle.py`,
`tests/production/test_workhorse_qualification.py`,
`tests/production/test_vp_insulating_oracle.py`, and
`tests/production/test_tc_vector_potential_oracle.py` gate the curl family at
`div B < 1e-12` across their stated resolutions and amplitudes). They are
qualification flags, not a cross-geometry convergence comparison. The PCF GPU evidence is recorded in
[`promotions/fj03_gpu_smoke_findings.md`](promotions/fj03_gpu_smoke_findings.md);
the full-artifact values come from the committed golden diagnostics.

One honest measurement nuance for Taylor-Couette: the reported TC `div B`
witness is the divergence of the forward-projected coefficient representation
of `b = curl(A)` (the representation the current density and diagnostics use),
so it carries the spectrally convergent quadrature error of the cylindrical
`1/r` projections. It is both resolution- and amplitude-dependent (`m=1`:
`1.8e-12` at `Nr=24` vs `3.7e-15` at `Nr=40` for the same seed scale) but
must remain at the corresponding projection floor instead of growing
secularly. The shipped TC specs therefore state tiered absolute ceilings
(`1e-9` for the deliberately coarse smoke tier and `1e-12` for start and
production); the runner enforces the selected ceiling every health block and
emits it as `divergence_b_guard_l2`. A start-tier `m=1` eigenmode amplified to
a saturation-scale `0.1` is an explicit regression case. The underlying
pointwise `curl A` field is solenoidal to roundoff identically. The slab (PCF)
witness has no `1/r` factors and sits at `~1e-16` independent of resolution.

### Adaptive CFL stepping

Both vector-potential runners accept an optional `time.adaptive_cfl` block
(`true` or `{}` for defaults, or `{"target", "safety", "dt_min", "dt_max",
"check_every", "growth_cap", "grow_when_below"}`). The horizon is a **time**
target (the elapsed time the fixed-`dt` run would cover, i.e. the spec
`final_time` when no step override is given): `dt` changes alter the step
count and the endpoint schedule is adjusted so the run lands exactly on the
requested time — a grown `dt` cannot overshoot the saturation window and a
shrunk one cannot end early. The CFL is measured on the state **before**
every compiled block (a pre-flight check covers the initial state), so an
unsafe starting `dt` is shrunk before any stepping instead of tripping the
production health gate. A block whose evolved state exceeds the hard
`CFL=1` ceiling aborts immediately: shrinking the next block cannot repair
statistics already produced in an unstable block. Safe values above the
target still shrink before the next solve. On a change the implicit
factorizations are rebuilt at the new `dt` (`solver.set_dt`), with the CNAB2
family restarting its IMEX-Euler bootstrap so no stale multistep history is
extrapolated. Elapsed time is accumulated exactly, every `dt` change is
recorded (scalars `n_dt_changes`, controller `dt_final`, actual
`dt_last_used`, `dt_min_used`, `dt_max_used`,
`adaptive_steps_taken`, `adaptive_final_step_clipped`,
`cfl_total_max_observed`; per-row `dt` and `cfl_total` in the time series).
The endpoint step is never allowed below `dt_min`: a tiny remainder is
redistributed over the final steps (subject to `dt_max` and projected CFL),
and an impossible sub-floor horizon is rejected before solving. `dt_final`
is restored to the controller value after an endpoint adjustment, while
`dt_last_used` records the actual final step.
The solenoidal gates run every block. Adaptive runs are currently wired
for fresh starts (no resume/quench/checkpoint-bank) and write a final
checkpoint only. Passing `checkpoint_every` or `snapshot_every` is rejected
explicitly rather than silently dropping output; tests:
`tests/production/test_adaptive_cfl.py`. Fixed-`dt`
remains the default and the committed goldens' semantics.

### Magnetic wall-condition menu (what exists, what does not)

- `conducting`: primitive families (PCF, TC) and vector-potential families
  (PCF `A=0` convention; TC exact `E_tang=0`).
- `insulating` (true vacuum matching): vector-potential families only, both
  geometries; the primitive families have no insulating implementation. The
  TC linear insulating eigensolver remains `m=0` only.
- `pseudo_vacuum` (`b_tang=0`, the cheap insulating surrogate): primitive PCF
  family only (`exp_pcf_mri_pseudo_vacuum`), not solenoidal-preserving; a
  vector-potential pseudo-vacuum variant is not implemented. With true
  insulating walls now available in the solenoidal-preserving family,
  pseudo-vacuum is useful mainly for literature comparison.
- Stress-free velocity walls: not implemented in any family (all wall-bounded
  solvers are no-slip); listed here so the absence is explicit rather than
  implied.

## Support matrix

| Geometry | Physics path | Support state | Internal formulation | jaxfun source files | Axis mapping | Boundary/sign conventions | Golden / fallback | Divergence keys | Tests |
|---|---|---|---|---|---|---|---|---|---|
| pcf | hydro | qualified_candidate | cheap and linear-window DNS parity, wired nonlinear runner, smoke coverage, and a legacy full GPU saturation artifact; regenerate under the current contract before release | `examples/pcf_fluctuations_jax.py`, `examples/pcf_linear_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip moving walls, `U_b=U_wall*x e_y` | `pcf_hydro_laminar_v1`; DNS `pcf_hydro_primitive_dns_v1`; heavy `pcf_fluct_re400` | `divergence_l2`; DNS `divergence_u` | cheap/DNS parity, smoke, and legacy saturation regressions |
| channel | hydro | production_ready_limited_scope | driven KMM pressure-gradient steady state with golden-normalized Poiseuille observables | `examples/channelflow_kmm.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip walls, pressure-gradient drive | `channel_poiseuille_hydro_v1` | `divergence_l2` | KMM steady-profile regression and CLI golden comparison |
| pcf | plain mhd, primitive `b` | finite_divergence_only | direct magnetic components; cheap linear parity and low-amplitude/decaying nonlinear execution work, but the full candidate decayed and was not promoted | `examples/pcf_mhd_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting walls; `B0=0.05`; `Omega=0` | `pcf_mhd_conducting_v1`; failed `pcf_mhd_divfree` candidate | `divergence_u_l2`, `divergence_b_l2` | finite-divergence smoke and failed-candidate regression |
| pcf | mri, primitive `b` | quarantined | direct magnetic components do not preserve the solenoidal constraint at finite MRI amplitude; linear and short DNS parity remain diagnostic-only | `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting or pseudo-vacuum walls, Coriolis/shear, imposed `B0 e_z` | cheap `pcf_mri_shearbox_v1`; DNS `pcf_mri_primitive_dns_v1`; quarantined `exp_pcf_mri_shearbox_growth` | `divergence_u_l2`, `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | quarantine and divergence-guard regressions |
| pcf | mri, vector potential | selected_workhorse_pending_full_run | nonlinear `B=B0+curl(A)` workhorse; current 300-step CPU qualification and GPU start-tier smoke pass, with checkpoint/resume/quench support and optional adaptive-CFL stepping; full production golden pending | `examples/pcf_mhd_mri_shearpy_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting (`A=0`) or insulating (exact per-mode vacuum matching) walls, Coriolis/shear, imposed `B0 e_z` | fallback rungs 1/2/3; full curl golden not yet committed | `divergence_u_l2`, `divergence_b_l2`, insulating adds `insulating_bc_residual` | vector-potential oracle, insulating oracle + linear-anchor tests, long nonlinear workhorse qualification |
| taylor_couette | mhd/mri, vector potential | CPU-anchored candidate | full 3D `(theta,z,r)` nonlinear `B=B0 e_z+curl(A)` DNS (CNAB2, per-mode coupled blocks); conducting rows are the exact resistive perfect-conductor set (`E_tang=0`); insulating rows are per-mode Bessel vacuum matching for **all** `(m,kz)`; checkpoint/resume and optional adaptive-CFL stepping wired; no GPU artifact yet | `examples/taylor_couette_vp_jax.py`, `production/oracles.py` | `axis_0=theta`, `axis_1=z`, `axis_2=r` (native); canonical `x=r`, `y=theta`, `z=z` | rotating no-slip cylinders; conducting `{A_theta=A_z=0, (r A_r)'=0}` or insulating Bessel matching; imposed `B0 e_z` | fallback rungs 1/2/3 via `exp_tc_mri_vector_potential`, `exp_tc_mri_vp_insulating`; no committed golden | `divergence_u_l2`, `divergence_b_l2`, insulating adds `insulating_bc_residual` | TC vector-potential oracle: eigenvalue anchors (`m=0`, `m=1`, insulating `m=0`), primitive-parity, whole-horizon solenoidal gates |
| taylor_couette | hydro | qualified_candidate | cheap and linear-window DNS parity, wired nonlinear runner, smoke coverage, and a legacy full GPU saturation artifact; current TC diagnostic contract remains open | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_linear_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | rotating no-slip cylinders, `V=A*r+B/r` | `taylor_couette_hydro_v1`; DNS `taylor_couette_hydro_dns_v1`; heavy `tc_supercritical_saturation` | `divergence_l2`; DNS `divergence_linf` | cheap/DNS parity, smoke, and legacy saturation regressions |
| taylor_couette | mhd/mri conducting, primitive `b` | finite_divergence_only | direct magnetic components; cheap and linear-window DNS parity plus a legacy full GPU saturation artifact, but not roundoff-solenoidal and not on the current TC diagnostic contract | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | conducting walls, Alfven units, `Pm=Rm/Re` | `taylor_couette_mhd_conducting_v1`; DNS `taylor_couette_mhd_dns_v1`; legacy `tc_mri_nonlinear_saturation` | `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | cheap/DNS parity, smoke, and legacy saturation regressions |
| taylor_couette | mhd insulating (linear/flux path) | production_ready_limited_scope (linear) | the axisymmetric flux-function insulating **eigensolver** remains `m=0, kz!=0` only and anchors the nonlinear vector-potential insulating DNS above (which itself has no such mode restriction) | `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | insulating eigensolver only for `m=0`, `kz!=0` | `taylor_couette_mhd_insulating_v1` | `divergence_b_l2` | linear golden; loader rejection for `m!=0` applies to the non-VP path only |
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
