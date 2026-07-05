# Production-Readiness Review — Couette/Taylor-Couette solver stack

**Date:** 2026-07-04 (branch `review-validation-hardening`, HEAD `f44ea6d`)
**Scope:** the plane-Couette (PCF) and Taylor-Couette (TC) hydro/MHD solver stack and everything a production campaign would rely on: `couette/` (vendored shenfun reference implementations), `examples/*_jax.py` (the jaxfun ports), `src/jaxfun/` (framework), `production/` (runner, goldens, gates), `tests/`.
**Intended production use cases reviewed against:** (1) direct-adjoint-looping (DAL) / minimal-seed search, (2) ROM training-data generation at scale, (3) comparisons with weakly/strongly nonlinear theory.

> **Layout note.** `couette/` contains the *shenfun* reference implementations (every file there imports `shenfun`, none import `jaxfun`). The jaxfun implementations you will actually run live in `examples/` (`pcf_fluctuations_jax.py`, `pcf_mhd_jax.py`, `pcf_mri_primitive_jax.py`, `taylor_couette_dns_jax.py`, …) driven by `production/run_problem.py`. The `couette/` scripts are the parity anchors; bugs there poison validation, so both layers were reviewed.

---

## 0. Implementation status update (2026-07-05)

This branch now implements the review's non-long-run fixes. The original
sections below are kept as the audit record; this section is the current status.
No intentionally long saturation campaigns were rerun as part of the fix pass.

### 0.1 Follow-up review disposition (2026-07-05)

The follow-up code review correctly identified one blocker in the F-13 hardening:
the saturation oracles built stationarity collectors but did not pass them into
the cadenced solve helper. That made full generated-saturation runs unsatisfiable
because the stationarity series collapsed to first/last rows only. This is now
fixed for the four saturation oracle paths, and a tiny full-scope regression
exercises the pass branch without `--steps` or a reduced resolution tier.

The same pass also fixes the adjacent production hazards that would have affected
the next GPU campaign: stationarity means/sample counts now get nonzero golden
tolerances; production checkpoints are compact latest-step HDF5 files written via
temp-file atomic replace rather than copy-appending an ever-growing file;
in-process device capture labels the live dtype unless the CLI explicitly applies
the production dtype policy, and checkpoint dtype metadata falls back to field
dtypes; snapshot writes now use atomic per-step HDF5 shards plus an
external-link index and pass a tensor-product function space so XDMF/HDF5 meshes
carry physical coordinates; the no-cadence fast solve path is restored;
the TC-hydro resumed time-series seed row no longer mixes seed energy with
resumed-state velocity; and the x64 guard reports worker-side dtype pollution
under xdist.

Still open from that review: `pcf_mhd_divfree` still needs the deferred long
float64 GPU campaign after these infrastructure fixes. Snapshot atomicity,
compilation-cache restoration, duplicate primitive-PCF reference helpers,
redundant cadence diagnostics, and divergence-key prefix matching are now fixed.

| Finding | Current status |
|---|---|
| F-1 dtype/test pollution | Fixed. Production dtype helpers are side-effect-free by default; only CLI entrypoints apply the process dtype policy. In-process capture now labels the live dtype, checkpoint metadata falls back to field dtypes, and `tests/conftest.py` guards x64 state including xdist workers. |
| F-2 live shenfun TC parity | Fixed. The parity harness masks the documented TC Nyquist convention and accounts for the axisymmetric MRI amplitude convention. Full live parity now passes. |
| F-2b primitive PCF finite-amplitude guard | Fixed. Axisymmetric and 3D primitive PCF finite-amplitude coefficient parity tests were added against live shenfun. |
| F-2c validation entrypoint semantics | Fixed. Default heavy/all validation prints a `SMOKE ONLY` banner, smoke rows remain scoped as `bounded_saturation_smoke`, generated goldens carry run provenance, and `parity-saturation` compares promoted saturation goldens. |
| F-3 checkpoint resume | Fixed. `run_problem.py --resume` reloads the latest checkpoint, validates spec/dtype metadata, reconstructs solver state, continues `tstep`, and appends diagnostics. |
| F-4 adjoint memory/remat | Fixed for the supported solve paths. KMM and CNAB2 solves use `scan_steps` with single-device `jax.checkpoint`; multi-device paths avoid remat where it conflicts with sharded execution. |
| F-5 MHD/TC differentiability | Fixed. `MHDState` is a pytree, CNAB2 `have_old` is a real leaf, and differentiability tests now cover MHD, primitive PCF, and TC states. |
| F-6 snapshots | Fixed for geometry and crash-safety. `--snapshot-every` writes atomic per-step HDF5 shards, rebuilds `snapshots.h5` as an external-link index via atomic replace, and emits XDMF plus a manifest with function-space mesh metadata for physical coordinates. |
| F-7 dtype policy | Fixed for the reviewed production contracts. CLI production remains float32 by default, objectives require x64, parity uses float64, and a dealiased primitive PCF short-window float32-vs-float64 regression test was added. |
| F-8 throughput visibility | Mitigated. KMM eager-loop dispatch was removed, no-cadence solves use the fast `solver.solve` path again, cadenced diagnostics are reused for divergence checks, persistent compilation cache is configured per run and restored after importable `run_problem()` calls, and metadata records `solver_steps`, `ms_per_step`, and `steps_per_second`. Further primitive-solver kernel optimization remains performance work, not a correctness gate. |
| F-9 minimal-seed loop | Fixed for the hydro-KMM DAL path. `minimal_seed_ascent` adds projected ascent, retraction to fixed energy, backtracking, and history. |
| F-10 PCF-MHD saturation | Not rerun; long simulation skipped as requested. `pcf_mhd_divfree` remains documented as a retained failed generated-saturation candidate. A longer/higher-resolution GPU campaign is still required to decide physics versus horizon/resolution. |
| F-11 production-envelope validation | Improved. Primitive PCF now has a dealiased live parity case, a float32-vs-float64 short-window check, and full live parity passes locally against shenfun 4.2.2. |
| F-12 parameter sensitivity | Fixed by contract. Production objectives require x64 and expose `finite_difference_parameter_sensitivity`; solver parameters remain static and should be changed by rebuilding solvers. |
| F-13 saturation gate holes | Fixed after follow-up. Full generated saturation requires finite numeric diagnostics, boolean `saturation_check_passed`, and a windowed stationarity check; the stationarity collector is wired through cadenced diagnostics and covered by a full-scope pass-branch regression. |
| F-14 mid-run NaN blindness | Fixed. Production cadence monitoring checks state finiteness and divergence drift during long runs. |
| F-15 primitive-MHD divergence drift | Mitigated. Divergence diagnostics remain gated and a mid-run divergence guard now fails early on drift; no long-run projection campaign was run. |
| F-16 checkpoint atomicity | Fixed and bounded. Production checkpoints now rewrite a compact latest-step HDF5 file through a temp file followed by atomic replace, avoiding the prior copy-append O(N^2) growth path. |
| F-17 secondary integrity items | Fixed or documented: explicit goldens validate fully, tolerance models are checked and hashed, reports can take exact metadata paths, same-backend device comparisons fail unless allowed, implemented-spec allowlists/schema validation are enforced, `saturation_check_passed` is type-checked, shearpy defaults match the reference, CI pins shenfun 4.2.2, and PCF SPMD parity was added. |

Current verification from this fix pass:

| Tier | Command | Result |
|---|---|---|
| Production suite | `.venv/bin/python -m pytest -q tests/production` | 137 passed |
| Live shenfun parity | `.venv/bin/python -m pytest -q -m "integration and live_shenfun" tests/couette/test_live_shenfun_parity.py` | 24 passed |
| Couette focused solver checks | `.venv/bin/python -m pytest -q tests/couette/test_pcf_mri_primitive_jax.py tests/couette/test_differentiability_jax.py tests/couette/test_pcf_mhd_jax.py tests/couette/test_pcf_mhd_mri_shearpy_jax.py` | 23 passed |
| SPMD parity | `.venv/bin/python -m pytest -q -m spmd tests/couette/test_sharding_parity_jax.py --num-devices=2` | 14 passed |
| Syntax/format checks | `py_compile` on edited Python files, `bash -n production/validate_gpu.sh`, `git diff --check` | passed |

## 1. Executive summary

The **physics is clean everywhere it was checked** — a line-by-line audit of the shenfun reference layer found zero operator bugs (independently corroborated by the separate eigensolver implementations), and term-by-term comparison of every jaxfun port (PCF KMM/MHD/shearbox/primitive, TC hydro/MRI, axisym+3D) found no wrong sign, no missing metric term, no dealiasing inconsistency. The production runner has honest validation-scope labeling with real fail-closed gates, and the June review's DAL projection bug is correctly fixed.

**It is not yet ready for unattended production campaigns.** Empirically, this review found: the live-shenfun parity tier currently red (4 Taylor-Couette tests — convention regressions, not physics); a test-suite global-state pollution bug that can silently evaporate coverage; and four structural gaps that bear directly on the stated use cases — no checkpoint-resume wiring, no rematerialization for adjoint memory (measured DAL horizon cap ≈ 20 steps at production resolution on the 24 GB GPU), `jax.grad` failing outright on every MHD/TC solver state, and a full-run gate hole that lets NaN theory-observables into passing saturation goldens. Everything below is ranked and comes with a file:line and a fix.

**Verdict by use case**

| Use case | Status | Blocking items |
|---|---|---|
| DAL / minimal seed (PCF hydro, KMM path) | Scaffold correct, campaign-infeasible at production scale | F-4 (adjoint memory), F-9 (no optimizer loop); F-1 hygiene |
| DAL / minimal seed (MHD, MRI, TC) | Blocked | F-5 (no MHD/TC solver state is grad-compatible), then F-4 |
| ROM data generation | Feasible with a small custom driver; not turnkey | F-3 (no resume), F-6 (no snapshot wiring in runner), F-7 (float32 policy), F-8 (KMM throughput) |
| Nonlinear-theory comparison | Linear layer solid; finite-amplitude TC validation currently red; primitive PCF unguarded; theory observables can silently null | F-2 (parity regressions), F-2b (primitive PCF has no finite-amp parity), F-13 (NaN observables pass full-run gates), F-10 (pcf_mhd_divfree unresolved) |

---

## 2. Empirical validation status (all run during this review, RTX 5090 laptop GPU, jax 0.10.1)

| Tier | Command | Result |
|---|---|---|
| Default suite | `uv run pytest -q` | **1 failed**, 1006 passed — the failure is real (F-1) |
| Live shenfun parity | `-m integration` with `SHENFUN_PYTHON` (shenfun 4.2.2 env) | **4 failed**, 328 passed — all four TC (F-2) |
| Committed-golden parity | `make -C production parity-cheap` / `parity-dns` | 9/9 and 13/13 passed (float64, rtol 1e-10 tier) |
| SPMD | `-m spmd --num-devices=2` | 54 passed |
| Production smoke run | `run_problem.py --config production/runs/tc_supercritical_saturation.json --resolution-tier smoke --steps 4 --checkpoint-every 2` | Works end-to-end; metadata honestly labeled `bounded_saturation_smoke`; checkpoint reads back via `jaxfun.io.read_checkpoint` |
| DAL gradient probe | `gain_and_projected_gradient` on KMM at N=(33,32,32), float64 | steps 5/20/50 → peak GPU memory 0.39/1.31/3.17 GB (linear ≈ 0.06 GB/step) |
| DAL gradient probe, CNAB2 family | `jax.grad` through `PCFMRIDNSJax.solve` | **TypeError**: state pytree contains a bool leaf (F-5) |
| DAL gradient probe, MHD family | `jax.grad` through `PlaneCouetteMHDJax.solve` | **TypeError**: `MHDState` is not a registered pytree (F-5) |
| Forward throughput, KMM start tier | `PlaneCouetteFluctuationJax` N=(32,64,32), float64 | construct 7.4 s + first-step 9.0 s, then 48 ms/step (eager loop) |
| Forward throughput, KMM production tier | same solver at N=(64,128,64) | construction + first 100 steps **did not complete in ~70 min** (killed); time-to-first-step is the bottleneck (F-8) |
| Forward throughput, primitive 3D | `PCFMRIDNSJax` N=(32,32,32), dealias (1.5,1.5,1), float64 | construct 8.5 s + scan compile 3.8 s, then **2179 ms/step** over 500 steps (F-8) |

The five promoted saturation goldens were generated in **float32** (`production_run_dtype: float32`); their `saturation_checks.passed = True` gates are genuinely enforced (verified in `run_problem.py:144-153, 473-495` — failure writes `status: failed` metadata and raises). The `pcf_mhd_divfree` saturation candidate decayed and was correctly *not* promoted (F-10).

---

## 3. Findings (severity-ranked)

### F-1 · HIGH · Test-suite global-state pollution: production dtype helpers flip x64 off mid-suite
`production/device.py:14-35` (`configure_production_dtype`) defaults to float32 and then **mutates the calling process**: sets `JAXFUN_ENABLE_X64=0` / `JAX_ENABLE_X64=0` in `os.environ` and `jax.config.update("jax_enable_x64", False)` via `capture_device_record` (`production/device.py:52-66`), which `run_problem.py:79` calls in-process. Reproduced deterministically: `pytest tests/production/test_run_problem.py tests/test_x64_default.py -n 0` → the x64 sentinel fails. Under the default parallel run (`-n logical`) this is scheduling-dependent; today it manifests as the 1 failure in the default suite.

*Why it matters beyond one red test:* dozens of couette tests are decorated `skipif(not x64)` (`tests/couette/test_taylor_couette_dns_jax.py:48,81,204,252,299`, `test_differentiability_jax.py`, `test_pcf_mhd_jax.py:30`, …). Any of them scheduled *after* a polluting production test in the same xdist worker will **silently skip** — the historically recurring failure mode of this repo (green suite, evaporated coverage). The same mechanism is a footgun for your own DAL/ROM driver scripts: importing and calling any production helper (`capture_device_record`, `production_run_env`) silently downgrades an already-configured float64 process to float32.

*Fix:* make `configure_production_dtype` side-effect-free by default (return the policy; only the `run_problem.py` CLI entrypoint applies it), or snapshot/restore env+config in the production tests (autouse fixture), and add a conftest guard asserting x64 unchanged at session end.

### F-2 · HIGH · Live shenfun parity tier is red: two convention regressions from commit `753650f` (2026-06-05)
Failing: `test_tc_3d_dns_matches_live_shenfun_diagnostics_and_coeffs`, `test_tc_axisymmetric_mri_dns_matches_live_shenfun_diagnostics_and_coeffs`, and both `*_finite_amplitude_coeffs_match_live_shenfun` MRI tests. Two independent root causes, both introduced by the "Fix Couette adjoint review issues" commit and both convention breaks against the vendored reference rather than wrong physics:

1. **Nyquist masking added to TC nonlinear products** (`examples/taylor_couette_dns_jax.py`, `TD.mask_nyquist(...)` around every nonlinear `scalar_product`) while `couette/taylor_couette_dns.py` contains zero `mask_nyquist` calls. At finite amplitude the reference populates the Nyquist mode (observed ≈ 1.8e-8 after 50 steps at amp 1e-3) where jaxfun now has exactly 0 → coefficient parity fails at the 1e-8 active floor. Masking the unpaired Nyquist mode in a full-complex layout is arguably *better* numerics; but it must be a documented, gated deviation, not a silent divergence from the parity contract.
2. **Axisymmetric/MRI eigenmode seeding halved** (`taylor_couette_dns_jax.py`: `arr.at[kpos].set(0.5*block)` / `0.5*conj(block)` for ± pairs). This made jaxfun internally consistent with its 3D class, but the shenfun reference seeder kept the old convention: live energies now differ by exactly (1/2)² — reproduced: axisym MRI `Ekin` obtained `1.2428e-17` vs expected `4.9713e-17` at amp 1e-8. (The reference audit independently confirmed the ~2× axisym-vs-3D seed-amplitude convention still present in `couette/taylor_couette_dns.py` — harmless for growth rates, decisive for absolute-amplitude parity.)

*Why it matters:* the live tier is the sharpest gate you have (raw-coefficient, live reference); it has been red for a month while the committed-golden tier stayed green (its observables — growth rates, normalized profiles — are insensitive to both changes). Every TC MHD/MRI production claim currently rests on goldens only. Note CI's `full` job *does* run this tier against a checked-out shenfun on PRs to main (`.github/workflows/pytest.yml:94-109`), so the branch will fail CI there — the gate is doing its job; the finding is that the regression shipped and sat red for a month of local work.

*Fix (choose deliberately):* either (a) mirror both conventions into `couette/taylor_couette_dns.py` and re-vendor goldens, or (b) make the parity harness convention-neutral (zero Nyquist modes on both sides before comparing; pass the amplitude through one shared seeding protocol). Option (b) keeps the reference pristine. Either way, get this tier green again *before* long production runs, and add it to the pre-run checklist.

### F-2b · HIGH · The primitive PCF workhorse has no independent finite-amplitude validation at all
`examples/pcf_mri_primitive_jax.py` — the solver behind three of the five production specs (`pcf_mhd_divfree`, `exp_pcf_mri_shearbox_growth`, and the PCF-MHD/MRI support rows) — is **absent from `tests/couette/test_live_shenfun_parity.py`** (see its imports, lines 5–15). Its nonlinear terms are exercised only in the linear regime (growth tests seed amp≈1e-7, so quadratic terms sit at ~1e-14) plus a self-generated regression golden. A sign error introduced tomorrow in `(u·∇)u`, `(b·∇)b`, or `curl(u×b)` would pass every automated test. For contrast, both other families *do* carry finite-amplitude coefficient-level live parity: the KMM-family PCF solvers (hydro amp 0.05; MHD u/g/A at amp 0.05/0.05, 50 steps, rtol 1e-8; MRI-shearbox α/stresses) and the TC family (amp 1e-3/1e-4 — currently red for convention reasons, F-2). Term-by-term hand-verification against `couette/pcf_mri_primitive.py` found the current code correct — so this is a *guard* gap, not a bug — but for a production workhorse the guard is the point.
*Fix:* add a finite-amplitude (amp ~1e-3, ≥50-step) live-parity coefficient test for both the axisymmetric and 3D primitive PCF classes, mirroring the TC MRI ones.

### F-2c · HIGH · The one-command validation entrypoint is green-but-vacuous by default, and saturated physics has no regression tier
`production/validate_gpu.sh all|heavy` defaults every heavy spec to `--resolution-tier start --steps 2` (`validate_gpu.sh:160-173`): every row is labeled `bounded_saturation_smoke`, mapped to "skipped" by `report.py`, and the script exits 0. So `validate-all` green certifies only finiteness/divergence floors — real validation requires the non-obvious `--full`. Separately, **no target ever `--compare-golden`s the five generated saturation goldens** (`all`/`heavy` always `--write-golden`, regenerating rather than comparing; `cheap`/`dns` cover only the 13 vendored shenfun IDs) — the saturated-regime physics that your ROM datasets will sample has zero regression protection. The per-row labels are honest (verified); the *entrypoint semantics* and headline `failed: 0` are what mislead.
*Aggravators (verified):* the saturation goldens' tolerance model is exact-scalar (e.g. `kinetic_energy` tol 1e-6) — meaningful for steady saturations (axisym Taylor vortices are a fixed point) but scientifically meaningless for chaotic 3D states, where only a statistical band can gate; `golden.json` records no generation-mode provenance (no steps/tier/bounded_smoke fields in `_write_golden`, `run_problem.py:596-623`), so a 2-step smoke golden is byte-indistinguishable from a full one — promotion to `production/goldens/` is a manual `cp` enforced only by discipline (the failed `pcf_mhd_divfree` candidate's metadata was necessarily hand-authored, since the runner raises before writing goldens on failed saturation).
*Fix:* make `validate-all` print an explicit "SMOKE ONLY — nothing validated at production tier" banner (or require a `--smoke` flag for the bounded mode); add a `parity-saturation` target that `--compare-golden`s current code against the promoted saturation goldens with statistical tolerances; stamp `golden.json` with steps/tier provenance so smoke artifacts cannot masquerade.

### F-3 · HIGH · Restart is proven possible but not wired: no `--resume` in the runner
`production/checkpoint.py` writes a complete restart envelope (spec hash, dtype/shape metadata, device record, diagnostics pointer, PRNG slot, full CNAB2 history incl. `nonlinear_old`/`have_old`), and `tests/production/test_run_problem.py:868-945` *proves bit-exact continuation* (checkpoint at step 4 + 2 more steps == 6 direct steps, atol 1e-12) — by hand-reconstructing the state in the test. But no production code consumes checkpoints: `run_problem.py` has no `--resume`, and the state reconstruction is done ad hoc inside that one test rather than exposed per state-kind. Operationally, a multi-day saturation run or ROM sweep that dies at 90% restarts from zero unless you hand-roll the reconstruction the test demonstrates.
*Fix:* `run_problem.py --resume <run_dir>`: validate `spec_hash` + dtype metadata, rebuild the solver from the spec, reconstruct the state (promote the test's logic into `production/oracles.py` per state-kind), continue `tstep` and append diagnostics.

### F-4 · HIGH (DAL) · No rematerialization anywhere: adjoint memory is O(steps), capping horizons at ~tens of steps
There is no `jax.checkpoint`/`remat` (nor custom VJP for the linear solves) in `src/jaxfun/` or `examples/` (grep-verified). Reverse-mode through `solve` stores every step's residuals. Measured on the KMM solver at toy resolution (33×32×32, float64): 0.39 → 1.31 → 3.17 GB peak for 5 → 20 → 50 steps (≈0.06 GB/step). Scaling ~16× to the `pcf_fluct_re400` production grid (64×128×64) gives ≈1 GB/step → **the 24 GB GPU holds a horizon of roughly 20 steps (t ≈ 0.2 advective units at dt 0.01)**. Minimal-seed literature horizons (T ≈ 10–100) need hundreds to tens of thousands of steps.
*Additional aggravator:* `KMM.solve` (`examples/channelflow_kmm.py:518-522`) is a Python loop, so the autodiff graph *unrolls* — compile time also grows linearly in steps (observed: 20 s trace/compile for a 5-step gradient).
*Fix:* route KMM `solve` through `scan_steps` like the CNAB2 solvers; wrap the step in `jax.checkpoint` (per-step remat ≈ O(1) residual memory × 2 compute), and consider nested/binomial checkpointing for long horizons. This is the difference between "DAL demo" and "DAL campaign".

### F-5 · HIGH (DAL) · No MHD or TC solver is differentiable end-to-end today — only the hydro KMM path
Two independent mechanisms, both verified empirically this review:
1. **CNAB2 family (primitive PCF MRI, all TC solvers): boolean pytree leaf.** `AxisymmetricPCFState.have_old: bool` (`examples/pcf_mri_primitive_jax.py:61-75`) and both TC states (`examples/taylor_couette_dns_jax.py:95,117`) flatten `have_old` into the leaves (`jnp.asarray(self.have_old)` → bool array). `jax.grad` through `PCFMRIDNSJax.solve` raises `TypeError: grad requires real- or complex-valued inputs ... got bool`.
2. **KMM-MHD family (`pcf_mhd_jax.py`, and the MRI-shearbox solver built on it): state not a pytree at all.** `MHDState` (`examples/pcf_mhd_jax.py:31-36`) is a plain frozen dataclass with no `register_pytree_node_class` (contrast `KMMState`, `channelflow_kmm.py:38`); `jax.grad` w.r.t. it raises `TypeError: ... is not a valid JAX type`. The forward Python-loop `solve` never notices; anything needing flattening (grad, jit, scan, tree_map) does.
The differentiability test suite (`tests/couette/test_differentiability_jax.py`) only exercises the hydro KMM path, which is why both gaps are invisible to the suite.
Compounding detail: because `MHDState` is opaque to `tree_util`, the minimal-seed tree helpers (`tree_scale`, `tree_add_scaled`) also fail on it — MHD DAL is doubly broken until registration.
*Fix:* register `MHDState` as a pytree (one decorator + flatten/unflatten, mirroring `KMMState`); carry `have_old` as a **float 0/1 leaf** (it only feeds `jnp.where` in `ab2_extrapolate`, `src/jaxfun/integrators/cnab2.py:33-45`). Do *not* move `have_old` to static/aux metadata: it flips after the first step, and a treedef change mid-`lax.scan` is an error; it is also load-bearing in the checkpoint payload (`oracles.py:1315-1332`) and in the bit-exact restart test — a float leaf satisfies grad, scan, and checkpoint round-trip simultaneously. Then add differentiability tests for one CNAB2 solver and one MHD solver.
*Workaround until fixed:* differentiate w.r.t. the field coefficients only, closing over the flag — e.g. `jax.grad(lambda u: loss(solve(State(x=u, p=p0, nonlinear_old=n0, have_old=False), steps)))`. The step internals themselves are AD-clean (the AB2 bootstrap is a `jnp.where`, the per-mode LU solves are `lu_solve` calls, and `scan_steps` is `lax.scan` on a single device).

### F-6 · MEDIUM (ROM) · Snapshot pipeline exists but is not wired to the production runner
`jaxfun.io` has `Cadence(snapshot_every=…)`, `write_uniform_snapshot` (verified against shenfun's own `ShenfunFile` output in `tests/io/test_hdf5.py`), `generate_xdmf`, and every solver's `solve_with_cadence` accepts `on_snapshot` — but `production/oracles.py`/`run_problem.py` never pass it and the CLI has no `--snapshot-every`; `write_uniform_snapshot` is called only from tests. ROM training data (fields at regular intervals, physical space, mesh metadata) currently requires a custom driver per problem, and even then there is no compression/chunking, no float32 storage cast for float64 runs, no dataset manifest/loader, no per-snapshot physical-parameter attrs, and no SPMD-parallel write.
*Fix:* add `--snapshot-every K` symmetric to `--checkpoint-every`, writing `snapshots.h5` + XDMF next to the checkpoints with the run's spec/dtype attrs (chunked, optionally cast); a manifest JSON per run directory closes the loader gap.

### F-7 · MEDIUM · float32 production default deserves an explicit decision per use case
`JAXFUN_PRODUCTION_DTYPE` defaults to float32 (`production/device.py:17`), all five promoted saturation goldens are float32 artifacts, and committed-golden parity gates run float64 via `JAXFUN_VALIDATE_PARITY_DTYPE`. That split is coherent for throughput, but: (a) DAL gradients and objective Hessians in float32 through thousands of spectral steps will be noise-limited; (b) weakly-nonlinear comparisons (Landau coefficients, amplitude scalings near criticality) generally need float64; (c) any user driver that imports production helpers inherits float32 silently (F-1's mechanism). Nothing in the test suite exercises solver physics at float32 beyond smoke tiers, so float32-specific pathologies (e.g. Helmholtz/biharmonic solve conditioning at Nx=64) are unprobed.
*Fix:* keep float32 as the data-generation default if you like, but pin `float64` in the DAL/theory-comparison entrypoints, and add one float32-vs-float64 agreement test on a short DNS window.

### F-8 · MEDIUM (ROM/perf) · Throughput hazards in both solver families — measure before you budget GPU-months
Measured this review (float64, RTX 5090; production float32 should be up to ~2× better):
- **KMM family** (eager per-step Python loop, `channelflow_kmm.py:518`, `pcf_mhd_jax.py:260`): serviceable at start tier — N=(32,64,32): 7.4 s construct, 9.0 s first step, then **48 ms/step**. But at the `pcf_fluct_re400` production tier N=(64,128,64), **construction + the first 100 steps did not complete within ~70 minutes** (probe killed; host pegged dispatching small kernels). The promoted J5 golden proves it does complete, but time-to-first-step at production N (sympy operator assembly + per-mode solver builds) is re-paid on every restart, dtype change, or resolution change, and no persistent compilation cache is configured.
- **Primitive/CNAB2 family** (`lax.scan`, compiled once): compile is cheap (3.8 s) but the *step itself* is heavy — N=(32,32,32) with production dealiasing runs at **2179 ms/step** (500-step average). That is ~45× the KMM per-step cost at comparable size; the 12,000-step `exp_pcf_mri_shearbox_growth` spec implies ~7 GPU-hours in float64. The prime suspect is the per-mode dense saddle-point `lu_solve` over all (ky,kz) modes each step plus padded-transform round-trips — profile before scaling N.
*Fix:* `scan_steps` for KMM `solve` (F-4 fix kills both the unrolled-gradient and the eager-dispatch problems); profile the primitive step (batched LU layout, avoid re-forming dense blocks); enable `jax.config.jax_compilation_cache_dir`; and record `ms/step` in run metadata so throughput regressions are visible (the golden metadata currently stores no timing).

### F-9 · MEDIUM (DAL) · The minimal-seed scaffold is pieces, not a loop
`examples/pcf_minimal_seed_jax.py` provides exactly what its docstring promises — normalization, gain, tangent projection — and (verified this review) the June bug is fixed correctly: the projection now subtracts along `conj(g)` with denominator `Re⟨conj(g),conj(g)⟩ = Σ|g|²` (`pcf_minimal_seed_jax.py:112-123`), and the `Re(Σ g·d)` pairing matches JAX's `grad(|z|²)=2conj(z)` convention. What does not exist: the outer optimization loop (gradient ascent/rotation on the E₀ sphere with retraction — the projection gives a tangent step but nothing renormalizes after the update), line search/step control, horizon (T) and energy (E₀) continuation, convergence criteria, restart/checkpoint of the optimization state, and any test with a mixed-phase state at steps>0 beyond the small differentiability suite.
*Fix:* a ~150-line driver (project → step → renormalize-to-E₀ → track gain; bisection over E₀) plus F-4; recommend writing it against `production/objectives.minimal_seed_value_and_projected_gradient` so it inherits the tested path.
*Scope note:* the helpers hardcode the hydro-KMM interface (`solver.perturbation_energy`, `KMMState` tree helpers — `pcf_minimal_seed_jax.py:106,118`; `production/objectives.py:129`); on the MHD solver they fail on the missing `.u`, on the primitive solver on the missing `perturbation_energy`. Extending DAL beyond hydro needs a small energy-interface shim in addition to F-5.

### F-10 · MEDIUM (physics coverage) · PCF-MHD saturation target unresolved; one production spec is a placeholder
`production/README.md` records that the `pcf_mhd_divfree` full run (N=32×64×32, final_time 10) **decayed** and is retained as a failed candidate — so the PCF-MHD nonlinear saturation regime, one of your stated targets, has no validated golden. `stab_PCF_MRI_stability` is declared `config-undetermined placeholder / not executable`. Also note `final_time: 10.0` is short for PCF-MHD transition; decay vs. sustainment at these parameters may be physical or resolution/horizon-limited — currently indistinguishable.
*Fix:* treat as an open physics-validation task: rerun with larger N/longer horizon in float64 once F-3 (resume) exists; keep the failed-candidate labeling (it is honest and correct).

### F-11 · MEDIUM · The exact configuration you will run in production is not the configuration that is parity-validated
Two axes, same pattern (validated ≠ deployed):
- **Dealiasing:** every end-to-end TC parity test — including the finite-amplitude MHD nonlinear cases — runs `dealias=1.0`, while the production specs run `dealias=1.5` (TC) / `[1,1.5,1.5]` (PCF). The padded-transform round-trip is statically correct (the `T0p.forward` fallback on standard-size input reduces to the standard forward) and runs stably, but no live-parity case pins the padded path's coefficients against shenfun.
- **dtype:** all parity is float64; all promoted saturation goldens and the runner default are float32 (see F-7).
*Fix:* one finite-amplitude live-parity case at `dealias=1.5`, and one float32-vs-float64 short-window agreement test, then the validated envelope covers the production envelope.

### F-12 · MEDIUM (theory comparison) · Parameter gradients are structurally zero-capable: operator caches are baked at `__init__`
All solver families factorize their implicit operators once at construction (e.g. per-mode LU via `lu_factor` in the TC/primitive solvers; `Biharmonic`/`Helmholtz` solvers in KMM, `channelflow_kmm.py:141-236`) with `nu`, `dt`, `B0`, `Re` folded into concrete arrays. Gradients w.r.t. the *initial state* flow correctly (that path is probed), but gradients w.r.t. *physical parameters* cannot flow through the cached factorizations — differentiating an objective w.r.t. `Re` or `B0` will silently miss the implicit-operator dependence (worse than an error). This blocks AD-based Landau-coefficient extraction, critical-parameter continuation, and adjoint parameter-sensitivity studies — part of your weakly/strongly-nonlinear program.
*Fix:* either construct operators inside the traced function from parameter arrays (JAX-native assembly), or expose a documented "params are static — re-instantiate the solver per parameter value; do not `jax.grad` w.r.t. them" contract plus finite-difference helpers for parameter sensitivities.

### F-13 · HIGH · The saturation gate verifies transition, not saturation — and a *passing* full-run golden can carry NaN observables
Three holes in the same gate, all verified in code by the lead reviewer:
1. **Transition, not stationarity:** `_saturation_passed` (`production/oracles.py:19-35`) requires only `energy_growth_factor > threshold` (e.g. 1e3 for TC hydro, 2.0 for `pcf_fluct_re400`) plus finite, non-negative finals. A run still in transient growth, or decaying after a peak, passes identically. `generated_saturated_golden` certifies *instability triggered, amplified, stayed bounded* — not statistical stationarity.
2. **Only the energy tuple is finiteness-checked on full runs.** The oracle passes a specific `final_energies` tuple into the gate; `reynolds_stress`, `maxwell_stress`, `transport_alpha`, `growth_rate`, … are not in it. And the all-scalars nonfinite floor is `required` **only for smoke scopes** — `generated_saturated_golden` is absent from `_VALIDATION_FLOOR_SCOPES` (`run_problem.py:359-371`), so `validation_floor.passed = True` vacuously on full runs. A full saturation run whose `transport_alpha` went NaN (e.g. zero pressure normalization) still passes both gates, and `_json_ready` (`run_problem.py:626-627`) silently rewrites NaN→`null` in the written golden — green artifact, null theory observables. This directly poisons the nonlinear-theory-comparison use case.
3. **Gate coverage:** the pass-branch is never exercised end-to-end in pytest (in-suite saturation tests force `steps=2`, N≈8; gate helpers unit-tested on hand-built dicts only).
*Fix:* include `generated_saturated_golden` in the validation-floor scopes (all-scalars finiteness on full runs is cheap and strictly safer); add a windowed stationarity criterion (last-quartile mean vs previous quartile) with window statistics stored in the golden; add one small end-to-end pass-branch test. See F-2c for the absent regression tier over these goldens.

### F-14 · MEDIUM · Long production runs are NaN-blind until the end
The production execution path has no mid-run finiteness monitoring: `_solve_with_optional_checkpoints` (`production/oracles.py:1267-1307`) passes only `Cadence(checkpoint_every=…)` — no `diagnostics_every`, no `should_stop` — and without `--checkpoint-every` it is a single `solver.solve(state, steps)` call (for the CNAB2 solvers, one `lax.scan` with zero host visibility until it returns). A blow-up at step 100 of a 16,000-step MRI saturation run propagates NaNs on the GPU for the remaining 15,900 steps and is caught only by the final saturation/floor gate. The reference layer has the same blind spot on its hydro-PCF path (`couette/ChannelFlow.py:4` suppresses *all* warnings globally and `KMM.solve` never checks `isfinite`, unlike the MHD/primitive/TC reference solvers which do assert finiteness).
*Fix:* wire `should_stop=lambda t, ts, s: ~isfinite(E)` (or a cheap coefficient-max check) at a modest cadence into the production oracles; scope the reference's warnings filter.

### F-15 · MEDIUM (long MHD runs) · Primitive-variable MRI solvers never project div(b)
`examples/pcf_mri_primitive_jax.py` and the TC MRI solvers (mirroring their references) preserve div(b)=0 only analytically through the induction curl; it is monitored, never cleaned. Axisymmetric cases stay at roundoff (the source terms are exactly div-free), but the 3D saturation tests tolerate div(b) < 1e-3 over long runs — i.e. there *is* bounded drift, and nothing bounds it beyond test horizons. A months-of-GPU-time 3D MRI ROM dataset could accumulate div(b) past that, corrupting the Lorentz force late in the series. The vector-potential PCF solver (`pcf_mhd_jax.py`) is immune by construction.
*Fix:* keep `divergence_b` in the gated diagnostics for every production MHD run (it is already in the divergence keys); add a periodic projection (or a cleaning pass) for 3D runs beyond ~10⁴ steps, or prefer the vector-potential formulation where the physics allows.

### F-16 · LOW · Checkpoint writes are not atomic
`jaxfun.io.write_checkpoint` appends into one `checkpoints.h5` (`mode="a"`); a crash mid-write can corrupt the file including *earlier* steps (HDF5 has no journal). For long runs the checkpoint file is the recovery mechanism — it should not be the crash casualty.
*Fix:* write to `checkpoints.h5.tmp` copy-on-write or one file per step (`chk_<tstep>.h5`) + atomic rename; keep last N.

### F-17 · MEDIUM/LOW · Assorted, worth a line each
Gate-integrity secondary paths (all avoidable by staying on the default paths):
- `run_problem.py --shenfun-golden PATH` (`run_problem.py:579-587`) bypasses `validate_golden` (uses only the scalar-hash check): an empty-scalars golden with a matching empty-hash passes vacuously (`compare_goldens.py:187-201`). The default resolved-golden path is sound (rejects empty scalars, missing tolerances, hash/spec mismatches, NaN diffs).
- Tolerances are checked for *presence*, never for magnitude/finiteness (`compare_goldens.py:313-320`), and `scalar_hash` covers only the scalars — a golden's tolerances can be loosened post-hoc without breaking any hash. Consider hashing the tolerance model too.
- `report.py` globs whatever metadata exists (`report.py:148-152`): a run that dies before its first metadata write is simply absent (headline `failed: 0` can hide a crashed launch — only the shell exit code sees it), and the cumulative glob re-counts stale prior runs until `runs/` is cleaned. Bounded-smoke rows map to "skipped", so read the scope column, not the headline.
- `compare_devices.py` with default `--device-b auto` on a CUDA-less machine compares CPU to CPU — green while proving nothing about portability; its atol 1e-5/rtol 1e-6 sit close to float32 eps for grid-summed energies (expect cry-wolf failures on real GPU/CPU comparisons).
- `tests/production/test_objectives.py:49` skips the whole module when x64 is off — the only automated DAL-objective correctness check reports "skipped" under the float32 production default (compounds F-1's pollution).
- `problem_spec._reject_jaxfun_unimplemented` (`problem_spec.py:375-379`) is an empty no-op (dispatch fails later, message just worse); the committed JSON schema is never actually enforced by `validate_spec`; `validate_golden` passes `allow_unsupported=True`, so a golden for an unsupported subcase validates structurally.
- `_saturation_check_metadata` records `bool(raw_passed)` (`run_problem.py:353`) — `bool(float('nan'))` is `True`, so the gate trusts truthiness rather than type. Unreachable today (oracles emit real bools); one careless oracle edit away from slipping NaN through. Compare `is True` or assert the type.
- Energy diagnostics carry no ½ factor (`Epert`/`Etot` = 2× kinetic energy — deliberately matching the shenfun reference). Harmless internally, but calibrate before comparing amplitudes/energies against theory papers, and note eigenmode seeding scales the eigenvector norm, not a physical energy.

Solver/driver layer:
- `PCFMRIDNSJax(dealias=1.0)` default means an undealiased quadratic nonlinearity unless the spec overrides (production specs do set `[1,1.5,1.5]`; smoke tiers deliberately run undealiased). A safer default for a "production workhorse" class is 3/2 in the Fourier directions.
- `production/objectives.py:111-115` catches `AttributeError` from `maxwell_stress_objective` to detect hydro solvers — this also swallows genuine `AttributeError` bugs raised *inside* `fields_physical`. Prefer an explicit `has_magnetic` capability check.
- `production/objectives.py:235-264` stress objectives integrate with trapezoid weights on the *spectral* wall-normal axis instead of the solver's quadrature-exact `integrate()`. As an optimization objective any consistent functional is legitimate; but optimized Reynolds/Maxwell-stress *values* will be biased relative to solver diagnostics and theory — use the solver quadrature before comparing numbers. Related: `_taylor_couette_domain_weights` labels axis 1 `z` (it is θ for 3D states); numerically harmless for means, misleading for extensions.
- No PCF sharding-parity test exists (`tests/couette/test_sharding_parity_jax.py` covers TC only), and the multi-device branch of `scan_steps` (`cnab2.py:79`, eager Python loop) is unvalidated for replicated==sharded equivalence — stay single-device for PCF production until covered.
- Solver defaults drift from the references in two places (`pcf_mhd_mri_shearpy_jax`: `background_b z=0.1` vs reference `0.025`; `magnetic_amplitude=0.05` vs `0.0`) — parity tests pass explicit values so this only bites ad-hoc CLI comparisons.
- `dt` is baked into every cached factorization with no setter (see also F-12): dt-continuation or CFL-adaptive stepping requires solver re-instantiation today.
- `production/compare_devices.py` is exercised in tests only as cpu-vs-cpu determinism (`tests/production/test_compare_devices.py:83,116`); the actual GPU-vs-CPU / float32-vs-float64 agreement claim is unexercised by the suite — run it manually once per problem before trusting it.
- CI checks out `spectralDNS/shenfun` **unpinned** (`.github/workflows/pytest.yml:96-99`): the parity target drifts with upstream; pin a commit.
- The live-parity sentinel (`tests/conftest.py:87-107`) fail-hards correctly when parity tests are selected but zero references execute — but it guards *all-skip* only (one successful reference run satisfies the session) and is inert in default runs where parity is deselected. Green default `pytest` proves zero live parity by construction.
- `run_with_cadence` recompiles the scan for each distinct block length; with mixed cadences (`diagnostics_every=10`, `checkpoint_every=25`) block lengths vary (10,5,…) and each new length pays a compile. Choose commensurate cadences.
- (Meta: my own probe scripts piped pytest through `tail` and nearly mis-read exit codes twice — a reminder that anyone scripting around these suites should check `PIPESTATUS`, though `validate_gpu.sh` itself handles status correctly, see §6.4.)

---

## 4. What was verified correct (do not re-litigate)

- **DAL projection math** (`pcf_minimal_seed_jax.py`): fixed per the June review; denominator positive-definite, tangency exact (`Re Σ g·(d−λ·conj(g)) = 0` with `λ = ReΣg·d / Σ|g|²`). The JAX complex-gradient pairing without conjugate is correct — do not "fix" it.
- **KMM (0,0) mean-mode reality** is enforced (`channelflow_kmm.py:411-423, 464-465, 479-484`) — the June concern is resolved.
- **Eigenmode seeding determinism**: `_positive_pivot_phase` pins the eigenvector phase (reproducible seeds across BLAS/LAPACK builds).
- **Production gates fail closed** (verified by code path and by exercising the runner): solver exceptions write `status: failed` metadata then re-raise; missing/false saturation checks and validation-floor failures raise after metadata is written; failed golden comparisons raise. Validation-scope labels (`bounded_saturation_smoke` vs `generated_saturated_golden` vs `golden_comparison`) are honest and machine-readable.
- **Committed-golden parity green at float64:** 13/13, including both pipe hydro goldens and the four linear-window DNS goldens.
- **Checkpoint envelope**: spec-hash, dtype/shape, device record and diagnostics pointer all round-trip. The CNAB2 payload correctly includes the AB2 history (`nonlinear_old`, `have_old`) — restart is bit-exact (tested for TC). `KMMState` checkpoints only `{u, g}`, which is *complete by construction*: the IMEX-RK steppers are single-step methods with no cross-step history, so KMM resume needs no extra state (though only the TC restart has an automated fidelity test — add a KMM one when wiring F-3).
- **Golden provenance** is explicit (`production/goldens/PROVENANCE.json`: vendored from `fn_shenfun@dcaa42c`), and generated jaxfun goldens are distinguishable from vendored shenfun ones by directory + metadata.
- **IMEX-RK tableaux and staging** (`src/jaxfun/integrators/imex_rk.py`): IMEXRK222 is the standard ARS(2,2,2) (γ=(2−√2)/2, δ=1−1/(2γ)), IMEXRK443 the ARS(4,4,3) used by shenfun; the stage RHS accumulates `M u⁰ + dt Σ_j b[k,j] N_j + dt Σ_{m≥1} a[k,m] L u_m` with a verified-constant DIRK diagonal folded into the cached Helmholtz/Biharmonic solves — matches `shenfun/utilities/integrators.py`.
- **TC term-by-term physics fidelity** (sub-audit, spot-checked): axisym+3D hydro metric terms (`−u_θ²/r`, `+u_r u_θ/r`, `(u_θ/r)∂_θ`), EMF curl signs incl. the 3D `(1/r)∂_θ` and `−ε_θ/r` pieces, MHD linear operator (viscous `∇²−1/r²`, `∓2/r²·∂_θ` r–θ cross-coupling for u and b, Coriolis/field-stretch `rΩ′ b_r → b_θ`, `+B0∂_z`), and conducting-wall BCs (b_r Dirichlet, b_θ Robin with `R_wall/J`, b_z Neumann) all match `couette/taylor_couette_dns.py`.

## 5. Recommended order of work

Before any long production run:
1. **Re-green the live parity tier (F-2).** Decide the Nyquist and seeding conventions once, apply them on both sides of the contract (or make the harness convention-neutral), and record the decision in `couette/taylor_couette_notes.md`. Everything else stands on this gate.
2. **Fix the test pollution (F-1)** so the default suite's coverage is deterministic; add a session-end x64 guard.
2b. **Add the missing parity guards (F-2b/F-2c):** a finite-amplitude live-parity case for the primitive PCF family, and a `parity-saturation` compare target for the five promoted goldens.
3. **Wire `--resume` (F-3)**, make checkpoint writes atomic (F-16), and add a mid-run finiteness `should_stop` (F-14). Long runs are not production-safe without these.
3b. **Close the full-run gate holes (F-13):** add `generated_saturated_golden` to the validation-floor scopes (one-line change, all-scalars finiteness on full runs) and a windowed stationarity check.

To unlock the DAL campaign (in order):
4. Make all states grad-compatible (F-5: register `MHDState`; float-ify `have_old`) and add MHD/CNAB2 differentiability tests.
5. Convert `KMM.solve` to `scan_steps` and add `jax.checkpoint` around the step (F-4/F-8); re-run the memory probe to establish the feasible horizon (expected: hundreds–thousands of steps instead of ~20).
6. Write the minimal-seed outer loop (F-9) against `production/objectives.py`, in float64 (F-7).

To unlock ROM data generation:
7. Add `--snapshot-every` to the runner (F-6); decide float32 vs float64 per dataset deliberately (F-7); add a stationarity check before trusting "saturated" windows (F-13); keep `divergence_b` gated and plan a div(b) cleaning strategy for long 3D MHD series (F-15).

For the nonlinear-theory program:
8. Add one dealias=1.5 finite-amplitude live-parity case and a float32↔float64 agreement window (F-11); resolve the PCF-MHD saturation question at higher resolution/longer horizon (F-10); decide the parameter-sensitivity strategy (F-12).

---

## 6. Per-area audit summaries

### 6.1 `couette/` shenfun reference layer — **trustworthy as the parity anchor**
Line-by-line audit with first-principles re-derivation of every operator. **Zero physics or numerics bugs found.** Verified: the KMM velocity-vorticity system (wall-normal forcing `−(∂yy+∂zz)N₀ + ∂x∂y N₁ + ∂x∂z N₂` re-derived from curl-curl; exact `compute_vw` reconstruction; correct mean-pressure-gradient signs on the (0,0) modes); PCF fluctuation shear-production term present and correct; vector-potential MHD (`dA/dt = U×B + η∇²A`, correct J×B sign, div(B)=0 by construction); shearing-box Coriolis giving net `(S−2Ω)u_x`; the full cylindrical TC hydro + MRI operators (all metric terms, Ω-effect `rΩ′ = −2b/r²`, EMF curl with `−ε_θ/r`, resistive vector-Laplacian cross-couplings); conducting-wall Robin BC `c = r_wall/J` correctly derived; `CircularCouette` a,b identities (`2Ω+rΩ′=2a`) that feed every DNS operator. Crucially, the DNS operators are independently corroborated by the *separate* eigensolver reimplementations (`taylor_couette_linear/mri.assemble_parts`, `_pcf_linear`) which agree term-for-term — two independent derivations agreeing. Dealiasing is applied to all quadratic products (incl. MHD) when enabled.
**Reference-side caveats for long runs** (mirrored or inherited by the port, folded into F-14/F-15/F-17): global `warnings.filterwarnings('ignore')` + no NaN guard in `KMM.solve`; primitive-MRI div(b) monitored but never cleaned; `pcf_mri_primitive` dealias defaults OFF (TC defaults ON); a sign-flipped roll *visualization* diagnostic (`pcf_fluctuations_corrected.py:441`, viz-only); shearpy channel/parasite diagnostics valid only under serial/x-slab decomposition. The reference's own default test suite has essentially zero nonlinear-dynamics coverage of the PCF family (1–3-step runs, amp 1e-7 seeds); the TC family carries three always-run machine-precision operator-sign tests that directly pin the metric/EMF terms.
**Implication:** a parity failure is a port bug (or a deliberate convention change) — never ambiguity about the anchor.

### 6.2 Taylor-Couette jaxfun ports — **faithful port; gaps are coverage, not physics**
Term-by-term comparison of all three files against the reference found **no wrong-physics bug**: hydro metric terms, EMF curl (incl. 3D `(1/r)∂_θ` and `−ε_θ/r`), the full MHD linear operator (`_add_mhd_terms` vs reference `_Lxx` — viscous `∇²−1/r²`, `∓2/r²∂_θ` cross-couplings for u *and* b, `−Ω∂_θ`, Coriolis, field-stretch, `+B0∂_z`, pressure/continuity `1/r` terms), conducting BCs, and pressure pinning (saddle solves match dense reference solves to 1e-11 in x64). The MRI eigensolver blocks used for seeding match the evolved operator's signs. Prior-review defects (Nyquist, seed-2×, DAL denominator) are fixed *on the jaxfun side* — which is exactly what broke live parity against the unchanged reference (F-2); the fix now needs to be completed on the contract side. Named gaps, absorbed into findings: the 3/2-dealiased nonlinear path is never end-to-end parity-tested (F-11; statically verified correct — the padded-transform round-trip reduces to the identity on standard-grid input — and empirically stable, div ~1e-7, reality ~5e-16); float32 entirely unvalidated for the precision-sensitive per-mode saddle LU (F-7/F-11); full-complex θ,z layout relies on unmonitored conjugate-symmetry preservation (fine over 50 steps in x64; a reality/div monitor is cheap insurance for 10⁵-step float32 runs); finite-amplitude parity uses an active-coefficient floor that can mask discrepancies confined to small coefficients; exact CNAB2 restart requires the *full* state pytree (`nonlinear_old` + `have_old` are in the checkpoint payload — a fields-only restart would silently re-bootstrap IMEX-Euler).

### 6.3 Plane-Couette jaxfun ports — **no physics/sign bugs; gaps are wiring and coverage**
Term-by-term verification against the references found the ports faithful: KMM `_nonlinear_rhs` signs exactly match `ChannelFlow.py:152-163` (`Hu = ∂x∂y H₁ + ∂x∂z H₂ − ∂yy H₀ − ∂zz H₀`, `Hg = ∂z H₁ − ∂y H₂`, mean modes `−M00·H + dpdy`); the ARS stage accumulation (`src/jaxfun/integrators/coupled.py:13`, driving `KMM.step`) matches shenfun's `PDEIMEXRK` staging; the fluctuation base-flow terms (lift-up on streamwise only) match; the MHD solver applies `−J×B` into H and uses total velocity + background B0 in both Lorentz and EMF; the shearbox Coriolis (`n0 −= 2Ω u₁`, `n1 += 2Ω u₀`, `Ub = −S·x`) and the primitive MRI linear terms (`+2Ω u_y`, `(S−2Ω)u_x`, `−S b_x` Ω-effect, `+B0 ∂z` couplings, CNAB2 `M/dt − L/2`, pressure pinned at mode 0, inf-sup `Nx−2` truncation) all match the references. PCF seeding has no axisym-vs-3D 2× discrepancy (that was TC-only). Named gaps absorbed into findings: F-2b (primitive family unvalidated at finite amplitude), F-5/F-9 (grad wiring), F-4 (no remat; max gradient-test horizon anywhere is 3 steps), F-7 (float32), plus the F-17 items (stress-objective quadrature bias, no PCF sharding-parity test, default drift vs references, unjitted KMM step).

### 6.4 Production infrastructure — **truth-telling holds on the default paths; secondary paths can mislead**
Verified sound: `validate_gpu.sh` exit-code handling (`set -euo pipefail`, explicit status capture — no tee/pipe swallowing; timeouts counted as failures); the default golden-compare path (`validate_golden` rejects empty scalars, missing tolerances, hash/spec-hash mismatches; comparisons fail on NaN/missing/None-tolerance); fail-closed runner gates with correct report outcome mapping (completed-without-gate maps to *failed*, never a vacuous pass); honest per-row validation-scope labels; `tests/production` largely runs real solvers at small N (including a genuine bit-exact checkpoint-restart test) rather than mocking. The trust boundary: with one exception, everything vacuous or misleading sits *off* the default path — the `validate-all` smoke default and absent saturation-regression tier (F-2c), the explicit `--shenfun-golden` bypass, post-hoc-loosenable tolerances, missing-run-blind report aggregation, CPU-vs-CPU device "comparison", and the objectives test module that skips entirely under the float32 production default (all in F-17). The exception is F-13's floor-scope hole, which sits *on* the default full-run path: non-energy observables are not finiteness-checked for `generated_saturated_golden` runs, so a green full-saturation artifact can carry `null` theory observables. Net: trustworthy if you drive `parity-cheap`/`parity-dns`/`--full`, read exit codes, and close F-13; misleading if you read `validate-all` + `results.json` headlines at face value.

### 6.5 What the test suite actually proves (claim → strongest evidence)

| Claim | Strongest evidence | Regime | Tier |
|---|---|---|---|
| PCF hydro nonlinear correct | live parity, coeff+diag | amp 0.05 | `integration` |
| PCF MHD (KMM/A) nonlinear correct | live parity, full u/g/A coeffs, rtol 1e-8 | amp 0.05, 1/5/50 steps | `integration` |
| PCF MRI-shearbox α/stresses | live parity | amp 0.05 | `integration` |
| **PCF primitive (3 production specs)** | hand-audit + self-golden only | linear-regime tests only | **untested vs reference** |
| TC axisym/3D hydro nonlinear | live parity (coeffs) | amp 1e-4, ≤100 steps | `integration` — **currently red** (F-2) |
| TC MRI nonlinear (axisym+3D) | live parity, active-coeff floor 1e-8 | amp 1e-3, 50 steps | `integration` — **currently red** (F-2) |
| Linear eigen/non-modal (both geometries) | live parity + recorded goldens + analytic anchors (e.g. MRI Keplerian 15/16) | rtol ≤1e-11 | mixed, part default |
| DNS reproduces linear growth | DNS vs independent eigensolver | amp 1e-8, 50–100 steps | default (x64) |
| Gradients correct (hydro KMM) | finite-difference, full complex mixed-phase state | amp 0.15, 3 steps | default (x64) |
| Uniform snapshot format | parity vs shenfun `ShenfunFile` | — | `integration` |
| Saturation-regime parity vs shenfun | — | — | **does not exist** (self-goldens only) |
| Any solver physics at float32 | — | — | **does not exist** |
| GPU-vs-CPU agreement | cpu-vs-cpu determinism test only | — | manual only |

Reading the table: the default `uv run pytest` green proves linear-operator + solver-plumbing health in x64; every nonlinear-correctness proof lives behind `-m integration` with a live shenfun; float32 and saturation regimes have no reference-anchored proof at all.

---

*Review method: full manual read of the DAL scaffold, objectives, KMM solver, integrators, io/cadence, checkpoint and gate layers; five parallel deep-read audits over the reference layer, PCF ports, TC ports, production infra, and test integrity; plus the empirical runs in §2. All CRITICAL/HIGH findings above were reproduced or verified in code by the lead reviewer, not taken on faith from sub-audits.*
