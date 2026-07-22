# JAXfun PCF performance refactor plan

**Status:** implementation handoff
**Local qualification hardware:** NVIDIA GeForce RTX 5090 Laptop GPU, GPU 0 only
**Later comparison hardware:** NVIDIA A100; cross-code comparisons are explicitly deferred
**Scope repository:** `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun`
**Reference commit at diagnosis:** `47955c5dbdf7ef23cedeee5c51a7acc930725ad7`

## 1. Objective

Reduce the steady-state GPU cost of the JAXfun plane-Couette-flow (PCF) hydro and vector-potential MHD solvers without weakening their boundary conditions, dealiasing, divergence constraints, float64 behavior, or formal temporal order.

The immediate goal is to make the JAXfun implementation materially more efficient on the local RTX 5090. Do not claim speedups against ShearPy or TorchChannel from this work; repeat the matched cross-code benchmark later on the A100.

The primary targets are:

- hydro: `examples/channelflow_kmm.py` and `examples/pcf_fluctuations_jax.py`;
- vector-potential MHD: `examples/pcf_mhd_jax.py` and `examples/pcf_mhd_mri_shearpy_jax.py`;
- tensor-product transforms: `src/jaxfun/galerkin/tensorproductspace.py`;
- per-wavenumber implicit solves: `src/jaxfun/la/tpmatrix.py`;
- benchmark and correctness infrastructure under `production/` and `tests/`.

## 2. Constraints and non-goals

1. Use only GPU 0:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   export JAX_ENABLE_X64=1
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```

2. Do not run any case larger than `128^3`, or an anisotropic case with more than the equivalent number of collocation points. Prefer `64^3` for iteration and use `96^3`/`128^3` only for bounded scaling checks.
3. Compilation time must be recorded separately and excluded from warm-step timing.
4. Every timed result must synchronize the returned JAX state. Prefer the existing `production.benchmark` harness, which already warms up, uses compiled `lax.scan` rollouts, and calls `block_until_ready`.
5. Preserve float64. Do not obtain speedups by enabling TF32, float32, relaxed precision, weaker dealiasing, fewer evolved fields, or changed boundary conditions.
6. Preserve the existing implementation as a selectable reference until each replacement passes parity and convergence tests. Optimize one concern at a time; do not combine the integrator, nonlinear formulation, storage format, and linear solver in one unreviewable change.
7. Do not create GitHub issues or PRs as part of this task. Local commits are optional, but each phase should leave a clean, independently testable checkpoint.
8. A generic rewrite of all JAXfun Fourier spaces is not the first milestone. The PCF hot paths should demonstrate the benefit before a library-wide storage migration is attempted.

## 3. Diagnosed baseline

The existing A100 matched benchmark found the following warm per-step costs:

| Problem | Resolution | JAXfun | ShearPy | TorchChannel compiled |
| --- | ---: | ---: | ---: | ---: |
| Hydro, third-order | `64^3` class | 59.7 ms | 4.0 ms | 1.5 ms |
| Hydro, third-order | `128^3` class | 225.8 ms | 11.1 ms | 11.6 ms |
| MHD, CNAB2 | `64^3` class | 30.5 ms | 4.9 ms | 2.9 ms |
| MHD, CNAB2 | `128^3` class | 120.6 ms | 18.7 ms | 30.3 ms |

These values are context, not local acceptance thresholds. Re-establish an RTX 5090 baseline before changing code because absolute timings are hardware- and software-version-dependent.

The diagnosis identified these main costs:

- A hydro ARS IMEXRK443 step evaluates the expensive nonlinear path four times. A local one-evaluation CNAB2 diagnostic took about 21.7 ms versus about 100 ms for the four-stage third-order path at the same small resolution. CNAB2 is not a valid third-order replacement; the measurement isolates stage-count cost.
- A traced ARS hydro step contains 76 high-level tensor-product transform/projection calls: 4 `TB.backward`, 4 `TD.backward`, 16 `TB.backward_primitive`, 36 `TD.backward_primitive`, and 16 forward/scalar-product calls. Optimized HLO contained roughly 280 FFT operations. HLO operation counts are diagnostic, not equivalent to kernel-launch counts.
- `KMM.convection` computes all nine velocity gradients in physical space. `_nonlinear_rhs` then receives coefficient-space `H` but performs another physical round trip before projecting the KMM right-hand sides.
- `_reconstruct_velocity` similarly differentiates `u0` by transforming to physical space and then forward-transforming into `TD`.
- JAXfun uses full-complex Fourier coefficients and complex physical intermediates where a real/Hermitian representation could reduce work and memory.
- At the `128^3` class size, the default Chebyshev `Su` and `Sg` runtime factors occupy roughly 1.5 GB. Local compiled-step temporary storage was about 5.8 GB. The dense-upper factor representation scales approximately as periodic modes times the square of wall-normal order.
- The optional `JAXFUN_WAVENUMBER_SOLVER=pallas-triton` reduced local `64^3` hydro and MHD times by approximately 1.2-1.3x and reduced temporary memory, but gave only about 1.05x hydro speedup at the `128^3` class size. It is useful but is not the root-cause fix.
- The three MHD vector-potential component solves use the same factor but are invoked separately. They took about 9.6 ms together in the local small-case diagnostic.

## 4. Required benchmark artifacts

Before implementation, add or adapt a small benchmark driver that emits JSON and can run hydro and vector-potential MHD at explicit resolutions. Reuse `production.benchmark.benchmark_step`; do not duplicate its synchronization logic. The driver should record:

- Git commit and dirty status;
- JAX, jaxlib, CUDA, driver, GPU, and Python versions;
- solver, basis family, domain, resolution, padding, Reynolds numbers, `dt`, integrator, and wavenumber-solver backend;
- compile seconds;
- median and p90 warm milliseconds per physical step;
- number of warmup blocks, timed blocks, and physical steps per block;
- generated-code, argument, output, alias, and temporary bytes from compiled memory analysis;
- maximum allocated GPU memory when available;
- final state checksum or a compact set of physical diagnostics.

Use separate compilation-cache directories per code variant so a stale executable cannot contaminate comparisons. A suitable local protocol is:

- 2 untimed warmup blocks;
- 10 timed blocks;
- 25 physical steps per compiled rollout block;
- at least 3 complete process repetitions for important before/after claims;
- report the median of process medians and retain all raw JSON;
- `64^3` class for every checkpoint, `96^3` after a promising change, and `128^3` only at phase gates.

The existing production path can be invoked along these lines:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
JAX_COMPILATION_CACHE_DIR=benchmark-cache/baseline-mhd \
.venv/bin/python -m production.benchmark \
  --config production/runs/exp_pcf_mri_vector_potential.json \
  --tiers start --warmup-steps 2 --timed-steps 10 \
  --rollout-steps 25 --dt-transition-probes 3 \
  --out benchmark-results/baseline-mhd.json
```

Add an equivalent hydro configuration or a thin driver around `benchmark_step`, because the current run-spec inventory does not expose an obvious standalone hydro benchmark configuration.

For each phase, compare against the immediately preceding accepted checkpoint and the original baseline. A change is not a performance win if it merely shifts work outside the timed region, recompiles on every `dt` transition, or increases memory enough to threaten the `128^3` case.

## 5. Implementation phases

### Phase 0: lock down correctness and cost attribution

1. Run the relevant existing tests before modifying code:

   ```bash
   CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 uv run pytest -q \
     tests/integrators/test_cnab2.py \
     tests/integrators/test_imex_rk.py \
     tests/la/test_tpmatrices_solvers.py \
     tests/couette/test_pcf_fluctuations_jax.py \
     tests/couette/test_pcf_mhd_jax.py \
     tests/couette/test_pcf_mhd_mri_shearpy_jax.py \
     tests/couette/test_pcf_mri_cnab2_order_jax.py \
     tests/production/test_benchmark.py \
     tests/production/test_benchmark_real_measurement.py
   ```

   Run live-Shenfun tests only if their documented reference environment is configured.

2. Save original RTX 5090 benchmark JSON for hydro and MHD at the `64^3`, `96^3`, and bounded `128^3` classes.
3. Add a transform-call census test/helper for one eager `step`, plus a reproducible `make_jaxpr`/lowered-HLO summary. Keep it diagnostic rather than asserting unstable raw HLO text. Stable assertions may cover high-level calls under controlled monkeypatching.
4. Record separate timings for convection, `_nonlinear_rhs`, velocity reconstruction, `Su`, `Sg`, and the three-component `SA` solve. Component timings are attribution only; the compiled full rollout is the acceptance metric.

**Exit gate:** baseline tests pass; JSON artifacts are reproducible within approximately 10% across process medians; no resolution exceeds the stated cap.

### Phase 1: remove coefficient/physical round trips

This phase should preserve the current gradient-form convection and time integrators, making numerical parity relatively easy to establish.

1. Rewrite `KMM._nonlinear_rhs(H)` to form `Nu` and `Ng` from coefficients using `derivative_orthogonal_coeffs` and `project_from_orthogonal`, or equivalent sparse coefficient operators. `H` is already in `TD`; it should not require six inverse derivative transforms followed by physical scalar products.
2. Rewrite `_reconstruct_velocity` so the wall-normal derivative of the `TB` coefficient field is converted/projected into `TD` without a backward-to-physical/forward round trip.
3. Ensure the new coefficient path applies the same Nyquist masking, normalization, test-space projection, and mean-mode treatment as the original.
4. Keep private reference implementations during development and add direct randomized parity tests for Chebyshev and Legendre at small sizes. Compare each returned coefficient array, not only aggregate diagnostics.

Suggested tolerances in float64 are `rtol=2e-11, atol=2e-12` initially; tighten them where the existing transform path agrees more closely. Any systematic normalization factor or mean-mode discrepancy must be fixed, not hidden by a loose tolerance.

**Exit gate:** all existing PCF/MHD tests pass; coefficient-level parity passes for both basis families; high-level transform count is materially reduced; `64^3` full-rollout time improves by at least 10% or the phase is documented as neutral and kept only if it significantly reduces memory/complexity.

### Phase 2: batch compatible transforms and repeated solves

1. Add a multi-right-hand-side path to `TPMatricesWavenumberSolver` or a solver-level wrapper that places the RHS-component axis where one compiled solve can reuse the same factors. Validate the layout from lowered HLO and profiling; a Python loop hidden under `jit` is not sufficient evidence of factor reuse.
2. Use it for the three identical `SA` vector-potential solves in `PlaneCouetteMHDJax._A_solve`.
3. Batch the two mean tangential-velocity solves that use `S00_factor`.
4. In `update_B_from_A` and `update_J_from_B`, avoid repeated `to_orthogonal` conversion of the same component. Convert a component once, derive all required directions, then project into the heterogeneous target spaces. Preserve conducting and insulating space overrides.
5. Group backward/forward transforms only when spaces, sizes, normalization, padding, and dtype are identical. The `TB`, `TD`, and `TC` wall spaces cannot be indiscriminately stacked.

**Exit gate:** solver parity passes for multiple RHS counts, Chebyshev/Legendre, representative complex arrays, zero/Nyquist modes, and both `jax` and `pallas-triton` backends. MHD `64^3` warm-step time should improve by at least 10% without hydro regression.

### Phase 3: rotational nonlinear formulation

Replace nine-gradient convection with curl/cross-product operations, but preserve the mathematical sign convention and fluctuation/base-flow decomposition exactly.

1. Implement a reference-tested coefficient-space vorticity construction. Reuse or improve `velocity_vorticity_physical`; avoid six separately converted derivatives when components can share conversions.
2. Form the incompressible nonlinear term using the rotational identity, with the gradient contribution absorbed by the pressure projection. Do not guess whether the implementation needs `u x omega` or `omega x u`: derive it from the sign of the current `H`, KMM right-hand side, and Lorentz term, then prove it with coefficient parity tests.
3. For plane Couette fluctuations, include the base velocity and base vorticity consistently. The current `_add_base_convection` hooks and subclasses must remain physically equivalent. Test total-field and fluctuation-field formulations against the original gradient path.
4. For MHD, share the already computed physical fields and build compatible batches for `u`, `omega`, `B`, and `J`. Form momentum and induction products with `physical_cross`, respecting the existing signs for convection, `J x B`, and `u x B`.
5. Keep `nonlinear_form="gradient"|"rotational"` temporarily, with gradient as an oracle. Change a production default only after all parity and convergence gates pass.

Because the two forms differ by a pressure gradient, compare the projected KMM right-hand sides and one-step state, not necessarily the unprojected physical vector pointwise. Add tests for:

- zero field;
- laminar Couette flow;
- a manufactured divergence-free Fourier/Chebyshev field;
- randomized valid coefficients with Nyquist modes masked;
- conducting and insulating MHD;
- energy-transfer identities, including near-zero inviscid self-work of the rotational term within discretization tolerance.

**Exit gate:** projected right-hand sides and trajectories agree with the gradient oracle; divergence and wall constraints are unchanged; hydro and MHD `64^3` rollouts each improve materially. Target at least 1.4x for the nonlinear hot path, while accepting that whole-step gain will be smaller.

### Phase 4: add a third-order one-evaluation multistep integrator

Implement an IMEX SBDF3/BDF3-EXT3 option for hydro and MHD. This is expected to be the largest hydro improvement.

For fixed `dt`, the intended structure for `M du/dt = L u + N(u)` is:

```text
(11/6 M - dt L) u[n+1]
  = M (3 u[n] - 3/2 u[n-1] + 1/3 u[n-2])
    + dt (3 N[n] - 3 N[n-1] + N[n-2]).
```

Confirm signs against the existing solver definitions rather than copying this formula mechanically.

1. Add a reusable integrator/history helper under `src/jaxfun/integrators/`, with unit tests independent of PCF.
2. Extend `KMMState` and `MHDState` with fixed-shape two-level solution and nonlinear history. Avoid `None`-driven structural changes after initialization; compiled scan state structure must remain constant.
3. Use a third-order-compatible startup for the first two steps. The safest initial implementation is the already validated four-stage third-order method for startup only, followed by SBDF3. Do not bootstrap with plain Euler/CNAB2 and then claim third-order global convergence without demonstrating it.
4. Build only the single SBDF3 implicit factor needed per active `dt`, rather than stage-specific ARS factors.
5. Apply the same scheme to `u0`, `g`, the two mean modes, and all three components of `A`. Evaluate the full nonlinear hydro/MHD operator once per steady-state step.
6. Decide variable-step scope explicitly. If production permits changing `dt`, either implement variable-step BDF3/EXT3 coefficients with ratio guards and tests, or reject unsupported transitions clearly. Never silently use constant-step coefficients after `dt` changes. The existing benchmark's `dt_transition_probes` must show correct results and no unexpected recompilation for supported transitions.
7. Update run-spec validation and integrator string dispatch without changing historical configs unexpectedly.

Required temporal tests:

- scalar split linear equation with known solution;
- diffusion-only KMM-compatible mode;
- nonlinear manufactured solution if available;
- hydro PCF convergence across at least four `dt` values at fixed spatial resolution;
- MHD/MRI convergence using existing CNAB2-order-test patterns adapted to third order;
- restart/history serialization and compiled `lax.scan` equivalence;
- startup transition and dynamic-`dt` behavior.

Accept observed order conservatively in a band such as 2.7-3.3 after confirming the asymptotic regime. Also compare diagnostics and coefficients with a small-`dt` ARS IMEXRK443 reference over the same physical time.

**Exit gate:** demonstrated third-order convergence; one nonlinear evaluation per post-startup step; no wall/divergence regression; at least 2.5x hydro whole-step speedup over the original four-stage baseline at `64^3`. MHD may gain little from stage count because its current CNAB2 path already evaluates once, but SBDF3 should provide comparable formal order at similar per-step cost.

### Phase 5: qualify the Pallas wavenumber solver and reduce factor traffic

1. Expand `tests/la/test_tpmatrices_solvers.py` to cover the actual PCF `Su`, `Sg`, `S00`, and `SA` factor shapes, multiple RHS, complex float64, boundary rows, zero modes, and the largest bounded local size that is practical.
2. Compare default and `pallas-triton` output to approximately `2e-12` in small cases and use residual norms for larger cases.
3. Record compile time, warm solve time, whole-step time, argument bytes, and temporary bytes for both backends. Do not make Pallas the default based only on isolated-solve timing.
4. Investigate a structured Shen/quasi-inverse/ultraspherical solve that does not store dense-upper factors per Fourier mode. Begin with an operator/factor-memory design note and prototype on `Sg`; do not entangle this research change with the already qualified Pallas path.

**Exit gate for Pallas:** all solver and production parity tests pass on GPU 0; no regression at `96^3`/`128^3`; meaningful whole-step or memory benefit. The structured solver is a separate milestone and should not block earlier gains.

### Phase 6: prototype real/Hermitian periodic storage

This is high-reward but high-risk and should start only after Phases 1-5 provide a stable optimized baseline.

1. Write a focused design note defining shapes, owned modes, normalization, conjugate reconstruction, Nyquist masking, derivatives, padding/truncation, and interaction with sharding.
2. Prototype a PCF-specific tensor-product path using a real transform on one periodic axis while retaining the wall-normal polynomial transform and the other Fourier axis as needed.
3. Do not change the generic `Fourier`/`TensorProduct` public representation until forward/backward, derivative, projection, dealiasing, and MPI/SPMD parity are proven.
4. Verify that all physical velocity, vorticity, magnetic, and current fields are real to roundoff and that nonlinear products match the full-complex reference.
5. Measure both arithmetic and memory. A representation change is worthwhile only if the full rollout improves substantially; transform microbenchmarks alone are insufficient.

**Exit gate:** coefficient/physical parity, Nyquist and derivative tests, dealiased nonlinear parity, PCF/MHD trajectory agreement, and at least 20% full-rollout improvement or a compelling memory-capacity improvement at the bounded largest case.

## 6. Correctness matrix

Every accepted phase must cover the relevant rows below.

| Dimension | Required cases |
| --- | --- |
| Physics | hydro PCF; conducting vector-potential MHD; MRI/shearing-box-derived PCF path; insulating path when shared code changes |
| Basis | Chebyshev mandatory; Legendre for generic transform/solver changes |
| Resolution | tiny unit grids; `64^3` performance; `96^3` scaling; bounded `128^3` final local gate |
| Time | one step; compiled multi-step scan; equal physical-time trajectory; restart/history; convergence in `dt` |
| Constraints | no-slip wall values; velocity divergence; magnetic divergence; mean modes; Hermitian/Nyquist consistency |
| Quantities | coefficient arrays; kinetic/magnetic energies; dissipation/enstrophy where available; wall shear; mean shear; growth rate for MRI cases |
| Execution | eager reference; `jax.jit`; production `lax.scan`; supported `dt` transitions |

Do not update golden data merely because an optimized path disagrees. First determine whether the difference is roundoff, the pressure-gradient equivalence of nonlinear forms, or a genuine physics/normalization error.

## 7. Performance acceptance and regression policy

Use the original baseline, the previous checkpoint, and an unchanged reference path in every phase report. Report medians, p90, memory, and raw artifacts.

Final local goals, treated as targets rather than promises:

- hydro: 4-8x faster than the original ARS IMEXRK443 path at the `64^3` class size, with third-order accuracy retained;
- MHD: 2-4x faster than the original vector-potential implementation at the `64^3` class size, with at least third-order temporal accuracy if SBDF3 is selected;
- no worse than 5% regression at `96^3` or `128^3` for any accepted default;
- substantially smaller transform count and compiled temporary/argument memory;
- no new compilation during steady-state rollouts or supported same-shape `dt` changes.

Do not multiply isolated speedup factors to predict the final result; transform, solve, memory, and integrator gains overlap.

If a change improves `64^3` but regresses `128^3`, keep it behind an experimental option until the scaling cause is understood. If performance changes by less than approximately 5-10%, repeat in fresh processes and inspect variance before drawing a conclusion.

## 8. Recommended checkpoint sequence

Keep changes reviewable in this order:

1. benchmark harness and baseline artifacts;
2. coefficient-space `_nonlinear_rhs` and reconstruction;
3. multi-RHS solves and repeated conversion elimination;
4. rotational nonlinear form behind an option;
5. SBDF3/EXT3 plus history, startup, convergence, and restart tests;
6. qualify Pallas and decide its default status;
7. structured wall-solver research prototype;
8. real/Hermitian storage prototype;
9. remove reference paths only after full local qualification and the later A100 comparison.

At each checkpoint, write a short result note containing commit, commands, JSON paths, correctness status, measured deltas, memory deltas, and the decision to keep/revert/continue.

## 9. A100 handoff package

When local work is complete, prepare—but do not run on the RTX 5090 as a substitute for—the later matched A100 comparison package:

- exact JAXfun commit and clean-tree status;
- environment lock/version output;
- selected integrator, nonlinear form, transform representation, and wavenumber backend;
- immutable problem manifests for hydro and MHD;
- `64^3`, `96^3`, and `128^3` commands, never larger;
- compilation-separated synchronized benchmark commands;
- local correctness/convergence summary and raw RTX 5090 JSON;
- original and optimized transform census and compiled-memory analysis;
- explicit note that ShearPy remains the baseline and that TorchChannel/ShearPy numbers must be generated on the same A100 allocation, GPU 0, with matched normalization and physics.

The A100 test should decide the externally reported speed comparison. The RTX 5090 results decide whether each JAXfun refactor is internally worthwhile and correct.

---

**Signed:** Codex (OpenAI)
**Date:** 2026-07-22
