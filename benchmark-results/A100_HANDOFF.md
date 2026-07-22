# A100 matched-comparison handoff

Use the exact implementation tree described in `FINAL_LOCAL_CHECKPOINT.md`.
Record a clean commit before externally reporting results, reproduce the
environment from `uv.lock`, and retain the emitted provenance in every JSON.
Use GPU 0, float64, no preallocation, synchronized rollouts, and a fresh cache
directory for every process repetition.

Selected optimized configuration:

- integrator: `SBDF3`;
- nonlinear form: `rotational`;
- coefficient path: `optimized`;
- solve batching: `batched`;
- wavenumber backend: first `jax`, then the qualified `pallas-triton` variant;
- periodic padding: `(1.0, 1.5, 1.5)`;
- protocol: 2 warmup blocks, 10 timed blocks, 25 steps/block, 3 processes.

For each solver, backend, and resolution in `(65,64,64)`, `(97,96,96)`, and
`(128,128,128)`, run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false JAXFUN_WAVENUMBER_SOLVER=jax \
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 \
NUMEXPR_NUM_THREADS=6 \
.venv/bin/python -m production.benchmark_pcf_refactor \
  --solver hydro --resolution NX NY NZ \
  --variant optimized-rotational --integrator SBDF3 \
  --padding 1.0 1.5 1.5 --warmup-blocks 2 --timed-blocks 10 \
  --rollout-steps 25 --dt-transition-probes 0 \
  --cache-dir benchmark-cache/a100-hydro-jax-NXxNYxNZ-rN \
  --out benchmark-results/a100-hydro-jax-NXxNYxNZ-rN.json
```

Repeat with `--solver mhd`. Repeat both with
`JAXFUN_WAVENUMBER_SOLVER=pallas-triton` and distinct cache/output names.
Fixed-step SBDF3 intentionally rejects dt-transition probes; record this scope
rather than changing dt inside a history.

Before the performance run, execute the default regression matrix, the slow
SBDF3 order tests, and the GPU Pallas tests. Compare coefficient diagnostics,
divergence, wall values, energy conventions, padding, Reynolds numbers, and
physical timestep with ShearPy and TorchChannel manifests. Generate all three
codes' numbers on the same A100 allocation. The A100 result—not the RTX 5090
laptop result—decides any externally reported cross-code comparison.
