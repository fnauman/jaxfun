# Final local PCF performance-refactor checkpoint

Date: 2026-07-22
Reference commit: `47955c5dbdf7ef23cedeee5c51a7acc930725ad7` (dirty implementation tree)
Hardware: NVIDIA GeForce RTX 5090 Laptop GPU, GPU 0

The JSON artifacts in this checkpoint were captured before benchmark checksums
were narrowed to primary physical fields. Their full-state checksum aggregates
are retained as historical measurements and must not be compared across time
integrators; newly generated results identify the checksum scope explicitly.
Software: Python 3.12.3, JAX/jaxlib 0.10.1, CUDA runtime 13.0, driver 580.159.04

## Accepted implementation

- coefficient-space KMM right-hand sides and velocity reconstruction, with the
  transform implementation retained as an oracle;
- multi-RHS wavenumber solves for JAX and Pallas and batched MHD/mean solves;
- conversion reuse in magnetic curl construction;
- selectable gradient and rotational nonlinear forms;
- fixed-step SBDF3/BDF3-EXT3 with IMEXRK3 startup, fixed-shape two-level
  solution/nonlinear history, exact checkpoint restart, and one nonlinear
  evaluation per steady step;
- a steady-only compiled rollout that excludes startup factor traffic;
- host-resident startup-only factors so the steady GPU holds one factor per
  active implicit block;
- actual-shape Pallas qualification and separate structured-Sg and Hermitian
  representation research prototypes. Neither prototype changes production
  defaults.

Historical configurations and defaults remain unchanged. `SBDF3`, rotational
nonlinearity, and Pallas are explicit selections.

## Correctness

- Full selected regression matrix: `85 passed, 84 deselected` in 490.26 s.
- Slow SBDF3 temporal gates: `3 passed` in 164.81 s.
- Observed hydro self-convergence orders: 3.09 and 3.05.
- Observed MRI vector-potential MHD orders: 3.07 and 3.04.
- Diffusion-only KMM mean-mode orders: 3.10, 3.04, and 3.02.
- Conducting and insulating startup/steady, exact restart, fixed-dt rejection,
  eager/compiled equivalence, and one-evaluation gates pass.
- Actual PCF Pallas `Su`, `Sg`, `S00`, and `SA` tests pass for Chebyshev and
  Legendre; the 65x64x64 large Sg residual gate passes.
- Hermitian prototype roundtrip, full-complex derivative, Nyquist, padding,
  truncation, dealiased-product, full/half coefficient reconstruction, and
  hydro/MHD trajectory parity pass. At 64 periodic points the stored last
  axis uses 33 rather than 64 complex modes (48.4% coefficient reduction).
- Compact Sg prototype matches production for both polynomial families and
  demonstrates the separable-operator persistent-memory floor.

## 65x64x64 synchronized results

Protocol: two warmup blocks, ten timed blocks, 25 physical steps per block,
three fresh processes, float64, 3/2 periodic padding, JAX wavenumber backend.

| Solver | Process medians (ms/step) | Median | p90 range | Arguments | Temporaries |
| --- | --- | ---: | ---: | ---: | ---: |
| Hydro, rotational SBDF3 | 27.479, 27.600, 27.552 | 27.552 | 27.656-27.794 | 245.1 MB | 231.5 MB |
| MHD, rotational SBDF3 | 45.851, 45.689, 45.790 | 45.790 | 45.837-46.002 | 375.1 MB | 428.2 MB |

Relative to the immediately preceding local checkpoints:

- hydro versus coefficient-gradient IMEXRK3 (101.286 ms): 3.68x faster;
- hydro versus rotational IMEXRK3 (84.740 ms): 3.08x faster;
- hydro versus the transform-gradient reference (114.031 ms): 4.14x faster;
- hydro argument bytes fell 59.5% and temporary bytes 38.1% versus rotational
  IMEXRK3;
- MHD versus separate coefficient solves (61.981 ms): 1.35x faster;
- MHD versus batched CNAB2 (49.731 ms): 1.09x faster while increasing formal
  temporal order from two to three.

The MHD 2-4x aspirational target was not reached locally. No cross-code speedup
is claimed.

## Scaling gates

| Solver | Resolution | Warm ms/step | Arguments | Temporaries | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hydro SBDF3 | 97x96x96 | 92.94 | 1.175 GB | 0.883 GB | 6.980 GB |
| MHD SBDF3 | 97x96x96 | 163.28 | 1.728 GB | 1.699 GB | 8.360 GB |
| Hydro SBDF3 | 128x128x128 | 219.72 | 3.531 GB | 4.491 GB | 12.628 GB |

The 128x128x128 MHD steady executable exceeds the available workspace on this
laptop GPU with both JAX and Pallas. The rollout itself reached execution in a
single-step Pallas attempt, but synchronization reported a 3.31 GB allocation
failure; the benchmark now propagates such failures instead of swallowing
them. Therefore 97x96x96 is the largest successful local MHD gate. This is an
explicit capacity result, not a passing 128 artifact.

## Artifact inventory

- Final hydro repetitions:
  `hydro-sbdf3-rotational-65x64x64-r2.json` through `r4.json`.
- Final MHD repetitions:
  `mhd-sbdf3-rotational-65x64x64-r1.json` through `r3.json`.
- Scaling:
  `hydro-sbdf3-rotational-97x96x96.json`,
  `mhd-sbdf3-rotational-97x96x96.json`, and
  `hydro-sbdf3-rotational-128cube.json`.
- Earlier accepted checkpoint: `CHECKPOINT_PHASES_1_3.md` and its raw JSON.
- Research notes: `DESIGN_STRUCTURED_SG.md` and
  `DESIGN_HERMITIAN_PERIODIC.md`.

Decision: keep Phases 1-4 and the qualified multi-RHS/Pallas support. Keep the
structured and Hermitian work experimental until their stated whole-rollout
gates are met. Proceed to the matched A100 qualification without changing the
historical production defaults.
