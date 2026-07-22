# JAXfun PCF performance checkpoint: Phases 1–3

- Commit: `47955c5dbdf7ef23cedeee5c51a7acc930725ad7` plus the dirty working-tree
  changes recorded in each JSON artifact.
- Device: NVIDIA GeForce RTX 5090 Laptop GPU, GPU 0, float64.
- Protocol: 2 warmup blocks, 10 timed blocks, 25 physical steps per block,
  synchronized production `lax.scan` rollouts, and 3 same-shape `dt` probes.
- Resolution: `65 x 64 x 64`, below the bounded `128^3` cap.

## Results

| Checkpoint | Median ms/step | p90 ms/step | Temporary bytes | Decision |
| --- | ---: | ---: | ---: | --- |
| Hydro transform-gradient reference | 114.03 | 114.51 | 452,498,888 | retain as oracle |
| Hydro coefficient-gradient | 101.29 | 102.69 | 450,888,936 | keep; 11.2% faster |
| Hydro optimized-rotational | 84.74 | 85.01 | 374,095,592 | keep behind option; 16.3% faster than coefficient-gradient |
| MHD coefficient path, separate solves | 61.98 | 62.55 | 414,160,472 | retain as solve oracle |
| MHD coefficient path, batched solves | 49.73 | 49.99 | 414,158,680 | keep; 19.8% faster |

The paired final checksums and diagnostics agree to roundoff for the coefficient
and batched-solve comparisons. Rotational and gradient hydro have matching energy,
wall, mean-shear, and divergence diagnostics at the recorded horizon; their state
checksums differ slightly because the discrete projected forms are pressure-equivalent
only up to the documented projection tolerance. All `dt` probes reused the compiled
rollout variant.

## Artifacts

- `hydro-transform-gradient-65x64x64-r1.json`
- `hydro-optimized-gradient-65x64x64-r1.json`
- `hydro-optimized-rotational-65x64x64-r1.json`
- `mhd-coefficient-separate-65x64x64-r1.json`
- `mhd-optimized-batched-65x64x64-r1.json`

## Correctness

- Required baseline plus new tests: `54 passed in 238.25s`.
- Pallas multi-RHS parity on GPU 0: `4 passed` for Chebyshev/Legendre and 1/3 RHS.
- Pallas real MHD CNAB2 step: `1 passed`.
- Rotational hydro suite: `4 passed`.
- Rotational conducting/insulating MHD suite: `2 passed`.

Decision: accept Phases 1 and 2 as defaults with selectable transform/separate
references. Keep the rotational form selectable while Phase 4 trajectory and
third-order convergence qualification proceeds.
