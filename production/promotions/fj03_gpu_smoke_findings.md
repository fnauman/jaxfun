# FJ-03 GPU smoke findings and solver decision (2026-07-10)

GPU smoke runs of the **corrected (numerics_contract_version=2)** pipeline, on the
CUDA device, at the `start` resolution tier.

## Results

| run | path | result | `div_u` | `div_b` |
|---|---|---|---:|---:|
| `pcf_mhd_divfree` (plain MHD, decaying) | primitive-b | **completed, solenoidal** | 1.05e-8 | 7.15e-6 |
| `exp_pcf_mri_shearbox_growth` (MRI growth) | primitive-b | **divergence guard tripped at t=30** | — | 1.34e-2 > 1e-2 |
| `pcf_mri_vector_potential` (MRI) | curl / `B = curl A` | completed, **solenoidal by construction** | — | ~1e-18 |

## Interpretation

The corrected dealiasing (FJ-01) is verified on GPU: the decaying plain-MHD case is
now cleanly solenoidal, and a short nonlinear run reproduces bounded `div_u`/`div_b`.

The **primitive-b path cannot hold the solenoidal constraint at finite MRI amplitude**:
`div_b` drifts past the 1e-2 guard by `t=30` for `exp_pcf_mri_shearbox_growth` *even with
the corrected padding*. The primitive solver evolves `b` directly and only relies on the
induction term to preserve `div b = 0`; at growing MRI amplitude that drifts. This is the
"real constraint failure" FJ-03 anticipated — not merely a dealiasing artifact.

The **curl / vector-potential path** (`PlaneCouetteMRIShearpyJax`, now wired as the
`representation: vector_potential` production oracle) keeps `div_b ~ 1e-18` by construction.

## Decision (FJ-03 item 4/5)

Use the **vector-potential / curl family as the MRI-saturation and ZNF workhorse.**
`exp_pcf_mri_shearbox_growth` stays quarantined for the primitive path — its saturated
golden must be regenerated through the curl oracle (or the primitive path must gain a
wall-compatible Helmholtz/constrained projection before it can carry a saturated MRI
reference). The plain-MHD decaying `pcf_mhd_divfree` case remains valid on the primitive
path (low amplitude, solenoidality preserved).

Full-resolution production goldens for the chosen curl path remain a campaign-scale GPU
step (GPU-hours); these smoke runs validate the corrected pipeline and settle the solver
choice.
