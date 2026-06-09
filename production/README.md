# jaxfun production layer

This package is the jaxfun side of the shared shenfun production contract. It
vendors the neutral problem-spec schema, example specs, and 13 committed shenfun
goldens so parity checks do not import or require a live shenfun process.

## Production-run inventory

| problem_id | geometry | physics | solver file | status | fallback rung |
|---|---|---|---|---|---|
| `pcf_fluct_re400` | pcf | hydro | `examples/pcf_fluctuations_jax.py` | not yet executed through production runner | rung 3 |
| `pcf_mhd_divfree` | pcf | mhd | `examples/pcf_mri_primitive_jax.py` | blocked on primitive-`b` PCF port | rung 3 |
| `exp_pcf_mri_shearbox_growth` | pcf | mri | `examples/pcf_mri_primitive_jax.py` | blocked on primitive-`b` PCF port | rung 1/2/3 |
| `tc_supercritical_saturation` | taylor_couette | hydro | `examples/taylor_couette_dns_jax.py` | not yet executed through production runner | rung 2/3 |
| `tc_mri_nonlinear_saturation` | taylor_couette | mri | `examples/taylor_couette_dns_jax.py` | not yet executed through production runner | rung 1/2/3 |
| `stab_PCF_MRI_stability` | pcf | mri | `examples/pcf_mhd_mri_shearpy_jax.py` | config-undetermined placeholder | not executable |

## Support matrix

| Geometry | Physics path | Support state | Internal formulation | jaxfun source files | Axis mapping | Boundary/sign conventions | Golden / fallback | Divergence keys | Tests |
|---|---|---|---|---|---|---|---|---|---|
| pcf | hydro | parity_pending | KMM velocity-vorticity / primitive diagnostics | `examples/pcf_fluctuations_jax.py`, `examples/pcf_linear_jax.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip moving walls, `U_b=U_wall*x e_y` | `pcf_hydro_laminar_v1`; DNS precheck `pcf_hydro_primitive_dns_v1`; heavy `pcf_fluct_re400` rung 3 | `divergence_l2`; DNS `divergence_u` | existing couette tests plus `tests/production` contract tests |
| channel | hydro | parity_pending | driven KMM channel | `examples/channelflow_kmm.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip walls, pressure-gradient drive | `channel_poiseuille_hydro_v1` | `divergence_l2` | production adapter/comparator tests; driven run wiring pending |
| pcf | mhd | parity_pending | cheap linear parity; DNS moving to primitive `b` | `examples/pcf_mhd_jax.py`, `examples/pcf_mri_primitive_jax.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting magnetic walls; `Rm=Re*Pm` | `pcf_mhd_conducting_v1`; heavy `pcf_mhd_divfree` rung 3 | `divergence_u_l2`, `divergence_b_l2` | observable-key and comparator tests; primitive DNS pending |
| pcf | mri | parity_pending | primitive `b` port required; old vector-potential port is not golden DNS path | `examples/pcf_mri_primitive_jax.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting walls, Coriolis/shear, imposed `B0 e_z` | `pcf_mri_shearbox_v1`, DNS `pcf_mri_primitive_dns_v1`, heavy `exp_pcf_mri_shearbox_growth` | `divergence_u_l2`, `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | primitive PCF DNS implementation pending |
| taylor_couette | hydro | parity_pending | primitive perturbation DNS / linear solver | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_linear_jax.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | rotating no-slip cylinders, `V=A*r+B/r` | `taylor_couette_hydro_v1`, DNS `taylor_couette_hydro_dns_v1`, heavy `tc_supercritical_saturation` | `divergence_l2`; DNS `divergence_linf` | existing TC tests plus production comparator tests |
| taylor_couette | mhd/mri conducting | parity_pending | primitive MHD/MRI DNS | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_mri_jax.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | conducting walls, Alfven units, `Pm=Rm/Re` | `taylor_couette_mhd_conducting_v1`, DNS `taylor_couette_mhd_dns_v1`, heavy `tc_mri_nonlinear_saturation` | `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | existing TC MHD tests plus production comparator tests |
| taylor_couette | mhd insulating | parity_pending | axisymmetric flux-function magnetic path | `examples/taylor_couette_mri_jax.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | insulating only for `m=0`, `kz!=0` | `taylor_couette_mhd_insulating_v1` | `divergence_b_l2` | loader rejection test for `m!=0` |
| pipe | hydro | parity_pending | missing axis-regularity radial basis | planned `examples/pipe_flow_dns_jax.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | no-slip wall, regular axis at `r=0` | `pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`; skipped until basis lands | `divergence_l2` | loader rejection test names missing basis and both required goldens |
| pipe | mhd/mri | unsupported | no shenfun production formulation | none | `axis_0=r`, `axis_1=theta`, `axis_2=z` | unsupported | rejected to match shenfun | n/a | loader rejection test |

## Validation scripts

Current implemented entry points:

- `production/device.py` captures backend/device/dtype/golden-policy metadata.
- `production/compare_goldens.py` validates `schema_version=1` goldens and compares
  tolerance-declared scalars.
- `production/run_problem.py --validate-only` validates a spec and writes metadata;
  full solver execution still fails explicitly until solver factories are wired.

Planned Phase J3/J6 entry points:

- `production/validate_gpu.sh`
- `production/report.py`
- `production/Makefile`
