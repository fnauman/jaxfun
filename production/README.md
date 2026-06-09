# jaxfun production layer

This package is the jaxfun side of the shared shenfun production contract. It
vendors the neutral problem-spec schema, example specs, and 13 committed shenfun
goldens so parity checks do not import or require a live shenfun process.

## Production-run inventory

| problem_id | geometry | physics | solver file | status | fallback rung |
|---|---|---|---|---|---|
| `pcf_fluct_re400` | pcf | hydro | `examples/pcf_fluctuations_jax.py` | production runner wired; bounded start-tier smoke tested; full KMM run pending | rung 3 |
| `pcf_mhd_divfree` | pcf | mhd | `examples/pcf_mri_primitive_jax.py` | production runner wired; bounded start-tier smoke tested; full saturation execution pending | rung 3 |
| `exp_pcf_mri_shearbox_growth` | pcf | mri | `examples/pcf_mri_primitive_jax.py` | production runner wired; bounded start-tier smoke tested; full saturation execution pending | rung 1/2/3 |
| `tc_supercritical_saturation` | taylor_couette | hydro | `examples/taylor_couette_dns_jax.py` | production runner wired; bounded start-tier smoke tested; full saturation execution pending | rung 2/3 |
| `tc_mri_nonlinear_saturation` | taylor_couette | mri | `examples/taylor_couette_dns_jax.py` | production runner wired; bounded start-tier smoke tested; full saturation execution pending | rung 1/2/3 |
| `stab_PCF_MRI_stability` | pcf | mri | `examples/pcf_mhd_mri_shearpy_jax.py` | config-undetermined placeholder | not executable |

## Support matrix

| Geometry | Physics path | Support state | Internal formulation | jaxfun source files | Axis mapping | Boundary/sign conventions | Golden / fallback | Divergence keys | Tests |
|---|---|---|---|---|---|---|---|---|---|
| pcf | hydro | parity_pending | laminar/linear cheap golden, primitive linear-window DNS golden, and `pcf_fluct_re400` runner path wired in `production/oracles.py` with `--resolution-tier start` smoke coverage; full KMM run still pending | `examples/pcf_fluctuations_jax.py`, `examples/pcf_linear_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip moving walls, `U_b=U_wall*x e_y` | `pcf_hydro_laminar_v1`; DNS precheck `pcf_hydro_primitive_dns_v1`; heavy `pcf_fluct_re400` rung 3 | `divergence_l2`; DNS `divergence_u` | `tests/production/test_run_problem.py`; DNS golden comparison wired, full KMM run pending |
| channel | hydro | production | driven KMM pressure-gradient steady state wired in `production/oracles.py`, with golden-normalized Poiseuille observables | `examples/channelflow_kmm.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | no-slip walls, pressure-gradient drive | `channel_poiseuille_hydro_v1` | `divergence_l2` | `tests/production/test_run_problem.py`; KMM steady-profile regression and CLI golden comparison |
| pcf | mhd | parity_pending | cheap linear golden and `pcf_mhd_divfree` primitive-`b` runner path wired in `production/oracles.py` with `--resolution-tier start` smoke coverage; full saturation run still pending | `examples/pcf_mhd_jax.py`, `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting magnetic walls; `Rm=Re*Pm` | `pcf_mhd_conducting_v1`; heavy `pcf_mhd_divfree` rung 3 | `divergence_u_l2`, `divergence_b_l2` | `tests/production/test_run_problem.py`; primitive 3D smoke tests and production smoke wired |
| pcf | mri | parity_pending | cheap shearbox linear golden, axisymmetric primitive `b` DNS golden, and `exp_pcf_mri_shearbox_growth` primitive-`b` runner path wired in `production/oracles.py` with `--resolution-tier start` smoke coverage; full saturation run still pending | `examples/pcf_mri_primitive_jax.py`, `production/oracles.py` | `axis_0=x`, `axis_1=y`, `axis_2=z` | conducting walls, Coriolis/shear, imposed `B0 e_z` | `pcf_mri_shearbox_v1`, DNS `pcf_mri_primitive_dns_v1`, heavy `exp_pcf_mri_shearbox_growth` | `divergence_u_l2`, `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | `tests/production/test_run_problem.py`; primitive PCF DNS golden comparison and production smoke wired |
| taylor_couette | hydro | parity_pending | linear/laminar cheap golden, linear-window DNS golden, and `tc_supercritical_saturation` runner path wired in `production/oracles.py` with `--resolution-tier start` smoke coverage; full saturation run still pending | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_linear_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | rotating no-slip cylinders, `V=A*r+B/r` | `taylor_couette_hydro_v1`, DNS `taylor_couette_hydro_dns_v1`, heavy `tc_supercritical_saturation` | `divergence_l2`; DNS `divergence_linf` | `tests/production/test_run_problem.py`; DNS golden comparison wired, full saturation run pending |
| taylor_couette | mhd/mri conducting | parity_pending | cheap linear golden, conducting linear-window DNS golden, and `tc_mri_nonlinear_saturation` runner path wired in `production/oracles.py` with `--resolution-tier start` smoke coverage; full saturation run still pending | `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | conducting walls, Alfven units, `Pm=Rm/Re` | `taylor_couette_mhd_conducting_v1`, DNS `taylor_couette_mhd_dns_v1`, heavy `tc_mri_nonlinear_saturation` | `divergence_b_l2`; DNS `divergence_u`, `divergence_b` | `tests/production/test_run_problem.py`; DNS golden comparison wired, full saturation run pending |
| taylor_couette | mhd insulating | parity_pending | cheap insulating linear golden wired in `production/oracles.py` | `examples/taylor_couette_mri_jax.py`, `production/oracles.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | insulating only for `m=0`, `kz!=0` | `taylor_couette_mhd_insulating_v1` | `divergence_b_l2` | `tests/production/test_run_problem.py`; loader rejection test for `m!=0` |
| pipe | hydro | parity_pending | missing axis-regularity radial basis; local promotion record `production/promotions/pipe_hydro_axis_regular_basis.md` | planned `examples/pipe_flow_dns_jax.py` | `axis_0=r`, `axis_1=theta`, `axis_2=z` | no-slip wall, regular axis at `r=0` | `pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`; skipped until basis lands | `divergence_l2` | loader rejection test names missing basis, promotion record, and both required goldens |
| pipe | mhd/mri | unsupported | no shenfun production formulation | none | `axis_0=r`, `axis_1=theta`, `axis_2=z` | unsupported | rejected to match shenfun | n/a | loader rejection test |

## Validation scripts

Current implemented entry points:

- `production/device.py` captures backend/device/dtype/golden-policy metadata.
- `production/compare_goldens.py` validates `schema_version=1` goldens and compares
  tolerance-declared scalars.
- `production/run_problem.py --validate-only` validates a spec and writes metadata;
  unsupported or intentionally unwired specs still fail explicitly before solver allocation.
- `make -C production parity-cheap` runs the seven non-pipe cheap golden comparisons and writes `runs/_report/results.json`; pipe hydro goldens are skipped until the axis-regular radial basis lands.
- `make -C production parity-dns` runs the four committed non-pipe linear-window DNS golden comparisons and writes `runs/_report/results.json`; `parity-dns-pcf` and `parity-dns-tc` run geometry-specific subsets.
- `production/validate_gpu.sh cheap|dns|dns-pcf|dns-tc` runs the same wired parity groups with a 30-minute timeout per run, writes `logs/<problem_id>.log` with command/status/duration, and writes `runs/_report/results.json`; strict parity subprocesses default to `JAXFUN_VALIDATE_PARITY_DTYPE=float64` for the committed `1e-10` goldens. `all`, `heavy`, and direct production run IDs execute bounded start-tier float32 smoke by default (`--resolution-tier start --steps 2`) unless `--full`, `--validate-only`, `--smoke`, or explicit run args are supplied; `--smoke` selects the lighter checked-in `smoke` resolution tier. Non-validate heavy runs also write `golden/golden.json` and `checkpoints/checkpoints.h5` by default.
- `production/run_problem.py --checkpoint-every K` writes HDF5 coefficient checkpoints for wired DNS paths through `production/checkpoint.py`, including spec hash, dtype/shape metadata, device metadata, and diagnostics pointer attrs.
- `production/objectives.py` exposes differentiable final-energy, integrated-energy, stress/alpha, growth-proxy, and PCF minimal-seed objectives with finite-difference tests.
- `production/compare_devices.py` runs the same config in separate device-specific subprocesses and compares final numeric diagnostics for CPU/GPU agreement checks.
- `production/report.py` builds machine-readable summaries from run metadata.

Long-run Phase J5/J6 entry point:

- `production/validate_gpu.sh --full` long-form heavy-run execution mode for saturation specs
