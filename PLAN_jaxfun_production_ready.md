# Plan: Make jaxfun the production GPU solver and execute the heavy production runs

## File location and authority

This plan is authored at `/home/nauman/cfd/PLAN_jaxfun_production_ready.md`, and an
**identical copy is kept in the fork** at
`shenfun_jaxfun_spectralDNS/fork_jaxfun/PLAN_jaxfun_production_ready.md` (reconciled
2026-06-09 — the two are byte-identical). The Phase-J scaffolding this plan
describes (the `production/` package, the `couette/` reference ports, the
`tests/_parity.py` harness, the `.venv`) all live under
`/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/`.

**Keep the two copies in sync.** When this document changes, re-copy it into the
fork (or leave the fork copy a one-line redirect to this path) so they never
diverge again. The implementing agent follows whichever copy it opens — they are
the same content. All work products described below are created under
`shenfun_jaxfun_spectralDNS/fork_jaxfun/` regardless of where this plan text lives.

## Purpose

This document is an implementation brief for an autonomous agent. The goal is to
turn the jaxfun solver family into the production execution layer for the heavy
DNS runs that shenfun specified but could not afford to run on CPU, and to make
PCF / Taylor-Couette / pipe parity between shenfun and jaxfun a **completed,
tested fact now** — not a deferred aspiration.

Two things changed since the previous revision of this plan:

1. **shenfun productionization is essentially complete.** The shenfun production
   reference layer at `fn_shenfun/demo/production/` is fully built out per
   `PLAN_shenfun_production_ready.md` (phases S0-S6): a framework-neutral problem
   spec + JSON schema, a canonical observables layer, a golden writer/validator,
   a verification floor, a CLI runner, **13 committed goldens (9 cheap + 4
   nonlinear-DNS), and 13 promotion records**, and has since rewritten the PCF
   MHD/MRI DNS in primitive variables (see the currency section below). The shenfun
   promotion gate is therefore **satisfied**. jaxfun parity is **unblocked**.

2. **shenfun cannot run the heavy *saturated* DNS.** shenfun runs on CPU by default
   and it is hard to get enough CPUs for heavy DNS, so the nonlinear *saturated*
   production runs that shenfun *specified* (PCF Re400 fluctuation DNS, PCF MHD,
   PCF MRI net-flux, TC supercritical / MRI saturation) were left planned-not-run.
   shenfun's committed goldens are either **cheap** linear-eigenvalue / analytic
   scalar artifacts (`final_time=0.01`, `N<=64`) or **linear-regime DNS** goldens
   that seed the leading eigenmode at tiny amplitude and pin the measured DNS
   growth/decay rate to the linear eigenvalue (`pcf_hydro_primitive_dns_v1`,
   `pcf_mri_primitive_dns_v1`, `taylor_couette_hydro_dns_v1`,
   `taylor_couette_mhd_dns_v1`) — neither captures the saturated end state. jaxfun
   runs on GPU (verified: this box is an NVIDIA RTX 5090 Laptop GPU, `jax 0.10.1`,
   `default_backend='gpu'`), so the saturated runs are migrated here and executed on
   GPU, with the committed DNS goldens as the rung-2 pre-check.

Use jaxfun for:

- the heavy GPU DNS production runs migrated from shenfun (Phase J5 inventory);
- GPU-accelerated DNS and linearized workflows;
- differentiable objectives, minimal-seed / optimization / sensitivity work;
- production runs promoted from verified shenfun goldens (cheap-golden parity now,
  DNS goldens generated on the GPU run and back-compared on CPU where cheap).

Do **not** treat jaxfun as the first source of truth for *new* physics. New
physics is still implemented and verified in shenfun first. But the physics that
shenfun already specified (all four geometries, hydro/MHD/MRI as supported) is no
longer "later" — it is the active core of this plan.

## Shenfun currency baseline and the primitive-variable rewrite (updated 2026-06-09)

**Read this first.** An earlier revision of this plan was written against shenfun
commit `1507c20` (the commit stamped in the original 9 cheap goldens). shenfun has
since advanced to HEAD `0e19d19` (2026-06-09) with a burst of DNS work that
**invalidates several premises** of the older text. The jaxfun fork (`753650f`,
2026-06-05) predates this and now lags shenfun. Concretely:

1. **PCF MHD/MRI DNS was rewritten from vector potential to primitive variables.**
   shenfun's canonical plane-Couette / shearing-box MHD DNS is now
   `demo/pcf_mri_primitive.py` — classes `AxisymmetricPCFMRIDNS` (ky=0 channel
   mode) and `PCFMRIDNS` (full 3D, ky!=0) — evolving the magnetic field `b`
   **directly** in primitive variables `(u_x,u_y,u_z,p,b_x,b_y,b_z)`, with a
   CN(viscous/resistive + Coriolis + shear + imposed-field) / AB2(quadratic + EMF)
   CNAB2 step and a coupled velocity-pressure+magnetic saddle point solved per `k_z`
   via `BlockMatrixSolver`. shenfun describes it as *"the Cartesian analogue of the
   (validated) cylindrical `taylor_couette_dns.TaylorCouetteMRIDNS`"*. Motivation:
   the old vector-potential solver (`pcf_mhd_divfree.py`, evolving `A` with
   `B=curl A`) makes eigenmode seeding a delicate gauge problem; evolving `b`
   directly lets the DNS reproduce the linear eigenvalue to spectral accuracy via a
   direct block-copy of the eigenvector. The vector-potential `pcf_mhd_divfree.py` /
   `pcf_mhd_mri_shearpy.py` are retained (the latter still used for some 3D
   shearing-box runs) but are **no longer the golden DNS path**. shenfun also added
   primitive linear tooling that jaxfun should mirror for apples-to-apples linear
   parity: `pcf_galerkin_linear.py` (`PlaneCouetteGalerkinLinear`, primitive
   Galerkin generalized eigenproblem, same primitive vars as the TC linear solvers)
   and `pcf_imexrk_linear.py` (IMEXRK time-stepper, the PCF counterpart of
   `taylor_couette_imexrk.py`).

2. **jaxfun's PCF MHD/MRI ports are the superseded vector-potential formulation.**
   `examples/pcf_mhd_jax.py` ("evolves a magnetic vector potential A ... recomputes
   B=curl(A)") and `examples/pcf_mhd_mri_shearpy_jax.py` (which extends it) mirror
   the OLD `couette/pcf_mhd_divfree.py` / `pcf_mhd_mri_shearpy.py`. They do **not**
   match shenfun's current primitive DNS golden. **The fix is not "harden the
   existing vector-potential code."** It is to add a primitive-`b` PCF MHD/MRI DNS
   that is the Cartesian analogue of jaxfun's OWN already-primitive TC MHD DNS
   (`examples/taylor_couette_dns_jax.py`, `AxisymmetricMRIDNSJax`, which already
   evolves `b` directly). This is exactly the move shenfun made; jaxfun has the TC
   primitive pattern in hand and must replicate it for the Cartesian/shearbox case.

3. **DNS goldens now exist — there is no "no DNS golden anywhere" gap, and no new
   golden schema is needed.** shenfun committed four `nonlinear-DNS` goldens, all at
   `schema_version=1` / `physics-regression` (the SAME envelope as the cheap
   goldens, with appropriate per-scalar tolerances): `pcf_hydro_primitive_dns_v1`,
   `pcf_mri_primitive_dns_v1`, `taylor_couette_hydro_dns_v1`,
   `taylor_couette_mhd_dns_v1`. Each is a *linear-regime DNS validation*: it seeds
   the leading eigenmode at tiny amplitude and records that the MEASURED DNS
   growth/decay rate matches the linear eigenvalue to ~1e-6..3e-8 (the
   "growth-vs-linear" / "decay-vs-linear" gate), carrying both `growth_rate`
   (measured, tol ~1e-6) and `growth_rate_linear` (eigenvalue, tol ~1e-10). They
   are **not** saturated-turbulence goldens — heavy nonlinear saturation remains
   deferred on both sides and is still the GPU production target (Phase J5). The
   earlier plan's invented `schema_version=2` / `dns-regression` contract is
   therefore **dropped**: jaxfun matches the committed `schema_version=1` DNS
   goldens directly, exactly as it matches the cheap goldens.

The sections below are updated accordingly. Where an older paragraph still implies
vector-potential PCF MHD or "no committed DNS golden," the updated text supersedes it.

**Local reference copies (vendored 2026-06-09).** The shenfun couette-family
reference demos are now vendored **byte-identically** into `fork_jaxfun/couette/`
(synced to shenfun HEAD `0e19d19`), so the implementing agent reads the port source
in-repo without a sibling shenfun checkout. This includes the new
`couette/pcf_mri_primitive.py`, `couette/test_pcf_mri_primitive.py`,
`couette/pcf_galerkin_linear.py`, `couette/pcf_imexrk_linear.py`,
`couette/_demo_utils.py`, plus the refreshed `couette/taylor_couette_dns.py`,
`couette/_pcf_linear.py`, `couette/_linear_analysis.py`,
`couette/pcf_mhd_mri_shearpy.py`, and the `.md` notes. These files still
`import shenfun` and run only in the shenfun conda env — they are **read references
for porting**, not jaxfun-runnable; the native GPU ports live in `examples/*_jax.py`.
Below, a `couette/<file>` path is the in-fork vendored copy of the corresponding
shenfun `demo/<file>` (identical content); `pipe_flow_dns.py` was **not** vendored
(it is outside the couette mirror — read `fn_shenfun/demo/pipe_flow_dns.py` directly).

## Required References

Keep `solver_survey.md` available to the agent. It is **required as a convention
and parity reference**, especially Part 0, Part I.C, Part IV, Part V, and Appendix A.
For the still-deferred heavy *saturated* runs (no committed golden), see
`solver_survey.md:2449` for the DNS validation model (growth-factor binary checks,
stress-sign checks, `div(b) < 1e-10`, dt-tightening). NOTE: shenfun's four committed
**DNS goldens** do **not** use a separate model — they reuse the cheap goldens'
`schema_version=1` / `physics-regression` per-scalar absolute tolerances (Phase J2),
so no new golden schema is built. Only the additional saturated-run checks are
asserted in the runner (Phase J5).

Important constraint: use `solver_survey.md` as a map, not as proof. Appendix A
marks claims as `VERIFIED` / `PARTIAL` / `UNSUPPORTED`; any `PARTIAL` claim must be
checked against current source before implementation.

Also keep available, and treat as the source of truth for specs, goldens, and
promotions:

- `PLAN_shenfun_production_ready.md`
- `fn_shenfun/demo/production/` — the neutral spec contract, schema, examples,
  goldens, promotions, runner, observables, verification.
- `fn_shenfun/demo/production/README.md` — the shenfun support matrix.

jaxfun source root:

- `shenfun_jaxfun_spectralDNS/fork_jaxfun/`

Runtime:

- `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/.venv/bin/python`
  (CUDA jaxlib build; GPU is live by default; `import jaxfun` enables x64).

## The Shared Contract jaxfun Must Consume (no longer optional, no longer later)

The neutral problem spec lives **entirely in the shenfun repo** under
`fn_shenfun/demo/production/`. jaxfun must consume it as-is. Concretely:

- **Schema:** `fn_shenfun/demo/production/schemas/problem_spec.schema.json`
  (draft 2020-12). 22 required top-level keys: `family` (const `shenfun`),
  `problem_id` (`^[a-z0-9][a-z0-9_.-]*$`), `geometry` in
  `{pcf, channel, taylor_couette, pipe}`, `physics` in `{hydro, mhd, mri}`,
  `support_state` in `{production, experimental, unsupported}`, `formulation`,
  `evolved_variables`, `diagnostic_variables`, `canonical_axes` (requires
  `{x,y,z}`), `native_axes` (requires `{axis_0,axis_1,axis_2}`),
  `nondimensional_groups`, `boundary_conditions` (requires `velocity`; magnetic
  required for mhd/mri), `domain`, `resolution`, `time` (requires
  `{integrator, dt, final_time}`), `initial_condition`, `forcing`, `diagnostics`,
  `expected_oracle` (requires `{type, source}`), `tolerance_model` (requires
  `{kind, scalars}`), `golden` (requires `{artifact_id, regeneration_command}`),
  `unsupported_subcases`.

- **Stricter Python validator:** `fn_shenfun/demo/production/problem_spec.py`.
  Integrators `{analytic, IMEXRK222, CNAB2, linear_eigenproblem}`. Magnetic BCs
  `{None, conducting, insulating, pseudo_vacuum, dirichlet}`. Cross-field
  invariants: `Pm == Rm/Re` (rel_tol 1e-10), `Re,Rm,Pm > 0`, TC `radius_ratio`
  in `(0,1)`, MHD/MRI must declare `boundary_conditions.magnetic`. `spec_hash` =
  SHA256 over sorted-key JSON minus the `spec_hash` field.

- **Unsupported-subcase rejection rules** (jaxfun must mirror these, rejecting
  before any JAX compilation): pipe MHD/MRI rejected
  (`"pipe MHD/MRI is unsupported in shenfun production specs"`); channel MHD/MRI
  deferred (`"channel MHD/MRI is deferred; use pcf MHD/MRI production specs"`);
  MRI only on `pcf` + `taylor_couette`; PCF MHD/MRI conducting walls only; TC
  MHD/MRI conducting or insulating only; TC insulating only axisymmetric `m=0`
  with nonzero `kz`. Two committed negative examples live at
  `fn_shenfun/demo/production/examples/unsupported/`.

- **Goldens (the targets to match):** **13** committed `golden/golden.json` files
  under `fn_shenfun/demo/production/goldens/<id>/golden/golden.json`, each
  `schema_version=1` with `artifact_id`, `problem_id`, `spec_hash`, `environment`,
  `git`, `source_anchors`, `tolerance_model`, `diagnostics{scalars, time_series}`,
  `comparison_fields{scalars_sha256}`. The support matrix tags each with a **golden
  method**:
  - **9 cheap goldens** (`analytic-oracle` or `linear-eigenproblem`,
    `final_time=0.01`, byte-reproducible): `pcf_hydro_laminar_v1`,
    `channel_poiseuille_hydro_v1`, `pcf_mhd_conducting_v1`, `pcf_mri_shearbox_v1`,
    `taylor_couette_hydro_v1`, `taylor_couette_mhd_conducting_v1`,
    `taylor_couette_mhd_insulating_v1`, `pipe_hagen_poiseuille_v1`,
    `pipe_womersley_v1`. (The two pipe goldens are themselves DNS-*measured* by
    `pipe_flow_dns.py` seeded from the exact laminar/Womersley IC, but are cheap.)
  - **4 nonlinear-DNS goldens** (measured from a real time-integrated primitive
    DNS, still `schema_version=1` / `physics-regression`):
    `pcf_hydro_primitive_dns_v1`, `pcf_mri_primitive_dns_v1`,
    `taylor_couette_hydro_dns_v1`, `taylor_couette_mhd_dns_v1`. Each is a
    linear-regime DNS validation: tiny-amplitude eigenmode seed; the golden records
    `growth_rate` (measured, tol ~1e-6) == `growth_rate_linear` (eigenvalue, tol
    ~1e-10) to spectral accuracy, plus energies and measured divergence. They do
    **not** capture nonlinear saturation (that remains deferred — Phase J5).
  The comparator targets `golden/golden.json` (the rich file with
  `diagnostics.scalars` + `tolerance_model.scalars` +
  `comparison_fields.scalars_sha256`), **not** the slim `metadata.json`, for both
  cheap and DNS goldens.

- **The scalar keys are geometry/physics-conditional AND differ between cheap and
  DNS goldens** (verified by reading the committed goldens — jaxfun observables and
  the comparator must honor this, never assume a uniform set):
  - cheap hydro goldens (`pcf_hydro_laminar_v1`, `channel_poiseuille_hydro_v1`,
    `taylor_couette_hydro_v1`, `pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`)
    carry `divergence_l2`;
  - cheap PCF MHD/MRI goldens (`pcf_mhd_conducting_v1`, `pcf_mri_shearbox_v1`) carry
    **both** `divergence_u_l2` and `divergence_b_l2`;
  - cheap TC MHD goldens (`taylor_couette_mhd_conducting_v1`,
    `taylor_couette_mhd_insulating_v1`) carry **only** `divergence_b_l2`;
  - **DNS goldens drop the `_l2` suffix**: `pcf_hydro_primitive_dns_v1` carries
    `divergence_u`; `pcf_mri_primitive_dns_v1` and `taylor_couette_mhd_dns_v1` carry
    `divergence_u`+`divergence_b`; `taylor_couette_hydro_dns_v1` carries
    `divergence_linf`. The comparator must compare exactly the keys the resolved
    golden carries.

- **Promotion records:** 9 `fn_shenfun/demo/production/promotions/<id>.md`
  (dated 2026-06-07), validated by
  `fn_shenfun/demo/production/check_promotions.py`. Uniform jaxfun-feature
  requirement: "jaxfun must implement the same geometry, canonical axes,
  diagnostics, and tolerance model; no live shenfun import is required."

**No-import rule (unchanged, enforced):** the in-process jaxfun pipeline must not
`import shenfun`. The jaxfun library (`src/jaxfun`) and the JAX ports
(`examples/`) currently contain zero real `import shenfun`; the only hits are in
the verbatim `couette/` shenfun reference demos (run only inside the separate
shenfun conda env) and in subprocess source strings in `tests/_parity.py`. Keep it
that way. The production comparator reads committed golden JSON; it does not import
shenfun.

### Golden-path resolution policy (so the comparator has a concrete source)

The shenfun goldens are committed **only** in `fn_shenfun/demo/production/goldens/`;
jaxfun commits none and there is no link today. Pick **one** policy and implement
it (the agent must not leave this unresolved — `compare_goldens.py` needs a source):

- **Vendored copy (default, recommended):** copy the 9 `golden/golden.json` files
  (plus their `spec.json`) into
  `shenfun_jaxfun_spectralDNS/fork_jaxfun/production/goldens/<id>/` and commit them.
  This makes jaxfun parity reproducible without the sibling shenfun checkout, and
  is the only policy compatible with the no-shenfun-import rule on a clean CI.
  Record the source commit (`1507c20..`, branch `main`) in
  `production/goldens/PROVENANCE.json`.
- **Env-var path (fallback):** read goldens from `$SHENFUN_GOLDENS_ROOT` (default
  `../../fn_shenfun/demo/production/goldens`). Only acceptable if the vendored copy
  is impractical; document that CI must have the path populated.

`compare_goldens.py` resolves a golden by `problem_id` -> `<root>/<id>/golden/golden.json`,
where `<root>` is the vendored dir if present else `$SHENFUN_GOLDENS_ROOT`. The
chosen policy and the resolved path are recorded in every run's metadata.

## Shenfun Promotion Gate (REFRAMED: satisfied — parity is now)

The previous revision made this plan "intentionally downstream" and required
physics rows to stay `parity_pending` until shenfun goldens existed. **That gate
is now satisfied.** The shenfun goldens and promotion records exist (paths above).
Therefore:

- PCF and Taylor-Couette parity work is the **immediate, active priority**, not
  deferred. Pipe parity is also active but blocked on a new basis (Phase J4.6).
- The shenfun goldens remain the correctness oracle, and the neutral spec remains
  the shared contract. jaxfun must not invent its own schema; it vendors/loads the
  shenfun one.
- Importing shenfun into jaxfun tests remains **forbidden**. Parity is established
  against committed golden JSON, not against a live shenfun process.

**Updated golden coverage, with a narrowed fallback.** shenfun now provides
committed goldens at two tiers, so most of the migration target is no longer a gap:

- **cheap linear/analytic goldens** for every family (9 of them);
- **primitive-variable DNS goldens** for four families
  (`pcf_hydro_primitive_dns_v1`, `pcf_mri_primitive_dns_v1`,
  `taylor_couette_hydro_dns_v1`, `taylor_couette_mhd_dns_v1`) — committed,
  `schema_version=1`, pinning the measured DNS growth/decay rate to the linear
  eigenvalue. A jaxfun DNS for these families now has a **real committed DNS golden
  to match** (`growth_rate` within ~1e-6, plus the energies and divergence the
  golden carries).

What remains genuinely un-goldened is the **heavy nonlinear saturated** regime
(turbulent SSP, MRI saturation, supercritical TC saturation) and a few explicitly
deferred subcases (full 3D ky!=0 PCF MRI golden; PCF plain-MHD DNS golden;
insulating-wall TC MRI DNS golden; pipe beyond the committed analytic-seeded DNS).
For those, and only those, use this fallback ladder (strongest applicable rung):

1. **Analytic / local oracle where one exists** (local ideal MRI optimum
   `s_max = 0.75*Omega` at `(k vA)^2 = 15/16*Omega^2`; Hagen-Poiseuille flow rate;
   Womersley Bessel profile; epicyclic frequency; Ohmic/Alfven for the *linear*
   decay/oscillation only). Must pass first **where it applies**; a turbulent or
   saturated state has no rung-1 oracle.
2. **Committed shenfun golden for the same family.** Prefer the family's DNS golden
   (the four above) when one exists: the jaxfun DNS must reproduce its
   `growth_rate` / `growth_rate_linear` to the golden tolerance at the same linear
   seeding. Else the cheap linear golden, only when its eigenvalue actually bounds
   the DNS quantity. Note `pcf_hydro_laminar_v1` is a *decaying* eigenvalue
   (`growth_rate = -0.0034674`), and `pcf_hydro_primitive_dns_v1` confirms the DNS
   reproduces a decay rate (`-0.41946`) for its own seeded mode — neither bounds a
   *turbulent* SSP state, so the heavy `pcf_fluct_Re400` run is rung-3-only.
3. **Generate the golden on the GPU run itself, then back-compare on CPU**, for the
   still-deferred heavy saturated runs. Run the heavy DNS on GPU, emit a golden in
   the **same `schema_version=1` shenfun format** (extend, do not replace, the
   committed contract; add the saturation series to `time_series`), and run a
   reduced-resolution CPU instance of the same spec to confirm reproducibility and
   CPU/GPU agreement. Where a run is also cheap enough to execute in shenfun, a
   human/agent **may** regenerate the shenfun DNS golden offline (the one place
   shenfun is run outside the jaxfun process) and commit it into
   `fn_shenfun/demo/production/goldens/<id>/golden/golden.json`, then vendor it per
   the resolution policy above.

Record which rung each run used, in the run's output metadata and the support matrix.

## Ambiguity Closure Requirement (updated)

This plan is not complete until jaxfun's support status is machine-checkable for
each shenfun production family:

- `pcf` / plane Couette;
- `channel` (driven KMM);
- `taylor_couette`;
- `pipe`.

The agent must leave a support matrix in
`shenfun_jaxfun_spectralDNS/fork_jaxfun/production/README.md` with one row per
geometry and physics path. For every row, the matrix must state:

- support state: `production`, `parity_pending`, or `unsupported`;
- internal formulation: primitive variables, flux function, vector potential,
  pressure formulation, or mixed formulation;
- the exact jaxfun source file(s) that implement the path (see Phase J4 — every
  non-pipe path already has code; the agent hardens existing files, it does not
  start blank);
- mapping from the neutral shenfun spec into native jaxfun arrays (canonical ->
  native axis transform);
- boundary conditions, including magnetic/electromagnetic wall conventions;
- nondimensionalization and sign conventions;
- shenfun golden artifact id and tolerance model, **or** the explicit DNS-golden
  fallback rung (above) used when no committed golden exists;
- the geometry/physics-conditional divergence observable key(s) the row emits
  (`divergence_l2` hydro / `divergence_u_l2`+`divergence_b_l2` PCF MHD/MRI /
  `divergence_b_l2` TC MHD);
- parity tests, gradient tests where relevant, and restart tests;
- for heavy runs: a link to the Phase J5 production-run inventory entry;
- explicit reason for any unsupported or deferred subcase.

Additionally the matrix must carry, near the top, a **production-run inventory
table** (the Phase J5 table) and a pointer to the **GPU validation scripts**
(Phase J6). Unsupported is acceptable only if schema validation rejects the spec,
the support matrix names the gap, and a test asserts the rejection. Silent partial
support is not allowed.

## Target End State

At the end of this plan, jaxfun should provide:

1. A production package `production/` that consumes the same neutral problem-spec
   format as shenfun, with a loader that mirrors shenfun's rejection rules.
2. Canonical observables matching shenfun golden artifacts within declared
   tolerance, emitting the geometry/physics-conditional divergence keys, plus the
   DNS observables (Reynolds/Maxwell stress, transport alpha, growth factor) for
   the heavy runs.
3. PCF / channel / Taylor-Couette hydro/MHD/MRI parity **passing now** against the
   committed cheap shenfun goldens (the 7 non-pipe cheap goldens), and pipe
   classified per the survey recommendation (build axis-regularity for hydro parity
   now, or tested rejection with the work flagged — see Phase J4.6).
4. **Executed heavy GPU production runs** (Phase J5 inventory), each with logs,
   checkpoints, diagnostics, a golden in shenfun's `schema_version=1` format, and a
   recorded fallback-oracle rung.
5. **One-command GPU validation scripts** (Phase J6) that autodetect device, pin
   x64, run one or all production runs, compare to goldens with per-observable
   tolerances, and emit a single machine-readable results report.
6. JIT-stable, shape-stable execution on CPU and GPU; differentiable objectives
   with finite-difference gradient checks; checkpoint/restart with reproducible
   metadata; a documented sharding-later path.

## Non-Goals

- Do not silently accept pipe specs without a verified implementation. Either
  build the axis-regularity machinery and pass the pipe hydro goldens, or reject
  pipe specs through schema validation with a documented promotion issue (Phase
  J4.6).
- Do not chase TPU / multi-GPU sharding before single-device GPU is reliable.
- Do not port every shenfun research branch; the Phase J5 inventory is the scope.
- Do not introduce shenfun imports into the in-process jaxfun pipeline or tests.
  Use committed shenfun golden artifacts and neutral specs.
- Do not compare wall-bounded MRI growth to the ideal `0.75*Omega` except in the
  local 4x4 oracle; wall-bounded growth compares to a shenfun golden or to the
  DNS-golden fallback.
- Do not apply a fixed `~7x` growth-factor gate to a *saturated* run; the `~7x`
  expectation is a linear-window check only (see Phase J2/J5).

---

## Phase J0: Baseline Run Envelope and Device Capture

### Implement

First, reconcile the plan file location: overwrite
`shenfun_jaxfun_spectralDNS/fork_jaxfun/PLAN_jaxfun_production_ready.md` with this
document (or a one-line redirect to it) so the agent and any future reader follow a
single source.

Then create the production metadata directory (it does not exist yet — confirmed by
the survey):

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/
  README.md          # support matrix + production-run inventory + script pointers
  run_env.json       # captured at first run (see below)
  commands.md
  __init__.py
```

`run_env.json` must record (query live, do not hardcode):

- interpreter path `/home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun/.venv/bin/python`;
- `jax.__version__`, `jaxlib.__version__` (currently 0.10.1 / 0.10.1);
- `jax.default_backend()` and `jax.devices()` (currently `gpu`,
  `[CudaDevice(id=0)]`, NVIDIA RTX 5090 Laptop GPU);
- default dtype policy (x64 enabled at `import jaxfun` via
  `src/jaxfun/__init__.py`; verified by `tests/test_x64_default.py`);
- `XLA_PYTHON_CLIENT_PREALLOCATE` setting;
- GPU-availability detection command and result;
- the golden-path resolution policy in effect (vendored dir or
  `$SHENFUN_GOLDENS_ROOT`) and the resolved root;
- current test commands and known long-running / gated tests (the
  `live_shenfun` and `spmd` markers; x64-gated tests);
- output locations for runs / checkpoints / goldens.

### Verify

```bash
cd /home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun
.venv/bin/python -c "import jax,jaxlib; print(jax.__version__, jaxlib.__version__); print(jax.devices()); print(jax.default_backend())"
.venv/bin/python -m pytest -q tests/test_x64_default.py
.venv/bin/python -m pytest -q tests/couette/test_taylor_couette_linear_jax.py
```

Discover additional quick tests with `rg --files tests examples | rg 'test_.*\.py$'`.

Acceptance gate:

- the fork copy of this plan is reconciled (overwritten or redirected);
- `run_env.json` exists and records backend/device/dtype/versions and the
  golden-path policy;
- x64 default is verified;
- at least one PCF/channel and one Taylor-Couette smoke path runs;
- failures are either fixed or documented with command and traceback.

## Phase J1: Adopt the Shared Problem Contract

### Implement

Consume the shenfun problem-spec schema rather than inventing a JAX-only schema.

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/problem_spec.py
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/adapters.py
```

`problem_spec.py` must vendor `schemas/problem_spec.schema.json` (copy it into
`production/schemas/`) and port the cross-field invariants from
`fn_shenfun/demo/production/problem_spec.py` (Pm == Rm/Re; Re,Rm,Pm > 0; TC
radius_ratio in (0,1); MHD/MRI must declare a magnetic BC; the exact
unsupported-subcase rejection rules listed above) **without importing shenfun**.

`adapters.py` must:

- validate the neutral spec and reject unsupported cases before any solver
  allocation or JAX compilation;
- map `canonical_axes` -> `native_axes` for each geometry (cartesian:
  `axis_0=x` wall-normal, `axis_1=y` streamwise, `axis_2=z` spanwise; TC:
  `axis_0=r` radial, `axis_1=theta` azimuthal, `axis_2=z` axial) and centralize
  the shenfun-rfft conjugate-symmetric Fourier coefficient layout transform that
  is currently handled ad hoc in `tests/couette/test_live_shenfun_parity.py`;
- map nondimensional groups into solver constructor arguments;
- record every convention transformation in output metadata.

Supported first-pass specs (each already has a jaxfun solver — see Phase J4):

- plane Couette hydro; plane Couette MHD; plane Couette MRI/shearing-box analogue;
- channel KMM hydro (driven, `dpdy != 0`);
- Taylor-Couette hydro; Taylor-Couette MHD/MRI (linear + DNS).

Explicitly reject (mirroring shenfun): pipe MHD/MRI; channel MHD/MRI; MRI off
`pcf`/`taylor_couette`; non-conducting-wall PCF MHD/MRI; TC insulating with
`m != 0` or `kz == 0`; and pipe hydro **only if** Phase J4.6 lands on the
rejection branch (otherwise pipe hydro is accepted).

### Verify

For every accepted example spec from `fn_shenfun/demo/production/examples/`:

- loader returns a typed config object;
- native axes and canonical axes are both recorded;
- unsupported specs (including the two committed negative examples) fail before
  JAX compilation with a message naming the geometry/physics and the reason;
- no solver is instantiated from partially validated config.

Acceptance gate:

- jaxfun reads the same neutral config family shenfun uses for goldens;
- the rejection rules are tested and match shenfun's messages in intent.

## Phase J2: Canonical Observables and Golden Comparison

### Implement

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/observables.py
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/compare_goldens.py
```

`observables.py` must emit the same canonical scalar **names** as the shenfun
golden scalars (this name parity is currently unverified and is a real gap), and
critically must emit the **geometry/physics-conditional divergence keys** — not a
uniform set:

- `kinetic_energy`, `magnetic_energy`, `total_energy`;
- divergence (cheap goldens): emit `divergence_l2` for hydro rows
  (PCF/channel/TC/pipe hydro); **both** `divergence_u_l2` and `divergence_b_l2` for
  PCF MHD/MRI; **only** `divergence_b_l2` for TC MHD/MRI. Divergence (DNS goldens):
  the `_l2` suffix is dropped — emit `divergence_u` (PCF hydro DNS),
  `divergence_u`+`divergence_b` (PCF MRI DNS, TC MHD DNS), `divergence_linf` (TC
  hydro DNS). (Verified against the committed goldens. Using the wrong key breaks
  the per-observable comparison because the golden has no such key.)
- `reynolds_stress`, `maxwell_stress_xy` (defined with the leading minus sign,
  matching shenfun's `observables.py`), `transport_alpha`;
- `flow_rate` (pipe `2*pi*int(u*r)`, channel/pcf `int(u)`), `flow_rate_exact`
  (pipe hydro goldens), `wall_flux`, `torque`, `wall_shear_lower/upper`;
- `growth_rate` (measured, `0.5 * slope of log E`), `growth_rate_linear` (the run's
  linear eigenvalue, carried by the DNS goldens for the growth-vs-linear gate),
  `growth_rate_from_energy`;
- `q_shear`, `local_mri_growth`, `local_mri_smax_over_omega`,
  `eigenvalue_real/imag`, `magnetic_bc`, `rayleigh_stable`, `pressure_gradient`.

Map jaxfun's internal diagnostic names (`Epert`, `Etot`, `Emag`, `divu_l2`,
`divb_l2`, ...) onto these golden keys; that mapping must be explicit and tested,
and must select the correct divergence key(s) from geometry+physics.

`compare_goldens.py` (CLI + importable) must:

- resolve the golden by `problem_id` per the golden-path resolution policy
  (vendored `production/goldens/<id>/golden/golden.json`, else
  `$SHENFUN_GOLDENS_ROOT`);
- load `golden/golden.json` (schema_version 1), read `diagnostics.scalars`,
  `tolerance_model.scalars`, and `comparison_fields.scalars_sha256`;
- verify `scalars_sha256` and `spec_hash`;
- assert **per-observable** absolute tolerances from the golden metadata — never a
  single global tolerance — comparing only the keys the golden actually carries
  (so a hydro comparison never demands `divergence_b_l2`);
- on failure, report observable, expected value, actual value, tolerance, and
  convention metadata;
- not import shenfun (the logic in shenfun's `goldens.py:scalar_differences` /
  `validate_golden` is a reference, re-implemented here).

The committed DNS goldens (`pcf_hydro_primitive_dns_v1`, `pcf_mri_primitive_dns_v1`,
`taylor_couette_hydro_dns_v1`, `taylor_couette_mhd_dns_v1`) are **also**
`schema_version=1` / `physics-regression` and compare through the **same**
per-scalar absolute-tolerance path — there is no separate schema and no "DNS
branch." Their scalar set is `growth_rate` (measured DNS, tol ~1e-6),
`growth_rate_linear` (eigenvalue, tol ~1e-10), `kinetic_energy`, `magnetic_energy`,
and the `_l2`-free divergence keys (`divergence_u` / `divergence_b` /
`divergence_linf`). The parity assertion for a jaxfun DNS on these families is: the
jaxfun-measured `growth_rate` matches the golden to its tolerance, and equals the
run's own linear eigenvalue (the growth-vs-linear / decay-vs-linear gate that the
vendored `couette/test_pcf_mri_primitive.py` and `couette/test_taylor_couette_dns.py`
enforce).

**Heavy saturated runs (still deferred, no committed golden — Phase J5 rung 3)**
generate their own `schema_version=1` golden and additionally check saturation
behavior. Do not conflate the two growth checks:

- *Linear-window quantitative check* (the regime the committed DNS goldens live in):
  the measured DNS growth rate matches the linear eigenvalue,
  `|gamma_dns - Re(s_lin)| < 2e-3*|s_lin|` (and, where seeded into the exponential
  window, an energy-growth factor consistent with `exp(2 s_lin Δt)`).
- *Saturation-run binary check* (heavy runs only): `E_mag[-1] > 2*E_mag[0]` **AND**
  monotone-increasing-then-plateau; the saturation level is a **qualitative** check.
  A saturated run grows far more than any fixed factor before saturating — never
  apply a fixed linear-window growth factor to a saturated run.

### Verify

Add tests that load committed shenfun goldens and compare a jaxfun run on the same
spec — both a cheap golden (`pcf_hydro_laminar_v1`, `channel_poiseuille_hydro_v1`,
`pcf_mhd_conducting_v1`, `taylor_couette_hydro_v1`) and a DNS golden
(`taylor_couette_hydro_dns_v1` is the lowest-risk start since jaxfun's TC DNS is
already primitive; then `pcf_mri_primitive_dns_v1` once the primitive PCF port
exists). Assert the cheap hydro comparison uses `divergence_l2`, the cheap PCF MHD
comparison uses `divergence_u_l2`+`divergence_b_l2`, and the DNS comparison uses the
`_l2`-free keys plus the `growth_rate` / `growth_rate_linear` pair.

Acceptance gate:

- failed comparison reports observable / expected / actual / tolerance / convention;
- comparison does not import shenfun;
- observable-name mapping jaxfun-internal -> golden-key is tested, including the
  conditional divergence-key selection;
- the golden-path resolution policy is exercised (the comparator finds a vendored
  golden by `problem_id` with no shenfun checkout present).

## Phase J3: Production Runner API

### Implement

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/run_problem.py
```

Interface (mirrors shenfun's `run_problem.py`):

```bash
cd /home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun
.venv/bin/python production/run_problem.py \
  --config path/to/problem.json \
  --out runs/<problem_id>/<timestamp> \
  --compare-golden \
  [--shenfun-golden path/to/golden.json] \
  [--write-golden] [--device auto|cpu|cuda] [--steps N] [--checkpoint-every K]
```

(`--shenfun-golden` is optional: if omitted, the golden is resolved by
`problem_id` per the golden-path policy.)

Runner responsibilities:

- validate spec via Phase J1 loader;
- instantiate the solver through the adapter (Phase J4 file map);
- autodetect device and pin x64 (delegates to Phase J6 device logic);
- compile / warm up deterministic kernels (one untimed warm-up step);
- run a fixed-shape time loop (shapes constructor-baked so no per-step recompile);
- emit diagnostics (`diagnostics.jsonl`), checkpoints (via `src/jaxfun/io`), and a
  final golden (`golden/golden.json`) when `--write-golden`, always in shenfun's
  `schema_version=1` format (cheap and DNS goldens share it); heavy saturated runs
  additionally record the saturation series in `time_series` (Phase J5);
- compare to a golden when `--compare-golden` is given (resolved per policy);
- record hardware and JAX metadata into the output (backend, device, dtype,
  versions, spec hash, fallback-oracle rung, golden-path policy).

This runner is the single entrypoint for both the cheap parity runs and the heavy
Phase J5 runs (driven by `--steps` / `final_time` from the spec).

### Verify

End-to-end tiny run test: load config, run CPU, optionally run GPU if present,
write output, compare scalar observables to a tiny golden, restart from the
checkpoint and confirm bit-identical continuation (the `src/jaxfun/io`
checkpoint/restart is already tested as a library — wire it here).

Acceptance gate:

- a new agent can run a jaxfun production problem from config alone, no example
  script reading required;
- checkpoint emission and restart are exercised from the runner, not only the
  library tests.

## Phase J4: Parity Work Order (NOW ACTIVE — harden existing code)

Implement parity in this strict order. **Every non-pipe path below already has
jaxfun code**; the agent hardens it against the shenfun goldens and wires it into
the Phase J3 runner. Each subsection names the exact files and the current state
from the capability survey.

These 7 non-pipe cheap goldens are the "parity now" set: `pcf_hydro_laminar_v1`,
`channel_poiseuille_hydro_v1`, `pcf_mhd_conducting_v1`, `pcf_mri_shearbox_v1`,
`taylor_couette_hydro_v1`, `taylor_couette_mhd_conducting_v1`,
`taylor_couette_mhd_insulating_v1`. The two pipe goldens
(`pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`) are **conditional** on Phase J4.6
landing the axis-regularity basis; until then they are rejection-tested, not run.

### J4.1 PCF / Channel Hydro — state: implemented+tested

Files: `examples/channelflow_kmm.py`, `examples/pcf_fluctuations_jax.py`,
`examples/pcf_fluctuations_divv_jax.py`, `examples/pcf_linear_jax.py`,
`couette/ChannelFlow.py`. Tests: `tests/couette/test_pcf_fluctuations_jax.py`,
`tests/couette/test_linear_analysis_jax.py`,
`tests/couette/test_live_shenfun_parity.py`.

Formulation: KMM velocity-vorticity (wall-normal `u` on clamped biharmonic basis,
wall-normal vorticity `g` on Dirichlet basis, horizontal velocity reconstructed
from incompressibility), fluctuations about `Ub = U_wall*x` (`dpdy=0`),
IMEXRK222/IMEXRK3, pressure via Poisson.

Channel note: the `dpdy` driving term is wired in
(`channelflow_kmm.py:74,232-236,411`) but **never instantiated with `dpdy != 0`**
and has no standalone example/test (state: partial). For channel parity, add a
driven-channel run (`dpdy != 0`) and compare to `channel_poiseuille_hydro_v1`.

Observable note: both `pcf_hydro_laminar_v1` and `channel_poiseuille_hydro_v1`
carry `divergence_l2` (hydro key). Emit that, not `divergence_u_l2`.

Verification: laminar plane Couette / channel profile; divergence floor; temporal
order smoke; golden comparison (`pcf_hydro_laminar_v1`,
`channel_poiseuille_hydro_v1`); CPU vs GPU scalar agreement where GPU is available.

Gate: PCF and driven-channel hydro match shenfun goldens within declared
truncation tolerance.

### J4.2 PCF MHD — state: linear parity OK; DNS path is vector-potential (lags shenfun)

Files: `examples/pcf_mhd_jax.py`, `couette/pcf_mhd_divfree.py`,
`couette/pcf_mhd_divfree_notes.md`. Tests: `tests/couette/test_pcf_mhd_jax.py`,
`tests/couette/test_live_shenfun_parity.py`.

Formulation: magnetic vector potential `A` in `TD^3`, advanced by
`dA/dt = U x B + eta*lap(A)`; `B = curl(A)`, `J = curl(B)` recomputed on demand so
`div(B) = div(curl(A)) = 0` by construction; Lorentz `= cross(J,B)`,
EMF `= cross(U,B)`; Rm/eta; Lorentz prefactor 1; `Rm = Re*Pm`; conducting walls
`A = 0`.

Currency note (2026-06-09): the cheap golden `pcf_mhd_conducting_v1` is a
*linear-eigenproblem* golden, so the existing jaxfun linear MHD path can match it
and cheap PCF MHD parity is unaffected by the formulation change. But this DNS path
(`examples/pcf_mhd_jax.py`) evolves a vector potential `A`, which shenfun has
abandoned for the DNS path in favor of primitive `b` (see the currency section and
J4.3). There is **no** committed PCF plain-MHD *DNS* golden yet (deferred), so no
DNS parity is required for J4.2 specifically; the primitive-`b` solver built for
J4.3 is the vehicle if a PCF-MHD DNS golden is later added.

Observable note: `pcf_mhd_conducting_v1` carries **both** `divergence_u_l2` and
`divergence_b_l2`. Emit both.

Verification: `div(B)` floor (x64); Ohmic decay; Alfven phase; magnetic/kinetic
energy diagnostics; golden comparison (`pcf_mhd_conducting_v1`, linear).

Gate: all MHD scalar oracles pass on CPU and GPU.

### J4.3 PCF MRI / Rotation-Shear — state: vector-potential port LAGS; primitive-`b` re-port required (primary gap)

This is the headline of the 2026-06-09 currency update. shenfun's canonical PCF MRI
DNS is now the **primitive-`b`** `couette/pcf_mri_primitive.py` (vendored from
shenfun `demo/pcf_mri_primitive.py`; classes `AxisymmetricPCFMRIDNS` ky=0,
`PCFMRIDNS` full 3D), the Cartesian analogue of the cylindrical TC MHD DNS. jaxfun's `examples/pcf_mhd_mri_shearpy_jax.py` (extending
the vector-potential `examples/pcf_mhd_jax.py`) is the **superseded** formulation.

Existing (old) jaxfun files: `examples/pcf_mhd_mri_shearpy_jax.py`,
`couette/pcf_mhd_mri_shearpy.py`, `couette/pcf_mhd_mri_notes.md`. Tests:
`tests/couette/test_pcf_mhd_mri_shearpy_jax.py` (one-step finiteness only),
`tests/couette/test_live_shenfun_parity.py`.

Required work — write a primitive-`b` PCF/shearbox MRI DNS in jaxfun, e.g.
`examples/pcf_mri_primitive_jax.py`, by porting the vendored
`couette/pcf_mri_primitive.py` **and reusing jaxfun's own already-primitive TC MHD
DNS pattern**
(`examples/taylor_couette_dns_jax.py`, `AxisymmetricMRIDNSJax`, which evolves `b`
directly): primitive `(u_x,u_y,u_z,p,b_x,b_y,b_z)`; Coriolis `2*Omega`, linear
shear base `Ub = -S*x e_y`, omega-effect `-S b_x`, imposed vertical `B0`; total
pressure; conducting magnetic walls (`b_x=0` Dirichlet, `d_x b_y = d_x b_z = 0`
Neumann); CNAB2 (CN linear, AB2 quadratic+EMF) coupled velocity-pressure+magnetic
saddle point per `k_z`. The payoff (shenfun's reason for the rewrite): the linear
MRI eigenmode is injected by a direct block-copy, so the DNS growth rate matches the
linear eigenvalue to spectral accuracy — making `pcf_mri_primitive_dns_v1` an
achievable parity target. `q_shear = S/Omega`, `kappa2 = 2*Omega*(2*Omega - S)`.
Wall-bounded (no-slip radial), not shearing-periodic. This is the ECS / MRI-dynamo
target.

Observable note: cheap `pcf_mri_shearbox_v1` carries `divergence_u_l2` +
`divergence_b_l2`, `local_mri_growth`, `local_mri_smax_over_omega` (=0.75),
`q_shear` (=1.5), `maxwell_stress_xy`. The DNS golden `pcf_mri_primitive_dns_v1`
carries the `_l2`-free `divergence_u` + `divergence_b`, plus `growth_rate`
(=0.44239) == `growth_rate_linear` (=0.44239), `kinetic_energy`, `magnetic_energy`,
`magnetic_bc` (=conducting).

Verification: (1) local ideal MRI dispersion (`s_max = 0.75*Omega` at
`(k vA)^2 = 15/16*Omega^2`, 4x4 oracle); epicyclic oscillator; shear-winding.
(2) cheap linear parity: wall-bounded MRI growth vs `pcf_mri_shearbox_v1`
(`growth_rate`). (3) primitive DNS parity: the new primitive-`b` solver reproduces
`pcf_mri_primitive_dns_v1` `growth_rate` / `growth_rate_linear` (growth-vs-linear
gate, tiny-amplitude seed) on CPU and GPU.

Gate: a primitive-`b` PCF MRI DNS exists in jaxfun and matches both the cheap linear
golden and the `pcf_mri_primitive_dns_v1` DNS golden. Do not compare wall-bounded
growth to ideal `0.75*Omega` except in the local 4x4 oracle.

### J4.4 Taylor-Couette Hydro — state: implemented+tested

Files: `examples/taylor_couette_dns_jax.py`, `examples/taylor_couette_linear_jax.py`,
`couette/taylor_couette_dns.py`, `couette/taylor_couette_linear.py`. Tests:
`tests/couette/test_taylor_couette_dns_jax.py`,
`tests/couette/test_taylor_couette_linear_jax.py`,
`tests/couette/test_linear_analysis_jax.py`,
`tests/couette/test_sharding_parity_jax.py`,
`tests/couette/test_live_shenfun_parity.py`.

Formulation: primitive perturbation `(u_r,u_theta,u_z,p)` about exact
circular-Couette base `V = a*r + b/r`; Cartesian tensor-product spaces with
explicit cylindrical `1/r` factors; no-slip Dirichlet velocity; truncated
orthogonal pressure (inf-sup `P_N/P_{N-2}`); coupled velocity-pressure BlockMatrix
solve per Fourier mode; CNAB2. Classes `AxisymmetricTCDNSJax`,
`TaylorCouetteDNSJax` (3D).

Observable note: `taylor_couette_hydro_v1` carries `divergence_l2` (hydro key),
plus `rayleigh_stable` (=false), `eigenvalue_real/imag`, `growth_rate` (=0.371384).

Verification: laminar TC profile; divergence floor; linear onset/growth golden
(`taylor_couette_hydro_v1`); **DNS golden `taylor_couette_hydro_dns_v1`** (the
`AxisymmetricTCDNSJax` run reproduces measured `growth_rate` == `growth_rate_linear`,
divergence key `divergence_linf`); restart; shenfun comparison. This is the
lowest-risk DNS parity in the plan, since jaxfun's TC DNS is already primitive-`b`.

Gate: Taylor-Couette hydro is production-runnable from config and matches both the
linear and the DNS golden.

### J4.5 Taylor-Couette MHD/MRI — state: implemented+tested

Files: `examples/taylor_couette_mri_jax.py`, `examples/taylor_couette_dns_jax.py`,
`couette/taylor_couette_mri.py`, `couette/taylor_couette_notes.md`. Tests:
`tests/couette/test_taylor_couette_mri_jax.py`,
`tests/couette/test_taylor_couette_dns_jax.py`,
`tests/couette/test_live_shenfun_parity.py`.

Formulation: resistive viscous MHD in primitive vars
`q = (u_r,u_theta,u_z,Pi,b_r,b_theta,b_z)`, `Pi` = total pressure, Alfven units,
uniform axial `B0`. Conducting walls (`b_r=0`, `d(r b_theta)/dr=0`, `b_z'=0`) and
insulating `m=0` poloidal flux-function `chi` with Robin vacuum match. DNS classes
`AxisymmetricMRIDNSJax`, `TaylorCouetteMRIDNSJax` (3D). Validated against local WKB
MRI (Keplerian `s_max = 0.75*Omega`) and global resistive onset
(Rudiger 2023: conducting `Rm_min ~ 24.7`, insulating `Rm_min ~ 16.5`).

Observable note: both `taylor_couette_mhd_conducting_v1` and
`taylor_couette_mhd_insulating_v1` carry **only** `divergence_b_l2` (no
`divergence_u_l2`, no `divergence_l2`), plus `magnetic_bc`, `magnetic_energy`,
`total_energy`, `growth_rate` (conducting 0.256286, insulating 0.259951).

Verification: conducting-wall linear MHD/MRI golden
(`taylor_couette_mhd_conducting_v1`); insulating-wall golden
(`taylor_couette_mhd_insulating_v1`, sign-distinguishing); conducting/insulating
comparisons must pin `magnetic_bc` identically; **DNS golden
`taylor_couette_mhd_dns_v1`** — the primitive-`b` `AxisymmetricMRIDNSJax` reproduces
measured MRI `growth_rate` (=0.3404) == `growth_rate_linear` (keys `divergence_u` +
`divergence_b`, conducting walls). **jaxfun's `AxisymmetricMRIDNSJax` is already
primitive-`b` and is the reference pattern for the PCF MRI re-port in J4.3.**

Gate: selected TC MHD/MRI cases match shenfun linear and DNS growth-rate goldens
within tolerance.

### J4.6 Pipe — state: MISSING (decision: build axis-regularity for hydro parity now)

The user wants pipe parity. shenfun pipe is fully implemented, tested, and
production-promoted with two hydro goldens (`pipe_hagen_poiseuille_v1`,
`pipe_womersley_v1`, both `divergence_l2`); pipe MHD/MRI is schema-rejected on both
sides. jaxfun has **zero** pipe support, and — critically — the existing TC radial
machinery does **not** extend to `r=0`:

- The only native-jaxfun cylindrical DNS, `examples/taylor_couette_dns_jax.py`,
  uses `Domain(R1,R2)` with `bc=(0,0)` no-slip at **both** walls. A pipe has a
  wall only at `r=R` and an axis at `r=0`; `bc=(0,0)` on `u_z` would force
  `u_z(0)=0`, which is fatal because Hagen-Poiseuille has its **maximum** on the
  axis. The repo's own `couette/taylor_couette_notes.md` (note 4) states the
  annulus `R1>0` case needs no special-casing "unlike the pipe/disc demos where
  `r=0` makes `m=0` a singular 1-D BVP needing `bc=(None,0)`".
- jaxfun's `BoundaryConditions` (`src/jaxfun/galerkin/composite.py`) is strictly
  two-sided left/right Dirichlet/Neumann/Robin; there is **no** `None`/free
  regularity option and **no** Fourier-Bessel basis. The curvilinear *measure*
  exists (`src/jaxfun/coordinates.py` computes `sqrt_det_g = r`; `inner.py:306`
  multiplies by `system.sg`), so only the measure is reusable.

What shenfun does and jaxfun must replicate
(`fn_shenfun/demo/pipe_flow_dns.py`): a single velocity basis
`FunctionSpace(Nr,'C',bc=(None,0))` (free/regular at axis, no-slip Dirichlet at
wall) for **every** azimuthal mode `m`, with weighted-Galerkin singular penalties
(`-m^2/r^2` scalar; `-(m∓1)^2/r^2` on `u_r ± i u_theta`) selecting the regular
`r^|m|` solution; orthogonal Chebyshev pressure truncated 2 modes; CNAB2 coupled
saddle-point per `(m,kz)`; axial body force `fz` (constant -> Hagen-Poiseuille,
callable -> Womersley).

**Preferred path (build it now), in scope for this cycle if feasible:**

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/src/jaxfun/galerkin/   # add axis-regular radial basis
shenfun_jaxfun_spectralDNS/fork_jaxfun/examples/pipe_flow_dns_jax.py   # new pipe solver
```

1. Add an axis-regularity radial basis = an analogue of shenfun's one-sided-free
   `bc=(None, 0)` (free/regular at `r=0`, Dirichlet at `r=R`). This means
   extending `composite.py`'s `BoundaryConditions` to admit a `None`/free end, or
   adding a Fourier-Bessel basis.
2. Add the `m`-dependent `r^|m|` pole selection via the singular weighted-Galerkin
   penalties (or via the Bessel basis), reusing the existing `sqrt_det_g = r`
   curvilinear measure.
3. Build the pipe DNS (`u_r,u_theta,u_z,p`, Fourier(theta) x Fourier(z) x radial,
   CNAB2 coupled saddle-point per `(m,kz)`, axial body force `fz`).
4. Implement pipe observables (`flow_rate` via curvilinear `inner(1,u_z)/Lz`,
   `flow_rate_exact`, `kinetic_energy`, `divergence_l2`, `forcing_phase`) and
   compare against `pipe_hagen_poiseuille_v1` (tol 1e-10) and `pipe_womersley_v1`
   (tol 1e-8). Use the hydro divergence key `divergence_l2`.
5. Verify axis regularity with a manufactured mixed-`m` probe (shenfun reached
   `1.7e-13`); the `bc=(0,0)` annulus pattern gives garbage `O(20)` and is the
   wrong tool — confirm the new basis reproduces the manufactured solution.

**Fallback (only if the basis work is genuinely infeasible this cycle):** keep
pipe specs rejected by the loader, add a test asserting the rejection names `pipe`
and the missing axis-regularity solver path and the required golden ids
(`pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`), and record in the support matrix
a promotion issue with the concrete missing feature list above. Also reject pipe
MHD/MRI to match shenfun. **If this branch is taken, flag it loudly**: the support
matrix row for pipe must read `parity_pending` (not `unsupported` in the sense of
"never"), with the work itemized.

Gate: pipe hydro is either production-runnable against the two shenfun pipe goldens
(preferred), or explicitly rejected with schema validation, an itemized
support-matrix promotion issue, and a regression test that names the missing basis
work. Pipe MHD/MRI is rejected to match shenfun in both branches.

---

## Phase J5: Production Runs on GPU (the heavy DNS migrated from shenfun)

This is the heart of the migration. shenfun *specified* these heavy nonlinear DNS
runs but left them planned-not-run because CPU compute was the bottleneck. They are
now executed in jaxfun on GPU. Each run below is driven by a neutral spec
(authored under `production/runs/<problem_id>.json`, geometry/physics matching the
shenfun families) and launched through the Phase J3 runner and the Phase J6
validation scripts.

For each run: the canonical observables must be emitted (with the correct
geometry/physics-conditional divergence key); the fallback rung is stated; a golden
is written in shenfun's `schema_version=1` format (see below); and CPU/GPU agreement
is checked at reduced resolution where cheap. Four of these families now have a
committed **linear-regime DNS golden** (`pcf_hydro_primitive_dns_v1`,
`pcf_mri_primitive_dns_v1`, `taylor_couette_hydro_dns_v1`,
`taylor_couette_mhd_dns_v1`) that the jaxfun DNS must reproduce *before* it is pushed
into saturation — that golden is the rung-2 pre-check; the saturated end state is the
new, still-un-goldened work.

### DNS golden format (shenfun already defines it — schema_version=1)

The earlier revision invented a `schema_version=2` / `dns-regression` contract on the
premise that no DNS golden existed. **That premise is now false.** shenfun's four
committed DNS goldens (`pcf_hydro_primitive_dns_v1`, `pcf_mri_primitive_dns_v1`,
`taylor_couette_hydro_dns_v1`, `taylor_couette_mhd_dns_v1`) reuse the **same**
`schema_version=1` / `physics-regression` envelope as the cheap goldens, with
per-scalar **absolute** tolerances appropriate to a measured DNS (e.g. `growth_rate`
~1e-6, `growth_rate_linear` ~1e-10, energies/divergence ~1e-9). jaxfun therefore
matches DNS goldens through the **same** comparator path as cheap goldens (Phase J2)
— no second schema, no `dns-regression` branch.

For the still-deferred **heavy saturated** runs that have no committed golden, the
jaxfun-generated golden (fallback rung 3) stays in this same `schema_version=1`
shape and simply adds:

- the saturation energy history in `diagnostics.time_series` (so a
  monotone-then-plateau check is reconstructable);
- the binary saturation checks asserted in the runner/comparator rather than in the
  golden schema: `E_mag[-1] > 2*E_mag[0]` and monotone-then-plateau; stress signs
  `maxwell_stress_xy = <-B_x B_y> > 0`, `reynolds_stress = <u_x u_y> > 0`; measured
  divergence at the DNS floor.

If a human later regenerates one of these in shenfun (rung 3), it is written by
shenfun's existing `goldens.py` in the same `schema_version=1` format — no shenfun
schema change is needed.

### Inventory

Resolution and key params are pinned to the survey. `divergence` column lists the
exact golden key each row's observables layer must emit.

| problem_id | geometry | physics | jaxfun solver file | resolution (production / start) | key params | integrator / final time | canonical observables (divergence key) | golden to match / fallback rung | GPU cost class |
|---|---|---|---|---|---|---|---|---|---|
| `pcf_fluct_Re400` | pcf | hydro | `examples/pcf_fluctuations_jax.py` | scale up from `N=(32,64,32)` toward `(64,128,64)`+ | Re=400, U_wall=1, nu=1/400, x∈[-1,1], Ly=4π, Lz=2π, pert_amp=0.1, dealias (1,1.5,1.5) | IMEXRK222, dt=0.01, end_time=50 (longer for production) | kinetic_energy, wall_shear_lower/upper, mean shear, growth_rate, **divergence_l2**, RMS profiles, energy spectra, SSP streak/roll | no committed golden; `pcf_hydro_laminar_v1` is a *decaying* laminar eigenvalue (`growth_rate=-0.0034674`) that does **not** bound turbulent SSP growth → **rung 3 only** (generate DNS golden on GPU, back-compare on CPU). shenfun demo ran this to t=50 at N=(32,64,32) (`data/PCF_fluct_Re400.chk.h5`, 1.6 MB) as a back-compare reference | medium-heavy (3D, long) |
| `pcf_mhd_divfree` | pcf | mhd | `examples/pcf_mri_primitive_jax.py` (NEW primitive-`b`, see J4.3; old `pcf_mhd_jax.py` is vector-potential) | `N=(32,64,32)`, scale up | Re=Rm=400 (Pm=1), nu=1/400, eta=1/Rm, imposed Bz, pert_amp=0.1, mag_amp=0.05, conducting walls A=0 | IMEXRK222, dt=0.01, end_time=10 | kinetic_energy, magnetic_energy, total_energy, maxwell_stress_xy, **divergence_u_l2 + divergence_b_l2**, growth/decay rate | no committed golden; linear `pcf_mhd_conducting_v1` decay eigenvalue does not bound the nonlinear Lorentz-coupled state → **rung 3 only** (Ohmic/Alfven rung-1 bounds the *linear* decay only, used as a sanity floor, not parity). shenfun never integrated this (chk t=0) | medium (3D) |
| `exp_PCF_MRI_shearbox_growth` | pcf (shearbox analogue) | mri | `examples/pcf_mri_primitive_jax.py` (NEW primitive-`b` 3D port, J4.3; old shearpy port is vector-potential) | `N=(32,32,32)`, Lx=4, Ly=4, Lz=1 (CI `(16,32,16)`) | Re=Rm=1000 (Pm=1), S=1, Omega=0.6667 (q=3/2 Keplerian), by=0, bz=0.025 (v_A=bz), pert_amp=1e-3 seeding k_y=0,k_z={1,2,3}, conducting walls A=0 | IMEXRK222, dt=0.005, end_time=60 | magnetic_energy + saturation, growth_rate (linear-window only), maxwell_stress_xy (`<-B_x B_y>`), reynolds_stress (`<u_x u_y>`), transport_alpha, butterfly `<B_y>(x,t)`, **divergence_b_l2** | rung 1 (local ideal MRI `s_max=0.75·Omega`) → rung 2 (linear `pcf_mri_shearbox_v1` `growth_rate`, plus committed linear-regime DNS golden `pcf_mri_primitive_dns_v1` as the growth-vs-linear pre-check) → rung 3 (saturated DNS golden, generated on GPU). Growth check: `~7x` over t∈[1,3] **linear window**; saturation (t=60) uses `E_mag[-1]>2·E_mag[0]` + monotone-then-plateau, qualitative — do NOT apply `~7x` at t=60. shenfun placeholders empty | heavy (3D, long, MRI saturation) |
| `tc_supercritical_saturation` | taylor_couette | hydro | `examples/taylor_couette_dns_jax.py` (`AxisymmetricTCDNSJax`) | `Nr=40, Nz=16` (CI), scale up | R1=1, R2=2 (eta=0.5), Omega1=1, Omega2=0, nu=1e-2 ⇒ Re=100 > Re_c~68, kz=3.13, Lz=2π/kz, dealias=1.5, seed eigenmode amp=1e-4 | CNAB2, dt=4e-3, ~80 time units (20×4.0) | kinetic_energy (>1e3× then plateau, rel change <1e-2), radial velocity finite (Taylor vortices), **divergence_l2** (<1e-2), torque | rung 2 (`taylor_couette_hydro_v1` linear onset `growth_rate=0.371384`, plus committed DNS golden `taylor_couette_hydro_dns_v1` growth-vs-linear pre-check) → rung 3 (saturated DNS golden) | medium (axisymmetric; 3D heavier) |
| `tc_mri_nonlinear_saturation` | taylor_couette | mri | `examples/taylor_couette_dns_jax.py` (`AxisymmetricMRIDNSJax` / `TaylorCouetteMRIDNSJax`) | `Nr=40, Nz=24` (CI), scale up | quasi-Keplerian base, B0=0.1 axial, nu=eta=1e-3 (Pm=1), kz=6, Lz=2π/kz, dealias=1.5, seed kz_mode=1 amp=1e-4 | CNAB2, dt=2e-3, ~32 time units (16×2.0) | magnetic_energy (>1e3× then saturate), late-time growth_rate << linear MRI rate (~0.34), maxwell_stress_xy, **divergence_b_l2** (<1e-4) | rung 1 (WKB Keplerian `s_max=0.75·Omega`) → rung 2 (linear `taylor_couette_mhd_conducting_v1`/`_insulating_v1`, plus committed DNS golden `taylor_couette_mhd_dns_v1` growth-vs-linear pre-check) → rung 3 (saturated DNS golden). Saturation uses `E_mag[-1]>2·E_mag[0]` + monotone-then-plateau, not a `~7x` gate | heavy (MRI saturation; 3D heaviest) |

**Config-undetermined placeholder (not an executable row):**
`stab_PCF_MRI_stability`. The survey is explicit that its configuration is
**undetermined** — it was inferred from empty 800-byte placeholder files
(`demo/stab_PCF_MRI_*.h5`) with no recorded driver config, distinguished only as a
presumed "stability" counterpart to the `exp_` growth run. **Do not execute it as a
production run.** Carry it as a placeholder: same shearbox solver
(`examples/pcf_mhd_mri_shearpy_jax.py`), but the `(bz, Lz, N, T)` and the
growth-vs-stability intent must be **confirmed from the original CLI args** before
any run or golden. List it in the support matrix as `config-undetermined` with the
open question, not as a first-class run with an acceptance gate.

Pipe is **not** in this heavy-DNS inventory: the shenfun pipe runs are
analytic-oracle scalar goldens already committed (`pipe_hagen_poiseuille_v1`,
`pipe_womersley_v1`), the compute is trivial (Nr=32, ~10 steps), and the blocker
is the missing axis-regularity basis (Phase J4.6), not GPU compute. Once J4.6 lands
the pipe runs are validated through the Phase J6 scripts like any cheap parity run.

### Implement

- Author `production/runs/<problem_id>.json` neutral specs for the 5 executable
  inventory rows, with `support_state` and `unsupported_subcases` set per the
  shenfun rules. The spec carries the production resolution and `final_time`; a
  `--steps` override and a reduced-resolution variant support the CPU smoke / dev
  path.
- Add the DNS-golden writer (shenfun's `schema_version=1` format, per "DNS golden
  format" above) to the runner's `--write-golden` path for heavy runs; for saturated
  runs append the saturation `time_series` and the binary saturation checks.
- Wire each solver's `main()` to emit checkpoints and snapshots (currently the
  example runners pass `on_checkpoint=None`/`on_snapshot=None`; the
  `src/jaxfun/io` library is tested but unwired). The Phase J3 runner supplies the
  callbacks.
- Resolve the open parameter decisions before the heavy run (these are real gaps
  from the survey): the exact production `(N, T)` for `pcf_fluct_Re400` and
  `pcf_mhd_divfree`; the MRI `(bz, Lz, N, T)` tuple for a saturated state (notes
  suggest raising `bz` toward 0.1 and growing `Lz`); the `PCF_MHD_A` vs
  `PCF_MHD_Ac` variant distinction; and whether a Pm/Rm scan is in scope
  (default: single Pm=1 point per run). The `stab_PCF_MRI_stability` config is a
  separate open question (see placeholder above) and is not part of this set.
  Record the chosen tuple in `run_env.json` and the run metadata.

### Verify

For each executable run:

- the fallback-oracle rung (1/2/3) is recorded in the output metadata, and matches
  the inventory (note: `pcf_fluct_Re400` and `pcf_mhd_divfree` are rung-3-only;
  rung-2 is not available for them);
- rung-1 analytic oracles pass before the heavy run **where they apply** (local
  ideal MRI, Hagen, Womersley, epicyclic, linear onset; Ohmic/Alfven as a linear
  sanity floor for `pcf_mhd_divfree`);
- the heavy GPU run completes, emits a `schema_version=1` golden, checkpoints, and
  diagnostics; where a committed DNS golden exists for the family
  (`pcf_mri_primitive_dns_v1`, `taylor_couette_hydro_dns_v1`,
  `taylor_couette_mhd_dns_v1`), the linear-regime growth-vs-linear pre-check passes
  against it before saturation;
- a reduced-resolution CPU instance of the same spec reproduces the GPU result
  within the golden's per-scalar tolerances (CPU/GPU agreement);
- MHD/MRI runs pass the binary checks: stress-sign, `divergence_b_l2 < 1e-10`, and
  the **correct** growth check — linear-window `~7x` over t∈[1,3] for the early
  growth phase, and `E_mag[-1]>2·E_mag[0]` + monotone-then-plateau for the
  saturation run (never `~7x` at saturation).

Acceptance gate:

- every executable inventory row (5 rows) has an executed GPU run (or, where
  production `(N,T)` is still under decision, a documented decision plus an executed
  reduced-but-representative run) with a `schema_version=1` golden and a recorded
  fallback rung;
- `stab_PCF_MRI_stability` is carried only as a `config-undetermined` placeholder
  with its open question documented — it is **not** required to be executed and is
  not a gate;
- no executable run is left in placeholder state.

## Phase J6: GPU Server Validation Scripts (single-command launch + report)

Build the easy-to-run validation harness so any production run can be launched and
validated on a GPU server with one command, and so the same scripts run here for
development (falling back to CPU smoke, clearly labeled).

### Implement

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/run_problem.py     # Phase J3 entrypoint (single run)
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/validate_gpu.sh    # batch driver
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/Makefile           # convenience targets
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/device.py          # device/x64 autodetect helper
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/report.py          # machine-readable results report
```

`device.py` must:

- read `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES`; if a CUDA device is present,
  select it; otherwise fall back to CPU and set a `degraded=true, mode="cpu_smoke"`
  flag in the run metadata;
- ensure x64 is enabled (it already is at `import jaxfun`, but assert it);
- set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if unset;
- return a structured device record (backend, device string, dtype, prealloc) for
  `run_env.json` and per-run metadata.

`validate_gpu.sh` must accept a run id or `all`, and for each run:

- launch `run_problem.py` with the run's spec, the device autodetect, the
  production resolution (or a reduced smoke resolution if `--smoke` / no GPU);
- write per-run `logs/<problem_id>.log`, `checkpoints/`, `diagnostics.jsonl`,
  `golden/golden.json`, and `metadata.json` under
  `runs/<problem_id>/<timestamp>/`;
- compare to the golden (resolved per the golden-path policy) or run the
  fallback-oracle rung via `compare_goldens.py` with per-observable tolerances;
- append a pass/fail record to a single machine-readable report.

`report.py` (invoked by the batch driver) must emit one
`runs/_report/results.json` (and a short `results.md` summary) with: per-run
`problem_id`, device/backend/mode (`gpu` or `cpu_smoke`), wall time, observables
compared, per-observable pass/fail with expected/actual/tolerance, fallback rung,
and an overall `summary{passed, failed, skipped}`.

`Makefile` targets:

```make
validate-all:    # run every executable production run id, GPU if present else CPU smoke
validate-%:      # run a single problem_id, e.g. make validate-pcf_fluct_Re400
parity-cheap:    # run the 7 non-pipe cheap shenfun-golden parity runs; if J4.6 landed the
                 #   basis, also run the 2 pipe goldens, else the 2 pipe goldens are skipped (rejection-tested)
report:          # regenerate runs/_report/results.{json,md}
```

`parity-cheap` runs the 7 non-pipe cheap goldens unconditionally
(`pcf_hydro_laminar_v1`, `channel_poiseuille_hydro_v1`, `pcf_mhd_conducting_v1`,
`pcf_mri_shearbox_v1`, `taylor_couette_hydro_v1`,
`taylor_couette_mhd_conducting_v1`, `taylor_couette_mhd_insulating_v1`). The two
pipe goldens are added only when Phase J4.6 has landed the axis-regularity basis;
otherwise they are covered by the rejection regression test, not by `parity-cheap`,
and the target reports them `skipped` with the reason.

### Example invocations

```bash
cd /home/nauman/cfd/shenfun_jaxfun_spectralDNS/fork_jaxfun

# Single heavy run on GPU (autodetects CUDA), compares to golden/fallback, writes report:
.venv/bin/python production/run_problem.py \
  --config production/runs/pcf_fluct_Re400.json \
  --out runs/pcf_fluct_Re400/$(date +%Y%m%dT%H%M%S) \
  --device auto --compare-golden --write-golden

# Batch over all executable production runs (GPU server):
production/validate_gpu.sh all

# Force GPU / force CPU:
JAX_PLATFORMS=cuda production/validate_gpu.sh all
JAX_PLATFORMS=cpu  production/validate_gpu.sh all --smoke

# Cheap parity (7 non-pipe goldens; pipe pair only if J4.6 landed):
make -C production parity-cheap

# One run by id, then read the report:
production/validate_gpu.sh pcf_mhd_divfree
.venv/bin/python production/report.py --print runs/_report/results.json
```

### No-GPU behavior (so it is runnable here for development)

When no CUDA device is visible, `validate_gpu.sh` must:

- run a **reduced-resolution CPU smoke** of each requested run (use the spec's
  reduced variant / `--steps` cap), not the full production resolution;
- label every such run `mode="cpu_smoke", degraded=true` in logs and the report;
- still perform golden / fallback-oracle comparison where the reduced run is
  meaningful:
  - cheap parity runs (the 7 non-pipe goldens) compare fully;
  - heavy DNS runs that **have** a rung-1 analytic oracle or rung-2 linear golden
    assert that oracle at smoke scale (`exp_PCF_MRI_shearbox_growth`,
    `tc_supercritical_saturation`, `tc_mri_nonlinear_saturation`);
  - `pcf_fluct_Re400` and `pcf_mhd_divfree` have **no** nonlinear-state analytic
    oracle (rung-3-only), so CPU-smoke asserts only finiteness and the divergence
    floor (`divergence_l2`, or `divergence_u_l2`+`divergence_b_l2` for the MHD run)
    — the report must say so rather than imply parity was checked;
- exit nonzero only on a real comparison failure, not merely because no GPU was
  present (absence of GPU is a labeled fallback, not an error).

### Verify

- `validate_gpu.sh all --smoke` runs end-to-end on this box (CPU fallback labeled
  correctly) and produces `runs/_report/results.json`;
- on a GPU box (or this box's GPU), at least one heavy run completes and the report
  records `mode="gpu"`, wall time, and per-observable pass/fail;
- the report is valid JSON with the documented schema and a correct
  `summary{passed, failed, skipped}`;
- a single intentionally-broken tolerance produces a `failed` entry with
  observable/expected/actual/tolerance.

Acceptance gate:

- a production run can be launched and validated on a GPU server with **one
  command**;
- the same command runs here in clearly-labeled CPU smoke mode;
- a single machine-readable results report summarizes pass/fail per run.

## Phase J7: Autograd Production Hardening

### Implement

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/objectives.py
```

Required objectives (supporting the Phase J5 runs and the ECS/MRI workflows):

- final perturbation energy; time-integrated energy;
- Reynolds/Maxwell stress objective; transport-alpha objective;
- growth-rate proxy where differentiable;
- minimal-seed objective for PCF.

Rules: objectives are pure functions of config/state/control; static arguments
explicit; no hidden global mutable state; gradients flow through the intended
solver path (the Phase J3/J5 step), not a diagnostic shortcut.

### Verify

For small problems: `jax.value_and_grad` finite; finite-difference directional
derivative agrees with AD gradient; gradient stable across JIT/non-JIT where both
exist; no accidental `stop_gradient` or host callback in the objective path.

Acceptance gate:

- every advertised differentiable objective has a finite-difference gradient check.

## Phase J8: JIT/GPU Reliability (supports the production runs)

### Implement

Add performance/reliability instrumentation used by the Phase J3 runner and the
Phase J6 scripts:

- compile-time logging (first untimed warm-up step), per-step runtime logging;
- memory notes where available; shape-stability checks;
- CPU/GPU comparison mode (drives the DNS-tolerance CPU/GPU agreement check);
- explicit dtype policy.

Rules: production loops must not recompile per step (shapes are constructor-baked;
`scan_steps` uses `lax.scan` single-device and an eager loop on multi-device);
dynamic Python branching in JIT loops eliminated or made static; arrays on the
intended device; x64 remains default. Note the time step is not whole-jitted (JIT
is per-operator); this is shape-stable and correct, just many small kernel
launches — acceptable for now, instrument it.

### Verify

- one CPU smoke run; one GPU smoke run if GPU exists;
- repeated run confirms no unintended recompilation (verified live: identical-shape
  solves measured ~0.45s with no recompile);
- CPU/GPU scalar diagnostics agree within tolerance;
- output metadata records backend and device.

Acceptance gate:

- single-device GPU path is stable before any sharding work begins.

## Phase J9: Checkpoint/Restart and Output Schema

### Implement

Wire the existing, tested `src/jaxfun/io` checkpoint library into the production
runner (it is currently unwired — example runners pass `on_checkpoint=None`):

```text
shenfun_jaxfun_spectralDNS/fork_jaxfun/production/checkpoint.py   # thin wrapper over src/jaxfun/io
```

Checkpoint must include: spec hash; solver/schema version; timestep/time; state
arrays (coefficient HDF5); diagnostic history or pointer; dtype/device metadata;
PRNG state if any.

### Verify

Restart test: run to `t_mid`, checkpoint, restart to `t_final`, compare against
uninterrupted run (the library already has a bit-identical restart test —
`tests/io/test_hdf5.py`; here exercise it from `run_problem.py`).

Acceptance gate:

- checkpoint/restart is part of production readiness and is driven by the runner,
  not only the IO library tests; the Phase J5 heavy runs checkpoint at cadence so a
  GPU run can resume after preemption.

## Phase J10: Sharding/TPU Later

Only start after J0-J9 are green on single-device CPU/GPU. The heavy Phase J5 runs
are sized to fit single-GPU first; multi-device is an optimization, not a
correctness requirement.

### Implement

Use existing `src/jaxfun/sharding.py` (1D mesh `'k'`, opt-in `spectral_sharding`
`P('k')` / `physical_sharding` `P(None,'k')`) as the basis for explicit sharding.
Keep it opt-in. Note `all_to_all` requires the split axis divisible by device
count, which the Phase J5 production resolutions must satisfy.

### Verify

- two-device smoke where available (today only simulated CPU devices via
  `--num-devices=2`; bit-identical parity tested there, never on >1 physical GPU);
- same scalar diagnostics as single-device;
- no shape-specific hidden assumptions.

Acceptance gate:

- sharding is not allowed to change physics observables.

## Final Acceptance Checklist

- This plan reconciled with the fork copy at
  `shenfun_jaxfun_spectralDNS/fork_jaxfun/PLAN_jaxfun_production_ready.md`
  (overwritten or redirected) so a single file is authoritative.
- `solver_survey.md` used as required reference and partial claims rechecked; the
  `solver_survey.md:2449` DNS validation model applied to the still-deferred
  saturated runs only — the committed DNS goldens use `schema_version=1` /
  `physics-regression` like the cheap goldens (no separate schema).
- Shenfun promotion gate confirmed **satisfied** and **current to HEAD `0e19d19`**:
  the neutral spec, schema, **13 goldens (9 cheap + 4 nonlinear-DNS), and 13
  promotion records** exist under `fn_shenfun/demo/production/`; the jaxfun support
  matrix is refreshed from them.
- **Currency reconciled (2026-06-09):** the plan reflects shenfun's primitive-`b`
  PCF MHD/MRI DNS rewrite (`demo/pcf_mri_primitive.py`: `AxisymmetricPCFMRIDNS`,
  `PCFMRIDNS`). jaxfun's vector-potential `examples/pcf_mhd_jax.py` /
  `pcf_mhd_mri_shearpy_jax.py` are flagged as lagging; the J4.3 task is a
  primitive-`b` re-port mirroring jaxfun's own primitive TC DNS
  (`AxisymmetricMRIDNSJax`).
- **DNS-golden parity** (committed, `schema_version=1`): jaxfun matches
  `taylor_couette_hydro_dns_v1` (lowest-risk; jaxfun TC DNS already primitive),
  `taylor_couette_mhd_dns_v1`, and `pcf_mri_primitive_dns_v1` (after the J4.3 port)
  on `growth_rate == growth_rate_linear` with the `_l2`-free divergence keys.
- jaxfun consumes the same neutral problem specs as shenfun (loader vendors the
  shenfun schema, mirrors its rejection rules, imports no shenfun).
- Golden-path resolution policy chosen and implemented (vendored
  `production/goldens/` by default), so `compare_goldens.py` has a concrete source
  with no shenfun checkout.
- Observables emit the correct **geometry/physics-conditional divergence key**
  (`divergence_l2` hydro / `divergence_u_l2`+`divergence_b_l2` PCF MHD/MRI /
  `divergence_b_l2` TC MHD), and the comparator only compares keys the golden
  carries.
- Production runner exists and is the single entrypoint, not an example script.
- **Parity passes now** against the 7 non-pipe committed cheap shenfun goldens:
  - PCF / driven-channel hydro (`pcf_hydro_laminar_v1`, `channel_poiseuille_hydro_v1`);
  - PCF MHD (`pcf_mhd_conducting_v1`);
  - PCF MRI / rotation-shear against the wall-bounded golden (`pcf_mri_shearbox_v1`);
  - Taylor-Couette hydro (`taylor_couette_hydro_v1`);
  - selected TC MHD/MRI (`taylor_couette_mhd_conducting_v1`, `_insulating_v1`),
    pinning `magnetic_bc` identically.
- The two pipe cheap goldens (`pipe_hagen_poiseuille_v1`, `pipe_womersley_v1`) are
  in `parity-cheap` **only if** Phase J4.6 landed the axis-regularity basis;
  otherwise they are rejection-tested and reported `skipped`.
- **The Phase J5 production-run inventory is executed on GPU**: each of the 5
  executable rows has an executed run (or documented `(N,T)` decision +
  representative run), a `schema_version=1` golden (matching the committed DNS
  golden where one exists for the family), recorded fallback-oracle rung, and
  CPU/GPU agreement where cheap.
  `stab_PCF_MRI_stability` is carried as a `config-undetermined` placeholder, not
  executed, with its open question documented.
- The DNS growth checks are split correctly: linear-window `~7x` over t∈[1,3] for
  early growth, `E_mag[-1]>2·E_mag[0]` + monotone-then-plateau (qualitative) for
  saturation — `~7x` is never applied to a saturated run.
- `pcf_fluct_Re400` and `pcf_mhd_divfree` are recorded as **rung-3-only** (their
  decaying linear goldens do not bound a nonlinear state); the other three
  executable runs use rung 1/2 where the oracle/linear golden applies.
- **The Phase J6 GPU validation scripts exist and work**: one-command launch
  (`validate_gpu.sh all`), device/x64 autodetect, per-run logs/checkpoints/
  diagnostics, per-observable golden comparison, a single machine-readable report,
  and a clearly-labeled CPU-smoke fallback that runs on this box.
- Pipe is either production-runnable against the two shenfun pipe goldens
  (axis-regularity basis built — preferred) **or** rejected by schema validation
  with an itemized `parity_pending` promotion issue naming the missing
  axis-regularity basis, the singular-penalty assembly, and the required golden
  ids, plus a regression test asserting the rejection. Pipe MHD/MRI rejected to
  match shenfun in both branches.
- Ambiguity-closure support matrix exists for plane Couette, channel,
  Taylor-Couette, and pipe, including the production-run inventory table and the
  validation-script pointer.
- Autograd objectives have finite-difference gradient checks.
- CPU/GPU agreement is tested; single-device GPU is stable before sharding.
- Checkpoint/restart is tested and driven from the runner.
- TPU/sharding remains deferred until single-device GPU is stable.
