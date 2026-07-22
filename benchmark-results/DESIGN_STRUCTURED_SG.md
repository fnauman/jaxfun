# Structured Sg factor-memory prototype

The production solver stores banded LU rows for every `(ky, kz)` mode.  Its
memory scales as `n_modes * n_x * (n_L + n_U)`; the upper bandwidth grows with
the Shen/quasi-inverse discretization.  `production.structured_sg_prototype`
instead retains only the unfactored DIA rows, assembles each mode in a compiled
call, and uses a pivoted dense solve.  This is a correctness and memory-floor
prototype, not a candidate default: it trades persistent factor traffic for
large temporary dense matrices and repeated factorization.

The next viable algorithmic milestone is an ultraspherical or Shen recurrence
that preserves the compact operator rows while solving in linear work per
mode.  It must retain the current boundary rows, mean/zero-mode treatment,
complex float64 residuals, multi-RHS lanes, and GPU batching.  Qualification
must compare whole-step time and compiled argument/temporary bytes at 64, 96,
and bounded 128 classes; isolated factor-byte reduction is insufficient.
