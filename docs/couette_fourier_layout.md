# Couette Fourier layout

The jaxfun Couette ports use the existing full-complex Fourier space on every
periodic axis. This differs from the shenfun KMM reference, where the spanwise
axis may use a real-Fourier half spectrum (`dtype="d"`).

Rationale:

- Full-complex FFT layout is already implemented and differentiable in jaxfun.
- The same layout works on CPU, GPU and TPU without an rfft-specific branch.
- The `(0, 0)` KMM mode remains the global `[0, 0]` Fourier coefficient.
- Nyquist filtering is explicit through `TensorProductSpace.mask_nyquist()`.

When comparing to shenfun's real-Fourier output, expand the shenfun half
spectrum to full FFT ordering before coefficient-wise comparisons. Diagnostics
such as energy, divergence and wall shear can be compared directly because they
are computed in physical space.
