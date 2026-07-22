# PCF real/Hermitian periodic-storage prototype

`jaxfun.galerkin.hermitian` keeps the wall-normal coefficient axis unchanged,
uses full complex FFT ordering in `y`, and stores only `kz >= 0` from a real
FFT in `z`.  The shape contract is `(..., Ny, Nz//2+1)`.  Inversion first
performs the complex inverse `y` FFT and then `irfft` in `z`; normalization is
`norm="forward"`, matching the existing Fourier spaces.

Odd derivatives zero the corresponding even-grid Nyquist plane.  Padding
inserts zeros between positive and negative `ky` blocks and extends the
nonnegative `kz` tail; truncation is the exact inverse for masked spectra.
Ownership under sharding should remain on the full `ky` axis initially, so the
half-spectrum axis is local.  A later distributed implementation must define
whether the `kz=0` and Nyquist planes have special owners.

The prototype is intentionally separate from TensorProduct's public
representation.  Unit tests cover round trips, full-complex derivative parity,
Nyquist behavior, dealiased nonlinear-product parity, and the expected storage
reduction.  Production adoption still requires end-to-end PCF/MHD trajectory
and sharding parity plus a measured >=20% rollout or compelling capacity gain.
