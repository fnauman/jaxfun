"""Full 3D Taylor-Couette MHD/MRI DNS with a magnetic vector potential.

The magnetic unknown is the fluctuation vector potential ``A(theta, z, r)``
with total field ``B = B0 e_z + curl(A)``, so ``div B = div(curl A) = 0`` is
an analytic identity of the evolved representation -- the direct counterpart
of the plane-Couette curl workhorse (``examples/pcf_mhd_mri_shearpy_jax.py``)
in cylindrical geometry.  The evolution gauge is the resistive gauge

    dA/dt = (U + u) x (B0 + b) + eta * lap_vec(A),

whose curl is exactly the induction equation for any wall condition on ``A``.
The steady, curl-free base EMF ``U x B0 = U_theta B0 e_r`` is omitted (it is
a pure gauge gradient and would only accumulate a secular curl-free part).

Discretisation mirrors ``examples/taylor_couette_dns_jax.py``: complex
Fourier modes in ``theta`` and ``z``, a Galerkin radial direction, one dense
coupled block per ``(m, kz)`` mode solved by a batched pinned LU, and CNAB2
time stepping (Crank-Nicolson linear terms, Adams-Bashforth-2 nonlinear terms
with an IMEX-Euler bootstrap).  The implicit block couples
``(u_r, u_theta, u_z, Pi, A_r, A_theta, A_z)``: rotation and base advection
for the velocity, the linear Lorentz force ``J(A) x B0 e_z`` (plain-pressure
form), the linear EMFs ``u x B0 e_z`` and ``U x b(A)``, and the eta vector
Laplacian of ``A``.

Magnetic wall conditions (``magnetic_bc``):

* ``conducting`` -- ``A_theta = A_z = 0`` (Dirichlet) and ``(r A_r)' = 0``
  (Robin).  The Robin row is ``div A = 0`` evaluated at the wall, so the
  electrostatic potential of the resistive gauge (``phi = -eta div A``)
  vanishes there and ``E_tang = 0`` holds *exactly*: this is the resistive
  perfectly-conducting condition, on-shell equivalent to the primitive set
  ``{b_r = 0, (r b_theta)' = 0, b_z' = 0}`` (with ``E_tang = eta j_tang`` at
  a no-slip wall).  All three conditions live in fixed composite bases.
* ``insulating`` -- the fluctuation field matches a current-free exterior
  potential field, decaying for ``r > R2`` and regular for ``r < R1``.  Per
  mode ``(m, kz != 0)`` the exterior solutions are ``K_m(|kz| r)`` and
  ``I_m(|kz| r)``, giving two matching rows per wall with modified-Bessel
  ratio coefficients plus the wall gauge ``A_r = 0``; for ``kz = 0, m != 0``
  the potential is ``r^{+-|m|}``; the ``(0, 0)`` mean mode uses
  ``b_theta = 0`` at both walls (no net axial current), ``b_z(R2) = 0``
  (finite exterior energy), and the exact trapped-flux Faraday row
  ``(R1/2) d b_z(R1)/dt = eta d b_z/dr(R1)`` for the inner vacuum column.
  Because the coefficients depend on the mode, ``A`` lives in orthogonal
  radial bases and the boundary rows are imposed tau-style per mode.

The reported ``div b`` witness is measured from the forward-projected
coefficient representation of ``b`` (the representation the current density
and diagnostics use), matching how the primitive family reports it; the
underlying pointwise ``curl A`` field is solenoidal to roundoff by
construction, while the projected witness carries the (spectrally small,
non-growing) quadrature error of the cylindrical ``1/r`` projections.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import sympy as sp
from jax import Array
from scipy.special import ive, kve

try:
    from examples.taylor_couette_dns_jax import (
        AxisymmetricTCDNSJax,
        _cylindrical_laplacian_3d,
        _positive_pivot_phase,
        _require_resolved_m,
    )
    from examples.taylor_couette_linear_jax import CircularCouette
    from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax
except ModuleNotFoundError:  # direct script execution from examples/
    from taylor_couette_dns_jax import (
        AxisymmetricTCDNSJax,
        _cylindrical_laplacian_3d,
        _positive_pivot_phase,
        _require_resolved_m,
    )
    from taylor_couette_linear_jax import CircularCouette
    from taylor_couette_mri_jax import TaylorCouetteMRIJax

from jaxfun import Domain, Dx
from jaxfun.diagnostics import coefficient_wall_linf, cylindrical_energy_parts
from jaxfun.galerkin import (
    CoupledSpace,
    FunctionSpace,
    TensorProduct,
    TestFunction,
    TrialFunction,
)
from jaxfun.galerkin.Chebyshev import Chebyshev
from jaxfun.galerkin.Fourier import Fourier
from jaxfun.galerkin.inner import integrate
from jaxfun.galerkin.Legendre import Legendre
from jaxfun.integrators.cnab2 import ScanRolloutCache, ScanRolloutCacheInfo, cnab2_rhs
from jaxfun.io import Cadence, run_with_cadence

type Velocity = tuple[Array, Array, Array]


def _bessel_i_log_derivative(m: int, x: float) -> float:
    """Return I_m'(x)/I_m(x) using overflow-safe scaled Bessel functions."""
    m = abs(int(m))
    num = 0.5 * (ive(m - 1, x) + ive(m + 1, x)) if m > 0 else ive(1, x)
    return float(num / ive(m, x))


def _bessel_k_log_derivative(m: int, x: float) -> float:
    """Return K_m'(x)/K_m(x) (negative) using scaled Bessel functions."""
    m = abs(int(m))
    num = -0.5 * (kve(m - 1, x) + kve(m + 1, x)) if m > 0 else -kve(1, x)
    return float(num / kve(m, x))


def _require_non_nyquist_kz(kz_mode: int, nz: int, *, allow_zero: bool) -> None:
    """Require an axial Fourier mode that survives Nyquist masking."""

    kz_mode = int(kz_mode)
    nz = int(nz)
    if not allow_zero and kz_mode == 0:
        raise ValueError("eigenmode seeding requires kz_mode != 0")
    if 2 * abs(kz_mode) >= nz:
        raise ValueError(
            f"axial mode |kz_mode|={abs(kz_mode)} is unresolved by Nz={nz}; "
            "require 2|kz_mode| < Nz so Nyquist masking cannot erase the seed"
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TCVPState:
    """Coefficient state for the vector-potential Taylor-Couette DNS."""

    u: Velocity
    p: Array
    A: Velocity
    nonlinear_old: tuple[Array, ...]
    have_old: float | Array = 0.0

    def tree_flatten(self):
        return (
            self.u,
            self.p,
            self.A,
            self.nonlinear_old,
            self.have_old,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        u, p, A, nonlinear_old, have_old = children
        return cls(u=u, p=p, A=A, nonlinear_old=nonlinear_old, have_old=have_old)


class TaylorCouetteVPMRIDNSJax:
    """Full 3D vector-potential Taylor-Couette MHD/MRI DNS.

    See the module docstring for formulation and boundary conditions.  The
    per-mode dense-block machinery follows
    :class:`examples.taylor_couette_dns_jax.TaylorCouetteMRIDNSJax`.
    """

    MAGNETIC_BCS = ("conducting", "insulating")

    def __init__(
        self,
        base: CircularCouette,
        B0: float = 0.1,
        nu: float = 1.0e-3,
        eta_mag: float = 1.0e-3,
        Nr: int = 20,
        Ntheta: int = 8,
        Nz: int = 16,
        Lz: float | None = None,
        dt: float = 2.0e-3,
        family: str = "C",
        dealias: float = 1.5,
        magnetic_bc: str = "conducting",
    ) -> None:
        if magnetic_bc not in self.MAGNETIC_BCS:
            raise ValueError(
                f"magnetic_bc must be one of {self.MAGNETIC_BCS}, got {magnetic_bc!r}"
            )
        self.base = base
        self.B0 = float(B0)
        self.nu = float(nu)
        self.eta_mag = float(eta_mag)
        self.Nr = int(Nr)
        self.Ntheta = int(Ntheta)
        self.Nz = int(Nz)
        self.dt = float(dt)
        self._dt_array = jnp.asarray(self.dt)
        self.family = family.upper()
        self.dealias = float(dealias)
        self.magnetic_bc = magnetic_bc
        self.Lz = float(Lz) if Lz is not None else 2.0 * math.pi / 3.0 * base.gap
        self.Re = base.Omega1 * base.R1 * base.gap / self.nu
        self.Rm = base.Omega1 * base.R1 * base.gap / self.eta_mag
        self.Pm = self.nu / self.eta_mag
        self.Jm = 0.5 * (base.R2 - base.R1)

        family_cls = self._family_class(self.family)
        dom = Domain(base.R1, base.R2)
        self.Ft = FunctionSpace(
            self.Ntheta, Fourier, domain=Domain(0.0, 2.0 * math.pi), name="Ftv"
        )
        self.Fz = FunctionSpace(
            self.Nz, Fourier, domain=Domain(0.0, self.Lz), name="Fzv"
        )
        self.SD = FunctionSpace(self.Nr, family_cls, bc=(0, 0), domain=dom, name="SDv3")
        self.S0 = FunctionSpace(self.Nr, family_cls, domain=dom, name="S0v3")
        self.SP = FunctionSpace(
            self.Nr, family_cls, domain=dom, num_dofs=self.Nr - 2, name="SPv3"
        )
        if magnetic_bc == "conducting":
            # (r A_r)' = 0: u + (r_w / Jm) * du/dX = 0 at each wall, i.e. the
            # same Robin composite used for (r b_theta)' = 0 in the primitive
            # family; A_theta and A_z are Dirichlet.
            self.SAr = FunctionSpace(
                self.Nr,
                family_cls,
                domain=dom,
                bc={
                    "left": {"R": (base.R1 / self.Jm, 0)},
                    "right": {"R": (base.R2 / self.Jm, 0)},
                },
                name="SArv3",
            )
            self.SAt = self.SD
            self.SAz = self.SD
        else:
            self.SAr = self.S0
            self.SAt = self.S0
            self.SAz = self.S0

        self.TD = TensorProduct(self.Ft, self.Fz, self.SD, name="TDvp3")
        self.T0 = TensorProduct(self.Ft, self.Fz, self.S0, name="T0vp3")
        self.TP = TensorProduct(self.Ft, self.Fz, self.SP, name="TPvp3")
        self.TAr = TensorProduct(self.Ft, self.Fz, self.SAr, name="TArvp3")
        self.TAt = TensorProduct(self.Ft, self.Fz, self.SAt, name="TAtvp3")
        self.TAz = TensorProduct(self.Ft, self.Fz, self.SAz, name="TAzvp3")
        self.VQ = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TP, self.TAr, self.TAt, self.TAz),
            name="VQvp3",
        )
        self.VE = CoupledSpace(
            (self.TD, self.TD, self.TD, self.TAr, self.TAt, self.TAz),
            name="VEvp3",
        )

        self.theta, self.z, self.r = self.TD.system.base_scalars()
        self.VQ_mode_indices = AxisymmetricTCDNSJax._mode_indices(self.VQ)
        self.VE_mode_indices = AxisymmetricTCDNSJax._mode_indices(self.VE)
        self.Theta, self.Z, self.R = self.T0.mesh()
        self.inv_r = 1.0 / self.R
        if self.dealias > 1.0:
            self.T0p = self.T0.get_dealiased((self.dealias, self.dealias, self.dealias))
            self.padded_counts = self.T0p.num_quad_points
            _thp, _zp, Rp = self.T0p.mesh()
            self.inv_r_p = 1.0 / Rp
        else:
            self.T0p = None
            self.padded_counts = None
            self.inv_r_p = self.inv_r

        self._build_wall_rows()
        mass_q, phys_q, mass_e, phys_e = self._build_operator_parts()
        extract = AxisymmetricTCDNSJax._extract_mode_matrices
        self._mass_q_modes = extract(mass_q, self.VQ_mode_indices)
        self._phys_q_modes = extract(phys_q, self.VQ_mode_indices)
        self._mass_e_modes = extract(mass_e, self.VE_mode_indices)
        self._phys_e_modes = extract(phys_e, self.VE_mode_indices)
        self._refresh_dt_operators()
        self._rollout_cache = ScanRolloutCache(
            self._step_with_operators,
            dynamic_args=lambda: (
                self._dt_array,
                self.Lexp_modes,
                *self.Limp_lu,
            ),
        )

    def _refresh_dt_operators(self) -> None:
        """(Re)combine the dt-independent operator parts and refactorize.

        ``L_imp = M/dt - 0.5 L`` and ``L_exp = M/dt + 0.5 L`` share the same
        mass and physics blocks, so an adaptive-CFL dt change only recombines
        and re-runs the batched LU (plus the insulating tau-row surgery, whose
        trapped-flux Faraday row carries dt).
        """
        inv_dt = 1.0 / self.dt
        limp = inv_dt * self._mass_q_modes + self._phys_q_modes
        self.Lexp_modes = inv_dt * self._mass_e_modes + self._phys_e_modes
        if self.magnetic_bc == "insulating":
            limp = self._apply_insulating_rows(limp)
        self.Limp_modes = limp
        self.Limp_lu = jax.vmap(jsp_linalg.lu_factor)(
            self._pin_pressure_modes(self.Limp_modes)
        )

    def set_dt(self, dt: float) -> None:
        """Adopt a new time step (adaptive-CFL support) and refactorize."""
        self.dt = float(dt)
        self._dt_array = jnp.asarray(self.dt)
        self._refresh_dt_operators()

    # ------------------------------------------------------------------
    # assembly helpers (mirroring the primitive TC DNS class)
    # ------------------------------------------------------------------
    @staticmethod
    def _family_class(family: str):
        if family.startswith("L"):
            return Legendre
        if family.startswith("C"):
            return Chebyshev
        raise ValueError("family must be 'L' or 'C'")

    def _lap(self, u: sp.Expr) -> sp.Expr:
        return _cylindrical_laplacian_3d(u, self.r)

    _dense = staticmethod(AxisymmetricTCDNSJax._dense)
    _put_block = staticmethod(AxisymmetricTCDNSJax._put_block)
    _scatter_modes = staticmethod(AxisymmetricTCDNSJax._scatter_modes)

    def _add_form(self, A, test_space, trial_space, i, j, expr):
        return self._put_block(
            A,
            test_space.block_slices[i],
            trial_space.block_slices[j],
            self._dense(expr),
        )

    def _add_vp_terms(
        self,
        A: Array,
        test_space: CoupledSpace,
        trial_space: CoupledSpace,
        idx: dict[str, int],
        fields: dict[str, sp.Expr],
        tests: dict[str, sp.Expr],
        sign: float,
    ) -> Array:
        r = self.r
        nu, eta, B0 = self.nu, self.eta_mag, self.B0
        a = self.base.a
        omega = self.base.a + self.base.b / r**2
        u_theta_base = omega * r

        def dz(f):
            return Dx(f, 1, 1)

        def dth(f):
            return Dx(f, 0, 1)

        Ar, At, Az = fields["Ar"], fields["At"], fields["Az"]

        def dr(f):
            return Dx(f, 2, 1)

        simple_terms = [
            # --- velocity block: viscosity, base advection, rotation ---
            ("ur", "ur", tests["ur"] * (sign * nu * self._lap(fields["ur"]))),
            ("ur", "ur", tests["ur"] * (sign * (-nu) * (1 / r**2) * fields["ur"])),
            ("ur", "ut", tests["ur"] * (sign * (-nu) * (2 / r**2) * dth(fields["ut"]))),
            ("ur", "ur", tests["ur"] * (sign * (-omega) * dth(fields["ur"]))),
            ("ur", "ut", tests["ur"] * (sign * (2 * omega) * fields["ut"])),
            ("ut", "ut", tests["ut"] * (sign * nu * self._lap(fields["ut"]))),
            ("ut", "ut", tests["ut"] * (sign * (-nu) * (1 / r**2) * fields["ut"])),
            ("ut", "ur", tests["ut"] * (sign * nu * (2 / r**2) * dth(fields["ur"]))),
            ("ut", "ut", tests["ut"] * (sign * (-omega) * dth(fields["ut"]))),
            ("ut", "ur", tests["ut"] * (sign * (-2 * a) * fields["ur"])),
            ("uz", "uz", tests["uz"] * (sign * nu * self._lap(fields["uz"]))),
            ("uz", "uz", tests["uz"] * (sign * (-omega) * dth(fields["uz"]))),
            # --- induction: eta vector Laplacian of A ---
            ("Ar", "Ar", tests["Ar"] * (sign * eta * self._lap(fields["Ar"]))),
            ("Ar", "Ar", tests["Ar"] * (sign * (-eta) * (1 / r**2) * fields["Ar"])),
            (
                "Ar",
                "At",
                tests["Ar"] * (sign * (-eta) * (2 / r**2) * dth(fields["At"])),
            ),
            ("At", "At", tests["At"] * (sign * eta * self._lap(fields["At"]))),
            ("At", "At", tests["At"] * (sign * (-eta) * (1 / r**2) * fields["At"])),
            ("At", "Ar", tests["At"] * (sign * eta * (2 / r**2) * dth(fields["Ar"]))),
            ("Az", "Az", tests["Az"] * (sign * eta * self._lap(fields["Az"]))),
            # --- linear EMF u x B0 e_z = B0 (u_theta, -u_r, 0) ---
            ("Ar", "ut", tests["Ar"] * (sign * B0 * fields["ut"])),
            ("At", "ur", tests["At"] * (sign * (-B0) * fields["ur"])),
        ]
        # Compound couplings, hand-expanded per trial column so every
        # `_add_form` call carries exactly one trial function (nested Dx form,
        # which the separable assembler parses).  With b = curl(A):
        #   j_theta(A) = dz(b_r) - dr(b_z)
        #   j_r(A)     = (1/r) dth(b_z) - dz(b_theta)
        j_t_cols = [
            ("Ar", -(1 / r**2) * dth(Ar) + (1 / r) * dr(dth(Ar))),
            ("At", -dz(dz(At)) - dr(dr(At)) - (1 / r) * dr(At) + At / r**2),
            ("Az", (1 / r) * dz(dth(Az))),
        ]
        j_r_cols = [
            ("Ar", -(1 / r**2) * dth(dth(Ar)) - dz(dz(Ar))),
            ("At", (1 / r) * dth(dr(At)) + (1 / r**2) * dth(At)),
            ("Az", dz(dr(Az))),
        ]
        # U x b(A) = U_theta * (b_z e_r - b_r e_z)
        uxb_r_cols = [
            ("Ar", -u_theta_base * (1 / r) * dth(Ar)),
            ("At", u_theta_base * (dr(At) + At / r)),
        ]
        uxb_z_cols = [
            ("At", u_theta_base * dz(At)),
            ("Az", -u_theta_base * (1 / r) * dth(Az)),
        ]
        compound_terms = (
            [("ur", col, sign * B0 * expr) for col, expr in j_t_cols]
            + [("ut", col, sign * (-B0) * expr) for col, expr in j_r_cols]
            + [("Ar", col, sign * expr) for col, expr in uxb_r_cols]
            + [("Az", col, sign * expr) for col, expr in uxb_z_cols]
        )
        for row, col, expr in simple_terms + [
            (row, col, tests[row] * expr) for row, col, expr in compound_terms
        ]:
            A = self._add_form(A, test_space, trial_space, idx[row], idx[col], expr)
        return A

    def _build_operator_parts(self) -> tuple[Array, Array, Array, Array]:
        """Assemble dt-independent (mass, physics) parts of L_imp and L_exp."""
        r = self.r
        dtype = jnp.result_type(jnp.asarray(1.0), jnp.asarray(1.0j))

        names_q = ("ur", "ut", "uz", "p", "Ar", "At", "Az")
        spaces_q = (self.TD, self.TD, self.TD, self.TP, self.TAr, self.TAt, self.TAz)
        idx_q = {name: i for i, name in enumerate(names_q)}
        fields_q = {
            name: TrialFunction(space, name=f"tr_{name}")
            for name, space in zip(names_q, spaces_q, strict=True)
        }
        tests_q = {
            name: TestFunction(space, name=f"te_{name}")
            for name, space in zip(names_q, spaces_q, strict=True)
        }

        mass_q = jnp.zeros((self.VQ.dim, self.VQ.dim), dtype=dtype)
        phys_q = jnp.zeros_like(mass_q)
        for name in ("ur", "ut", "uz", "Ar", "At", "Az"):
            mass_q = self._add_form(
                mass_q,
                self.VQ,
                self.VQ,
                idx_q[name],
                idx_q[name],
                tests_q[name] * fields_q[name],
            )
        phys_q = self._add_vp_terms(
            phys_q, self.VQ, self.VQ, idx_q, fields_q, tests_q, sign=-0.5
        )
        p = fields_q["p"]
        q = tests_q["p"]
        ur, ut, uz = fields_q["ur"], fields_q["ut"], fields_q["uz"]
        vr, vt, vz = tests_q["ur"], tests_q["ut"], tests_q["uz"]
        phys_q = self._add_form(phys_q, self.VQ, self.VQ, 0, 3, vr * Dx(p, 2, 1))
        phys_q = self._add_form(
            phys_q, self.VQ, self.VQ, 1, 3, vt * (1 / r) * Dx(p, 0, 1)
        )
        phys_q = self._add_form(phys_q, self.VQ, self.VQ, 2, 3, vz * Dx(p, 1, 1))
        phys_q = self._add_form(phys_q, self.VQ, self.VQ, 3, 0, q * Dx(ur, 2, 1))
        phys_q = self._add_form(phys_q, self.VQ, self.VQ, 3, 0, q * (1 / r) * ur)
        phys_q = self._add_form(
            phys_q, self.VQ, self.VQ, 3, 1, q * (1 / r) * Dx(ut, 0, 1)
        )
        phys_q = self._add_form(phys_q, self.VQ, self.VQ, 3, 2, q * Dx(uz, 1, 1))

        names_e = ("ur", "ut", "uz", "Ar", "At", "Az")
        spaces_e = (self.TD, self.TD, self.TD, self.TAr, self.TAt, self.TAz)
        idx_e = {name: i for i, name in enumerate(names_e)}
        fields_e = {
            name: TrialFunction(space, name=f"etr_{name}")
            for name, space in zip(names_e, spaces_e, strict=True)
        }
        tests_e = {
            name: TestFunction(space, name=f"ete_{name}")
            for name, space in zip(names_e, spaces_e, strict=True)
        }
        mass_e = jnp.zeros((self.VE.dim, self.VE.dim), dtype=dtype)
        phys_e = jnp.zeros_like(mass_e)
        for name in names_e:
            mass_e = self._add_form(
                mass_e,
                self.VE,
                self.VE,
                idx_e[name],
                idx_e[name],
                tests_e[name] * fields_e[name],
            )
        phys_e = self._add_vp_terms(
            phys_e, self.VE, self.VE, idx_e, fields_e, tests_e, sign=0.5
        )
        return mass_q, phys_q, mass_e, phys_e

    # ------------------------------------------------------------------
    # insulating wall rows (tau method, per mode)
    # ------------------------------------------------------------------
    def _build_wall_rows(self) -> None:
        """Radial-basis evaluation rows at the walls for tau rows/diagnostics."""
        bounds = jnp.asarray([self.base.R1, self.base.R2], dtype=float)
        ref = self.S0.map_reference_domain(bounds)
        df = float(self.S0.domain_factor)
        self._wall_d0 = np.asarray(self.S0.evaluate_basis_derivative(ref, 0))
        self._wall_d1 = np.asarray(self.S0.evaluate_basis_derivative(ref, 1)) * df
        self._wall_d2 = np.asarray(self.S0.evaluate_basis_derivative(ref, 2)) * df**2

    def _mode_wavenumbers(self) -> tuple[np.ndarray, np.ndarray]:
        """Physical (m, kz) per flattened (theta, z) mode, matching the
        row-major mode enumeration of ``_mode_indices``."""
        m_modes = np.fft.fftfreq(self.Ntheta, d=1.0 / self.Ntheta)
        kz_modes = 2.0 * math.pi * np.fft.fftfreq(self.Nz, d=self.Lz / self.Nz)
        mm, kk = np.meshgrid(m_modes, kz_modes, indexing="ij")
        return mm.reshape(-1), kk.reshape(-1)

    def _insulating_mode_rows(
        self, m: int, kz: float
    ) -> list[tuple[str, int, dict[str, np.ndarray]]]:
        """Return the six tau rows of one mode as (component, local_row,
        {component: row_vector}) entries."""
        R1, R2 = self.base.R1, self.base.R2
        d0, d1, d2 = self._wall_d0, self._wall_d1, self._wall_d2
        n = d0.shape[1]
        zero = np.zeros(n, dtype=complex)
        rows: list[tuple[str, int, dict[str, np.ndarray]]] = []

        def b_rows(wall: int, R: float):
            """Wall-evaluation row vectors of (b_r, b_theta, b_z) in A dofs."""
            im = 1j * m
            ikz = 1j * kz
            br = {"Ar": zero, "At": -ikz * d0[wall], "Az": (im / R) * d0[wall]}
            bt = {"Ar": ikz * d0[wall], "At": zero, "Az": -d1[wall]}
            bz = {
                "Ar": -(im / R) * d0[wall],
                "At": d1[wall] + d0[wall] / R,
                "Az": zero,
            }
            return br, bt, bz

        def combine(*weighted):
            out = {"Ar": zero.copy(), "At": zero.copy(), "Az": zero.copy()}
            for coeff, row in weighted:
                for key in out:
                    out[key] = out[key] + coeff * row[key]
            return out

        gauge_in = {"Ar": d0[0].astype(complex), "At": zero, "Az": zero}
        gauge_out = {"Ar": d0[1].astype(complex), "At": zero, "Az": zero}
        rows.append(("Ar", 0, gauge_in))
        rows.append(("Ar", 1, gauge_out))

        if m == 0 and kz == 0.0:
            # Mean mode: b_theta = -Az' = 0 at both walls; b_z(R2) = 0;
            # trapped-flux Faraday row at R1 (state-dependent rhs, see step()).
            rows.append(("Az", 0, {"Ar": zero, "At": zero, "Az": d1[0] + 0j}))
            rows.append(("Az", 1, {"Ar": zero, "At": zero, "Az": d1[1] + 0j}))
            e_bz_in = d1[0] + d0[0] / R1
            e_dbz_in = d2[0] + d1[0] / R1 - d0[0] / R1**2
            dyn = (R1 / (2.0 * self.dt)) * e_bz_in - 0.5 * self.eta_mag * e_dbz_in
            rows.append(("At", 0, {"Ar": zero, "At": dyn + 0j, "Az": zero}))
            e_bz_out = d1[1] + d0[1] / R2
            rows.append(("At", 1, {"Ar": zero, "At": e_bz_out + 0j, "Az": zero}))
            return rows

        for wall, R in ((0, R1), (1, R2)):
            br, bt, bz = b_rows(wall, R)
            if kz == 0.0:
                # Exterior potential ~ r^{+-|m|}: b_z = 0 and
                # |m| b_theta -+ i m b_r = 0 (inner grows, outer decays).
                sgn = -1.0 if wall == 0 else 1.0
                rows.append(
                    (
                        "At",
                        wall,
                        combine((1.0, bz)),
                    )
                )
                rows.append(
                    (
                        "Az",
                        wall,
                        combine((abs(m), bt), (sgn * 1j * m, br)),
                    )
                )
                continue
            x = abs(kz) * R
            if wall == 0:
                ratio = abs(kz) * _bessel_i_log_derivative(m, x)
            else:
                ratio = abs(kz) * _bessel_k_log_derivative(m, x)
            # Matching b to the exterior potential psi with b_r = psi',
            # b_theta = (i m / r) psi, b_z = i kz psi:
            #   b_theta * ratio - b_r * (i m / R) = 0
            #   b_z * ratio - b_r * (i kz) = 0
            rows.append(("Az", wall, combine((ratio, bt), (-1j * m / R, br))))
            rows.append(("At", wall, combine((ratio, bz), (-1j * kz, br))))
        return rows

    def _a_block_offsets(self) -> dict[str, tuple[int, int]]:
        """Per-mode (offset, size) of each A component inside a mode block."""
        sizes = [int(space.num_dofs[-1]) for space in self.VQ]
        starts = np.concatenate(([0], np.cumsum(sizes)))
        names = ("ur", "ut", "uz", "p", "Ar", "At", "Az")
        return {
            name: (int(starts[i]), int(sizes[i]))
            for i, name in enumerate(names)
            if name in ("Ar", "At", "Az")
        }

    def _apply_insulating_rows(self, modes: Array) -> Array:
        """Replace the last two radial Galerkin rows of each A component with
        the per-mode vacuum-matching tau rows."""
        blocks = self._a_block_offsets()
        m_all, kz_all = self._mode_wavenumbers()
        modes_np = np.array(modes)  # writable host copy for the row surgery
        dyn_positions = np.full(modes_np.shape[0], -1, dtype=int)
        n = blocks["Ar"][1]
        residual_rows = np.zeros((modes_np.shape[0], 6, 3, n), dtype=modes_np.dtype)
        residual_mask = np.zeros((modes_np.shape[0], 6), dtype=bool)
        component_index = {name: i for i, name in enumerate(("Ar", "At", "Az"))}
        for mode in range(modes_np.shape[0]):
            m = int(round(float(m_all[mode])))
            kz = float(kz_all[mode])
            for slot, (comp, local, row) in enumerate(
                self._insulating_mode_rows(m, kz)
            ):
                off, size = blocks[comp]
                target = off + size - 2 + local
                modes_np[mode, target, :] = 0.0
                for name, vec in row.items():
                    o2, s2 = blocks[name]
                    modes_np[mode, target, o2 : o2 + s2] = vec
                if m == 0 and kz == 0.0 and comp == "At" and local == 0:
                    dyn_positions[mode] = target
                else:
                    residual_mask[mode, slot] = True
                    for name, vec in row.items():
                        residual_rows[mode, slot, component_index[name], :] = vec
        self._insulating_dyn_row = int(dyn_positions.max())
        self._insulating_dyn_mode = int(np.argmax(dyn_positions))
        self._tau_row_positions = self._collect_tau_positions(blocks)
        # Diagnostics reuse these already-derived static rows.  Keeping the
        # contraction on device avoids the previous per-mode NumPy sync and
        # repeated ive/kve work on every diagnostics cadence.
        self._insulating_residual_rows = jnp.asarray(residual_rows)
        self._insulating_residual_mask = jnp.asarray(residual_mask)
        return jnp.asarray(modes_np)

    def _collect_tau_positions(self, blocks) -> np.ndarray:
        positions = []
        for comp in ("Ar", "At", "Az"):
            off, size = blocks[comp]
            positions.extend([off + size - 2, off + size - 1])
        return np.asarray(sorted(positions), dtype=int)

    def _pin_pressure_modes(self, modes: Array) -> Array:
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        modes = modes.at[0, pressure_row, :].set(0)
        return modes.at[0, pressure_row, pressure_row].set(1)

    # ------------------------------------------------------------------
    # state construction
    # ------------------------------------------------------------------
    def zero_state(self) -> TCVPState:
        u = tuple(
            jnp.zeros(space.num_dofs, dtype=self.Limp_modes.dtype)
            for space in (self.TD, self.TD, self.TD)
        )
        A = tuple(
            jnp.zeros(space.num_dofs, dtype=self.Limp_modes.dtype)
            for space in (self.TAr, self.TAt, self.TAz)
        )
        p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp_modes.dtype)
        nold = tuple(jnp.zeros_like(x) for x in (*u, *A))
        return TCVPState(u=u, p=p, A=A, nonlinear_old=nold, have_old=0.0)

    def state_from_physical(self, u_phys: Velocity, a_phys: Velocity) -> TCVPState:
        u = tuple(self.TD.forward(v) for v in u_phys)
        A = tuple(
            space.forward(v)
            for space, v in zip((self.TAr, self.TAt, self.TAz), a_phys, strict=True)
        )
        p = jnp.zeros(self.TP.num_dofs, dtype=u[0].dtype)
        nold = tuple(jnp.zeros_like(x) for x in (*u, *A))
        return TCVPState(u=u, p=p, A=A, nonlinear_old=nold, have_old=0.0)

    # ------------------------------------------------------------------
    # physics evaluation
    # ------------------------------------------------------------------
    def _phys4(self, coeff: Array, space, padded: bool = True):
        N = self.padded_counts if padded else None
        value = space.backward(coeff, N=N)
        radial = space.backward_primitive(coeff, (0, 0, 1), N=N)
        theta = space.backward_primitive(coeff, (1, 0, 0), N=N)
        axial = space.backward_primitive(coeff, (0, 1, 0), N=N)
        return value, radial, theta, axial

    def b_physical(self, A: Velocity, padded: bool = True) -> Velocity:
        """Fluctuation b = curl(A) evaluated pointwise on the quadrature grid."""
        N = self.padded_counts if padded else None
        invr = (
            self.inv_r_p if (padded and self.padded_counts is not None) else self.inv_r
        )
        Ar_t = self.TAr.backward_primitive(A[0], (1, 0, 0), N=N)
        Ar_z = self.TAr.backward_primitive(A[0], (0, 1, 0), N=N)
        if self.TAt is self.TAz:
            tangential_value, tangential_radial, tangential_theta, tangential_axial = (
                jax.vmap(lambda coeff: self._phys4(coeff, self.TAt, padded=padded))(
                    jnp.stack(A[1:])
                )
            )
            At_v = tangential_value[0]
            At_r, Az_r = tangential_radial
            Az_t = tangential_theta[1]
            At_z = tangential_axial[0]
        else:
            At_v = self.TAt.backward(A[1], N=N)
            At_r = self.TAt.backward_primitive(A[1], (0, 0, 1), N=N)
            At_z = self.TAt.backward_primitive(A[1], (0, 1, 0), N=N)
            Az_t = self.TAz.backward_primitive(A[2], (1, 0, 0), N=N)
            Az_r = self.TAz.backward_primitive(A[2], (0, 0, 1), N=N)
        br = invr * Az_t - At_z
        bt = Ar_z - Az_r
        bz = At_r + invr * At_v - invr * Ar_t
        return br, bt, bz

    def b_coefficients(
        self, A: Velocity, *, b_phys: Velocity | None = None
    ) -> Velocity:
        """Forward-projected coefficient representation of b (T0 spaces)."""
        br, bt, bz = self.b_physical(A, padded=False) if b_phys is None else b_phys
        return (
            self.T0.mask_nyquist(self.T0.forward(br)),
            self.T0.mask_nyquist(self.T0.forward(bt)),
            self.T0.mask_nyquist(self.T0.forward(bz)),
        )

    def _current_physical(self, B: Velocity, *, padded: bool = True) -> Velocity:
        """Evaluate ``curl(B)`` with only the seven required transforms."""

        N = self.padded_counts if padded else None
        invr = (
            self.inv_r_p if (padded and self.padded_counts is not None) else self.inv_r
        )
        br, bt, bz = B
        bt_value = self.T0.backward(bt, N=N)
        bt_r, bz_r = jax.vmap(
            lambda coeff: self.T0.backward_primitive(coeff, (0, 0, 1), N=N)
        )(jnp.stack((bt, bz)))
        br_t, bz_t = jax.vmap(
            lambda coeff: self.T0.backward_primitive(coeff, (1, 0, 0), N=N)
        )(jnp.stack((br, bz)))
        br_z, bt_z = jax.vmap(
            lambda coeff: self.T0.backward_primitive(coeff, (0, 1, 0), N=N)
        )(jnp.stack((br, bt)))
        return (
            invr * bz_t - bt_z,
            br_z - bz_r,
            bt_r + invr * bt_value - invr * br_t,
        )

    def _dealias_to_standard(self, values: Array) -> Array:
        if self.T0p is None:
            return values
        coeff = self.T0p.forward(values)
        return self.T0.backward(coeff)

    def nonlinear(self, state: TCVPState) -> tuple[Array, ...]:
        """Explicit nonlinear rows: advection - j_f x b_f for u, u x b for A."""
        velocity, velocity_radial, velocity_theta, velocity_axial = jax.vmap(
            lambda coeff: self._phys4(coeff, self.TD)
        )(jnp.stack(state.u))
        ur, ut, uz = velocity
        urr, utr, uzr = velocity_radial
        urt, utt, uzt = velocity_theta
        urz, utz, uzz = velocity_axial
        invr = self.inv_r_p
        au_r = ur * urr + (ut * invr) * urt + uz * urz - ut * ut * invr
        au_t = ur * utr + (ut * invr) * utt + uz * utz + ur * ut * invr
        au_z = ur * uzr + (ut * invr) * uzt + uz * uzz

        br, bt, bz = self.b_physical(state.A, padded=True)
        Bc = self.b_coefficients(state.A)
        j_r, j_t, j_z = self._current_physical(Bc)
        lor_r = j_t * bz - j_z * bt
        lor_t = j_z * br - j_r * bz
        lor_z = j_r * bt - j_t * br

        nu_r = self._dealias_to_standard(au_r - lor_r)
        nu_t = self._dealias_to_standard(au_t - lor_t)
        nu_z = self._dealias_to_standard(au_z - lor_z)

        emf_r = self._dealias_to_standard(ut * bz - uz * bt)
        emf_t = self._dealias_to_standard(uz * br - ur * bz)
        emf_z = self._dealias_to_standard(ur * bt - ut * br)
        return (
            self.TD.mask_nyquist(self.TD.scalar_product(nu_r)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_t)),
            self.TD.mask_nyquist(self.TD.scalar_product(nu_z)),
            # A-forcing carries a minus sign inside cnab2_rhs (rhs - N), so
            # store the negative EMF here.
            self.TAr.mask_nyquist(self.TAr.scalar_product(-emf_r)),
            self.TAt.mask_nyquist(self.TAt.scalar_product(-emf_t)),
            self.TAz.mask_nyquist(self.TAz.scalar_product(-emf_z)),
        )

    def _apply_lexp(
        self, x: tuple[Array, ...], Lexp_modes: Array | None = None
    ) -> tuple[Array, ...]:
        flat = self.VE.flatten(x)
        modes = flat[self.VE_mode_indices]
        if Lexp_modes is None:
            Lexp_modes = self.Lexp_modes
        out_modes = jnp.einsum("kij,kj->ki", Lexp_modes, modes)
        out = self._scatter_modes(out_modes, self.VE_mode_indices, self.VE.dim)
        return self.VE.unflatten(out)

    def _dyn_row_rhs(self, state: TCVPState, dt: Array) -> Array:
        """Right-hand side of the trapped-flux Faraday row (insulating (0,0))."""
        R1 = self.base.R1
        d0 = jnp.asarray(self._wall_d0[0])
        d1 = jnp.asarray(self._wall_d1[0])
        d2 = jnp.asarray(self._wall_d2[0])
        e_bz = d1 + d0 / R1
        e_dbz = d2 + d1 / R1 - d0 / R1**2
        at00 = state.A[1][0, 0, :]
        bz_wall = jnp.sum(e_bz * at00)
        dbz_wall = jnp.sum(e_dbz * at00)
        return (R1 / (2.0 * dt)) * bz_wall + 0.5 * self.eta_mag * dbz_wall

    def _solve_limp(
        self,
        rhs: Array,
        state: TCVPState,
        dt: Array,
        Limp_lu: tuple[Array, Array] | None = None,
    ) -> Array:
        rhs_modes = rhs[self.VQ_mode_indices]
        pressure_row = sum(int(space.num_dofs[-1]) for space in self.VQ[:3])
        rhs_modes = rhs_modes.at[0, pressure_row].set(0)
        if self.magnetic_bc == "insulating":
            rhs_modes = rhs_modes.at[:, self._tau_row_positions].set(0.0)
            rhs_modes = rhs_modes.at[
                self._insulating_dyn_mode, self._insulating_dyn_row
            ].set(self._dyn_row_rhs(state, dt))
        lu, piv = self.Limp_lu if Limp_lu is None else Limp_lu
        sol_modes = jax.vmap(
            lambda lu_i, piv_i, b_i: jsp_linalg.lu_solve((lu_i, piv_i), b_i)
        )(lu, piv, rhs_modes)
        return self._scatter_modes(sol_modes, self.VQ_mode_indices, self.VQ.dim)

    def step(
        self,
        state: TCVPState,
        dt: Array | None = None,
        Lexp_modes: Array | None = None,
        Limp_lu: tuple[Array, Array] | None = None,
    ) -> TCVPState:
        if dt is None:
            dt = self._dt_array
        n_hat = self.nonlinear(state)
        rhs_e = self._apply_lexp((*state.u, *state.A), Lexp_modes)
        rhs_x = cnab2_rhs(rhs_e, n_hat, state.nonlinear_old, state.have_old)
        rhs_p = jnp.zeros(self.TP.num_dofs, dtype=self.Limp_modes.dtype)
        rhs = self.VQ.flatten((*rhs_x[:3], rhs_p, *rhs_x[3:]))
        sol = self.VQ.unflatten(self._solve_limp(rhs, state, dt, Limp_lu))
        return TCVPState(
            u=(sol[0], sol[1], sol[2]),
            p=sol[3],
            A=(sol[4], sol[5], sol[6]),
            nonlinear_old=n_hat,
            have_old=jnp.ones_like(state.have_old),
        )

    def _step_with_operators(
        self, state: TCVPState, dt: Array, Lexp_modes: Array, lu: Array, piv: Array
    ) -> TCVPState:
        """Advance one compiled rollout step with runtime operator data."""
        return self.step(state, dt, Lexp_modes, (lu, piv))

    def solve(self, state: TCVPState, steps: int) -> TCVPState:
        return self._rollout_cache(state, int(steps))

    def rollout_cache_info(self) -> ScanRolloutCacheInfo:
        return self._rollout_cache.info()

    def solve_with_cadence(
        self,
        state: TCVPState,
        steps: int,
        cadence: Cadence,
        *,
        block_size: int = 1,
        on_diagnostics=None,
        on_snapshot=None,
        on_checkpoint=None,
        should_stop=None,
        t0: float = 0.0,
        tstep0: int = 0,
    ) -> TCVPState:
        return run_with_cadence(
            self.solve,
            state,
            steps=steps,
            dt=self.dt,
            cadence=cadence,
            block_size=block_size,
            diagnostics=self.diagnostics,
            on_diagnostics=on_diagnostics,
            on_snapshot=on_snapshot,
            on_checkpoint=on_checkpoint,
            should_stop=should_stop,
            t0=t0,
            tstep0=tstep0,
        )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def velocity_physical(self, state: TCVPState) -> Velocity:
        return tuple(self.TD.backward(ui) for ui in state.u)

    def fields_physical(self, state: TCVPState) -> tuple[Array, ...]:
        """Physical (u_r, u_t, u_z, b_r, b_t, b_z) fluctuation fields."""
        up = self.velocity_physical(state)
        bp = self.b_physical(state.A, padded=False)
        return (*up, *bp)

    def velocity_divergence(self, state: TCVPState) -> Array:
        dur_dr = self.TD.backward_primitive(state.u[0], (0, 0, 1))
        dut_dt = self.TD.backward_primitive(state.u[1], (1, 0, 0))
        duz_dz = self.TD.backward_primitive(state.u[2], (0, 1, 0))
        ur = self.TD.backward(state.u[0])
        return dur_dr + ur * self.inv_r + dut_dt * self.inv_r + duz_dz

    def magnetic_divergence(
        self, state: TCVPState, *, coefficients: Velocity | None = None
    ) -> Array:
        """div b of the projected coefficient representation of b = curl(A)."""
        Bc = self.b_coefficients(state.A) if coefficients is None else coefficients
        dbr_dr = self.T0.backward_primitive(Bc[0], (0, 0, 1))
        dbt_dt = self.T0.backward_primitive(Bc[1], (1, 0, 0))
        dbz_dz = self.T0.backward_primitive(Bc[2], (0, 1, 0))
        br = self.T0.backward(Bc[0])
        return dbr_dr + br * self.inv_r + dbt_dt * self.inv_r + dbz_dz

    def _weighted_l2(self, values: Array) -> Array:
        squared = jnp.real(jnp.conj(values) * values) * self.R
        return jnp.sqrt(jnp.real(integrate(squared, self.T0)))

    def divergence_linf(
        self,
        state: TCVPState,
        *,
        divu_field: Array | None = None,
        divb_field: Array | None = None,
    ) -> tuple[Array, Array]:
        divu_field = (
            self.velocity_divergence(state) if divu_field is None else divu_field
        )
        divb_field = (
            self.magnetic_divergence(state) if divb_field is None else divb_field
        )
        return (
            jnp.max(jnp.abs(divu_field)),
            jnp.max(jnp.abs(divb_field)),
        )

    def energy_parts(
        self, state: TCVPState, *, fields: tuple[Array, ...] | None = None
    ) -> tuple[Array, Array]:
        fields = self.fields_physical(state) if fields is None else fields
        return cylindrical_energy_parts(fields[:3], fields[3:], self.R, self.T0)

    def energy(self, state: TCVPState) -> Array:
        ek, em = self.energy_parts(state)
        return ek + em

    def _volume(self) -> float:
        R1, R2 = self.base.R1, self.base.R2
        return math.pi * (R2**2 - R1**2) * self.Lz

    def stresses(
        self, state: TCVPState, *, fields: tuple[Array, ...] | None = None
    ) -> tuple[Array, Array]:
        """Volume-mean Reynolds and Maxwell r-theta stresses.

        ``integrate`` over the 3D tensor space already covers the azimuthal
        quadrature, so the volume mean is the r-weighted integral over volume.
        """
        fields = self.fields_physical(state) if fields is None else fields
        volume = self._volume()
        reynolds = (
            integrate(jnp.real(fields[0] * jnp.conj(fields[1])) * self.R, self.T0)
            / volume
        )
        maxwell = (
            -integrate(jnp.real(fields[3] * jnp.conj(fields[4])) * self.R, self.T0)
            / volume
        )
        return jnp.real(reynolds), jnp.real(maxwell)

    def mean_bz_total(self, state: TCVPState, *, bz: Array | None = None) -> Array:
        """Volume mean of the total axial field (imposed B0 included)."""
        if bz is None:
            _, _, bz = self.b_physical(state.A, padded=False)
        fluct = integrate(jnp.real(bz) * self.R, self.T0) / self._volume()
        return self.B0 + jnp.real(fluct)

    def insulating_bc_residual(self, state: TCVPState) -> Array:
        """Max wall residual of the vacuum-matching rows applied to A."""
        if self.magnetic_bc != "insulating":
            return jnp.asarray(0.0)
        n = int(state.A[0].shape[-1])
        a_modes = jnp.stack(
            [component.reshape((-1, n)) for component in state.A], axis=1
        )
        values = jnp.einsum("mscn,mcn->ms", self._insulating_residual_rows, a_modes)
        residuals = jnp.where(self._insulating_residual_mask, jnp.abs(values), 0.0)
        return jnp.max(residuals)

    def diagnostics(self, state: TCVPState) -> dict[str, Array]:
        velocity = self.velocity_physical(state)
        magnetic = self.b_physical(state.A, padded=False)
        fields = (*velocity, *magnetic)
        b_coefficients = self.b_coefficients(state.A, b_phys=magnetic)
        divu_field = self.velocity_divergence(state)
        divb_field = self.magnetic_divergence(state, coefficients=b_coefficients)
        ek, em = self.energy_parts(state, fields=fields)
        divu, divb = self.divergence_linf(
            state, divu_field=divu_field, divb_field=divb_field
        )
        reynolds, maxwell = self.stresses(state, fields=fields)
        out = {
            "Ekin": ek,
            "Emag": em,
            "E": ek + em,
            "divu": divu,
            "divb": divb,
            "divu_l2": self._weighted_l2(divu_field),
            "divb_l2": self._weighted_l2(divb_field),
            "reynolds_rt": reynolds,
            "maxwell_rt": maxwell,
            "total_stress": reynolds + maxwell,
            "mean_bz": self.mean_bz_total(state, bz=magnetic[2]),
            "wall_u": coefficient_wall_linf(state.u, (self.TD,) * 3),
        }
        if self.magnetic_bc == "insulating":
            out["insulating_bc_residual"] = self.insulating_bc_residual(state)
        return out

    def growth_rate(self, state: TCVPState, steps: int) -> tuple[Array, TCVPState]:
        e0 = self.energy(state)
        out = self.solve(state, steps)
        e1 = self.energy(out)
        elapsed = int(steps) * self.dt
        return 0.5 * jnp.log(e1 / e0) / elapsed, out

    # ------------------------------------------------------------------
    # eigenmode seeding (A reconstructed in closed form per mode)
    # ------------------------------------------------------------------
    def _radial_antiderivative(self, values: np.ndarray) -> np.ndarray:
        """Antiderivative on the radial grid via an exact polynomial fit."""
        r = np.asarray(jnp.squeeze(self.R))
        deg = len(r) - 1
        fit_re = np.polynomial.legendre.Legendre.fit(
            r, values.real, deg, domain=[self.base.R1, self.base.R2]
        )
        fit_im = np.polynomial.legendre.Legendre.fit(
            r, values.imag, deg, domain=[self.base.R1, self.base.R2]
        )
        anti_re = fit_re.integ(lbnd=self.base.R1)
        anti_im = fit_im.integ(lbnd=self.base.R1)
        return anti_re(r) + 1j * anti_im(r)

    def _a_profiles_from_b(
        self, m: int, kz: float, br: np.ndarray, bt: np.ndarray, bz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closed-form A(r) with curl(A) = b for a single (m, kz != 0) mode.

        Gauge: A_r = c / r (satisfying the conducting Robin row identically),
        with the constant fixed so the A_z Dirichlet compatibility
        ``ikz * int A_r dr = int b_theta dr`` holds; the ``b_z`` component of
        curl(A) then matches automatically because both fields are
        divergence-free.
        """
        if kz == 0.0:
            raise ValueError("eigenmode seeding requires kz != 0")
        r = np.asarray(jnp.squeeze(self.R))
        R1, R2 = self.base.R1, self.base.R2
        # Evaluate int_{R1}^{R2} b_theta dr from the polynomial fit directly.
        fit_re = np.polynomial.legendre.Legendre.fit(
            r, bt.real, len(r) - 1, domain=[R1, R2]
        )
        fit_im = np.polynomial.legendre.Legendre.fit(
            r, bt.imag, len(r) - 1, domain=[R1, R2]
        )
        total_bt = fit_re.integ(lbnd=R1)(R2) + 1j * fit_im.integ(lbnd=R1)(R2)
        c = total_bt / (1j * kz * math.log(R2 / R1))
        a_r = c / r
        # A_z' = i kz A_r - b_theta, A_z(R1) = 0.
        a_z = self._radial_antiderivative(1j * kz * a_r - bt)
        # b_r = (i m / r) A_z - i kz A_theta -> A_theta algebraically.
        a_t = ((1j * m / r) * a_z - br) / (1j * kz)
        return a_r, a_t, a_z

    def seed_linear_eigenmode(
        self, m: int = 0, kz_mode: int = 1, amp: float = 1.0e-6, which: int = 0
    ) -> tuple[TCVPState, complex]:
        """Seed the real part of a linear MHD eigenmode of the matching walls.

        Conducting walls use the primitive 7-field eigensolver at any ``m``;
        insulating walls use the axisymmetric flux-function eigensolver
        (available for ``m = 0`` only, the current insulating linear anchor).
        """
        _require_resolved_m(m, self.Ntheta)
        _require_non_nyquist_kz(kz_mode, self.Nz, allow_zero=False)
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        lin = TaylorCouetteMRIJax(
            self.base,
            B0=self.B0,
            nu=self.nu,
            eta_mag=self.eta_mag,
            N=self.Nr,
            family=self.family,
            magnetic_bc=self.magnetic_bc,
        )
        w, vecs = lin.eigs(m=m, kz=kz, n_return=which + 1)
        vec = vecs[:, which]
        if int(m) == 0:
            vec = _positive_pivot_phase(vec)
        n = lin.n
        r = np.asarray(jnp.squeeze(self.R))

        if self.magnetic_bc == "conducting":
            blocks = {
                name: np.asarray(vec[i * n : (i + 1) * n])
                for i, name in enumerate(("ur", "ut", "uz", "p", "br", "bt", "bz"))
            }
            u_prof = [
                np.asarray(lin.SDv.backward(jnp.asarray(blocks[name])))
                for name in ("ur", "ut", "uz")
            ]
            br = np.asarray(lin.SDv.backward(jnp.asarray(blocks["br"])))
            bt = np.asarray(lin.Sbt.backward(jnp.asarray(blocks["bt"])))
            bz = np.asarray(lin.Sbz.backward(jnp.asarray(blocks["bz"])))
            a_r, a_t, a_z = self._a_profiles_from_b(int(m), kz, br, bt, bz)
        else:
            if int(m) != 0:
                raise NotImplementedError(
                    "insulating eigenmode seeding is anchored to the m=0 "
                    "flux-function eigensolver"
                )
            Schi, Sbth = lin._flux_bases(kz)
            blocks = {
                name: np.asarray(vec[i * n : (i + 1) * n])
                for i, name in enumerate(("ur", "ut", "uz", "p", "chi", "bt"))
            }
            u_prof = [
                np.asarray(lin.SDv.backward(jnp.asarray(blocks[name])))
                for name in ("ur", "ut", "uz")
            ]
            chi = np.asarray(Schi.backward(jnp.asarray(blocks["chi"])))
            bt = np.asarray(Sbth.backward(jnp.asarray(blocks["bt"])))
            # m = 0: A_theta = chi / r reproduces (b_r, b_z); A_z' = -b_theta.
            a_t = chi / r
            a_z = self._radial_antiderivative(-bt)
            a_r = np.zeros_like(a_t)

        state = self.zero_state()
        mpos = int(m) % self.Ntheta
        mneg = (-int(m)) % self.Ntheta
        kpos = int(kz_mode) % self.Nz
        kneg = (-int(kz_mode)) % self.Nz

        def to_coeffs(space1d, profile):
            return np.asarray(space1d.forward(jnp.asarray(profile)))

        comps_u = list(state.u)
        comps_a = list(state.A)
        u_spaces = (self.SD, self.SD, self.SD)
        a_spaces = (self.SAr, self.SAt, self.SAz)
        for i in range(3):
            cu = to_coeffs(u_spaces[i], u_prof[i]) * amp
            ca = to_coeffs(a_spaces[i], (a_r, a_t, a_z)[i]) * amp
            for comps, coeffs in ((comps_u, cu), (comps_a, ca)):
                arr = comps[i]
                if mpos == mneg and kpos == kneg:
                    arr = arr.at[mpos, kpos, : len(coeffs)].set(
                        jnp.real(jnp.asarray(coeffs))
                    )
                else:
                    arr = arr.at[mpos, kpos, : len(coeffs)].set(
                        0.5 * jnp.asarray(coeffs)
                    )
                    arr = arr.at[mneg, kneg, : len(coeffs)].set(
                        0.5 * jnp.conj(jnp.asarray(coeffs))
                    )
                comps[i] = arr
        nold = tuple(jnp.zeros_like(x) for x in (*comps_u, *comps_a))
        return (
            TCVPState(
                u=tuple(comps_u),
                p=state.p,
                A=tuple(comps_a),
                nonlinear_old=nold,
                have_old=0.0,
            ),
            complex(w[which]),
        )

    def add_symmetry_breaking_perturbation(
        self, state: TCVPState, amp: float, m: int = 1, kz_mode: int = 1
    ) -> TCVPState:
        """Superpose a non-axisymmetric solenoidal magnetic perturbation.

        An axisymmetric eigenmode seed stays axisymmetric under nonlinear
        evolution, so a small ``m != 0`` perturbation is required for a run
        to exercise the non-axisymmetric dynamics (and, for insulating walls,
        the non-axisymmetric Bessel matching rows).  The perturbation adds

            dA_theta = dA_z = amp * w(r) * cos(m theta) * cos(kz z),
            w(r) = sin^2(pi (r - R1) / (R2 - R1)),

        so ``b = curl(dA)`` is solenoidal by construction and carries
        ``(+-m, +-kz)`` content, while ``w = w' = 0`` at both cylinders makes
        every wall row -- conducting Dirichlet/Robin and insulating vacuum
        matching alike, all linear in wall values and first derivatives of
        ``A`` -- hold identically at t = 0.  Velocity and the AB2 history are
        untouched apart from restarting the IMEX-Euler bootstrap.
        """
        _require_resolved_m(m, self.Ntheta)
        if int(m) == 0:
            raise ValueError("the symmetry-breaking perturbation requires m != 0")
        _require_non_nyquist_kz(kz_mode, self.Nz, allow_zero=True)
        kz = 2.0 * math.pi * int(kz_mode) / self.Lz
        gap = self.base.R2 - self.base.R1
        w = jnp.sin(math.pi * (self.R - self.base.R1) / gap) ** 2
        field = float(amp) * w * jnp.cos(int(m) * self.Theta) * jnp.cos(kz * self.Z)
        d_at = self.TAt.mask_nyquist(self.TAt.forward(field))
        d_az = self.TAz.mask_nyquist(self.TAz.forward(field))
        A = (state.A[0], state.A[1] + d_at, state.A[2] + d_az)
        return TCVPState(
            u=state.u,
            p=state.p,
            A=A,
            nonlinear_old=state.nonlinear_old,
            have_old=0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D vector-potential Taylor-Couette MHD/MRI DNS"
    )
    # Demo defaults are a light smoke footprint (tests/test_demos.py executes
    # every example main under xdist); production resolutions live in the
    # run specs (production/runs/exp_tc_mri_*.json).
    parser.add_argument("--Nr", type=int, default=16)
    parser.add_argument("--Ntheta", type=int, default=4)
    parser.add_argument("--Nz", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--nu", type=float, default=1.0e-3)
    parser.add_argument("--eta-mag", type=float, default=1.0e-3)
    parser.add_argument("--B0", type=float, default=0.1)
    parser.add_argument("--family", choices=("L", "C"), default="C")
    parser.add_argument("--dealias", type=float, default=1.5)
    parser.add_argument(
        "--magnetic-bc", choices=("conducting", "insulating"), default="conducting"
    )
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--kz-mode", type=int, default=1)
    parser.add_argument("--amp", type=float, default=1.0e-6)
    args = parser.parse_args()

    base = CircularCouette(1.0, 2.0, 1.0, 0.5**1.5)
    solver = TaylorCouetteVPMRIDNSJax(
        base,
        B0=args.B0,
        nu=args.nu,
        eta_mag=args.eta_mag,
        Nr=args.Nr,
        Ntheta=args.Ntheta,
        Nz=args.Nz,
        dt=args.dt,
        family=args.family,
        dealias=args.dealias,
        magnetic_bc=args.magnetic_bc,
    )
    state, eigenvalue = solver.seed_linear_eigenmode(
        m=args.m, kz_mode=args.kz_mode, amp=args.amp
    )
    print(f"seeded eigenvalue: {eigenvalue.real:+.6e} {eigenvalue.imag:+.3e}i")
    state = solver.solve(state, args.steps)
    diag = solver.diagnostics(state)
    print(" ".join(f"{key}={float(value):.6e}" for key, value in diag.items()))


if __name__ == "__main__":
    main()
