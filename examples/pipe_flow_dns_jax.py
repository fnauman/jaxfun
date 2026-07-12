"""Axisymmetric pipe-flow production helpers for jaxfun.

The full shenfun pipe DNS uses a 3D cylindrical saddle-point solve with an
axis-regular radial basis.  The two committed production pipe goldens are
axisymmetric hydro checks, so this module implements the same scalar contract
without importing shenfun:

* Hagen-Poiseuille: exact parabolic profile.
* Womersley: Bessel-mode Crank-Nicolson recurrence with midpoint forcing,
  matching the shenfun CNAB2 linear step used to generate the golden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import jn_zeros, jv


@dataclass(frozen=True)
class PipeDiagnostics:
    """Scalar diagnostics for an axisymmetric pipe hydro state."""

    kinetic_energy: float
    flow_rate: float
    divergence_l2: float = 0.0
    forcing_phase: float | None = None
    flow_rate_exact: float | None = None

    def scalars(self) -> dict[str, float]:
        out = {
            "kinetic_energy": self.kinetic_energy,
            "flow_rate": self.flow_rate,
            "divergence_l2": self.divergence_l2,
        }
        if self.forcing_phase is not None:
            out["forcing_phase"] = self.forcing_phase
        if self.flow_rate_exact is not None:
            out["flow_rate_exact"] = self.flow_rate_exact
        return out


def hagen_poiseuille_profile(
    r: np.ndarray, *, fz: float, nu: float, radius: float
) -> np.ndarray:
    """Return ``u_z(r) = fz * (R^2 - r^2) / (4 nu)``."""

    return (fz / (4.0 * nu)) * (radius**2 - np.asarray(r, dtype=float) ** 2)


def hagen_poiseuille_flow_rate(*, fz: float, nu: float, radius: float) -> float:
    """Exact volumetric flow rate for a forced pipe."""

    return math.pi * radius**4 * fz / (8.0 * nu)


def hagen_poiseuille_diagnostics(
    *,
    fz: float,
    nu: float,
    radius: float,
    length: float,
) -> PipeDiagnostics:
    """Exact axisymmetric Hagen-Poiseuille diagnostics."""

    q = hagen_poiseuille_flow_rate(fz=fz, nu=nu, radius=radius)
    kinetic = math.pi * length * radius**6 * fz**2 / (96.0 * nu**2)
    return PipeDiagnostics(
        kinetic_energy=kinetic,
        flow_rate=q,
        flow_rate_exact=q,
    )


def womersley_profile(
    r: np.ndarray,
    *,
    t: float,
    amplitude: float,
    omega: float,
    nu: float,
    radius: float,
) -> np.ndarray:
    """Continuous Womersley profile for ``f_z(t)=amplitude*cos(omega*t)``."""

    alpha = radius * math.sqrt(omega / nu)
    i32 = np.exp(1j * 3.0 * np.pi / 4.0)
    j0a = jv(0, i32 * alpha)
    prof = 1.0 - jv(0, i32 * alpha * np.asarray(r, dtype=float) / radius) / j0a
    value = (amplitude / (1j * omega)) * prof * np.exp(1j * omega * t)
    return np.real(value)


def womersley_cn_diagnostics(
    *,
    amplitude: float,
    omega: float,
    nu: float,
    radius: float,
    length: float,
    dt: float,
    final_time: float,
    n_modes: int = 512,
) -> tuple[PipeDiagnostics, PipeDiagnostics]:
    """Return initial and final Womersley diagnostics using shenfun's CN step.

    The axisymmetric linear pipe equation is diagonal in the regular
    ``J0(alpha_n r/R)`` basis.  Shenfun's pipe DNS evaluates the oscillatory body
    force at ``t^(n+1/2)`` and uses Crank-Nicolson for viscosity.  Replaying that
    recurrence in Bessel modes reproduces the committed pipe Womersley golden
    without requiring the full shenfun basis stack.
    """

    zeros = jn_zeros(0, n_modes)
    j1 = jv(1, zeros)
    norm = 0.5 * radius**2 * j1**2
    forcing_coeff = 2.0 / (zeros * j1)
    lambdas = nu * zeros**2 / radius**2
    coeff = _womersley_initial_coefficients(
        zeros,
        norm,
        amplitude=amplitude,
        omega=omega,
        nu=nu,
        radius=radius,
    )
    initial = _bessel_diagnostics(
        coeff,
        zeros,
        j1,
        norm,
        radius=radius,
        length=length,
        forcing_phase=math.cos(0.0),
    )

    n_steps = int(round(final_time / dt))
    t = 0.0
    for _ in range(n_steps):
        force = amplitude * math.cos(omega * (t + 0.5 * dt))
        coeff = ((1.0 - 0.5 * dt * lambdas) * coeff + dt * forcing_coeff * force) / (
            1.0 + 0.5 * dt * lambdas
        )
        t += dt

    final = _bessel_diagnostics(
        coeff,
        zeros,
        j1,
        norm,
        radius=radius,
        length=length,
        forcing_phase=math.cos(omega * t),
    )
    return initial, final


def _womersley_initial_coefficients(
    zeros: np.ndarray,
    norm: np.ndarray,
    *,
    amplitude: float,
    omega: float,
    nu: float,
    radius: float,
) -> np.ndarray:
    grid, weights = np.polynomial.legendre.leggauss(2048)
    r = 0.5 * radius * (grid + 1.0)
    wr = 0.5 * radius * weights
    profile = womersley_profile(
        r,
        t=0.0,
        amplitude=amplitude,
        omega=omega,
        nu=nu,
        radius=radius,
    )
    basis = jv(0, np.outer(r / radius, zeros))
    integrals = (profile[:, None] * basis * r[:, None] * wr[:, None]).sum(axis=0)
    return integrals / norm


def _bessel_diagnostics(
    coeff: np.ndarray,
    zeros: np.ndarray,
    j1: np.ndarray,
    norm: np.ndarray,
    *,
    radius: float,
    length: float,
    forcing_phase: float,
) -> PipeDiagnostics:
    radial_integral = radius**2 * np.sum(coeff * j1 / zeros)
    flow_rate = float(2.0 * math.pi * radial_integral)
    kinetic = float(0.5 * (2.0 * math.pi) * length * np.sum(coeff**2 * norm))
    return PipeDiagnostics(
        kinetic_energy=kinetic,
        flow_rate=flow_rate,
        forcing_phase=float(forcing_phase),
    )
