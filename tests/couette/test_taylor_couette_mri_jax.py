import jax
import numpy as np
import pytest

from examples.taylor_couette_linear_jax import CircularCouette
from examples.taylor_couette_mri_jax import TaylorCouetteMRIJax, mri_keplerian_optimum


def _keplerian_base():
    eta = 0.5
    return CircularCouette(1.0, 2.0, 1.0, eta**1.5)


def test_local_keplerian_mri_optimum():
    opt = mri_keplerian_optimum()

    assert opt["s_max_over_Omega"] == pytest.approx(0.75, rel=1e-3)
    assert opt["wa2_opt_over_O2"] == pytest.approx(15.0 / 16.0, rel=2e-3)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="shenfun parity reference requires x64"
)
@pytest.mark.parametrize(
    ("magnetic_bc", "expected"),
    [
        (
            "conducting",
            np.array(
                [
                    0.25628761535339467 + 1.588556792647881e-16j,
                    0.09823607358977877 + 1.0327826189548287e-17j,
                    -0.009000000000000001 + 0j,
                    -0.01921903715666608 + 0j,
                    -0.03595125026732483 + 0.622060190485423j,
                    -0.035951250267324875 - 0.6220601904854233j,
                ]
            ),
        ),
        (
            "insulating",
            np.array(
                [
                    0.25995005500337837 + 5.232643537701433e-17j,
                    0.11307480123119808 + 8.46344257953988e-16j,
                    -0.01831294114132462 + 1.2614175475477918e-16j,
                    -0.036777045403937884 - 0.6214752030458406j,
                    -0.03677704540393851 + 0.6214752030458399j,
                    -0.07059336855644499 + 0.49761699975570717j,
                ]
            ),
        ),
    ],
)
def test_taylor_couette_mri_matches_shenfun_reference_spectrum(magnetic_bc, expected):
    solver = TaylorCouetteMRIJax(
        _keplerian_base(),
        B0=0.1,
        nu=0.001,
        eta_mag=0.001,
        N=12,
        family="L",
        magnetic_bc=magnetic_bc,
    )

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)

    assert np.allclose(w, expected, rtol=1e-11, atol=1e-11)
    assert solver.growth_rate(m=0, kz=3.0) == pytest.approx(expected[0].real, abs=1e-12)
