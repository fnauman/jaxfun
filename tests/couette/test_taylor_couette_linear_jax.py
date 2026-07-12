import jax
import numpy as np
import pytest

from examples.taylor_couette_linear_jax import CircularCouette, TaylorCouetteLinearJax


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="recorded golden spectrum requires x64"
)
def test_taylor_couette_linear_matches_recorded_golden_spectrum():
    solver = TaylorCouetteLinearJax(CircularCouette(), nu=0.002, N=12, family="L")

    w, _ = solver.eigs(m=0, kz=3.0, n_return=6)

    expected = np.array(
        [
            0.36073352898670064 + 4.797857668758156e-22j,
            0.10328890448038253 - 7.084450497656995e-18j,
            -0.07718676501148072 + 0j,
            -0.23738479237266985 + 0j,
            -0.3467838262905014 + 0j,
            -0.3833172863333123 + 0j,
        ]
    )
    assert np.allclose(w, expected, rtol=1e-11, atol=1e-11)
    assert solver.growth_rate(m=0, kz=3.0) == pytest.approx(expected[0].real, abs=1e-12)
