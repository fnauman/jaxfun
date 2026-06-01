def test_generalized_eig_filters_singular_mass_eigenvalues():
    import numpy as np

    from jaxfun.la import generalized_eig

    L = np.diag([2.0, 3.0])
    M = np.diag([1.0, 0.0])

    w = generalized_eig(L, M)

    assert w.shape == (1,)
    assert np.allclose(w, [2.0])
