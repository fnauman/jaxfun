from production.observables import (
    canonicalize_scalars,
    energy_convention_for_spec,
    expected_divergence_keys,
)


def test_hydro_cheap_uses_divergence_l2():
    assert expected_divergence_keys(
        geometry="pcf",
        physics="hydro",
        artifact_id="pcf_hydro_laminar_v1",
    ) == ("divergence_l2",)
    scalars = canonicalize_scalars(
        {"Epert": 1.0, "divL2": 2.0},
        geometry="pcf",
        physics="hydro",
        artifact_id="pcf_hydro_laminar_v1",
    )
    assert scalars["kinetic_energy"] == 1.0
    assert scalars["divergence_l2"] == 2.0


def test_pcf_mhd_cheap_uses_u_and_b_l2_divergence_keys():
    assert expected_divergence_keys(
        geometry="pcf",
        physics="mhd",
        artifact_id="pcf_mhd_conducting_v1",
    ) == ("divergence_u_l2", "divergence_b_l2")
    scalars = canonicalize_scalars(
        {"divu_l2": 1.0e-12, "divB_L2": 2.0e-12, "Emag": 0.5},
        geometry="pcf",
        physics="mhd",
        artifact_id="pcf_mhd_conducting_v1",
    )
    assert scalars["divergence_u_l2"] == 1.0e-12
    assert scalars["divergence_b_l2"] == 2.0e-12
    assert scalars["magnetic_energy"] == 0.5


def test_tc_mhd_cheap_uses_only_b_l2_divergence_key():
    assert expected_divergence_keys(
        geometry="taylor_couette",
        physics="mhd",
        artifact_id="taylor_couette_mhd_conducting_v1",
    ) == ("divergence_b_l2",)
    scalars = canonicalize_scalars(
        {"divu_l2": 3.0, "divb_l2": 4.0},
        geometry="taylor_couette",
        physics="mhd",
        artifact_id="taylor_couette_mhd_conducting_v1",
    )
    assert "divergence_u_l2" not in scalars
    assert scalars["divergence_b_l2"] == 4.0


def test_dns_goldens_drop_l2_suffixes():
    assert expected_divergence_keys(
        geometry="pcf",
        physics="mri",
        artifact_id="pcf_mri_primitive_dns_v1",
    ) == ("divergence_u", "divergence_b")
    scalars = canonicalize_scalars(
        {"divu_l2": 1.0e-10, "divb_l2": 2.0e-10, "growth_rate": 0.4},
        geometry="pcf",
        physics="mri",
        artifact_id="pcf_mri_primitive_dns_v1",
    )
    assert scalars["divergence_u"] == 1.0e-10
    assert scalars["divergence_b"] == 2.0e-10
    assert scalars["growth_rate"] == 0.4


def test_energy_convention_follows_solver_family():
    assert (
        energy_convention_for_spec(
            {
                "geometry": "taylor_couette",
                "physics": "mri",
                "representation": "vector_potential",
            }
        )
        == "half_integral_abs2_annulus"
    )
    assert (
        energy_convention_for_spec(
            {
                "geometry": "pcf",
                "physics": "mri",
                "representation": "vector_potential",
            }
        )
        == "integral_abs2"
    )
