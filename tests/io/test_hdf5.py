import jax
import jax.numpy as jnp
import pytest

from jaxfun.galerkin import Array, Chebyshev, JAXFunction, TensorProduct
from jaxfun.io import (
    Cadence,
    cadence_due,
    generate_xdmf,
    read_checkpoint,
    run_with_cadence,
    write_checkpoint,
    write_uniform_snapshot,
)


def test_checkpoint_roundtrip_preserves_nested_coefficients(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "restart.h5"
    fields = {
        "u": (
            jnp.arange(4.0),
            jnp.arange(4.0, 8.0) + 1j * jnp.arange(4.0),
        ),
        "meta": {"g": jnp.eye(2)},
    }

    write_checkpoint(path, fields, t=0.25, tstep=7, attrs={"solver": "unit"})
    record = read_checkpoint(path)

    assert record.t == 0.25
    assert record.tstep == 7
    assert record.attrs["solver"] == "unit"
    assert jnp.array_equal(record.fields["u"][0], fields["u"][0])
    assert jnp.array_equal(record.fields["u"][1], fields["u"][1])
    assert jnp.array_equal(record.fields["meta"]["g"], fields["meta"]["g"])


def test_uniform_snapshot_from_jaxfunction_and_physical_array(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fields.h5"
    C = Chebyshev.Chebyshev(6)
    T = TensorProduct(C, C)
    coeffs = jnp.zeros(T.num_dofs).at[1, 2].set(0.5)
    function = JAXFunction(coeffs, T, name="u")
    physical = Array.from_coefficients(coeffs, T, name="u_phys")

    write_uniform_snapshot(
        path,
        {"u": function, "u_phys": physical},
        t=0.5,
        tstep=2,
        N={"u": (8, 8)},
    )

    with h5py.File(path, "r") as h5:
        u = jnp.asarray(h5["snapshots/2/fields/u"][()])
        u_phys = jnp.asarray(h5["snapshots/2/fields/u_phys"][()])
    assert u.shape == (8, 8)
    assert jnp.allclose(u, T.evaluate_mesh(coeffs, kind="uniform", N=(8, 8)))
    assert jnp.allclose(u_phys, physical.backward())

    xdmf = generate_xdmf(path)
    text = xdmf.read_text()
    assert "fields.h5:/snapshots/2/fields/u" in text
    assert "u_phys" in text


def test_checkpoint_restart_continues_bit_identically(tmp_path):
    pytest.importorskip("h5py")
    path = tmp_path / "restart.h5"

    def step(state):
        u, g = state
        return (1.25 * u - 0.5 * g, g + 0.125 * u)

    state = (jnp.arange(5.0), jnp.linspace(1.0, 2.0, 5))
    for _ in range(4):
        state = step(state)
    write_checkpoint(path, {"state": state}, t=0.4, tstep=4)

    restarted = read_checkpoint(path).fields["state"]
    continued = restarted
    direct = state
    for _ in range(20):
        continued = step(continued)
        direct = step(direct)

    assert jnp.array_equal(continued[0], direct[0])
    assert jnp.array_equal(continued[1], direct[1])



def test_run_with_cadence_hits_exact_boundaries():
    @jax.jit(static_argnums=1)
    def advance(state, nsteps):
        def body(_, value):
            return 1.25 * value + 0.5

        return jax.lax.fori_loop(0, nsteps, body, state)

    events = []

    def diagnostics(state):
        return {"norm": jnp.linalg.norm(state)}

    def on_diagnostics(t, tstep, diag):
        events.append(("diag", t, tstep, float(diag["norm"])))

    def on_snapshot(t, tstep, state):
        events.append(("snap", t, tstep, float(state[0])))

    def on_checkpoint(t, tstep, state):
        events.append(("ckpt", t, tstep, float(state[0])))

    out = run_with_cadence(
        advance,
        jnp.arange(3.0),
        steps=7,
        dt=0.1,
        block_size=5,
        cadence=Cadence(diagnostics_every=2, snapshot_every=3, checkpoint_every=4),
        diagnostics=diagnostics,
        on_diagnostics=on_diagnostics,
        on_snapshot=on_snapshot,
        on_checkpoint=on_checkpoint,
    )

    direct = jnp.arange(3.0)
    for _ in range(7):
        direct = 1.25 * direct + 0.5
    assert jnp.allclose(out, direct)
    assert [(kind, tstep) for kind, _, tstep, _ in events] == [
        ("diag", 2),
        ("snap", 3),
        ("diag", 4),
        ("ckpt", 4),
        ("diag", 6),
        ("snap", 6),
    ]
    assert cadence_due(8, 4)
    assert not cadence_due(7, 4)
