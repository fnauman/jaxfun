import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def clear_jax_caches_after_module():
    yield

    import jax

    jax.clear_caches()


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--float64",
        action="store_true",
        default=True,
        help="Deprecated compatibility flag; tests run in x64 by default.",
    )
    parser.addoption(
        "--num-devices",
        type=int,
        default=1,
        help="Number of JAX CPU devices to expose (must be >1 to run spmd-marked tests).",  # noqa: E501
    )


def pytest_configure(config) -> None:
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if not hasattr(config, "workerinput"):
        fd, sentinel = tempfile.mkstemp(
            prefix="jaxfun_shenfun_parity_", suffix=".sentinel"
        )
        os.close(fd)
        config._jaxfun_shenfun_parity_sentinel = Path(sentinel)
        config._jaxfun_live_shenfun_selected = 0
        os.environ["JAXFUN_SHENFUN_PARITY_SENTINEL"] = sentinel
    import jax

    jax.config.update("jax_enable_x64", True)
    n = config.getoption("--num-devices")
    if n > 1:
        jax.config.update("jax_num_cpu_devices", n)
    os.environ["PYTEST"] = "True"


def _count_live_shenfun_items(items) -> int:
    return sum(1 for item in items if "live_shenfun" in item.keywords)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    n = config.getoption("--num-devices")
    if n > 1:
        selected = [item for item in items if "spmd" in item.keywords]
        deselected = [item for item in items if "spmd" not in item.keywords]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    selected_count = _count_live_shenfun_items(items)
    if hasattr(config, "workerinput"):
        config.workeroutput["jaxfun_live_shenfun_selected"] = selected_count
    else:
        config._jaxfun_live_shenfun_selected = selected_count


def pytest_runtest_setup(item) -> None:
    if "spmd" in item.keywords:
        import jax

        if jax.device_count() < 2:
            pytest.skip("spmd tests require --num-devices=2 (or more)")


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    selected = int(node.workeroutput.get("jaxfun_live_shenfun_selected", 0))
    node.config._jaxfun_live_shenfun_selected = (
        getattr(node.config, "_jaxfun_live_shenfun_selected", 0) + selected
    )


def pytest_sessionfinish(session, exitstatus) -> None:
    config = session.config
    if hasattr(config, "workerinput"):
        return
    expected = int(getattr(config, "_jaxfun_live_shenfun_selected", 0))
    if expected == 0 or exitstatus != pytest.ExitCode.OK:
        return
    sentinel = getattr(config, "_jaxfun_shenfun_parity_sentinel", None)
    count = 0
    if sentinel is not None and sentinel.exists():
        count = len(
            [line for line in sentinel.read_text().splitlines() if line.strip()]
        )
    if count == 0:
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "live shenfun parity tests were selected, but no reference "
                "comparisons executed"
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config) -> None:
    os.environ.pop("PYTEST", None)
    if not hasattr(config, "workerinput"):
        os.environ.pop("JAXFUN_SHENFUN_PARITY_SENTINEL", None)
