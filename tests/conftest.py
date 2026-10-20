import matplotlib
import pytest

matplotlib.use("Agg")

from scales import config, SupplyChainBackbone, ScbConverter, SupplyChain


@pytest.fixture(autouse=True)
def output_root(tmp_path):
    """Every test writes its default outputs into its own temporary folder."""
    config.set_output_root(tmp_path)
    yield tmp_path
    config._output_root = None


@pytest.fixture
def backbone():
    return SupplyChainBackbone(size=[6, 12, 6, 2], prune=True, beta=4, w=0.1, betawin=3, betawout=10, seed=7)


@pytest.fixture
def converter(backbone):
    return ScbConverter(backbone, seed=7)


@pytest.fixture
def supply_chain(converter):
    return SupplyChain.from_converter(converter, duration=40, supply=1e9, demand=10)
