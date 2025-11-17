import os
import pytest

from rtk.utils import get_logger, set_hydra_configuration

logger = get_logger(f"rtk.tests.{os.path.basename(__file__)}")


@pytest.fixture
def test_config_name():
    """Fixture for the test configuration name."""
    return "tests"


@pytest.fixture
def test_config_dir():
    """Fixture for the test configuration directory."""
    config_path = os.path.abspath("configs")
    logger.debug(f"Using config path: {config_path}")
    return config_path


@pytest.fixture
def test_config(test_config_name: str, test_config_dir: os.PathLike):
    """Fixture for the rtk configuration."""
    return set_hydra_configuration(
        config_name=test_config_name,
        init_method_kwargs={"config_dir": test_config_dir},
    )
