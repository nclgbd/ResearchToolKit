"""
Tests for the `rtk.models` module.
"""

import pytest
from collections import Counter
from omegaconf import OmegaConf, DictConfig, ListConfig

# torch
import torch

# rtk
from rtk import models


class TestModels:

    @pytest.fixture
    def model_config(self, test_config: DictConfig) -> DictConfig:
        """Fixture for the model configuration."""
        return test_config.models

    def test_load_hf_model(
        self,
        test_config: DictConfig,
    ):
        """Test the initialization of the Encoder class."""
        models.load_hf_model(test_config)
        assert True
