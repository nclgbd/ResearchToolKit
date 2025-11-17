"""
Tests for the `rtk.config` module.
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

    def test_create_open_clip_encoder(
        self,
        test_config: DictConfig,
    ):
        """Test the initialization of the Encoder class."""
        encoder: models.Encoder = models.create_open_clip_model(test_config)
        assert isinstance(encoder, models.Encoder)
