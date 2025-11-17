from omegaconf import DictConfig


class TestConfig:
    def test_config(self, test_config: DictConfig):
        assert test_config is not None
