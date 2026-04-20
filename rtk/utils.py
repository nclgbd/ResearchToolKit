"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

import hydra
import logging
import os
import yaml
from argparse import Namespace
from dotenv import load_dotenv
from logging import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.markdown import Markdown

# hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def setup_torch_backends():
    """
    Configure PyTorch backends for optimal performance on Blackwell GPUs.

    Optimizations:
    - TF32: Enables TensorFloat-32 for matrix multiplications, providing ~3x speedup with minimal precision loss. Blackwell architecture has dedicated TF32 cores.
    - cuDNN benchmark: Runs multiple convolution algorithms to find the fastest one. Best when input sizes are consistent (as in this retrieval pipeline).
    - Flash/Memory-efficient attention: Uses optimized SDPA kernels when available.
    """
    import torch

    # Enable TF32 for matmuls (significant speedup on Ampere/Blackwell architecture)
    # TF32 uses 19 bits (10 mantissa) vs FP32's 32 bits, ~3x faster with <0.1% precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuner - benchmarks algorithms to find fastest for given input sizes
    # First iteration is slower (benchmarking), subsequent iterations are faster
    torch.backends.cudnn.benchmark = True

    # Disable deterministic mode for maximum performance
    # Set to True if exact reproducibility is required
    torch.backends.cudnn.deterministic = False

    # Enable optimized Scaled Dot-Product Attention kernels
    # Flash Attention: O(N) memory instead of O(N²), faster for long sequences
    # Memory-efficient: Good fallback when Flash Attention constraints aren't met
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    logger.info(
        f"PyTorch backends configured: TF32={torch.backends.cuda.matmul.allow_tf32}, "
        f"cuDNN benchmark={torch.backends.cudnn.benchmark}, "
        f"Flash SDP=enabled"
    )


def intro(args: DictConfig, title: str = "", console: Console = None):

    env_file = args.get("env_file", "../.env")
    load_dotenv(env_file)
    console.clear()
    console.print(Markdown(f"# {title}"))
    config_str = OmegaConf.to_yaml(args, resolve=True)
    console.print(Markdown("## Configuration\n\n"))
    config_str = f"```yaml\n{config_str}```"
    console.print(Markdown(config_str))
    return config_str


def get_console(**kwargs) -> Console:
    """
    Gets the rich console object.

    ## Returns:
    * `Console`: Rich console object.

    """
    return Console(record=True, **kwargs)


def get_logger(
    name: str = None,
    level: int = logging.INFO,
    console: Console = None,
) -> Logger:
    """
    Function to get a logger with a `RichHandler`. Sets up the logger with a custom format and a `StreamHandler`.

    ## Args:
    * `name` (`str`): The name of the logger. Defaults to `None`.
    * `level` (`int`): The level of the logger. Defaults to `logging.INFO`.

    ## Returns:
    * `logging.Logger`: The logger.
    """

    logger: Logger = logging.getLogger(name)
    logger.setLevel(level=level)

    return logger


logger = get_logger(__name__)


def set_hydra_configuration(
    config_name: str,
    init_method: callable = initialize_config_dir,
    init_method_kwargs: dict = {},
    **compose_kwargs,
) -> DictConfig:
    """
    Creates and returns a hydra configuration.

    ## Args:
    * `config_name` (`str`): The name of the config (usually the file name without the .yaml extension).
    * `init_method` (`function`, optional): The initialization method to use. Should be either [`initialize`, `initialize_config_module`, `initialize_config_dir`].
    Defaults to `initialize_config_dir`.
    * `init_method_kwargs` (`dict`, optional): Keyword arguments for the `init_method` function.
    * `compose_kwargs` (`dict`, optional): Keyword arguments for the `compose` function.

    ## Returns:
    * `DictConfig`: The hydra configuration.
    """
    logger.info(f"Creating configuration: '{config_name}'\n")
    GlobalHydra.instance().clear()
    init_method(version_base="1.1", **init_method_kwargs)
    conf: DictConfig = compose(config_name=config_name, **compose_kwargs)
    return conf


def hydra_instantiate(args: DictConfig, **kwargs):
    """
    Instantiates an object from a configuration.

    ## Args:
    * `args` (`DictConfig`): The Hydra config.
    * `**kwargs`: Keyword arguments for the object.
    ## Returns:
    * `Any`: The instantiated class.
    """
    return hydra.utils.instantiate(args, **kwargs)


def yaml_to_namespace(yaml_file: os.PathLike):
    """
    Converts a YAML file to a `Namespace` object.
    ## Args:
    * `yaml_file` (`os.PathLike`): The path to the YAML file.

    ## Returns:
    * `Namespace`: The `Namespace` object.

    """
    with open(yaml_file, "r") as f:
        return Namespace(**yaml.safe_load(f))


def yaml_to_configuration(file_path: str):
    cfg = OmegaConf.load(file_path)
    del cfg["defaults"]
    cfg = DictConfig(cfg)
    return cfg


def namespace_to_configuration(namespace: Namespace):
    return DictConfig(vars(namespace))


def strip_target(_dict: dict, lower=False):
    target_name: str = _dict["_target_"].split(".")[-1]
    if lower:
        target_name = target_name.lower()
    return target_name
