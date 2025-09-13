"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# imports
import logging
import os
import yaml
import hydra
from copy import deepcopy
from argparse import Namespace
from colorlog import ColoredFormatter
from logging import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler

__all__ = [
    "_console",
    "_logger",
    "COLOR_LOGGER_FORMAT",
    "get_console",
    "get_logger",
    "login",
    "repl",
]

LOG_TIME_FORMAT = "[%X]"
COLOR_LOGGER_FORMAT: logging.Formatter = ColoredFormatter(
    fmt="%(name)s: %(message)s", datefmt=LOG_TIME_FORMAT
)


def get_console(**kwargs) -> Console:
    """
    Gets the rich console object.

    ## Returns:
    * `Console`: Rich console object.

    """

    return kwargs.get("console", Console(record=True, **kwargs))


_console = get_console()


def get_logger(name: str = None, level: int = logging.INFO):
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
    rich_handler = RichHandler(
        rich_tracebacks=True,
        level=level,
        log_time_format=LOG_TIME_FORMAT,
        console=_console,
    )
    rich_handler.setFormatter(COLOR_LOGGER_FORMAT)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


def hydra_instantiate(cfg: DictConfig, **kwargs):
    """
    Instantiates an object from a configuration.

    ## Args:
    * `cfg` (`DictConfig`): The Hydra config.
    * `**kwargs`: Keyword arguments for the object.
    ## Returns:
    * `Any`: The instantiated class.
    """
    target_class_name = cfg["_target_"].split(".")[-1]
    _logger.debug(
        "Instantiating object '{}' from configuration".format(target_class_name)
    )
    return hydra.utils.instantiate(cfg, **kwargs)


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


def strip_target(_dict: dict, lower=False):
    target_name: str = _dict["_target_"].split(".")[-1]
    if lower:
        target_name = target_name.lower()
    return target_name


_logger = get_logger(__name__)
