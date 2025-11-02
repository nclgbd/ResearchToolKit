"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# imports
import hydra
import logging
import os
import textwrap
import yaml
from argparse import Namespace
from colorlog import ColoredFormatter
from dotenv import load_dotenv
from logging import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown


# LOGGING_DIR = "logs"
LOG_TIME_FORMAT = "[%X]".strip()
COLOR_LOGGER_FORMAT: logging.Formatter = ColoredFormatter(
    fmt="%(name)s - %(message)s".strip(),
    reset=False,
)
# Color settings
rich_handler = RichHandler(
    # rich_tracebacks=True,
    # console=console,
    log_time_format=LOG_TIME_FORMAT,
)
rich_handler.setFormatter(COLOR_LOGGER_FORMAT)


def get_console(**kwargs) -> Console:
    """
    Gets the rich console object.

    ## Returns:
    * `Console`: Rich console object.

    """
    return kwargs.get("console", Console(record=True, **kwargs))


_console = get_console()


def intro(
    args: DictConfig, title: str = "SigLIP Training", console: Console = _console
):

    env_file = args.get("env_file", "../.env")
    load_dotenv(env_file)
    console.clear()
    console.print(Markdown(f"# {title}"))
    config_str = OmegaConf.to_yaml(args, resolve=True)
    console.print(Markdown("## Configuration\n\n"))
    config_str = textwrap.dedent(
        f"""
        ```yaml
{config_str}
        """
    ).strip()
    console.print(Markdown(config_str))
    return config_str


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
