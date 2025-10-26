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
from logging import Logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown


__all__ = [
    # "COLOR_LOGGER_FORMAT",
    # "LOG_TIME_FORMAT",
    # "_console",
    "get_console",
    "get_logger",
    "hydra_instantiate",
    "intro",
    "rich_handler",
    "strip_target",
    "yaml_to_configuration",
    "yaml_to_namespace",
]

# LOGGING_DIR = "logs"
LOG_TIME_FORMAT = "[%X]".strip()
COLOR_LOGGER_FORMAT: logging.Formatter = ColoredFormatter(
    fmt="%(name)s - %(message)s".strip(),
    # datefmt=LOG_TIME_FORMAT,
    reset=False,
)
# Color settings
rich_handler = RichHandler(
    # rich_tracebacks=True,
    # console=console,
    log_time_format=LOG_TIME_FORMAT,
)
rich_handler.setFormatter(COLOR_LOGGER_FORMAT)
# logging.basicConfig(
#     level=logging.INFO, datefmt="[%X]", handlers=[rich_handler], force=True
# )


def get_console(**kwargs) -> Console:
    """
    Gets the rich console object.

    ## Returns:
    * `Console`: Rich console object.

    """

    # log_file = kwargs.get("file", None)
    # if log_file:
    #     file_io = open(log_file, "a")
    #     kwargs["file"] = file_io
    return kwargs.get("console", Console(record=True, **kwargs))


_console = get_console()


def intro(
    args: DictConfig, title: str = "SigLIP Training", console: Console = _console
):

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

    # # File settings
    # # curr_dir = os.getcwd()
    # # os.makedirs("logs", exist_ok=True)
    # # file_handler = logging.FileHandler(f"logs/{name}.log")
    # # file_handler.setFormatter(COLOR_LOGGER_FORMAT)
    # # logger.addHandler(file_handler)

    # logger.addHandler(rich_handler)
    # logger.propagate = False

    return logger


def hydra_instantiate(args: DictConfig, **kwargs):
    """
    Instantiates an object from a configuration.

    ## Args:
    * `cfg` (`DictConfig`): The Hydra config.
    * `**kwargs`: Keyword arguments for the object.
    ## Returns:
    * `Any`: The instantiated class.
    """
    # target_class_name = cfg["_target_"].split(".")[-1]
    # _logger.debug(
    #     "Instantiating object '{}' from configuration".format(target_class_name)
    # )
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


def strip_target(_dict: dict, lower=False):
    target_name: str = _dict["_target_"].split(".")[-1]
    if lower:
        target_name = target_name.lower()
    return target_name


# _logger = get_logger(__name__)
