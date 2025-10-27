"""
For quick debugging and testing.
"""
from dotenv import load_dotenv
from rich import pretty, traceback
from rich import inspect as rich_inspect

from rtk.utils import get_console, get_logger

logger = get_logger(__name__)
console = get_console()


def install(
    max_depth: int = 3,
    max_length: int = 7,
    show_locals: bool = False,
    _pretty: bool = True,
    _traceback: bool = True,
):
    """
    Installs the rich traceback hook and pretty install.

    ## Args:
        `max_depth` (`int`, optional): Defaults to `3`.
        `max_length` (`int`, optional): Defaults to `7`.
        `show_locals` (`bool`, optional): Defaults to `True`.
    """
    logger.debug("Installing rich.pretty and rich.traceback")
    if _pretty:
        pretty.install(max_depth=max_depth, max_length=max_length, console=console)
    if _traceback:
        traceback.install(show_locals=show_locals, console=console)


def prepare_console(**kwargs):
    from rtk.utils import login

    install(**kwargs)
    ws = login()
    console = console
    console.clear()

    return ws, console


def inspect(obj, *args, private=True, methods=True, **kwargs):
    rich_inspect(obj, *args, private=private, methods=methods, **kwargs)
