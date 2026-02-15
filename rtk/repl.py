"""
For quick debugging and testing.
"""

from dotenv import load_dotenv
from rich import pretty, traceback
from rich import inspect as rich_inspect

from rtk import console
from rtk.utils import get_logger

logger = get_logger(__name__, console=console)


def install(
    max_depth: int = 3,
    max_length: int = 7,
    show_locals: bool = False,
    _pretty: bool = True,
    _traceback: bool = True,
):
    logger.debug("Installing rich.pretty and rich.traceback")
    if _pretty:
        pretty.install(max_depth=max_depth, max_length=max_length, console=console)
    if _traceback:
        traceback.install(show_locals=show_locals, console=console)


def prepare_console(**kwargs):
    from rtk.utils import login

    install(**kwargs)
    try:
        ws = login()
    except Exception as e:
        logger.error(f"Login failed: {e}")
        ws = None
    console.clear()

    return ws, console


def inspect(obj, *args, private=True, methods=True, **kwargs):
    rich_inspect(obj, *args, private=private, methods=methods, **kwargs)
