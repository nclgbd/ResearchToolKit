import os
import pandas as pd

# rtk
from rtk.utils import get_logger, get_console

logger = get_logger(__name__)
console = get_console()


def login(
    from_config=True,
    **kwargs,
):
    """
    Login to AzureML workspace. If path is provided, will load from the specified config file.

    ## Args:
    * `from_config` (`bool`, optional): Whether to load from config file or provide the `subscription_id`. Defaults to `True`.
    * `kwargs` (`dict`): Keyword arguments for `Workspace()`.

    ## Returns:
    * `Workspace`: AzureML workspace object.
    """
    from azureml.core import Workspace
    from azureml.core.dataset import Dataset

    if from_config:
        ws = Workspace.from_config()

    else:
        ws = Workspace(**kwargs)

    logger.debug("Workspace: {}".format(ws.name))
    return ws
