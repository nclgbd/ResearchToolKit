import os
import pandas as pd

# azureml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# rtk
from rtk import console
from rtk.utils import get_logger

logger = get_logger(__name__, console=console)


def login(
    from_config=True,
    **kwargs,
):
    if from_config:
        try:
            azml = MLClient.from_config(credential=DefaultAzureCredential())
            logger.info("Successfully connected to the Azure ML workspace.")
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    else:
        raise NotImplementedError("Manual login not implemented yet.")

    return azml


def get_azml_client(azml: MLClient = login()):
    """Get Azure ML client using default credentials and config file."""
    mlflow_tracking_uri = azml.workspaces.get(azml.workspace_name).mlflow_tracking_uri

    logger.debug(f"Azure MLflow Tracking URI: {mlflow_tracking_uri}")
    return azml
