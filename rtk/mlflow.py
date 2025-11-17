#
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# mlflow
import mlflow
from mlflow.data.huggingface_dataset import *

# huggingface
from datasets import DatasetDict

# rtk
from rtk.utils import get_logger

logger = get_logger(__name__)


def prepare_mlflow(args: DictConfig, **kwargs):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = kwargs.get(
        "experiment_name", args.get("experiment_name", HydraConfig.get().job.name)
    )

    postfix = "-test" if args.get("dry_run", True) else ""
    experiment_name = f"{experiment_name}{postfix}"
    mlflow.set_experiment(experiment_name)

    mlflow_args = {
        "experiment_name": experiment_name,
    }
    return mlflow_args
