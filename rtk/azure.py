import os
import pandas as pd

# azureml
from azureml.core import Workspace
from azureml.core.dataset import Dataset

# rtk
from rtk import DEFAULT_DATA_PATH
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

    if from_config:
        ws = Workspace.from_config()

    else:
        ws = Workspace(**kwargs)

    logger.debug("Workspace: {}".format(ws.name))
    return ws


def load_patient_dataset(
    ws: Workspace,
    patient_data_name: str,
    patient_data_version="latest",
    data_dir: os.PathLike = DEFAULT_DATA_PATH,
    pandas_read_fn: callable = pd.read_csv,
    **kwargs,
) -> pd.DataFrame:
    """
    Load a patient dataset from AzureML. If the dataset is not found locally, it will be downloaded from AzureML and saved to the local cache.

    ## Args:
    * `ws` (`Workspace`): The AzureML workspace.
    * `patient_dataset_name` (`str`): The name of the patient dataset.
    * `patient_dataset_version` (`str`, optional): The version of the patient dataset. Defaults to `"latest"`.
    * `data_dir` (`os.PathLike`, optional): The path to the data directory. Defaults to `DEFAULT_DATA_PATH`.
    * `pandas_read_fn` (`callable`, optional): The function to use to read the CSV file. Defaults to `pd.read_csv`.
    * `**kwargs`: Keyword arguments for `pandas_read_fn`.
    ## Returns:
    * `pd.DataFrame`: The patient dataset.
    """

    console.log(f"Patient dataset: '{patient_data_name}'")
    _patients_csv_path = os.path.join(
        data_dir, "patients", f"{patient_data_name}:{patient_data_version}.csv"
    )
    patients_csv_path = os.path.abspath(_patients_csv_path)
    patient_df: pd.DataFrame
    try:
        logger.debug(
            f"Attempting to load patient dataset from: '{patients_csv_path}'..."
        )
        patient_df: pd.DataFrame = pandas_read_fn(patients_csv_path, **kwargs)

    except FileNotFoundError:
        logger.warning(
            f"Patient dataset '{patient_data_name}' not found. Downloading from AzureML..."
        )
        patient_df: pd.DataFrame = Dataset.get_by_name(
            ws, name=patient_data_name, version=patient_data_version
        ).to_pandas_dataframe()
        os.makedirs(os.path.dirname(patients_csv_path), exist_ok=True)
        patient_df.to_csv(patients_csv_path, index=False)

    return patient_df


def load_scan_dataset(
    ws: Workspace,
    scan_dataset_name: str,
    scan_dataset_version="latest",
    data_dir: os.PathLike = DEFAULT_DATA_PATH,
    mount=True,
):
    """
    Load a scan dataset from AzureML. If the dataset is not found locally, it will be downloaded from AzureML and saved to the local cache.

    ## Args:
    * `ws` (`Workspace`): The AzureML workspace.
    * `scan_dataset_name` (`str`): The name of the scan dataset.
    * `scan_dataset_version` (`str`, optional): The version of the scan dataset. Defaults to `"latest"`.
    * `data_dir` (`os.PathLike`, optional): The path to the data directory. Defaults to `DEFAULT_DATA_PATH`.
    * `mount` (`bool`, optional): Whether to mount the dataset or download it. Defaults to `True`.
    """

    console.log(f"Scan dataset: '{scan_dataset_name}'\n")

    scan_dataset: Dataset = Dataset.get_by_name(
        ws, name=scan_dataset_name, version=scan_dataset_version
    )
    if mount:
        scan_mount = scan_dataset.mount()
        console.log(f"Mounting scan dataset to '{scan_mount.mount_point}'.")
        return scan_mount
    else:
        target_path = os.path.join(
            data_dir, "scans", f"{scan_dataset_name}:{scan_dataset_version}"
        )
        console.log(f"Downloading scan dataset to '{data_dir}'.")
        scan_dataset.download(target_path=target_path, overwrite=True)
        return scan_dataset
