import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from hydra.utils import instantiate

from sklearn.preprocessing import MultiLabelBinarizer

# :huggingface:
from datasets import Dataset as HGFDataset

# sklearn
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# monai
import monai

# rtk
from rtk import *
from rtk._datasets import *
from rtk.config import *
from rtk.utils import get_logger, _console


logger = get_logger(__name__)
console = _console

MINORITY_CLASS = "Hernia"
MINORITY_CLASS_COUNT = 227
DATA_ENTRY_PATH = (
    "/home/nicoleg/workspaces/dissertation/.data/CHEST_XRAY_14/Data_Entry_2017.csv"
)
NIH_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def nih_get_target_counts(df: pd.DataFrame, target: str = ""):
    return Counter(",".join(df[target]).replace("|", ",").split(","))


def load_nih_dataset(
    cfg: ImageClassificationConfiguration = None,
    save_metadata=True,
    return_metadata=False,
    subset_to_positive_class=False,
    **kwargs,
):
    dataset_cfg: ImageDatasetConfiguration = kwargs.get("dataset_cfg", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.datasets

    scan_path = dataset_cfg.scan_data
    target = kwargs.get("target", dataset_cfg.target)

    preprocessing_cfg = kwargs.get("preprocessing_cfg", None)
    if preprocessing_cfg is None:
        preprocessing_cfg = dataset_cfg.preprocessing

    positive_class = kwargs.get("positive_class", None)
    if positive_class == None:
        positive_class = preprocessing_cfg.get(
            "positive_class", preprocessing_cfg.get("positive_class", "Pneumonia")
        )

    random_state = kwargs.get("random_state", None)
    if random_state is None:
        random_state = cfg.random_state

    nih_metadata = load_metadata(
        dataset_cfg.index,
        patient_data_name=dataset_cfg.patient_data,
        patient_data_version=dataset_cfg.patient_data_version,
    )

    # remove all of the negative class for diffusion
    if subset_to_positive_class:
        console.log("Removing all negative classes...")
        nih_metadata = nih_metadata[nih_metadata[positive_class] == 1]

    # split metadata
    with open(os.path.join(scan_path, "train_val_list.txt"), "r") as f:
        train_val_list = [idx.strip() for idx in f.readlines()]

    train_metadata = nih_metadata[nih_metadata.index.isin(train_val_list)]
    stratify = kwargs.get("stratify", train_metadata["class_conditioned_labels"]).values
    X = train_metadata.drop(columns=[target]).reset_index(drop=False)
    X_columns = X.columns
    X = X.values
    y = train_metadata[target].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.05, random_state=random_state, stratify=stratify
    )
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    # train split
    train_metadata = np.concatenate((X_train, y_train), axis=1)
    columns = X_columns.values.tolist()
    columns.append(target)
    train_metadata = pd.DataFrame(train_metadata, columns=columns)
    if preprocessing_cfg.use_sampling and subset_to_positive_class == False:
        train_metadata = resample_to_value(
            train_metadata,
            NIH_CLASS_NAMES,
            dataset_cfg=dataset_cfg,
            preprocessing_cfg=preprocessing_cfg,
            sampling_strategy=target,
            random_state=random_state,
        )
    train_class_counts = get_class_counts(train_metadata, NIH_CLASS_NAMES)
    console.log(f"Train class counts:\n{train_class_counts}")

    # val split
    val_metadata = np.concatenate((X_val, y_val), axis=1)
    val_metadata = pd.DataFrame(val_metadata, columns=columns)

    val_class_counts = get_class_counts(val_metadata, NIH_CLASS_NAMES)
    console.log(f"Validation class counts:\n{val_class_counts}")

    # test split
    with open(os.path.join(scan_path, "test_list.txt"), "r") as f:
        test_list = [idx.strip() for idx in f.readlines()]
    test_metadata = nih_metadata[nih_metadata.index.isin(test_list)]

    test_class_counts = get_class_counts(test_metadata, NIH_CLASS_NAMES)
    console.log(f"Test class counts:\n{test_class_counts}")

    if return_metadata:
        return (
            train_metadata,
            val_metadata,
            test_metadata,
        )

    # create datasets
    ## prepare transforms
    train_transforms = kwargs.get(
        "train_transforms",
        None,
    )
    if train_transforms is None:
        train_transforms = create_transforms(
            cfg,
            use_transforms=cfg.use_transforms,
        )

    eval_transforms = kwargs.get(
        "eval_transforms",
        None,
    )
    if eval_transforms is None:
        eval_transforms = create_transforms(
            cfg,
            use_transforms=False,
        )
    ## train
    train_labels = list(train_metadata[target].values.tolist())
    if train_metadata.index.dtype == "int64":
        train_image_files = np.array(
            [
                os.path.join(scan_path, filename)
                for filename in train_metadata["Image Index"].values
            ]
        )
    else:
        train_image_files = np.array(
            [
                os.path.join(scan_path, filename)
                for filename in train_metadata.index.values
            ]
        )

    train_dataset: monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=list(train_image_files),
        labels=list(train_labels),
        transform=train_transforms,
    )

    ## validation
    val_labels = val_metadata[target].values.tolist()
    if val_metadata.index.dtype == "int64":
        val_image_files = np.array(
            [
                os.path.join(scan_path, filename)
                for filename in val_metadata["Image Index"].values
            ]
        )
    else:
        val_image_files = np.array(
            [
                os.path.join(scan_path, filename)
                for filename in val_metadata.index.values
            ]
        )
    val_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=list(val_image_files),
        labels=list(val_labels),
        transform=train_transforms,
    )

    ## test
    test_labels = test_metadata[target].values.tolist()
    test_image_files = np.array(
        [os.path.join(scan_path, filename) for filename in test_metadata.index.values]
    )
    test_dataset = monai.data.Dataset = instantiate(
        config=dataset_cfg.instantiate,
        image_files=list(test_image_files),
        labels=list(test_labels),
        transform=eval_transforms,
    )

    if save_metadata:
        train_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "nih_train_metadata.csv")
        )
        val_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "nih_val_metadata.csv")
        )
        test_metadata.to_csv(
            os.path.join(DEFAULT_DATA_PATH, "patients", "nih_test_metadata.csv")
        )

    return train_dataset, val_dataset, test_dataset
