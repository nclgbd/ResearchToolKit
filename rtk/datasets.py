# imports
import numpy as np
import os
import pandas as pd
from PIL import Image

# torch
import torch
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
    ToTensor,
)

# 🤗
from datasets import Dataset, DatasetDict, load_dataset

# rtk
from rtk.utils import get_logger

logger = get_logger(__name__)
MIMIC_CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]
CHEXAGENT_PREDICTION_LABELS = (
    "/data/nicoleg/workspaces/dissertation/.data/CHEXAGENT_PREDICTION_LABELS.txt"
)


def load_mimic_gt_data(positive_class: str = "Pneumonia") -> pd.DataFrame:
    gt = load_dataset("vllm-pneumonia-detection/mimic-500-gt", split="test")
    gt = gt.to_pandas()
    gt = gt[["dicom_id", "image_files", "reports", positive_class]]
    gt["y_true"] = gt[positive_class].apply(lambda x: 1 if x == 1 else 0)
    gt.drop(columns=[positive_class], inplace=True)
    return gt


def set_transforms(
    data: DatasetDict, data_dir: str = ".", size: int = 448, split: str = "", **kwargs
):

    def _load_image_as_pil(examples: dict):

        image_files = [os.path.join(data_dir, image_file) for image_file in examples["image_files"]]
        images = [Image.open(image_file).convert("RGB") for image_file in image_files]
        return images

    _apply_train_transforms = kwargs.get(
        "apply_train_transforms", "train" in data or split == "train"
    )
    if _apply_train_transforms:
        logger.info("Setting train transforms...")
        _train_transforms = kwargs.get(
            "train_transforms",
            Compose(
                [
                    Resize((size, size)),
                    RandomRotation(90),
                    RandomAdjustSharpness(2),
                ]
            ),
        )

        def train_transforms(examples: dict):
            images = _load_image_as_pil(examples)
            examples["image"] = [_train_transforms(image) for image in images]
            return examples

        data["train"].set_transform(train_transforms)

    _apply_eval_transforms = kwargs.get(
        "apply_eval_transforms",
        "validate" in data or "test" in data or split == "val" or split == "test",
    )
    if _apply_eval_transforms:

        def val_transforms(examples: dict):

            _val_transforms = kwargs.get(
                "eval_transforms",
                Compose(
                    [
                        Resize((size, size)),
                    ]
                ),
            )
            images = _load_image_as_pil(examples)
            examples["image"] = [_val_transforms(image) for image in images]
            return examples

        if "validate" in data:
            logger.info("Setting validation transforms...")
            data["validate"].set_transform(val_transforms)

        if "test" in data:
            logger.info("Setting test transforms...")
            data["test"].set_transform(val_transforms)
