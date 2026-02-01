# imports
import numpy as np
import os
import pandas as pd
import re
from PIL import Image
from dotenv import load_dotenv
from omegaconf import DictConfig

# torch
import torch
import torch.linalg as L
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
    ToTensor,
)

# 🤗
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

# rtk
from rtk import console
from rtk.models import Encoder
from rtk.utils import get_logger

logger = get_logger(__name__, console=console)
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
DTYPE = torch.bfloat16
EMBED_COLUMN = "embeddings"

load_dotenv()


def set_custom_clip_embeddings(
    args: DictConfig, dataset: Dataset, model: Encoder, batch_size=16, **kwargs
) -> Dataset:
    retrieval_modality: str = kwargs.get("retrieval_modality", args.retrieval_modality)
    embed_column = kwargs.get("embed_column", EMBED_COLUMN)

    def process(sample: dict):
        if retrieval_modality not in ["image", "text", "indication"]:
            raise ValueError(
                f"Invalid retrieval_modality: '{retrieval_modality}'. Must be 'image', 'text', or 'indication'."
            )
        if retrieval_modality == "image":
            encode_func = model.encode_images
            input_data = sample["image_files"]
        elif retrieval_modality in ["text", "indication"]:
            encode_func = model.encode_text
            input_data = sample["reports"]
            if retrieval_modality == "indication":
                input_data = [extract_indication(report) for report in input_data]
        else:
            raise NotImplementedError(f"Unsupported retrieval_modality: {retrieval_modality}")
        embeds = encode_func(input_data)
        embeds /= L.vector_norm(embeds, dim=1, keepdim=True)
        sample[embed_column] = embeds
        return sample

    return dataset.map(process, batched=True, batch_size=batch_size, **kwargs)


def save_tensor_memmap(tensor: torch.Tensor, filepath: str) -> None:
    """
    Save a tensor to disk using torch.save for efficient loading.

    Args:
        tensor: PyTorch tensor to save
        filepath: Path to save the .pt file (without extension)
    """
    pt_path = f"{filepath}.pt"
    torch.save(tensor.detach().cpu(), pt_path)


def load_tensor_memmap(filepath: str, copy_to_memory: bool = True) -> torch.Tensor:
    """
    Load a tensor from disk using torch.load.

    Args:
        filepath: Path to the saved file (without extension)
        copy_to_memory: If True, load into RAM; if False, use mmap (slower but saves RAM)

    Returns:
        PyTorch tensor
    """
    pt_path = f"{filepath}.pt"

    if copy_to_memory:
        # Load directly into RAM for faster access
        tensor: torch.Tensor = torch.load(pt_path, map_location="cuda", weights_only=True)

    else:
        # Use memory-mapped loading to save RAM (slower access)
        tensor = torch.load(pt_path, map_location="cpu", weights_only=True, mmap=True)
    tensor = tensor.to(dtype=DTYPE)
    return tensor


def extract_indication(report: str) -> str:
    """
    Extract the INDICATION section from a radiology report.

    Args:
        report: A string containing the radiology report text

    Returns:
        The indication text, stripped of extra whitespace, or an empty string if not found
    """
    # Pattern to match indication-like sections, capturing text until the next section header
    # Matches: INDICATION, REASON FOR EXAMINATION, HISTORY, CLINICAL INFORMATION
    # Stops at: blank line, line starting with uppercase section header + colon, or end of string
    logger.debug(f"Report:\n{report}\n")
    pattern = r"(?:HISTORY|INDICATION|REASON FOR EXAM|REASON FOR EXAMINATION|CLINICAL INFORMATION):\s*(.*?)(?=\n[ \t]*\n|\n\s*[A-Z]{2,}:|\Z)"
    match = re.search(pattern, report.strip(), re.DOTALL)

    if match:
        indication = match.group(1).strip()
        indication = re.sub(r"\s+", " ", indication)  # Normalize whitespace
        indication = indication.replace("\n", " ").strip()
        logger.debug(f"Extracted Indication:\n{indication}\n{'='*40}")
        return indication

    return ""


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
