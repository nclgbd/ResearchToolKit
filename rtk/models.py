# imports
import os
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

from azureml.core import Model, Workspace

# torch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# open clip
import open_clip

# rtk
from rtk import DEFAULT_MODEL_PATH
from rtk.utils import get_logger

logger = get_logger(__name__)


def print_trainable_parameters(model: nn.Module):
    """
    Adapted from: https://huggingface.co/docs/peft/v0.6.2/en/task_guides/image_classification_lora#load-and-prepare-a-model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param:.2f}"
    )


def create_clip_model(args: DictConfig, return_processors=False, **kwargs):
    data_dir: str = kwargs.get("data_dir", args.get("data_dir", os.getenv("DATA_DIR")))
    caption_column: str = kwargs.get("caption_column", "reports")
    image_column: str = kwargs.get("image_column", "image_files")
    model_path: str = kwargs.get("model_path", args.models.model_path)
    pretrained: str = kwargs.get("pretrained", args.models.pretrained)

    model, train_image_processor, val_image_processor = (
        open_clip.create_model_and_transforms(model_path, pretrained=pretrained)
    )
    model.eval()
    tokenizer: open_clip.tokenizer.HFTokenizer = open_clip.get_tokenizer(
        model_path, context_length=model.context_length
    )

    def load_images_as_pil(examples: dict):
        images = [
            Image.open(os.path.join(data_dir, image_file))
            for image_file in examples[image_column]
        ]
        return images

    def tokenize_captions(examples: dict):
        captions = list(examples[caption_column])
        text = tokenizer(captions)
        examples["text"] = text
        return examples

    def train_transform_images(examples: dict):
        images = load_images_as_pil(examples)
        examples["pixel_values"] = [train_image_processor(image) for image in images]
        return examples

    def val_transform_images(examples: dict):
        images = load_images_as_pil(examples)
        examples["pixel_values"] = [val_image_processor(image) for image in images]
        return examples

    if return_processors:
        return (
            model,
            tokenizer,
            tokenize_captions,
            train_transform_images,
            val_transform_images,
            train_image_processor,
            val_image_processor,
        )
    else:
        return (
            model,
            tokenizer,
            tokenize_captions,
            train_transform_images,
            val_transform_images,
        )
