# imports
import os
import warnings
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# torch imports
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# huggingface
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    ProcessorMixin,
    SiglipModel,
    logging,
)

# open clip
import open_clip

# rtk
from rtk import DEFAULT_MODEL_PATH, console
from rtk.utils import get_logger

logger = get_logger(__name__, console=console)
logging.set_verbosity_error()


def load_hf_model(args: DictConfig, **kwargs):
    pretrained_model_id = kwargs.get(
        "pretrained_model_id", args.models.get("pretrained_model_id", "")
    )
    config: PretrainedConfig = AutoConfig.from_pretrained(pretrained_model_id, **kwargs)
    model = PreTrainedModel(config=config)
    processor = AutoProcessor.from_pretrained(model.name_or_path)
    return model, processor


def download_checkpoint_from_hugging_face(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading checkpoint from Hugging Face hub for repo_id: {repo_id}")
    checkpoint_path = hf_hub_download(repo_id=repo_id, subfolder="checkpoints", filename=filename)
    return checkpoint_path


class Encoder:

    def __init__(
        self,
        args: DictConfig,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        tokenizer=None,
        **kwargs,
    ):
        self.args = args
        self.model_name: str = kwargs.get("model_name", args.get("model_name"))
        self.data_dir: str = kwargs.get("data_dir", args.get("data_dir", os.getenv("DATA_DIR")))
        self.model_args: dict = args.models
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        logger.debug(f"Initialized Encoder with model: '{self.model_name}'")

    @torch.no_grad()
    def encode_text(self, text):
        if not isinstance(text, list):
            text = [text]
        if "siglip" in self.model_name:
            inputs = self.processor(
                text=text,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to("cuda")
            embeddings = self.model.get_text_features(**inputs)
            return embeddings

        if "biomed" in self.model_name:
            inputs = self.tokenizer(text).to("cuda")
            embeddings = self.model.encode_text(inputs)
            return embeddings

        if "medclip" in self.model_name:
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")
            inputs.pop("token_type_ids")
            embeddings = self.model.encode_text(**inputs)
            return embeddings

    @torch.no_grad()
    def encode_images(self, image_paths: list) -> torch.Tensor:

        if "biomed" in self.model_name:
            images = [
                Image.open(os.path.join(self.data_dir, p)).convert("RGB") for p in image_paths
            ]
            inputs = torch.stack([self.processor(im) for im in images]).to("cuda")
            embeddings: torch.Tensor = self.model.encode_image(inputs)
            return embeddings

        if "siglip" in self.model_name:
            images = [
                Image.open(os.path.join(self.data_dir, p)).convert("RGB") for p in image_paths
            ]
            inputs = self.processor(images=images, return_tensors="pt").to("cuda")
            embeddings: torch.Tensor = self.model.get_image_features(**inputs)
            return embeddings

        if "medclip" in self.model_name:
            images = [Image.open(os.path.join(self.data_dir, p)) for p in image_paths]
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")
            embeddings = self.model.encode_image(**inputs)
            return embeddings

    @torch.no_grad()
    def encode(self, text, image_paths: list) -> torch.Tensor:
        # Preprocessed in val_transform_images
        if "biomed" in self.model_name:
            image_embeddings = self.encode_images(image_paths)
            text_embeddings = self.encode_text(text)
            return image_embeddings, text_embeddings
        if "siglip" in self.model_name:
            image_embeddings = self.encode_images(image_paths)
            text_embeddings = self.encode_text(text)
            return image_embeddings, text_embeddings
        if "medclip" in self.model_name:
            if not isinstance(text, list):
                text = [text]
            images = [
                Image.open(os.path.join(self.data_dir, p)).convert("RGB") for p in image_paths
            ]
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                padding=False,
            )
            output = self.model(**inputs)
            image_embeddings = output["img_embeds"]
            text_embeddings = output["text_embeds"]
            return image_embeddings, text_embeddings


def _create_medclip_model(args: DictConfig, **kwargs):
    from medclip import MedCLIPModel, MedCLIPProcessor

    model_name: str = args.model_name
    if model_name == "medclip-resnet":
        from medclip import MedCLIPVisionModel

        vision_cls = MedCLIPVisionModel
    elif model_name == "medclip-vit":
        from medclip import MedCLIPVisionModelViT

        vision_cls = MedCLIPVisionModelViT
    model = MedCLIPModel(vision_cls=vision_cls)
    model.to("cuda").eval()
    processor = MedCLIPProcessor()
    encoder = Encoder(args, model, processor, **kwargs)

    return encoder


def _create_open_clip_model(args: DictConfig, **kwargs):
    model_args: dict = args.models
    model_path: str = model_args["model_id"]
    pretrained: bool = model_args.get("pretrained", None)
    if pretrained:
        pretrained_kw = model_args["pretrained_weights"]
        checkpoint_path = download_checkpoint_from_hugging_face(**pretrained_kw)
        model, _, processor = open_clip.create_model_and_transforms(
            model_path, pretrained=checkpoint_path
        )
    else:
        model, _, processor = open_clip.create_model_and_transforms(model_path)
    model = model.to("cuda")
    tokenizer: open_clip.tokenizer.HFTokenizer = open_clip.get_tokenizer(
        model_path, context_length=model.context_length
    )
    model.eval()
    encoder = Encoder(args, model, tokenizer=tokenizer, processor=processor, **kwargs)

    return encoder


def _create_siglip_model(args: DictConfig, **kwargs):
    from transformers import SiglipProcessor

    model_args: dict = args.models
    model_id: str = model_args["model_id"]
    model: SiglipModel = SiglipModel.from_pretrained(model_id, device_map="cuda")
    processor: SiglipProcessor = SiglipProcessor.from_pretrained(model_id)
    model.eval()
    encoder = Encoder(args, model, processor, **kwargs)

    return encoder


def create_retrieval_model(args: DictConfig, **kwargs) -> Encoder:
    model_args: dict = args.models
    model_name: str = kwargs.get("model_name", model_args.get("name", ""))

    if "biomed" in model_name:
        return _create_open_clip_model(args, **kwargs)
    if "medclip" in model_name:
        return _create_medclip_model(args, **kwargs)
    if "siglip" in model_name:
        return _create_siglip_model(args, **kwargs)


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
    console.print(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param:.2f}"
    )


def create_clip_model(
    args: DictConfig, return_processors=False, caption_column: str = "reports", **kwargs
):
    data_dir: str = kwargs.get("data_dir", args.get("data_dir", os.getenv("DATA_DIR")))
    image_column: str = kwargs.get("image_column", "image_files")
    model_path: str = kwargs.get("model_path", args.models.model_path)
    pretrained: str = kwargs.get("pretrained", args.models.pretrained)

    model, train_image_processor, val_image_processor = open_clip.create_model_and_transforms(
        model_path, pretrained=pretrained
    )
    model.eval()
    tokenizer: open_clip.tokenizer.HFTokenizer = open_clip.get_tokenizer(
        model_path, context_length=model.context_length
    )

    def load_images_as_pil(examples: dict):
        images = [
            Image.open(os.path.join(data_dir, image_file)) for image_file in examples[image_column]
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
