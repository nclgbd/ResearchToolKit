# imports
import os
import warnings
from PIL import Image
from hydra.utils import instantiate
from omegaconf import DictConfig

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# torch imports
from open_clip import tokenizer
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
    SiglipProcessor,
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
        processor=None,
        tokenizer=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
    ):
        self.args = {} if args is None else args
        self.model_name: str = kwargs.get("model_name", self.args.get("model_name", ""))
        self.data_dir: str = kwargs.get(
            "data_dir", self.args.get("data_dir", os.getenv("DATA_DIR"))
        )
        self.model_args: dict = {} if args is None else self.args.models
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.device = device
        self.image_transform = kwargs.get("image_transform", None)
        logger.debug(f"Initialized Encoder with model: '{self.model_name}'")

    @torch.no_grad()
    def encode_text(self, text, **kwargs):
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

        if "chexficient" in self.model_name:
            max_length: int = kwargs.get("max_length", self.model_args.get("max_bert_length", 256))
            inputs = self.tokenizer(
                text,
                padding="longest",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            for key in inputs:
                inputs[key] = inputs[key].to(
                    next(self.model.parameters()).device, non_blocking=True
                )
            embeddings = self.model.encode_text(inputs)
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

        if "chexficient" in self.model_name:
            images = [
                Image.open(os.path.join(self.data_dir, p)).convert("RGB") for p in image_paths
            ]
            inputs = torch.stack([self.image_transform(im) for im in images]).to("cuda")
            embeddings: torch.Tensor = self.model.encode_image(inputs)
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

        if "chexficient" in self.model_name:
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

        if "radir" in self.model_name:
            from rtk.datasets import load_2d_image_to_tensor

            modality_dict = {"CT": 0, "CXR": 1}
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            is_abs = os.path.isabs(image_paths[0])
            if is_abs:
                images = image_paths
            else:
                images = [os.path.join(self.data_dir, p) for p in image_paths]
            image_tensors = [load_2d_image_to_tensor(image_path) for image_path in images]
            batched_images = torch.stack(
                image_tensors, dim=0
            )  # [B, C, D, H, W] -> [2, 1, 1, 480, 480]
            batched_images = batched_images.to(self.device)

            text_tokens = self.tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
            ).to(self.device)
            logger.debug(text_tokens)

            modal_indexs = torch.tensor([modality_dict["CXR"]] * len(image_paths)).to(self.device)

            image_embeddings, text_embeddings, _, _ = self.model(
                text_tokens,
                image=batched_images,
                device=self.device,
                is_condition=False,
                return_latents=True,
                modal_indexs=modal_indexs,
                modal_embedding=True,
            )

            return image_embeddings, text_embeddings


def _create_chexficient_model(args: DictConfig = None, **kwargs):
    import torchvision.transforms as transforms
    from chexficient import CheXficient

    image_size = kwargs.get("image_size", args.models.get("image_size", 224))
    model: torch.nn.Module = CheXficient(image_size=image_size)
    model.to(torch.device("cuda"))
    tokenizer = model.text_encoder.tokenizer
    image_transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ]
    )

    weights_path = f"{DEFAULT_MODEL_PATH}/{args.models.name}/pytorch_model.pth"
    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False,
    )["model"]
    _ = model.load_state_dict(state_dict, strict=False)
    model.eval()
    encoder = Encoder(args, model, tokenizer=tokenizer, image_transform=image_transform)

    return encoder


def _create_radir_model(args: DictConfig = None, **kwargs):
    from radir import RADIR
    from transformer_maskgit import CTViT
    from transformers import BertTokenizer, BertModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args: dict = {} if args is None else args.models
    model_id: str = "microsoft/BiomedVLP-CXR-BERT-specialized"
    checkpoint_path: str = "/data/nicoleg/workspaces/RadIR/models/RadIR.pt"
    tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=True)
    text_encoder = BertModel.from_pretrained(model_id)
    text_encoder = text_encoder.to(device)

    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=8,
        temporal_depth=6,
        cls_depth=4,
        dim_head=32,
        heads=8,
    ).to(device)

    Rad_IR = RADIR(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dim_text=768,
        dim_image=512,
        dim_latent=512,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    ).to(device)
    Rad_IR.load(checkpoint_path)
    Rad_IR.eval()

    encoder = Encoder(args, Rad_IR, model_name="radir", tokenizer=tokenizer)
    return encoder


def _create_medclip_model(args: DictConfig = None, **kwargs):
    from medclip import MedCLIPModel, MedCLIPProcessor

    model_name: str = "" if args is None else args.model_name
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


def _create_open_clip_model(args: DictConfig = None, **kwargs):
    model_args: dict = {} if args is None else args.models
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


def _create_siglip_model(args: DictConfig = None, **kwargs):
    from transformers import SiglipProcessor

    model_args: dict = {} if args is None else args.models
    model_id: str = model_args["model_id"]
    model: SiglipModel = SiglipModel.from_pretrained(model_id, device_map="cuda")
    processor: SiglipProcessor = SiglipProcessor.from_pretrained(model_id)
    model.eval()
    encoder = Encoder(args, model, processor, **kwargs)

    return encoder


def create_retrieval_model(args: DictConfig = None, **kwargs) -> Encoder:
    model_args: dict = {} if args is None else args.models
    model_name: str = kwargs.get("model_name", model_args.get("name", ""))

    if "biomed" in model_name:
        return _create_open_clip_model(args, **kwargs)
    if "medclip" in model_name:
        return _create_medclip_model(args, **kwargs)
    if "siglip" in model_name:
        return _create_siglip_model(args, **kwargs)
    if "radir" in model_name:
        return _create_radir_model(args, **kwargs)
    if "chexficient" in model_name:
        return _create_chexficient_model(args, **kwargs)


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
