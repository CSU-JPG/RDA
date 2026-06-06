import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.residual_vq_model_feat import ResidualVQ_models
from tokenizer.tokenizer_image.vq_model import VQ_models


@dataclass
class RDAOutput:
    gt: Image.Image
    vq: Image.Image
    residual: Image.Image
    resvq: Image.Image
    prediction: Image.Image
    comparison: Image.Image


def resolve_model_path(pretrained_model_name_or_path: Union[str, os.PathLike]) -> Path:
    path = Path(pretrained_model_name_or_path)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub to load a Hugging Face repo id.") from e
    return Path(snapshot_download(str(pretrained_model_name_or_path)))


def resolve_checkpoint_path(checkpoint_or_repo_id: Union[str, os.PathLike], filename: str) -> Path:
    path = Path(checkpoint_or_repo_id)
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub to load a Hugging Face checkpoint repo id.") from e
    return Path(hf_hub_download(str(checkpoint_or_repo_id), filename))


def load_config(pretrained_model_name_or_path: Optional[Union[str, os.PathLike]]) -> Dict:
    if pretrained_model_name_or_path is None:
        return {}
    model_dir = resolve_model_path(pretrained_model_name_or_path)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def load_state_dict(model: nn.Module, checkpoint_path: Union[str, os.PathLike], map_location: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def load_rda_model(
    checkpoint_path: Union[str, os.PathLike],
    *,
    resvq_model: str = "VQ-16-feat",
    resvq_codebook_size: int = 16384,
    resvq_codebook_embed_dim: int = 16,
    commit_loss_beta: float = 0.25,
    entropy_loss_ratio: float = 0.0,
    dropout_p: float = 0.0,
    image_size: Optional[int] = None,
    map_location: str = "cpu",
) -> nn.Module:
    model = ResidualVQ_models[resvq_model](
        codebook_size=resvq_codebook_size,
        codebook_embed_dim=resvq_codebook_embed_dim,
        commit_loss_beta=commit_loss_beta,
        entropy_loss_ratio=entropy_loss_ratio,
        dropout_p=dropout_p,
        img_size=image_size,
    )
    load_state_dict(model, checkpoint_path, map_location=map_location)
    return model.eval()


class RDATokenizer(nn.Module):
    def __init__(
        self,
        vq_model: nn.Module,
        resvq_model: nn.Module,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.vq_model = vq_model.eval()
        self.resvq_model = resvq_model.eval()
        self.image_size = image_size
        transform_steps = []
        if image_size is not None:
            transform_steps.append(transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)))
        else:
            transform_steps.append(transforms.Lambda(self._center_crop_to_multiple_of_16))
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.transform = transforms.Compose(transform_steps)

        for param in self.vq_model.parameters():
            param.requires_grad = False
        for param in self.resvq_model.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        *,
        vq_ckpt: Optional[Union[str, os.PathLike]] = None,
        rda_ckpt: Optional[Union[str, os.PathLike]] = None,
        resvq_ckpt: Optional[Union[str, os.PathLike]] = None,
        map_location: str = "cpu",
        **kwargs,
    ):
        config = load_config(pretrained_model_name_or_path)
        config.update(kwargs)

        if pretrained_model_name_or_path is not None:
            model_dir = resolve_model_path(pretrained_model_name_or_path)
            vq_ckpt = vq_ckpt or model_dir / config.get("vq_checkpoint", "vq_model.pt")
            rda_ckpt = rda_ckpt or resvq_ckpt or model_dir / config.get("rda_checkpoint", "rda_model.pt")
        else:
            rda_ckpt = rda_ckpt or resvq_ckpt
            if rda_ckpt is not None:
                rda_ckpt = resolve_checkpoint_path(rda_ckpt, "rda_model.pt")

        if vq_ckpt is None:
            raise ValueError("Please provide vq_ckpt or a pretrained model directory with vq_model.pt.")
        if rda_ckpt is None:
            raise ValueError("Please provide rda_ckpt/resvq_ckpt or a pretrained model directory with rda_model.pt.")

        vq_model_name = config.get("vq_model", "VQ-16")
        resvq_model_name = config.get("resvq_model", "VQ-16-feat")
        image_size = config.get("image_size")
        image_size = int(image_size) if image_size is not None else None

        vq_model = VQ_models[vq_model_name](
            codebook_size=int(config.get("codebook_size", 16384)),
            codebook_embed_dim=int(config.get("codebook_embed_dim", 8)),
            commit_loss_beta=float(config.get("commit_loss_beta", 0.25)),
            entropy_loss_ratio=float(config.get("entropy_loss_ratio", 0.0)),
            dropout_p=float(config.get("dropout_p", 0.0)),
        )
        resvq_model = load_rda_model(
            rda_ckpt,
            resvq_model=resvq_model_name,
            resvq_codebook_size=int(config.get("resvq_codebook_size", 16384)),
            resvq_codebook_embed_dim=int(config.get("resvq_codebook_embed_dim", 16)),
            commit_loss_beta=float(config.get("commit_loss_beta", 0.25)),
            entropy_loss_ratio=float(config.get("entropy_loss_ratio", 0.0)),
            dropout_p=float(config.get("dropout_p", 0.0)),
            image_size=image_size,
            map_location=map_location,
        )

        load_state_dict(vq_model, vq_ckpt, map_location=map_location)

        return cls(vq_model=vq_model, resvq_model=resvq_model, image_size=image_size)

    def forward(self, image: Union[str, os.PathLike, Image.Image]) -> RDAOutput:
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        device = next(self.parameters()).device
        img = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            vq_recons, _, vq_info, quant_embeddings = self.vq_model(img, return_quant=True)
            quant = self.vq_model.post_quant_conv(quant_embeddings)
            residual = img - vq_recons
            vq_indices = vq_info[2].reshape(vq_recons.shape[0], -1)
            resvq_recons, _, _ = self.resvq_model(residual, vq_indices, quant)
            prediction = vq_recons + resvq_recons

        return self.make_output(img, residual, vq_recons, resvq_recons, prediction)

    def make_output(self, gt, residual, vq, resvq, prediction) -> RDAOutput:
        gt_img = self._inverse_transform(gt[0].cpu())
        residual_img = self._inverse_transform(residual[0].cpu())
        vq_img = self._inverse_transform(vq[0].cpu())
        resvq_img = self._inverse_transform(resvq[0].cpu())
        prediction_img = self._inverse_transform(prediction[0].cpu())
        comparison_img = self._make_comparison(gt_img, residual_img, vq_img, resvq_img, prediction_img)
        return RDAOutput(
            gt=gt_img,
            residual=residual_img,
            vq=vq_img,
            resvq=resvq_img,
            prediction=prediction_img,
            comparison=comparison_img,
        )

    @staticmethod
    def save_output(output: RDAOutput, output_dir: Union[str, os.PathLike]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output.gt.save(output_dir / "input.png")
        output.vq.save(output_dir / "vq_reconstruction.png")
        output.residual.save(output_dir / "residual.png")
        output.resvq.save(output_dir / "rda_residual.png")
        output.prediction.save(output_dir / "final_reconstruction.png")
        output.comparison.save(output_dir / "comparison.png")

    @staticmethod
    def _center_crop_to_multiple_of_16(image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_width = width - width % 16
        crop_height = height - height % 16
        if crop_width == width and crop_height == height:
            return image
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))

    @staticmethod
    def _inverse_transform(tensor: torch.Tensor) -> Image.Image:
        x_inv_norm = (tensor + 1) / 2
        x_inv_img = (x_inv_norm * 255).clamp(0, 255).byte()
        return transforms.ToPILImage()(x_inv_img)

    def _make_comparison(self, gt, residual, vq, resvq, prediction):
        width, height = gt.size
        comparison = Image.new("RGB", (width * 5, height))
        comparison.paste(gt, (0, 0))
        comparison.paste(residual, (width, 0))
        comparison.paste(vq, (width * 2, 0))
        comparison.paste(resvq, (width * 3, 0))
        comparison.paste(prediction, (width * 4, 0))
        return comparison
