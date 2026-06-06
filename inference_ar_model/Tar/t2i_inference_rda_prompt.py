import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoTokenizer, Qwen2ForCausalLM

TAR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TAR_ROOT.parents[1]
for path in [TAR_ROOT, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tok.mm_autoencoder import MMAutoEncoder
from tokenizer.tokenizer_image.rda_model import load_config, load_rda_model, resolve_model_path


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class T2IConfig:
    model_path: str = "csuhan/Tar-1.5B"
    ar_path: str = ""
    encoder_path: str = ""
    decoder_path: str = ""
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    scale: int = 0
    seq_len: int = 729
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200
    cfg_scale: float = 4.0


@dataclass
class RDAConfig:
    resvq_model: str = "VQ-16-feat"
    codebook_size: int = 16384
    codebook_embed_dim: int = 16
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    dropout_p: float = 0.0
    checkpoint_path: Union[str, os.PathLike] = ""
    image_size: int = 512


class TarRDAInference:
    def __init__(self, t2i_config: T2IConfig, rda_config: RDAConfig):
        self.t2i_config = t2i_config
        self.rda_config = rda_config
        self.device = torch.device(t2i_config.device)
        torch.set_grad_enabled(False)
        self._load_models()

    def _load_models(self):
        self.model = Qwen2ForCausalLM.from_pretrained(
            self.t2i_config.model_path,
            torch_dtype=self.t2i_config.dtype,
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.t2i_config.model_path)

        self.visual_tokenizer = MMAutoEncoder(
            ar_path=self.t2i_config.ar_path,
            encoder_path=self.t2i_config.encoder_path,
            decoder_path=self.t2i_config.decoder_path,
            encoder_args={"input_type": "rec"},
            decoder_args={},
        ).eval().to(dtype=self.t2i_config.dtype, device=self.device)
        self.visual_tokenizer.ar_model.cls_token_num = self.t2i_config.seq_len
        self.visual_tokenizer.encoder.pool_scale = self.t2i_config.scale + 1

        self.rda = load_rda_model(
            self.rda_config.checkpoint_path,
            resvq_model=self.rda_config.resvq_model,
            resvq_codebook_size=self.rda_config.codebook_size,
            resvq_codebook_embed_dim=self.rda_config.codebook_embed_dim,
            commit_loss_beta=self.rda_config.commit_loss_beta,
            entropy_loss_ratio=self.rda_config.entropy_loss_ratio,
            dropout_p=self.rda_config.dropout_p,
            image_size=self.rda_config.image_size,
        ).to(self.device)

    @torch.inference_mode()
    def generate_ar_codes(self, prompt: str):
        input_text = self._format_prompt(prompt)
        inputs = self.tokenizer(input_text, return_tensors="pt")
        gen_ids = self.model.generate(
            inputs.input_ids.to(self.device),
            max_new_tokens=self.t2i_config.seq_len,
            do_sample=True,
            temperature=self.t2i_config.temperature,
            top_p=self.t2i_config.top_p,
            top_k=self.t2i_config.top_k,
        )

        gen_text = self.tokenizer.batch_decode(gen_ids)[0]
        ar_codes = [int(x) for x in re.findall(r"<I(\d+)>", gen_text)]
        ar_codes = ar_codes[: self.t2i_config.seq_len] + [0] * max(0, self.t2i_config.seq_len - len(ar_codes))
        return torch.tensor(ar_codes, device=self.device).unsqueeze(0)

    @torch.inference_mode()
    def decode_vq_image(self, ar_codes: torch.Tensor):
        _, quant_embeddings, vq_image, vq_ids = self.visual_tokenizer.decode_from_encoder_indices_rec(
            ar_codes,
            {"cfg_scale": self.t2i_config.cfg_scale},
        )
        return vq_image, vq_ids, quant_embeddings

    @torch.inference_mode()
    def decode_rda_residual(self, vq_ids: torch.Tensor, quant_embeddings: torch.Tensor):
        rda_residual_image, _, _ = self.rda(None, vq_ids, quant_embeddings)
        return rda_residual_image

    @torch.inference_mode()
    def decode_with_rda(self, ar_codes: torch.Tensor):
        vq_image, vq_ids, quant_embeddings = self.decode_vq_image(ar_codes)
        rda_residual_image = self.decode_rda_residual(vq_ids, quant_embeddings)
        prediction_image = vq_image + rda_residual_image
        return vq_image, rda_residual_image, prediction_image

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return input_text + f"<im_start><S{self.t2i_config.scale}>"

    def make_output(self, vq_image, rda_residual_image, prediction_image):
        output = {
            "vq": self._to_pil(vq_image),
            "rda_residual": self._to_pil(rda_residual_image),
            "prediction": self._to_pil(prediction_image),
        }
        output["comparison"] = make_comparison(output)
        return output

    @staticmethod
    def save_output(output, output_dir: Union[str, os.PathLike], prompt: Optional[str] = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output["vq"].save(output_dir / "vq.png")
        output["rda_residual"].save(output_dir / "rda_residual.png")
        output["prediction"].save(output_dir / "prediction.png")
        output["comparison"].save(output_dir / "comparison.png")
        if prompt is not None:
            with open(output_dir / "prompt.txt", "w") as f:
                f.write(prompt)

    @staticmethod
    def _to_pil(tensor: torch.Tensor) -> Image.Image:
        image = (tensor[0].detach().float().cpu() + 1) / 2
        image = (image * 255).clamp(0, 255).byte()
        return Image.fromarray(image.permute(1, 2, 0).numpy())


def resolve_file(path_or_repo_id: Optional[str], filename: str, default_repo_id: str) -> Path:
    source = path_or_repo_id or default_repo_id
    path = Path(source)
    if path.exists():
        if path.is_dir():
            return path / filename
        return path
    return Path(hf_hub_download(source, filename))


def resolve_rda_model(path_or_repo_id: str):
    path = Path(path_or_repo_id)
    if path.exists() and path.is_file():
        return path, {}

    config = load_config(path_or_repo_id)
    model_dir = resolve_model_path(path_or_repo_id)
    return model_dir / config.get("rda_checkpoint", "rda_model.pt"), config


def make_comparison(images) -> Image.Image:
    ordered = [images["vq"], images["rda_residual"], images["prediction"]]
    width, height = ordered[0].size
    comparison = Image.new("RGB", (width * len(ordered), height))
    for index, image in enumerate(ordered):
        comparison.paste(image, (width * index, 0))
    return comparison


def parse_dtype(dtype: str):
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def main(args):
    torch.manual_seed(args.seed)

    ar_filename = args.ar_dtok_filename or f"ar_dtok_lp_{args.image_size}px.pth"
    ar_path = resolve_file(args.ar_dtok_path, ar_filename, "csuhan/TA-Tok")
    encoder_path = resolve_file(args.encoder_path, "ta_tok.pth", "csuhan/TA-Tok")
    decoder_path = resolve_file(args.vq_ckpt, "vq_ds16_t2i.pt", "peizesun/llamagen_t2i")
    rda_path, rda_hf_config = resolve_rda_model(args.rda_model_path)

    t2i_config = T2IConfig(
        model_path=args.model_path,
        ar_path=str(ar_path),
        encoder_path=str(encoder_path),
        decoder_path=str(decoder_path),
        device=args.device,
        dtype=parse_dtype(args.dtype),
        scale=args.scale,
        seq_len=args.seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        cfg_scale=args.cfg_scale,
    )
    rda_config = RDAConfig(
        resvq_model=args.rda_model or rda_hf_config.get("resvq_model", "VQ-16-feat"),
        codebook_size=args.rda_codebook_size or int(rda_hf_config.get("resvq_codebook_size", 16384)),
        codebook_embed_dim=args.rda_codebook_embed_dim or int(rda_hf_config.get("resvq_codebook_embed_dim", 16)),
        commit_loss_beta=args.commit_loss_beta if args.commit_loss_beta is not None else float(rda_hf_config.get("commit_loss_beta", 0.25)),
        entropy_loss_ratio=args.entropy_loss_ratio if args.entropy_loss_ratio is not None else float(rda_hf_config.get("entropy_loss_ratio", 0.0)),
        dropout_p=args.dropout_p if args.dropout_p is not None else float(rda_hf_config.get("dropout_p", 0.0)),
        checkpoint_path=rda_path,
        image_size=args.image_size,
    )

    inference = TarRDAInference(t2i_config, rda_config)

    ar_codes = inference.generate_ar_codes(args.prompt)
    vq_image, vq_ids, quant_embeddings = inference.decode_vq_image(ar_codes)
    rda_residual_image = inference.decode_rda_residual(vq_ids, quant_embeddings)
    prediction_image = vq_image + rda_residual_image
    images = inference.make_output(vq_image, rda_residual_image, prediction_image)

    inference.save_output(images, args.output_dir, args.prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tar text-to-image inference with an RDA decoder adapter for one prompt.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/tar_rda_demo")
    parser.add_argument("--model-path", type=str, default="csuhan/Tar-1.5B", help="Tar HF repo id or local model directory")
    parser.add_argument("--rda-model-path", type=str, default="CSU-JPG/RDA_llamagen", help="RDA HF repo id, local HF-style directory, or local .pt checkpoint")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="Base LlamaGen VQ checkpoint path. If omitted, download from Hugging Face")
    parser.add_argument("--encoder-path", type=str, default=None, help="TA-Tok encoder checkpoint path. If omitted, download from Hugging Face")
    parser.add_argument("--ar-dtok-path", type=str, default=None, help="AR-DTok checkpoint path. If omitted, download from Hugging Face")
    parser.add_argument("--ar-dtok-filename", type=str, default=None, help="AR-DTok filename in csuhan/TA-Tok")
    parser.add_argument("--rda-model", type=str, default=None)
    parser.add_argument("--rda-codebook-size", type=int, default=None)
    parser.add_argument("--rda-codebook-embed-dim", type=int, default=None)
    parser.add_argument("--commit-loss-beta", type=float, default=None)
    parser.add_argument("--entropy-loss-ratio", type=float, default=None)
    parser.add_argument("--dropout-p", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=729)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=1200)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
