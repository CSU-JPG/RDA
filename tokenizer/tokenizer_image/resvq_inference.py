import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from dataset.build import build_dataset
from tokenizer.tokenizer_image.rda_model import RDATokenizer
from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.residual_vq_model_feat import ResidualVQ_models


def variable_size_collate(batch):
    return tuple(zip(*batch))

def main(args):
    assert torch.cuda.is_available(), "Inference currently requires at least one GPU."

    init_distributed_mode(args)
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    if distributed:
        device = args.gpu
    else:
        device = args.gpu
        args.distributed = False

    assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    output_dirs = {
        "comparison": os.path.join(args.output_dir, "comparison"),
        "vq": os.path.join(args.output_dir, "vq"),
        "resvq": os.path.join(args.output_dir, "resvq"),
        "gt": os.path.join(args.output_dir, "gt"),
        "prediction": os.path.join(args.output_dir, "prediction"),
        "residual": os.path.join(args.output_dir, "residual"),
    }
    for output_dir in [args.output_dir, *output_dirs.values()]:
        os.makedirs(output_dir, exist_ok=True)

    if distributed:
        dist.barrier()

    if rank == 0:
        logger = create_logger(args.output_dir)
        logger.info(f"Output directory created at {args.output_dir}")
    else:
        logger = create_logger(None)

    logger.info(f"{args}")
    logger.info(f"Starting inference rank={rank}, seed={seed}, world_size={world_size}.")

    rda_model_path = args.rda_model_path or args.resvq_ckpt
    if rda_model_path is None:
        raise ValueError("Please provide --rda-model-path or --resvq-ckpt.")

    model = RDATokenizer.from_pretrained(
        vq_ckpt=args.vq_ckpt,
        rda_ckpt=rda_model_path,
        vq_model=args.vq_model,
        resvq_model=args.resvq_model,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        resvq_codebook_size=args.resvq_codebook_size,
        resvq_codebook_embed_dim=args.resvq_codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        image_size=args.image_size,
    ).to(device)
    vq = model.vq_model
    rda = model.resvq_model
    logger.info(f"Loaded base VQ checkpoint from {args.vq_ckpt}")
    logger.info(f"Loaded RDA model from {rda_model_path}")

    if args.compile:
        logger.info("compiling the RDA model... (may take several minutes)")
        rda = torch.compile(rda)

    dataset = build_dataset(args, transform=model.transform)
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=variable_size_collate
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    # Inference loop
    logger.info("Starting inference...")
    with torch.no_grad():
        for _, (images, image_paths) in enumerate(tqdm(loader)):
            for image_tensor, image_path in zip(images, image_paths):
                inputs = image_tensor.unsqueeze(0).to(device, non_blocking=True)
                vq_image, _, vq_info, quant_embeddings = vq(inputs, return_quant=True)
                vq_latent = vq.post_quant_conv(quant_embeddings)
                vq_ids = vq_info[2].reshape(vq_image.shape[0], -1)

                residual_image = inputs - vq_image
                rda_residual_image, _, _ = rda(residual_image, vq_ids, vq_latent)
                prediction_image = vq_image + rda_residual_image

                output = model.make_output(inputs, residual_image, vq_image, rda_residual_image, prediction_image)
                image_name = os.path.basename(image_path)
                output.gt.save(os.path.join(output_dirs["gt"], image_name))
                output.residual.save(os.path.join(output_dirs["residual"], image_name))
                output.vq.save(os.path.join(output_dirs["vq"], image_name))
                output.resvq.save(os.path.join(output_dirs["resvq"], image_name))
                output.prediction.save(os.path.join(output_dirs["prediction"], image_name))
                output.comparison.save(os.path.join(output_dirs["comparison"], image_name))
                    
    logger.info("Inference completed!")
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VQ + residual VQ image reconstruction inference.")
    parser.add_argument("--dataset", type=str, default='json_data')
    parser.add_argument("--data-path", type=str, nargs='+', required=True, help="Path(s) to JSON/YAML image list files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save gt/residual/vq/resvq/prediction/comparison images")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Checkpoint path for the pre-trained base VQ model")
    parser.add_argument("--resvq-model", type=str, choices=list(ResidualVQ_models.keys()), default="VQ-16-feat")
    parser.add_argument("--rda-model-path", type=str, default=None, help="RDA model path: local .pt checkpoint, local HF-style directory, or Hugging Face repo id")
    parser.add_argument("--resvq-ckpt", type=str, default=None, help="Deprecated alias for --rda-model-path")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--resvq-codebook-size", type=int, default=16384, help="codebook size for residual vector quantization")
    parser.add_argument("--resvq-codebook-embed-dim", type=int, default=8, help="codebook dimension for residual vector quantization")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--image-size", type=int, default=None, help="Optional center-crop size. If omitted, crop each image to the nearest size divisible by 16.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for single-process inference")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])

    args = parser.parse_args()
    main(args)