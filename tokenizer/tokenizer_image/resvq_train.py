# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.augmentation import random_crop_arr, center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.residual_vq_model_feat import ResidualVQ_models
from tokenizer.tokenizer_image.vq_loss import VQLoss, ResVQLoss
from tokenizer.tokenizer_image.resfreqvq_loss import ResFreqVQLoss

import warnings
import wandb
warnings.filterwarnings('ignore')
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
import uuid
#################################################################################
#                                  Training Loop                                #
#################################################################################

def inverse_transform(tensor):
    # 假设输入 tensor 是 [-1,1]
    x_inv_norm = (tensor + 1) / 2
    x_inv_img = (x_inv_norm * 255).clamp(0, 255).byte()
    return F.to_pil_image(x_inv_img)

def main(args):
    """
    Trains a new ResVQ model using a pre-trained VQ model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    # import pdb; pdb.set_trace()
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(args.cloud_save_path, exist_ok=True)
        model_string_name = args.resvq_model.replace("/", "-")

        experiment_index = len(glob(f"{args.cloud_save_path}/*"))

        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)

    dist.barrier()


    if rank == 0:
        logger = create_logger(cloud_checkpoint_dir)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")

        if args.use_wandb:
            run_id = uuid.uuid4().hex[:8] 

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                id=run_id,
                config=vars(args)
            )
    else:
        logger = create_logger(None) 

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load pre-trained VQ model and freeze its parameters
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
    )
    
    # Load pre-trained weights
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded pre-trained VQ model from {args.vq_ckpt}")
    else:
        raise ValueError("Must provide a pre-trained VQ model checkpoint!")
    
    # Freeze VQ model parameters
    for param in vq_model.parameters():
        param.requires_grad = False
    vq_model = vq_model.to(device)
    vq_model.eval()  # Set to eval mode
    
    # Create and initialize ResVQ model
    resvq_model = ResidualVQ_models[args.resvq_model](
        codebook_size=args.resvq_codebook_size,
        codebook_embed_dim=args.resvq_codebook_embed_dim,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        dropout_p=args.dropout_p,
        cross_attention=args.cross_attention,
        contact_conv=args.contact_conv,
    )
    
    
    # load pre-trained weights 
    if args.use_pretrained_vq:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        resvq_model.load_state_dict(checkpoint["model"])
        del checkpoint

    
    for param in resvq_model.encoder.parameters():
        param.requires_grad = False
    for param in resvq_model.quant_conv.parameters():
        param.requires_grad = False

    logger.info(f"ResVQ Model Parameters: {sum(p.numel() for p in resvq_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(resvq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"ResVQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    resvq_model = resvq_model.to(device)

    vq_loss = ResFreqVQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        args=args
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    optimizer = torch.optim.Adam(resvq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Setup data:
    transform = transforms.Compose([
        # transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = build_dataset(args, transform=transform)
    usable_len = (len(dataset) // args.global_batch_size) * args.global_batch_size
    dataset = torch.utils.data.Subset(dataset, range(usable_len))
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    # Prepare models for training:
    # import pdb; pdb.set_trace()
    if args.resvq_ckpt:
        checkpoint = torch.load(args.resvq_ckpt, map_location="cpu")
        resvq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.resvq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.resvq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, resvq_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        resvq_model = torch.compile(resvq_model) # requires PyTorch 2.0        
    
    resvq_model = DDP(resvq_model.to(device), device_ids=[args.gpu])
    resvq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_gen_loss = 0
    running_rec_loss = 0
    running_p_loss = 0
    running_sum_rec_loss = 0
    running_sum_p_loss = 0
    running_generator_adv_loss = 0
    running_codebook_vq_loss = 0
    running_codebook_commit_loss = 0
    running_codebook_entropy_loss = 0
    running_disc_real_loss = 0
    running_disc_fake_loss = 0
    running_disc_loss = 0
    running_dice_loss = 0
    running_sobel_recon_loss = 0
    running_freq_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in tqdm(range(start_epoch, args.epochs)):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for x, y in pbar:
            imgs = x.to(device, non_blocking=True)
            with torch.no_grad():
                vq_recons, _ , vq_info, quant_embeddings = vq_model(imgs, return_quant=True)
                
                quant = vq_model.post_quant_conv(quant_embeddings)
                # quant = torch.einsum('b c h w -> b h w c', quant)
                residual = imgs - vq_recons  # Calculate residual
                residual = residual
                vq_indices = vq_info[2].reshape(vq_recons.shape[0], -1)
            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                # resvq_recons, codebook_loss = resvq_model(residual)
                # import pdb; pdb.set_trace()
                resvq_recons, codebook_loss, _ = resvq_model(residual, vq_indices, quant)
                # resvq_recons, codebook_loss, _ = resvq_model(residual, vq_indices)

                # resvq_recons, codebook_loss, _ = resvq_model(residual)

                total_recons = vq_recons + resvq_recons

                # lable = imgs
                # predict = total_recons
                lable = residual
                predict = resvq_recons

                loss_gen, rec_loss, p_loss,sum_rec_loss,sum_p_loss, generator_adv_loss, codebook_vq_loss, codebook_commit_loss, codebook_entropy_loss, dice_loss, sobel_recon_loss, freq_loss = \
                    vq_loss(codebook_loss, residual, resvq_recons, imgs, total_recons, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=resvq_model.module.decoder.last_layer, 
                                   logger=logger, log_every=args.log_every)

            scaler.scale(loss_gen).backward()
            # import pdb; pdb.set_trace()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(resvq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, resvq_model.module._orig_mod if args.compile else resvq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):

                loss_disc, logits_real, logits_fake = vq_loss(codebook_loss, residual, resvq_recons, imgs, total_recons, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()

            running_gen_loss += loss_gen.item() if loss_gen is not None else 0
            running_rec_loss += rec_loss.item() if rec_loss is not None else 0
            running_p_loss += p_loss.item() if p_loss is not None else 0
            running_sum_rec_loss += sum_rec_loss.item() if sum_rec_loss is not None else 0
            running_sum_p_loss += sum_p_loss.item() if sum_p_loss is not None else 0
            running_generator_adv_loss += generator_adv_loss.item() if generator_adv_loss is not None else 0
            running_codebook_vq_loss += codebook_vq_loss.item() if codebook_vq_loss is not None else 0
            running_codebook_commit_loss += codebook_commit_loss.item() if codebook_commit_loss is not None else 0
            running_codebook_entropy_loss += codebook_entropy_loss.item() if codebook_entropy_loss is not None else 0
            running_disc_real_loss += logits_real.item() if logits_real is not None else 0
            running_disc_fake_loss += logits_fake.item() if logits_fake is not None else 0
            running_disc_loss += loss_disc.item() if loss_disc is not None else 0
            running_dice_loss += dice_loss.item() if dice_loss is not None else 0
            running_sobel_recon_loss += sobel_recon_loss.item() if sobel_recon_loss is not None else 0
            running_freq_loss += freq_loss.item() if freq_loss is not None else 0
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_gen_loss = torch.tensor(running_gen_loss / log_steps, device=device)
                avg_rec_loss = torch.tensor(running_rec_loss / log_steps, device=device)
                avg_p_loss = torch.tensor(running_p_loss / log_steps, device=device)
                avg_sum_rec_loss = torch.tensor(running_sum_rec_loss / log_steps, device=device)
                avg_sum_p_loss = torch.tensor(running_sum_p_loss / log_steps, device=device)
                avg_generator_adv_loss = torch.tensor(running_generator_adv_loss / log_steps, device=device)
                avg_codebook_vq_loss = torch.tensor(running_codebook_vq_loss / log_steps, device=device)
                avg_codebook_commit_loss = torch.tensor(running_codebook_commit_loss / log_steps, device=device)
                avg_codebook_entropy_loss = torch.tensor(running_codebook_entropy_loss / log_steps, device=device)
                avg_disc_real_loss = torch.tensor(running_disc_real_loss / log_steps, device=device)
                avg_disc_fake_loss = torch.tensor(running_disc_fake_loss / log_steps, device=device)
                avg_disc_loss = torch.tensor(running_disc_loss / log_steps, device=device)
                avg_dice_loss = torch.tensor(running_dice_loss / log_steps, device=device)
                avg_sobel_recon_loss = torch.tensor(running_sobel_recon_loss / log_steps, device=device)
                avg_freq_loss = torch.tensor(running_freq_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_gen_loss, op=dist.ReduceOp.SUM)
                avg_gen_loss = avg_gen_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_rec_loss, op=dist.ReduceOp.SUM)
                avg_rec_loss = avg_rec_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_p_loss, op=dist.ReduceOp.SUM)
                avg_p_loss = avg_p_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_sum_rec_loss, op=dist.ReduceOp.SUM)
                avg_sum_rec_loss = avg_sum_rec_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_sum_p_loss, op=dist.ReduceOp.SUM)
                avg_sum_p_loss = avg_sum_p_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_generator_adv_loss, op=dist.ReduceOp.SUM)
                avg_generator_adv_loss = avg_generator_adv_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_codebook_vq_loss, op=dist.ReduceOp.SUM)
                avg_codebook_vq_loss = avg_codebook_vq_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_codebook_commit_loss, op=dist.ReduceOp.SUM)
                avg_codebook_commit_loss = avg_codebook_commit_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_codebook_entropy_loss, op=dist.ReduceOp.SUM)
                avg_codebook_entropy_loss = avg_codebook_entropy_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_disc_real_loss, op=dist.ReduceOp.SUM)
                avg_disc_real_loss = avg_disc_real_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_disc_fake_loss, op=dist.ReduceOp.SUM)
                avg_disc_fake_loss = avg_disc_fake_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_disc_loss, op=dist.ReduceOp.SUM)
                avg_disc_loss = avg_disc_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_dice_loss, op=dist.ReduceOp.SUM)
                avg_dice_loss = avg_dice_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_sobel_recon_loss, op=dist.ReduceOp.SUM)
                avg_sobel_recon_loss = avg_sobel_recon_loss.item() / dist.get_world_size()
                dist.all_reduce(avg_freq_loss, op=dist.ReduceOp.SUM)
                avg_freq_loss = avg_freq_loss.item() / dist.get_world_size()
                # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                pbar.set_description(f"Epoch {epoch}/{args.epochs} Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.use_wandb and rank == 0:
                    wandb.log({
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "train_steps": train_steps,
                        "train_freq_loss": avg_freq_loss,
                        "epoch": epoch,
                        "train_gen_loss": avg_gen_loss,
                        "train_disc_loss": avg_disc_loss,
                        "train_rec_loss": avg_rec_loss,
                        "train_p_loss": avg_p_loss,
                        "train_sum_rec_loss": avg_sum_rec_loss,
                        "train_sum_p_loss": avg_sum_p_loss,
                        "train_generator_adv_loss": avg_generator_adv_loss,
                        "train_codebook_vq_loss": avg_codebook_vq_loss,
                        "train_codebook_commit_loss": avg_codebook_commit_loss,
                        "train_codebook_entropy_loss": avg_codebook_entropy_loss,
                        "train_disc_real_loss": avg_disc_real_loss,
                        "train_disc_fake_loss": avg_disc_fake_loss,
                        "train_dice_loss": avg_dice_loss,
                        "train_sobel_recon_loss": avg_sobel_recon_loss
                        
                    })
                # Reset monitoring variables:
                running_loss = 0
                running_gen_loss = 0
                running_rec_loss = 0
                running_p_loss = 0
                running_sum_rec_loss = 0
                running_sum_p_loss = 0
                running_generator_adv_loss = 0
                running_codebook_vq_loss = 0
                running_codebook_commit_loss = 0
                running_codebook_entropy_loss = 0
                running_disc_real_loss = 0
                running_disc_fake_loss = 0
                running_disc_loss = 0
                running_dice_loss = 0
                running_sobel_recon_loss = 0
                running_freq_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = resvq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = resvq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()

                    
                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    if rank == 0:
        if args.compile:
            model_weight = resvq_model.module._orig_mod.state_dict()
        else:
            model_weight = resvq_model.module.state_dict()  
        checkpoint = {
            "model": model_weight,
            "optimizer": optimizer.state_dict(),
            "discriminator": vq_loss.module.discriminator.state_dict(),
            "optimizer_disc": optimizer_disc.state_dict(),
            "steps": train_steps,
            "args": args
        }
        if args.ema:
            checkpoint["ema"] = ema.state_dict()
            
        # 保存云端checkpoint
        cloud_last_checkpoint_path = f"{cloud_checkpoint_dir}/last.pt"
        torch.save(checkpoint, cloud_last_checkpoint_path)
        logger.info(f"Saved last checkpoint in cloud to {cloud_last_checkpoint_path}")
    dist.barrier()

    resvq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if args.use_wandb:
        wandb.finish()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, nargs='+', required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="ckpt path for pre-trained vq model")
    parser.add_argument("--resvq-model", type=str, choices=list(ResidualVQ_models.keys()), default="VQ-16-feat")
    parser.add_argument("--resvq-ckpt", type=str, default=None, help="ckpt path for resume training resvq model")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained resvq model")
    parser.add_argument("--residual_scale", type=float, default=2, help="residual scale for residual vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--resvq-codebook-size", type=int, default=16384, help="codebook size for residual vector quantization")
    parser.add_argument("--resvq-codebook-embed-dim", type=int, default=8, help="codebook dimension for residual vector quantization")
    parser.add_argument("--codebook-l2-norm", action='store_true', default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--use-wandb", action="store_true", help="whether to use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="vq_tokenizer", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (team or user), optional")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name, optional")

    # loss function
    parser.add_argument("--res_p_loss", action="store_true", help="whether to use res_p_loss")
    parser.add_argument("--res_p_loss_weight", type=float, default=1.0, help="res_p_loss weight")

    parser.add_argument("--dice_loss", action="store_true", help="whether to use dice loss")
    parser.add_argument("--dice_loss_weight", type=float, default=0.5, help="dice loss weight")

    parser.add_argument("--sum_rec_loss", action="store_true", help="whether to use sum_rec_loss")
    parser.add_argument("--sum_rec_loss_weight", type=float, default=1.0, help="sum_rec_loss weight")
    parser.add_argument("--sum_p_loss", action="store_true", help="whether to use sum_p_loss")
    parser.add_argument("--sum_p_loss_weight", type=float, default=1.0, help="sum_p_loss weight")
    parser.add_argument("--sum_gan_loss", action="store_true", help="whether to use sum_gan_loss")
    parser.add_argument("--sum_gan_loss_weight", type=float, default=1.0, help="sum_gan_loss weight")

    parser.add_argument("--freq_loss", action="store_true", help="whether to use freq loss")
    parser.add_argument("--freq_loss_weight", type=float, default=1.0, help="freq loss weight")
    parser.add_argument("--freq_q", type=float, default=0.8, help="freq q")

    parser.add_argument("--sobel_recon_loss", action="store_true", help="whether to use sobel recon loss")
    parser.add_argument("--sobel_recon_loss_weight", type=float, default=2.0, help="sobel recon loss weight")
    

    parser.add_argument("--cross_attention", type=bool, default=False, help="whether to use cross attention")
    parser.add_argument("--contact_conv", type=bool, default=False, help="whether to use contact conv")
    

    parser.add_argument("--use_pretrained_vq", action="store_true", help="whether to use pretrained vq model")
    args = parser.parse_args()
    main(args) 