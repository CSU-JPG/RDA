
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator

import numpy as np

def unified_minmax_norm(x1, x2):
    combined = torch.cat([x1, x2], dim=0)
    x_min = combined.amin()
    x_max = combined.amax()
    x1_norm = (x1 - x_min) / (x_max - x_min + 1e-6)
    x2_norm = (x2 - x_min) / (x_max - x_min + 1e-6)
    return x1_norm.clamp(0, 1), x2_norm.clamp(0, 1)

def get_sobel_mask(inputs):

    gray = inputs.mean(dim=1, keepdim=True)


    sobel_x = torch.tensor([[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]], device=inputs.device).unsqueeze(0) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    threshold = grad_mag.mean() + 0.5 * grad_mag.std()
    mask = (grad_mag > threshold).float()  # shape: [B, 1, H, W]
    return mask

def masked_dice_loss(pred, target, mask, eps=1e-6):
    pred = pred * mask
    target = target * mask
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def cutoff_by_energy(fshift, ratio=0.9):

    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    radius = np.sqrt((X-ccol)**2 + (Y-crow)**2)

    mag2 = np.abs(fshift)**2
    # 将频率按半径排序
    idx = np.argsort(radius.flat)
    mag2_sorted = mag2.flat[idx]
    radius_sorted = radius.flat[idx]

    # 累积能量
    cumsum = np.cumsum(mag2_sorted)
    total = cumsum[-1]
    cutoff_idx = np.searchsorted(cumsum, ratio * total)
    r_c = radius_sorted[cutoff_idx]
    return r_c


class ResFreqVQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, args=None
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight



        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight
        self.args = args


    def tensor2freq(self, x):
        # crop patches
        patch_factor = self.patch_factor
        B, C, H, W = x.shape
        patch_h, patch_w = H // patch_factor, W // patch_factor
        patches = []
        for i in range(patch_factor):
            for j in range(patch_factor):
                patches.append(x[:, :, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w])
        y = torch.stack(patches, 1)  # [B, P*P, C, ph, pw]

        # 2D FFT
        freq = torch.fft.fft2(y, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], -1)  # [B,P*P,C,ph,pw,2]
        return freq

    def rgb_to_gray(self, x):
        # x: [B,3,H,W] → [B,1,H,W]
        return  x[:,0:1] +  x[:,1:2] +  x[:,2:3]

    def cutoff_by_energy(self, fshift, ratio=0.9):
        """fshift: [ph, pw] 单通道频谱 (complex)"""
        rows, cols = fshift.shape
        crow, ccol = rows // 2, cols // 2
        yy, xx = torch.meshgrid(torch.arange(rows, device=fshift.device),
                                torch.arange(cols, device=fshift.device),
                                indexing="ij")
        radius = torch.sqrt((xx-ccol)**2 + (yy-crow)**2)

        mag2 = torch.abs(fshift) ** 2
        flat_radius = radius.flatten()
        flat_mag2 = mag2.flatten()

        idx = torch.argsort(flat_radius)
        mag2_sorted = flat_mag2[idx]
        radius_sorted = flat_radius[idx]

        cumsum = torch.cumsum(mag2_sorted, dim=0)
        total = cumsum[-1]
        cutoff_idx = torch.searchsorted(cumsum, ratio * total)
        r_c = radius_sorted[cutoff_idx]
        return r_c.item()

    def make_highfreq_mask(self, gray_img):

        B, _, H, W = gray_img.shape
        f = torch.fft.fft2(gray_img[0,0], norm="ortho")  
        fshift = torch.fft.fftshift(f)

        r_c = self.cutoff_by_energy(fshift, self.ratio)

        yy, xx = torch.meshgrid(torch.arange(H, device=gray_img.device),
                                torch.arange(W, device=gray_img.device),
                                indexing="ij")
        radius = torch.sqrt((xx-W//2)**2 + (yy-H//2)**2)
        mask = (radius > r_c).float()  
        return mask[None,None,:,:]  # [1,1,H,W]
        
    def freq_high(self, target,pred, q=0.6):

        B,C,H,W = pred.shape
        yy,xx = torch.meshgrid(torch.arange(H, device=pred.device),
                            torch.arange(W, device=pred.device), indexing="ij")
        R = min(H,W)/2
        r = R * (1.0 - q)**0.5                  
        mask = ( ((xx-W/2)**2 + (yy-H/2)**2).sqrt() > r ).float()  # [H,W]
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        Fp = torch.fft.fft2(pred,   norm="ortho")
        Ft = torch.fft.fft2(target, norm="ortho")
        diff2 = (Fp - Ft).abs().pow(2) * mask   #
        return diff2.mean()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, gt_img, sum_recons, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())


            # import pdb; pdb.set_trace()
            freq_loss = self.freq_high(inputs, reconstructions, self.args.freq_q)
            # perceptual loss

            res_p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            res_p_loss = torch.mean(res_p_loss)

            # sum perceptual loss
            sum_p_loss = self.perceptual_loss(gt_img.contiguous(), sum_recons.contiguous())
            sum_p_loss = torch.mean(sum_p_loss)

            # sum reconstruction loss
            sum_rec_loss= F.mse_loss(gt_img.contiguous(), sum_recons.contiguous())
            sum_rec_loss= torch.mean(sum_rec_loss)

            # discriminator loss
            logits_fake = self.discriminator(sum_recons.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * res_p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            

            sobel_mask = get_sobel_mask(inputs)

 
            reconstructions_norm, inputs_norm = unified_minmax_norm(reconstructions, inputs)

            dice_loss = masked_dice_loss(reconstructions_norm, inputs_norm, sobel_mask)

            sobel_recon_loss = self.rec_loss(inputs.contiguous()*sobel_mask, reconstructions.contiguous()*sobel_mask)

            loss = self.rec_weight * rec_loss
            
            if self.args.res_p_loss:
                loss += self.args.res_p_loss_weight * res_p_loss
            if self.args.sum_gan_loss:
                loss += disc_adaptive_weight * generator_adv_loss * disc_weight
            if self.args.sum_p_loss:
                loss += self.args.sum_p_loss_weight * sum_p_loss
            if self.args.dice_loss:
                loss += self.args.dice_loss_weight * dice_loss
            if self.args.sobel_recon_loss:
                loss += self.args.sobel_recon_loss_weight * sobel_recon_loss

            if self.args.sum_rec_loss:
                loss += self.args.sum_rec_loss_weight * sum_rec_loss

            if self.args.freq_loss:
                loss += self.args.freq_loss_weight * freq_loss

            rec_loss = self.rec_weight * rec_loss
            res_p_loss = self.perceptual_weight * res_p_loss
            sum_p_loss = self.perceptual_weight * sum_p_loss
            sum_rec_loss = self.rec_weight * sum_rec_loss
            freq_loss = self.args.freq_loss_weight * freq_loss
            generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            dice_loss = self.args.dice_loss_weight * dice_loss
            sobel_recon_loss = self.args.sobel_recon_loss_weight * sobel_recon_loss

            return loss, rec_loss, res_p_loss, sum_rec_loss, sum_p_loss, generator_adv_loss, codebook_loss[0], codebook_loss[1], codebook_loss[2], dice_loss, sobel_recon_loss, freq_loss

        # discriminator update
        if optimizer_idx == 1:


            logits_real = self.discriminator(gt_img.contiguous().detach())
            logits_fake = self.discriminator(sum_recons.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            
            # if global_step % log_every == 0:
            logits_real = logits_real.detach().mean()
            logits_fake = logits_fake.detach().mean()
            # logger.info(f"(Discriminator) " 
            #             f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
            #             f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
            return d_adversarial_loss, logits_real, logits_fake
        