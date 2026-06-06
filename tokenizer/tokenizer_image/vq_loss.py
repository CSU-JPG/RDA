# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator



def unified_minmax_norm(x1, x2):
    combined = torch.cat([x1, x2], dim=0)
    x_min = combined.amin()
    x_max = combined.amax()
    x1_norm = (x1 - x_min) / (x_max - x_min + 1e-6)
    x2_norm = (x2 - x_min) / (x_max - x_min + 1e-6)
    return x1_norm.clamp(0, 1), x2_norm.clamp(0, 1)

def get_sobel_mask(inputs):
    # 转灰度
    gray = inputs.mean(dim=1, keepdim=True)

    # 定义 Sobel 卷积核
    sobel_x = torch.tensor([[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]], device=inputs.device).unsqueeze(0) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    # 计算梯度幅值
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # 自适应 threshold
    threshold = grad_mag.mean() + 0.5 * grad_mag.std()
    mask = (grad_mag > threshold).float()  # shape: [B, 1, H, W]
    return mask

def get_soft_sobel_mask(inputs, tau=0.2, q=0.8):
    gray = (0.299*inputs[:,0:1] + 0.587*inputs[:,1:2] + 0.114*inputs[:,2:3])
    sobel_x = torch.tensor([[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]],
                            device=inputs.device).unsqueeze(0)/8.0
    sobel_y = sobel_x.transpose(2,3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    # 分位数阈值 + Sigmoid 软化
    t = torch.quantile(grad_mag.flatten(1), q, dim=1, keepdim=True).view(-1,1,1,1)
    mask = torch.sigmoid((grad_mag - t) / tau)
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


class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, 
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

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # discriminator loss
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            # import pdb; pdb.set_trace()
            loss = self.rec_weight * rec_loss + \
                self.perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                p_loss = self.perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                            f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                            f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            
            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
            return d_adversarial_loss


class ResVQLoss(nn.Module):
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
            
            # loss = self.rec_weight * rec_loss + \
            #     self.perceptual_weight * p_loss + \
            #     disc_adaptive_weight * disc_weight * generator_adv_loss + \
            #     codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            # loss = self.rec_weight * rec_loss + \
            #     self.perceptual_weight * p_loss + \
            #     disc_adaptive_weight * disc_weight * generator_adv_loss 
            # import pdb; pdb.set_trace()
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

            rec_loss = self.rec_weight * rec_loss
            res_p_loss = self.perceptual_weight * res_p_loss
            sum_p_loss = self.perceptual_weight * sum_p_loss
            sum_rec_loss = self.rec_weight * sum_rec_loss
            generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            dice_loss = self.args.dice_loss_weight * dice_loss
            sobel_recon_loss = self.args.sobel_recon_loss_weight * sobel_recon_loss
            
            # logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
            #             f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
            #             f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
            #             f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss, rec_loss, res_p_loss, sum_rec_loss, sum_p_loss, generator_adv_loss, codebook_loss[0], codebook_loss[1], codebook_loss[2], dice_loss, sobel_recon_loss

        # discriminator update
        if optimizer_idx == 1:
            # logits_real = self.discriminator(inputs.contiguous().detach())
            # logits_fake = self.discriminator(reconstructions.contiguous().detach())
            # logits_real = self.discriminator(inputs.contiguous().detach())
            # logits_fake = self.discriminator(reconstructions.contiguous().detach())


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
        


class EncoderDecoderVQLoss(nn.Module):
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

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, gt_img, sum_recons, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(gt_img.contiguous(), sum_recons.contiguous())
            # sum perceptual loss
            sum_p_loss = self.perceptual_loss(gt_img.contiguous(), sum_recons.contiguous())
            sum_p_loss = torch.mean(sum_p_loss)

            # discriminator loss
            logits_fake = self.discriminator(sum_recons.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * sum_p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            
            # loss = self.rec_weight * rec_loss + \
            #     self.perceptual_weight * p_loss + \
            #     disc_adaptive_weight * disc_weight * generator_adv_loss + \
            #     codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            # loss = self.rec_weight * rec_loss + \
            #     self.perceptual_weight * p_loss + \
            #     disc_adaptive_weight * disc_weight * generator_adv_loss 
            # import pdb; pdb.set_trace()

            loss = self.rec_weight * rec_loss
            
            if self.args.sum_gan_loss:
                loss += disc_adaptive_weight * generator_adv_loss * disc_weight
            if self.args.sum_p_loss:
                loss += self.args.sum_p_loss_weight * sum_p_loss

            rec_loss = self.rec_weight * rec_loss
            sum_p_loss = self.perceptual_weight * sum_p_loss
            generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            # logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
            #             f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
            #             f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
            #             f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss, rec_loss, sum_p_loss, generator_adv_loss

        # discriminator update
        if optimizer_idx == 1:
            # logits_real = self.discriminator(inputs.contiguous().detach())
            # logits_fake = self.discriminator(reconstructions.contiguous().detach())
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
        
class StageVQLoss(nn.Module):
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

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, gt_img, sum_recons, optimizer_idx, global_step, stage='3', last_layer=None, 
                logger=None, log_every=100):
        # generator update
        if optimizer_idx == 0:
            # import pdb; pdb.set_trace()
            if stage == '1':
                rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())
                loss = self.rec_weight * rec_loss
                return loss, rec_loss, None, None, None, None, None, None, None, None
           
            
            # perceptual loss
            res_p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            res_p_loss = torch.mean(res_p_loss)

            # sum perceptual loss
            sum_p_loss = self.perceptual_loss(gt_img.contiguous(), sum_recons.contiguous())
            sum_p_loss = torch.mean(sum_p_loss)

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

            rec_loss = self.rec_weight * rec_loss
            res_p_loss = self.perceptual_weight * res_p_loss
            sum_p_loss = self.perceptual_weight * sum_p_loss
            generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            dice_loss = self.args.dice_loss_weight * dice_loss
            sobel_recon_loss = self.args.sobel_recon_loss_weight * sobel_recon_loss
            # logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
            #             f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
            #             f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
            #             f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss, rec_loss, res_p_loss, sum_p_loss, generator_adv_loss, codebook_loss[0], codebook_loss[1], codebook_loss[2], dice_loss, sobel_recon_loss

        # discriminator update
        if optimizer_idx == 1:
            # logits_real = self.discriminator(inputs.contiguous().detach())
            # logits_fake = self.discriminator(reconstructions.contiguous().detach())
            # import pdb; pdb.set_trace()
            if stage == '1':

                logits_real = None
                logits_fake = None
                disc_weight = 0
                d_adversarial_loss = 0
                return d_adversarial_loss, logits_real, logits_fake
            

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

class ResidualStrongLoss(nn.Module):
    def __init__(self, 
                 res_weight=1.0, 
                 sum_weight=1.0, 
                 contrast_weight=0.5,
                 mag_weight=0.1,
                 edge_alpha=2.0,
                 min_magnitude=0.05):
        super().__init__()
        self.lpips_loss = LPIPS().eval()
        self.res_weight = res_weight
        self.sum_weight = sum_weight
        self.contrast_weight = contrast_weight
        self.mag_weight = mag_weight
        self.edge_alpha = edge_alpha
        self.min_magnitude = min_magnitude

    def forward(self, imgs, vq_recons, resvq_recons):
        sum_recons = vq_recons + resvq_recons
        residual_gt = imgs - vq_recons

        # -------- Residual loss (带权重) --------
        sobel_mask = get_sobel_mask(residual_gt)
        res_diff = torch.abs(residual_gt - resvq_recons)
        weight = (1 + self.edge_alpha * sobel_mask + residual_gt.abs() / (residual_gt.abs().mean() + 1e-6))
        res_loss = (res_diff * weight).mean()

        # -------- Full image loss --------
        sum_res_loss = F.l1_loss(sum_recons, imgs) 
        sum_p_loss = torch.mean(self.lpips_loss(imgs, sum_recons))

        # -------- Min magnitude penalty --------
        mag_loss = F.relu(self.min_magnitude - torch.abs(resvq_recons)).mean()

        # -------- Contrastive perceptual loss --------
        lpips_vq = torch.mean(self.lpips_loss(imgs, vq_recons))
        lpips_sum = torch.mean(self.lpips_loss(imgs, sum_recons))
        # margin = 0.01  # 要求 sum 至少比 vq 好一点
        # contrastive_lpips = torch.clamp(lpips_sum - lpips_vq + margin, min=0.0)
        eps = 1e-4
        diff = torch.log((lpips_sum + eps) / (lpips_vq + eps))

        reward_weight = 0.1
        contrastive_lpips = torch.clamp(diff, min=0.0) - reward_weight * torch.clamp(-diff, min=0.0)


        # -------- Final total loss --------
        total_loss = (
            self.res_weight * res_loss +
            self.sum_weight * sum_res_loss +
            self.sum_weight * sum_p_loss +
            self.mag_weight * mag_loss +
            self.contrast_weight * contrastive_lpips
        )

        # return {
        #     "total_loss": total_loss,
        #     "res_loss": res_loss,
        #     "sum_loss": sum_loss,
        #     "mag_loss": mag_loss,
        #     "contrastive_lpips": contrastive_lpips
        # }
        return total_loss, res_loss, sum_res_loss, sum_p_loss, mag_loss, contrastive_lpips