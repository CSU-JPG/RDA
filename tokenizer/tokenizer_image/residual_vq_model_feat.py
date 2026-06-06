# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq_model import VQModel
# from .vq_quantize import VectorQuantizer
from .ablation_cross_atten import CrossAttentionFuse
from .ablation_contact_fuse import ConcatConvFuse

@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    img_size: int = 256
    is_cascade: bool = False
    cross_attention: bool = False
    contact_conv: bool = False

class ResidualVQModel(nn.Module):
    """
    Residual VQ model that uses the same codebook indices as the base VQ model
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        # self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        self.post_quant_conv = nn.Conv2d(config.z_channels, config.z_channels, 1)

        if config.cross_attention:

            self.cross_attention = CrossAttentionFuse(dim=config.z_channels, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0)
        if config.contact_conv:
            self.contact_conv = ConcatConvFuse(dim=config.z_channels, bottleneck=0.5, use_se=True)
        self.quant_post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels,1)
        self.img_size = config.img_size
        self.is_cascade = config.is_cascade

    def forward(self, x, vq_indices=None, additional_feat=None):
        """
        Args:
            x: Input residual tensor
            vq_indices: Codebook indices from the base VQ model
        """
        # Encode
        # import pdb; pdb.set_trace()
        if vq_indices is not None:
            if additional_feat is not None:
                latent_h, latent_w = additional_feat.shape[-2:]
            else:
                latent_h, latent_w = x.shape[-2] // 16, x.shape[-1] // 16
            quant = self.quantize.embedding(vq_indices).reshape(vq_indices.shape[0], latent_h, latent_w, -1)

            quant = torch.einsum('b h w c -> b c h w', quant)
            # import pdb; pdb.set_trace()

            # quant = self.post_quant_conv(quant)
            quant = self.quant_post_quant_conv(quant)
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            # quant = torch.einsum('b c h w -> b h w c', quant)
            # import pdb; pdb.set_trace()
            if additional_feat is not None:
                if self.config.cross_attention:
                    quant = self.cross_attention(quant, additional_feat)
                if self.config.contact_conv:
                    quant = self.contact_conv(quant, additional_feat)
                else:
                    quant = quant + additional_feat
            diff =(None, None, None)
            info = (None, None, None)
        else:
            quant, diff, info = self.encode(x)
        
        # Decode
        x_recon = self.decode(quant)
        
        if self.is_cascade:
            return x_recon, diff, info, quant
        else:
            return x_recon, diff, info

    def encode(self, x, vq_indices=None):
        """
        Encode the input residual
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, vq_indices)
        return quant, emb_loss, info

    def decode(self, quantized):
        """
        Decode the quantized features
        """
        quantized = self.post_quant_conv(quantized)
        x_recon = self.decoder(quantized)
        return x_recon

    def decode_code(self, code_b, shape=None, channel_first=True):
        """
        Decode from codebook indices
        """
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec



class Encoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, resamp_with_conv=True, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions-1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        
        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))


    def forward(self, z, vq_indices=None):
        # 统一到 BHWC 方便后面按 (H,W) reshape
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        B, H, W, C = z.shape

        # >>> FAST PATH: 固定ID，直接查表，跳过距离/熵计算 <<<
        if vq_indices is not None:
            # vq_indices 可能是 [B, H*W] 或 [B, H, W]
            if vq_indices.dim() == 2:
                idx = vq_indices.view(B, H, W)
            else:
                idx = vq_indices
                assert idx.shape[1] == H and idx.shape[2] == W, "vq_indices 的网格需与 z 匹配"

            emb = self.embedding.weight
            if self.l2_norm:
                emb = F.normalize(emb, p=2, dim=-1)  # 只归一化码本向量

            # 直接 gather 对应的码字向量 → BHWC
            z_q = emb[idx.view(-1)]                # [B*H*W, C]
            z_q = z_q.view(B, H, W, C)            # [B, H, W, C]

            # 直通梯度：保留 z 的梯度通路
            z_q = z + (z_q - z).detach()

            # 统计使用率（可选）
            perplexity = None
            min_encodings = None
            codebook_usage = 0
            if self.show_usage and self.training:
                flat_idx = idx.reshape(-1)
                # 简易统计：当次 batch 的 unique 比例
                codebook_usage = flat_idx.unique().numel() / float(self.n_e)

            # 回到 NCHW
            z_q = torch.einsum('b h w c -> b c h w', z_q)
            # 不返回 vq/commit/entropy（固定ID下无意义）
            return z_q, (None, None, None, codebook_usage), (perplexity, min_encodings, idx)

        # ===== 原路径：需要重新分配最近邻时才走 =====
        z_flattened = z.view(-1, C)
        if self.l2_norm:
            z     = F.normalize(z, p=2, dim=-1)
            zf    = F.normalize(z_flattened, p=2, dim=-1)
            emb_w = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            zf, emb_w = z_flattened, self.embedding.weight

        # 距离矩阵 & argmin
        d = torch.sum(zf**2, dim=1, keepdim=True) + \
            torch.sum(emb_w**2, dim=1) - 2 * (zf @ emb_w.t())
        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = emb_w[min_encoding_indices].view(B, H, W, C)
        perplexity = None
        min_encodings = None
        vq_loss = torch.mean((z_q - z.detach()) ** 2)
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
        entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d) if self.entropy_loss_ratio > 0 else None

        z_q = z + (z_q - z).detach()
        z_q = torch.einsum('b h w c -> b c h w', z_q)
        codebook_usage = 0
        if self.show_usage and self.training:
            codebook_usage = min_encoding_indices.unique().numel() / float(self.n_e)

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, None, min_encoding_indices.view(B, H, W))


    # def forward(self, z, vq_indices=None):
    #     # reshape z -> (batch, height, width, channel) and flatten
    #     z = torch.einsum('b c h w -> b h w c', z).contiguous()
    #     z_flattened = z.view(-1, self.e_dim)
    #     # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

    #     if self.l2_norm:
    #         z = F.normalize(z, p=2, dim=-1)
    #         z_flattened = F.normalize(z_flattened, p=2, dim=-1)
    #         embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
    #     else:
    #         embedding = self.embedding.weight

    #     d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
    #         torch.sum(embedding**2, dim=1) - 2 * \
    #         torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

    #     if vq_indices is not None:
    #         min_encoding_indices = vq_indices
    #     else:
    #         min_encoding_indices = torch.argmin(d, dim=1)

    #     z_q = embedding[min_encoding_indices].view(z.shape)
    #     perplexity = None
    #     min_encodings = None
    #     vq_loss = None
    #     commit_loss = None
    #     entropy_loss = None
    #     codebook_usage = 0

    #     if self.show_usage and self.training:
    #         cur_len = min_encoding_indices.shape[0]
    #         self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
    #         self.codebook_used[-cur_len:] = min_encoding_indices
    #         codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

    #     # compute loss for embedding
    #     if self.training and vq_indices is None:
    #         vq_loss = torch.mean((z_q - z.detach()) ** 2) 
    #         commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
    #         entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
    #     elif self.training and vq_indices is not None:

    #         # During training, use the provided VQ indices
    #         # 直接使用VQ模型的indices获取对应的embeddings
    #         # import pdb; pdb.set_trace()
            
    #         vq_loss = torch.mean((z_q - z.detach()) ** 2) 
    #         commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            
    #         # 计算entropy loss
    #         entropy_loss = 0.0
    #         if self.entropy_loss_ratio > 0:
    #             entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-torch.zeros(1, device=z.device))

    #     # preserve gradients
    #     z_q = z + (z_q - z).detach()


    #     # reshape back to match original input shape
    #     z_q = torch.einsum('b h w c -> b c h w', z_q)
    #     return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
def VQ_8(**kwargs):
    return ResidualVQModel(ModelArgs(encoder_ch_mult=[1, 2, 2, 4], decoder_ch_mult=[1, 2, 2, 4], **kwargs))

def VQ_16(**kwargs):
    return ResidualVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

def VQ_16_feat(**kwargs):
    return ResidualVQModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 2, 4, 4, 8], **kwargs))

ResidualVQ_models = {'VQ-16': VQ_16, 'VQ-8': VQ_8, 'VQ-16-feat': VQ_16_feat}