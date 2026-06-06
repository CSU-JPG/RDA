import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionFuse(nn.Module):
    """
    F_dec:  (B, C, H, W)  -> 作为 Query
    F_hint: (B, C, H, W)  -> 作为 Key/Value
    return: (B, C, H, W)  -> 残差回写到 F_dec
    """
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 conv 等价于逐像素线性投影，保空间尺寸
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可选：对输出做轻量FFN再回写（更稳）
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1),
        )

        # 可选：门控，抑制噪声融合（更稳）
        self.gate = nn.Parameter(torch.tensor(0.0))  # 学习一个标量门，初始为0，渐进引入注意力

    def forward(self, F_dec, F_hint):
        B, C, H, W = F_dec.shape
        N = H * W

        # 1) 线性投影
        q = self.q_proj(F_dec)  # (B, C, H, W)
        k = self.k_proj(F_hint)
        v = self.v_proj(F_hint)

        # 2) 展平到 token 序列，并分头
        #    -> (B, heads, N, head_dim)
        def reshape_heads(x):
            x = x.view(B, C, N).transpose(1, 2)              # (B, N, C)
            x = x.view(B, N, self.num_heads, self.head_dim)  # (B, N, H, D)
            return x.permute(0, 2, 1, 3)                     # (B, H, N, D)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # 3) 注意力 (全局：q @ k^T)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, N, D)

        # 4) 合并头，并还原到 (B, C, H, W)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)   # (B, C, H, W)

        # 5) 输出投影 + Dropout
        out = self.proj(out)
        out = self.proj_drop(out)

        # 6) 残差 + 轻量FFN（可关掉）
        # 学习门控避免训练初期不稳：out_scale = sigmoid(gate)
        out = F_dec + torch.sigmoid(self.gate) * out
        out = out + self.ffn(out)

        return out