import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatConvFuse(nn.Module):
    """
    F_dec, F_hint: (B, C, H, W)  -> 返回 (B, C, H, W)
    思路：cat → 1x1降维 → 3x3融合 → (可选SE/门控) → 残差
    """
    def __init__(self, dim=256, bottleneck=0.5, use_se=True):
        super().__init__()
        mid = int(dim * (1 + 1) * bottleneck)  # 拼接后降维到 mid
        self.reduce = nn.Sequential(
            nn.Conv2d(dim*2, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU()
        )
        self.mix = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU()
        )
        self.proj = nn.Conv2d(mid, dim, kernel_size=1, bias=True)

        # 学习型门控，避免训练初期不稳（初值很小，相当于逐步引入融合）
        self.gate = nn.Parameter(torch.tensor(0.0))

        # 可选 SE 注意力
        self.use_se = use_se
        if use_se:
            self.se_fc1 = nn.Conv2d(dim, dim//8, 1)
            self.se_fc2 = nn.Conv2d(dim//8, dim, 1)

        self.out_bn = nn.BatchNorm2d(dim)

    def forward(self, F_dec, F_hint):
        # 可选：先对 hint 做一个轻去噪（注释掉即可）
        # F_hint = F_hint + F.avg_pool2d(F_hint, 3, 1, 1) * 0.0

        x = torch.cat([F_dec, F_hint], dim=1)   # (B, 2C, H, W)
        x = self.reduce(x)                      # (B, mid, H, W)
        x = self.mix(x)                         # (B, mid, H, W)
        x = self.proj(x)                        # (B, C, H, W)

        if self.use_se:
            w = F.adaptive_avg_pool2d(x, 1)
            w = torch.sigmoid(self.se_fc2(F.gelu(self.se_fc1(w))))
            x = x * w

        # 残差 + 门控（sigmoid让其∈[0,1]，初期≈0更稳）
        out = F_dec + torch.sigmoid(self.gate) * x
        return self.out_bn(out)
