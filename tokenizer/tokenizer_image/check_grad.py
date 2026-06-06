import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleQuantFromIndices(nn.Module):
    def __init__(self, num_embeddings=16, embedding_dim=8, img_size=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.img_size = img_size
        self.grid_size = img_size // 16  # assume encoder outputs 16x downsampling

    def forward(self, vq_indices):
        # vq_indices: [B, H, W]
        B, H, W = vq_indices.shape

        # 查表（embedding lookup）
        quant = F.embedding(vq_indices, self.embedding.weight)  # [B, H, W, C]
        quant = quant.reshape(B, H, W, -1)
        quant = torch.einsum("b h w c -> b c h w", quant)

        # ✅ 使用 STE：让 embedding 可导
        quant_st = quant + (quant - quant).detach()
        return quant_st

# ---- 模拟输入 ----
B, H, W, C = 4, 16, 16, 8
num_embeddings = 16
img_size = 256

# 模拟来自 base VQ 的索引
vq_indices = torch.randint(0, num_embeddings, size=(B, H, W)).long()
# 模拟 VQ encoder 结果（想拟合的目标）
lable = torch.randn(B, C, H, W)

# ---- 构建模型并训练 ----
model = SimpleQuantFromIndices(num_embeddings=num_embeddings, embedding_dim=C, img_size=img_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

vq_indices = vq_indices.cuda()
lable = lable.cuda()
model = model.cuda()

for step in range(1, 6):
    optimizer.zero_grad()
    quant_recons = model(vq_indices)  # 查表得到重建

    loss = F.mse_loss(quant_recons, lable)
    loss.backward()

    print(f"[Step {step}] Loss: {loss.item():.4f}")
    grad_norm = model.embedding.weight.grad.norm().item() if model.embedding.weight.grad is not None else "None"
    print(f"    Grad norm: {grad_norm}")

    optimizer.step()
    print(f"    Weight norm after step: {model.embedding.weight.data.norm().item():.4f}")

