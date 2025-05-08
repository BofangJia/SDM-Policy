import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    



class SinusoidalPosEmb_gan(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # 确保 x 和 emb 的维度匹配进行广播操作
        x = x.unsqueeze(-1)  # 将 x 扩展为 [128, 1] 形状
        emb = emb.unsqueeze(0).expand(x.size(0), -1)  # 将 emb 扩展为 [128, 64]

        emb = x * emb  # 逐元素乘法
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 拼接 sin 和 cos
        return emb
