import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BilateralCrossAttention(nn.Module):
    """
    Cross-attention between the two eyes.

    The right eye queries the left eye's context and vice versa.
    This lets the model pick up on systemic diseases that affect
    both eyes (like DR or hypertension) which wouldn't be obvious
    looking at a single eye alone.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        # separate projections for each eye so the roles are explicit
        self.q_od = nn.Linear(embed_dim, embed_dim)
        self.k_od = nn.Linear(embed_dim, embed_dim)
        self.v_od = nn.Linear(embed_dim, embed_dim)

        self.q_os = nn.Linear(embed_dim, embed_dim)
        self.k_os = nn.Linear(embed_dim, embed_dim)
        self.v_os = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.norm_od = nn.LayerNorm(embed_dim)
        self.norm_os = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def _attend(self, q, k, v, B):
        H = self.num_heads
        dk = self.d_k
        q = q.view(B, H, dk)
        k = k.view(B, H, dk)
        v = v.view(B, H, dk)
        attn = torch.softmax((q * k).sum(-1, keepdim=True) * self.scale, dim=1)
        return (attn * v).view(B, -1)

    def forward(self, f_od: torch.Tensor, f_os: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = f_od.size(0)

        # OD attends to OS
        ctx_od = self._attend(self.q_od(f_od), self.k_os(f_os), self.v_os(f_os), B)
        ctx_od = self.drop(ctx_od)
        out_od = self.norm_od(f_od + ctx_od[:, :f_od.size(-1)])

        # OS attends to OD
        ctx_os = self._attend(self.q_os(f_os), self.k_od(f_od), self.v_od(f_od), B)
        ctx_os = self.drop(ctx_os)
        out_os = self.norm_os(f_os + ctx_os[:, :f_os.size(-1)])

        fused = torch.cat([out_od, out_os], dim=-1)
        fused = self.out_proj(fused)
        return fused, out_od, out_os
