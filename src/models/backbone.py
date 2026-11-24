import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, Optional


class SEBlock(nn.Module):
    """Channel attention block - squeeze and excite."""

    def __init__(self, channels: int, r: int = 16):
        super().__init__()
        mid = max(8, channels // r)
        self.fc1 = nn.Linear(channels, mid, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mid, channels, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        z = x.mean(dim=(2, 3))
        s = self.sig(self.fc2(self.relu(self.fc1(z)))).view(b, c, 1, 1)
        return x * s


class SiameseEfficientNetBackbone(nn.Module):
    """
    Shared-weight backbone for processing bilateral fundus image pairs.
    Both OD and OS go through the same weights so the learned features
    are truly comparable across eyes.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        reduction_ratio: int = 16,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        # figure out how many channels come out of the last feature stage
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            feats = self.backbone(dummy)
            self.in_channels = feats[-1].shape[1]

        self.se = SEBlock(self.in_channels, r=reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(self.in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def extract(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        top = feats[-1]
        recal = self.se(top)
        emb = self.proj(self.pool(recal).flatten(1))
        return emb, recal

    def forward(self, x_od, x_os):
        f_od, map_od = self.extract(x_od)
        f_os, map_os = self.extract(x_os)
        return f_od, f_os, map_od, map_os
