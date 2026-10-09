import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from models.backbone import SiameseEfficientNetBackbone
from models.bilateral_attention import BilateralCrossAttention
from models.quantum_circuit import QuantumVQCBranch


class ClassicalHead(nn.Module):
    """Dense classification head on top of the fused bilateral features."""

    def __init__(self, in_features: int = 512, hidden: int = 128, num_classes: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.bn = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        if h.size(0) > 1:
            h = self.bn(h)
        h = self.drop(h)
        return self.fc2(h), h


class ETHBQFramework(nn.Module):
    """
    Explainable Tri-Hybrid Bilateral Quantum (ETH-BQ) model.

    Architecture overview:
      1. Siamese EfficientNet extracts features from OD and OS separately
         (same weights for both so the representations are comparable)
      2. Bilateral cross-attention lets each eye 'look at' what the other eye sees
      3. The fused representation goes into two parallel branches:
         - Classical dense head
         - 8-qubit variational quantum circuit head
      4. Logits from both branches are averaged for the final prediction

    The quantum branch helps separate co-morbid edge cases that sit close
    together in classical feature space.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        embed_dim: int = 256,
        num_heads: int = 4,
        n_qubits: int = 8,
        n_quantum_layers: int = 3,
        num_classes: int = 8,
        pretrained: bool = True,
        use_quantum: bool = True,
        use_bilateral_attention: bool = True,
    ):
        super().__init__()
        self.use_quantum = use_quantum
        self.use_attn = use_bilateral_attention
        self.embed_dim = embed_dim

        self.backbone = SiameseEfficientNetBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            embed_dim=embed_dim,
        )

        fused_dim = embed_dim * 2

        if use_bilateral_attention:
            self.attn = BilateralCrossAttention(embed_dim=embed_dim, num_heads=num_heads)

        self.classical_head = ClassicalHead(fused_dim, hidden=128, num_classes=num_classes)

        if use_quantum:
            self.quantum_head = QuantumVQCBranch(
                in_features=fused_dim,
                n_qubits=n_qubits,
                n_layers=n_quantum_layers,
                num_classes=num_classes,
            )

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x_od: torch.Tensor,
        x_os: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:

        f_od, f_os, map_od, map_os = self.backbone(x_od, x_os)

        if self.use_attn:
            fused, att_od, att_os = self.attn(f_od, f_os)
        else:
            fused = torch.cat([f_od, f_os], dim=-1)
            att_od, att_os = f_od, f_os

        cls_logits, hidden = self.classical_head(fused)

        if self.use_quantum:
            q_logits = self.quantum_head(fused)
            w = torch.sigmoid(self.fusion_weight)
            logits = w * cls_logits + (1 - w) * q_logits
        else:
            logits = cls_logits

        meta = None
        if return_features:
            meta = {
                "f_od": f_od.unsqueeze(0) if f_od.dim() == 1 else f_od,
                "f_os": f_os.unsqueeze(0) if f_os.dim() == 1 else f_os,
                "fused": fused,
                "map_od": map_od,
                "map_os": map_os,
                "att_od": att_od,
                "att_os": att_os,
                "hidden": hidden,
            }

        return logits, meta
