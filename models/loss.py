import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    ASL - works much better than plain cross-entropy for our heavily
    imbalanced dataset (hypertension is only ~4.6% of samples).

    Key idea: treat positives and negatives with different focusing factors
    so the model doesn't just learn to predict the majority class.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        margin: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.size(-1)).float()
        else:
            targets = targets.float()

        probs = torch.sigmoid(logits)
        probs_neg = torch.clamp(probs - self.margin, min=0.0, max=1.0)

        # positive side
        pos_w = (1.0 - probs).pow(self.gamma_pos)
        pos = targets * pos_w * torch.log(torch.clamp(probs, min=self.eps))

        # negative side - harder focusing
        neg_w = probs_neg.pow(self.gamma_neg)
        neg = (1.0 - targets) * neg_w * torch.log(torch.clamp(1.0 - probs_neg, min=self.eps))

        loss = -(pos + neg)

        if self.reduction == "mean":
            return loss.sum() / targets.size(0)
        elif self.reduction == "sum":
            return loss.sum()
        return loss
