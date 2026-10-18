import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Dict, Tuple


class BilateralGradCAMPlusPlus:
    """
    Grad-CAM++ for bilateral models.

    We hook into the backbone's last convolutional feature map,
    compute second-order gradients, and produce a heatmap showing
    which regions of the fundus the model is paying attention to.
    """

    def __init__(self, model):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}

    def _register_hooks(self, eye: str):
        se_block = self.model.backbone.se

        def fwd_hook(_, __, out):
            self.activations[eye] = out.detach()

        def bwd_hook(_, __, grad_out):
            self.gradients[eye] = grad_out[0].detach()

        fh = se_block.register_forward_hook(fwd_hook)
        bh = se_block.register_full_backward_hook(bwd_hook)
        return fh, bh

    def _compute_cam(self, eye: str) -> np.ndarray:
        A = self.activations.get(eye)
        G = self.gradients.get(eye)
        if A is None or G is None:
            return np.zeros((16, 16))

        # Grad-CAM++ weights: second-order gradient weighting
        G2 = G ** 2
        G3 = G ** 3
        denom = 2 * G2 + (A * G3).sum(dim=(2, 3), keepdim=True)
        alpha = G2 / (denom + 1e-8)
        weights = (alpha * F.relu(G)).sum(dim=(2, 3))  # [B, C]

        cam = (weights[0].unsqueeze(-1).unsqueeze(-1) * A[0]).sum(0)  # [H, W]
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def _overlay(self, cam: np.ndarray, image: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_uint8 = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        return (alpha * heatmap.astype(np.float32) + (1 - alpha) * img_uint8.astype(np.float32)).clip(0, 255).astype(np.uint8)

    def generate(
        self,
        t_od: torch.Tensor,
        t_os: torch.Tensor,
        raw_od: np.ndarray,
        raw_os: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:

        fh_od, bh_od = self._register_hooks("od")
        fh_os, bh_os = self._register_hooks("os")

        self.model.zero_grad()
        with torch.enable_grad():
            t_od_g = t_od.clone().requires_grad_(True)
            t_os_g = t_os.clone().requires_grad_(True)

            logits, _ = self.model(t_od_g, t_os_g)
            tc = target_class if target_class is not None else int(logits.argmax(-1)[0])
            score = logits[0, tc]
            score.backward()

        cam_od = self._compute_cam("od")
        cam_os = self._compute_cam("os")

        for h in [fh_od, bh_od, fh_os, bh_os]:
            h.remove()
        self.activations.clear()
        self.gradients.clear()

        return {
            "cam_od": cam_od,
            "cam_os": cam_os,
            "overlay_od": self._overlay(cam_od, raw_od),
            "overlay_os": self._overlay(cam_os, raw_os),
        }
