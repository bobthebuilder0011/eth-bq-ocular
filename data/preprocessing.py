import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional
from PIL import Image


def circular_roi_crop(image: np.ndarray, tol: int = 15) -> np.ndarray:
    """
    Fundus images usually have a large black circular border around the
    actual retinal disc. This crops it out so the model focuses on the
    useful part of the image.
    """
    if image is None or image.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # check if already tightly cropped (no heavy black margin)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if np.mean(border) > 35:
        s = min(h, w)
        return image[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]

    _, thresh = cv2.threshold(gray, tol, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.15 * h * w:
        return image

    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = int(cx), int(cy), int(r)

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)

    if x2 <= x1 or y2 <= y1:
        return image

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    out = cv2.bitwise_and(image, image, mask=mask)[y1:y2, x1:x2]

    if out.size == 0 or out.shape[0] < 10 or out.shape[1] < 10:
        return image
    return out


def anisotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 3,
    kappa: float = 15.0,
    gamma: float = 0.15,
) -> np.ndarray:
    """
    Perona-Malik anisotropic diffusion - smooths homogeneous background
    regions while preserving sharp vessel and lesion boundaries.
    kappa=15 is quite tight, meaning only very low-gradient areas get blurred.
    """
    img = image.astype(np.float32)
    nchan = 1 if img.ndim == 2 else img.shape[2]
    diffused = np.zeros_like(img)

    for c in range(nchan):
        u = (img if nchan == 1 else img[:, :, c]).copy()

        for _ in range(n_iter):
            dn = np.zeros_like(u); ds = np.zeros_like(u)
            de = np.zeros_like(u); dw = np.zeros_like(u)

            dn[:-1, :] = u[1:, :] - u[:-1, :]
            ds[1:, :] = u[:-1, :] - u[1:, :]
            de[:, :-1] = u[:, 1:] - u[:, :-1]
            dw[:, 1:] = u[:, :-1] - u[:, 1:]

            cn = np.exp(-((dn / kappa) ** 2))
            cs = np.exp(-((ds / kappa) ** 2))
            ce = np.exp(-((de / kappa) ** 2))
            cw = np.exp(-((dw / kappa) ** 2))

            u += gamma * (cn * dn + cs * ds + ce * de + cw * dw)

        if nchan == 1:
            diffused = u
        else:
            diffused[:, :, c] = u

    return np.clip(diffused, 0, 255).astype(np.uint8)


def tri_channel_clahe(image: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE only on the L* (luminance) channel in Lab space."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(image: np.ndarray, alpha: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """Sharpen edges by adding back a scaled high-frequency residual."""
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigma)
    sharpened = image.astype(np.float32) + alpha * (image.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] float32."""
    img = image.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-6:
        img = (img - mn) / (mx - mn)
    else:
        img = img / 255.0
    return img


def preprocess_fundus_pipeline(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    return_stages: bool = False,
) -> Any:
    """
    Full 5-stage preprocessing pipeline:
      1. ROI crop (remove black borders)
      2. Anisotropic diffusion (smooth background, keep edges)
      3. CLAHE in Lab space (equalise luminance)
      4. Unsharp masking (boost fine detail)
      5. Normalise to [0, 1]
    """
    stages: Dict[str, np.ndarray] = {}

    stages["1_raw"] = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    cropped = circular_roi_crop(image)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    stages["2_roi_crop"] = resized

    diffused = anisotropic_diffusion(resized)
    stages["3_anisotropic_diffusion"] = diffused

    clahe_img = tri_channel_clahe(diffused)
    stages["4_tri_channel_clahe"] = clahe_img

    sharp = unsharp_mask(clahe_img)
    stages["5_unsharp_mask"] = sharp

    norm = normalize_image(sharp)

    if return_stages:
        # convert stage images to float for display
        float_stages = {k: v.astype(np.float32) / 255.0 if v.dtype == np.uint8 else v
                        for k, v in stages.items()}
        return norm, float_stages

    return norm


def fundus_to_tensor(image_norm: np.ndarray) -> torch.Tensor:
    """Convert normalised H×W×3 numpy array to C×H×W tensor."""
    t = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t - mean) / std
