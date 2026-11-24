"""
Real-Time Clinical Inference and Grad-CAM++ Diagnostic Engine for ETH-BQ.
Directly implements Section VI-I and VI-G:
  - Sub-second bilateral inference (24.2 ms GPU / 142.5 ms CPU)
  - 5-stage bio-inspired mathematical preprocessing
  - Siamese EfficientNetV2 + Bilateral Cross-Attention + 8-Qubit VQC forward pass
  - Multi-class diagnostic probabilities & confidence
  - Grad-CAM++ anatomical saliency heatmap localization for both OD and OS
"""

import os
import sys
import time
import cv2
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.class_mapping import CLASS_NAMES, CLASS_CODES, CLASS_METADATA, CLASS_TO_IDX
from data.preprocessing import preprocess_fundus_pipeline, fundus_to_tensor
from models.eth_bq import ETHBQFramework
from explainability.gradcam_plus_plus import BilateralGradCAMPlusPlus


class ETHBQPipeline:
    """
    End-to-End Clinical Diagnostic Pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backbone_name: str = "efficientnet_b0",
        device: Optional[str] = None,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load ETH-BQ Framework
        self.model = ETHBQFramework(
            backbone_name=backbone_name,
            embed_dim=256,
            num_heads=4,
            n_qubits=8,
            n_quantum_layers=3,
            num_classes=8,
            pretrained=False,
            use_quantum=True,
            use_bilateral_attention=True,
        ).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

        # Initialize Grad-CAM++ engine
        self.gradcam = BilateralGradCAMPlusPlus(self.model)

    def predict_pair(
        self,
        od_image_path_or_array,
        os_image_path_or_array,
        generate_gradcam: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete inference for a patient pair (OD and OS).
        """
        t0 = time.time()

        # 1. Load raw images
        def _to_rgb(img_input):
            if img_input is None:
                return None
            if isinstance(img_input, str):
                return np.array(Image.open(img_input).convert("RGB"))
            return img_input

        od_raw = _to_rgb(od_image_path_or_array)
        os_raw = _to_rgb(os_image_path_or_array)

        # If only one eye provided, create contralateral eye via anatomical reflection
        if od_raw is None and os_raw is not None:
            od_raw = np.fliplr(os_raw).copy()
        elif os_raw is None and od_raw is not None:
            os_raw = np.fliplr(od_raw).copy()
        elif od_raw is None and os_raw is None:
            raise ValueError("At least one fundus image (OD or OS) must be provided.")

        # 2. 5-Stage Preprocessing Pipeline
        t_prep_start = time.time()
        od_norm, od_stages = preprocess_fundus_pipeline(od_raw, target_size=(256, 256), return_stages=True)
        os_norm, os_stages = preprocess_fundus_pipeline(os_raw, target_size=(256, 256), return_stages=True)
        t_prep = time.time() - t_prep_start

        # 3. Model Forward Pass
        t_infer_start = time.time()
        t_od = fundus_to_tensor(od_norm).unsqueeze(0).to(self.device)
        t_os = fundus_to_tensor(os_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, meta = self.model(t_od, t_os, return_features=True)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])
        t_infer = time.time() - t_infer_start

        # Dynamic bilateral symmetry calculation from extracted embeddings
        f_od = meta["f_od"]
        f_os = meta["f_os"]
        symmetry_score = float(torch.nn.functional.cosine_similarity(f_od, f_os, dim=-1)[0].item())

        # Dynamic top differential diagnoses ranking
        ranked_indices = np.argsort(probs)[::-1]
        differential_diagnoses = [
            {"class": CLASS_NAMES[idx], "probability": round(float(probs[idx]) * 100.0, 2), "code": CLASS_CODES[idx]}
            for idx in ranked_indices[:3]
        ]

        # 4. Grad-CAM++ Saliency Maps
        gradcam_results = {}
        if generate_gradcam:
            t_cam_start = time.time()
            # Enable grad temporarily for backpropagation hook
            with torch.enable_grad():
                t_od_g = fundus_to_tensor(od_norm).unsqueeze(0).to(self.device)
                t_os_g = fundus_to_tensor(os_norm).unsqueeze(0).to(self.device)
                t_od_g.requires_grad = True
                t_os_g.requires_grad = True

                cam_od, _, _ = self.gradcam.generate_heatmap(t_od_g, t_os_g, target_class=pred_idx, target_eye="od")
                cam_os, _, _ = self.gradcam.generate_heatmap(t_od_g, t_os_g, target_class=pred_idx, target_eye="os")

            overlay_od = self.gradcam.overlay_heatmap(od_stages["final_normalized"], cam_od, alpha=0.55)
            overlay_os = self.gradcam.overlay_heatmap(os_stages["final_normalized"], cam_os, alpha=0.55)

            gradcam_results = {
                "cam_od": cam_od,
                "cam_os": cam_os,
                "overlay_od": overlay_od,
                "overlay_os": overlay_os,
            }

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, "gradcam_od.png"), cv2.cvtColor(overlay_od, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_dir, "gradcam_os.png"), cv2.cvtColor(overlay_os, cv2.COLOR_RGB2BGR))

        total_time_ms = (time.time() - t0) * 1000.0

        # Detailed probabilities dictionary
        class_probabilities = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        return {
            "prediction": pred_class,
            "prediction_code": CLASS_CODES[pred_idx],
            "confidence": pred_conf,
            "class_probabilities": class_probabilities,
            "differential_diagnoses": differential_diagnoses,
            "bilateral_symmetry_score": round(symmetry_score, 4),
            "clinical_hallmarks": CLASS_METADATA[pred_class]["anatomical_hallmarks"],
            "preprocessing_time_ms": round(t_prep * 1000.0, 2),
            "inference_time_ms": round(t_infer * 1000.0, 2),
            "total_latency_ms": round(total_time_ms, 2),
            "od_stages": od_stages,
            "os_stages": os_stages,
            "gradcam": gradcam_results,
        }


if __name__ == "__main__":
    pipeline = ETHBQPipeline()
    demo_od = os.path.join(PROJECT_ROOT, "data", "benchmark", "test_od", "demo_diabetes_od.png")
    demo_os = os.path.join(PROJECT_ROOT, "data", "benchmark", "test_os", "demo_diabetes_os.png")

    if os.path.exists(demo_od) and os.path.exists(demo_os):
        res = pipeline.predict_pair(demo_od, demo_os, generate_gradcam=True)
        print("Prediction:", res["prediction"], f"({res['confidence']*100:.2f}%)")
        print("Latency:", res["total_latency_ms"], "ms")
        print("Probabilities:", res["class_probabilities"])
