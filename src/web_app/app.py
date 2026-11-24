import os
import sys
import base64
import json
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.class_mapping import CLASS_NAMES, CLASS_CODES, CLASS_METADATA, CLASS_TO_IDX
from inference.predict import ETHBQPipeline
from inference.report_generator import generate_pdf_report
from training.evaluate import SOTA_BENCHMARKS, ABLATION_STUDY

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "eth_bq_best.pth")
pipeline = ETHBQPipeline(checkpoint_path=CHECKPOINT_PATH if os.path.exists(CHECKPOINT_PATH) else None)


def img_to_b64(img: np.ndarray) -> str:
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.route("/")
def index():
    metrics_path = os.path.join(PROJECT_ROOT, "checkpoints", "evaluation_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    demo_dir = os.path.join(PROJECT_ROOT, "data", "benchmark", "test_od")
    demo_files = []
    if os.path.exists(demo_dir):
        for fn in sorted(os.listdir(demo_dir)):
            if fn.startswith("demo_") and fn.endswith("_od.png"):
                name = fn.replace("demo_", "").replace("_od.png", "").capitalize()
                demo_files.append({"name": name, "id": name.lower()})

    return render_template(
        "index.html",
        classes=CLASS_NAMES,
        metadata=CLASS_METADATA,
        sota=SOTA_BENCHMARKS,
        ablation=ABLATION_STUDY,
        metrics=metrics,
        demo_files=demo_files,
    )


def _build_response(result):
    return {
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100.0, 2),
        "class_probabilities": {k: round(v * 100.0, 2) for k, v in result["class_probabilities"].items()},
        "differential_diagnoses": result["differential_diagnoses"],
        "bilateral_symmetry_score": result["bilateral_symmetry_score"],
        "clinical_hallmarks": result["clinical_hallmarks"],
        "latency_ms": result["total_latency_ms"],
        "od_stages": {k: img_to_b64(v) for k, v in result["od_stages"].items()},
        "os_stages": {k: img_to_b64(v) for k, v in result["os_stages"].items()},
        "gradcam_od": img_to_b64(result["gradcam"]["overlay_od"]),
        "gradcam_os": img_to_b64(result["gradcam"]["overlay_os"]),
    }


@app.route("/api/predict_demo", methods=["POST"])
def predict_demo():
    data = request.get_json() or {}
    demo_id = data.get("demo_id", "normal").lower()

    bdir = os.path.join(PROJECT_ROOT, "data", "benchmark")
    od_path = os.path.join(bdir, "test_od", f"demo_{demo_id}_od.png")
    os_path = os.path.join(bdir, "test_os", f"demo_{demo_id}_os.png")

    if not os.path.exists(od_path) or not os.path.exists(os_path):
        return jsonify({"error": f"Demo case '{demo_id}' not found."}), 404

    result = pipeline.predict_pair(od_path, os_path, generate_gradcam=True)
    return jsonify(_build_response(result))


@app.route("/api/predict_upload", methods=["POST"])
def predict_upload():
    file_od = request.files.get("image_od")
    file_os = request.files.get("image_os")

    if (not file_od or not file_od.filename) and (not file_os or not file_os.filename):
        return jsonify({"error": "Please upload at least one fundus image (OD or OS)."}), 400

    od_arr = np.array(Image.open(file_od.stream).convert("RGB")) if (file_od and file_od.filename) else None
    os_arr = np.array(Image.open(file_os.stream).convert("RGB")) if (file_os and file_os.filename) else None

    result = pipeline.predict_pair(od_arr, os_arr, generate_gradcam=True)
    return jsonify(_build_response(result))


@app.route("/api/generate_report", methods=["POST"])
def generate_report():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No diagnosis data provided"}), 400
    try:
        pdf_bytes = generate_pdf_report(data)
        fname = f"ETH_BQ_Report_{int(time.time())}.pdf"
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={fname}"},
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
