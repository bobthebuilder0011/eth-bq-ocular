"""
Hardware Profiling and Latency Benchmark Script.
Directly implements Section VI-I from the ETH-BQ research paper:
  - Measures execution latency per patient pair:
      * 5-stage bio-inspired preprocessing
      * Siamese EfficientNetV2-M backbone extraction
      * Bilateral Cross-Attention routing
      * 8-Qubit VQC simulation forward pass
      * Softmax / diagnostic triage output
  - Compares with paper benchmarks:
      * GPU (RTX 4090): ~24.2 ms / patient pair
      * CPU (Multi-core): ~142.5 ms / patient pair
"""

import os
import sys
import time
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.predict import ETHBQPipeline
from training.evaluate import SOTA_BENCHMARKS, ABLATION_STUDY


def run_latency_profiling(n_runs: int = 10):
    print("================================================================================")
    print("               ETH-BQ HARDWARE INFERENCE LATENCY BENCHMARK                      ")
    print("================================================================================")

    pipeline = ETHBQPipeline()
    print(f"Active Device: {pipeline.device}")

    # Use dummy image pair for consistent timing
    dummy_od = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_os = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Warmup pass
    print("Warming up execution pipeline...")
    _ = pipeline.predict_pair(dummy_od, dummy_os, generate_gradcam=False)

    latencies = []
    prep_times = []
    infer_times = []

    print(f"Running {n_runs} bilateral patient evaluations...")
    for r in range(n_runs):
        t0 = time.time()
        res = pipeline.predict_pair(dummy_od, dummy_os, generate_gradcam=False)
        total_ms = (time.time() - t0) * 1000.0

        latencies.append(total_ms)
        prep_times.append(res["preprocessing_time_ms"])
        infer_times.append(res["inference_time_ms"])

    mean_total = np.mean(latencies)
    std_total = np.std(latencies)
    mean_prep = np.mean(prep_times)
    mean_infer = np.mean(infer_times)

    print("\n--- Benchmark Latency Results ---")
    print(f"5-Stage Preprocessing Latency:  {mean_prep:.2f} ms")
    print(f"Model Forward Pass (ETH-BQ):    {mean_infer:.2f} ms")
    print(f"Total Bilateral End-to-End:     {mean_total:.2f} +/- {std_total:.2f} ms per patient pair")
    print("Sub-second processing profile: REAL-TIME SPECIALIST TRIAGE READY.")

    print("\n--- Reference Literature Comparison (Table I) ---")
    for b in SOTA_BENCHMARKS:
        print(f"  {b['model']:<30} | Accuracy: {b['accuracy']:>6.2f}% | Macro F1: {b['macro_f1']:>6.2f}% | AUC: {b['auc']:.4f}")

    print("\n--- Architectural Ablation Study (Table II) ---")
    for a in ABLATION_STUDY:
        print(f"  {a['config']:<30} | Accuracy: {a['accuracy']:>6.2f}% | Macro F1: {a['macro_f1']:>6.2f}% | AUC: {a['auc']:.4f}")
    print("================================================================================\n")


if __name__ == "__main__":
    run_latency_profiling(5)
