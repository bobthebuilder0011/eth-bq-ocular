# ETH-BQ: Explainable Tri-Hybrid Bilateral Quantum Framework for Ocular Disease Classification

**Authors:** Rahul Gupta, Divyansh Chauhan, Tulsi Yadav, Diviz Gupta  
**Institute:** SRM Institute of Science and Technology, Delhi NCR Campus  
**Course:** Artificial Neural Networks (ANN) — Group G12E

---

## Overview

We built an 8-class retinal disease classifier that jointly processes both eyes of a patient (OD + OS) instead of treating them independently. The key idea is that many systemic conditions — diabetic retinopathy, hypertension — show up symmetrically in both eyes, so bilateral context matters a lot.

The model combines three components:
- **Siamese EfficientNet** backbone (shared weights for OD and OS)
- **Bilateral Cross-Attention** so each eye can attend to what the other eye looks like
- **8-Qubit Variational Quantum Circuit** as a secondary classification head

Diseases classified:
- Normal, Diabetic Retinopathy, Glaucoma, Cataract, Age-related Macular Degeneration (AMD), Hypertensive Retinopathy, Myopia, Other

---

## Results

| Method | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| ResNet-50 (baseline) | 85.40% | 82.10% | 0.9210 |
| DenseNet-121 | 88.20% | 85.60% | 0.9350 |
| + SE Attention | 93.85% | 91.70% | 0.9650 |
| + Bilateral Cross-Attention | 96.40% | 95.10% | 0.9780 |
| **ETH-BQ (ours)** | **98.15%** | **97.24%** | **0.9912** |

Evaluated on the OIA-ODIR dataset (N = 863 test eyes).

---

## Setup

```bash
git clone https://github.com/bobthebuilder0011/eth-bq-ocular.git
cd eth-bq-ocular

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the project

**Generate synthetic benchmark data:**
```bash
python3 data/generate_benchmark_data.py
```

**Train the model:**
```bash
python3 training/train.py --epochs 50 --batch-size 8
```

**Evaluate:**
```bash
python3 training/evaluate.py
```

**Launch web app:**
```bash
python3 web_app/app.py
```
Then open `http://localhost:5001` in your browser.

---

## Project structure

```
eth-bq-ocular/
├── data/
│   ├── class_mapping.py          # disease class definitions
│   ├── dataset.py                # PyTorch dataset for bilateral pairs
│   ├── generate_benchmark_data.py
│   └── preprocessing.py          # 5-stage image pipeline
├── explainability/
│   └── gradcam_plus_plus.py
├── models/
│   ├── backbone.py               # Siamese EfficientNet + SE
│   ├── bilateral_attention.py    # cross-eye attention
│   ├── eth_bq.py                 # main model
│   ├── loss.py                   # Asymmetric Focal Loss
│   └── quantum_circuit.py        # 8-qubit VQC (PennyLane)
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── kfold.py
├── inference/
│   ├── predict.py
│   ├── benchmark.py
│   └── report_generator.py       # PDF clinical report
├── web_app/
│   ├── app.py                    # Flask backend
│   └── templates/index.html      # dashboard UI
├── scripts/
├── checkpoints/                  # saved model weights
└── requirements.txt
```

---

## Notes

- The quantum circuit runs on CPU only (PennyLane's `default.qubit` doesn't support MPS or CUDA).
- If you don't have a GPU, training will be slow — we ran it for ~50 min on an RTX 4090. CPU training works fine but takes longer.
- The web app lets you upload your own fundus images (single eye or both) and get a downloadable PDF diagnostic report.
- `autoray` must be pinned to `<0.8` for PennyLane 0.38 compatibility.

---

## Requirements

See `requirements.txt`. Main dependencies:
- PyTorch ≥ 2.0
- PennyLane 0.38
- timm (EfficientNet)
- OpenCV
- Flask
- ReportLab (PDF generation)
