# ETH-BQ: Explainable Tri-Hybrid Bilateral Quantum Framework for Ocular Disease Classification

**Students:**
- Divyansh Chauhan (RA2411026030280)
- Tulsi Yadav (RA2411026030251)
- Harshit Chaudhary (RA2411026030268)

**Department:** Department of Computer Science and Engineering  
**Institution:** SRM Institute of Science and Technology, Delhi NCR Campus, Modinagar  
**Course:** Machine Learning (21CSC305P) — B.Tech CSE (AIML), 5th Semester  
**Faculty In-Charge:** Dr. Saurabh Gupta  
**Head of Department:** Dr. Neelam Singh  

---

## Academic Documents & Attached Deliverables

- **`ETH_BQ_Project_Report.pdf`** — Official project report titled *"Multi-Ocular Disease Classification via Siamese EfficientNetV2, Bilateral Cross-Attention and Variational Quantum Circuits"*, signed and submitted for the viva-voce.
- **`ETH_BQ_Presentation.pptx`** — Project presentation deck detailing system architecture, 5-stage image preprocessing pipeline, benchmark results, and Grad-CAM++ clinical explainability.
- **`ML_Lab_Practical_File_Divyansh_Chauhan.pdf`** — Complete Machine Learning Laboratory Practical File (13 experiments covering Iris classification, Linear/Bayesian regression, SVM, K-Means, GMM, PCA, HMM, CART, Ensemble models, KNN, Naive Bayes, Factor Analysis, and Agglomerative/Divisive clustering).

---

## Project Overview

We built an 8-class retinal disease classifier that jointly processes both eyes of a patient (OD: Right Eye, OS: Left Eye) instead of treating them in isolation. Systemic pathologies (e.g., diabetic retinopathy, hypertensive retinopathy) present bilateral manifestations, making contralateral context essential for high-fidelity clinical screening.

The ETH-BQ framework integrates three core architectural components:
1. **Siamese EfficientNetV2-M Backbone:** Strict weight-sharing ($W_{OD} = W_{OS}$) with Squeeze-and-Excitation (SE, $r = 16$) channel recalibration to isolate pathology-discriminative features.
2. **Bilateral Cross-Attention Module:** 4-head Query-Key-Value attention mechanism enabling bidirectional contralateral feature routing ($F_{\text{fused}} \in \mathbb{R}^{512}$).
3. **8-Qubit Parameterized Variational Quantum Circuit (VQC):** PennyLane-based angle embedding ($R_Y$) into a 256-dimensional Hilbert space ($2^8 = 256$) with 3 strongly entangling layers and Pauli-Z expectation value readouts ($\langle Z_j \rangle$).
4. **Asymmetric Focal Loss (ASL):** $\gamma_+ = 0, \gamma_- = 4, m = 0.05$ to mitigate extreme long-tail class imbalance (hypertensive retinopathy ~4.6% vs normal ~28.9%).
5. **Grad-CAM++ Explainability:** 2nd- and 3rd-order gradient-weighted visual activation heatmaps ($\alpha = 0.55$) localizing anatomical lesions (microaneurysms, hemorrhages, optic cup enlargement, drusen).

---

## Key Results on OIA-ODIR Cohort (N = 863 Test Eyes)

- **Overall Classification Accuracy:** **98.15%** (847 / 863 eyes)
- **Macro F1-Score:** **97.24%**
- **Macro AUC-ROC:** **0.9912**
- **Quadratic Weighted Cohen's Kappa:** **0.9765**
- **Inference Latency:** 24.2 ms on GPU / 142.5 ms on CPU

| Model Architecture | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Baseline ResNet-50 | 85.40% | 82.10% | 0.9210 |
| Standard DenseNet-121 | 88.20% | 85.60% | 0.9350 |
| Baseline DIA-VXNET | 56.50% | 50.07% | 0.7739 |
| Bilateral Cross-Attention | 97.75% | 96.69% | 0.9879 |
| **Proposed ETH-BQ (Tri-Hybrid)** | **98.15%** | **97.24%** | **0.9912** |

---

## Repository Structure

```
eth-bq-ocular/
├── ETH_BQ_Project_Report.pdf                 # Complete project report with signed certificates
├── ETH_BQ_Presentation.pptx                 # Project presentation deck
├── ML_Lab_Practical_File_Divyansh_Chauhan.pdf# Machine Learning practical file (13 experiments)
├── README.md
├── requirements.txt
├── .gitignore
└── src/                                      # Complete project implementation
    ├── data/
    │   ├── class_mapping.py                  # 8 disease classes & ICD-10 metadata
    │   ├── dataset.py                        # Bilateral fundus pair PyTorch Dataset
    │   ├── generate_benchmark_data.py        # Benchmark data generator
    │   └── preprocessing.py                  # 5-stage bio-inspired image pipeline
    ├── models/
    │   ├── backbone.py                       # Siamese EfficientNetV2 + SE block
    │   ├── bilateral_attention.py            # Bilateral cross-modal attention
    │   ├── quantum_circuit.py                # 8-qubit VQC in PennyLane
    │   ├── loss.py                           # Asymmetric Focal Loss (ASL)
    │   └── eth_bq.py                         # Master Tri-Hybrid framework
    ├── explainability/
    │   └── gradcam_plus_plus.py              # Bilateral Grad-CAM++ engine
    ├── training/
    │   ├── train.py                          # Training loop with AdamW & Cosine Annealing
    │   ├── evaluate.py                       # Empirical evaluation & ROC/CM generation
    │   └── kfold.py                          # Patient-level stratified 5-fold CV
    ├── inference/
    │   ├── predict.py                        # Real-time inference engine
    │   ├── benchmark.py                      # Latency profiling
    │   └── report_generator.py               # Clinical PDF diagnostic report generator
    ├── web_app/
    │   ├── app.py                            # Flask server backend
    │   ├── templates/index.html              # Diagnostic dashboard UI
    │   └── static/                           # Assets and images
    ├── scripts/
    └── checkpoints/                          # Training history & evaluation metrics
```

---

## Setup & Execution

### 1. Environment Setup
```bash
git clone https://github.com/bobthebuilder0011/eth-bq-ocular.git
cd eth-bq-ocular

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Inference & Interactive Web Application
```bash
cd src
python3 web_app/app.py
```
Open **`http://localhost:5001`** to:
- Test 8 disease presets or upload custom bilateral/unilateral fundus images.
- Inspect the 5-stage image preprocessing stages.
- View bilateral Grad-CAM++ saliency heatmaps.
- Generate and download comprehensive clinical PDF diagnostic reports.

### 3. Model Training & Evaluation
```bash
cd src
python3 training/train.py --epochs 50 --batch-size 8
python3 training/evaluate.py
```
