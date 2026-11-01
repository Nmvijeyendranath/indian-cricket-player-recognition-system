---
title: Indian Cricket Player Face Recognition
emoji: 🏏
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Indian Cricket Team Member Face Identification

Production-quality facial identification system for 323 Indian cricket players using pretrained **ArcFace (buffalo_l)** embeddings with a **Calibrated Linear Support Vector Machine (SVM)** and **confidence-based unknown-person rejection**, optimized for GPU hardware acceleration and streaming chunked data processing.

---

## 1. Architecture Overview

```
Input Image 
    │
    ▼
Face Detection & Alignment (InsightFace buffalo_l on NVIDIA GPU / CUDA)
    │  [Single-face enforcement: rejects 0 faces or >1 faces]
    ▼
512-Dimensional ArcFace Metric Embedding (L2-normalized)
    │
    ▼
Calibrated Linear SVM (Balanced Class Weights, Sigmoid Calibration)
    │
    ├── Calibrated Probability Distribution across 323 Players
    └── Dynamic Rejection Thresholding (Tuned on Held-Out Validation Split)
         ├── Confidence >= 0.10: Return Identified Player + Top-3 Candidates
         └── Confidence <  0.10: Return unknown: True, identity: null
```

---

## 2. Dataset Audit & Preprocessing

The raw dataset contains 7,687 images across 324 folders. A complete automated audit was executed:
- **Corrupt / Null Bytes**: 13 files completely zero-padded (`0x00`) excluded (9 in `Ravi Shastri`, 4 in `Randhir Singh`). `Ravi Shastri` was excluded due to 0 valid images.
- **Cross-Identity Conflicts**: 40 images (20 identical file pairs) assigned to conflicting players were purged to prevent label poisoning.
- **Same-Identity Duplicates**: 3 redundant exact duplicates removed.
- **Folder Names Cleaned**: Scraper artifacts stripped (e.g. `M.S. Dhoni1` $\rightarrow$ `M.S. Dhoni`, `Rahul Dravid2` $\rightarrow$ `Rahul Dravid`).
- **Clean Retained Dataset**: 7,631 images across 323 distinct player classes.
- **Stratified Leakage-Safe Splits**:
  - `manifest_train.csv`: 5,215 images (~68.3%)
  - `manifest_val.csv`: 1,208 images (~15.8%)
  - `manifest_test.csv`: 1,208 images (~15.8%)

---

## 3. Measured Evaluation Results (Held-Out Test Set)

Both architectures were trained and evaluated on the exact same dataset splits on an **NVIDIA GeForce RTX 3050 6GB Laptop GPU**:

| Metric | ArcFace + Calibrated SVM (Primary) | EfficientNet-B0 (Experimental CNN) |
| :--- | :---: | :---: |
| **Model Type** | Pretrained Metric Learning + SVM | Transfer Learning (ImageNet weights) |
| **Top-1 Accuracy** | **71.55%** | 14.24% |
| **Top-3 Accuracy** | **79.71%** | 23.10% |
| **Accuracy on Accepted Faces** | **80.58%** | N/A (no rejection head) |
| **Macro Precision / Recall** | 59.28% / 67.78% | N/A |
| **Macro / Weighted F1** | 61.51% / 65.43% | N/A |
| **Rejection Rate (Threshold = 0.10)**| 13.50% | N/A |
| **Inference Latency** | ~38 ms / image (GPU) | 14.66 ms / image (GPU) |

*Analysis*: With only ~14 training samples per player across 323 classes, standard CNNs overfit severely. ArcFace hyperspherical embeddings provide robust identity separation out of the box, achieving high accuracy with zero dataset leakage.

---

## 4. Quick Start & Execution

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Reproduce Dataset Cleaning & Stratification
```bash
python -m scripts.prepare_dataset
```

### Reproduce Feature Extraction & Model Training (GPU-Accelerated)
```bash
python -m scripts.train \
  --manifest data/manifest_train.csv \
  --val-manifest data/manifest_val.csv \
  --root "data/archive (3)/indian cricketer" \
  --out models/face_index.joblib \
  --cache-dir data/embeddings \
  --chunk-size 256
```

### Run Model Evaluation & Generate Reports
```bash
python -m scripts.evaluate \
  --manifest data/manifest_test.csv \
  --model models/face_index.joblib \
  --root "data/archive (3)/indian cricketer" \
  --cache-dir data/embeddings \
  --reports-dir reports \
  --chunk-size 256
```

### Train Experimental EfficientNet-B0 Baseline
```bash
python -m scripts.train_efficientnet \
  --train-manifest data/manifest_train.csv \
  --val-manifest data/manifest_val.csv \
  --test-manifest data/manifest_test.csv \
  --root "data/archive (3)/indian cricketer"
```

### Run Unit Tests
```bash
pytest
```

### Launch API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- Interactive Web Demo: `http://localhost:8000`
- Swagger API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- REST Inference: `curl -F file=@cricketer.jpg 'http://localhost:8000/identify?top_k=3'`
