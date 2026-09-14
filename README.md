# MEDGUARD AI: Multimodal Explainable Early Warning System for ICU Deterioration

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-ff4b4b.svg)](https://streamlit.io)

---

> **CRITICAL MEDICAL DISCLAIMER: RESEARCH PROTOTYPE ONLY**
> This application is a university research / machine learning prototype developed strictly for experimental and academic evaluation. **It is NOT a certified medical device and must never be used for actual clinical diagnosis, triage, treatment recommendation, or patient management.**

---

## 1. Executive Summary & Core Research Question

In intensive care units (ICUs), patient deterioration often manifests across two distinct clinical modalities:
1. **High-frequency quantitative physiological streams:** Vital signs (heart rate, mean arterial pressure, respiratory rate, oxygen saturation) and laboratory biomarkers (lactate, creatinine, platelets).
2. **Rich unstructured clinical narratives:** Contemporaneous nursing notes, physician progress reports, and multidisciplinary assessments describing bedside nuances (e.g., vasopressor titrations, ventilatory changes, subtle neurological shifts).

### Primary Research Question:
> **"Does incorporating unstructured clinical notes improve early ICU mortality prediction compared with structured physiological and laboratory data alone?"**

MEDGUARD AI provides a rigorous, research-grade, end-to-end framework to answer this question.

---

## 2. Benchmark Model Architectures

| Category | Model | Description | Modality |
| :--- | :--- | :--- | :--- |
| **Baseline 1** | **Logistic Regression** | L2-regularized linear model on 24h summary statistics | Structured Tabular |
| **Baseline 2** | **XGBoost / LightGBM** | Gradient-boosted decision trees on 24h summary statistics | Structured Tabular |
| **Deep Learning** | **Temporal GRU / LSTM** | 2-layer Recurrent Neural Network with missingness masking | Structured Temporal |
| **Clinical NLP** | **ClinicalBERT / NLP** | Clinical transformer document encoder with attention pooling | Unstructured Text |
| **Fusion A** | **Intermediate Concat** | Joint projection of structured ($e_s$) and text ($e_t$) embeddings | Multimodal |
| **Fusion B** | **Gated Multimodal Fusion** | Learned sigmoid gating $g = \sigma(W_s e_s + W_t e_t)$ | Multimodal |
| **Fusion C** | **Cross-Modal Attention** | Multi-head attention across sequence states and text tokens | Multimodal |
| **Clinical Score**| **SOFA / SAPS II Proxies** | Standard ICU severity scoring benchmarks | Classical Clinical |

---

## 3. Leakage Prevention Protocol

1. **Patient-Level Splitting:** Splits are strictly performed on `subject_id`. No patient can ever appear in both training and test/validation folds.
2. **24-Hour Observation Window:** Only vitals, labs, and notes recorded within $[t_{\text{icu\_in}}, t_{\text{icu\_in}} + 24\text{h}]$ are utilized.
3. **Exclusion of Retrospective Notes:** Discharge summaries and notes written $>24\text{h}$ are strictly excluded.
4. **No Premature Scaling:** Scalers and imputers are fitted solely on the training fold.

---

## 4. Quickstart & Installation

### Prerequisites
- Python 3.10 or 3.11
- PyTorch 2.1+

### Installation
```bash
git clone https://github.com/medguard-ai/medguard-ai.git
cd medguard-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run End-to-End Pipeline
```bash
# 1. Generate Synthetic Demo Cohort & Preprocess
python scripts/generate_synthetic_data.py
python scripts/prepare_data.py --config configs/config.yaml

# 2. Train All Models
python scripts/train.py --config configs/config.yaml

# 3. Evaluate & Generate Research Benchmark
python scripts/evaluate.py --config configs/config.yaml
python scripts/generate_report.py
```

---

## 5. Launching Applications

### A. Streamlit Web Dashboard
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### B. FastAPI REST Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

### C. Docker Deployment
```bash
docker-compose up --build
```

---

## 6. Project Structure

```
medguard-ai/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── configs/
│   ├── config.yaml          # Global pipeline & horizon parameters
│   ├── features.yaml        # Biomarkers, normal bounds, clipping
│   └── model_config.yaml    # Network architectures & hyperparameters
│
├── data/
│   ├── raw/                 # MIMIC-IV / MIMIC-IV-Note directory
│   ├── processed/           # 24h temporal tensors (N, 24, 18)
│   └── synthetic/           # Synthetic demo dataset
│
├── src/
│   └── medguard/
│       ├── data/            # Cohort selection, MIMIC loader, synthetic generator
│       ├── preprocessing/   # Temporal discretization, text cleaner, scaling
│       ├── features/        # Tabular summary extractor, SOFA/SAPS II scores
│       ├── models/          # Baselines, Temporal GRU, Clinical NLP, Gated Fusion
│       ├── calibration/     # Temperature scaling, ECE, reliability curves
│       ├── explainability/  # SHAP explainer, text span saliency highlighter
│       ├── evaluation/      # Metrics, bootstrap CIs, ablation, fairness, robustness
│       └── monitoring/      # Feature & prediction drift (PSI, KS-test)
│
├── api/                     # FastAPI backend & Pydantic schemas
├── app/                     # Streamlit clinical intelligence dashboard
├── scripts/                 # CLI runners for data prep, training, eval, and reporting
├── tests/                   # Pytest unit & integration test suite
└── reports/                 # Generated research reports & markdown summaries
```

---

## 7. Running Tests
```bash
pytest tests/ -v
```

---

## 8. Citation
```bibtex
@article{medguard2026,
  title={MEDGUARD AI: Multimodal Explainable Early Warning System for ICU Deterioration},
  author={MEDGUARD AI Research Team},
  year={2026}
}
```
