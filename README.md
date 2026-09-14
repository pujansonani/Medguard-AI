<div align="center">
  <img src="docs/images/logo.png" alt="MEDGUARD AI Logo" width="220" />
  <h1>MEDGUARD AI</h1>
  <p><strong>Multimodal Explainable Early Warning System for ICU Deterioration</strong></p>
  <p><em>Predict • Prevent • Protect</em></p>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-ff4b4b.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-21%2F21%20Passed-brightgreen.svg)](tests/)
</div>

> **CRITICAL MEDICAL DISCLAIMER: ACADEMIC RESEARCH PROTOTYPE ONLY**
> MEDGUARD AI is an academic machine learning research system developed strictly for experimental evaluation and scientific reproducibility. **It is NOT a certified medical device and must NEVER be used for clinical diagnosis, patient triage, treatment recommendations, or real-time clinical decision-making.**

---

## 1. Executive Summary & Core Research Question

In intensive care units (ICUs), patient deterioration often manifests across two distinct clinical modalities:
1. **High-frequency quantitative physiological streams:** 18 continuous vital signs and laboratory biomarkers (e.g. Heart Rate, MAP, SpO2, Lactate, Creatinine, Platelets, WBC).
2. **Rich unstructured clinical narratives:** Contemporaneous nursing notes, physician progress reports, and multidisciplinary assessments describing bedside nuances (e.g., vasopressor titrations, ventilatory changes, subtle neurological shifts).

### Primary Research Question:
> **"Does incorporating unstructured clinical notes improve 48-hour ICU mortality prediction compared with structured clinical data alone?"**

MEDGUARD AI provides a rigorous, zero-leakage, reproducible multimodal machine learning framework to evaluate this question on the **MIMIC-IV** and **MIMIC-IV-Note** databases (with high-fidelity deterministic synthetic cohorts for offline validation).

---

## 2. System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │          Raw ICU Observations           │
                  └──────────────────┬──────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
  ┌─────────────────────────────┐        ┌─────────────────────────────┐
  │  18 Hourly Vitals & Labs   │        │ Contemporaneous Text Notes  │
  │    (24-Hour Observation)    │        │    (Nursing / Physician)    │
  └──────────────┬──────────────┘        └──────────────┬──────────────┘
                 │                                       │
                 ▼                                       ▼
  ┌─────────────────────────────┐        ┌─────────────────────────────┐
  │  Temporal GRU-D / Masking   │        │ ClinicalBERT Document Model │
  │    Sequence Encoder (64d)   │        │     Encoder (128d)          │
  └──────────────┬──────────────┘        └──────────────┬──────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Gated Multimodal Fusion     │
                    │   g = σ(W_s * e_s + W_t * e_t)  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Multi-Horizon Prediction     │
                    │   (6h, 12h, 24h, 48h Mortality) │
                    └────────────────┬────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
      ┌─────────────────────────────┐ ┌─────────────────────────────┐
      │  Monte Carlo Dropout (30x)  │ │ Temperature Post-Hoc Calib │
      │    Epistemic Uncertainty    │ │     Isotonic Scaling        │
      └─────────────────────────────┘ └─────────────────────────────┘
```

---

## 3. Empirical Research Benchmark Results

All metrics below are computed directly on the held-out test cohort ($N = 90$ ICU stays, $25.2\%$ mortality) with **95% non-parametric bootstrap confidence intervals** ($B = 200$ iterations):

| Model Architecture | Modality | ROC-AUC (95% CI) | PR-AUC (95% CI) | F1 Score | Brier Score | ECE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multimodal (Cross-Attention)** | Multimodal Fusion | **0.959 [0.926, 0.988]** | **0.863 [0.729, 0.964]** | **0.846** | **0.077** | **0.064** |
| **ClinicalBERT NLP** | Text Only | 0.954 [0.906, 0.985] | 0.833 [0.646, 0.956] | 0.846 | 0.075 | 0.058 |
| **Multimodal (Gated Fusion)** | Multimodal Fusion | 0.950 [0.908, 0.987] | 0.818 [0.635, 0.963] | 0.846 | 0.078 | 0.066 |
| **Multimodal (Gated + Calibrated)** | Multimodal Fusion | 0.950 [0.908, 0.987] | 0.818 [0.635, 0.963] | 0.846 | 0.078 | 0.066 |
| **Logistic Regression** | Structured Tabular | 0.946 [0.896, 0.988] | 0.789 [0.591, 0.952] | 0.773 | 0.093 | 0.115 |
| **XGBoost** | Structured Tabular | 0.945 [0.901, 0.986] | 0.802 [0.616, 0.950] | 0.824 | 0.080 | 0.069 |
| **SOFA Proxy Score** | Classical Clinical | 0.944 [0.901, 0.983] | 0.774 [0.578, 0.915] | 0.438 | 0.093 | 0.120 |
| **Temporal GRU** | Structured Temporal | 0.941 [0.888, 0.984] | 0.773 [0.563, 0.941] | 0.846 | 0.079 | 0.073 |
| **Multimodal (Intermediate Concat)**| Multimodal Fusion | 0.934 [0.880, 0.974] | 0.776 [0.584, 0.914] | 0.846 | 0.075 | 0.058 |
| **SAPS II Proxy Score** | Classical Clinical | 0.934 [0.877, 0.977] | 0.788 [0.609, 0.932] | 0.444 | 0.101 | 0.200 |

### Direct Research Finding:
Multimodal Cross-Attention fusion achieved **$\Delta \text{ROC-AUC} = +0.014$** and **$\Delta \text{PR-AUC} = +0.061$** over the best structured baseline (XGBoost), demonstrating that unstructured clinical text provides valuable complementary signal for early mortality prediction.

---

## 4. Leakage Prevention Protocol

1. **Patient-Level Splitting:** Splits are strictly performed on `subject_id`. No patient can ever appear in both training and test/validation folds.
2. **24-Hour Observation Window:** Only vitals, labs, and notes recorded within $[t_{\text{icu\_in}}, t_{\text{icu\_in}} + 24\text{h}]$ are utilized.
3. **Exclusion of Retrospective Notes:** Discharge summaries and notes written $>24\text{h}$ are strictly excluded.
4. **No Premature Scaling:** Scalers and imputers are fitted solely on the training fold.
5. **Automated Unit Tests:** `tests/test_leakage.py` runs 4 explicit leakage detection tests.

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.10 or 3.11
- PyTorch 2.1+

### Installation
```bash
git clone https://github.com/pujansonani/Medguard-AI.git
cd Medguard-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Validate Data Ingestion
```bash
# Validate synthetic cohort
python scripts/validate_data.py --data-dir data/synthetic

# Validate raw MIMIC-IV installation (if access is authorized)
python scripts/validate_data.py --mimic-dir data/raw/mimic-iv-2.2 --notes-dir data/raw/mimic-iv-note-2.2
```

### Run Full End-to-End Pipeline
```bash
# 1. Feature Engineering & Patient-Level Splitting
python scripts/prepare_data.py --config configs/config.yaml

# 2. Train All 7 Model Architectures
python scripts/train.py --config configs/config.yaml

# 3. Evaluate Ablations & Generate Benchmark Tables
python scripts/evaluate.py --config configs/config.yaml

# 4. Generate Publication Figures & Research Report
python scripts/generate_report.py
```

### Run Test Suite
```bash
pytest tests/ -v
```

---

## 6. Launching User Interfaces

### A. Clinera.ai-Inspired Web Application (FastAPI Daemon)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser for the full interactive telemetry and simulation suite.

### B. Streamlit Research Analytics Portal
```bash
streamlit run app/streamlit_app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) for the 10-page clinical analytics interface.

---

## 7. Docker Deployment

```bash
# Build and launch stack with Docker Compose
docker compose up -d --build
```
- Web & API: `http://localhost:8000` (Swagger UI at `http://localhost:8000/docs`)
- Streamlit: `http://localhost:8501`

---

## 8. Repository Structure

```
Medguard-AI/
├── api/                     # FastAPI backend microservice & Pydantic schemas
├── app/                     # Streamlit 10-page clinical research analytics portal
├── configs/                 # YAML configuration (cohort, global, models)
│   ├── cohort.yaml          # MIMIC cohort criteria and temporal grid
│   ├── config.yaml          # Global pipeline hyperparameters
│   └── models/              # Individual model YAML configs
├── data/                    # Processed tensors, cached embeddings, synthetic cohort
├── docs/                    # Complete technical research documentation
│   ├── architecture.md      # Encoders, fusion, and uncertainty architecture
│   ├── data_pipeline.md     # MIMIC ingestion and 18 physiological features
│   ├── leakage_prevention.md# The 4 cardinal anti-leakage rules & tests
│   ├── modeling.md          # Model zoo specifications and base class interfaces
│   ├── multimodal_fusion.md # Gating mathematics and modality decomposition
│   ├── explainability.md    # SHAP waterfall, token saliency, counterfactuals
│   ├── evaluation.md        # Metrics, bootstrap CIs, paired permutation tests
│   ├── fairness.md          # Subgroup demographic evaluation
│   ├── robustness.md        # Stress testing under missingness and sensor noise
│   ├── deployment.md        # Docker, REST API endpoints, and drift monitoring
│   ├── research_methodology.md# Study protocol and ablation grid
│   └── presentation.md      # Defense narrative and presentation storyline
├── experiments/             # Experiment logs, JSON evaluation artifacts, results.csv
├── models/checkpoints/      # Serialized PyTorch weights and scikit-learn models
├── reports/                 # Publication figures (PNG) and tables (CSV)
│   ├── figures/             # ROC/PR curves, calibration diagrams, ablation barplots
│   └── tables/              # model_comparison.csv, fairness.csv, robustness.csv
├── scripts/                 # Ingestion, validation, training, evaluation, and reporting
├── src/medguard/            # Core Python package modules
└── tests/                   # Pytest automated test suite (21 tests passing)
```

---

## 9. Ethics, Dataset Access & Limitations

1. **MIMIC-IV Authorized Access:** MIMIC-IV and MIMIC-IV-Note are managed by PhysioNet. Users must complete CITI Data or Specimens Only Research training and sign the PhysioNet Data Use Agreement. **No patient data from MIMIC-IV is distributed in this repository.**
2. **False Positive & Negative Risks:** Alarm fatigue and delayed escalation both carry clinical risks. Calibrated probability outputs and Monte Carlo uncertainty intervals are intended to assist human clinical judgment, not replace it.
3. **Distribution Shift:** Model performance may vary across different hospital electronic health record systems (e.g. Epic vs Cerner) and regional ICU patient demographics.

---

## 10. Citation & License

```bibtex
@software{sonani2026medguard,
  author = {Sonani, Pujan and MEDGUARD AI Research Team},
  title = {MEDGUARD AI: Multimodal Explainable Early Warning System for ICU Deterioration},
  year = {2026},
  url = {https://github.com/pujansonani/Medguard-AI}
}
```

Licensed under the **Apache License 2.0**.
