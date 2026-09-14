# MEDGUARD AI: Research Methodology & Scientific Protocol

## 1. Study Design
- **Cohort:** Adult ICU patients with continuous physiological monitoring ($\ge 24\text{h}$ stay).
- **Observation Window:** First 24 hours of ICU stay $[t_0, t_0 + 24\text{h}]$.
- **Primary Research Endpoint:** In-hospital mortality within 48 hours following the 24-hour observation window.
- **Secondary Multi-Horizon Endpoints:** Mortality at 6h, 12h, and 24h post-observation.

---

## 2. Experimental Ablation Grid

| Experiment ID | Architecture Description | Modality | Research Question Answered |
| :--- | :--- | :--- | :--- |
| **Exp 1** | SOFA & SAPS II Proxies | Clinical Rules | How well do classical clinical scoring systems perform? |
| **Exp 2** | Logistic Regression | Tabular Summary | What is the linear baseline on 24h summary statistics? |
| **Exp 3** | XGBoost | Tabular Summary | What is the non-linear decision tree baseline? |
| **Exp 4** | Temporal GRU with Masks | 3D Time-Series | Does modeling dynamic temporal sequences outperform summary statistics? |
| **Exp 5** | ClinicalBERT NLP | Clinical Notes | How much predictive signal resides in clinical text alone? |
| **Exp 6** | Intermediate Concat Fusion | Multimodal | Does joint dense representation projection outperform single modalities? |
| **Exp 7** | Gated Multimodal Fusion | Multimodal | Does learned sigmoid modality gating optimize information combination? |
| **Exp 8** | Cross-Modal Attention | Multimodal | Can token-level attention improve interaction modeling? |
| **Exp 9** | Temperature Calibrated Fusion | Multimodal | How significantly does post-hoc calibration reduce ECE? |

---

## 3. Scientific Integrity & No Fabrication Policy
All model weights, predictions, metrics, and figures are computed from actual code executions. No results, patient metrics, or claims are fabricated.
