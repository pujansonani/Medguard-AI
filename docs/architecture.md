# MEDGUARD AI: System & Model Architecture

## 1. Overview
MEDGUARD AI is a research-grade multimodal machine learning system designed to answer a central clinical machine learning question:
> **"Does incorporating unstructured clinical notes improve 48-hour ICU mortality prediction compared with structured physiological data alone?"**

The platform ingests two asynchronous, heterogeneous clinical data modalities within the initial **24-hour observation window**:
1. **Structured Multidimensional Time-Series:** 18 hourly vital signs (e.g., HR, MAP, SpO2, Temperature) and laboratory biomarkers (e.g., Lactate, Creatinine, WBC, Platelets, Potassium).
2. **Unstructured Bedside Clinical Notes:** Nursing progress notes, physician assessments, and multidisciplinary reports written within $[t_0, t_0 + 24\text{h}]$.

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

## 2. Encoder Components

### A. Structured Temporal Encoder (GRU-D with Missingness Masks)
- **Input Tensor:** $X \in \mathbb{R}^{B \times 24 \times 18}$ with corresponding observation mask $M \in \{0, 1\}^{B \times 24 \times 18}$.
- **Imputation:** Forward fill with decaying memory toward the population training baseline.
- **Recurrent Architecture:** 2-layer Gated Recurrent Unit (GRU) with hidden dimension 64 and dropout $p = 0.25$.
- **Output Representation:** $e_s \in \mathbb{R}^{64}$.

### B. Unstructured Clinical Text Encoder (ClinicalBERT)
- **Input Document:** Concatenated clinical narratives recorded $\le 24$ hours into ICU stay.
- **Chunked Tokenizer:** Maximum sequence length of 256 tokens per chunk with 32-token overlap.
- **Backbone:** ClinicalBERT / Bio_ClinicalBERT transformer with attention pooling over chunk embeddings.
- **Output Representation:** $e_t \in \mathbb{R}^{128}$.

---

## 3. Multimodal Fusion Mechanisms

### 1. Late Fusion (Ensemble Logit Combination)
Combines predictions from independently trained structured and text models:
$$\hat{y}_{\text{late}} = \alpha \cdot \hat{y}_{\text{struct}} + (1 - \alpha) \cdot \hat{y}_{\text{text}}$$

### 2. Intermediate Fusion (Dense Concatenation)
Concatenates learned representations into a multilayer perceptron:
$$e_f = \text{ReLU}(W_c [e_s \,\|\, e_t] + b_c)$$

### 3. Gated Multimodal Fusion (Sigmoid Modality Gating)
Dynamically learns patient-specific modality weighting based on the informational richness of each channel:
$$g = \sigma(W_g [e_s \,\|\, e_t] + b_g) \in [0, 1]^{d_f}$$
$$e_f = g \odot \text{Proj}_s(e_s) + (1 - g) \odot \text{Proj}_t(e_t)$$

### 4. Cross-Modal Attention
Computes multi-head cross-attention between token representations and physiological time states:
$$\text{Attn}(Q_{\text{text}}, K_{\text{struct}}, V_{\text{struct}})$$

---

## 4. Multi-Horizon Prediction & Uncertainty
The prediction head outputs logits for 4 clinical lead times:
- **6 Hours** post-observation
- **12 Hours** post-observation
- **24 Hours** post-observation
- **48 Hours** post-observation (Primary research endpoint)

**Uncertainty Quantification:** Epistemic uncertainty is estimated via Monte Carlo Dropout across $K = 30$ stochastic forward passes:
$$\mu_p = \frac{1}{K} \sum_{k=1}^K \sigma(\hat{y}^{(k)}), \quad \sigma_p = \sqrt{\frac{1}{K} \sum_{k=1}^K (\sigma(\hat{y}^{(k)}) - \mu_p)^2}$$
