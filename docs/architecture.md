# MEDGUARD AI: Technical Architecture & Methodology

## 1. System Overview
MEDGUARD AI investigates whether unstructured clinical notes provide complementary predictive value beyond structured physiological time-series for predicting ICU patient deterioration (mortality within 48 hours post-observation).

```
                      ┌────────────────────────────────────────┐
                      │    24-Hour ICU Observation Window      │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │  Structured Time-Series   │                   │  Clinical Narrative Notes │
    │  (18 Vitals & Biomarkers) │                   │  (Nursing, Physician)     │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │ Temporal Discretization & │                   │ De-ID Cleaning, Chunking  │
    │ Missingness Mask Stacking │                   │ & Transformer Tokenizer   │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  ▼                                               ▼
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │    Temporal GRU Encoder   │                   │   Clinical NLP Encoder    │
    │  Structured State e_s     │                   │     Text State e_t        │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────┐
                            │   Gated Multimodal Fusion │
                            │  g = σ(W_s e_s + W_t e_t) │
                            └─────────────┬─────────────┘
                                          │
                                          ▼
                            ┌───────────────────────────┐
                            │ Multi-Horizon Prediction  │
                            │   6h | 12h | 24h | 48h    │
                            └─────────────┬─────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
       ┌───────────────────────────┐             ┌───────────────────────────┐
       │   Explainability Engine   │             │ Uncertainty & Calibration │
       │  • SHAP Waterfall / Bars  │             │ • Monte Carlo Dropout     │
       │  • Temporal Dynamics      │             │ • Temperature Scaling     │
       │  • Text Span Saliency     │             │ • ECE & Reliability Curve │
       └───────────────────────────┘             └───────────────────────────┘
```

## 2. Mathematical Formulation
- Let $\mathbf{X}_s \in \mathbb{R}^{T \times D}$ be the discretized 24-hour physiological sequence ($T=24, D=18$).
- Let $\mathbf{M} \in \{0, 1\}^{T \times D}$ be the binary missingness observation mask.
- Let $\mathbf{X}_t$ be the clinical notes occurring during $[t_0, t_0 + 24\text{h}]$.

### Modality Encoders:
$$\mathbf{e}_s = \text{Encoder}_{\text{GRU}}(\mathbf{X}_s \oplus \mathbf{M}) \in \mathbb{R}^{d_s}$$
$$\mathbf{e}_t = \text{Encoder}_{\text{NLP}}(\text{Tokenize}(\mathbf{X}_t)) \in \mathbb{R}^{d_t}$$

### Gated Multimodal Fusion:
$$\mathbf{g} = \sigma(\mathbf{W}_s \mathbf{e}_s + \mathbf{W}_t \mathbf{e}_t + \mathbf{b}_g)$$
$$\mathbf{e}_f = \mathbf{g} \odot \text{Proj}_s(\mathbf{e}_s) + (1 - \mathbf{g}) \odot \text{Proj}_t(\mathbf{e}_t)$$

### Multi-Horizon Head & Epistemic Uncertainty:
$$\hat{\mathbf{y}} = \sigma(\mathbf{W}_h \text{Dropout}(\mathbf{e}_f) + \mathbf{b}_h) \in \mathbb{R}^4$$
Uncertainty $\sigma_{\text{epistemic}}$ is quantified via $K=30$ Monte Carlo dropout forward passes.
