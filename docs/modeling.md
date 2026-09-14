# MEDGUARD AI: Model Zoo & Architectures

## 1. Unified Model Interface
All models in MEDGUARD AI inherit from `BaseMedguardModel` and implement:
- `fit(X, y, ...)`: Trains model strictly on training fold.
- `predict_proba(X, ...)`: Generates continuous risk probabilities.
- `save(filepath)`: Serializes model artifacts and metadata to disk.
- `load(filepath)`: Restores model weights for inference.

---

## 2. Model Zoo Summary

| Model | Class | Modality | Key Parameters |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `LogisticRegressionBaseline` | Tabular Summary | $L_2$ regularized, balanced class weight |
| **XGBoost** | `XGBoostBaseline` | Tabular Summary | `max_depth=4`, `lr=0.05`, `scale_pos_weight=3.5` |
| **Temporal GRU** | `TemporalGRUModel` | 3D Time-Series | 2-layer GRU, `hidden_dim=64`, missingness mask |
| **ClinicalBERT NLP** | `ClinicalNLPModel` | Clinical Text | Clinical document encoder with attention pooling |
| **Late Fusion** | `MedguardLateFusion` | Multimodal | Calibrated logit ensembling |
| **Intermediate Concat** | `MedguardMultimodalModel` | Multimodal | Dense MLP projection of $[e_s \,\|\, e_t]$ |
| **Gated Multimodal Fusion** | `MedguardMultimodalModel` | Multimodal | Learned sigmoid gating $g = \sigma(W [e_s \,\|\, e_t])$ |
| **Cross-Modal Attention** | `MedguardMultimodalModel` | Multimodal | 4-head multi-head cross attention |
| **SOFA / SAPS II / APACHE II**| `ClinicalScoreCalculator` | Clinical Benchmark | Standard ICU organ failure scores |
