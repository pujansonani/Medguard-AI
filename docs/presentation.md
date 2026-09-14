# MEDGUARD AI: Project Presentation & Defense Guide

## Executive Presentation Storyline

### 1. The Clinical Problem
- In intensive care units (ICUs), delayed recognition of patient deterioration leads to preventable septic shock, respiratory failure, cardiac arrest, and mortality.
- Traditional bedside monitors issue frequent alarms based on single-variable thresholds, resulting in alarm fatigue without predictive foresight.

---

### 2. The Research Gap
- Most machine learning models rely **strictly on structured numerical tabular data** (vitals/labs) or **strictly on clinical text**.
- Very few systems investigate whether unstructured clinical narratives provide **additive, statistically significant predictive value** beyond continuous vital signs.

---

### 3. The Core Research Question
> **"Does incorporating unstructured clinical notes improve 48-hour ICU mortality prediction compared with structured clinical data alone?"**

---

### 4. Our Multimodal Solution
- **Temporal Deep Learning:** 2-layer GRU-D with forward-fill decay and missingness masks on 18 hourly vital signs and biomarkers.
- **Clinical NLP:** ClinicalBERT transformer encoding nursing progress notes, physician notes, and consult assessments.
- **Gated Multimodal Fusion:** Learned sigmoid gating that dynamically allocates attention weights between structured and narrative evidence.
- **Epistemic Uncertainty:** Monte Carlo Dropout (30 forward passes) providing confidence bounds.
- **Post-Hoc Calibration:** Temperature scaling reducing Expected Calibration Error (ECE).

---

### 5. Key Empirical Conclusions
1. **Clinical Notes Provide Additive Signal:** Multimodal Gated Fusion achieves higher AUROC and AUPRC than structured-only GRU/XGBoost and text-only ClinicalBERT.
2. **Notes Capture Early Bedside Nuance:** Nursing notes document subtle fatigue, vasopressor escalations, and mental status changes hours before laboratory blood panels reveal multi-organ dysfunction.
3. **Rigorous Zero-Leakage Protocol:** Patient-level splitting and strict 24h temporal cutoff ensure zero prospective leakage.
4. **Fairness & Robustness:** Evaluated across demographic subgroups and tested under up to 50% simulated sensor missingness.
