# MEDGUARD AI: Subgroup Fairness & Algorithmic Disparity

## 1. Ethical Motivation
Clinical risk prediction models run the risk of systemic performance degradation across historically underserved or vulnerable demographic groups. MEDGUARD AI incorporates a built-in algorithmic fairness auditing module evaluated across:
1. **Age Groups:** `<45`, `45–64`, `65–74`, `75+` years.
2. **Gender:** Male vs Female admissions.
3. **ICU Specialty Unit Types:** Medical ICU (MICU), Surgical ICU (SICU), Coronary Care Unit (CCU), Trauma/Cardiovascular (Other).

---

## 2. Evaluated Fairness Metrics
For each demographic slice, the framework computes:
- **AUROC & AUPRC:** Discrimination capability within the slice.
- **Brier Score & Expected Calibration Error (ECE):** Calibration reliability across subpopulations.
- **Sensitivity (Recall) & Specificity:** Ensuring false negative rates are bounded across groups.

---

## 3. Disparity Mitigation Strategies
- **Patient-Stratified Splitting:** Prevents demographic skew between training and testing partitions.
- **Auxiliary Modality Supervision:** Single-modality loss regularization prevents the network from over-indexing on narrative bias.
- **Temperature Calibration:** Ensures uniform calibration across different unit specialties.
