# MEDGUARD AI: Evaluation Metrics, Calibration & Statistical Significance

## 1. Metrics Suite
MEDGUARD AI employs a comprehensive evaluation protocol across multiple evaluation criteria:

### Discrimination Metrics
- **AUROC (Area Under Receiver Operating Characteristic Curve):** Measures ranking discrimination across the entire decision threshold spectrum.
- **AUPRC (Area Under Precision-Recall Curve):** Primary metric under high class imbalance (~18% mortality rate in ICU cohorts).
- **F1 Score, Sensitivity (Recall), Specificity, Precision, MCC (Matthews Correlation Coefficient).**

### Calibration Metrics
- **Brier Score:** Mean squared error of probabilistic predictions:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
- **Expected Calibration Error (ECE):** Weighted absolute difference between confidence and empirical accuracy across 10 quantile bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

---

## 2. Non-Parametric Bootstrap 95% Confidence Intervals
To ensure statistical rigor, all reported metrics are paired with 95% confidence intervals derived from $B = 200$ bootstrap iterations on held-out test data:
$$\text{Metric} = \hat{\theta} \; [95\%\; \text{CI}: \theta_{\text{low}}, \theta_{\text{high}}]$$

---

## 3. Paired Statistical Significance Testing
To rigorously confirm that multimodal fusion statistically outperforms structured and text baselines, we execute an empirical paired permutation test:
- **Null Hypothesis ($H_0$):** $\text{AUROC}_{\text{Fusion}} = \text{AUROC}_{\text{Structured}}$.
- Computes empirical p-value over $N = 2,000$ permutations.
