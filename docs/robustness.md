# MEDGUARD AI: Robustness & Stress Testing Under Missingness

## 1. Missingness in Critical Care
In routine ICU operations, sensor leads disconnect, arterial lines are recalibrated, and laboratory blood draws occur at irregular intervals. Real-world models must exhibit graceful degradation under severe data missingness rather than catastrophic failure.

---

## 2. Experimental Stress Testing Protocols

### A. Induced Missingness Decay Test
Evaluates model performance when random physiological variables are masked at increasing drop rates:
- **Missingness Rates Evaluated:** $0\%, 10\%, 20\%, 30\%, 50\%$.
- **Mechanism:** Randomly drop observed channels and observe AUROC, AUPRC, and Brier Score degradation.

### B. Sensor Noise Stress Test
Simulates sensor degradation and electrical interference by injecting Gaussian noise into vital sign channels:
- **Noise Levels Evaluated:** $\sigma \in \{0.0, 0.05, 0.10, 0.20, 0.40\}$.

---

## 3. Results & Observations
- **GRU-D Imputation with Missingness Masks:** Maintaining binary observation masks ($M_{t, f}$) allows the recurrent network to distinguish between true normal measurements and forward-filled imputations.
- **Multimodal Resilience:** Under extreme physiological channel drops (50% missingness), clinical notes provide a vital stabilizing signal that prevents severe degradation in AUROC.
