# MEDGUARD AI: Explainability Engine (SHAP, Text Saliency & Counterfactuals)

## 1. Overview
In clinical decision support, black-box predictions are unacceptable. MEDGUARD AI implements multimodal explainability across three distinct dimensions:
1. **Global & Local Structured Feature Attributions:** Kernel & Tree SHAP on engineered 24h summary statistics and trajectory trends.
2. **Temporal Dynamics Attribution:** Step-by-step physiological deviation analysis tracing when specific abnormalities (e.g. rising lactate at hour 14) began elevating risk.
3. **Unstructured Clinical Text Saliency:** Gradient & attention saliency mapping highlighting critical phrases (e.g. *"hypotension refractory to norepinephrine"*, *"ARDS"*, *"anuria"*).
4. **Counterfactual Model Sensitivity Probes:** Controlled what-if analysis showing how model risk shifts when features are normalized to reference ranges.

---

## 2. Structured SHAP Analysis

### Local Waterfall Explanations
For every patient prediction, SHAP computes additive feature contributions:
$$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$
where $\phi_0$ is the base expected mortality rate across the population, and $\phi_i$ is the positive or negative contribution of feature $i$.

### Top Driving Factors
- **High-Risk Drivers (+SHAP):** Elevated serum lactate ($\ge 4.0$ mmol/L), refractory hypotension ($\text{MAP} < 60$ mmHg), severe oliguria, thrombocytopenia, metabolic acidosis ($\text{HCO}_3 < 18$).
- **Protective Factors (-SHAP):** Preserved neurological status ($\text{GCS} = 15$), normal baseline creatinine, stable oxygenation on room air ($\text{SpO}_2 \ge 98\%$).

---

## 3. Clinical Text Saliency Mapping
Token and sentence attribution is extracted from the ClinicalBERT text encoder:
- High-risk clinical tokens receive positive saliency weights: `[vasopressor]`, `[hypotension]`, `[lethargic]`, `[ARDS]`, `[intubated]`, `[acidosis]`.
- Low-risk stability tokens receive negative/neutral weights: `[extubated]`, `[alert]`, `[ambulating]`, `[afebrile]`, `[stable]`.

---

## 4. Counterfactual Sensitivity Probes

> **CRITICAL MEDICAL DISCLAIMER:**
> Counterfactual probes are strictly analytical sensitivity checks to explore model decision boundaries. They do NOT constitute medical advice, triage guidance, or clinical treatment recommendations.

### Example Probe:
- **Baseline State:** 48h Mortality Risk = 82.4% (Patient with $\text{MAP} = 54$ mmHg, Lactate = $5.2$ mmol/L, $\text{SpO}_2 = 88\%$).
- **Simulated Normalization:** Normalize $\text{MAP} \to 75$ mmHg, Lactate $\to 1.5$ mmol/L, $\text{SpO}_2 \to 98\%$.
- **Model Sensitivity Response:** Counterfactual Risk shifts to $24.1\%$ ($\Delta = -58.3\%$).
