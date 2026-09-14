# MEDGUARD AI: Multimodal Fusion & Modality Contribution

## 1. Multimodal Gating Hypothesis
In intensive care units, neither quantitative vital sign monitors nor qualitative nursing narratives are sufficient on their own:
- **Structured Physiological Data:** Provides objective, high-frequency, quantitative trend measurements (e.g. rising serum lactate, dropping blood pressure, declining urine output).
- **Unstructured Clinical Notes:** Documents context, bedside impressions, clinical reasoning, ventilatory interventions, vasopressor titrations, and early subtle clinical deterioration (e.g., increased work of breathing, mottled extremities, confusion) hours before vital signs trigger acute alarms.

**Hypothesis:** A gated multimodal architecture that dynamically weights representations based on each modality's informational richness will statistically outperform structured-alone and text-alone models in predicting 48-hour mortality.

---

## 2. Mathematical Formulation

Given structured time-series embedding $e_s \in \mathbb{R}^{d_s}$ and clinical text embedding $e_t \in \mathbb{R}^{d_t}$:

### Step 1: Linear Projections
$$h_s = \text{LayerNorm}(\text{ReLU}(W_{ps} e_s + b_{ps})) \in \mathbb{R}^{d_f}$$
$$h_t = \text{LayerNorm}(\text{ReLU}(W_{pt} e_t + b_{pt})) \in \mathbb{R}^{d_f}$$

### Step 2: Learned Sigmoid Modality Gating
$$g = \sigma(W_{g2} \cdot \text{ReLU}(W_{g1} [e_s \,\|\, e_t] + b_{g1}) + b_{g2}) \in [0, 1]^{d_f}$$

### Step 3: Weighted Representation & Residual Connection
$$e_f = g \odot h_s + (1 - g) \odot h_t$$

### Step 4: Multi-Horizon Classification Head
$$\hat{y} = \sigma(W_o e_f + b_o) \in [0, 1]^4$$

---

## 3. Patient-Level Modality Risk Decomposition
For every assessed patient, the system exposes:
- **$P_{\text{struct}}$:** Mortality probability if only structured vitals/labs were available.
- **$P_{\text{text}}$:** Mortality probability if only clinical notes were available.
- **$P_{\text{multi}}$:** Joint multimodal fused mortality probability.
- **$g$:** Learned gating weight allocated to structured vs text information.
- **$\Delta_{\text{gain}}$:** Increment in predictive certainty from multimodal fusion.
