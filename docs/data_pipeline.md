# MEDGUARD AI: MIMIC-IV Data Ingestion & Cohort Pipeline

## 1. Data Sources
MEDGUARD AI supports two operational execution modes:
1. **MIMIC Mode:** Ingestion of authenticated raw MIMIC-IV v2.2 (hosp, icu) and MIMIC-IV-Note v2.2 datasets from PhysioNet.
2. **DEMO Mode:** High-fidelity synthetic clinical cohort generated deterministically for offline development, integration testing, and interactive web demonstration.

---

## 2. Cohort Construction Waterfall

```
Raw MIMIC-IV Stays (icustays.csv)
  │
  ▼ [Filter 1: Adult Admissions (anchor_age >= 18)]
Adult ICU Patients
  │
  ▼ [Filter 2: First ICU Admission Only (first_icu_stay_only = True)]
Primary ICU Admissions
  │
  ▼ [Filter 3: Complete 24h Window (los_hours >= 24.0)]
Valid 24h Observation Stays
  │
  ▼ [Filter 4: Required Physiological Streams Present (HR, MAP, SpO2)]
Sufficient Continuous Data
  │
  ▼ [Filter 5: Target Mortality Label Well-Defined within 48h Horizon]
Final Analytical Cohort (configs/cohort.yaml)
```

---

## 3. Physiological Features Monitored
The 18 continuous hourly physiological streams include:

| Variable | Clinical Significance | Normal Adult Range | Imputation Strategy |
| :--- | :--- | :--- | :--- |
| **Heart Rate** | Hemodynamic stability | 60–100 bpm | Forward fill + decay |
| **Systolic BP (SBP)** | Perfusion pressure | 90–140 mmHg | Forward fill + decay |
| **Diastolic BP (DBP)** | Coronary perfusion | 60–90 mmHg | Forward fill + decay |
| **Mean Arterial Pressure (MAP)**| Organ perfusion indicator | 65–105 mmHg | Forward fill + decay |
| **Respiratory Rate** | Respiratory fatigue / failure | 12–20 bpm | Forward fill + decay |
| **Temperature** | Fever / hypothermia / sepsis | 36.5–37.5 °C | Forward fill + decay |
| **SpO2** | Arterial oxygenation | 95–100% | Forward fill + decay |
| **Glasgow Coma Scale (GCS)** | Neurological consciousness | 3–15 | Forward fill + decay |
| **Blood Glucose** | Glycemic dysregulation | 70–140 mg/dL | Forward fill + decay |
| **Serum Creatinine** | Acute Kidney Injury (AKI) | 0.6–1.2 mg/dL | Forward fill + decay |
| **Hemoglobin** | Anemia / hemorrhage | 12.0–17.0 g/dL | Forward fill + decay |
| **White Blood Cells (WBC)** | Infection / systemic inflammation | 4.0–11.0 K/uL | Forward fill + decay |
| **Platelets** | Coagulation / DIC risk | 150–450 K/uL | Forward fill + decay |
| **Sodium (Na+)** | Fluid balance / osmolality | 135–145 mEq/L | Forward fill + decay |
| **Potassium (K+)** | Arrhythmia risk | 3.5–5.0 mEq/L | Forward fill + decay |
| **Serum Bicarbonate (HCO3-)** | Metabolic acidosis / shock | 22–28 mEq/L | Forward fill + decay |
| **Serum Lactate** | Tissue hypoperfusion / sepsis | 0.5–2.0 mmol/L | Forward fill + decay |
| **Blood Urea Nitrogen (BUN)** | Renal dysfunction / azotemia | 7–20 mg/dL | Forward fill + decay |

---

## 4. Ingestion Validation
Run the built-in validator to inspect data integrity:
```bash
# Validate synthetic or processed cohort
python scripts/validate_data.py --data-dir data/synthetic

# Validate raw MIMIC-IV installation
python scripts/validate_data.py --mimic-dir /path/to/mimic-iv-2.2 --notes-dir /path/to/mimic-iv-note-2.2
```
