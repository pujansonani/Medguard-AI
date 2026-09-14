# MEDGUARD AI: Zero-Leakage Protocol & Verification

## 1. Research Principles for Leakage Prevention
Data leakage is one of the most pervasive flaws in published clinical AI literature. MEDGUARD AI implements a multi-layer verification protocol to guarantee zero temporal and patient-level leakage:

```
                  ┌──────────────────────────────────────────────┐
                  │           MIMIC-IV Patient Cohort            │
                  └──────────────────────┬───────────────────────┘
                                         │
                        [Strict Patient-Level Hash Split]
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼
   ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
   │    Train Fold     │       │  Validation Fold  │       │     Test Fold     │
   │   (70% Patients)  │       │   (15% Patients)  │       │   (15% Patients)  │
   │  Unique subject_id│       │  Unique subject_id│       │  Unique subject_id│
   └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
             │                           │                           │
   [Fit Scalers/Imputers]                │                           │
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │     Temporal Window Boundary: t in [0, 24h]  │
                 │      (Discard all vitals/labs > 24.0h)       │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │    Text Notes Boundary: charttime <= 24.0h   │
                 │     (Strictly exclude Discharge Summaries)   │
                 └──────────────────────────────────────────────┘
```

---

## 2. Four Cardinal Anti-Leakage Rules

### Rule 1: Patient-Level Splitting (Zero Subject Overlap)
- Splits are stratified by 48-hour mortality outcome strictly at the `subject_id` level.
- Patients with multiple ICU admissions will **never** have admissions split between train and test/validation folds.
- **Verification Test:** `test_patient_overlap()` asserts that $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, and $\text{Val} \cap \text{Test} = \emptyset$.

### Rule 2: Strict 24-Hour Observation Window
- Only physiological data (heart rate, blood pressure, lab draws) recorded within $[t_{\text{icu\_intime}}, t_{\text{icu\_intime}} + 24\text{h}]$ are included.
- Any observation with timestamp $> 24.0\text{h}$ is automatically discarded before tensor discretization.
- **Verification Test:** `test_temporal_leakage()` feeds a 30-hour sequence and confirms that only hours $0 \dots 23$ are present in the output tensor.

### Rule 3: Exclusion of Retrospective Clinical Notes
- **Discharge Summaries** are strictly excluded because they are compiled retrospectively after patient outcome is known.
- Any nursing notes, radiology reports, or consult notes charted with $\text{charttime} > 24.0\text{h}$ are excluded.
- **Verification Test:** `test_future_notes_excluded()` verifies that discharge summaries and post-24h notes are filtered out.

### Rule 4: Training-Fold-Only Feature Preprocessing
- All imputation medians, normalization means, standard deviations, and token vocabularies are fitted **exclusively on the training fold**.
- Test and validation tensors are transformed using the fitted training parameters without re-estimating statistics.
- **Verification Test:** `test_preprocessing_fit_only_train()` asserts that test set distribution shifts do not alter scaler parameters.

---

## 3. Automated Leakage Test Suite
Run the dedicated leakage suite at any time:
```bash
pytest tests/test_leakage.py -v
```
All 4 automated tests must pass before any model weights or benchmark results are registered.
