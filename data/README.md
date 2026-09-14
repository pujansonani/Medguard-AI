# MEDGUARD AI: Data Directory & MIMIC-IV Ingestion Guide

## Directory Structure
```
data/
├── raw/                 <- Raw MIMIC-IV and MIMIC-IV-Note CSV/Parquet files (PhysioNet credentialed access)
│   ├── mimic-iv-2.2/    <- icustays, admissions, patients, chartevents, labevents
│   └── mimic-iv-note-2.2/ <- discharge.csv, nursing.csv, physician.csv
├── interim/             <- Temporary pre-filtered cohorts and intermediate files
├── processed/           <- Extracted 24h temporal tensors and tokenized notes (patient-split)
├── synthetic/           <- Clinically realistic synthetic demo cohort for testing & demonstrations
└── README.md            <- This documentation
```

## MIMIC-IV Compliance & Setup Instructions
To run MEDGUARD AI on real clinical data:
1. Complete CITI Human Subjects Research & HIPAA training on [PhysioNet](https://physionet.org/).
2. Request credentialed access to [MIMIC-IV (v2.2)](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV-Note (v2.2)](https://physionet.org/content/mimic-iv-note/2.2/).
3. Place raw files or configure your paths in `.env` or `configs/config.yaml`.
4. Run the data preparation pipeline:
   ```bash
   python scripts/prepare_data.py --config configs/config.yaml
   ```

## Leakage Prevention Protocol
1. **Patient-Level Splitting:** Splits are strictly performed on `subject_id`. No patient can appear in both training and test/validation folds.
2. **24-Hour Observation Window:** Only vitals, labs, and clinical notes timestamped within $[t_{\text{icu\_in}}, t_{\text{icu\_in}} + 24\text{h}]$ are utilized.
3. **Exclusion of Retrospective Notes:** Discharge summaries and any notes recorded after $t_{\text{icu\_in}} + 24\text{h}$ are strictly excluded to avoid future leakage.
4. **No Premature Scaling:** Imputation statistics (medians) and standard scalers are fitted solely on the training fold and applied to validation/test folds.
