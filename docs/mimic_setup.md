# MIMIC-IV Clinical Dataset Setup Guide

## 1. Requesting PhysioNet Access
1. Register on [PhysioNet](https://physionet.org/).
2. Complete CITI training course: "Human Subjects Research - Data or Specimens Only".
3. Sign the Data Use Agreement (DUA) for:
   - [MIMIC-IV (v2.2)](https://physionet.org/content/mimiciv/2.2/)
   - [MIMIC-IV-Note (v2.2)](https://physionet.org/content/mimic-iv-note/2.2/)

## 2. Directory Placement
Download the compressed CSV or Parquet files and place them under `data/raw/`:
```
data/raw/
├── mimic-iv-2.2/
│   ├── icu/icustays.csv.gz
│   ├── hosp/patients.csv.gz
│   ├── hosp/admissions.csv.gz
│   ├── icu/chartevents.parquet
│   └── hosp/labevents.parquet
└── mimic-iv-note-2.2/
    ├── discharge.csv.gz
    └── radiology.csv.gz
```

## 3. Ingestion & Preprocessing
Configure your `.env` file:
```bash
MEDGUARD_MODE=MIMIC
MIMIC_IV_DATA_DIR=data/raw/mimic-iv-2.2
MIMIC_IV_NOTE_DIR=data/raw/mimic-iv-note-2.2
```
Then run the data preparation pipeline:
```bash
python scripts/prepare_data.py --config configs/config.yaml
```
