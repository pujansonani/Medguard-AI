"""
Data Preparation & Feature Engineering Pipeline:
Loads raw MIMIC or synthetic cohort, performs patient-level splitting, extracts 3D temporal tensors,
tabular summary features, and tokenized clinical note texts.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.data.cohort import CohortSelector, split_cohort_patient_level
from medguard.data.synthetic_generator import SyntheticDataGenerator
from medguard.data.mimic_loader import MIMICDataLoader
from medguard.preprocessing.time_series import TimeSeriesPreprocessor
from medguard.preprocessing.text_cleaner import ClinicalTextPreprocessor
from medguard.preprocessing.scaling import TemporalStandardScaler
from medguard.features.temporal_extractor import TabularFeatureExtractor
from medguard.features.clinical_scores import ClinicalScoreCalculator
from medguard.utils.logging import get_logger
from medguard.utils.config import get_project_root, load_all_configs

logger = get_logger("medguard.scripts.prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and features for MEDGUARD AI.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to main config")
    parser.add_argument("--force_synthetic", action="store_true", help="Force using synthetic data")
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    mode = cfg.get("mode", "DEMO")
    if args.force_synthetic:
        mode = "DEMO"

    raw_mimic_dir = root / cfg.get("paths", {}).get("raw_mimic_dir", "data/raw/mimic-iv-2.2")
    mimic_loader = MIMICDataLoader(raw_mimic_dir=raw_mimic_dir)

    if mode == "MIMIC" and mimic_loader.check_availability():
        logger.info("MIMIC-IV raw dataset detected. Ingesting MIMIC-IV cohort...")
        df_patients = mimic_loader.load_cohort()
        # In a full MIMIC-IV ingestion, load chartevents and labevents into df_timeseries and notes
        raise NotImplementedError("For MIMIC-IV full extraction, configure local database connections.")
    else:
        logger.info("Using DEMO / Synthetic Cohort Pipeline.")
        synth_dir = root / cfg.get("paths", {}).get("synthetic_dir", "data/synthetic")
        
        # Load or generate synthetic cohort
        if (synth_dir / "patients.parquet").exists():
            logger.info("Loading pre-generated synthetic files...")
            df_patients = pd.read_parquet(synth_dir / "patients.parquet")
            df_timeseries = pd.read_parquet(synth_dir / "timeseries.parquet")
            df_notes = pd.read_parquet(synth_dir / "notes.parquet")
        else:
            logger.info("Generating synthetic files on the fly...")
            generator = SyntheticDataGenerator(seed=cfg.get("seed", 42))
            df_patients, df_timeseries, df_notes = generator.generate_cohort(
                num_patients=cfg.get("synthetic", {}).get("num_patients", 600)
            )
            synth_dir.mkdir(parents=True, exist_ok=True)
            df_patients.to_parquet(synth_dir / "patients.parquet", index=False)
            df_timeseries.to_parquet(synth_dir / "timeseries.parquet", index=False)
            df_notes.to_parquet(synth_dir / "notes.parquet", index=False)

    # 1. Cohort Filtering
    cohort_cfg = cfg.get("cohort", {})
    selector = CohortSelector(
        min_age=cohort_cfg.get("min_age", 18.0),
        min_icu_los_hours=cohort_cfg.get("min_icu_los_hours", 24.0),
    )
    df_cohort = selector.filter_cohort(df_patients)

    # 2. Patient-Level Splitting (Strictly Leakage-Free)
    split_cfg = cfg.get("splitting", {})
    df_train, df_val, df_test = split_cohort_patient_level(
        df=df_cohort,
        patient_col="subject_id",
        stratify_col=split_cfg.get("stratify_col", "mortality_48h"),
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
    )

    # 3. Temporal 3D Tensor Extraction
    ts_preprocessor = TimeSeriesPreprocessor(seq_len=24)
    # Fit strictly on training fold time-series observations
    train_stay_ids = df_train["stay_id"].tolist()
    val_stay_ids = df_val["stay_id"].tolist()
    test_stay_ids = df_test["stay_id"].tolist()

    df_ts_train = df_timeseries[df_timeseries["stay_id"].isin(set(train_stay_ids))]
    ts_preprocessor.fit(df_ts_train)

    X_train_raw, mask_train = ts_preprocessor.transform(df_timeseries, train_stay_ids)
    X_val_raw, mask_val = ts_preprocessor.transform(df_timeseries, val_stay_ids)
    X_test_raw, mask_test = ts_preprocessor.transform(df_timeseries, test_stay_ids)

    # 4. Fit Temporal Standard Scaler (strictly on training fold)
    scaler = TemporalStandardScaler()
    scaler.fit(X_train_raw, mask_train)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Extract Tabular Summary Features
    tab_extractor = TabularFeatureExtractor()
    df_tab_train = tab_extractor.transform(X_train_raw, mask_train, df_train)
    df_tab_val = tab_extractor.transform(X_val_raw, mask_val, df_val)
    df_tab_test = tab_extractor.transform(X_test_raw, mask_test, df_test)

    # 6. Compute Clinical Severity Scores (SOFA & SAPS II)
    score_calc = ClinicalScoreCalculator()
    df_scores_train = score_calc.calculate_scores_df(df_tab_train)
    df_scores_val = score_calc.calculate_scores_df(df_tab_val)
    df_scores_test = score_calc.calculate_scores_df(df_tab_test)

    # 7. Aggregate Clinical Notes (strictly within 24h cutoff)
    text_cleaner = ClinicalTextPreprocessor()
    notes_dict_train = text_cleaner.aggregate_patient_notes(df_notes, train_stay_ids, max_cutoff_hour=24.0)
    notes_dict_val = text_cleaner.aggregate_patient_notes(df_notes, val_stay_ids, max_cutoff_hour=24.0)
    notes_dict_test = text_cleaner.aggregate_patient_notes(df_notes, test_stay_ids, max_cutoff_hour=24.0)

    texts_train = [notes_dict_train[s] for s in train_stay_ids]
    texts_val = [notes_dict_val[s] for s in val_stay_ids]
    texts_test = [notes_dict_test[s] for s in test_stay_ids]

    # 8. Targets: Multi-horizon (6h, 12h, 24h, 48h)
    target_cols = ["mortality_6h", "mortality_12h", "mortality_24h", "mortality_48h"]
    y_train = df_train[target_cols].values
    y_val = df_val[target_cols].values
    y_test = df_test[target_cols].values

    # 9. Save Processed Artifacts
    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "mask_train.npy", mask_train)
    np.save(proc_dir / "y_train.npy", y_train)

    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "mask_val.npy", mask_val)
    np.save(proc_dir / "y_val.npy", y_val)

    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "mask_test.npy", mask_test)
    np.save(proc_dir / "y_test.npy", y_test)

    df_tab_train.to_parquet(proc_dir / "df_tab_train.parquet", index=False)
    df_tab_val.to_parquet(proc_dir / "df_tab_val.parquet", index=False)
    df_tab_test.to_parquet(proc_dir / "df_tab_test.parquet", index=False)

    df_scores_train.to_parquet(proc_dir / "df_scores_train.parquet", index=False)
    df_scores_val.to_parquet(proc_dir / "df_scores_val.parquet", index=False)
    df_scores_test.to_parquet(proc_dir / "df_scores_test.parquet", index=False)

    df_train.to_parquet(proc_dir / "patients_train.parquet", index=False)
    df_val.to_parquet(proc_dir / "patients_val.parquet", index=False)
    df_test.to_parquet(proc_dir / "patients_test.parquet", index=False)

    with open(proc_dir / "texts_train.json", "w", encoding="utf-8") as f:
        json.dump(texts_train, f)
    with open(proc_dir / "texts_val.json", "w", encoding="utf-8") as f:
        json.dump(texts_val, f)
    with open(proc_dir / "texts_test.json", "w", encoding="utf-8") as f:
        json.dump(texts_test, f)

    joblib.dump(scaler, proc_dir / "temporal_scaler.joblib")
    joblib.dump(ts_preprocessor, proc_dir / "ts_preprocessor.joblib")

    logger.info(
        f"Data preparation complete! Artifacts saved to {proc_dir.resolve()}. "
        f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )


if __name__ == "__main__":
    main()
