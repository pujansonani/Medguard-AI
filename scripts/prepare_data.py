"""
Data Preparation & Feature Engineering Pipeline:
Loads raw MIMIC-IV or synthetic cohort, performs patient-level splitting, extracts 3D temporal tensors,
tabular summary features, and tokenized clinical note texts.
Saves cohort.csv, splits/ IDs, and cohort_report.json.
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
from medguard.utils.config import get_project_root, load_all_configs, load_yaml

logger = get_logger("medguard.scripts.prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and features for MEDGUARD AI.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to main config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--mimic-dir", type=str, default=None, help="Path to raw MIMIC-IV directory")
    parser.add_argument("--notes-dir", type=str, default=None, help="Path to raw MIMIC-IV-Note directory")
    parser.add_argument("--force_synthetic", action="store_true", help="Force using synthetic data")
    parser.add_argument("--max_patients", type=int, default=None, help="Limit cohort size for quick testing")
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]
    data_cfg = load_yaml(root / args.data_config) if (root / args.data_config).exists() else {}

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = proc_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mimic_path = Path(args.mimic_dir) if args.mimic_dir else root / data_cfg.get("data", {}).get("mimic_iv_path", "./data/raw/mimiciv")
    notes_path = Path(args.notes_dir) if args.notes_dir else root / data_cfg.get("data", {}).get("mimic_iv_note_path", "./data/raw/mimiciv_note")

    mimic_loader = MIMICDataLoader(
        raw_mimic_dir=mimic_path,
        raw_notes_dir=notes_path,
        config=data_cfg,
        obs_window_hours=cfg.get("cohort", {}).get("observation_window_hours", 24.0),
        pred_horizon_hours=cfg.get("cohort", {}).get("prediction_horizon_hours", 48.0),
    )

    is_mimic_mode = not args.force_synthetic and mimic_loader.check_availability()
    
    if is_mimic_mode:
        logger.info(f"MIMIC-IV raw dataset detected at {mimic_path}. Ingesting real MIMIC-IV cohort...")
        cohort_cfg = cfg.get("cohort", {})
        df_cohort, cohort_report = mimic_loader.load_cohort(
            min_age=cohort_cfg.get("min_age", 18.0),
            min_los_hours=cohort_cfg.get("min_icu_los_hours", 24.0),
            max_patients=args.max_patients,
        )
        # Extract observations and notes from MIMIC-IV
        df_timeseries = mimic_loader.extract_structured_timeseries(df_cohort)
        df_notes = mimic_loader.extract_clinical_notes(df_cohort)
    else:
        logger.info("Using DEMO / Synthetic Cohort Pipeline.")
        synth_dir = root / cfg.get("paths", {}).get("synthetic_dir", "data/synthetic")
        synth_dir.mkdir(parents=True, exist_ok=True)

        if (synth_dir / "patients.parquet").exists() and not args.force_synthetic:
            logger.info("Loading pre-generated synthetic files...")
            df_patients = pd.read_parquet(synth_dir / "patients.parquet")
            df_timeseries = pd.read_parquet(synth_dir / "timeseries.parquet")
            df_notes = pd.read_parquet(synth_dir / "notes.parquet")
        else:
            logger.info("Generating high-fidelity synthetic clinical files...")
            generator = SyntheticDataGenerator(seed=cfg.get("seed", 42))
            num_pts = args.max_patients or cfg.get("synthetic", {}).get("num_patients", 600)
            df_patients, df_timeseries, df_notes = generator.generate_cohort(num_patients=num_pts)
            df_patients.to_parquet(synth_dir / "patients.parquet", index=False)
            df_timeseries.to_parquet(synth_dir / "timeseries.parquet", index=False)
            df_notes.to_parquet(synth_dir / "notes.parquet", index=False)

        cohort_cfg = cfg.get("cohort", {})
        selector = CohortSelector(
            min_age=cohort_cfg.get("min_age", 18.0),
            min_icu_los_hours=cohort_cfg.get("min_icu_los_hours", 24.0),
        )
        df_cohort = selector.filter_cohort(df_patients)
        cohort_report = {
            "initial_icu_stays": int(len(df_patients)),
            "final_cohort_stays": int(len(df_cohort)),
            "mortality_count": int(df_cohort["mortality_48h"].sum()),
            "survival_count": int(len(df_cohort) - df_cohort["mortality_48h"].sum()),
            "mortality_rate": round(float(df_cohort["mortality_48h"].mean()), 4),
            "age_mean": round(float(df_cohort["age"].mean()), 2),
            "age_std": round(float(df_cohort["age"].std()), 2),
            "icu_type_distribution": df_cohort["icu_type"].value_counts().to_dict(),
        }

    # Save Cohort CSV and Report
    df_cohort.to_csv(proc_dir / "cohort.csv", index=False)
    with open(reports_dir / "cohort_report.json", "w") as f:
        json.dump(cohort_report, f, indent=2)
    logger.info(f"Saved cohort.csv and cohort_report.json ({len(df_cohort):,} stays).")

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

    # Save Persistent Split IDs
    df_train[["subject_id", "stay_id", "mortality_48h"]].to_csv(splits_dir / "train_ids.csv", index=False)
    df_val[["subject_id", "stay_id", "mortality_48h"]].to_csv(splits_dir / "val_ids.csv", index=False)
    df_test[["subject_id", "stay_id", "mortality_48h"]].to_csv(splits_dir / "test_ids.csv", index=False)
    logger.info(f"Saved persistent split IDs: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 3. Temporal 3D Tensor Extraction
    ts_preprocessor = TimeSeriesPreprocessor(
        obs_window_hours=cfg.get("cohort", {}).get("observation_window_hours", 24.0),
        time_step_hours=cfg.get("cohort", {}).get("time_step_hours", 1.0),
    )
    
    # Fit strictly on training fold
    train_stay_ids = df_train["stay_id"].tolist()
    val_stay_ids = df_val["stay_id"].tolist()
    test_stay_ids = df_test["stay_id"].tolist()

    train_ts_df = df_timeseries[df_timeseries["stay_id"].isin(train_stay_ids)]
    ts_preprocessor.fit(train_ts_df)

    X_train_raw, mask_train = ts_preprocessor.transform(df_timeseries, train_stay_ids)
    X_val_raw, mask_val = ts_preprocessor.transform(df_timeseries, val_stay_ids)
    X_test_raw, mask_test = ts_preprocessor.transform(df_timeseries, test_stay_ids)

    # 4. Standardize Tensors (Fit Scaler Strictly on Train)
    scaler = TemporalStandardScaler()
    scaler.fit(X_train_raw, mask=mask_train)

    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Extract Tabular Summary & Volatility Features
    tab_extractor = TabularFeatureExtractor()
    df_tab_train = tab_extractor.transform(X_train_raw, mask_train, df_train)
    df_tab_val = tab_extractor.transform(X_val_raw, mask_val, df_val)
    df_tab_test = tab_extractor.transform(X_test_raw, mask_test, df_test)

    # 6. Calculate Clinical Baseline Scores (SOFA, SAPS II, APACHE II)
    score_calc = ClinicalScoreCalculator()
    df_scores_train = score_calc.calculate_scores_df(df_tab_train)
    df_scores_val = score_calc.calculate_scores_df(df_tab_val)
    df_scores_test = score_calc.calculate_scores_df(df_tab_test)

    # 7. Clean and Aggregate Unstructured Clinical Notes
    text_cleaner = ClinicalTextPreprocessor(
        obs_window_hours=cfg.get("cohort", {}).get("observation_window_hours", 24.0),
        exclude_discharge_summaries=True,
    )
    texts_train_dict = text_cleaner.aggregate_patient_notes(df_notes, train_stay_ids)
    texts_val_dict = text_cleaner.aggregate_patient_notes(df_notes, val_stay_ids)
    texts_test_dict = text_cleaner.aggregate_patient_notes(df_notes, test_stay_ids)

    texts_train = [texts_train_dict[sid] for sid in train_stay_ids]
    texts_val = [texts_val_dict[sid] for sid in val_stay_ids]
    texts_test = [texts_test_dict[sid] for sid in test_stay_ids]

    # Target multi-horizon arrays (6h, 12h, 24h, 48h)
    label_cols = ["mortality_6h", "mortality_12h", "mortality_24h", "mortality_48h"]
    y_train = df_train[label_cols].values
    y_val = df_val[label_cols].values
    y_test = df_test[label_cols].values

    # 8. Save Processed Artifacts
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

    with open(proc_dir / "texts_train.json", "w") as f:
        json.dump(texts_train, f)
    with open(proc_dir / "texts_val.json", "w") as f:
        json.dump(texts_val, f)
    with open(proc_dir / "texts_test.json", "w") as f:
        json.dump(texts_test, f)

    joblib.dump(ts_preprocessor, proc_dir / "ts_preprocessor.joblib")
    joblib.dump(scaler, proc_dir / "temporal_scaler.joblib")

    logger.info(
        f"Data preparation complete! Artifacts saved to {proc_dir.resolve()}. "
        f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
    )


if __name__ == "__main__":
    main()
