"""
Tests for Data Loaders, Cohort Selection, and Patient-Level Splitting.
"""

import numpy as np
import pandas as pd
import pytest

from medguard.data.cohort import CohortSelector, split_cohort_patient_level
from medguard.data.synthetic_generator import SyntheticDataGenerator


def test_cohort_selector_filters():
    df_raw = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "stay_id": [10, 20, 30, 40],
        "age": [16.0, 45.0, 72.0, 60.0], # Patient 1 is pediatric (< 18)
        "los_hours": [36.0, 12.0, 48.0, 72.0], # Patient 2 has LOS < 24h
        "mortality_48h": [0, 0, 1, 0],
    })

    selector = CohortSelector(min_age=18.0, min_icu_los_hours=24.0)
    df_filtered = selector.filter_cohort(df_raw)

    assert len(df_filtered) == 2
    assert set(df_filtered["subject_id"]) == {3, 4}


def test_patient_level_splitting_no_leakage():
    # Multiple stays for some patients
    df = pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "stay_id": list(range(100, 111)),
        "mortality_48h": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    })

    df_train, df_val, df_test = split_cohort_patient_level(
        df, patient_col="subject_id", stratify_col="mortality_48h",
        train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=42
    )

    train_pts = set(df_train["subject_id"])
    val_pts = set(df_val["subject_id"])
    test_pts = set(df_test["subject_id"])

    # Assert strict zero patient overlap
    assert len(train_pts.intersection(val_pts)) == 0, "Leakage between train and val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Leakage between train and test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Leakage between val and test!"


def test_synthetic_data_generator():
    gen = SyntheticDataGenerator(seed=42)
    df_patients, df_ts, df_notes = gen.generate_cohort(num_patients=20, obs_window_hours=24)

    assert len(df_patients) == 20
    assert len(df_ts) == 20 * 24 # 24 hourly rows per patient
    assert len(df_notes) > 0

    # Test multi-horizon hierarchy: m6 <= m12 <= m24 <= m48
    for _, row in df_patients.iterrows():
        assert row["mortality_6h"] <= row["mortality_12h"]
        assert row["mortality_12h"] <= row["mortality_24h"]
        assert row["mortality_24h"] <= row["mortality_48h"]

    # Test notes strictly <= 24h
    assert (df_notes["chart_time_hour"] <= 24.0).all()
