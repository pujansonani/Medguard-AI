"""
MEDGUARD AI: Automated Data Leakage Verification Test Suite

Explicitly validates zero-leakage research principles:
1. test_patient_overlap(): Confirms disjoint sets of subject_ids between train, validation, and test splits.
2. test_temporal_leakage(): Ensures no physiological observations beyond the 24-hour observation window enter the feature pipeline.
3. test_future_notes_excluded(): Verifies that clinical notes recorded after the 24h cutoff or discharge summaries are strictly excluded.
4. test_preprocessing_fit_only_train(): Verifies scalers/imputers are strictly fitted on training splits with no test/val data leaking into parameters.
"""

import pytest
import numpy as np
import pandas as pd
from medguard.data.cohort import split_cohort_patient_level
from medguard.preprocessing.scaling import TemporalStandardScaler
from medguard.preprocessing.time_series import TimeSeriesPreprocessor
from medguard.preprocessing.text_cleaner import ClinicalTextPreprocessor


def test_patient_overlap():
    """Verify strictly zero patient overlap between train, val, and test splits."""
    # Create synthetic dataframe where patients have multiple rows
    np.random.seed(42)
    subjects = [f"P_{i:04d}" for i in range(100)]
    records = []
    for s in subjects:
        # Each patient has 1-3 stays
        n_stays = np.random.randint(1, 4)
        for stay_idx in range(n_stays):
            records.append({
                "subject_id": s,
                "stay_id": f"{s}_stay_{stay_idx}",
                "mortality_48h": np.random.choice([0, 1], p=[0.8, 0.2]),
                "age": np.random.randint(20, 85),
                "los_hours": np.random.uniform(24, 120),
            })
    df = pd.DataFrame(records)

    train_df, val_df, test_df = split_cohort_patient_level(
        df,
        patient_col="subject_id",
        stratify_col="mortality_48h",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    train_pts = set(train_df["subject_id"].unique())
    val_pts = set(val_df["subject_id"].unique())
    test_pts = set(test_df["subject_id"].unique())

    # Check pairwise disjoint
    assert len(train_pts.intersection(val_pts)) == 0, "Leakage detected: Patient overlap between Train and Val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Leakage detected: Patient overlap between Train and Test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Leakage detected: Patient overlap between Val and Test!"
    
    # Check completeness
    assert len(train_pts) + len(val_pts) + len(test_pts) == len(set(subjects))


def test_temporal_leakage():
    """Verify that any observations with hour > 24 are filtered out and cannot influence features."""
    raw_timeseries = pd.DataFrame({
        "stay_id": [101] * 30,
        "hour": list(range(30)), # Hours 0 to 29 (Hours 24-29 are future/post-window)
        "heart_rate": [70 + i * 2 for i in range(30)],
        "map": [80 - i for i in range(30)],
        "spo2": [98.0] * 30,
    })

    preprocessor = TimeSeriesPreprocessor(
        obs_window_hours=24,
        time_step_hours=1.0,
        vital_cols=["heart_rate", "map", "spo2"]
    )
    
    # Preprocess must only contain 24 steps
    tensor, mask = preprocessor.transform_single_stay(raw_timeseries)
    assert tensor.shape[0] == 24, f"Expected 24 time steps, got {tensor.shape[0]}"
    assert mask.shape[0] == 24, f"Expected 24 mask steps, got {mask.shape[0]}"
    
    # Verify values correspond only to hours 0-23
    assert np.isclose(tensor[0, 0], 70.0) # Hour 0 HR
    assert np.isclose(tensor[23, 0], 70.0 + 23 * 2) # Hour 23 HR


def test_future_notes_excluded():
    """Verify that clinical notes recorded after 24h or categorized as discharge summaries are rejected."""
    notes_df = pd.DataFrame({
        "stay_id": [201, 201, 201, 201],
        "charttime_hours": [4.0, 18.0, 26.5, 52.0], # Hours 26.5 and 52.0 are post-window
        "note_category": ["Nursing", "Physician", "Nursing", "Discharge Summary"],
        "text": [
            "Early admission nursing note. Patient alert.",
            "Physician 18h progress report. Vitals stable.",
            "Post-window note: Patient arrested at hour 26.5.", # Future leakage!
            "Discharge Summary: Patient survived to discharge." # Retrospective leakage!
        ]
    })

    text_preprocessor = ClinicalTextPreprocessor(
        obs_window_hours=24.0,
        exclude_discharge_summaries=True
    )
    
    valid_notes = text_preprocessor.filter_notes_for_stay(notes_df, stay_id=201)
    
    # Should only retain notes with charttime_hours <= 24.0 and category != Discharge Summary
    assert len(valid_notes) == 2, f"Expected 2 notes within 24h window, got {len(valid_notes)}"
    assert all(valid_notes["charttime_hours"] <= 24.0), "Future note leaked into training!"
    assert all(valid_notes["note_category"] != "Discharge Summary"), "Discharge summary leaked into training!"


def test_preprocessing_fit_only_train():
    """Verify that temporal scalers fit exclusively on train set and don't leak test distribution statistics."""
    np.random.seed(42)
    # Train set centered at 100, Test set shifted to 200
    train_data = np.random.normal(loc=100.0, scale=15.0, size=(50, 24, 3))
    test_data = np.random.normal(loc=200.0, scale=15.0, size=(20, 24, 3))

    scaler = TemporalStandardScaler()
    scaler.fit(train_data)

    # Scaler mean must match train mean (~100.0), NOT combined mean (~128.5)
    train_mean = np.mean(train_data, axis=(0, 1))
    assert np.allclose(scaler.means_, train_mean, atol=1e-5), "Scaler means do not match train mean!"
    assert not np.allclose(scaler.means_, np.mean(np.concatenate([train_data, test_data], axis=0), axis=(0, 1)))

    # Transform test data using strictly train parameters
    scaled_test = scaler.transform(test_data)
    # Test data should have mean ~ (200 - 100)/15 = ~6.67, not 0.0
    assert np.mean(scaled_test) > 5.0, "Test data was normalized with its own statistics (leakage)!"
