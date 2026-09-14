"""
Tests for Preprocessing, Scaling, and Feature Extraction.
"""

import numpy as np
import pandas as pd
import pytest

from medguard.preprocessing.time_series import TimeSeriesPreprocessor, FEATURE_COLUMNS
from medguard.preprocessing.text_cleaner import ClinicalTextPreprocessor
from medguard.preprocessing.scaling import TemporalStandardScaler
from medguard.features.temporal_extractor import TabularFeatureExtractor
from medguard.features.clinical_scores import ClinicalScoreCalculator


def test_time_series_preprocessor_and_mask():
    df_ts = pd.DataFrame({
        "stay_id": [101, 101, 101],
        "hour": [0, 5, 23],
        "heart_rate": [80.0, 95.0, 110.0],
        "map": [85.0, np.nan, 65.0],
    })

    preproc = TimeSeriesPreprocessor(seq_len=24)
    preproc.fit(df_ts)

    X, mask = preproc.transform(df_ts, stay_ids=[101])

    assert X.shape == (1, 24, len(FEATURE_COLUMNS))
    assert mask.shape == (1, 24, len(FEATURE_COLUMNS))

    # Check that heart_rate at hour 0, 5, 23 has mask = 1.0
    hr_idx = FEATURE_COLUMNS.index("heart_rate")
    assert mask[0, 0, hr_idx] == 1.0
    assert mask[0, 5, hr_idx] == 1.0
    assert mask[0, 1, hr_idx] == 0.0 # hour 1 was imputed/forward-filled


def test_clinical_text_cleaning_and_cutoff():
    raw_note = "Patient [**2180-05-12**] admitted with [**Hospital 1**] hypotension. --- ===== ___"
    clean = ClinicalTextPreprocessor.clean_text(raw_note)

    assert "[**" not in clean
    assert "[REDACTED]" in clean
    assert "hypotension" in clean

    # Test cutoff filter
    df_notes = pd.DataFrame({
        "stay_id": [201, 201],
        "chart_time_hour": [12.0, 36.0], # 36h is past 24h window
        "text": ["Early note.", "Late note that must be excluded."],
    })

    cleaner = ClinicalTextPreprocessor()
    res = cleaner.aggregate_patient_notes(df_notes, stay_ids=[201], max_cutoff_hour=24.0)

    assert "Early note" in res[201]
    assert "Late note" not in res[201]


def test_clinical_score_calculator():
    calc = ClinicalScoreCalculator()
    sofa = calc.compute_sofa_proxy(
        map_min=50.0, gcs_min=8.0, platelets_min=45.0, creatinine_max=3.8, spo2_min=85.0
    )
    assert 10 <= sofa <= 24
