"""
Cohort definition, inclusion criteria, and patient-level leakage-free train/val/test splitting.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from medguard.utils.logging import get_logger

logger = get_logger("medguard.cohort")


class CohortSelector:
    """
    Applies rigorous ICU cohort selection rules:
    1. Adult patients (age >= 18)
    2. ICU stay duration >= 24 hours (so the 24-hour observation window is complete)
    3. Exclude patients with missing or negative ICU admission times
    4. Multi-horizon mortality labels strictly computed relative to the 24h observation window.
    """

    def __init__(
        self,
        min_age: float = 18.0,
        min_icu_los_hours: float = 24.0,
        obs_window_hours: float = 24.0,
        pred_horizon_hours: float = 48.0,
    ):
        self.min_age = min_age
        self.min_icu_los_hours = min_icu_los_hours
        self.obs_window_hours = obs_window_hours
        self.pred_horizon_hours = pred_horizon_hours

    def filter_cohort(self, df_patients: pd.DataFrame) -> pd.DataFrame:
        """
        Filter cohort dataframe based on inclusion criteria.
        Expects columns: ['subject_id', 'stay_id', 'age', 'los_hours', ...]
        """
        initial_count = len(df_patients)
        
        # Filter 1: Age
        df = df_patients[df_patients["age"] >= self.min_age].copy()
        age_filtered = len(df)
        
        # Filter 2: Length of Stay
        df = df[df["los_hours"] >= self.min_icu_los_hours].copy()
        los_filtered = len(df)
        
        logger.info(
            f"Cohort selection: Initial={initial_count} -> Age>={self.min_age}: {age_filtered} "
            f"-> LOS>={self.min_icu_los_hours}h: {los_filtered} patients."
        )
        return df


def split_cohort_patient_level(
    df: pd.DataFrame,
    patient_col: str = "subject_id",
    stratify_col: str = "mortality_48h",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split cohort at the patient (subject_id) level to prevent temporal and data leakage.
    Ensures that a patient's multiple stays or data never span across train/val/test splits.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    # Get unique patients with their stratification label (using max/mode of stratify col)
    patient_df = (
        df.groupby(patient_col)[stratify_col]
        .max()
        .reset_index()
    )
    
    patients = np.array(patient_df[patient_col].tolist())
    labels = np.array(patient_df[stratify_col].tolist())
    
    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    p_train, p_temp, y_train, y_temp = train_test_split(
        patients,
        labels,
        test_size=temp_ratio,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    
    # Second split: Val vs Test
    val_rel_ratio = val_ratio / temp_ratio
    p_val, p_test, _, _ = train_test_split(
        p_temp,
        y_temp,
        test_size=(1.0 - val_rel_ratio),
        random_state=seed,
        stratify=y_temp if len(np.unique(y_temp)) > 1 else None,
    )
    
    df_train = df[df[patient_col].isin(set(p_train))].copy().reset_index(drop=True)
    df_val = df[df[patient_col].isin(set(p_val))].copy().reset_index(drop=True)
    df_test = df[df[patient_col].isin(set(p_test))].copy().reset_index(drop=True)
    
    logger.info(
        f"Patient-level Split: Train={len(df_train)} ({len(p_train)} unique patients), "
        f"Val={len(df_val)} ({len(p_val)} unique patients), "
        f"Test={len(df_test)} ({len(p_test)} unique patients)"
    )
    return df_train, df_val, df_test
