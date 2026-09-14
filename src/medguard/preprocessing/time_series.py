"""
Temporal preprocessing, hourly discretization, forward-fill imputation, and missingness masking.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from medguard.utils.logging import get_logger

logger = get_logger("medguard.preprocessing.ts")

FEATURE_COLUMNS = [
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "temperature", "spo2", "gcs",
    "glucose", "creatinine", "hemoglobin", "wbc", "platelets", "sodium", "potassium",
    "bicarbonate", "lactate", "bun"
]


class TimeSeriesPreprocessor:
    """
    Transforms tabular observation logs into fixed-length 3D temporal tensors (N_patients, Seq_len, N_features).
    Generates accompanying binary missingness masks: mask[t, f] = 1 if observed, 0 if missing.
    Strictly enforces the observation window cutoff (e.g. 24 hours).
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 24,
        obs_window_hours: float = 24.0,
        time_step_hours: float = 1.0,
        vital_cols: Optional[List[str]] = None,
        default_fill_values: Optional[Dict[str, float]] = None,
    ):
        self.feature_cols = feature_cols or vital_cols or FEATURE_COLUMNS
        self.seq_len = int(obs_window_hours / time_step_hours) if obs_window_hours else seq_len
        self.obs_window_hours = obs_window_hours
        self.time_step_hours = time_step_hours
        self.default_fill_values = default_fill_values or {
            "heart_rate": 80.0, "sbp": 120.0, "dbp": 75.0, "map": 85.0,
            "resp_rate": 16.0, "temperature": 37.0, "spo2": 98.0, "gcs": 15.0,
            "glucose": 100.0, "creatinine": 0.9, "hemoglobin": 13.5, "wbc": 7.5,
            "platelets": 220.0, "sodium": 140.0, "potassium": 4.0,
            "bicarbonate": 24.0, "lactate": 1.2, "bun": 15.0
        }
        self.medians_: Dict[str, float] = {}

    def fit(self, df_timeseries: pd.DataFrame) -> "TimeSeriesPreprocessor":
        """Compute training set medians for initial feature imputation (fitted ONLY on training fold)."""
        for col in self.feature_cols:
            if col in df_timeseries.columns:
                valid_vals = df_timeseries[col].dropna()
                self.medians_[col] = float(valid_vals.median()) if len(valid_vals) > 0 else self.default_fill_values.get(col, 0.0)
            else:
                self.medians_[col] = self.default_fill_values.get(col, 0.0)
        return self

    def transform_single_stay(self, df_stay: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single ICU stay into (seq_len, n_features) and (seq_len, n_features) mask.
        Strictly filters out observations beyond obs_window_hours.
        """
        # Determine hour column
        hr_col = "hour" if "hour" in df_stay.columns else ("hour_bin" if "hour_bin" in df_stay.columns else None)
        
        if hr_col is not None:
            # Filter strictly within observation window
            valid_df = df_stay[(df_stay[hr_col] >= 0) & (df_stay[hr_col] < self.obs_window_hours)].copy()
        else:
            valid_df = df_stay.iloc[:self.seq_len].copy()

        n_feats = len(self.feature_cols)
        tensor = np.zeros((self.seq_len, n_feats), dtype=np.float32)
        mask = np.zeros((self.seq_len, n_feats), dtype=np.float32)

        for col_idx, col in enumerate(self.feature_cols):
            default_val = self.medians_.get(col, self.default_fill_values.get(col, 0.0))
            
            # Map observations to hourly bins
            observed_series = np.full(self.seq_len, np.nan, dtype=np.float32)
            if col in valid_df.columns:
                if hr_col is not None:
                    for _, row in valid_df.iterrows():
                        h = int(row[hr_col])
                        if 0 <= h < self.seq_len and not np.isnan(row[col]):
                            observed_series[h] = float(row[col])
                else:
                    for idx, val in enumerate(valid_df[col].values[:self.seq_len]):
                        if not np.isnan(val):
                            observed_series[idx] = float(val)

            # Record mask where observation was directly measured
            obs_mask = ~np.isnan(observed_series)
            mask[:, col_idx] = obs_mask.astype(np.float32)

            # Forward fill with fallback to population median
            filled = observed_series.copy()
            last_val = default_val
            for t in range(self.seq_len):
                if not np.isnan(filled[t]):
                    last_val = filled[t]
                else:
                    filled[t] = last_val
            tensor[:, col_idx] = filled

        return tensor, mask

    def transform(
        self,
        df_timeseries: pd.DataFrame,
        stay_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time-series dataframe into 3D tensors:
        X: (N_stays, seq_len, N_features)
        mask: (N_stays, seq_len, N_features)
        """
        N = len(stay_ids)
        T = self.seq_len
        D = len(self.feature_cols)

        X = np.zeros((N, T, D), dtype=np.float32)
        M = np.zeros((N, T, D), dtype=np.float32)

        grouped = df_timeseries.groupby("stay_id")

        for i, sid in enumerate(stay_ids):
            if sid in grouped.groups:
                stay_df = grouped.get_group(sid)
                t_arr, m_arr = self.transform_single_stay(stay_df)
                X[i] = t_arr
                M[i] = m_arr
            else:
                for d, col in enumerate(self.feature_cols):
                    default_v = self.medians_.get(col, self.default_fill_values.get(col, 0.0))
                    X[i, :, d] = default_v
                    M[i, :, d] = 0.0

        return X, M
