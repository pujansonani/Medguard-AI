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
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 24,
        default_fill_values: Optional[Dict[str, float]] = None,
    ):
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.seq_len = seq_len
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

    def transform(
        self,
        df_timeseries: pd.DataFrame,
        stay_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time-series dataframe into (X_tensor, mask_tensor).

        Returns:
            X: np.ndarray of shape (N_stays, seq_len, N_features)
            mask: np.ndarray of shape (N_stays, seq_len, N_features) - 1.0 where observed, 0.0 where imputed
        """
        num_stays = len(stay_ids)
        num_feats = len(self.feature_cols)

        X = np.zeros((num_stays, self.seq_len, num_feats), dtype=np.float32)
        mask = np.zeros((num_stays, self.seq_len, num_feats), dtype=np.float32)

        # Index dataframe for rapid lookup
        grouped = df_timeseries.groupby("stay_id")

        for i, stay_id in enumerate(stay_ids):
            if stay_id not in grouped.groups:
                # Fill with global training medians if completely missing
                for f_idx, col in enumerate(self.feature_cols):
                    X[i, :, f_idx] = self.medians_.get(col, 0.0)
                continue

            stay_data = grouped.get_group(stay_id).sort_values("hour")
            
            # Reindex to full sequence 0..(seq_len-1)
            full_idx = pd.DataFrame({"hour": np.arange(self.seq_len)})
            merged_stay = full_idx.merge(stay_data, on="hour", how="left")

            for f_idx, col in enumerate(self.feature_cols):
                if col not in merged_stay.columns:
                    col_vals = pd.Series([np.nan] * self.seq_len)
                else:
                    col_vals = merged_stay[col]

                # Binary observation mask: 1 if observed in raw data, 0 if missing
                col_mask = (~col_vals.isna()).astype(float).values
                mask[i, :, f_idx] = col_mask

                # Forward-fill, then backward-fill, then fill with training fold median
                imputed_series = col_vals.ffill().bfill().fillna(self.medians_.get(col, 0.0))
                X[i, :, f_idx] = imputed_series.values[:self.seq_len]

        return X, mask
