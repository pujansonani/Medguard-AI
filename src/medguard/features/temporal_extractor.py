"""
Extracts statistical aggregations and dynamic trajectory features from 24h ICU time-series.
Used by tabular machine learning models (Logistic Regression, XGBoost, LightGBM).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from medguard.preprocessing.time_series import FEATURE_COLUMNS


class TabularFeatureExtractor:
    """
    Transforms 3D temporal arrays (N, T, D) and static demographics (N, S)
    into a rich tabular representation:
    - Min, Max, Mean, Std, First, Last, Delta (Last - First), Linear Trend
    - Missingness rate per variable
    - Static demographics (One-hot encoded)
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.column_names_: List[str] = []

    def transform(
        self,
        X_tensor: np.ndarray,
        mask_tensor: np.ndarray,
        df_demographics: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Args:
            X_tensor: (N, T, D) float array of vitals/labs across 24 hours.
            mask_tensor: (N, T, D) binary mask (1=observed, 0=imputed).
            df_demographics: DataFrame containing ['age', 'gender', 'icu_type', 'admission_type'].

        Returns:
            pd.DataFrame of shape (N, Total_Engineered_Features).
        """
        N, T, D = X_tensor.shape
        records = []
        time_steps = np.arange(T)

        for i in range(N):
            row_dict: Dict[str, float] = {}

            # Static demographic features
            demo_row = df_demographics.iloc[i]
            row_dict["age"] = float(demo_row.get("age", 60.0))
            row_dict["is_male"] = 1.0 if str(demo_row.get("gender", "M")).upper() == "M" else 0.0
            
            # ICU type one-hot
            icu = str(demo_row.get("icu_type", "MICU")).upper()
            row_dict["icu_micu"] = 1.0 if "MICU" in icu else 0.0
            row_dict["icu_sicu"] = 1.0 if "SICU" in icu else 0.0
            row_dict["icu_ccu"] = 1.0 if "CCU" in icu or "CVICU" in icu else 0.0
            row_dict["icu_other"] = 1.0 if not (row_dict["icu_micu"] or row_dict["icu_sicu"] or row_dict["icu_ccu"]) else 0.0
            
            # Admission type
            adm = str(demo_row.get("admission_type", "EMERGENCY")).upper()
            row_dict["is_emergency"] = 1.0 if "EMERGENCY" in adm or "URGENT" in adm else 0.0

            # Dynamic time-series statistics
            for d, feat in enumerate(self.feature_names):
                series = X_tensor[i, :, d]
                obs_mask = mask_tensor[i, :, d]

                row_dict[f"{feat}_mean"] = float(np.mean(series))
                row_dict[f"{feat}_min"] = float(np.min(series))
                row_dict[f"{feat}_max"] = float(np.max(series))
                row_dict[f"{feat}_std"] = float(np.std(series))
                row_dict[f"{feat}_first"] = float(series[0])
                row_dict[f"{feat}_last"] = float(series[-1])
                row_dict[f"{feat}_delta"] = float(series[-1] - series[0])
                
                # Linear trend (slope over time)
                if np.std(series) > 1e-4:
                    slope, _ = np.polyfit(time_steps, series, 1)
                    row_dict[f"{feat}_trend"] = float(slope)
                else:
                    row_dict[f"{feat}_trend"] = 0.0

                # Missingness fraction (signals data availability / testing frequency)
                row_dict[f"{feat}_missing_pct"] = float(1.0 - np.mean(obs_mask))

            records.append(row_dict)

        df_out = pd.DataFrame(records)
        self.column_names_ = list(df_out.columns)
        return df_out
