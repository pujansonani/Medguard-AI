"""
Data & Model Monitoring: Feature drift, prediction drift, and missingness rate monitoring.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between reference baseline and current deployment batch.
    PSI < 0.1: No significant shift
    0.1 <= PSI < 0.2: Moderate shift
    PSI >= 0.2: Significant drift detected
    """
    ref = np.array(reference).ravel()
    cur = np.array(current).ravel()
    
    # Remove NaNs
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts + 1e-4) / (len(ref) + 1e-4 * num_bins)
    cur_pct = (cur_counts + 1e-4) / (len(cur) + 1e-4 * num_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


class DriftMonitor:
    """Monitors physiological feature distribution drift and prediction confidence shifts."""

    def __init__(self, reference_df: pd.DataFrame, reference_predictions: np.ndarray):
        self.ref_df = reference_df
        self.ref_preds = reference_predictions

    def monitor_batch(
        self,
        current_df: pd.DataFrame,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate Kolmogorov-Smirnov and PSI metrics across numeric features and output probabilities.
        """
        feature_drift = {}
        numeric_cols = self.ref_df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in current_df.columns:
                ref_vals = self.ref_df[col].dropna().values
                cur_vals = current_df[col].dropna().values

                if len(ref_vals) > 5 and len(cur_vals) > 5:
                    ks_stat, p_val = ks_2samp(ref_vals, cur_vals)
                    psi = compute_psi(ref_vals, cur_vals)
                    feature_drift[col] = {
                        "psi": round(psi, 4),
                        "ks_stat": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 4),
                        "status": "Drift Alert" if psi >= 0.2 or p_val < 0.01 else ("Moderate Shift" if psi >= 0.1 else "Stable"),
                    }

        pred_psi = compute_psi(self.ref_preds, current_predictions)
        pred_ks, pred_pval = ks_2samp(self.ref_preds, current_predictions)

        return {
            "prediction_drift": {
                "psi": round(pred_psi, 4),
                "ks_stat": round(float(pred_ks), 4),
                "p_value": round(float(pred_pval), 4),
                "status": "Drift Alert" if pred_psi >= 0.2 else ("Moderate Shift" if pred_psi >= 0.1 else "Stable"),
            },
            "feature_drift": feature_drift,
        }
