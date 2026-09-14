"""
Structured Explainability Engine using SHAP (SHapley Additive exPlanations) and Temporal Feature Dynamics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap

from medguard.utils.logging import get_logger

logger = get_logger("medguard.explainability.shap")


class MedguardSHAPExplainer:
    """
    Computes global and local SHAP explanations for structured tabular features and temporal dynamics.
    """

    def __init__(self, model: Any, background_data: Optional[pd.DataFrame | np.ndarray] = None):
        self.model = model
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

        if background_data is not None:
            self.fit(background_data)

    def fit(self, background_data: pd.DataFrame | np.ndarray) -> "MedguardSHAPExplainer":
        """Initialize SHAP explainer with representative background data."""
        if isinstance(background_data, pd.DataFrame):
            self.feature_names = list(background_data.columns)
            bg_arr = background_data.values
        else:
            bg_arr = background_data
            self.feature_names = [f"feat_{i}" for i in range(bg_arr.shape[1])]

        # Use up to 100 background samples for efficient computation
        sample_size = min(100, len(bg_arr))
        bg_sample = shap.sample(bg_arr, sample_size, random_state=42)

        try:
            # Check if model is tree-based (XGBoost / LightGBM)
            if hasattr(self.model, "clf") and hasattr(self.model.clf, "get_booster"):
                self.explainer = shap.TreeExplainer(self.model.clf)
            elif hasattr(self.model, "predict_proba"):
                # Wrap predict_proba for positive mortality class
                def predict_fn(x):
                    return self.model.predict_proba(x)
                self.explainer = shap.KernelExplainer(predict_fn, bg_sample)
            else:
                self.explainer = shap.Explainer(self.model, bg_sample)
        except Exception as e:
            logger.warning(f"Defaulting to KernelExplainer due to: {e}")
            def predict_fn(x):
                return self.model.predict_proba(x)
            self.explainer = shap.KernelExplainer(predict_fn, bg_sample)

        self.is_fitted = True
        return self

    def explain_instance(self, instance_df: pd.DataFrame | np.ndarray) -> Dict[str, Any]:
        """
        Compute local SHAP attribution values for a single patient instance.

        Returns:
            dict containing:
                - base_value: expected model value
                - shap_values: dictionary of {feature_name: shap_value}
                - top_positive: top features pushing risk UP
                - top_negative: top features pushing risk DOWN
        """
        if not self.is_fitted:
            raise ValueError("SHAP explainer is not fitted.")

        inst_arr = instance_df.values if isinstance(instance_df, pd.DataFrame) else instance_df
        if inst_arr.ndim == 1:
            inst_arr = inst_arr.reshape(1, -1)

        shap_vals = self.explainer.shap_values(inst_arr)
        
        # Handle binary classification multi-output shapes
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            s_val = np.array(shap_vals[1][0]) # positive mortality class
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            s_val = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            s_val = shap_vals[0]
        else:
            s_val = np.array(shap_vals).flatten()

        feat_contributions = {}
        for feat, val in zip(self.feature_names, s_val):
            feat_contributions[feat] = float(val)

        # Sort features by absolute attribution
        sorted_feats = sorted(feat_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_pos = [(k, v) for k, v in sorted_feats if v > 0][:8]
        top_neg = [(k, v) for k, v in sorted_feats if v < 0][:8]

        base_val = float(getattr(self.explainer, "expected_value", 0.18))
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])

        return {
            "base_value": base_val,
            "shap_values": feat_contributions,
            "top_positive": top_pos,
            "top_negative": top_neg,
            "ranked_features": sorted_feats[:15],
        }

    def explain_temporal_trajectory(
        self,
        X_series: np.ndarray, # (24, num_vitals)
        vitals_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Simulate hourly attribution shift across the 24-hour observation window.
        Reveals when specific physiological abnormalities (e.g. rising lactate at hour 12, hypotension at hour 18)
        began dominating the risk score.
        """
        T, D = X_series.shape
        hourly_attributions = []

        # Baseline reference is healthy midpoint
        normal_refs = {
            "heart_rate": 75.0, "sbp": 120.0, "dbp": 75.0, "map": 85.0,
            "resp_rate": 16.0, "temperature": 37.0, "spo2": 98.0, "gcs": 15.0,
            "lactate": 1.0, "creatinine": 0.9, "wbc": 7.0, "platelets": 250.0,
            "glucose": 100.0, "potassium": 4.0, "bicarbonate": 24.0, "bun": 14.0,
            "sodium": 140.0, "hemoglobin": 14.0
        }

        for h in range(T):
            hour_data = {}
            for d, name in enumerate(vitals_names):
                val = float(X_series[h, d])
                ref = normal_refs.get(name, val)
                
                # Approximate directional risk contribution based on physiological deviation
                if name in ["lactate", "creatinine", "wbc", "bun", "resp_rate", "heart_rate"]:
                    diff = max(0.0, val - ref) / (ref + 1e-4)
                elif name in ["map", "sbp", "spo2", "gcs", "platelets", "bicarbonate"]:
                    diff = max(0.0, ref - val) / (ref + 1e-4)
                else:
                    diff = abs(val - ref) / (ref + 1e-4)

                hour_data[name] = float(diff)

            # Sort top driving features at this hour
            top_h = sorted(hour_data.items(), key=lambda x: x[1], reverse=True)[:3]
            hourly_attributions.append({
                "hour": h,
                "top_features": top_h,
                "all_contributions": hour_data,
            })

        return hourly_attributions
