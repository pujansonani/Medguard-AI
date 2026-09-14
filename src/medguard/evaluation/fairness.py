"""
Subgroup Fairness and Disparity Audit across Demographics and Care Units.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from medguard.evaluation.metrics import compute_all_metrics


class FairnessAuditor:
    """
    Evaluates model performance disparities across:
    - Age groups (< 65 vs >= 65)
    - Gender (M vs F)
    - ICU types (MICU, SICU, CCU, CVICU, etc.)
    """

    @staticmethod
    def audit_subgroups(
        df_patients: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute performance metrics for each demographic and care unit subgroup.
        """
        df = df_patients.copy().reset_index(drop=True)
        df["y_true"] = np.array(y_true).ravel()
        df["y_prob"] = np.array(y_prob).ravel()
        df["y_pred"] = (df["y_prob"] >= threshold).astype(int)

        # Categorize Age into clinical strata
        df["age_group"] = np.where(df["age"] >= 65, "Age >= 65", "Age < 65")

        audit_results = {}

        # 1. Age Audit
        audit_results["age_group"] = {}
        for grp, grp_df in df.groupby("age_group"):
            audit_results["age_group"][str(grp)] = {
                "count": len(grp_df),
                "mortality_rate": round(float(grp_df["y_true"].mean()), 3),
                "metrics": compute_all_metrics(grp_df["y_true"].values, grp_df["y_prob"].values, threshold),
            }

        # 2. Gender Audit
        audit_results["gender"] = {}
        for grp, grp_df in df.groupby("gender"):
            audit_results["gender"][str(grp)] = {
                "count": len(grp_df),
                "mortality_rate": round(float(grp_df["y_true"].mean()), 3),
                "metrics": compute_all_metrics(grp_df["y_true"].values, grp_df["y_prob"].values, threshold),
            }

        # 3. ICU Care Unit Audit
        audit_results["icu_type"] = {}
        for grp, grp_df in df.groupby("icu_type"):
            if len(grp_df) >= 5: # Only report if subgroup has minimal samples
                audit_results["icu_type"][str(grp)] = {
                    "count": len(grp_df),
                    "mortality_rate": round(float(grp_df["y_true"].mean()), 3),
                    "metrics": compute_all_metrics(grp_df["y_true"].values, grp_df["y_prob"].values, threshold),
                }

        return audit_results
