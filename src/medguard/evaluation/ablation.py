"""
Ablation Study Orchestrator:
Runs and aggregates rigorous comparisons across all model variants to answer the core research question:
"Does incorporating unstructured clinical notes improve early ICU mortality prediction compared with structured data alone?"
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from medguard.evaluation.metrics import compute_all_metrics, compute_bootstrap_ci
from medguard.utils.logging import get_logger

logger = get_logger("medguard.evaluation.ablation")


class AblationStudyRunner:
    """
    Standardized benchmark comparator.
    Collects validation/test predictions from each trained model architecture and produces an ablation table with 95% CIs.
    """

    def __init__(self, y_true: np.ndarray, num_bootstrap: int = 200, seed: int = 42):
        self.y_true = np.array(y_true).ravel()
        self.num_bootstrap = num_bootstrap
        self.seed = seed
        self.results: Dict[str, Dict[str, Any]] = {}

    def register_model_predictions(
        self,
        model_name: str,
        modality_type: str, # "Structured Tabular", "Structured Temporal", "Text Only", "Multimodal Fusion", "Clinical Score"
        y_prob: np.ndarray,
        description: str = "",
    ) -> None:
        """Register model predictions on the test set."""
        p = np.array(y_prob).ravel()
        point_metrics = compute_all_metrics(self.y_true, p)
        ci_metrics = compute_bootstrap_ci(self.y_true, p, n_iterations=self.num_bootstrap, seed=self.seed)

        self.results[model_name] = {
            "model_name": model_name,
            "modality_type": modality_type,
            "description": description,
            "metrics": point_metrics,
            "ci": ci_metrics,
            "y_prob": p.tolist(),
        }
        logger.info(
            f"Registered [{model_name}] ({modality_type}): "
            f"AUROC={point_metrics['auroc']} ({ci_metrics['auroc']['ci_str']}), "
            f"AUPRC={point_metrics['auprc']} ({ci_metrics['auprc']['ci_str']}), "
            f"Brier={point_metrics['brier_score']}"
        )

    def get_summary_table(self) -> pd.DataFrame:
        """Return formatted comparison table for research reporting and UI display."""
        rows = []
        for name, data in self.results.items():
            m = data["metrics"]
            ci = data["ci"]
            rows.append({
                "Model": name,
                "Modality": data["modality_type"],
                "AUROC (95% CI)": ci["auroc"]["ci_str"],
                "AUPRC (95% CI)": ci["auprc"]["ci_str"],
                "F1 Score": f"{m['f1']:.3f}",
                "Brier Score": f"{m['brier_score']:.3f}",
                "ECE": f"{m['ece']:.3f}",
                "Sensitivity": f"{m['sensitivity']:.3f}",
                "Specificity": f"{m['specificity']:.3f}",
                "AUROC_raw": m["auroc"],
                "AUPRC_raw": m["auprc"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="AUROC_raw", ascending=False).reset_index(drop=True)
        return df
