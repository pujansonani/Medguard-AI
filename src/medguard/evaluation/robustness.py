"""
Stress-testing and Robustness Evaluation under Sensor Noise and Increased Missingness.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from medguard.evaluation.metrics import compute_all_metrics


class RobustnessTester:
    """
    Evaluates degradation in model discrimination under:
    1. Additional synthetic missingness (masking 10%, 20%, 30%, 50% of observed vitals/labs).
    2. Additive Gaussian noise to physiological sensors (simulating calibration drift or measurement noise).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def test_missingness_decay(
        self,
        model: Any,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
        y_true: np.ndarray,
        missing_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate performance degradation as missing data increases.
        """
        results = []
        y = np.array(y_true).ravel()

        for rate in missing_rates:
            # Clone and inject additional missingness
            corrupted_mask = mask_struct.copy()
            corrupted_X = X_struct.copy()

            if rate > 0.0:
                drop_mask = self.rng.uniform(0, 1, size=corrupted_mask.shape) < rate
                corrupted_mask[drop_mask] = 0.0
                corrupted_X[drop_mask] = 0.0 # Re-zero or median fill

            if hasattr(model, "predict_proba"):
                if "texts" in model.predict_proba.__code__.co_varnames:
                    probs = model.predict_proba(corrupted_X, corrupted_mask, texts)[:, 3] # 48h horizon
                else:
                    probs = model.predict_proba(corrupted_X, corrupted_mask)[:, 3]
            else:
                probs = np.full(len(y), 0.5)

            metrics = compute_all_metrics(y, probs)
            results.append({
                "missingness_rate": rate,
                "missingness_pct": f"{int(rate * 100)}%",
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
            })

        return results

    def test_sensor_noise_decay(
        self,
        model: Any,
        X_struct: np.ndarray,
        mask_struct: np.ndarray,
        texts: List[str],
        y_true: np.ndarray,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate performance degradation under Gaussian noise injection.
        """
        results = []
        y = np.array(y_true).ravel()

        for noise_std in noise_levels:
            corrupted_X = X_struct.copy()
            if noise_std > 0.0:
                noise = self.rng.normal(0, noise_std, size=corrupted_X.shape)
                corrupted_X = corrupted_X + noise

            if hasattr(model, "predict_proba"):
                if "texts" in model.predict_proba.__code__.co_varnames:
                    probs = model.predict_proba(corrupted_X, mask_struct, texts)[:, 3]
                else:
                    probs = model.predict_proba(corrupted_X, mask_struct)[:, 3]
            else:
                probs = np.full(len(y), 0.5)

            metrics = compute_all_metrics(y, probs)
            results.append({
                "noise_std": noise_std,
                "noise_label": f"σ = {noise_std:.2f}",
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier_score": metrics["brier_score"],
            })

        return results
