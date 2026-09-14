"""
Post-hoc Temperature Scaling and Calibration Metrics (Brier Score, Expected Calibration Error).
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss


class TemperatureScaler:
    """
    Calibrates binary prediction logits via temperature parameter T:
    p_calibrated = sigmoid(logit / T)
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted = False

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """
        Fit optimal temperature T on validation probabilities to minimize cross-entropy loss.
        """
        # Convert probabilities to logits
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        y = np.array(y_true).ravel()

        def nll_loss(T: np.ndarray) -> float:
            scaled_logits = logits / T[0]
            # stable log_loss
            loss = np.mean(
                (1.0 - y) * scaled_logits + np.log(1.0 + np.exp(-scaled_logits))
            )
            return float(loss)

        res = minimize(nll_loss, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        self.is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to input probabilities."""
        if not self.is_fitted:
            return probs
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        scaled_logits = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE):
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    y = np.array(y_true).ravel()
    p = np.array(probs).ravel()
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (p >= bin_lower) & (p < bin_upper) if i < n_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y[in_bin])
            bin_conf = np.mean(p[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute bin statistics for reliability diagrams.
    """
    y = np.array(y_true).ravel()
    p = np.array(probs).ravel()
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (p >= bin_lower) & (p < bin_upper) if i < n_bins - 1 else (p >= bin_lower) & (p <= bin_upper)
        bin_size = int(np.sum(in_bin))
        bin_counts.append(bin_size)

        if bin_size > 0:
            bin_accs.append(float(np.mean(y[in_bin])))
            bin_confs.append(float(np.mean(p[in_bin])))
        else:
            bin_accs.append(float((bin_lower + bin_upper) / 2.0))
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))

    return {
        "bin_confs": np.array(bin_confs),
        "bin_accs": np.array(bin_accs),
        "bin_counts": np.array(bin_counts),
        "brier_score": float(brier_score_loss(y, p)),
        "ece": compute_ece(p, y, n_bins=n_bins),
    }
