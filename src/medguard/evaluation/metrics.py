"""
Evaluation metrics suite with non-parametric bootstrap confidence intervals.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

from medguard.calibration.temperature_scaling import compute_ece


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for binary ICU deterioration prediction."""
    y = np.array(y_true).ravel()
    p = np.array(y_prob).ravel()
    y_pred = (p >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Handle single class edge case in bootstrap
    try:
        auroc = float(roc_auc_score(y, p))
    except Exception:
        auroc = 0.5

    try:
        auprc = float(average_precision_score(y, p))
    except Exception:
        auprc = float(np.mean(y))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "ece": round(compute_ece(p, y), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute non-parametric bootstrap confidence intervals: [lower_bound, mean, upper_bound].
    """
    rng = np.random.default_rng(seed)
    y = np.array(y_true).ravel()
    p = np.array(y_prob).ravel()
    N = len(y)

    boot_metrics: Dict[str, List[float]] = {
        "auroc": [], "auprc": [], "f1": [], "brier_score": [], "ece": [], "recall": [], "specificity": []
    }

    for _ in range(n_iterations):
        indices = rng.integers(0, N, size=N)
        y_boot = y[indices]
        p_boot = p[indices]

        # Only compute if both classes are sampled
        if len(np.unique(y_boot)) < 2:
            continue

        res = compute_all_metrics(y_boot, p_boot)
        for k in boot_metrics:
            boot_metrics[k].append(res[k])

    alpha = (1.0 - confidence_level) / 2.0
    results_ci: Dict[str, Dict[str, float]] = {}

    for k, vals in boot_metrics.items():
        if vals:
            low = float(np.percentile(vals, alpha * 100))
            high = float(np.percentile(vals, (1.0 - alpha) * 100))
            mean = float(np.mean(vals))
            results_ci[k] = {
                "mean": round(mean, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "ci_str": f"{mean:.3f} [{low:.3f}, {high:.3f}]",
            }
        else:
            results_ci[k] = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_str": "N/A"}

    return results_ci


def get_roc_and_pr_curves(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Generate curve coordinates for ROC and Precision-Recall visualization."""
    y = np.array(y_true).ravel()
    p = np.array(y_prob).ravel()

    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"precision": prec.tolist(), "recall": rec.tolist()},
    }
