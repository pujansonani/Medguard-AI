"""
Tests for Evaluation Metrics, Bootstrap Confidence Intervals, and Calibration.
"""

import numpy as np
import pytest

from medguard.evaluation.metrics import compute_all_metrics, compute_bootstrap_ci
from medguard.calibration.temperature_scaling import TemperatureScaler, compute_ece, compute_reliability_curve


def test_compute_all_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.85])

    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert metrics["auroc"] > 0.80
    assert metrics["accuracy"] == 1.0


def test_bootstrap_confidence_intervals():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1] * 5)
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.85] * 5)

    ci_results = compute_bootstrap_ci(y_true, y_prob, n_iterations=30, confidence_level=0.95, seed=42)

    assert "auroc" in ci_results
    assert ci_results["auroc"]["ci_low"] <= ci_results["auroc"]["mean"] <= ci_results["auroc"]["ci_high"]
    assert "ci_str" in ci_results["auroc"]


def test_temperature_scaling_and_ece():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.2, 0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    ece_before = compute_ece(y_prob, y_true, n_bins=5)
    scaler = TemperatureScaler()
    scaler.fit(y_prob, y_true)
    y_calib = scaler.transform(y_prob)
    ece_after = compute_ece(y_calib, y_true, n_bins=5)

    assert scaler.is_fitted
    assert scaler.temperature > 0.0
