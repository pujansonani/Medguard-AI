"""
Feature scaling utilities fitted strictly on the training fold.
Supports 3D temporal tensors (N, T, D) and 2D tabular summary arrays.
"""

from typing import Optional
import numpy as np


class TemporalStandardScaler:
    """Standardizes 3D time-series tensors along the feature dimension using training statistics."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> "TemporalStandardScaler":
        """
        Fit mean and standard deviation on training tensor (N, T, D).
        If mask (N, T, D) is provided, computes statistics strictly over observed non-missing values.
        """
        N, T, D = X.shape
        self.mean_ = np.zeros(D, dtype=np.float32)
        self.std_ = np.ones(D, dtype=np.float32)

        for d in range(D):
            if mask is not None:
                obs_mask = mask[:, :, d] > 0.5
                vals = X[:, :, d][obs_mask]
            else:
                vals = X[:, :, d].flatten()

            if len(vals) > 0:
                self.mean_[d] = float(np.mean(vals))
                s = float(np.std(vals))
                self.std_[d] = s if s > self.eps else 1.0
            else:
                self.mean_[d] = 0.0
                self.std_[d] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize tensor: (X - mean) / std."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        # Broadcast (D,) over (N, T, D)
        return (X - self.mean_.reshape(1, 1, -1)) / (self.std_.reshape(1, 1, -1) + self.eps)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Invert scaling back to original clinical units."""
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return X_scaled * self.std_.reshape(1, 1, -1) + self.mean_.reshape(1, 1, -1)
