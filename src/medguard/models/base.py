"""
Base model interfaces for MEDGUARD AI.
Enforces standard .fit(), .predict_proba(), .save(), and .load() methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseMedguardModel(ABC):
    """Abstract base class for all MEDGUARD models (tabular, temporal, NLP, and fusion)."""

    def __init__(self, name: str, version: str = "0.1.0"):
        self.name = name
        self.version = version
        self.is_fitted = False

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseMedguardModel":
        """Fit model to training dataset."""
        pass

    @abstractmethod
    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        """Predict probabilities of mortality. Returns array of shape (N,) or (N, Horiz)."""
        pass

    @abstractmethod
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model artifacts to disk."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> "BaseMedguardModel":
        """Load model artifacts from disk."""
        pass


class StructuredModel(BaseMedguardModel):
    """Base class for structured tabular or temporal sequence models."""
    pass


class TextModel(BaseMedguardModel):
    """Base class for unstructured clinical text NLP models."""
    pass


class MultimodalModel(BaseMedguardModel):
    """Base class for multimodal fusion architectures."""
    
    @abstractmethod
    def get_modality_weights(self, *args, **kwargs) -> Optional[np.ndarray]:
        """Return learned gating or attention weights for modality importance analysis."""
        pass
