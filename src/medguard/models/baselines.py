"""
Structured Tabular Baselines: Logistic Regression, XGBoost, and LightGBM.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from medguard.models.base import StructuredModel
from medguard.utils.logging import get_logger

logger = get_logger("medguard.models.baselines")


class LogisticRegressionBaseline(StructuredModel):
    """L2-regularized Logistic Regression baseline fitted on scaled tabular summary features."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        super().__init__(name="LogisticRegression", version="0.1.0")
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> "LogisticRegressionBaseline":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.array(y).ravel()
        X_scaled = self.scaler.fit_transform(X_arr)
        self.clf.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_arr)
        # Return probability of positive class (mortality = 1)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return standardized model coefficients as importance scores."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        coefs = self.clf.coef_[0]
        return {feat: float(coef) for feat, coef in zip(self.feature_names_, coefs)}

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "clf": self.clf, "feature_names": self.feature_names_}, filepath)

    def load(self, filepath: Union[str, Path]) -> "LogisticRegressionBaseline":
        data = joblib.load(filepath)
        self.scaler = data["scaler"]
        self.clf = data["clf"]
        self.feature_names_ = data["feature_names"]
        self.is_fitted = True
        return self


class XGBoostBaseline(StructuredModel):
    """XGBoost Classifier baseline on tabular features."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 3.5,
        random_state: int = 42,
    ):
        super().__init__(name="XGBoost", version="0.1.0")
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": 1,
            "eval_metric": "logloss",
        }
        self.clf = xgb.XGBClassifier(**self.params)
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> "XGBoostBaseline":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.array(y).ravel()
        self.clf.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.clf.predict_proba(X_arr)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        importances = self.clf.feature_importances_
        return {feat: float(imp) for feat, imp in zip(self.feature_names_, importances)}

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "feature_names": self.feature_names_}, filepath)

    def load(self, filepath: Union[str, Path]) -> "XGBoostBaseline":
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.feature_names_ = data["feature_names"]
        self.is_fitted = True
        return self


class LightGBMBaseline(StructuredModel):
    """LightGBM Classifier baseline on tabular features."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 3.5,
        random_state: int = 42,
    ):
        super().__init__(name="LightGBM", version="0.1.0")
        self.clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | pd.Series) -> "LightGBMBaseline":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.array(y).ravel()
        self.clf.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.clf.predict_proba(X_arr)[:, 1]

    def save(self, filepath: Union[str, Path]) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "feature_names": self.feature_names_}, filepath)

    def load(self, filepath: Union[str, Path]) -> "LightGBMBaseline":
        data = joblib.load(filepath)
        self.clf = data["clf"]
        self.feature_names_ = data["feature_names"]
        self.is_fitted = True
        return self
