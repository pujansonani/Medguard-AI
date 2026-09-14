"""
Model Registry & Service Cache:
Loads pre-trained model weights, scalers, and explainers into memory on application startup.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from medguard.models.baselines import LogisticRegressionBaseline, XGBoostBaseline
from medguard.models.temporal_nn import TemporalGRUModel
from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.models.fusion import MedguardMultimodalModel
from medguard.calibration.temperature_scaling import TemperatureScaler
from medguard.explainability.shap_explainer import MedguardSHAPExplainer
from medguard.explainability.text_attribution import TextAttributionHighlighter
from medguard.preprocessing.time_series import FEATURE_COLUMNS
from medguard.utils.config import get_project_root, load_all_configs
from medguard.utils.logging import get_logger

logger = get_logger("medguard.api.dependencies")


class ModelService:
    """Singleton service managing all in-memory model instances and artifacts."""

    def __init__(self):
        self.root = get_project_root()
        self.configs = load_all_configs(self.root)
        self.cfg = self.configs["global"]
        
        self.ckpt_dir = self.root / self.cfg.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
        self.proc_dir = self.root / self.cfg.get("paths", {}).get("processed_dir", "data/processed")
        self.exp_dir = self.root / self.cfg.get("paths", {}).get("experiments_dir", "experiments/artifacts")

        self.multimodal_model: Optional[MedguardMultimodalModel] = None
        self.gru_model: Optional[TemporalGRUModel] = None
        self.xgb_model: Optional[XGBoostBaseline] = None
        self.nlp_model: Optional[ClinicalNLPModel] = None
        self.calibrator: Optional[TemperatureScaler] = None
        self.shap_explainer: Optional[MedguardSHAPExplainer] = None
        self.evaluation_results: Dict[str, Any] = {}
        self.is_loaded = False

    def load_artifacts(self) -> None:
        """Load trained model checkpoints and scalers."""
        device = self.cfg.get("device", "cpu")
        logger.info(f"Loading MEDGUARD AI artifacts from {self.ckpt_dir} (Device: {device})...")

        # 1. Multimodal Gated Model (Core)
        gated_path = self.ckpt_dir / "multimodal_gated.pt"
        if gated_path.exists():
            self.multimodal_model = MedguardMultimodalModel(fusion_type="gated", device=device)
            self.multimodal_model.load(gated_path)
            logger.info("Loaded Multimodal Gated Model.")

        # 2. Temporal GRU
        gru_path = self.ckpt_dir / "temporal_gru.pt"
        if gru_path.exists():
            self.gru_model = TemporalGRUModel(input_dim=len(FEATURE_COLUMNS), device=device)
            self.gru_model.load(gru_path)
            logger.info("Loaded Temporal GRU Model.")

        # 3. XGBoost
        xgb_path = self.ckpt_dir / "xgboost.joblib"
        if xgb_path.exists():
            self.xgb_model = XGBoostBaseline()
            self.xgb_model.load(xgb_path)
            logger.info("Loaded XGBoost Model.")

        # 4. Clinical NLP
        nlp_path = self.ckpt_dir / "clinical_nlp.pt"
        if nlp_path.exists():
            self.nlp_model = ClinicalNLPModel(device=device)
            self.nlp_model.load(nlp_path)
            logger.info("Loaded Clinical NLP Model.")

        # 5. Temperature Calibrator
        calib_path = self.ckpt_dir / "temperature_scaler.joblib"
        if calib_path.exists():
            self.calibrator = joblib.load(calib_path)
            logger.info(f"Loaded Temperature Scaler (T = {self.calibrator.temperature:.3f}).")

        # 6. SHAP Explainer
        if self.xgb_model is not None and (self.proc_dir / "df_tab_train.parquet").exists():
            df_bg = pd.read_parquet(self.proc_dir / "df_tab_train.parquet")
            self.shap_explainer = MedguardSHAPExplainer(self.xgb_model, df_bg)
            logger.info("Initialized SHAP Explainer.")

        # 7. Evaluation results
        eval_path = self.exp_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                self.evaluation_results = json.load(f)
            logger.info("Loaded Evaluation Benchmark Results.")

        self.is_loaded = True


# Global Singleton Instance
model_service = ModelService()
