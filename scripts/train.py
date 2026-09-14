"""
Model Training Script for MEDGUARD AI:
Trains all structured baselines, temporal deep learning architectures, clinical NLP models, and multimodal fusion variants.
Saves model weights, feature transformers, and calibrators to models/checkpoints/.
"""

import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.models.baselines import LogisticRegressionBaseline, XGBoostBaseline, LightGBMBaseline
from medguard.models.temporal_nn import TemporalGRUModel
from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.models.fusion import MedguardMultimodalModel
from medguard.calibration.temperature_scaling import TemperatureScaler
from medguard.utils.logging import get_logger
from medguard.utils.config import get_project_root, load_all_configs
from medguard.utils.seed import set_seed

logger = get_logger("medguard.scripts.train")


def main():
    parser = argparse.ArgumentParser(description="Train all MEDGUARD AI models.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]
    set_seed(cfg.get("seed", 42))

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    ckpt_dir = root / cfg.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not (proc_dir / "X_train.npy").exists():
        logger.error(f"Processed data not found at {proc_dir}. Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    logger.info("Loading preprocessed training and validation sets...")
    X_train = np.load(proc_dir / "X_train.npy")
    mask_train = np.load(proc_dir / "mask_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")

    X_val = np.load(proc_dir / "X_val.npy")
    mask_val = np.load(proc_dir / "mask_val.npy")
    y_val = np.load(proc_dir / "y_val.npy")

    df_tab_train = pd.read_parquet(proc_dir / "df_tab_train.parquet")
    df_tab_val = pd.read_parquet(proc_dir / "df_tab_val.parquet")

    with open(proc_dir / "texts_train.json", "r") as f:
        texts_train = json.load(f)
    with open(proc_dir / "texts_val.json", "r") as f:
        texts_val = json.load(f)

    device = cfg.get("device", "cpu")
    y_train_48h = y_train[:, 3] # 48-hour mortality target
    y_val_48h = y_val[:, 3]

    # =========================================================================
    # 1. Baseline A: Logistic Regression (Tabular Summary)
    # =========================================================================
    logger.info("Training Model 1/7: Logistic Regression Baseline...")
    lr_model = LogisticRegressionBaseline(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(df_tab_train, y_train_48h)
    lr_model.save(ckpt_dir / "logistic_regression.joblib")

    # =========================================================================
    # 2. Baseline B: XGBoost (Tabular Gradient Boosting)
    # =========================================================================
    logger.info("Training Model 2/7: XGBoost Baseline...")
    xgb_model = XGBoostBaseline(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_model.fit(df_tab_train, y_train_48h)
    xgb_model.save(ckpt_dir / "xgboost.joblib")

    # =========================================================================
    # 3. Model C: Temporal GRU (Structured Time-Series Deep Learning)
    # =========================================================================
    logger.info("Training Model 3/7: Temporal GRU (Time-Series Neural Net)...")
    gru_model = TemporalGRUModel(
        input_dim=X_train.shape[2],
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        epochs=cfg.get("training", {}).get("epochs", 25),
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        device=device,
    )
    gru_model.fit(X_train, mask_train, y_train, X_val, mask_val, y_val)
    gru_model.save(ckpt_dir / "temporal_gru.pt")

    # =========================================================================
    # 4. Model D: Clinical NLP Model (Text-Only ClinicalBERT / Transformer)
    # =========================================================================
    logger.info("Training Model 4/7: Clinical NLP Model (Text Only)...")
    nlp_model = ClinicalNLPModel(
        embedding_dim=128,
        epochs=15,
        batch_size=32,
        device=device,
    )
    nlp_model.fit(texts_train, y_train, texts_val, y_val)
    nlp_model.save(ckpt_dir / "clinical_nlp.pt")

    # =========================================================================
    # 5. Model E: Gated Multimodal Fusion (Structured + Clinical Text Notes)
    # =========================================================================
    logger.info("Training Model 5/7: Gated Multimodal Fusion Model (Core Architecture)...")
    gated_fusion = MedguardMultimodalModel(
        fusion_type="gated",
        lr=0.001,
        epochs=cfg.get("training", {}).get("epochs", 25),
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        device=device,
    )
    gated_fusion.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
    gated_fusion.save(ckpt_dir / "multimodal_gated.pt")

    # =========================================================================
    # 6. Model F: Intermediate Concatenation Multimodal Fusion
    # =========================================================================
    logger.info("Training Model 6/7: Intermediate Concat Fusion Model...")
    inter_fusion = MedguardMultimodalModel(
        fusion_type="intermediate",
        epochs=20,
        batch_size=32,
        device=device,
    )
    inter_fusion.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
    inter_fusion.save(ckpt_dir / "multimodal_intermediate.pt")

    # =========================================================================
    # 7. Model G: Cross-Modal Attention Fusion (Experimental)
    # =========================================================================
    logger.info("Training Model 7/7: Cross-Modal Attention Fusion Model...")
    cross_fusion = MedguardMultimodalModel(
        fusion_type="cross_attention",
        epochs=20,
        batch_size=32,
        device=device,
    )
    cross_fusion.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
    cross_fusion.save(ckpt_dir / "multimodal_cross_attention.pt")

    # =========================================================================
    # 8. Temperature Calibrator for Production Gated Fusion
    # =========================================================================
    logger.info("Fitting post-hoc Temperature Calibrator on validation set...")
    val_probs_gated = gated_fusion.predict_proba(X_val, mask_val, texts_val)[:, 3]
    calibrator = TemperatureScaler()
    calibrator.fit(val_probs_gated, y_val_48h)
    joblib.dump(calibrator, ckpt_dir / "temperature_scaler.joblib")
    logger.info(f"Learned Optimal Calibration Temperature T = {calibrator.temperature:.3f}")

    logger.info(f"All models trained successfully and saved to {ckpt_dir.resolve()}!")


if __name__ == "__main__":
    main()
