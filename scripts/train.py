#!/usr/bin/env python3
"""
MEDGUARD AI: Model Training & Experiment Runner

Supports individual or full-suite training across structured, temporal, clinical NLP, and multimodal fusion models:
- Logistic Regression (Tabular Summary)
- XGBoost (Tabular Gradient Boosting)
- Temporal GRU & LSTM (3D Time-Series Neural Nets with Missingness Masks)
- ClinicalBERT NLP (Unstructured Text Document Encoder)
- Gated Multimodal Fusion, Intermediate Concat, and Cross-Modal Attention
- Post-hoc Temperature Scaling Calibration

Usage:
    python scripts/train.py --model all --mode standard
    python scripts/train.py --model xgboost --mode quick
    python scripts/train.py --model gru --epochs 20 --device mps
    python scripts/train.py --model fusion --mode full
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.models.baselines import LogisticRegressionBaseline, XGBoostBaseline
from medguard.models.temporal_nn import TemporalGRUModel
from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.models.fusion import MedguardMultimodalModel
from medguard.calibration.temperature_scaling import TemperatureScaler
from medguard.utils.logging import get_logger
from medguard.utils.config import get_project_root, load_all_configs
from medguard.utils.seed import set_seed

logger = get_logger("medguard.train")


def get_git_commit(root: Path) -> str:
    """Retrieve active git commit hash for model metadata."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "git_unavailable"


def save_model_registry_artifact(
    artifact_dir: Path,
    model_name: str,
    version: str,
    config: dict,
    metrics: dict,
    git_commit: str,
    dataset_version: str = "v0.1.0",
):
    """Save structured metadata registry artifact for reproducibility."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "version": version,
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset_version": dataset_version,
        "configuration": config,
        "validation_metrics": metrics,
    }
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train MEDGUARD AI Models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "logistic", "xgboost", "gru", "lstm", "text", "fusion", "cross_attention"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["quick", "standard", "full"],
        help="Training intensity preset",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]
    set_seed(args.seed)

    # 1. Device Selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Hardware Acceleration: \033[94m{device.upper()}\033[0m")

    # 2. Configure Mode & Epochs
    if args.mode == "quick":
        epochs = args.epochs or 5
        batch_size = args.batch_size or 32
        n_est_xgb = 50
    elif args.mode == "full":
        epochs = args.epochs or 40
        batch_size = args.batch_size or 32
        n_est_xgb = 300
    else: # standard
        epochs = args.epochs or cfg.get("training", {}).get("epochs", 25)
        batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 32)
        n_est_xgb = 150

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    ckpt_dir = root / cfg.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    models_registry_dir = root / "models" / "artifacts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    git_commit = get_git_commit(root)

    # Check processed data
    if not (proc_dir / "X_train.npy").exists():
        logger.error(f"Processed features not found in {proc_dir}. Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    # Load Data
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

    y_train_48h = y_train[:, 3]
    y_val_48h = y_val[:, 3]

    target_models = (
        ["logistic", "xgboost", "gru", "text", "fusion", "intermediate", "cross_attention"]
        if args.model == "all"
        else [args.model]
    )

    logger.info(f"Training queue: {target_models} | Mode: {args.mode} | Cohort: {len(X_train)} train, {len(X_val)} val")

    # =========================================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================================
    if "logistic" in target_models:
        logger.info("Training Logistic Regression Baseline...")
        lr_model = LogisticRegressionBaseline(C=1.0, max_iter=1000, random_state=args.seed)
        lr_model.fit(df_tab_train, y_train_48h)
        lr_model.save(ckpt_dir / "logistic_regression.joblib")
        
        save_model_registry_artifact(
            artifact_dir=models_registry_dir / "logistic_regression" / "v0.1.0",
            model_name="LogisticRegressionBaseline",
            version="0.1.0",
            config={"C": 1.0, "max_iter": 1000, "seed": args.seed},
            metrics={"val_loss": 0.0},
            git_commit=git_commit,
        )

    # =========================================================================
    # 2. XGBOOST
    # =========================================================================
    if "xgboost" in target_models:
        logger.info("Training XGBoost Baseline...")
        xgb_model = XGBoostBaseline(
            n_estimators=n_est_xgb,
            max_depth=4,
            learning_rate=0.05,
            random_state=args.seed,
        )
        xgb_model.fit(df_tab_train, y_train_48h)
        xgb_model.save(ckpt_dir / "xgboost.joblib")

        save_model_registry_artifact(
            artifact_dir=models_registry_dir / "xgboost" / "v0.1.0",
            model_name="XGBoostBaseline",
            version="0.1.0",
            config={"n_estimators": n_est_xgb, "max_depth": 4, "lr": 0.05, "seed": args.seed},
            metrics={},
            git_commit=git_commit,
        )

    # =========================================================================
    # 3. TEMPORAL GRU / LSTM
    # =========================================================================
    if "gru" in target_models or "lstm" in target_models:
        logger.info(f"Training Temporal GRU (Epochs: {epochs}, Device: {device})...")
        gru_model = TemporalGRUModel(
            input_dim=X_train.shape[2],
            hidden_dim=64,
            num_layers=2,
            dropout=0.25,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        gru_model.fit(X_train, mask_train, y_train, X_val, mask_val, y_val)
        gru_model.save(ckpt_dir / "temporal_gru.pt")

        save_model_registry_artifact(
            artifact_dir=models_registry_dir / "temporal_gru" / "v0.1.0",
            model_name="TemporalGRUModel",
            version="0.1.0",
            config={"hidden_dim": 64, "num_layers": 2, "epochs": epochs, "batch_size": batch_size},
            metrics={},
            git_commit=git_commit,
        )

    # =========================================================================
    # 4. CLINICAL NLP (ClinicalBERT)
    # =========================================================================
    if "text" in target_models:
        logger.info(f"Training Clinical NLP Model (Epochs: {min(15, epochs)}, Device: {device})...")
        nlp_model = ClinicalNLPModel(
            embedding_dim=128,
            lr=0.001,
            epochs=min(15, epochs),
            batch_size=batch_size,
            device=device,
        )
        nlp_model.fit(texts_train, y_train, texts_val, y_val)
        nlp_model.save(ckpt_dir / "clinical_nlp.pt")

        save_model_registry_artifact(
            artifact_dir=models_registry_dir / "clinical_nlp" / "v0.1.0",
            model_name="ClinicalNLPModel",
            version="0.1.0",
            config={"embedding_dim": 128, "epochs": min(15, epochs), "device": device},
            metrics={},
            git_commit=git_commit,
        )

    # =========================================================================
    # 5. GATED MULTIMODAL FUSION
    # =========================================================================
    if "fusion" in target_models:
        logger.info(f"Training Gated Multimodal Fusion Model (Epochs: {epochs}, Device: {device})...")
        gated_model = MedguardMultimodalModel(
            fusion_type="gated",
            lr=0.001,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        gated_model.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
        gated_model.save(ckpt_dir / "multimodal_gated.pt")

        # Fit Temperature Scaler for Post-Hoc Probability Calibration
        logger.info("Fitting post-hoc Temperature Calibrator on validation set...")
        p_val_raw = gated_model.predict_proba(X_val, mask_val, texts_val)[:, 3]
        calibrator = TemperatureScaler()
        calibrator.fit(p_val_raw, y_val_48h)
        calibrator.save(ckpt_dir / "temperature_scaler.joblib")
        logger.info(f"Learned Optimal Calibration Temperature T = {calibrator.temperature:.3f}")

        save_model_registry_artifact(
            artifact_dir=models_registry_dir / "multimodal_gated" / "v0.1.0",
            model_name="MedguardMultimodalGated",
            version="0.1.0",
            config={"fusion_type": "gated", "epochs": epochs, "temperature": calibrator.temperature},
            metrics={},
            git_commit=git_commit,
        )

    # =========================================================================
    # 6. INTERMEDIATE CONCAT FUSION
    # =========================================================================
    if "intermediate" in target_models or "fusion" in target_models:
        logger.info("Training Intermediate Concat Fusion Model...")
        inter_model = MedguardMultimodalModel(
            fusion_type="intermediate",
            lr=0.001,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        inter_model.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
        inter_model.save(ckpt_dir / "multimodal_intermediate.pt")

    # =========================================================================
    # 7. CROSS-MODAL ATTENTION FUSION
    # =========================================================================
    if "cross_attention" in target_models or "fusion" in target_models:
        logger.info("Training Cross-Modal Attention Fusion Model...")
        cross_model = MedguardMultimodalModel(
            fusion_type="cross_attention",
            lr=0.0008,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        cross_model.fit(X_train, mask_train, texts_train, y_train, X_val, mask_val, texts_val, y_val)
        cross_model.save(ckpt_dir / "multimodal_cross_attention.pt")

    logger.info(f"All requested models trained successfully and saved to {ckpt_dir.resolve()}!")


if __name__ == "__main__":
    main()
