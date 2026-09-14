"""
Model Evaluation and Research Benchmark Script:
Executes the full ablation study, subgroup fairness audit, and robustness stress testing on held-out test data.
Saves comprehensive results to experiments/artifacts/.
"""

import argparse
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.models.baselines import LogisticRegressionBaseline, XGBoostBaseline
from medguard.models.temporal_nn import TemporalGRUModel
from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.models.fusion import MedguardMultimodalModel
from medguard.evaluation.ablation import AblationStudyRunner
from medguard.evaluation.fairness import FairnessAuditor
from medguard.evaluation.robustness import RobustnessTester
from medguard.evaluation.metrics import get_roc_and_pr_curves
from medguard.calibration.temperature_scaling import compute_reliability_curve
from medguard.utils.logging import get_logger
from medguard.utils.config import get_project_root, load_all_configs

logger = get_logger("medguard.scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all MEDGUARD AI models and generate ablation benchmark.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    args = parser.parse_args()

    root = get_project_root()
    configs = load_all_configs(root)
    cfg = configs["global"]

    proc_dir = root / cfg.get("paths", {}).get("processed_dir", "data/processed")
    ckpt_dir = root / cfg.get("paths", {}).get("checkpoints_dir", "models/checkpoints")
    exp_dir = root / cfg.get("paths", {}).get("experiments_dir", "experiments/artifacts")
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test dataset and patient metadata...")
    X_test = np.load(proc_dir / "X_test.npy")
    mask_test = np.load(proc_dir / "mask_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")
    df_tab_test = pd.read_parquet(proc_dir / "df_tab_test.parquet")
    df_scores_test = pd.read_parquet(proc_dir / "df_scores_test.parquet")
    df_patients_test = pd.read_parquet(proc_dir / "patients_test.parquet")

    with open(proc_dir / "texts_test.json", "r") as f:
        texts_test = json.load(f)

    y_test_48h = y_test[:, 3]
    device = cfg.get("device", "cpu")

    # Initialize Ablation Runner
    ablation = AblationStudyRunner(
        y_true=y_test_48h,
        num_bootstrap=cfg.get("evaluation", {}).get("bootstrap_iterations", 200),
        seed=cfg.get("seed", 42),
    )

    # 1. Clinical Baseline: SOFA Proxy
    sofa_scores = df_scores_test["sofa_score"].values
    sofa_norm_probs = sofa_scores / 24.0 # Normalized score to [0, 1] probability proxy
    ablation.register_model_predictions(
        model_name="SOFA Proxy Score",
        modality_type="Clinical Score",
        y_prob=sofa_norm_probs,
        description="Classical ICU Organ Dysfunction Score (Max 24h severity)",
    )

    # 2. Clinical Baseline: SAPS II Proxy
    saps_scores = df_scores_test["saps_ii_score"].values
    saps_norm_probs = saps_scores / 100.0 # Normalized score proxy
    ablation.register_model_predictions(
        model_name="SAPS II Proxy Score",
        modality_type="Clinical Score",
        y_prob=saps_norm_probs,
        description="Simplified Acute Physiology Score II (Proxy)",
    )

    # 3. Logistic Regression
    lr_model = LogisticRegressionBaseline()
    lr_model.load(ckpt_dir / "logistic_regression.joblib")
    p_lr = lr_model.predict_proba(df_tab_test)
    ablation.register_model_predictions(
        model_name="Logistic Regression",
        modality_type="Structured Tabular",
        y_prob=p_lr,
        description="Linear model on summary statistics of 24h vitals/labs",
    )

    # 4. XGBoost
    xgb_model = XGBoostBaseline()
    xgb_model.load(ckpt_dir / "xgboost.joblib")
    p_xgb = xgb_model.predict_proba(df_tab_test)
    ablation.register_model_predictions(
        model_name="XGBoost",
        modality_type="Structured Tabular",
        y_prob=p_xgb,
        description="Gradient boosted trees on summary statistics of 24h vitals/labs",
    )

    # 5. Temporal GRU
    gru_model = TemporalGRUModel(input_dim=X_test.shape[2], device=device)
    gru_model.load(ckpt_dir / "temporal_gru.pt")
    p_gru_multi = gru_model.predict_proba(X_test, mask_test) # (N, 4)
    p_gru_48h = p_gru_multi[:, 3]
    ablation.register_model_predictions(
        model_name="Temporal GRU",
        modality_type="Structured Temporal",
        y_prob=p_gru_48h,
        description="Deep recurrent neural network on 24 hourly steps with missingness mask",
    )

    # 6. Clinical NLP (Text Only)
    nlp_model = ClinicalNLPModel(device=device)
    nlp_model.load(ckpt_dir / "clinical_nlp.pt")
    p_nlp_multi = nlp_model.predict_proba(texts_test)
    p_nlp_48h = p_nlp_multi[:, 3]
    ablation.register_model_predictions(
        model_name="ClinicalBERT NLP",
        modality_type="Text Only",
        y_prob=p_nlp_48h,
        description="Transformer clinical text encoder on nursing/physician notes",
    )

    # 7. Intermediate Concat Fusion
    inter_fusion = MedguardMultimodalModel(fusion_type="intermediate", device=device)
    inter_fusion.load(ckpt_dir / "multimodal_intermediate.pt")
    p_inter_multi = inter_fusion.predict_proba(X_test, mask_test, texts_test)
    p_inter_48h = p_inter_multi[:, 3]
    ablation.register_model_predictions(
        model_name="Multimodal (Intermediate Concat)",
        modality_type="Multimodal Fusion",
        y_prob=p_inter_48h,
        description="Concatenation of structured temporal and clinical text embeddings",
    )

    # 8. Cross-Attention Fusion
    cross_fusion = MedguardMultimodalModel(fusion_type="cross_attention", device=device)
    cross_fusion.load(ckpt_dir / "multimodal_cross_attention.pt")
    p_cross_multi = cross_fusion.predict_proba(X_test, mask_test, texts_test)
    p_cross_48h = p_cross_multi[:, 3]
    ablation.register_model_predictions(
        model_name="Multimodal (Cross-Attention)",
        modality_type="Multimodal Fusion",
        y_prob=p_cross_48h,
        description="Cross-attention between clinical text tokens and physiological sequence states",
    )

    # 9. Gated Multimodal Fusion (Uncalibrated)
    gated_fusion = MedguardMultimodalModel(fusion_type="gated", device=device)
    gated_fusion.load(ckpt_dir / "multimodal_gated.pt")
    p_gated_multi = gated_fusion.predict_proba(X_test, mask_test, texts_test)
    p_gated_48h = p_gated_multi[:, 3]
    ablation.register_model_predictions(
        model_name="Multimodal (Gated Fusion)",
        modality_type="Multimodal Fusion",
        y_prob=p_gated_48h,
        description="Learned sigmoid gating dynamically weighting structured vs text modalities",
    )

    # 10. Gated Multimodal Fusion (Calibrated)
    calibrator = joblib.load(ckpt_dir / "temperature_scaler.joblib")
    p_gated_calibrated = calibrator.transform(p_gated_48h)
    ablation.register_model_predictions(
        model_name="Multimodal (Gated + Calibrated)",
        modality_type="Multimodal Fusion",
        y_prob=p_gated_calibrated,
        description="Gated multimodal model with post-hoc temperature calibration",
    )

    # Save summary ablation table
    df_ablation = ablation.get_summary_table()
    df_ablation.to_csv(exp_dir / "ablation_table.csv", index=False)
    logger.info("\n" + df_ablation.to_string())

    # Curves for all models
    curves_dict = {}
    reliability_dict = {}
    for name, data in ablation.results.items():
        p_arr = np.array(data["y_prob"])
        curves_dict[name] = get_roc_and_pr_curves(y_test_48h, p_arr)
        rel = compute_reliability_curve(p_arr, y_test_48h, n_bins=10)
        reliability_dict[name] = {
            "bin_confs": rel["bin_confs"].tolist(),
            "bin_accs": rel["bin_accs"].tolist(),
            "bin_counts": rel["bin_counts"].tolist(),
            "brier_score": rel["brier_score"],
            "ece": rel["ece"],
        }

    # Subgroup Fairness Audit
    logger.info("Conducting Subgroup Fairness Audit on Gated Multimodal Model...")
    auditor = FairnessAuditor()
    fairness_results = auditor.audit_subgroups(df_patients_test, y_test_48h, p_gated_calibrated)

    # Robustness Stress Testing
    logger.info("Conducting Robustness Stress Tests...")
    rob_tester = RobustnessTester(seed=cfg.get("seed", 42))
    missingness_decay = rob_tester.test_missingness_decay(
        model=gated_fusion,
        X_struct=X_test,
        mask_struct=mask_test,
        texts=texts_test,
        y_true=y_test_48h,
        missing_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
    )
    noise_decay = rob_tester.test_sensor_noise_decay(
        model=gated_fusion,
        X_struct=X_test,
        mask_struct=mask_test,
        texts=texts_test,
        y_true=y_test_48h,
        noise_levels=[0.0, 0.05, 0.1, 0.2, 0.4],
    )

    # Assemble and save full evaluation results JSON
    full_eval_payload = {
        "ablation_results": ablation.results,
        "curves": curves_dict,
        "calibration": reliability_dict,
        "fairness": fairness_results,
        "robustness": {
            "missingness_decay": missingness_decay,
            "sensor_noise_decay": noise_decay,
        },
        "temperature_calibrator": {
            "temperature": calibrator.temperature,
        }
    }

    with open(exp_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(full_eval_payload, f, indent=2)

    logger.info(f"Evaluation finished! Results saved to {exp_dir.resolve()}")


if __name__ == "__main__":
    main()
