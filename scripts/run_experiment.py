#!/usr/bin/env python3
"""
MEDGUARD AI: Single-Command Full End-to-End Experiment Pipeline

Executes the entire research workflow:
1. Validates data integrity and schema
2. Builds ICU cohort and enforces zero-leakage patient splitting
3. Computes engineered 24h features & clinical baseline scores
4. Encodes and caches clinical text representations
5. Trains baseline, deep temporal, clinical NLP, and multimodal fusion architectures
6. Evaluates discrimination (ROC-AUC, PR-AUC), calibration (ECE), fairness, and robustness
7. Generates publication-ready figures, tables, and Markdown report

Usage:
    python scripts/run_experiment.py --mode quick
    python scripts/run_experiment.py --mode standard
    python scripts/run_experiment.py --mode full --device mps
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.utils.config import get_project_root
from medguard.utils.logging import get_logger

logger = get_logger("medguard.experiment_runner")


def run_stage(cmd: list[str], stage_name: str, root: Path):
    """Executes a sub-script stage and checks exit status."""
    logger.info(f"\n{'='*70}\n  STAGE: {stage_name.upper()}\n  Command: {' '.join(cmd)}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=root)
    elapsed = time.time() - t0
    if res.returncode != 0:
        logger.error(f"Stage '{stage_name}' failed with exit code {res.returncode}. Aborting experiment.")
        sys.exit(res.returncode)
    logger.info(f"Stage '{stage_name}' completed in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Run complete MEDGUARD AI end-to-end experiment.")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["quick", "standard", "full"],
        help="Experiment intensity preset",
    )
    parser.add_argument("--device", type=str, default=None, help="Acceleration device (cuda, mps, cpu)")
    parser.add_argument("--mimic-dir", type=str, default=None, help="Raw MIMIC-IV path if available")
    parser.add_argument("--notes-dir", type=str, default=None, help="Raw MIMIC-IV-Note path if available")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic demo data")
    args = parser.parse_args()

    root = get_project_root()
    py_bin = sys.executable

    logger.info(f"Initiating MEDGUARD AI End-to-End Research Experiment (Preset: {args.mode.upper()})...")

    # STAGE 1: Data Ingestion Validation
    val_cmd = [py_bin, "scripts/validate_data.py"]
    if args.mimic_dir:
        val_cmd.extend(["--mimic-dir", args.mimic_dir])
    if args.notes_dir:
        val_cmd.extend(["--notes-dir", args.notes_dir])
    run_stage(val_cmd, "1. Data Validation", root)

    # STAGE 2: Cohort Construction & Feature Extraction
    prep_cmd = [py_bin, "scripts/prepare_data.py"]
    if args.force_synthetic:
        prep_cmd.append("--force_synthetic")
    if args.mimic_dir:
        prep_cmd.extend(["--mimic-dir", args.mimic_dir])
    if args.notes_dir:
        prep_cmd.extend(["--notes-dir", args.notes_dir])
    if args.mode == "quick":
        prep_cmd.extend(["--max_patients", "200"])
    run_stage(prep_cmd, "2. Cohort Construction & Feature Engineering", root)

    # STAGE 3: Exploratory Data Analysis (EDA)
    eda_cmd = [py_bin, "scripts/run_eda.py"]
    run_stage(eda_cmd, "3. Exploratory Data Analysis (EDA)", root)

    # STAGE 4: Clinical Note Pre-Encoding & Caching
    encode_cmd = [py_bin, "scripts/encode_notes.py", "--mode", args.mode]
    if args.device:
        encode_cmd.extend(["--device", args.device])
    run_stage(encode_cmd, "4. Clinical Text Embedding & Caching", root)

    # STAGE 5: Multi-Model Training
    train_cmd = [py_bin, "scripts/train.py", "--model", "all", "--mode", args.mode]
    if args.device:
        train_cmd.extend(["--device", args.device])
    run_stage(train_cmd, "5. Multi-Model Benchmark Training", root)

    # STAGE 6: Evaluation & Ablation Benchmarking
    eval_cmd = [py_bin, "scripts/evaluate.py"]
    run_stage(eval_cmd, "6. Model Evaluation & Subgroup Fairness Audit", root)

    # STAGE 7: Publication Report & Figures Generation
    rep_cmd = [py_bin, "scripts/generate_report.py"]
    run_stage(rep_cmd, "7. Publication Report & Figures Generation", root)

    logger.info(
        "\n" + "=" * 70 + "\n"
        "  FULL EXPERIMENT COMPLETE!\n"
        "  - Ablation Results: experiments/artifacts/results.csv\n"
        "  - Publication Report: reports/generated/research_evaluation_report.md\n"
        "  - Figures: reports/figures/\n"
        "  - Tables: reports/tables/\n"
        "  - Launch Analytics Portal: streamlit run app/streamlit_app.py\n"
        "  - Launch API Server: uvicorn api.main:app --port 8000\n"
        + "=" * 70
    )


if __name__ == "__main__":
    main()
