#!/usr/bin/env python3
"""
MEDGUARD AI: Exploratory Data Analysis (EDA) Script

Generates comprehensive data science analytics, statistical distributions,
missingness heatmaps, temporal trajectories, and summary tables.
Saves publication-quality plots to reports/figures/eda/ and summary table to reports/tables/eda_summary.csv.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --output-dir reports/figures/eda
"""

from pathlib import Path
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.preprocessing.time_series import FEATURE_COLUMNS
from medguard.utils.config import get_project_root, load_all_configs
from medguard.utils.logging import get_logger

logger = get_logger("medguard.eda")


def run_eda(
    proc_dir: Path,
    synth_dir: Path,
    fig_dir: Path,
    tables_dir: Path,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Styling
    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "font.family": "sans-serif",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 1. Load Cohort
    cohort_path = proc_dir / "cohort.csv"
    if cohort_path.exists():
        df_cohort = pd.read_csv(cohort_path)
    elif (synth_dir / "patients.parquet").exists():
        df_cohort = pd.read_parquet(synth_dir / "patients.parquet")
    else:
        logger.error("No cohort data found. Run `python scripts/prepare_data.py` first.")
        return

    # Load Timeseries
    ts_path = synth_dir / "timeseries.parquet"
    df_ts = pd.read_parquet(ts_path) if ts_path.exists() else pd.DataFrame()

    # Load Notes
    notes_path = synth_dir / "notes.parquet"
    df_notes = pd.read_parquet(notes_path) if notes_path.exists() else pd.DataFrame()

    logger.info(f"Running EDA on cohort of {len(df_cohort):,} patients...")

    # =========================================================================
    # PLOT 1: Mortality & Class Balance
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Mortality count bar
    m_counts = df_cohort["mortality_48h"].value_counts().sort_index()
    labels = ["Survived", "Deteriorated / Deceased"]
    colors = ["#3ECFB2", "#FF5F56"]
    ax1.bar(labels, m_counts.values, color=colors, width=0.5, edgecolor="#0F2B46", lw=1.2)
    ax1.set_title("48-Hour ICU Mortality Class Distribution")
    ax1.set_ylabel("Patient Count")
    for i, v in enumerate(m_counts.values):
        pct = v / len(df_cohort) * 100
        ax1.text(i, v + 5, f"{v:,} ({pct:.1f}%)", ha="center", fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Multi-horizon mortality rates
    horizons = [6, 12, 24, 48]
    h_rates = []
    for h in horizons:
        col = f"mortality_{h}h"
        if col in df_cohort.columns:
            h_rates.append(df_cohort[col].mean() * 100)
        else:
            h_rates.append(0.0)

    ax2.plot([f"{h}h" for h in horizons], h_rates, "o-", color="#0F2B46", lw=2.2, markersize=8)
    ax2.set_title("Cumulative Mortality by Prediction Horizon")
    ax2.set_ylabel("Cumulative Mortality Rate (%)")
    ax2.set_xlabel("Horizon Post-24h Observation")
    for i, txt in enumerate(h_rates):
        ax2.annotate(f"{txt:.1f}%", (i, h_rates[i] + 0.8), ha="center", fontweight="bold", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(fig_dir / "mortality_and_class_balance.png")
    plt.close(fig)

    # =========================================================================
    # PLOT 2: Age Distribution & Stratification by Outcome
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Age histogram
    survivors = df_cohort[df_cohort["mortality_48h"] == 0]["age"]
    non_survivors = df_cohort[df_cohort["mortality_48h"] == 1]["age"]

    ax1.hist([survivors, non_survivors], bins=15, stacked=True, color=["#3ECFB2", "#FF5F56"],
             label=["Survived", "Deteriorated"], edgecolor="white", alpha=0.85)
    ax1.set_title("Age Distribution by ICU Outcome")
    ax1.set_xlabel("Patient Age (Years)")
    ax1.set_ylabel("Number of Patients")
    ax1.legend()
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Mortality rate by ICU Type
    if "icu_type" in df_cohort.columns:
        icu_mort = df_cohort.groupby("icu_type")["mortality_48h"].agg(["mean", "count"]).reset_index()
        icu_mort = icu_mort.sort_values(by="mean", ascending=False)
        ax2.barh(icu_mort["icu_type"], icu_mort["mean"] * 100, color="#0F2B46", edgecolor="#3ECFB2", height=0.55)
        ax2.set_title("Mortality Rate by ICU Specialty Unit")
        ax2.set_xlabel("Mortality Rate (%)")
        for i, row in enumerate(icu_mort.itertuples()):
            ax2.text(row.mean * 100 + 0.5, i, f"{row.mean*100:.1f}% (N={row.count})", va="center", fontsize=8.5)
        ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(fig_dir / "age_and_icu_type_distribution.png")
    plt.close(fig)

    # =========================================================================
    # PLOT 3: Missingness Heatmap Across Physiological Streams
    # =========================================================================
    if not df_ts.empty:
        vital_cols = [c for c in FEATURE_COLUMNS if c in df_ts.columns]
        miss_df = df_ts[vital_cols].isna().mean().sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#3ECFB2" if v < 0.20 else ("#E29578" if v < 0.60 else "#FF5F56") for v in miss_df.values]
        y_pos = np.arange(len(miss_df))
        ax.barh(y_pos, miss_df.values * 100, color=colors, height=0.6, edgecolor="#0F2B46")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(miss_df.index)
        ax.set_xlabel("Missingness Rate in Hourly Records (%)")
        ax.set_title("Physiological & Laboratory Variable Missingness Profile")
        ax.set_xlim(0, 100)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        for i, val in enumerate(miss_df.values):
            ax.text(val * 100 + 1.2, i, f"{val*100:.1f}%", va="center", fontsize=8.5)

        plt.tight_layout()
        fig.savefig(fig_dir / "missingness_profile.png")
        plt.close(fig)

    # =========================================================================
    # PLOT 4: Temporal Trajectories (Survivors vs Non-Survivors)
    # =========================================================================
    if not df_ts.empty and "hour" in df_ts.columns:
        # Merge mortality status
        m_map = df_cohort.set_index("stay_id")["mortality_48h"].to_dict()
        df_ts_labeled = df_ts.copy()
        df_ts_labeled["mortality_48h"] = df_ts_labeled["stay_id"].map(m_map)

        key_vitals = ["heart_rate", "map", "resp_rate", "lactate"]
        available_vitals = [v for v in key_vitals if v in df_ts.columns]

        if available_vitals:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            axes = axes.flatten()

            for idx, var in enumerate(available_vitals):
                ax = axes[idx]
                surv_traj = df_ts_labeled[df_ts_labeled["mortality_48h"] == 0].groupby("hour")[var].mean()
                nonsurv_traj = df_ts_labeled[df_ts_labeled["mortality_48h"] == 1].groupby("hour")[var].mean()

                ax.plot(surv_traj.index, surv_traj.values, label="Survivors", color="#3ECFB2", lw=2.2)
                ax.plot(nonsurv_traj.index, nonsurv_traj.values, label="Non-Survivors", color="#FF5F56", lw=2.2, linestyle="--")
                ax.set_title(f"24h Mean Trajectory: {var.upper()}")
                ax.set_xlabel("Hours into ICU Stay")
                ax.set_ylabel(var)
                ax.legend(frameon=True, fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            fig.savefig(fig_dir / "temporal_trajectories_by_outcome.png")
            plt.close(fig)

    # =========================================================================
    # PLOT 5: Clinical Notes Length & Frequency Distribution
    # =========================================================================
    if not df_notes.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # Note count per stay
        notes_per_stay = df_notes.groupby("stay_id").size()
        ax1.hist(notes_per_stay.values, bins=10, color="#0F2B46", edgecolor="#3ECFB2", alpha=0.85)
        ax1.set_title("Clinical Notes Charted per 24h Stay")
        ax1.set_xlabel("Number of Notes per Patient")
        ax1.set_ylabel("Patient Count")
        ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

        # Note character length
        char_lens = df_notes["text"].dropna().str.len()
        ax2.hist(char_lens.values, bins=20, color="#E29578", edgecolor="#0F2B46", alpha=0.85)
        ax2.set_title("Note Character Length Distribution")
        ax2.set_xlabel("Character Count")
        ax2.set_ylabel("Note Count")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.savefig(fig_dir / "clinical_notes_distribution.png")
        plt.close(fig)

    # =========================================================================
    # TABLE: EDA Summary CSV
    # =========================================================================
    eda_summary_rows = []
    eda_summary_rows.append({"Metric": "Total Cohort ICU Stays", "Value": f"{len(df_cohort):,}"})
    eda_summary_rows.append({"Metric": "Unique Patients", "Value": f"{df_cohort['subject_id'].nunique():,}"})
    eda_summary_rows.append({"Metric": "48h In-Hospital Mortality Rate", "Value": f"{df_cohort['mortality_48h'].mean():.2%}"})
    eda_summary_rows.append({"Metric": "Mean Patient Age (Std)", "Value": f"{df_cohort['age'].mean():.1f} ± {df_cohort['age'].std():.1f} yrs"})
    if "gender" in df_cohort.columns:
        male_pct = (df_cohort["gender"].astype(str).str.upper() == "M").mean() * 100
        eda_summary_rows.append({"Metric": "Male Patient Fraction", "Value": f"{male_pct:.1f}%"})
    if not df_notes.empty:
        eda_summary_rows.append({"Metric": "Total 24h Notes Available", "Value": f"{len(df_notes):,}"})
        eda_summary_rows.append({"Metric": "Mean Notes per Stay", "Value": f"{len(df_notes) / len(df_cohort):.2f}"})

    df_summary = pd.DataFrame(eda_summary_rows)
    df_summary.to_csv(tables_dir / "eda_summary.csv", index=False)

    logger.info(f"EDA completed! Generated 5 figures in {fig_dir} and summary table in {tables_dir / 'eda_summary.csv'}.")


def main():
    parser = argparse.ArgumentParser(description="Run MEDGUARD AI Exploratory Data Analysis.")
    parser.add_argument("--proc-dir", type=str, default="data/processed")
    parser.add_argument("--synth-dir", type=str, default="data/synthetic")
    parser.add_argument("--fig-dir", type=str, default="reports/figures/eda")
    parser.add_argument("--tables-dir", type=str, default="reports/tables")
    args = parser.parse_args()

    root = get_project_root()
    run_eda(
        proc_dir=root / args.proc_dir,
        synth_dir=root / args.synth_dir,
        fig_dir=root / args.fig_dir,
        tables_dir=root / args.tables_dir,
    )


if __name__ == "__main__":
    main()
