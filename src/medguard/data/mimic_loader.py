"""
MIMIC-IV and MIMIC-IV-Note Ingestion Pipeline.
Handles raw CSV/Parquet files from PhysioNet, extracts cohorts, temporal resamplings, and note aggregations.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from medguard.utils.logging import get_logger

logger = get_logger("medguard.mimic")


class MIMICDataLoader:
    """Ingests and maps raw MIMIC-IV (v2.2) data into standardized MEDGUARD clinical formats."""

    def __init__(
        self,
        raw_mimic_dir: str | Path,
        raw_notes_dir: Optional[str | Path] = None,
        obs_window_hours: float = 24.0,
        pred_horizon_hours: float = 48.0,
    ):
        self.raw_mimic_dir = Path(raw_mimic_dir)
        self.raw_notes_dir = Path(raw_notes_dir) if raw_notes_dir else None
        self.obs_window_hours = obs_window_hours
        self.pred_horizon_hours = pred_horizon_hours

    def check_availability(self) -> bool:
        """Check whether local MIMIC-IV files exist."""
        if not self.raw_mimic_dir.exists():
            return False
        # Check for core ICU stays file
        icustays_file = list(self.raw_mimic_dir.glob("**/icustays.*"))
        return len(icustays_file) > 0

    def load_cohort(self) -> pd.DataFrame:
        """
        Load patients, admissions, and icustays from MIMIC-IV.
        Computes ICU entry timestamp, LOS hours, and 48-hour post-observation mortality labels.
        """
        if not self.check_availability():
            raise FileNotFoundError(
                f"MIMIC-IV files not found at {self.raw_mimic_dir}. "
                "Please download MIMIC-IV from PhysioNet or use DEMO mode."
            )

        logger.info(f"Loading MIMIC-IV cohort from {self.raw_mimic_dir}...")
        
        # Load icustays
        icustays_path = next(self.raw_mimic_dir.glob("**/icustays.*"))
        df_icustays = pd.read_parquet(icustays_path) if icustays_path.suffix == ".parquet" else pd.read_csv(icustays_path)
        
        # Load patients for age / gender / deathtime
        patients_path = next(self.raw_mimic_dir.glob("**/patients.*"))
        df_patients = pd.read_parquet(patients_path) if patients_path.suffix == ".parquet" else pd.read_csv(patients_path)
        
        # Load admissions
        admissions_path = next(self.raw_mimic_dir.glob("**/admissions.*"))
        df_admissions = pd.read_parquet(admissions_path) if admissions_path.suffix == ".parquet" else pd.read_csv(admissions_path)

        # Standardize column naming
        df_icustays.columns = [c.lower() for c in df_icustays.columns]
        df_patients.columns = [c.lower() for c in df_patients.columns]
        df_admissions.columns = [c.lower() for c in df_admissions.columns]

        # Merge core tables
        merged = df_icustays.merge(df_patients, on="subject_id", how="inner")
        merged = merged.merge(df_admissions, on=["subject_id", "hadm_id"], how="inner", suffixes=("", "_adm"))

        merged["intime"] = pd.to_datetime(merged["intime"])
        merged["outtime"] = pd.to_datetime(merged["outtime"])
        merged["los_hours"] = (merged["outtime"] - merged["intime"]).dt.total_seconds() / 3600.0

        # Compute Age
        merged["anchor_age"] = merged.get("anchor_age", 60.0)
        merged["age"] = merged["anchor_age"]

        # Compute Target: Mortality within [24h, 72h] from ICU intime (48h after 24h observation window)
        deathtime = pd.to_datetime(merged.get("deathtime", merged.get("dod", pd.NaT)))
        death_hours_from_in = (deathtime - merged["intime"]).dt.total_seconds() / 3600.0
        
        merged["mortality_6h"] = ((death_hours_from_in > self.obs_window_hours) & (death_hours_from_in <= self.obs_window_hours + 6.0)).astype(int)
        merged["mortality_12h"] = ((death_hours_from_in > self.obs_window_hours) & (death_hours_from_in <= self.obs_window_hours + 12.0)).astype(int)
        merged["mortality_24h"] = ((death_hours_from_in > self.obs_window_hours) & (death_hours_from_in <= self.obs_window_hours + 24.0)).astype(int)
        merged["mortality_48h"] = ((death_hours_from_in > self.obs_window_hours) & (death_hours_from_in <= self.obs_window_hours + 48.0)).astype(int)

        merged["icu_type"] = merged.get("first_careunit", "MICU")
        merged["gender"] = merged.get("gender", "M")
        merged["admission_type"] = merged.get("admission_type", "EMERGENCY")

        return merged[[
            "subject_id", "stay_id", "hadm_id", "age", "gender", "icu_type",
            "admission_type", "intime", "outtime", "los_hours",
            "mortality_6h", "mortality_12h", "mortality_24h", "mortality_48h"
        ]]
