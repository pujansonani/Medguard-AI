"""
MIMIC-IV and MIMIC-IV-Note Ingestion Pipeline.
Handles raw CSV/Parquet files from PhysioNet, extracts cohorts, temporal resamplings, and note aggregations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from medguard.data.schema import MIMICSchemaAdapter
from medguard.utils.logging import get_logger

logger = get_logger("medguard.mimic")


class MIMICDataLoader:
    """
    Ingests and maps raw MIMIC-IV (v2.2) and MIMIC-IV-Note data into standardized MEDGUARD clinical formats.
    Enforces strict 24-hour observation windows and patient-level isolation.
    """

    def __init__(
        self,
        raw_mimic_dir: Union[str, Path] = "./data/raw/mimiciv",
        raw_notes_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        obs_window_hours: float = 24.0,
        pred_horizon_hours: float = 48.0,
    ):
        self.raw_mimic_dir = Path(raw_mimic_dir)
        self.raw_notes_dir = Path(raw_notes_dir) if raw_notes_dir else self.raw_mimic_dir
        self.obs_window_hours = obs_window_hours
        self.pred_horizon_hours = pred_horizon_hours
        self.adapter = MIMICSchemaAdapter(
            mimic_iv_dir=self.raw_mimic_dir,
            mimic_note_dir=self.raw_notes_dir,
            config=config,
        )

    def check_availability(self) -> bool:
        """Check whether core MIMIC-IV files exist."""
        icustays_p = self.adapter.locate_table("icustays", ["icu"])
        patients_p = self.adapter.locate_table("patients", ["hosp", "core"])
        admissions_p = self.adapter.locate_table("admissions", ["hosp", "core"])
        return bool(icustays_p and patients_p and admissions_p)

    def load_cohort(
        self,
        min_age: float = 18.0,
        min_los_hours: float = 24.0,
        max_patients: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ingests admissions, patients, and icustays.
        Computes accurate patient age, observation timestamps, and mortality outcomes.
        Returns:
            df_cohort: DataFrame of eligible ICU stays
            cohort_report: Structured dictionary of waterfall exclusion statistics
        """
        if not self.check_availability():
            raise FileNotFoundError(
                f"Required MIMIC-IV tables not found under {self.raw_mimic_dir}. "
                "Expected tables: hosp/patients, hosp/admissions, icu/icustays."
            )

        # 1. Load ICU Stays
        icustays_p = self.adapter.locate_table("icustays", ["icu"])
        df_icu = pd.read_parquet(icustays_p) if icustays_p.suffix == ".parquet" else pd.read_csv(icustays_p)
        initial_icu_stays = len(df_icu)

        # Convert timestamps
        df_icu["intime"] = pd.to_datetime(df_icu["intime"])
        df_icu["outtime"] = pd.to_datetime(df_icu["outtime"])
        df_icu["los_hours"] = (df_icu["outtime"] - df_icu["intime"]).dt.total_seconds() / 3600.0

        # 2. Load Patients
        patients_p = self.adapter.locate_table("patients", ["hosp", "core"])
        df_pts = pd.read_parquet(patients_p) if patients_p.suffix == ".parquet" else pd.read_csv(patients_p)
        df_pts["dod"] = pd.to_datetime(df_pts["dod"]) if "dod" in df_pts.columns else pd.NaT

        # 3. Load Admissions
        admissions_p = self.adapter.locate_table("admissions", ["hosp", "core"])
        df_adm = pd.read_parquet(admissions_p) if admissions_p.suffix == ".parquet" else pd.read_csv(admissions_p)
        df_adm["admittime"] = pd.to_datetime(df_adm["admittime"])
        df_adm["dischtime"] = pd.to_datetime(df_adm["dischtime"])
        df_adm["deathtime"] = pd.to_datetime(df_adm["deathtime"]) if "deathtime" in df_adm.columns else pd.NaT

        # 4. Merge Demographics & Admissions
        df = df_icu.merge(df_pts[["subject_id", "gender", "anchor_age", "anchor_year", "dod"]], on="subject_id", how="inner")
        df = df.merge(
            df_adm[["hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "hospital_expire_flag"]],
            on="hadm_id",
            how="inner",
        )

        # 5. Compute Adjusted Age at ICU Admission
        # In MIMIC-IV, age = anchor_age + (intime.year - anchor_year)
        df["age"] = df["anchor_age"] + (df["intime"].dt.year - df["anchor_year"])
        df["age"] = df["age"].clip(lower=18.0, upper=91.0) # MIMIC-IV de-identifies >89 as ~91

        # 6. Apply Inclusions / Exclusions Waterfall
        # Filter: Age >= min_age
        df_adult = df[df["age"] >= min_age].copy()
        age_excluded = len(df) - len(df_adult)

        # Filter: First ICU Stay per Patient
        df_first = df_adult.sort_values(by=["subject_id", "intime"]).groupby("subject_id").first().reset_index()
        repeat_stays_excluded = len(df_adult) - len(df_first)

        # Filter: Length of stay >= 24h
        df_cohort = df_first[df_first["los_hours"] >= min_los_hours].copy()
        los_excluded = len(df_first) - len(df_cohort)

        # 7. Compute 24h Prediction Timestamp & 48h Mortality Label
        df_cohort["prediction_timestamp"] = df_cohort["intime"] + pd.Timedelta(hours=self.obs_window_hours)
        horizon_end = df_cohort["prediction_timestamp"] + pd.Timedelta(hours=self.pred_horizon_hours)

        # Mortality is 1 if deathtime or dod falls within [prediction_timestamp, horizon_end]
        has_deathtime = df_cohort["deathtime"].notna() & (df_cohort["deathtime"] >= df_cohort["prediction_timestamp"]) & (df_cohort["deathtime"] <= horizon_end)
        has_dod = df_cohort["dod"].notna() & (df_cohort["dod"] >= df_cohort["prediction_timestamp"]) & (df_cohort["dod"] <= horizon_end)
        
        # Also flag hospital mortality if discharged deceased within horizon
        hosp_expire = (df_cohort["hospital_expire_flag"] == 1) & (df_cohort["dischtime"] <= horizon_end)
        
        df_cohort["mortality_48h"] = (has_deathtime | has_dod | hosp_expire).astype(int)

        # Multi-horizon labels
        for h in [6.0, 12.0, 24.0, 48.0]:
            h_end = df_cohort["prediction_timestamp"] + pd.Timedelta(hours=h)
            d_h = df_cohort["deathtime"].notna() & (df_cohort["deathtime"] <= h_end)
            dod_h = df_cohort["dod"].notna() & (df_cohort["dod"] <= h_end)
            exp_h = (df_cohort["hospital_expire_flag"] == 1) & (df_cohort["dischtime"] <= h_end)
            df_cohort[f"mortality_{int(h)}h"] = (d_h | dod_h | exp_h).astype(int)

        # Map care unit
        df_cohort["icu_type"] = df_cohort["first_careunit"].fillna("MICU")

        if max_patients and len(df_cohort) > max_patients:
            df_cohort = df_cohort.sample(n=max_patients, random_state=42).reset_index(drop=True)

        # Generate Waterfall Cohort Report
        report = {
            "initial_icu_stays": int(initial_icu_stays),
            "excluded_pediatric": int(age_excluded),
            "excluded_repeat_stays": int(repeat_stays_excluded),
            "excluded_los_under_24h": int(los_excluded),
            "final_cohort_stays": int(len(df_cohort)),
            "mortality_count": int(df_cohort["mortality_48h"].sum()),
            "survival_count": int(len(df_cohort) - df_cohort["mortality_48h"].sum()),
            "mortality_rate": round(float(df_cohort["mortality_48h"].mean()), 4),
            "age_mean": round(float(df_cohort["age"].mean()), 2),
            "age_std": round(float(df_cohort["age"].std()), 2),
            "icu_type_distribution": df_cohort["icu_type"].value_counts().to_dict(),
        }

        logger.info(
            f"MIMIC-IV Cohort Selected: {report['final_cohort_stays']:,} stays "
            f"({report['mortality_count']} deaths, {report['mortality_rate']:.2%} mortality rate)."
        )
        return df_cohort, report

    def extract_structured_timeseries(
        self,
        df_cohort: pd.DataFrame,
        chunksize: int = 250_000,
    ) -> pd.DataFrame:
        """
        Extracts 18 hourly physiological streams for the selected cohort stays within the 24h window.
        Uses memory-efficient chunked streaming over chartevents and labevents.
        """
        stay_info = df_cohort.set_index("stay_id")[["intime", "prediction_timestamp", "subject_id"]].to_dict(orient="index")
        valid_stay_ids = set(df_cohort["stay_id"].unique())
        valid_subject_ids = set(df_cohort["subject_id"].unique())

        records = []

        # 1. Process Chartevents (Vital Signs)
        chartevents_p = self.adapter.locate_table("chartevents", ["icu"])
        if chartevents_p:
            logger.info(f"Extracting vital signs from {chartevents_p} (chunksize={chunksize:,})...")
            cols = ["stay_id", "charttime", "itemid", "valuenum"]
            for chunk in self.adapter.read_table_chunks(chartevents_p, usecols=cols, chunksize=chunksize):
                # Filter chunk to target cohort
                filtered = chunk[chunk["stay_id"].isin(valid_stay_ids)].dropna(subset=["valuenum"])
                if filtered.empty:
                    continue

                filtered["charttime"] = pd.to_datetime(filtered["charttime"])

                for _, row in filtered.iterrows():
                    sid = row["stay_id"]
                    info = stay_info.get(sid)
                    if not info:
                        continue

                    t = row["charttime"]
                    intime = info["intime"]
                    pred_time = info["prediction_timestamp"]

                    # Enforce strict 24h observation window
                    if intime <= t < pred_time:
                        mapped = self.adapter.map_vital_measurement(int(row["itemid"]), float(row["valuenum"]))
                        if mapped:
                            concept, val = mapped
                            hour_bin = int((t - intime).total_seconds() / 3600.0)
                            hour_bin = max(0, min(23, hour_bin))
                            records.append({
                                "stay_id": sid,
                                "subject_id": info["subject_id"],
                                "hour": hour_bin,
                                "concept": concept,
                                "value": val,
                            })

        # 2. Process Labevents (Laboratory Biomarkers)
        labevents_p = self.adapter.locate_table("labevents", ["hosp"])
        if labevents_p:
            logger.info(f"Extracting laboratory biomarkers from {labevents_p}...")
            cols = ["subject_id", "charttime", "itemid", "valuenum"]
            for chunk in self.adapter.read_table_chunks(labevents_p, usecols=cols, chunksize=chunksize):
                filtered = chunk[chunk["subject_id"].isin(valid_subject_ids)].dropna(subset=["valuenum"])
                if filtered.empty:
                    continue

                filtered["charttime"] = pd.to_datetime(filtered["charttime"])

                for _, row in filtered.iterrows():
                    mapped = self.adapter.map_lab_measurement(int(row["itemid"]), float(row["valuenum"]))
                    if not mapped:
                        continue
                    concept, val = mapped
                    sub_id = row["subject_id"]
                    t = row["charttime"]

                    # Match with cohort stays for this subject
                    for sid in [k for k, v in stay_info.items() if v["subject_id"] == sub_id]:
                        info = stay_info[sid]
                        if info["intime"] <= t < info["prediction_timestamp"]:
                            hour_bin = int((t - info["intime"]).total_seconds() / 3600.0)
                            hour_bin = max(0, min(23, hour_bin))
                            records.append({
                                "stay_id": sid,
                                "subject_id": sub_id,
                                "hour": hour_bin,
                                "concept": concept,
                                "value": val,
                            })

        if not records:
            logger.warning("No physiological observations extracted. Returning empty dataframe.")
            return pd.DataFrame()

        # Pivot records to (stay_id, hour, [features])
        df_obs = pd.DataFrame(records)
        df_pivoted = (
            df_obs.groupby(["stay_id", "subject_id", "hour", "concept"])["value"]
            .mean()
            .unstack(level="concept")
            .reset_index()
        )
        return df_pivoted

    def extract_clinical_notes(
        self,
        df_cohort: pd.DataFrame,
        chunksize: int = 100_000,
    ) -> pd.DataFrame:
        """
        Extracts clinical notes recorded strictly within the 24-hour observation window.
        Strictly excludes discharge summaries.
        """
        valid_subject_ids = set(df_cohort["subject_id"].unique())
        stay_info = df_cohort.set_index("stay_id")[["intime", "prediction_timestamp", "subject_id"]].to_dict(orient="index")

        note_records = []
        
        # Check for radiology or other note tables
        for tbl in ["radiology", "discharge"]:
            p = self.adapter.locate_table(tbl, ["note"])
            if not p:
                continue

            # Strict exclusion of discharge summaries
            if "discharge" in tbl.lower():
                logger.info("Strictly skipping discharge summaries to prevent retrospective outcome leakage.")
                continue

            logger.info(f"Extracting clinical text from {p}...")
            cols = ["note_id", "subject_id", "charttime", "note_type", "text"]
            for chunk in self.adapter.read_table_chunks(p, usecols=cols, chunksize=chunksize):
                filtered = chunk[chunk["subject_id"].isin(valid_subject_ids)].dropna(subset=["text"])
                if filtered.empty:
                    continue

                filtered["charttime"] = pd.to_datetime(filtered["charttime"])

                for _, row in filtered.iterrows():
                    sub_id = row["subject_id"]
                    t = row["charttime"]

                    for sid in [k for k, v in stay_info.items() if v["subject_id"] == sub_id]:
                        info = stay_info[sid]
                        if info["intime"] <= t <= info["prediction_timestamp"]:
                            hour_into_stay = (t - info["intime"]).total_seconds() / 3600.0
                            note_records.append({
                                "stay_id": sid,
                                "subject_id": sub_id,
                                "note_id": row.get("note_id", "N/A"),
                                "charttime_hours": round(hour_into_stay, 2),
                                "note_category": row.get("note_type", "Radiology"),
                                "text": str(row["text"]),
                            })

        return pd.DataFrame(note_records)
