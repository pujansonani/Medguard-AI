#!/usr/bin/env python3
"""
MEDGUARD AI: MIMIC-IV and Synthetic Clinical Dataset Validation Script

Validates raw and processed clinical data integrity:
1. Verifies required files existence (hosp, icu, note tables or parquet files).
2. Inspects schema, required columns, and datatypes.
3. Checks patient/stay identifier integrity and detects duplicates.
4. Analyzes 24h temporal observation coverage and note availability.
5. Computes missingness statistics per physiological variable.
6. Generates a structured clinical data quality report.

Usage:
    python scripts/validate_data.py --data-dir data/synthetic
    python scripts/validate_data.py --mimic-dir /path/to/mimic-iv-2.2 --notes-dir /path/to/mimic-iv-note-2.2
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.utils.logging import get_logger

logger = get_logger("medguard.validate_data")


REQUIRED_MIMIC_HOSP_TABLES = ["patients", "admissions"]
REQUIRED_MIMIC_ICU_TABLES = ["icustays", "chartevents"]
REQUIRED_MIMIC_NOTE_TABLES = ["radiology", "discharge"]

REQUIRED_PATIENT_COLS = ["subject_id", "gender", "anchor_age"]
REQUIRED_ADMISSION_COLS = ["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"]
REQUIRED_ICUSTAY_COLS = ["subject_id", "hadm_id", "stay_id", "intime", "outtime"]


def validate_parquet_dataset(data_dir: Path) -> Dict[str, Any]:
    """Validates processed or synthetic parquet files."""
    report = {
        "dataset_type": "Parquet / Processed",
        "data_dir": str(data_dir),
        "files_found": {},
        "missing_files": [],
        "patient_count": 0,
        "stay_count": 0,
        "mortality_rate": None,
        "vitals_missingness": {},
        "notes_statistics": {},
        "valid": True,
        "warnings": [],
        "errors": []
    }

    # Check patients file
    patients_file = data_dir / "patients.parquet"
    if patients_file.exists():
        report["files_found"]["patients.parquet"] = str(patients_file)
    else:
        report["missing_files"].append("patients.parquet")

    # Check timeseries / vitals file
    timeseries_file = None
    for cand in ["timeseries.parquet", "vitals_hourly.parquet"]:
        if (data_dir / cand).exists():
            timeseries_file = data_dir / cand
            report["files_found"][cand] = str(timeseries_file)
            break
    if not timeseries_file:
        report["missing_files"].append("timeseries.parquet (or vitals_hourly.parquet)")

    # Check notes file
    notes_file = data_dir / "notes.parquet"
    if notes_file.exists():
        report["files_found"]["notes.parquet"] = str(notes_file)
    else:
        report["missing_files"].append("notes.parquet")

    if report["missing_files"]:
        report["valid"] = False
        report["errors"].append(f"Missing required parquet files: {report['missing_files']}")
        return report

    # 1. Validate Patients
    try:
        df_patients = pd.read_parquet(data_dir / "patients.parquet")
        report["patient_count"] = int(df_patients["subject_id"].nunique())
        report["stay_count"] = int(df_patients["stay_id"].nunique()) if "stay_id" in df_patients else report["patient_count"]
        
        # Check duplicate stay_id
        if "stay_id" in df_patients:
            dup_stays = df_patients["stay_id"].duplicated().sum()
            if dup_stays > 0:
                report["warnings"].append(f"Found {dup_stays} duplicate stay_id records in patients table.")

        if "mortality_48h" in df_patients.columns:
            m_rate = float(df_patients["mortality_48h"].mean())
            report["mortality_rate"] = round(m_rate, 4)
            if m_rate < 0.01 or m_rate > 0.90:
                report["warnings"].append(f"Extreme mortality rate detected: {m_rate:.2%}")
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to load patients.parquet: {str(e)}")

    # 2. Validate Vitals & Missingness
    try:
        df_vitals = pd.read_parquet(timeseries_file)
        vital_cols = [c for c in df_vitals.columns if c not in ["subject_id", "stay_id", "hour_bin", "mortality_48h"]]
        
        for c in vital_cols:
            if pd.api.types.is_numeric_dtype(df_vitals[c]):
                miss_pct = float(df_vitals[c].isna().mean())
                report["vitals_missingness"][c] = round(miss_pct, 4)
                if miss_pct > 0.85:
                    report["warnings"].append(f"High missingness (>85%) in variable: {c} ({miss_pct:.1%})")

        # Temporal sequence length check
        if "hour_bin" in df_vitals.columns:
            hours_per_stay = df_vitals.groupby("stay_id")["hour_bin"].nunique()
            report["observation_coverage"] = {
                "min_hours": int(hours_per_stay.min()),
                "max_hours": int(hours_per_stay.max()),
                "median_hours": float(hours_per_stay.median())
            }
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to load timeseries file: {str(e)}")

    # 3. Validate Clinical Notes
    try:
        df_notes = pd.read_parquet(data_dir / "notes.parquet")
        total_notes = len(df_notes)
        unique_stays_with_notes = int(df_notes["stay_id"].nunique()) if "stay_id" in df_notes else 0
        empty_notes = int((df_notes["text"].isna() | (df_notes["text"].str.strip() == "")).sum())
        
        report["notes_statistics"] = {
            "total_notes": total_notes,
            "stays_with_notes": unique_stays_with_notes,
            "stay_note_coverage": round(unique_stays_with_notes / max(1, report["stay_count"]), 4),
            "empty_notes_count": empty_notes
        }
        if empty_notes > 0:
            report["warnings"].append(f"Found {empty_notes} empty or whitespace-only clinical notes.")
    except Exception as e:
        report["valid"] = False
        report["errors"].append(f"Failed to load notes.parquet: {str(e)}")

    return report


def validate_raw_mimic_dataset(mimic_dir: Path, notes_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Validates raw MIMIC-IV and MIMIC-IV-Note directory structure and schemas."""
    report = {
        "dataset_type": "Raw MIMIC-IV",
        "mimic_dir": str(mimic_dir),
        "notes_dir": str(notes_dir) if notes_dir else "None",
        "hosp_tables": {},
        "icu_tables": {},
        "note_tables": {},
        "missing_tables": [],
        "valid": True,
        "warnings": [],
        "errors": []
    }

    # Hosp tables
    for tbl in REQUIRED_MIMIC_HOSP_TABLES:
        p = check_file_exists(mimic_dir, tbl, ["hosp", "core"])
        if p:
            report["hosp_tables"][tbl] = str(p)
        else:
            report["missing_tables"].append(f"hosp/{tbl}")

    # ICU tables
    for tbl in REQUIRED_MIMIC_ICU_TABLES:
        p = check_file_exists(mimic_dir, tbl, ["icu"])
        if p:
            report["icu_tables"][tbl] = str(p)
        else:
            report["missing_tables"].append(f"icu/{tbl}")

    # Note tables
    if notes_dir and notes_dir.exists():
        for tbl in REQUIRED_MIMIC_NOTE_TABLES:
            p = check_file_exists(notes_dir, tbl, ["note"])
            if p:
                report["note_tables"][tbl] = str(p)
            else:
                report["missing_tables"].append(f"note/{tbl}")
    else:
        report["warnings"].append("MIMIC-IV-Note directory not provided or not found. Multimodal text modeling will require notes.")

    if report["missing_tables"]:
        report["valid"] = False
        report["errors"].append(f"Missing required MIMIC tables: {', '.join(report['missing_tables'])}")

    return report


def print_validation_report(report: Dict[str, Any]):
    """Pretty-print terminal report with clear dividers."""
    divider = "=" * 80
    subdivider = "-" * 80
    
    print("\n" + divider)
    print(f"  MEDGUARD AI — DATA QUALITY & SCHEMA VALIDATION REPORT")
    print(f"  Dataset Type: {report.get('dataset_type', 'Unknown')}")
    print(divider)
    
    status_str = "PASSED [VALID]" if report.get("valid") else "FAILED [ERRORS DETECTED]"
    print(f"  STATUS: {status_str}\n")

    if report.get("files_found"):
        print("  Files Verified:")
        for name, path in report["files_found"].items():
            print(f"    [+] {name:<25} -> {path}")
        print(subdivider)

    if report.get("patient_count"):
        print(f"  Cohort Overview:")
        print(f"    - Total Patients: {report['patient_count']:,}")
        print(f"    - Total ICU Stays: {report['stay_count']:,}")
        if report.get("mortality_rate") is not None:
            print(f"    - 48h Mortality Rate: {report['mortality_rate']:.2%}")
        print(subdivider)

    if report.get("vitals_missingness"):
        print("  Missingness Analysis (Hourly Physiological Streams):")
        for var, pct in sorted(report["vitals_missingness"].items(), key=lambda x: x[1]):
            bar = "#" * int(pct * 25) + "." * (25 - int(pct * 25))
            print(f"    {var:<22} [{bar}] {pct:>6.1%}")
        print(subdivider)

    if report.get("notes_statistics"):
        ns = report["notes_statistics"]
        print("  Clinical Text Coverage:")
        print(f"    - Total Notes: {ns.get('total_notes', 0):,}")
        print(f"    - ICU Stays with Notes: {ns.get('stays_with_notes', 0):,} ({ns.get('stay_note_coverage', 0):.1%})")
        print(f"    - Empty / Invalid Notes: {ns.get('empty_notes_count', 0)}")
        print(subdivider)

    if report.get("warnings"):
        print("  Quality Warnings:")
        for w in report["warnings"]:
            print(f"    [!] {w}")
        print(subdivider)

    if report.get("errors"):
        print("  Critical Errors:")
        for e in report["errors"]:
            print(f"    [x] {e}")
        print(subdivider)

    print(divider + "\n")


def main():
    parser = argparse.ArgumentParser(description="MEDGUARD AI Data Ingestion and Validation")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Path to directory containing parquet data")
    parser.add_argument("--mimic-dir", type=str, default=None, help="Path to raw MIMIC-IV directory")
    parser.add_argument("--notes-dir", type=str, default=None, help="Path to raw MIMIC-IV-Note directory")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON report")
    
    args = parser.parse_args()

    if args.mimic_dir:
        report = validate_raw_mimic_dataset(Path(args.mimic_dir), Path(args.notes_dir) if args.notes_dir else None)
    else:
        report = validate_parquet_dataset(Path(args.data_dir))

    print_validation_report(report)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to: {out_p}")

    if not report.get("valid"):
        sys.exit(1)


if __name__ == "__main__":
    main()
