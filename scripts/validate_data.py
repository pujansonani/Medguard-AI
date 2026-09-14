#!/usr/bin/env python3
"""
MEDGUARD AI: MIMIC-IV & Clinical Dataset Integrity & Schema Validator

Inspects local MIMIC-IV installation or processed/synthetic datasets:
1. Verifies required files and directory structure.
2. Reports file sizes, columns, and data types.
3. Checks patient/stay identifiers, duplicates, and timestamp ranges.
4. Computes observation coverage and physiological missingness rates.
5. Emits clear, actionable status reports (READY vs INCOMPLETE) without crashing.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --mimic-dir ./data/raw/mimiciv --notes-dir ./data/raw/mimiciv_note
    python scripts/validate_data.py --data-dir ./data/synthetic
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure src is in pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.data.schema import MIMICSchemaAdapter
from medguard.utils.config import get_project_root, load_yaml
from medguard.utils.logging import get_logger

logger = get_logger("medguard.validate_data")


REQUIRED_HOSP_TABLES = ["patients", "admissions", "labevents"]
REQUIRED_ICU_TABLES = ["icustays", "chartevents"]
REQUIRED_NOTE_TABLES = ["radiology"]

TABLE_REQUIRED_COLUMNS = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "hospital_expire_flag"],
    "icustays": ["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
    "chartevents": ["stay_id", "charttime", "itemid", "valuenum"],
    "labevents": ["subject_id", "charttime", "itemid", "valuenum"],
    "radiology": ["note_id", "subject_id", "charttime", "text"],
}


def format_bytes(size_bytes: int) -> str:
    """Format file size in human-readable units."""
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    return f"{size_bytes} B"


def validate_mimic_installation(
    mimic_dir: Path,
    notes_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validates raw MIMIC-IV and MIMIC-IV-Note directory structure, schemas, and statistics."""
    adapter = MIMICSchemaAdapter(mimic_iv_dir=mimic_dir, mimic_note_dir=notes_dir, config=config)
    
    report: Dict[str, Any] = {
        "dataset_type": "Raw MIMIC-IV (PhysioNet)",
        "mimic_dir": str(mimic_dir),
        "notes_dir": str(notes_dir) if notes_dir else str(mimic_dir),
        "status": "READY",
        "tables_found": {},
        "missing_tables": [],
        "missing_columns": {},
        "statistics": {},
        "warnings": [],
        "errors": [],
    }

    all_required = REQUIRED_HOSP_TABLES + REQUIRED_ICU_TABLES
    
    # 1. Check Table Files Existence & Sizes
    for tbl in all_required:
        p = adapter.locate_table(tbl)
        if p:
            file_size = p.stat().st_size
            report["tables_found"][tbl] = {
                "path": str(p),
                "filename": p.name,
                "size_bytes": file_size,
                "size_human": format_bytes(file_size),
            }
        else:
            report["missing_tables"].append(tbl)

    # Check note tables
    p_note = adapter.locate_table("radiology", ["note"]) or adapter.locate_table("discharge", ["note"])
    if p_note:
        report["tables_found"]["clinical_notes"] = {
            "path": str(p_note),
            "filename": p_note.name,
            "size_bytes": p_note.stat().st_size,
            "size_human": format_bytes(p_note.stat().st_size),
        }
    else:
        report["warnings"].append("Clinical notes table (radiology or progress notes) not found. Text modeling will require notes.")

    if report["missing_tables"]:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Missing required MIMIC-IV tables: {report['missing_tables']}")
        return report

    # 2. Schema and Column Inspection
    for tbl, info in report["tables_found"].items():
        if tbl == "clinical_notes":
            continue
        p = Path(info["path"])
        try:
            head_df = adapter.read_table_head(p, n_rows=10)
            cols_present = set(head_df.columns)
            req_cols = set(TABLE_REQUIRED_COLUMNS.get(tbl, []))
            missing = req_cols - cols_present
            if missing:
                report["missing_columns"][tbl] = list(missing)
                report["status"] = "INCOMPLETE"
                report["errors"].append(f"Table '{tbl}' missing required columns: {list(missing)}")
        except Exception as e:
            report["warnings"].append(f"Could not read header of table '{tbl}': {e}")

    # 3. Fast Demographic & Stay Statistics
    try:
        p_pts = Path(report["tables_found"]["patients"]["path"])
        df_pts = adapter.read_table_head(p_pts, n_rows=500_000) if p_pts.suffix == ".parquet" else pd.read_csv(p_pts, usecols=["subject_id"])
        report["statistics"]["patient_count"] = int(df_pts["subject_id"].nunique())
        report["statistics"]["patient_duplicates"] = int(df_pts["subject_id"].duplicated().sum())
    except Exception as e:
        report["statistics"]["patient_count"] = "N/A"

    try:
        p_icu = Path(report["tables_found"]["icustays"]["path"])
        df_icu = adapter.read_table_head(p_icu, n_rows=500_000) if p_icu.suffix == ".parquet" else pd.read_csv(p_icu, usecols=["subject_id", "stay_id", "intime", "outtime"])
        report["statistics"]["icu_stay_count"] = int(df_icu["stay_id"].nunique())
        df_icu["intime"] = pd.to_datetime(df_icu["intime"])
        df_icu["outtime"] = pd.to_datetime(df_icu["outtime"])
        report["statistics"]["intime_min"] = str(df_icu["intime"].min())
        report["statistics"]["intime_max"] = str(df_icu["intime"].max())
    except Exception as e:
        report["statistics"]["icu_stay_count"] = "N/A"

    return report


def validate_parquet_dataset(data_dir: Path) -> Dict[str, Any]:
    """Validates processed or synthetic parquet files."""
    report = {
        "dataset_type": "Parquet / Processed / Synthetic",
        "data_dir": str(data_dir),
        "status": "READY",
        "tables_found": {},
        "missing_tables": [],
        "patient_count": 0,
        "stay_count": 0,
        "mortality_rate": None,
        "vitals_missingness": {},
        "notes_statistics": {},
        "warnings": [],
        "errors": []
    }

    # Check patients file
    patients_file = data_dir / "patients.parquet"
    if patients_file.exists():
        report["tables_found"]["patients.parquet"] = {
            "path": str(patients_file),
            "size_human": format_bytes(patients_file.stat().st_size)
        }
    else:
        report["missing_tables"].append("patients.parquet")

    # Check timeseries file
    timeseries_file = None
    for cand in ["timeseries.parquet", "vitals_hourly.parquet"]:
        if (data_dir / cand).exists():
            timeseries_file = data_dir / cand
            report["tables_found"][cand] = {
                "path": str(timeseries_file),
                "size_human": format_bytes(timeseries_file.stat().st_size)
            }
            break
    if not timeseries_file:
        report["missing_tables"].append("timeseries.parquet")

    # Check notes file
    notes_file = data_dir / "notes.parquet"
    if notes_file.exists():
        report["tables_found"]["notes.parquet"] = {
            "path": str(notes_file),
            "size_human": format_bytes(notes_file.stat().st_size)
        }
    else:
        report["missing_tables"].append("notes.parquet")

    if report["missing_tables"]:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Missing required parquet files: {report['missing_tables']}")
        return report

    # 1. Validate Patients
    try:
        df_patients = pd.read_parquet(patients_file)
        report["patient_count"] = int(df_patients["subject_id"].nunique())
        report["stay_count"] = int(df_patients["stay_id"].nunique()) if "stay_id" in df_patients else report["patient_count"]
        
        if "stay_id" in df_patients:
            dup_stays = int(df_patients["stay_id"].duplicated().sum())
            if dup_stays > 0:
                report["warnings"].append(f"Found {dup_stays} duplicate stay_id records in patients table.")

        if "mortality_48h" in df_patients.columns:
            report["mortality_rate"] = round(float(df_patients["mortality_48h"].mean()), 4)
    except Exception as e:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Failed reading patients.parquet: {e}")

    # 2. Validate Vitals & Missingness
    try:
        df_vitals = pd.read_parquet(timeseries_file)
        vital_cols = [c for c in df_vitals.columns if c not in ["subject_id", "stay_id", "hour_bin", "hour", "mortality_48h"]]
        for c in vital_cols:
            if pd.api.types.is_numeric_dtype(df_vitals[c]):
                miss_pct = float(df_vitals[c].isna().mean())
                report["vitals_missingness"][c] = round(miss_pct, 4)
    except Exception as e:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Failed reading timeseries file: {e}")

    # 3. Validate Clinical Notes
    try:
        df_notes = pd.read_parquet(notes_file)
        total_notes = len(df_notes)
        unique_stays = int(df_notes["stay_id"].nunique()) if "stay_id" in df_notes else 0
        empty_notes = int((df_notes["text"].isna() | (df_notes["text"].str.strip() == "")).sum())
        report["notes_statistics"] = {
            "total_notes": total_notes,
            "stays_with_notes": unique_stays,
            "stay_note_coverage": round(unique_stays / max(1, report["stay_count"]), 4),
            "empty_notes_count": empty_notes,
        }
    except Exception as e:
        report["status"] = "INCOMPLETE"
        report["errors"].append(f"Failed reading notes.parquet: {e}")

    return report


def print_validation_report(report: Dict[str, Any]):
    """Pretty-print terminal validation report with clear indicators and actionable guidance."""
    divider = "=" * 80
    subdivider = "-" * 80
    
    print("\n" + divider)
    print(f"  MEDGUARD AI — CLINICAL DATASET VALIDATION REPORT")
    print(f"  Dataset Type: {report.get('dataset_type', 'Unknown')}")
    print(divider)
    
    status = report.get("status", "INCOMPLETE")
    if status == "READY":
        print(f"  Dataset Status: \033[92mREADY [VALIDATED]\033[0m\n")
    else:
        print(f"  Dataset Status: \033[91mINCOMPLETE [MISSING TABLES OR DATA]\033[0m\n")

    if report.get("tables_found"):
        print("  Verified Tables & Files:")
        for name, info in report["tables_found"].items():
            size_str = info.get("size_human", "")
            print(f"    [+] {name:<22} ✓  ({size_str:>9}) -> {info['path']}")
        print(subdivider)

    stats = report.get("statistics", {})
    if stats:
        print("  Cohort Demographics & Timeline:")
        if "patient_count" in stats:
            print(f"    - Unique Patients: {stats['patient_count']:,}")
        if "icu_stay_count" in stats:
            print(f"    - Total ICU Stays: {stats['icu_stay_count']:,}")
        if "intime_min" in stats and "intime_max" in stats:
            print(f"    - Observation Time Range: {stats['intime_min']} to {stats['intime_max']}")
        print(subdivider)

    if report.get("patient_count"):
        print(f"  Cohort Overview:")
        print(f"    - Total Patients: {report['patient_count']:,}")
        print(f"    - Total ICU Stays: {report['stay_count']:,}")
        if report.get("mortality_rate") is not None:
            print(f"    - 48h Mortality Rate: {report['mortality_rate']:.2%}")
        print(subdivider)

    if report.get("vitals_missingness"):
        print("  Missingness Analysis (18 Hourly Physiological Streams):")
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
        print("  Warnings:")
        for w in report["warnings"]:
            print(f"    [!] {w}")
        print(subdivider)

    if report.get("errors"):
        print("  Missing Required Elements:")
        for e in report["errors"]:
            print(f"    [x] {e}")
        print("\n  ACTION REQUIRED:")
        print("  Place authorized MIMIC-IV tables under: data/raw/mimiciv/")
        print("    - hosp/patients.csv[.gz]")
        print("    - hosp/admissions.csv[.gz]")
        print("    - hosp/labevents.csv[.gz]")
        print("    - icu/icustays.csv[.gz]")
        print("    - icu/chartevents.csv[.gz]")
        print("  And notes under: data/raw/mimiciv_note/")
        print("    - note/radiology.csv[.gz]")
        print("  Or run synthetic demo generator: python scripts/generate_synthetic_data.py")
        print(subdivider)

    print(divider + "\n")


def main():
    parser = argparse.ArgumentParser(description="MEDGUARD AI Data Ingestion and Validation")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to parquet directory (e.g. data/synthetic)")
    parser.add_argument("--mimic-dir", type=str, default=None, help="Path to raw MIMIC-IV directory")
    parser.add_argument("--notes-dir", type=str, default=None, help="Path to raw MIMIC-IV-Note directory")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save JSON report")
    
    args = parser.parse_args()

    root = get_project_root()
    data_cfg = {}
    cfg_p = root / args.config
    if cfg_p.exists():
        try:
            data_cfg = load_yaml(cfg_p)
        except Exception:
            pass

    mimic_path = Path(args.mimic_dir) if args.mimic_dir else root / data_cfg.get("data", {}).get("mimic_iv_path", "./data/raw/mimiciv")
    notes_path = Path(args.notes_dir) if args.notes_dir else root / data_cfg.get("data", {}).get("mimic_iv_note_path", "./data/raw/mimiciv_note")

    # If user passed explicit data-dir, use parquet validator
    if args.data_dir:
        report = validate_parquet_dataset(Path(args.data_dir))
    elif mimic_path.exists() and (list(mimic_path.glob("**/icustays.*")) or list(mimic_path.glob("**/patients.*"))):
        report = validate_mimic_installation(mimic_path, notes_path if notes_path.exists() else None, data_cfg)
    else:
        # Default to synthetic if raw MIMIC not yet placed
        synth_dir = root / data_cfg.get("data", {}).get("synthetic_path", "./data/synthetic")
        if synth_dir.exists() and (synth_dir / "patients.parquet").exists():
            report = validate_parquet_dataset(synth_dir)
            report["warnings"].append(
                "MIMIC-IV raw directory not found. Currently validating DEMO synthetic cohort. "
                "Place authorized MIMIC-IV files under data/raw/mimiciv/ for clinical research."
            )
        else:
            report = {
                "dataset_type": "MIMIC-IV / Synthetic",
                "status": "INCOMPLETE",
                "errors": [f"No dataset found at {mimic_path} or {synth_dir}"],
                "warnings": ["Please place MIMIC-IV files under data/raw/mimiciv/ or run python scripts/generate_synthetic_data.py"]
            }

    print_validation_report(report)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to: {out_p}")

    if report.get("status") == "INCOMPLETE":
        sys.exit(1)


if __name__ == "__main__":
    main()
