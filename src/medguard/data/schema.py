"""
MIMIC-IV Schema Adapter & Table Locator.
Supports standard PhysioNet folder hierarchies, compressed CSVs (.csv.gz), raw CSVs, and Parquet.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from medguard.utils.logging import get_logger

logger = get_logger("medguard.data.schema")


class MIMICSchemaAdapter:
    """
    Flexible loader and column mapper for MIMIC-IV (hosp, icu, note).
    Resolves filenames across different directory layouts and compression formats.
    """

    def __init__(
        self,
        mimic_iv_dir: Union[str, Path] = "./data/raw/mimiciv",
        mimic_note_dir: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.mimic_iv_dir = Path(mimic_iv_dir)
        self.mimic_note_dir = Path(mimic_note_dir) if mimic_note_dir else self.mimic_iv_dir
        self.config = config or {}
        self.item_id_map = self.config.get("item_id_map", {})
        self.schema_cols = self.config.get("schema_columns", {})

    def locate_table(self, table_name: str, subdirs: Optional[List[str]] = None) -> Optional[Path]:
        """Find table file supporting .csv, .csv.gz, and .parquet across candidate directories."""
        subdirs = subdirs or ["", "hosp", "icu", "core", "note"]
        extensions = [".csv.gz", ".csv", ".parquet"]
        
        # Check both main MIMIC dir and Note dir
        search_roots = [self.mimic_iv_dir]
        if self.mimic_note_dir and self.mimic_note_dir != self.mimic_iv_dir:
            search_roots.append(self.mimic_note_dir)

        for root in search_roots:
            if not root.exists():
                continue
            for sub in subdirs:
                target_dir = root / sub if sub else root
                if not target_dir.exists():
                    continue
                for ext in extensions:
                    candidate = target_dir / f"{table_name}{ext}"
                    if candidate.exists():
                        return candidate
                    # Also try recursive search if direct lookup missed
                    matches = list(target_dir.glob(f"**/{table_name}{ext}"))
                    if matches:
                        return matches[0]
        return None

    def read_table_head(self, table_path: Path, n_rows: int = 10) -> pd.DataFrame:
        """Efficiently read header or sample rows of a table."""
        if table_path.suffix == ".parquet":
            return pd.read_parquet(table_path).head(n_rows)
        return pd.read_csv(table_path, nrows=n_rows)

    def read_table_chunks(
        self,
        table_path: Path,
        usecols: Optional[List[str]] = None,
        chunksize: int = 250_000,
    ):
        """Iterate over large CSV/GZ tables in memory-safe chunks."""
        if table_path.suffix == ".parquet":
            df = pd.read_parquet(table_path, columns=usecols)
            # Yield in slices matching chunksize
            for i in range(0, len(df), chunksize):
                yield df.iloc[i : i + chunksize]
        else:
            for chunk in pd.read_csv(table_path, usecols=usecols, chunksize=chunksize, low_memory=False):
                yield chunk

    def map_vital_measurement(self, itemid: int, valuenum: float) -> Optional[Tuple[str, float]]:
        """Map raw MIMIC chartevent itemid and value to standard concept name and normalized unit."""
        if np.isnan(valuenum):
            return None

        # Temperature: Fahrenheit -> Celsius conversion
        if itemid == 223761: # Fahrenheit
            return "temperature", (valuenum - 32.0) * 5.0 / 9.0
        elif itemid == 223762: # Celsius
            return "temperature", valuenum

        # Heart Rate
        if itemid == 220045:
            if 20.0 <= valuenum <= 250.0:
                return "heart_rate", valuenum

        # SBP
        if itemid in [220179, 220050]:
            if 30.0 <= valuenum <= 300.0:
                return "sbp", valuenum

        # DBP
        if itemid in [220180, 220051]:
            if 20.0 <= valuenum <= 200.0:
                return "dbp", valuenum

        # MAP
        if itemid in [220181, 220052, 225312]:
            if 20.0 <= valuenum <= 220.0:
                return "map", valuenum

        # Resp Rate
        if itemid in [220210, 224690]:
            if 4.0 <= valuenum <= 70.0:
                return "resp_rate", valuenum

        # SpO2
        if itemid == 220277:
            if 40.0 <= valuenum <= 100.0:
                return "spo2", valuenum

        # GCS components
        if itemid in [220739, 223900, 223901]:
            return "gcs", valuenum

        # Glucose
        if itemid in [220621, 225664]:
            if 10.0 <= valuenum <= 1000.0:
                return "glucose", valuenum

        return None

    def map_lab_measurement(self, itemid: int, valuenum: float) -> Optional[Tuple[str, float]]:
        """Map raw MIMIC labevent itemid and value to standard concept name."""
        if np.isnan(valuenum):
            return None

        lab_id_map = {
            50912: ("creatinine", (0.1, 30.0)),
            51006: ("bun", (1.0, 250.0)),
            50983: ("sodium", (90.0, 190.0)),
            50824: ("sodium", (90.0, 190.0)),
            50971: ("potassium", (1.0, 10.0)),
            50822: ("potassium", (1.0, 10.0)),
            50902: ("chloride", (60.0, 150.0)),
            50806: ("chloride", (60.0, 150.0)),
            50882: ("bicarbonate", (2.0, 60.0)),
            50803: ("bicarbonate", (2.0, 60.0)),
            51222: ("hemoglobin", (2.0, 26.0)),
            50811: ("hemoglobin", (2.0, 26.0)),
            51301: ("wbc", (0.1, 150.0)),
            51300: ("wbc", (0.1, 150.0)),
            51265: ("platelets", (2.0, 1500.0)),
            50813: ("lactate", (0.1, 35.0)),
            50931: ("glucose", (10.0, 1200.0)),
            50809: ("glucose", (10.0, 1200.0)),
        }

        if itemid in lab_id_map:
            concept, (min_v, max_v) = lab_id_map[itemid]
            if min_v <= valuenum <= max_v:
                return concept, valuenum

        return None
