"""
Clinical text cleaning, de-identification tag removal, negation preservation, and temporal note aggregation.
"""

import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from medguard.utils.logging import get_logger

logger = get_logger("medguard.preprocessing.text")


class ClinicalTextPreprocessor:
    """
    Cleans and prepares unstructured clinical notes:
    1. Removes MIMIC bracketed de-identification masks (e.g. `[**2100-01-01**]`, `[**First Name**]`).
    2. Strictly filters notes within the observation window (charttime <= obs_window_hours).
    3. Excludes retrospective notes (e.g. Discharge Summaries).
    4. Preserves clinical negations and medical abbreviations.
    5. Chunks and aggregates notes per patient stay into structured text representations.
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 256,
        chunk_overlap: int = 32,
        max_total_chars: int = 4000,
        obs_window_hours: float = 24.0,
        exclude_discharge_summaries: bool = True,
    ):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.max_total_chars = max_total_chars
        self.obs_window_hours = obs_window_hours
        self.exclude_discharge_summaries = exclude_discharge_summaries

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Strip de-id artifacts and normalize whitespace while preserving clinical abbreviations and negations."""
        if not isinstance(raw_text, str) or not raw_text.strip():
            return "No clinical notes documented."

        # Replace MIMIC bracketed de-id tags [** ... **]
        text = re.sub(r"\[\*\*.*?\*\*\]", " [REDACTED] ", raw_text)
        
        # Remove extra punctuation artifacts and repeated symbols
        text = re.sub(r"_{2,}", " ", text)
        text = re.sub(r"-{3,}", " ", text)
        text = re.sub(r"={3,}", " ", text)
        
        # Normalize multiple spaces and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def filter_notes_for_stay(self, df_notes: pd.DataFrame, stay_id: Optional[int] = None, max_cutoff_hour: Optional[float] = None) -> pd.DataFrame:
        """
        Filter notes for a single stay or full dataset strictly within observation window
        and excluding retrospective discharge summaries.
        """
        df = df_notes.copy()
        cutoff = max_cutoff_hour if max_cutoff_hour is not None else self.obs_window_hours
        
        if stay_id is not None and "stay_id" in df.columns:
            df = df[df["stay_id"] == stay_id]

        # Filter 1: Observation window cutoff
        time_cols = [c for c in ["chart_time_hour", "charttime_hours", "charttime_hour", "hour", "hours_into_stay", "charttime"] if c in df.columns]
        if time_cols:
            t_col = time_cols[0]
            df = df[df[t_col] <= cutoff]

        # Filter 2: Exclude discharge summaries
        if self.exclude_discharge_summaries and "note_category" in df.columns:
            df = df[~df["note_category"].astype(str).str.lower().str.contains("discharge")]
        elif self.exclude_discharge_summaries and "category" in df.columns:
            df = df[~df["category"].astype(str).str.lower().str.contains("discharge")]

        return df

    def aggregate_patient_notes(
        self,
        df_notes: pd.DataFrame,
        stay_ids: List[int],
        max_cutoff_hour: Optional[float] = None,
    ) -> Dict[int, str]:
        """
        Aggregate all clinical notes occurring on or before max_cutoff_hour for each patient stay.
        Strictly enforces that no notes beyond observation window are incorporated.
        """
        cutoff = max_cutoff_hour if max_cutoff_hour is not None else self.obs_window_hours
        result: Dict[int, str] = {}
        
        if df_notes is None or df_notes.empty or "stay_id" not in df_notes.columns:
            return {sid: "No clinical notes documented." for sid in stay_ids}

        # Apply global filters first with appropriate cutoff
        valid_notes = self.filter_notes_for_stay(df_notes, max_cutoff_hour=cutoff)
        if valid_notes.empty or "stay_id" not in valid_notes.columns:
            return {sid: "No clinical notes documented." for sid in stay_ids}

        grouped = valid_notes.groupby("stay_id")

        for sid in stay_ids:
            if sid in grouped.groups:
                stay_notes = grouped.get_group(sid)
                
                # Sort chronologically if time column exists
                time_col = next((c for c in ["charttime_hours", "hour", "charttime"] if c in stay_notes.columns), None)
                if time_col:
                    stay_notes = stay_notes.sort_values(by=time_col)

                cleaned_texts = []
                for _, row in stay_notes.iterrows():
                    txt = self.clean_text(str(row.get("text", "")))
                    if txt and txt != "No clinical notes documented.":
                        category = row.get("note_category", row.get("category", "Clinical Note"))
                        cleaned_texts.append(f"[{category}] {txt}")

                if cleaned_texts:
                    full_text = " | ".join(cleaned_texts)
                    # Truncate to maximum character limit if excessively long
                    if len(full_text) > self.max_total_chars:
                        full_text = full_text[: self.max_total_chars] + "..."
                    result[sid] = full_text
                else:
                    result[sid] = "No clinical notes documented within 24h window."
            else:
                result[sid] = "No clinical notes documented within 24h window."

        return result
