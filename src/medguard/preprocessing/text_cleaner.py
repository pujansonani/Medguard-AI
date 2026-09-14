"""
Clinical text cleaning, de-identification tag removal, and temporal note chunking/aggregation.
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
    2. Filters notes strictly within the 24-hour ICU observation window (timestamp <= 24.0).
    3. Aggregates and chunks notes per patient stay into structured text representations.
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 256,
        chunk_overlap: int = 32,
        max_total_chars: int = 4000,
    ):
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.max_total_chars = max_total_chars

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Strip de-id artifacts and normalize whitespace while preserving clinical abbreviations."""
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

    def aggregate_patient_notes(
        self,
        df_notes: pd.DataFrame,
        stay_ids: List[int],
        max_cutoff_hour: float = 24.0,
    ) -> Dict[int, str]:
        """
        Aggregate all clinical notes occurring on or before max_cutoff_hour for each patient stay.
        Strictly enforces that no notes beyond 24h are incorporated.
        """
        result: Dict[int, str] = {}
        
        if df_notes.empty:
            for s in stay_ids:
                result[s] = "No contemporaneous clinical notes available."
            return result

        # Filter notes <= cutoff hour
        filtered_notes = df_notes[df_notes["chart_time_hour"] <= max_cutoff_hour].copy()
        filtered_notes["cleaned_text"] = filtered_notes["text"].apply(self.clean_text)
        
        grouped = filtered_notes.groupby("stay_id")

        for stay_id in stay_ids:
            if stay_id not in grouped.groups:
                result[stay_id] = "No clinical notes documented within 24h observation window."
            else:
                stay_notes = grouped.get_group(stay_id).sort_values("chart_time_hour")
                # Combine notes with timestamp header
                combined_parts = []
                for _, row in stay_notes.iterrows():
                    hr_str = f"[Hour {row['chart_time_hour']:.0f} - {row.get('note_type', 'Clinical Note')}]: "
                    combined_parts.append(hr_str + row["cleaned_text"])
                
                full_patient_text = " | ".join(combined_parts)
                # Cap total characters to avoid out-of-memory while preserving early & middle signal
                if len(full_patient_text) > self.max_total_chars:
                    full_patient_text = full_patient_text[:self.max_total_chars] + "..."
                result[stay_id] = full_patient_text

        return result
