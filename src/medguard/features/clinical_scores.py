"""
Clinical ICU Severity Scoring Systems (SOFA and SAPS II Proxies).
Used as established clinical baseline comparators in research evaluation.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


class ClinicalScoreCalculator:
    """
    Computes standard ICU severity scoring approximations from the 24-hour observation window:
    1. SOFA (Sequential Organ Failure Assessment): Respiration, Coagulation, Liver, Cardiovasc, CNS, Renal.
    2. SAPS II (Simplified Acute Physiology Score II).
    """

    @staticmethod
    def compute_sofa_proxy(
        map_min: float,
        gcs_min: float,
        platelets_min: float,
        creatinine_max: float,
        spo2_min: float,
    ) -> int:
        """
        Compute SOFA score proxy (0 to 24).
        Uses worst values observed during the 24h window.
        """
        sofa = 0

        # 1. Cardiovascular (MAP)
        if map_min < 70.0:
            sofa += 1
        if map_min < 65.0:
            sofa += 1  # proxy for mild/mod vasopressor
        if map_min < 55.0:
            sofa += 2  # proxy for high-dose vasopressor

        # 2. Neurological (GCS)
        if 13 <= gcs_min <= 14:
            sofa += 1
        elif 10 <= gcs_min <= 12:
            sofa += 2
        elif 6 <= gcs_min <= 9:
            sofa += 3
        elif gcs_min < 6:
            sofa += 4

        # 3. Coagulation (Platelets K/uL)
        if 100.0 <= platelets_min < 150.0:
            sofa += 1
        elif 50.0 <= platelets_min < 100.0:
            sofa += 2
        elif 20.0 <= platelets_min < 50.0:
            sofa += 3
        elif platelets_min < 20.0:
            sofa += 4

        # 4. Renal (Creatinine mg/dL)
        if 1.2 <= creatinine_max < 2.0:
            sofa += 1
        elif 2.0 <= creatinine_max < 3.5:
            sofa += 2
        elif 3.5 <= creatinine_max < 5.0:
            sofa += 3
        elif creatinine_max >= 5.0:
            sofa += 4

        # 5. Respiration (SpO2 / proxy for PaO2/FiO2)
        if 92.0 <= spo2_min < 95.0:
            sofa += 1
        elif 88.0 <= spo2_min < 92.0:
            sofa += 2
        elif spo2_min < 88.0:
            sofa += 3

        return min(24, sofa)

    @staticmethod
    def compute_saps_ii_proxy(
        age: float,
        hr_max: float,
        sbp_min: float,
        temp_max: float,
        gcs_min: float,
        bun_max: float,
        wbc_max: float,
        potassium_max: float,
        sodium_min: float,
        bicarbonate_min: float,
    ) -> int:
        """Compute SAPS II score proxy (0 to 163)."""
        score = 0

        # Age
        if age < 40:
            score += 0
        elif age < 60:
            score += 7
        elif age < 70:
            score += 12
        elif age < 75:
            score += 15
        elif age < 80:
            score += 16
        else:
            score += 18

        # Heart rate
        if hr_max >= 160:
            score += 11
        elif hr_max >= 120:
            score += 4
        elif hr_max < 40:
            score += 11

        # SBP
        if sbp_min < 70:
            score += 13
        elif sbp_min < 100:
            score += 5
        elif sbp_min >= 200:
            score += 2

        # Temperature
        if temp_max >= 39.0:
            score += 3

        # GCS
        if gcs_min < 6:
            score += 26
        elif gcs_min < 9:
            score += 13
        elif gcs_min < 11:
            score += 7
        elif gcs_min < 14:
            score += 5

        # BUN
        if bun_max >= 84:
            score += 10
        elif bun_max >= 28:
            score += 6

        # WBC
        if wbc_max >= 20:
            score += 3
        elif wbc_max < 1.0:
            score += 12

        # Potassium
        if potassium_max >= 5.0 or potassium_max < 3.0:
            score += 3

        # Sodium
        if sodium_min < 125 or sodium_min >= 145:
            score += 1

        # Bicarbonate
        if bicarbonate_min < 15:
            score += 6
        elif bicarbonate_min < 20:
            score += 3

        return score

    def calculate_scores_df(self, df_tabular: pd.DataFrame) -> pd.DataFrame:
        """Compute SOFA and SAPS II scores for an entire tabular feature matrix."""
        df_scores = pd.DataFrame(index=df_tabular.index)
        sofa_list = []
        saps_list = []

        for _, row in df_tabular.iterrows():
            sofa = self.compute_sofa_proxy(
                map_min=row.get("map_min", 85.0),
                gcs_min=row.get("gcs_min", 15.0),
                platelets_min=row.get("platelets_min", 220.0),
                creatinine_max=row.get("creatinine_max", 0.9),
                spo2_min=row.get("spo2_min", 98.0),
            )
            saps = self.compute_saps_ii_proxy(
                age=row.get("age", 60.0),
                hr_max=row.get("heart_rate_max", 80.0),
                sbp_min=row.get("sbp_min", 120.0),
                temp_max=row.get("temperature_max", 37.0),
                gcs_min=row.get("gcs_min", 15.0),
                bun_max=row.get("bun_max", 15.0),
                wbc_max=row.get("wbc_max", 7.5),
                potassium_max=row.get("potassium_max", 4.0),
                sodium_min=row.get("sodium_min", 140.0),
                bicarbonate_min=row.get("bicarbonate_min", 24.0),
            )
            sofa_list.append(sofa)
            saps_list.append(saps)

        df_scores["sofa_score"] = sofa_list
        df_scores["saps_ii_score"] = saps_list
        return df_scores
