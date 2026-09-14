"""
High-fidelity, clinically realistic synthetic ICU data generator.
Generates matched structured 24-hour time-series and unstructured clinical text notes
strictly within the 24-hour observation window for DEMO/UI testing and offline research validation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from medguard.utils.logging import get_logger

logger = get_logger("medguard.synthetic")

CLINICAL_NOTE_TEMPLATES_DETERIORATING = [
    (
        "Nursing Progress Note (Hour {hour}): Patient exhibiting progressive hemodynamic instability. "
        "Blood pressure refractory to fluid bolus 1000 mL NS. Norepinephrine infusion initiated at 0.08 mcg/kg/min "
        "and titrated up to 0.18 mcg/kg/min to maintain MAP > 65 mmHg. Heart rate elevated at {hr} bpm with sinus tachycardia. "
        "Lactate rising to {lactate} mmol/L. Urine output decreased to < 15 mL/hr over past 4 hours. "
        "Patient is lethargic, Glasgow Coma Scale assessed at {gcs}. Critical care fellow notified."
    ),
    (
        "Physician ICU Day 1 Assessment (Hour {hour}): Acute respiratory distress syndrome (ARDS) and septic shock secondary to severe pneumonia. "
        "Patient remains intubated and mechanically ventilated on PRVC mode. P/F ratio has dropped to {pf_ratio}. "
        "SpO2 fluctuating around {spo2}% despite high PEEP (14 cmH2O) and FiO2 70%. "
        "Laboratory results notable for worsening leukocytosis (WBC {wbc} K/uL) and acute kidney injury with creatinine {creatinine} mg/dL. "
        "Arterial blood gas reveals severe metabolic acidosis with bicarbonate {bicarb} mEq/L and base deficit -8. "
        "Plan: Continue broad-spectrum antibiotics (Vancomycin + Cefepime), stress-dose hydrocortisone, tight glycemic control, and maintain lung-protective ventilation."
    ),
    (
        "Critical Care Consultation Note (Hour {hour}): Multi-organ dysfunction syndrome (MODS). "
        "Cardiovascular: High-dose vasopressor support required with persistent hypotension (MAP {map_val} mmHg). "
        "Renal: Oliguric acute renal failure, BUN {bun} mg/dL, creatinine {creatinine} mg/dL, nephrology consulted for urgent CRRT initiation. "
        "Hematology: Thrombocytopenia with platelets declining to {platelets} K/uL. "
        "Neurology: Poor responsiveness, pupils equal and sluggish. Prognosis guarded. Family updated on critical status."
    )
]

CLINICAL_NOTE_TEMPLATES_STABLE = [
    (
        "Nursing Progress Note (Hour {hour}): Patient resting comfortably in bed. Hemodynamically stable without vasoactive support. "
        "Vitals stable: HR {hr} bpm in normal sinus rhythm, BP {sbp}/{dbp} mmHg (MAP {map_val} mmHg), SpO2 {spo2}% on 2L nasal cannula. "
        "Lungs clear to auscultation bilaterally with regular respiratory effort (RR {rr} breaths/min). "
        "Adequate urine output averaging 45-60 mL/hr. Alert, oriented x 4, following commands appropriately (GCS {gcs}). "
        "Pain adequately controlled on PRN oral acetaminophen."
    ),
    (
        "Physician Daily ICU Progress Note (Hour {hour}): Patient admitted for post-operative observation following elective surgery. "
        "Clinical condition improving steadily. Vital signs within normal limits. "
        "Labs demonstrate stable hemoglobin ({hgb} g/dL), normal renal function with creatinine {creatinine} mg/dL, and resolving lactate ({lactate} mmol/L). "
        "Electrolytes balanced: Potassium {potassium} mEq/L, Sodium {sodium} mEq/L. "
        "Plan: Advance diet as tolerated, encourage early ambulation, continue DVT prophylaxis, anticipate transfer to general medical floor within 24 hours."
    ),
    (
        "Multidisciplinary Team Note (Hour {hour}): Patient meets criteria for weaning and transfer out of the intensive care unit. "
        "Afebrile with temperature {temp}°C. Weaned off supplemental oxygen to room air, maintaining SpO2 {spo2}%. "
        "Tolerating oral intake. Surgical incision clean, intact, no signs of infection or erythema. "
        "Physical therapy completed bedside evaluation and cleared patient for floor-level care."
    )
]


class SyntheticDataGenerator:
    """Generates synthetic patient cohorts with correlated time-series and timestamped clinical notes."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_cohort(
        self,
        num_patients: int = 600,
        base_mortality_rate: float = 0.18,
        obs_window_hours: int = 24,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate matched patient-level metadata, hourly structured time-series, and timestamped notes.

        Returns:
            df_patients: Static demographic and multi-horizon target labels.
            df_timeseries: Hourly physiological observations (vitals + labs) for 24 steps per patient.
            df_notes: Clinical text notes timestamped within [0, 24] hours.
        """
        logger.info(f"Generating synthetic cohort of {num_patients} ICU patients (Seed={self.seed})...")
        
        patient_records = []
        ts_records = []
        note_records = []

        icu_types = ["MICU", "SICU", "CCU", "CVICU", "TSICU", "Neuro ICU"]
        icu_probs = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
        admission_types = ["EMERGENCY", "URGENT", "ELECTIVE"]
        admission_probs = [0.70, 0.20, 0.10]
        genders = ["M", "F"]

        for i in range(num_patients):
            subject_id = 10000 + i
            stay_id = 30000 + i
            
            # Demographics
            age = float(np.clip(self.rng.normal(64.0, 14.0), 18.0, 92.0))
            gender = self.rng.choice(genders, p=[0.55, 0.45])
            icu_type = self.rng.choice(icu_types, p=icu_probs)
            admission_type = self.rng.choice(admission_types, p=admission_probs)
            
            # Risk modulation
            age_risk = (age - 60.0) / 40.0 * 0.1
            icu_risk = 0.08 if icu_type in ["MICU", "SICU"] else 0.02
            emer_risk = 0.06 if admission_type == "EMERGENCY" else -0.05
            latent_risk = float(np.clip(base_mortality_rate + age_risk + icu_risk + emer_risk + self.rng.normal(0, 0.08), 0.02, 0.95))
            
            # Multi-horizon mortality outcomes (hierarchical: 6h <= 12h <= 24h <= 48h)
            m48 = int(self.rng.uniform(0, 1) < latent_risk)
            m24 = int(m48 and (self.rng.uniform(0, 1) < 0.70))
            m12 = int(m24 and (self.rng.uniform(0, 1) < 0.60))
            m6 = int(m12 and (self.rng.uniform(0, 1) < 0.45))
            
            los_hours = float(np.clip(self.rng.exponential(48.0) + 24.0, 24.0, 360.0))

            patient_records.append({
                "subject_id": subject_id,
                "stay_id": stay_id,
                "age": round(age, 1),
                "gender": gender,
                "icu_type": icu_type,
                "admission_type": admission_type,
                "los_hours": round(los_hours, 1),
                "latent_risk": round(latent_risk, 4),
                "mortality_6h": m6,
                "mortality_12h": m12,
                "mortality_24h": m24,
                "mortality_48h": m48,
            })

            # Generate 24 Hourly Observations
            # Base states
            is_deteriorating = (m48 == 1) or (latent_risk > 0.40)
            
            # Trajectory drifts
            drift_direction = 1.0 if is_deteriorating else -0.2
            
            # Initial vital values
            hr_base = self.rng.normal(92.0 if is_deteriorating else 76.0, 8.0)
            sbp_base = self.rng.normal(108.0 if is_deteriorating else 125.0, 10.0)
            dbp_base = self.rng.normal(64.0 if is_deteriorating else 78.0, 7.0)
            rr_base = self.rng.normal(22.0 if is_deteriorating else 16.0, 3.0)
            temp_base = self.rng.normal(38.2 if is_deteriorating else 37.0, 0.6)
            spo2_base = self.rng.normal(93.0 if is_deteriorating else 98.0, 2.0)
            gcs_base = self.rng.normal(12.0 if is_deteriorating else 15.0, 1.5)
            
            # Initial lab values
            lactate_base = self.rng.normal(3.2 if is_deteriorating else 1.2, 0.8)
            creat_base = self.rng.normal(1.8 if is_deteriorating else 0.9, 0.4)
            wbc_base = self.rng.normal(15.0 if is_deteriorating else 8.0, 3.0)
            plat_base = self.rng.normal(140.0 if is_deteriorating else 240.0, 40.0)
            hgb_base = self.rng.normal(10.5 if is_deteriorating else 13.5, 1.5)
            glucose_base = self.rng.normal(160.0 if is_deteriorating else 110.0, 25.0)
            na_base = self.rng.normal(138.0, 3.0)
            k_base = self.rng.normal(4.6 if is_deteriorating else 4.1, 0.4)
            bicarb_base = self.rng.normal(18.0 if is_deteriorating else 24.0, 3.0)
            bun_base = self.rng.normal(35.0 if is_deteriorating else 16.0, 8.0)

            # Store recent values for note rendering
            latest_vals = {}

            for h in range(obs_window_hours):
                progress = h / float(obs_window_hours)
                noise_scale = 0.5
                
                # Dynamic evolving vitals
                hr = hr_base + (drift_direction * 20.0 * progress) + self.rng.normal(0, 3.0)
                sbp = sbp_base - (drift_direction * 25.0 * progress) + self.rng.normal(0, 4.0)
                dbp = dbp_base - (drift_direction * 15.0 * progress) + self.rng.normal(0, 3.0)
                map_val = (sbp + 2.0 * dbp) / 3.0
                rr = rr_base + (drift_direction * 8.0 * progress) + self.rng.normal(0, 1.5)
                temp = temp_base + (drift_direction * 0.8 * progress) + self.rng.normal(0, 0.2)
                spo2 = spo2_base - (drift_direction * 6.0 * progress) + self.rng.normal(0, 1.0)
                gcs = gcs_base - (drift_direction * 4.0 * progress) + self.rng.normal(0, 0.5)

                # Labs (drawn every 4-6 hours or sparse)
                lactate = lactate_base + (drift_direction * 2.5 * progress) + self.rng.normal(0, 0.3)
                creat = creat_base + (drift_direction * 1.2 * progress) + self.rng.normal(0, 0.1)
                wbc = wbc_base + (drift_direction * 6.0 * progress) + self.rng.normal(0, 0.8)
                plat = plat_base - (drift_direction * 40.0 * progress) + self.rng.normal(0, 8.0)
                hgb = hgb_base - (drift_direction * 1.5 * progress) + self.rng.normal(0, 0.4)
                glucose = glucose_base + (drift_direction * 30.0 * progress) + self.rng.normal(0, 10.0)
                na = na_base + self.rng.normal(0, 1.5)
                k = k_base + (drift_direction * 0.6 * progress) + self.rng.normal(0, 0.2)
                bicarb = bicarb_base - (drift_direction * 5.0 * progress) + self.rng.normal(0, 1.0)
                bun = bun_base + (drift_direction * 15.0 * progress) + self.rng.normal(0, 2.0)

                # Clipping to valid biological ranges
                hr = float(np.clip(hr, 30.0, 220.0))
                sbp = float(np.clip(sbp, 40.0, 260.0))
                dbp = float(np.clip(dbp, 25.0, 160.0))
                map_val = float(np.clip(map_val, 30.0, 190.0))
                rr = float(np.clip(rr, 6.0, 55.0))
                temp = float(np.clip(temp, 33.0, 42.0))
                spo2 = float(np.clip(spo2, 60.0, 100.0))
                gcs = float(np.clip(gcs, 3.0, 15.0))
                
                lactate = float(np.clip(lactate, 0.3, 20.0))
                creat = float(np.clip(creat, 0.2, 18.0))
                wbc = float(np.clip(wbc, 0.8, 65.0))
                plat = float(np.clip(plat, 8.0, 850.0))
                hgb = float(np.clip(hgb, 4.0, 20.0))
                glucose = float(np.clip(glucose, 30.0, 650.0))
                na = float(np.clip(na, 110.0, 170.0))
                k = float(np.clip(k, 1.8, 8.5))
                bicarb = float(np.clip(bicarb, 6.0, 45.0))
                bun = float(np.clip(bun, 3.0, 140.0))

                # Inject realistic missingness in laboratory measurements (labs are not drawn every single hour)
                # Vitals are recorded nearly every hour (5% missing); Labs are drawn ~every 4-8h (60% missing in raw hourly bins)
                is_lab_hour = (h % 6 == 0) or (h == 0) or (h == obs_window_hours - 1)
                
                ts_records.append({
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "hour": h,
                    "heart_rate": round(hr, 1) if self.rng.uniform(0, 1) > 0.04 else np.nan,
                    "sbp": round(sbp, 1) if self.rng.uniform(0, 1) > 0.05 else np.nan,
                    "dbp": round(dbp, 1) if self.rng.uniform(0, 1) > 0.05 else np.nan,
                    "map": round(map_val, 1) if self.rng.uniform(0, 1) > 0.04 else np.nan,
                    "resp_rate": round(rr, 1) if self.rng.uniform(0, 1) > 0.04 else np.nan,
                    "temperature": round(temp, 2) if self.rng.uniform(0, 1) > 0.10 else np.nan,
                    "spo2": round(spo2, 1) if self.rng.uniform(0, 1) > 0.03 else np.nan,
                    "gcs": round(gcs, 1) if self.rng.uniform(0, 1) > 0.15 else np.nan,
                    
                    # Labs
                    "glucose": round(glucose, 1) if is_lab_hour or self.rng.uniform(0, 1) < 0.20 else np.nan,
                    "creatinine": round(creat, 2) if is_lab_hour else np.nan,
                    "hemoglobin": round(hgb, 1) if is_lab_hour else np.nan,
                    "wbc": round(wbc, 1) if is_lab_hour else np.nan,
                    "platelets": round(plat, 1) if is_lab_hour else np.nan,
                    "sodium": round(na, 1) if is_lab_hour else np.nan,
                    "potassium": round(k, 2) if is_lab_hour else np.nan,
                    "bicarbonate": round(bicarb, 1) if is_lab_hour else np.nan,
                    "lactate": round(lactate, 2) if (is_lab_hour or is_deteriorating) and self.rng.uniform(0, 1) > 0.20 else np.nan,
                    "bun": round(bun, 1) if is_lab_hour else np.nan,
                })

                latest_vals = {
                    "hr": int(hr),
                    "sbp": int(sbp),
                    "dbp": int(dbp),
                    "map_val": int(map_val),
                    "rr": int(rr),
                    "temp": round(temp, 1),
                    "spo2": int(spo2),
                    "gcs": int(gcs),
                    "lactate": round(lactate, 1),
                    "creatinine": round(creat, 2),
                    "wbc": round(wbc, 1),
                    "platelets": int(plat),
                    "hgb": round(hgb, 1),
                    "potassium": round(k, 1),
                    "sodium": int(na),
                    "bicarb": int(bicarb),
                    "bun": int(bun),
                    "pf_ratio": int(max(75, 280 - (120 * progress if is_deteriorating else -20))),
                }

            # Generate Timestamped Clinical Notes within [0, 24] hours (e.g. at hour 4, 12, 20)
            num_notes = self.rng.integers(2, 5)
            note_hours = sorted(self.rng.choice(range(1, obs_window_hours), size=num_notes, replace=False))
            
            templates = CLINICAL_NOTE_TEMPLATES_DETERIORATING if is_deteriorating else CLINICAL_NOTE_TEMPLATES_STABLE
            
            for note_idx, note_hr in enumerate(note_hours):
                template = templates[note_idx % len(templates)]
                note_text = template.format(hour=note_hr, **latest_vals)
                
                note_records.append({
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "chart_time_hour": float(note_hr),
                    "note_type": "Nursing" if note_idx == 0 else ("Physician" if note_idx == 1 else "Consultation"),
                    "text": note_text,
                })

        df_patients = pd.DataFrame(patient_records)
        df_timeseries = pd.DataFrame(ts_records)
        df_notes = pd.DataFrame(note_records)

        logger.info(
            f"Generated synthetic dataset: {len(df_patients)} patients, "
            f"{len(df_timeseries)} hourly observations, {len(df_notes)} clinical notes. "
            f"Mortality 48h rate: {df_patients['mortality_48h'].mean():.2%}"
        )
        return df_patients, df_timeseries, df_notes
