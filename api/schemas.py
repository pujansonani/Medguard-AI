"""
Pydantic Data Validation Schemas for MEDGUARD AI REST API.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DemographicInput(BaseModel):
    subject_id: Optional[int] = Field(default=10001, description="Patient Identifier")
    stay_id: Optional[int] = Field(default=30001, description="ICU Stay Identifier")
    age: float = Field(default=64.0, ge=18.0, le=120.0, description="Patient Age")
    gender: str = Field(default="M", description="Patient Gender (M/F)")
    icu_type: str = Field(default="MICU", description="ICU Care Unit Type (MICU, SICU, CCU, etc.)")
    admission_type: str = Field(default="EMERGENCY", description="Admission Type")


class HourlyObservation(BaseModel):
    hour: int = Field(ge=0, le=23, description="Hour of ICU stay [0..23]")
    heart_rate: Optional[float] = Field(default=80.0, ge=20.0, le=250.0)
    sbp: Optional[float] = Field(default=120.0, ge=40.0, le=280.0)
    dbp: Optional[float] = Field(default=75.0, ge=20.0, le=180.0)
    map: Optional[float] = Field(default=85.0, ge=30.0, le=200.0)
    resp_rate: Optional[float] = Field(default=16.0, ge=4.0, le=60.0)
    temperature: Optional[float] = Field(default=37.0, ge=30.0, le=44.0)
    spo2: Optional[float] = Field(default=98.0, ge=50.0, le=100.0)
    gcs: Optional[float] = Field(default=15.0, ge=3.0, le=15.0)
    glucose: Optional[float] = Field(default=100.0, ge=20.0, le=800.0)
    creatinine: Optional[float] = Field(default=0.9, ge=0.1, le=25.0)
    hemoglobin: Optional[float] = Field(default=13.5, ge=3.0, le=25.0)
    wbc: Optional[float] = Field(default=7.5, ge=0.5, le=90.0)
    platelets: Optional[float] = Field(default=220.0, ge=5.0, le=1000.0)
    sodium: Optional[float] = Field(default=140.0, ge=100.0, le=180.0)
    potassium: Optional[float] = Field(default=4.0, ge=1.5, le=9.0)
    bicarbonate: Optional[float] = Field(default=24.0, ge=5.0, le=50.0)
    lactate: Optional[float] = Field(default=1.2, ge=0.2, le=25.0)
    bun: Optional[float] = Field(default=15.0, ge=2.0, le=150.0)


class PatientAnalysisRequest(BaseModel):
    demographics: DemographicInput = Field(default_factory=DemographicInput)
    timeseries: List[HourlyObservation] = Field(
        ...,
        description="24-hour sequence of hourly physiological observations [0..23]"
    )
    clinical_notes: str = Field(
        default="",
        description="Contemporaneous clinical text notes written strictly within the 24-hour window."
    )
    num_mc_samples: int = Field(default=30, ge=5, le=100, description="Monte Carlo dropout forward passes")


class MultiHorizonPrediction(BaseModel):
    prob_6h: float = Field(..., description="Mortality risk within 6 hours post-window")
    prob_12h: float = Field(..., description="Mortality risk within 12 hours post-window")
    prob_24h: float = Field(..., description="Mortality risk within 24 hours post-window")
    prob_48h: float = Field(..., description="Mortality risk within 48 hours post-window")


class PredictionResponse(BaseModel):
    patient_id: int
    stay_id: int
    mode: str = "DEMO"
    risk_category: str # "Low", "Moderate", "High"
    mortality_48h_risk: float
    calibrated_risk: float
    epistemic_uncertainty: float # standard deviation from MC Dropout
    confidence_score: float
    risk_trajectory: MultiHorizonPrediction
    modality_weights: Dict[str, float]
    sofa_proxy_score: int
    saps_ii_proxy_score: int
    disclaimer: str = (
        "RESEARCH PROTOTYPE ONLY: This system is not a medical device and must never be used "
        "for clinical decision-making, diagnosis, triage, or patient care."
    )


class ExplanationResponse(BaseModel):
    patient_id: int
    mortality_48h_risk: float
    top_positive_features: List[List[Union[str, float]]]
    top_negative_features: List[List[Union[str, float]]]
    shap_base_value: float
    hourly_feature_importance: List[Dict[str, Any]]
    highlighted_notes_html: str
    clinical_keywords_detected: List[str]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    fusion_type: str
    observation_window_hours: float
    prediction_horizon_hours: float
    calibration_temperature: float
    features_monitored: List[str]
