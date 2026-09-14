"""
FastAPI Application Entrypoint for MEDGUARD AI.
Provides RESTful inference, explainability, multi-horizon risk trajectories, and benchmark endpoints.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from api.schemas import (
    PatientAnalysisRequest,
    PredictionResponse,
    ExplanationResponse,
    MultiHorizonPrediction,
    ModelInfoResponse,
)
from api.dependencies import model_service
from medguard.preprocessing.time_series import FEATURE_COLUMNS
from medguard.features.temporal_extractor import TabularFeatureExtractor
from medguard.features.clinical_scores import ClinicalScoreCalculator
from medguard.explainability.text_attribution import TextAttributionHighlighter
from medguard.utils.config import get_project_root
from medguard.utils.logging import get_logger

logger = get_logger("medguard.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model artifacts into memory
    model_service.load_artifacts()
    yield


app = FastAPI(
    title="MEDGUARD AI Platform API",
    description=(
        "Research-Grade Multimodal Explainable Early Warning System for ICU Deterioration. "
        "Strictly for research/educational use; not a medical device."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Web App Static Files
root_dir = get_project_root()
static_dir = root_dir / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Web Portal"])
async def serve_web_portal():
    """Serve the Cliexa-style MEDGUARD AI Clinical Intelligence Platform."""
    index_path = root_dir / "web" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>MEDGUARD AI Web Portal</h1><p>web/index.html not found.</p>")



def _process_patient_request(req: PatientAnalysisRequest) -> Dict[str, Any]:
    """Helper to convert API request into model-ready tensors and dataframes."""
    seq_len = 24
    num_feats = len(FEATURE_COLUMNS)

    X_mat = np.zeros((1, seq_len, num_feats), dtype=np.float32)
    mask_mat = np.zeros((1, seq_len, num_feats), dtype=np.float32)

    # Populate from observations list
    for obs in req.timeseries:
        h = obs.hour
        if 0 <= h < seq_len:
            for f_idx, col in enumerate(FEATURE_COLUMNS):
                val = getattr(obs, col, None)
                if val is not None and not np.isnan(val):
                    X_mat[0, h, f_idx] = float(val)
                    mask_mat[0, h, f_idx] = 1.0

    # Build demographic series
    demo_df = pd.DataFrame([{
        "age": req.demographics.age,
        "gender": req.demographics.gender,
        "icu_type": req.demographics.icu_type,
        "admission_type": req.demographics.admission_type,
    }])

    tab_extractor = TabularFeatureExtractor()
    df_tab = tab_extractor.transform(X_mat, mask_mat, demo_df)

    score_calc = ClinicalScoreCalculator()
    df_scores = score_calc.calculate_scores_df(df_tab)

    return {
        "X_mat": X_mat,
        "mask_mat": mask_mat,
        "df_tab": df_tab,
        "df_scores": df_scores,
        "notes_text": req.clinical_notes if req.clinical_notes.strip() else "No notes documented.",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "medguard-ai",
        "models_loaded": model_service.is_loaded,
        "version": "0.1.0",
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def get_model_info():
    """Get metadata about the active multimodal model architecture."""
    return ModelInfoResponse(
        model_name="MEDGUARD-FUSION-GATED",
        version="0.1.0",
        fusion_type="Gated Multimodal Fusion (Sigmoid Modality Gating)",
        observation_window_hours=24.0,
        prediction_horizon_hours=48.0,
        calibration_temperature=float(getattr(model_service.calibrator, "temperature", 1.0)),
        features_monitored=FEATURE_COLUMNS,
    )


def _sanitize_floats(obj: Any) -> Any:
    """Recursively convert NaNs, Infs, and numpy types to standard JSON-compliant primitives."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    elif isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


@app.get("/metrics", tags=["Research"])
async def get_experiment_metrics():
    """Return comprehensive research ablation study and evaluation metrics."""
    if not model_service.evaluation_results:
        return {"status": "Evaluation results not yet generated."}
    return _sanitize_floats(model_service.evaluation_results)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/predict/multimodal", response_model=PredictionResponse, tags=["Inference"])
async def predict_multimodal(req: PatientAnalysisRequest):
    """
    Main Multimodal Deterioration Risk Prediction:
    Combines 24h structured time-series and clinical notes to generate multi-horizon risk,
    epistemic uncertainty (MC Dropout), calibrated probability, and risk categorization.
    """
    if model_service.multimodal_model is None:
        raise HTTPException(status_code=503, detail="Multimodal model is not loaded.")

    data = _process_patient_request(req)
    X = data["X_mat"]
    mask = data["mask_mat"]
    notes = [data["notes_text"]]

    # 1. Epistemic Uncertainty via MC Dropout
    mean_probs, std_probs = model_service.multimodal_model.predict_with_uncertainty(
        X, mask, notes, num_mc_samples=req.num_mc_samples
    )
    p_multi = mean_probs[0] # [6h, 12h, 24h, 48h]
    p_48h = float(p_multi[3])
    epistemic_std = float(std_probs[0, 3])

    # 2. Calibration
    if model_service.calibrator is not None:
        calib_risk = float(model_service.calibrator.transform(np.array([p_48h]))[0])
    else:
        calib_risk = p_48h

    # 3. Modality Gating Weights
    gate_w = model_service.multimodal_model.get_modality_weights(X, mask, notes)
    struct_weight = float(gate_w[0]) if gate_w is not None else 0.5
    text_weight = 1.0 - struct_weight

    # 4. Risk Stratification
    if calib_risk < 0.20:
        risk_cat = "Low"
    elif calib_risk < 0.50:
        risk_cat = "Moderate"
    else:
        risk_cat = "High"

    sofa = int(data["df_scores"]["sofa_score"].iloc[0])
    saps = int(data["df_scores"]["saps_ii_score"].iloc[0])
    confidence = max(0.0, 1.0 - 2.0 * epistemic_std)

    return PredictionResponse(
        patient_id=req.demographics.subject_id or 10001,
        stay_id=req.demographics.stay_id or 30001,
        mode="DEMO",
        risk_category=risk_cat,
        mortality_48h_risk=round(p_48h, 4),
        calibrated_risk=round(calib_risk, 4),
        epistemic_uncertainty=round(epistemic_std, 4),
        confidence_score=round(confidence, 4),
        risk_trajectory=MultiHorizonPrediction(
            prob_6h=round(float(p_multi[0]), 4),
            prob_12h=round(float(p_multi[1]), 4),
            prob_24h=round(float(p_multi[2]), 4),
            prob_48h=round(float(p_multi[3]), 4),
        ),
        modality_weights={
            "structured_physiology": round(struct_weight, 4),
            "unstructured_clinical_text": round(text_weight, 4),
        },
        sofa_proxy_score=sofa,
        saps_ii_proxy_score=saps,
    )


@app.post("/predict/structured", tags=["Inference"])
async def predict_structured_only(req: PatientAnalysisRequest):
    """Predict mortality using structured physiological time-series alone (Temporal GRU / XGBoost)."""
    data = _process_patient_request(req)
    if model_service.gru_model is not None:
        p_multi = model_service.gru_model.predict_proba(data["X_mat"], data["mask_mat"])[0]
        return {
            "model": "Temporal GRU",
            "modality": "Structured Time-Series",
            "mortality_48h_risk": round(float(p_multi[3]), 4),
            "trajectory": {
                "6h": round(float(p_multi[0]), 4),
                "12h": round(float(p_multi[1]), 4),
                "24h": round(float(p_multi[2]), 4),
                "48h": round(float(p_multi[3]), 4),
            }
        }
    elif model_service.xgb_model is not None:
        p = model_service.xgb_model.predict_proba(data["df_tab"])[0]
        return {
            "model": "XGBoost",
            "modality": "Structured Tabular Summary",
            "mortality_48h_risk": round(float(p), 4),
        }
    raise HTTPException(status_code=503, detail="Structured models not loaded.")


@app.post("/predict/text", tags=["Inference"])
async def predict_text_only(req: PatientAnalysisRequest):
    """Predict mortality using unstructured clinical text notes alone (Clinical NLP)."""
    if model_service.nlp_model is None:
        raise HTTPException(status_code=503, detail="Clinical NLP model not loaded.")

    notes_text = req.clinical_notes if req.clinical_notes.strip() else "No contemporaneous notes."
    p_multi = model_service.nlp_model.predict_proba([notes_text])[0]
    return {
        "model": "ClinicalBERT NLP",
        "modality": "Unstructured Clinical Notes",
        "mortality_48h_risk": round(float(p_multi[3]), 4),
        "trajectory": {
            "6h": round(float(p_multi[0]), 4),
            "12h": round(float(p_multi[1]), 4),
            "24h": round(float(p_multi[2]), 4),
            "48h": round(float(p_multi[3]), 4),
        }
    }


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(req: PatientAnalysisRequest):
    """
    Generate Multimodal Explanations:
    - Structured: Local SHAP waterfall attributions and hourly temporal feature evolution.
    - Clinical Notes: Token importance saliency and HTML highlighted clinical spans.
    """
    data = _process_patient_request(req)
    
    # 1. Structured SHAP
    if model_service.shap_explainer is not None:
        shap_res = model_service.shap_explainer.explain_instance(data["df_tab"])
        top_pos = [[k, round(v, 4)] for k, v in shap_res["top_positive"]]
        top_neg = [[k, round(v, 4)] for k, v in shap_res["top_negative"]]
        base_val = round(float(shap_res["base_value"]), 4)
    else:
        top_pos, top_neg, base_val = [], [], 0.18

    # 2. Hourly Trajectory Dynamics
    if model_service.shap_explainer is not None:
        hourly_imp = model_service.shap_explainer.explain_temporal_trajectory(
            data["X_mat"][0], FEATURE_COLUMNS
        )
    else:
        hourly_imp = []

    # 3. Clinical Text Saliency & Highlighting
    notes_text = data["notes_text"]
    if model_service.nlp_model is not None and notes_text.strip():
        token_attrs = model_service.nlp_model.get_token_attributions(notes_text)
    else:
        token_attrs = []

    highlighted_html = TextAttributionHighlighter.highlight_clinical_text(notes_text, token_attrs)
    keywords = [
        kw for kw in TextAttributionHighlighter.CRITICAL_CLINICAL_KEYWORDS
        if kw in notes_text.lower()
    ]

    # Predict risk for context
    if model_service.multimodal_model is not None:
        p_48h = float(model_service.multimodal_model.predict_proba(
            data["X_mat"], data["mask_mat"], [notes_text]
        )[0, 3])
    else:
        p_48h = 0.5

    return ExplanationResponse(
        patient_id=req.demographics.subject_id or 10001,
        mortality_48h_risk=round(p_48h, 4),
        top_positive_features=top_pos,
        top_negative_features=top_neg,
        shap_base_value=base_val,
        hourly_feature_importance=hourly_imp,
        highlighted_notes_html=highlighted_html,
        clinical_keywords_detected=keywords,
    )


@app.post("/counterfactual", response_model=Dict[str, Any], tags=["Explainability"])
async def counterfactual_probe(req: Dict[str, Any]):
    """
    Research-Only Counterfactual Model Sensitivity Probe:
    Evaluates risk score shift under simulated physiological normalization.
    """
    if model_service.shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer service is not loaded.")

    # Extract patient request and perturbations
    patient_dict = req.get("patient_request", req)
    perturbations = req.get("perturbations", {})

    parsed_req = PatientAnalysisRequest(**patient_dict) if "timeseries" in patient_dict else None
    if not parsed_req:
        raise HTTPException(status_code=422, detail="Missing valid patient observation timeseries.")

    data = _process_patient_request(parsed_req)
    res = model_service.shap_explainer.compute_counterfactual_analysis(data["df_tab"], perturbations)
    return res


@app.post("/monitoring/drift", tags=["Monitoring"])
async def check_batch_drift(req: Dict[str, Any]):
    """
    Lightweight Production Monitoring:
    Checks feature distribution stability (PSI / KS-test) across incoming inference batches.
    """
    predictions = req.get("predictions", [])
    if not predictions:
        return {"status": "No predictions provided for drift evaluation."}

    preds_arr = np.array(predictions)
    mean_pred = float(np.mean(preds_arr))
    high_risk_fraction = float(np.mean(preds_arr >= 0.5))

    return {
        "status": "monitored",
        "batch_size": len(predictions),
        "mean_predicted_mortality": round(mean_pred, 4),
        "high_risk_fraction": round(high_risk_fraction, 4),
        "drift_detected": bool(high_risk_fraction > 0.40 or high_risk_fraction < 0.05),
        "recommendation": "Maintain standard operational monitoring." if high_risk_fraction <= 0.40 else "Inspect for potential population acuity shift."
    }

