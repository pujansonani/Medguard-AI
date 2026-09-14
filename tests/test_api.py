"""
Integration tests for FastAPI endpoints.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "medguard-ai"


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "MEDGUARD-FUSION" in data["model_name"]
    assert len(data["features_monitored"]) == 18


def test_predict_multimodal_endpoint(client):
    # Construct synthetic 24-hour request payload
    timeseries = []
    for h in range(24):
        timeseries.append({
            "hour": h,
            "heart_rate": 85.0 + h * 0.5,
            "sbp": 120.0 - h * 1.0,
            "dbp": 75.0,
            "map": 80.0 - h * 0.5,
            "resp_rate": 18.0,
            "temperature": 37.2,
            "spo2": 96.0,
            "gcs": 14.0,
            "lactate": 1.2 + h * 0.1,
            "creatinine": 1.0,
        })

    payload = {
        "demographics": {
            "subject_id": 99999,
            "stay_id": 88888,
            "age": 67.0,
            "gender": "M",
            "icu_type": "MICU",
            "admission_type": "EMERGENCY",
        },
        "timeseries": timeseries,
        "clinical_notes": "Patient in septic shock, norepinephrine titrated up to maintain MAP. Rising lactate noted.",
        "num_mc_samples": 5,
    }

    response = client.post("/predict/multimodal", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "mortality_48h_risk" in res_data
    assert "risk_trajectory" in res_data
    assert "prob_48h" in res_data["risk_trajectory"]
    assert "epistemic_uncertainty" in res_data
    assert "modality_weights" in res_data


def test_explain_endpoint(client):
    timeseries = [{"hour": h, "heart_rate": 90.0, "map": 70.0, "lactate": 3.5} for h in range(24)]
    payload = {
        "demographics": {"subject_id": 99999, "age": 70.0, "gender": "F", "icu_type": "SICU"},
        "timeseries": timeseries,
        "clinical_notes": "Patient with refractory hypotension requiring vasopressor support.",
    }

    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    exp_data = response.json()
    assert "highlighted_notes_html" in exp_data
    assert "top_positive_features" in exp_data
