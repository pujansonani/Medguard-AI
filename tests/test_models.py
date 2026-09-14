"""
Tests for Models (Tabular, Temporal Deep Learning, Clinical NLP, Multimodal Fusion).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from medguard.models.baselines import LogisticRegressionBaseline, XGBoostBaseline
from medguard.models.temporal_nn import TemporalGRUModel
from medguard.models.clinical_nlp import ClinicalNLPModel
from medguard.models.fusion import MedguardMultimodalModel


def test_logistic_regression_and_xgboost():
    X = pd.DataFrame(np.random.randn(50, 10), columns=[f"f_{i}" for i in range(10)])
    y = np.random.randint(0, 2, size=50)

    lr = LogisticRegressionBaseline()
    lr.fit(X, y)
    probs_lr = lr.predict_proba(X)
    assert len(probs_lr) == 50
    assert ((probs_lr >= 0.0) & (probs_lr <= 1.0)).all()

    xgb = XGBoostBaseline(n_estimators=10)
    xgb.fit(X, y)
    probs_xgb = xgb.predict_proba(X)
    assert len(probs_xgb) == 50


def test_temporal_gru_multi_horizon():
    X = np.random.randn(20, 24, 18).astype(np.float32)
    mask = np.ones((20, 24, 18), dtype=np.float32)
    y = np.random.randint(0, 2, size=(20, 4))

    gru = TemporalGRUModel(input_dim=18, hidden_dim=32, epochs=2, batch_size=10)
    gru.fit(X, mask, y)
    probs = gru.predict_proba(X, mask)

    assert probs.shape == (20, 4)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    embs = gru.get_embeddings(X, mask)
    assert embs.shape == (20, 32)


def test_clinical_nlp_model():
    texts = [
        "Patient in septic shock with hypotension and high lactate.",
        "Patient resting comfortably, vital signs stable, recovering post-op.",
    ] * 5
    y = np.random.randint(0, 2, size=(10, 4))

    nlp = ClinicalNLPModel(embedding_dim=64, epochs=2, batch_size=4)
    nlp.fit(texts, y)
    probs = nlp.predict_proba(texts)

    assert probs.shape == (10, 4)
    attrs = nlp.get_token_attributions(texts[0])
    assert len(attrs) > 0


def test_multimodal_gated_fusion_and_mc_dropout():
    X = np.random.randn(10, 24, 18).astype(np.float32)
    mask = np.ones((10, 24, 18), dtype=np.float32)
    texts = ["Patient critical with ARDS.", "Patient stable."] * 5
    y = np.random.randint(0, 2, size=(10, 4))

    fusion = MedguardMultimodalModel(fusion_type="gated", epochs=2, batch_size=4)
    fusion.fit(X, mask, texts, y)

    # Multi-horizon output
    probs = fusion.predict_proba(X, mask, texts)
    assert probs.shape == (10, 4)

    # MC-Dropout uncertainty
    mean_p, std_p = fusion.predict_with_uncertainty(X, mask, texts, num_mc_samples=5)
    assert mean_p.shape == (10, 4)
    assert std_p.shape == (10, 4)
    assert (std_p >= 0.0).all()

    # Modality weights
    weights = fusion.get_modality_weights(X, mask, texts)
    assert len(weights) == 10
    assert ((weights >= 0.0) & (weights <= 1.0)).all()
