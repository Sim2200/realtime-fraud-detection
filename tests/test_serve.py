"""Tests for vertex.serve FastAPI prediction server."""

import json
import os
import tempfile

import numpy as np
import pytest

try:
    from fastapi.testclient import TestClient
    fastapi_available = True
except ImportError:
    fastapi_available = False

from sklearn.linear_model import LogisticRegression
from joblib import dump


@pytest.mark.skipif(not fastapi_available, reason="fastapi not installed")
def test_serve_health_and_predict():
    """Test /health and /predict endpoints with synthetic data."""
    # Create a temporary directory with a mock model
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate random 29-feature data and train a tiny logistic regression
        np.random.seed(42)
        n_samples = 100
        n_features = 29

        x = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.binomial(1, 0.2, n_samples)  # 20% positives

        model = LogisticRegression(random_state=0, max_iter=100)
        model.fit(x, y)

        # Save model and metadata
        model_path = os.path.join(tmpdir, "model.joblib")
        dump(model, model_path)

        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        meta = {
            "features": feature_names,
            "threshold": 0.5
        }
        meta_path = os.path.join(tmpdir, "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Set MODEL_DIR and import the app
        os.environ["MODEL_DIR"] = tmpdir
        from vertex.serve import app

        client = TestClient(app)

        # Test /health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # Test /predict endpoint with list of floats
        x_test = np.random.randn(2, 29).astype(np.float32)
        payload = {"instances": x_test.tolist()}

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        predictions = data["predictions"]
        assert len(predictions) == 2

        for pred in predictions:
            assert "score" in pred
            assert "fraud" in pred
            assert 0.0 <= pred["score"] <= 1.0
            assert isinstance(pred["fraud"], bool)

        # Test /predict endpoint with dict instances (feature names)
        x_dict = []
        for row in x_test:
            d = {feature_names[i]: float(row[i]) for i in range(29)}
            x_dict.append(d)

        payload = {"instances": x_dict}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert len(data["predictions"]) == 2
