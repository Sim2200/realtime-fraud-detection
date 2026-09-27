"""Vertex AI prediction server for fraud classification.

Loads model and metadata from cloud storage or disk; serves GET /health
and POST /predict endpoints per Vertex AI model serving contract.
"""

import json
import os

import numpy as np
from fastapi import FastAPI, Request
from google.cloud import storage
from joblib import load

import fraud  # noqa: F401

app = FastAPI()
_model = _meta = _features = _threshold = None


def _download_gcs(uri: str, filename: str) -> str:
    """Download file from GCS gs:// URI to /tmp."""
    bucket, path = uri.replace("gs://", "").split("/", 1)
    blob = storage.Client().bucket(bucket).blob(f"{path}/{filename}".lstrip("/"))
    local = f"/tmp/{filename}"
    blob.download_to_filename(local)
    return local


def load_model():
    """Load model and metadata (once, on the first request or at startup)."""
    global _model, _meta, _features, _threshold
    if _model is not None:
        return
    storage_uri = os.getenv("AIP_STORAGE_URI")
    model_dir = os.getenv("MODEL_DIR", "/tmp")
    if storage_uri:
        model_path = _download_gcs(storage_uri, "model.joblib")
        meta_path = _download_gcs(storage_uri, "model_meta.json")
    else:
        model_path = os.path.join(model_dir, "model.joblib")
        meta_path = os.path.join(model_dir, "model_meta.json")
    _model = load(model_path)
    with open(meta_path) as f:
        _meta = json.load(f)
    _features = _meta["features"]
    _threshold = _meta["threshold"]


app.router.on_startup.append(load_model)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint; loads the model if the startup hook has not run yet."""
    load_model()
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request) -> dict:
    """Prediction endpoint with instances (list of floats or dicts)."""
    load_model()
    instances = (await request.json()).get("instances", [])
    x_list = []
    for inst in instances:
        if isinstance(inst, dict):
            x = np.array([inst.get(f, 0.0) for f in _features], dtype=np.float32)
        else:
            x = np.array(inst, dtype=np.float32)
        x_list.append(x)
    x = np.array(x_list, dtype=np.float32)
    scores = _model.predict_proba(x)[:, 1]
    return {
        "predictions": [
            {"score": float(s), "fraud": bool(s >= _threshold)} for s in scores
        ]
    }
