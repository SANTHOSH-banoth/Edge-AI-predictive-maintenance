"""
api.py
--------
Wraps PredictiveMaintenancePipeline (from predict.py) in a real HTTP API.
This is the step that turns the project from "a script you run and read
console output from" into "a service other software can actually call" --
the difference between a demo and something deployable.

Endpoints:
  GET  /health              -- liveness check, confirms models loaded
  POST /predict              -- score ONE 30x18 sensor window
  POST /predict/batch        -- score MULTIPLE windows in one call
  GET  /predict/test-engine/{engine_id}  -- convenience: score a real
                                             held-out CMAPSS test engine
                                             by ID (1-100), for demoing
                                             without needing your own
                                             sensor data on hand

Design choices worth being able to explain:
  - The pipeline (both ONNX session and PyTorch autoencoder) loads ONCE at
    startup, not per-request -- reloading a model on every call would add
    real latency and defeats the point of the sub-millisecond ONNX
    inference you already benchmarked in edge_deployment_validation.py.
  - Input validation via Pydantic catches malformed requests (wrong shape,
    wrong feature count) before they ever reach the model, with a clear
    error message instead of an opaque numpy broadcasting failure.
  - No authentication/rate-limiting here -- this is a demo-quality API,
    not a hardened production one. Worth saying that plainly if asked,
    rather than implying more security maturity than exists.

Install (if not already):
    pip install fastapi uvicorn

Run:
    uvicorn scripts.api:app --reload --port 8000

Then either open http://127.0.0.1:8000/docs for interactive Swagger UI,
or test from another terminal with curl / the test_api.py script.
"""

import os
import sys
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Reuse the actual pipeline class -- no duplicated model-loading or
# alert-decision logic between the API and the script version.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import PredictiveMaintenancePipeline, SEQ_DIR

app = FastAPI(
    title="Turbofan Predictive Maintenance API",
    description="Anomaly detection + RUL estimation + maintenance alert decision "
                 "for CMAPSS turbofan sensor data.",
    version="1.0.0",
)

# Loaded once at startup -- see design notes above.
pipeline: Optional[PredictiveMaintenancePipeline] = None
test_engines_cache: Optional[np.ndarray] = None


@app.on_event("startup")
def load_models():
    global pipeline, test_engines_cache
    pipeline = PredictiveMaintenancePipeline()
    test_path = os.path.join(SEQ_DIR, "X_test.npy")
    if os.path.exists(test_path):
        test_engines_cache = np.load(test_path).astype(np.float32)
    print(f"API ready. Test engines cached: "
          f"{0 if test_engines_cache is None else len(test_engines_cache)}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class SensorWindow(BaseModel):
    """A single 30-cycle x 18-feature sensor window, already scaled the
    same way build_sequences.py's MinMaxScaler scaled training data.
    Scaling is NOT done inside this API -- the caller is responsible for
    applying sequence_scaler.pkl first. Worth stating clearly rather than
    silently assuming raw sensor units would work."""
    window: List[List[float]] = Field(
        ...,
        description="30 timesteps x 18 features, pre-scaled to match training data.",
        min_length=30,
        max_length=30,
    )


class PredictionResponse(BaseModel):
    predicted_rul: float
    is_anomaly: bool
    reconstruction_error: float
    alert_level: str
    reason: str


class BatchPredictionRequest(BaseModel):
    windows: List[SensorWindow]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if pipeline is not None else "models not loaded",
        "test_engines_available": 0 if test_engines_cache is None else len(test_engines_cache),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SensorWindow):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    window = np.array(payload.window, dtype=np.float32)
    if window.shape != (30, 18):
        raise HTTPException(
            status_code=422,
            detail=f"Expected window shape (30, 18), got {window.shape}. "
                    "Check feature count matches the 18 features used in training "
                    "(see data/cmapss/sequences/feature_columns.txt).",
        )

    result = pipeline.predict(window)
    return result


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    results = []
    for i, item in enumerate(payload.windows):
        window = np.array(item.window, dtype=np.float32)
        if window.shape != (30, 18):
            raise HTTPException(
                status_code=422,
                detail=f"Window {i}: expected shape (30, 18), got {window.shape}.",
            )
        results.append(pipeline.predict(window))
    return results


@app.get("/predict/test-engine/{engine_id}", response_model=PredictionResponse)
def predict_test_engine(engine_id: int):
    """Convenience endpoint: score a real held-out CMAPSS test engine by ID
    (1-100), so the API can be demoed without needing your own sensor data
    on hand -- useful for quick manual testing or a live demo."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if test_engines_cache is None:
        raise HTTPException(status_code=404, detail="Test engine data not found on server.")
    if not (1 <= engine_id <= len(test_engines_cache)):
        raise HTTPException(
            status_code=404,
            detail=f"engine_id must be between 1 and {len(test_engines_cache)}.",
        )

    window = test_engines_cache[engine_id - 1]
    result = pipeline.predict(window)
    return result
