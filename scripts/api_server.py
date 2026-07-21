"""
api_server.py
----------------
Week 5: FastAPI serving endpoint. This turns predict.py from "a scriptyou
run manually" into an actual HTTP service -- send sensor data, get a
maintenance decision back. This is what "deploying a model" concretely
means: everything before this point (training, tuning, ONNX export) was
about PRODUCING a good model; this is about making it USABLE by something
else (a dashboard, another service, a mobile app) without needing Python
or your training code installed alongside it.

Reuses PredictiveMaintenancePipeline from predict.py directly -- the API
layer adds HTTP handling and input validation, it doesn't duplicate the
anomaly-check/RUL-estimate/alert-decision logic. That logic lives in
exactly one place (predict.py), which is what you want: if you improve
the pipeline later, both the CLI script and the API automatically benefit.

Endpoints:
  GET  /health                     -- liveness check
  GET  /model_info                 -- expected input shape, thresholds, etc.
  POST /predict                    -- send a 30-cycle sensor window, get a
                                       full RUL + anomaly pipeline result back
  POST /predict/batch              -- same as /predict but for a list of windows
  GET  /demo/engine/{engine_id}    -- convenience endpoint: runs prediction on
                                       one of the real held-out test engines
                                       (1-100), no request body needed -- makes
                                       it trivial to demo without a client
  POST /predict/failure_risk       -- classification task (AI4I): given
                                       engineered sensor features, returns a
                                       failure-risk probability. Routes between
                                       the cloud model (RandomForest, higher
                                       accuracy, heavier) and the edge model
                                       (compact MLP via ONNX Runtime, smaller/
                                       faster) based on a `device_constraint`
                                       flag in the request -- this is the
                                       Week 5 "model-selector logic" deliverable.
                                       Unlike the RUL/anomaly models above,
                                       THIS task actually has two deployable
                                       model variants trained for exactly this
                                       tradeoff (see train_model.py / Week 1).

Run:
    uvicorn scripts.api_server:app --reload --port 8000

Then either open http://localhost:8000/docs for interactive Swagger UI
(FastAPI generates this automatically -- try it, it's the easiest way to
demo an API without writing a client), or:

    curl http://localhost:8000/health
    curl http://localhost:8000/demo/engine/1
"""

import os
import sys
import json
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal

sys.path.insert(0, os.path.dirname(__file__))
from predict import PredictiveMaintenancePipeline, RUL_URGENT_THRESHOLD, RUL_WATCH_THRESHOLD

SEQ_DIR = os.path.join("data", "cmapss", "sequences")
MODEL_DIR = os.path.join("models")

app = FastAPI(
    title="Edge-AI Predictive Maintenance API",
    description="Serves RUL prediction + anomaly detection + alert decisions "
                 "for turbofan engine sensor data (backed by an ONNX-deployed LSTM "
                 "and a PyTorch autoencoder), plus a failure-risk classifier with "
                 "edge/cloud model-selector logic (AI4I dataset).",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, reused across requests -- loading an ONNX session
# and a PyTorch model per-request would be needlessly slow.
pipeline: PredictiveMaintenancePipeline | None = None

# ---------------------------------------------------------------------------
# Failure-risk classifier artifacts (Week 1 models, loaded once at startup)
# ---------------------------------------------------------------------------
cloud_model = None          # RandomForest (sklearn) -- higher accuracy, heavier
edge_session = None         # MLP via ONNX Runtime -- smaller, faster
scaler = None
failure_feature_cols = None


@app.on_event("startup")
def load_pipeline():
    global pipeline, cloud_model, edge_session, scaler, failure_feature_cols

    pipeline = PredictiveMaintenancePipeline()
    print("RUL/anomaly pipeline loaded and ready to serve.")

    # ---- load the failure-risk classifier artifacts (cloud + edge) ----
    cloud_model = joblib.load(os.path.join(MODEL_DIR, "cloud_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    edge_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "edge_model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        failure_feature_cols = json.load(f)

    print(f"Failure-risk models loaded (cloud=RandomForest, edge=ONNX MLP), "
          f"{len(failure_feature_cols)} expected features.")


class SensorWindow(BaseModel):
    """A 30-cycle window of already-engineered, already-scaled sensor
    features -- i.e. the same format produced by build_sequences.py /
    signal_features.py. The API's job is serving predictions, not
    reimplementing feature engineering; in a full production system this
    would sit behind a separate feature-pipeline service."""
    window: List[List[float]] = Field(
        ..., description="Shape (30, n_features) -- 30 cycles x engineered features, pre-scaled"
    )


class PredictionResponse(BaseModel):
    predicted_rul: float
    is_anomaly: bool
    reconstruction_error: float
    alert_level: str
    reason: str


class BatchPredictionRequest(BaseModel):
    windows: List[SensorWindow]


class FailureRiskRequest(BaseModel):
    """Engineered (but NOT yet scaled) features for the AI4I failure-risk
    classifier, in the exact order feature_columns.json specifies. The
    server applies scaler.pkl before inference -- both the cloud and edge
    models were trained on scaled features, so this must happen either way
    regardless of which model ends up serving the request.

    device_constraint controls model-selector routing:
      "cloud" -> RandomForest (higher accuracy/recall, ~3.9MB, ~0.03ms/sample)
      "edge"  -> MLP via ONNX Runtime (~3KB, comparable latency, small
                 accuracy/precision tradeoff -- see models/metrics.json)
    """
    features: List[float] = Field(
        ..., description="Engineered feature vector, order matching feature_columns.json"
    )
    device_constraint: Literal["cloud", "edge"] = Field(
        "cloud", description="Which deployment target to route this prediction to"
    )


class FailureRiskResponse(BaseModel):
    failure_probability: float
    predicted_failure: bool
    model_used: Literal["cloud", "edge"]
    model_framework: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "failure_risk_models_loaded": cloud_model is not None and edge_session is not None,
    }


@app.get("/model_info")
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    input_shape = pipeline.rul_session.get_inputs()[0].shape
    return {
        "rul_anomaly_pipeline": {
            "expected_window_shape": "(30, n_features)",
            "onnx_input_shape": str(input_shape),
            "anomaly_error_threshold": pipeline.anomaly_threshold,
            "alert_thresholds": {
                "urgent_rul_cycles": RUL_URGENT_THRESHOLD,
                "watch_rul_cycles": RUL_WATCH_THRESHOLD,
            },
            "models": {
                "rul_model": "LSTM (ONNX Runtime) -- RMSE 12.8, CMAPSS score 267",
                "anomaly_model": "LSTM Autoencoder (native PyTorch, unsupervised)",
            },
        },
        "failure_risk_classifier": {
            "expected_n_features": len(failure_feature_cols) if failure_feature_cols else None,
            "feature_order": failure_feature_cols,
            "device_options": {
                "cloud": "RandomForest (200 trees) -- higher precision/F1, ~3.9MB",
                "edge": "MLP (16,8) via ONNX Runtime -- ~3KB, small accuracy tradeoff",
            },
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SensorWindow):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    window = np.array(payload.window, dtype=np.float32)
    expected_features = pipeline.rul_session.get_inputs()[0].shape[-1]

    if window.shape != (30, expected_features):
        raise HTTPException(
            status_code=422,
            detail=f"Expected window shape (30, {expected_features}), got {window.shape}"
        )

    result = pipeline.predict(window)
    return result


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    expected_features = pipeline.rul_session.get_inputs()[0].shape[-1]
    results = []
    for i, item in enumerate(payload.windows):
        window = np.array(item.window, dtype=np.float32)
        if window.shape != (30, expected_features):
            raise HTTPException(
                status_code=422,
                detail=f"windows[{i}]: expected shape (30, {expected_features}), got {window.shape}"
            )
        results.append(pipeline.predict(window))
    return results


@app.get("/demo/engine/{engine_id}", response_model=PredictionResponse)
def demo_engine(engine_id: int):
    """Convenience endpoint: runs a prediction on one of the real held-out
    CMAPSS test engines (1-100) without needing to construct a requestbody
    by hand -- the easiest way to demo this API live."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    X_test_path = os.path.join(SEQ_DIR, "X_test.npy")
    if not os.path.exists(X_test_path):
        raise HTTPException(status_code=404, detail="Test sequences not found -- run build_sequences.py first")

    X_test = np.load(X_test_path).astype(np.float32)
    if not (1 <= engine_id <= len(X_test)):
        raise HTTPException(status_code=404, detail=f"engine_id must be between 1 and {len(X_test)}")

    window = X_test[engine_id - 1]
    result = pipeline.predict(window)
    return result


@app.post("/predict/failure_risk", response_model=FailureRiskResponse)
def predict_failure_risk(payload: FailureRiskRequest):
    """
    Model-selector logic (Week 5 deliverable): routes the same engineered
    feature vector to either the cloud model (RandomForest) or the edge
    model (MLP via ONNX Runtime), based on the caller's device_constraint.

    This mirrors a real deployment decision -- a device with plenty of
    compute/connectivity (a server, a gateway) can afford the larger, more
    accurate cloud model; a constrained device (microcontroller, offline
    sensor node) needs the edge model's much smaller footprint, at a small,
    measured accuracy cost (see models/metrics.json for the full tradeoff:
    cloud F1=0.777 vs edge F1=0.710, edge is ~220x smaller).
    """
    if cloud_model is None or edge_session is None or scaler is None:
        raise HTTPException(status_code=503, detail="Failure-risk models not loaded yet")

    if len(payload.features) != len(failure_feature_cols):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(failure_feature_cols)} features "
                   f"(order: {failure_feature_cols}), got {len(payload.features)}"
        )

    X = np.array(payload.features, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)

    if payload.device_constraint == "cloud":
        proba = float(cloud_model.predict_proba(X_scaled)[0, 1])
        framework = "scikit-learn RandomForest (200 trees)"
    else:
        input_name = edge_session.get_inputs()[0].name
        output_names = [o.name for o in edge_session.get_outputs()]
        outputs = edge_session.run(output_names, {input_name: X_scaled.astype(np.float32)})
        # skl2onnx classifiers typically output [label, probability_dict/array]
        # at index 1; handle both plain-array and list-of-dict ONNX outputs.
        raw_proba = outputs[1]
        if isinstance(raw_proba, list):
            proba = float(raw_proba[0][1])
        else:
            proba = float(raw_proba[0][1])
        framework = "MLP (16,8) via ONNX Runtime"

    return FailureRiskResponse(
        failure_probability=round(proba, 4),
        predicted_failure=proba >= 0.5,
        model_used=payload.device_constraint,
        model_framework=framework,
    )

