"""
train_model.py
---------------
Trains two models on the machine sensor dataset:

1. "Cloud model"  -> RandomForestClassifier (higher accuracy, heavier, runs on a server)
2. "Edge model"   -> Compact MLPClassifier (smallenough for edge deployment)

The edge model is then exported to ONNX format (the standard lightweight
runtime format used to deploy models on edge devices / microcontrollers /
mobile, e.g. via ONNX Runtime Mobile). We compare:
  - model file size
  - inference latency per sample
  - accuracy / F1 / recall (recall matters most: missing a failure is costly)

Outputs saved to /models:
  cloud_model.pkl, edge_model.pkl, edge_model.onnx, scaler.pkl,
  metrics.json, feature_columns.json

All params/metrics/artifacts also logged to MLflow (experiment:
"week1_classification_rf_vs_edge_mlp") so this run is comparable
side-by-side with every other model trained in this project.
"""

import json
import time
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

DATA_PATH = "data/machine_sensor_data_engineered.csv"
MODEL_DIR = "models"

mlflow.set_experiment("week1_classification_rf_vs_edge_mlp")

# ---------------- Load & feature engineer ----------------
df = pd.read_csv(DATA_PATH)

# Feature engineering: derived physical features (this is the kind of thing
# to talk about in interviews - domain knowledge -> features)
df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
df["Power_W"] = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)
df["Wear_Torque_Product"] = df["Tool_wear_min"] * df["Torque_Nm"]

type_encoder = LabelEncoder()
df["Type_encoded"] = type_encoder.fit_transform(df["Type"])

feature_cols = [
    "Type_encoded", "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min",
    "Temp_diff_K", "Power_W", "Wear_Torque_Product",
    "vib_rms", "vib_crest_factor", "vib_bpfo_energy", "vib_bpfi_energy", "vib_bsf_energy",
]
X = df[feature_cols].values
y = df["Machine_failure"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def evaluate(model, name):
    start = time.perf_counter()
    preds = model.predict(X_test_s)
    elapsed = time.perf_counter() - start
    per_sample_ms = (elapsed / len(X_test_s)) * 1000

    proba = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "avg_latency_ms_per_sample": round(per_sample_ms, 5),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


with mlflow.start_run(run_name="rf_vs_edge_mlp"):

    # ---------------- Model 1: Cloud model (RandomForest) ----------------
    cloud_model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    cloud_model.fit(X_train_s, y_train)

    # ---------------- Handle class imbalance for the edge model ----------------
    # The failure class is only ~8.6% of the data. A plain MLP trained on this
    # directly collapses to "always predict healthy" (high accuracy, zero recall
    # -- useless for maintenance, since missed failures are the costly mistake).
    # SMOTE oversamples the minority (failure) class synthetically so the edge
    # model actually learns the failure boundary.
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)
    print(f"Before SMOTE: {np.bincount(y_train)} | After SMOTE: {np.bincount(y_train_bal)}")

    # ---------------- Model 2: Edge model (compact MLP) ----------------
    # Small architecture on purpose -> fewer parameters -> smaller footprint,
    # suitable for microcontrollers / edge gateways (ESP32, Raspberry Pi, etc.)
    edge_model = MLPClassifier(
        hidden_layer_sizes=(16, 8), activation="relu", max_iter=1000,
        random_state=42, early_stopping=True, learning_rate_init=0.005
    )
    edge_model.fit(X_train_bal, y_train_bal)

    # ---- log hyperparameters now that both models are configured ----
    mlflow.log_params({
        "cloud_model_type": "RandomForest",
        "cloud_n_estimators": cloud_model.n_estimators,
        "cloud_max_depth": cloud_model.max_depth,
        "cloud_class_weight": "balanced",
        "edge_model_type": "MLP",
        "edge_hidden_layer_sizes": str(edge_model.hidden_layer_sizes),
        "edge_activation": edge_model.activation,
        "edge_learning_rate_init": edge_model.learning_rate_init,
        "smote_applied": True,
        "smote_random_state": 42,
        "n_features": len(feature_cols),
        "test_size": 0.2,
    })

    cloud_metrics = evaluate(cloud_model, "Cloud Model (RandomForest)")
    edge_metrics = evaluate(edge_model, "Edge Model (Compact MLP, sklearn)")

    # ---------------- Save sklearn artifacts ----------------
    joblib.dump(cloud_model, f"{MODEL_DIR}/cloud_model.pkl")
    joblib.dump(edge_model, f"{MODEL_DIR}/edge_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(type_encoder, f"{MODEL_DIR}/type_encoder.pkl")

    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # ---------------- Export edge model to ONNX ----------------
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_sklearn(edge_model, initial_types=initial_type)
    with open(f"{MODEL_DIR}/edge_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    # ---------------- Compare model sizes ----------------
    import os
    cloud_size_kb = os.path.getsize(f"{MODEL_DIR}/cloud_model.pkl") / 1024
    edge_pkl_size_kb = os.path.getsize(f"{MODEL_DIR}/edge_model.pkl") / 1024
    edge_onnx_size_kb = os.path.getsize(f"{MODEL_DIR}/edge_model.onnx") / 1024

    # ---------------- ONNX Runtime inference benchmark (the real "edge" runtime) ----------------
    import onnxruntime as ort
    sess = ort.InferenceSession(f"{MODEL_DIR}/edge_model.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    X_test_f32 = X_test_s.astype(np.float32)
    start = time.perf_counter()
    for i in range(len(X_test_f32)):
        _ = sess.run(None, {input_name: X_test_f32[i:i + 1]})
    onnx_elapsed = time.perf_counter() - start
    onnx_latency_ms = (onnx_elapsed / len(X_test_f32)) * 1000

    summary = {
        "cloud_model": {
            **cloud_metrics,
            "model_size_kb": round(cloud_size_kb, 2),
            "framework": "scikit-learn RandomForest (200 trees)"
        },
        "edge_model_sklearn": {
            **edge_metrics,
            "model_size_kb": round(edge_pkl_size_kb, 2),
            "framework": "scikit-learn MLP (16,8)"
        },
        "edge_model_onnx_runtime": {
            "model_size_kb": round(edge_onnx_size_kb, 2),
            "avg_latency_ms_per_sample": round(onnx_latency_ms, 5),
            "framework": "ONNX Runtime (edge-deployable format)",
            "size_reduction_vs_cloud_pct": round((1 - edge_onnx_size_kb / cloud_size_kb) * 100, 1),
            "speedup_vs_cloud_x": round(cloud_metrics["avg_latency_ms_per_sample"] / onnx_latency_ms, 2)
        }
    }

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- log every metric from the summary dict ----
    mlflow.log_metrics({
        "cloud_accuracy": summary["cloud_model"]["accuracy"],
        "cloud_precision": summary["cloud_model"]["precision"],
        "cloud_recall": summary["cloud_model"]["recall"],
        "cloud_f1": summary["cloud_model"]["f1"],
        "cloud_roc_auc": summary["cloud_model"]["roc_auc"],
        "cloud_model_size_kb": summary["cloud_model"]["model_size_kb"],
        "cloud_latency_ms": summary["cloud_model"]["avg_latency_ms_per_sample"],
        "edge_accuracy": summary["edge_model_sklearn"]["accuracy"],
        "edge_precision": summary["edge_model_sklearn"]["precision"],
        "edge_recall": summary["edge_model_sklearn"]["recall"],
        "edge_f1": summary["edge_model_sklearn"]["f1"],
        "edge_roc_auc": summary["edge_model_sklearn"]["roc_auc"],
        "edge_model_size_kb": summary["edge_model_sklearn"]["model_size_kb"],
        "edge_onnx_size_kb": summary["edge_model_onnx_runtime"]["model_size_kb"],
        "edge_onnx_latency_ms": summary["edge_model_onnx_runtime"]["avg_latency_ms_per_sample"],
        "onnx_size_reduction_pct": summary["edge_model_onnx_runtime"]["size_reduction_vs_cloud_pct"],
        "onnx_speedup_x": summary["edge_model_onnx_runtime"]["speedup_vs_cloud_x"],
    })

    # ---- log the saved artifacts ----
    mlflow.log_artifact(f"{MODEL_DIR}/cloud_model.pkl")
    mlflow.log_artifact(f"{MODEL_DIR}/edge_model.pkl")
    mlflow.log_artifact(f"{MODEL_DIR}/edge_model.onnx")
    mlflow.log_artifact(f"{MODEL_DIR}/scaler.pkl")
    mlflow.log_artifact(f"{MODEL_DIR}/feature_columns.json")
    mlflow.log_artifact(f"{MODEL_DIR}/metrics.json")

    print("\n\n=== EDGE DEPLOYMENT COMPARISON ===")
    print(f"Cloud model (RandomForest):  {cloud_size_kb:.1f} KB, {cloud_metrics['avg_latency_ms_per_sample']:.4f} ms/sample")
    print(f"Edge model (ONNX Runtime):   {edge_onnx_size_kb:.1f} KB, {onnx_latency_ms:.4f} ms/sample")
    print(f"Size reduction: {summary['edge_model_onnx_runtime']['size_reduction_vs_cloud_pct']}%")
    print(f"Speedup: {summary['edge_model_onnx_runtime']['speedup_vs_cloud_x']}x")
    print("\nAll artifacts saved to /models")
    print("Run also logged to MLflow (experiment: week1_classification_rf_vs_edge_mlp)")

