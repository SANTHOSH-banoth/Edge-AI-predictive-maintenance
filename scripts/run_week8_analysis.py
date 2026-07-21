"""
run_week8_analysis.py
-----------------------
Week 8 deliverable: model selection across deployment constraints,
ensembling XGBoost + LSTM + CNN, and a cost-based maintenance decision
layer — all run against real trained models and the real CMAPSS test set.

Run:
    python scripts/run_week8_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm_rul import LSTMRegressor
from cnn_rul import CNNRegressor
from model_selector import ModelRegistry
from ensemble import compare_ensembles
from maintenance_decision import ActionThresholds, CostModel, estimate_savings, summarize_savings

MODEL_DIR = Path("models")
SEQ_DIR = Path("data") / "cmapss" / "sequences"
DEVICE = torch.device("cpu")


def load_xgboost_predictions(rul_clip):
    df = pd.read_csv(MODEL_DIR / "test_set_evaluation.csv")
    df = df.sort_values("unit_number").reset_index(drop=True)
    # IMPORTANT: this CSV's true_RUL column is the RAW, uncapped value from
    # RUL_FD001.txt (evaluate_test.py only clips it inline for scoring, never
    # rewrites the saved column). y_test.npy (used by LSTM/CNN) WAS capped
    # at rul_clip in build_sequences.py. Clip here too, or engines with true
    # RUL > rul_clip will show a false "misalignment" that's really just a
    # units mismatch, not a row-order problem.
    true_rul = df["true_RUL"].clip(upper=rul_clip).to_numpy()
    return true_rul, df["predicted_RUL"].to_numpy()


def load_lstm_predictions(X_test):
    ckpt = torch.load(MODEL_DIR / "lstm_rul.pt", map_location=DEVICE, weights_only=False)
    model = LSTMRegressor(
        n_features=ckpt["n_features"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds_norm = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return preds_norm * ckpt["rul_scale"]


def load_cnn_predictions(X_test):
    ckpt = torch.load(MODEL_DIR / "cnn_rul.pt", map_location=DEVICE, weights_only=False)
    model = CNNRegressor(
        n_features=ckpt["n_features"],
        channels=ckpt["channels"],
        kernel_size=ckpt["kernel_size"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return preds


def main():
    with open(MODEL_DIR / "best_params.json") as f:
        best_params_meta = json.load(f)
    rul_clip = best_params_meta["rul_clip"]

    # ================================================================
    # 1. MODEL SELECTOR — using real evaluated numbers from Weeks 5-7
    # ================================================================
    print("=" * 60)
    print("1. MODEL SELECTOR")
    print("=" * 60)

    registry = ModelRegistry()
    registry.register("xgboost_onnx", rmse=13.993, size_kb=744.0, latency_ms=0.028, cmapss_score=320.4)
    registry.register("lstm_onnx", rmse=12.803, size_kb=218.3, latency_ms=0.457, cmapss_score=267.0)
    registry.register("cnn_onnx", rmse=18.063, size_kb=134.2, latency_ms=0.684, cmapss_score=649.7)

    print(registry.summary().to_string(index=False))
    for constraint in ["cloud", "edge", "realtime"]:
        best = registry.select(constraint)
        print(f"\nBest for '{constraint}': {best.name} "
              f"(rmse={best.rmse}, size_kb={best.size_kb}, latency_ms={best.latency_ms})")

    best_name, ranked = registry.select_weighted({"rmse": 0.6, "size_kb": 0.2, "latency_ms": 0.2})
    print(f"\nWeighted (60% accuracy / 20% size / 20% latency) winner: {best_name}")
    print(ranked[["name", "weighted_score"]].to_string(index=False))

    # ================================================================
    # 2. ENSEMBLE — real predictions from all 3 models on same test engines
    # ================================================================
    print("\n" + "=" * 60)
    print("2. ENSEMBLE (XGBoost + LSTM + CNN)")
    print("=" * 60)

    X_test = np.load(SEQ_DIR / "X_test.npy")
    y_test_seq = np.load(SEQ_DIR / "y_test.npy")

    xgb_true, xgb_preds = load_xgboost_predictions(rul_clip)
    lstm_preds = load_lstm_predictions(X_test)
    cnn_preds = load_cnn_predictions(X_test)

    max_diff = np.abs(xgb_true - y_test_seq).max()
    print(f"Alignment check (XGBoost true_RUL vs sequence y_test, both clipped at {rul_clip}): "
          f"max diff = {max_diff:.4f}")
    if max_diff > 1.0:
        print("WARNING: alignment mismatch detected — do not trust the ensemble "
              "results below until this is resolved.")
    else:
        print("Alignment OK — safe to combine predictions engine-for-engine.\n")

    y_true = xgb_true
    predictions = {"xgboost": xgb_preds, "lstm": lstm_preds, "cnn": cnn_preds}

    report = compare_ensembles(predictions, y_true)
    print(report.to_string(index=False))
    print("\nWeighted-avg weights:", report.attrs["weighted_avg_weights"])
    print("Stacked meta-learner coefficients:", report.attrs["stacked_coefs"])

    # ================================================================
    # 3. COST-BASED MAINTENANCE DECISION — using the best model/ensemble
    # ================================================================
    print("\n" + "=" * 60)
    print("3. MAINTENANCE DECISION LAYER")
    print("=" * 60)

    best_method = report.iloc[0]["method"]
    print(f"Using best method from ensemble comparison: {best_method}")

    if best_method == "xgboost":
        best_preds = xgb_preds
    elif best_method == "lstm":
        best_preds = lstm_preds
    elif best_method == "cnn":
        best_preds = cnn_preds
    elif best_method == "ensemble_simple_avg":
        best_preds = np.column_stack([xgb_preds, lstm_preds, cnn_preds]).mean(axis=1)
    else:
        from ensemble import weighted_average, stacked_ensemble
        if best_method == "ensemble_weighted_avg":
            best_preds, _ = weighted_average(predictions, y_true)
        else:
            best_preds, _ = stacked_ensemble(predictions, y_true)

    thresholds = ActionThresholds(critical=15, warning=30, watch=60)
    costs = CostModel()
    savings_df = estimate_savings(y_true, best_preds, thresholds, costs)
    summary = summarize_savings(savings_df)

    print(savings_df.head(10).to_string(index=False))
    print("\n--- Fleet-wide savings summary (100 test engines) ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    registry.summary().to_csv(MODEL_DIR / "week8_model_registry.csv", index=False)
    report.to_csv(MODEL_DIR / "week8_ensemble_comparison.csv", index=False)
    savings_df.to_csv(MODEL_DIR / "week8_maintenance_savings.csv", index=False)
    print("\nAll Week 8 reports saved to models/week8_*.csv")


if __name__ == "__main__":
    main()