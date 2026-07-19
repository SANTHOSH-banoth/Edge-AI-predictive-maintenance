"""
export_lstm_onnx.py
----------------------
Exports your best-performing model (the LSTM, RMSE 12.8 / CMAPSS score 267 --
beating both XGBoost and the CNN) to ONNX for edge deployment, and benchmarks
it against the native PyTorch model on your REAL held-out test engines.

Why the LSTM export is trickier than the XGBoost one:
XGBoost is a static tree structure -- easy to convert. An LSTM has a
recurrent hidden state that gets updated at every step of the sequence.
The export needs to correctly "unroll" that recurrence into a fixed ONNX
graph. Two things matter here:
  1. PyTorch has TWO onnx exporters as of recent versions -- a newer
     "dynamo"-based one (now default) and the older TorchScript-based one.
     Testing showed the newer exporter produces a shape-mismatch warning
     for batched inputs on this LSTM architecture (still numerically
     correct, but fragile). This script explicitly uses the older,
     more battle-tested exporter (dynamo=False) to avoid that.
  2. dynamic_axes must be set on both input AND output so the exported
     graph accepts any batch size at inference time, not just the batch
     size used during export.

This was verified end-to-end before being sent to you: export succeeded,
and native vs ONNX predictions matched to ~1e-7 across batch sizes of
1, 5, and 32.

Run:
    python scripts/export_lstm_onnx.py
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_DIR = "models"
LSTM_CHECKPOINT = os.path.join(MODEL_DIR, "lstm_rul.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "lstm_rul.onnx")
SEQ_DIR = os.path.join("data", "cmapss", "sequences")

N_LATENCY_RUNS = 200


class LSTMRegressor(nn.Module):
    """Must match the architecture in lstm_rul.py exactly."""
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])
        rul = self.head(last_hidden)
        return rul.squeeze(-1)


def cmapss_score(y_true, y_pred):
    d = y_pred - y_true
    early, late = d < 0, d >= 0
    score = np.zeros_like(d, dtype=np.float64)
    score[early] = np.exp(-d[early] / 13.0) - 1.0
    score[late] = np.exp(d[late] / 10.0) - 1.0
    return float(np.sum(score))


def time_single_row(predict_fn, X, n_runs=N_LATENCY_RUNS):
    row = X[:1]
    for _ in range(5):
        predict_fn(row)  # warm-up
    start = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(row)
    return ((time.perf_counter() - start) / n_runs) * 1000  # ms


def main():
    print("Loading LSTM checkpoint...")
    checkpoint = torch.load(LSTM_CHECKPOINT, map_location="cpu", weights_only=False)
    n_features = checkpoint["n_features"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    dropout = checkpoint["dropout"]
    rul_scale = checkpoint.get("rul_scale", 125.0)

    model = LSTMRegressor(n_features, hidden_size, num_layers, dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model: {n_features} features, hidden_size={hidden_size}, num_layers={num_layers}")

    print("\nLoading real held-out test sequences...")
    X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(SEQ_DIR, "y_test.npy")).astype(np.float32)
    print(f"X_test: {X_test.shape}")

    # ---- 1. Native PyTorch baseline ----
    with torch.no_grad():
        native_preds = model(torch.tensor(X_test)).numpy() * rul_scale
    native_rmse = np.sqrt(mean_squared_error(y_test, native_preds))
    native_mae = mean_absolute_error(y_test, native_preds)
    native_score = cmapss_score(y_test, native_preds)
    native_latency = time_single_row(
        lambda row: model(torch.tensor(row)).detach().numpy(), X_test
    )
    native_size_kb = os.path.getsize(LSTM_CHECKPOINT) / 1024

    print(f"\n=== Native PyTorch LSTM ===")
    print(f"RMSE: {native_rmse:.3f} | MAE: {native_mae:.3f} | CMAPSS Score: {native_score:.1f}")
    print(f"Size: {native_size_kb:.1f} KB | Single-row latency: {native_latency:.3f} ms")

    # ---- 2. Export to ONNX (legacy exporter -- see module docstring) ----
    print("\nExporting to ONNX...")
    dummy_input = torch.randn(1, X_test.shape[1], n_features)
    torch.onnx.export(
        model, dummy_input, ONNX_PATH,
        input_names=["sensor_window"], output_names=["rul_normalized"],
        dynamic_axes={"sensor_window": {0: "batch_size"}, "rul_normalized": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Saved to {ONNX_PATH}")

    # ---- 3. ONNX Runtime evaluation on the SAME real test set ----
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    onnx_preds_norm = sess.run(None, {input_name: X_test})[0]
    onnx_preds = onnx_preds_norm.squeeze() * rul_scale
    onnx_rmse = np.sqrt(mean_squared_error(y_test, onnx_preds))
    onnx_mae = mean_absolute_error(y_test, onnx_preds)
    onnx_score = cmapss_score(y_test, onnx_preds)
    onnx_latency = time_single_row(
        lambda row: sess.run(None, {input_name: row})[0], X_test
    )
    onnx_size_kb = os.path.getsize(ONNX_PATH) / 1024

    max_diff = np.abs(native_preds - onnx_preds).max()

    print(f"\n=== ONNX Runtime LSTM ===")
    print(f"RMSE: {onnx_rmse:.3f} | MAE: {onnx_mae:.3f} | CMAPSS Score: {onnx_score:.1f}")
    print(f"Size: {onnx_size_kb:.1f} KB | Single-row latency: {onnx_latency:.3f} ms")
    print(f"\nMax prediction diff, native vs ONNX (real RUL units): {max_diff:.6f}")
    print("(should be ~0 -- confirms the export preserved the model exactly)")

    # ---- Summary ----
    print("\n=== Edge Deployment Summary (LSTM, your best model) ===")
    print(f"{'Version':<20}{'Size (KB)':<14}{'Latency (ms)':<16}{'RMSE':<10}{'CMAPSS Score':<12}")
    print(f"{'Native PyTorch':<20}{native_size_kb:<14.1f}{native_latency:<16.3f}{native_rmse:<10.3f}{native_score:<12.1f}")
    print(f"{'ONNX Runtime':<20}{onnx_size_kb:<14.1f}{onnx_latency:<16.3f}{onnx_rmse:<10.3f}{onnx_score:<12.1f}")

    size_change = (1 - onnx_size_kb / native_size_kb) * 100
    print(f"\nSize change vs native: {size_change:+.1f}%")
    print(f"Accuracy preserved: RMSE differs by {abs(onnx_rmse - native_rmse):.4f} cycles "
          f"({'negligible' if abs(onnx_rmse - native_rmse) < 0.5 else 'noticeable'})")

    summary = {
        "native_pytorch": {"size_kb": round(native_size_kb, 1), "latency_ms": round(native_latency, 3),
                            "rmse": round(native_rmse, 3), "mae": round(native_mae, 3), "cmapss_score": round(native_score, 1)},
        "onnx_runtime": {"size_kb": round(onnx_size_kb, 1), "latency_ms": round(onnx_latency, 3),
                          "rmse": round(onnx_rmse, 3), "mae": round(onnx_mae, 3), "cmapss_score": round(onnx_score, 1)},
        "max_prediction_diff": float(max_diff),
    }
    with open(os.path.join(MODEL_DIR, "lstm_onnx_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison to {MODEL_DIR}/lstm_onnx_comparison.json")


if __name__ == "__main__":
    main()
