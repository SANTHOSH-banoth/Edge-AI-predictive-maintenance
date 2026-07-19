"""
quantize_lstm.py
-------------------
Gap #2 from the FD001 completeness check: XGBoost got a full INT8
quantization benchmark (quantize_and_benchmark.py); the LSTM never did.
This mirrors that treatment for the LSTM.

HONESTY NOTE, consistent with the XGBoost version: quantization benefits
are NOT guaranteed to transfer between model types. XGBoost's tree
ensemble showed NO real benefit from INT8 quantization (size and latency
were statistically unchanged). An LSTM is a genuinely different case --
it has real weight matrices (input-to-hidden, hidden-to-hidden gates)
that dynamic quantization CAN meaningfully compress, unlike tree splits.
So there's a real chance quantization helps here where it didn't for
XGBoost -- but this script MEASURES that on your real held-out test set
rather than assuming either outcome.

Run (after export_lstm_onnx.py has produced models/lstm_rul.onnx):
    python scripts/quantize_lstm.py
"""

import os
import time
import json
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_DIR = "models"
LSTM_ONNX_FP32_PATH = os.path.join(MODEL_DIR, "lstm_rul.onnx")
LSTM_ONNX_INT8_PATH = os.path.join(MODEL_DIR, "lstm_rul_int8.onnx")
SEQ_DIR = os.path.join("data", "cmapss", "sequences")

RUL_SCALE = 125.0
N_LATENCY_RUNS = 200


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


def evaluate_onnx_session(path, X_test, y_test, label):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    preds_norm = sess.run(None, {input_name: X_test})[0]
    preds = preds_norm.squeeze() * RUL_SCALE

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    score = cmapss_score(y_test, preds)
    latency = time_single_row(lambda row: sess.run(None, {input_name: row})[0], X_test)
    size_kb = os.path.getsize(path) / 1024

    print(f"\n=== {label} ===")
    print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | CMAPSS Score: {score:.1f}")
    print(f"Size: {size_kb:.1f} KB | Single-row latency: {latency:.3f} ms")

    return {"rmse": rmse, "mae": mae, "cmapss_score": score,
            "latency_ms": latency, "size_kb": size_kb, "preds": preds}


def main():
    if not os.path.exists(LSTM_ONNX_FP32_PATH):
        raise FileNotFoundError(
            f"{LSTM_ONNX_FP32_PATH} not found -- run export_lstm_onnx.py first."
        )

    print("Loading real held-out test sequences...")
    X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(SEQ_DIR, "y_test.npy")).astype(np.float32)
    print(f"X_test: {X_test.shape}\n")

    fp32_result = evaluate_onnx_session(LSTM_ONNX_FP32_PATH, X_test, y_test, "ONNX fp32 (baseline)")

    print("\nApplying dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=LSTM_ONNX_FP32_PATH,
        model_output=LSTM_ONNX_INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    print(f"Saved to {LSTM_ONNX_INT8_PATH}")

    int8_result = evaluate_onnx_session(LSTM_ONNX_INT8_PATH, X_test, y_test, "ONNX int8 (quantized)")

    max_diff = np.abs(fp32_result["preds"] - int8_result["preds"]).max()
    print(f"\nMax prediction diff, fp32 vs int8: {max_diff:.4f} "
          f"(nonzero is expected -- quantifies the precision/size tradeoff)")

    # ---- Summary ----
    print("\n=== LSTM Quantization Comparison (real held-out test set) ===")
    print(f"{'Version':<20}{'Size (KB)':<14}{'Latency (ms)':<16}{'RMSE':<10}{'CMAPSS Score':<12}")
    print(f"{'ONNX fp32':<20}{fp32_result['size_kb']:<14.1f}{fp32_result['latency_ms']:<16.3f}"
          f"{fp32_result['rmse']:<10.3f}{fp32_result['cmapss_score']:<12.1f}")
    print(f"{'ONNX int8':<20}{int8_result['size_kb']:<14.1f}{int8_result['latency_ms']:<16.3f}"
          f"{int8_result['rmse']:<10.3f}{int8_result['cmapss_score']:<12.1f}")

    size_change = (1 - int8_result["size_kb"] / fp32_result["size_kb"]) * 100
    latency_change = (1 - int8_result["latency_ms"] / fp32_result["latency_ms"]) * 100
    rmse_change = int8_result["rmse"] - fp32_result["rmse"]

    print(f"\nSize change (int8 vs fp32): {size_change:+.1f}%")
    print(f"Latency change (int8 vs fp32): {latency_change:+.1f}% "
          f"({'faster' if latency_change > 0 else 'slower'})")
    print(f"RMSE change: {rmse_change:+.3f} cycles "
          f"({'negligible -- safe to deploy quantized' if abs(rmse_change) < 0.5 else 'noticeable -- weigh against size/latency gain'})")

    summary = {
        "onnx_fp32": {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in fp32_result.items() if k != "preds"},
        "onnx_int8": {k: round(v, 3) if isinstance(v, float) else v
                      for k, v in int8_result.items() if k != "preds"},
        "max_prediction_diff": float(max_diff),
    }
    with open(os.path.join(MODEL_DIR, "lstm_quantization_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison to {MODEL_DIR}/lstm_quantization_comparison.json")


if __name__ == "__main__":
    main()
