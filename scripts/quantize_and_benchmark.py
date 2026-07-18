"""
quantize_and_benchmark.py
--------------------------
Week 5: model compression for edge deployment.

HONESTY NOTE (worth being precise about in interviews): "quantization" for
a tree ensemble like XGBoost is NOT the same story as quantizing a neural
network. A CNN's size is dominated by huge weight matrices, so int8
quantization can shrink it 4x and dramatically cut matmul latency. A tree
ensemble's "weights" are just leaf values and split thresholds — there's
no matmul to accelerate. What this step actually buys you:
  1. ONNX conversion: a standardized, cross-platform inference format that
     runs efficiently on edge/embedded runtimes (ARM, mobile, microcontrollers
     via ONNX Runtime) instead of requiring the full XGBoost library on-device.
  2. INT8 quantization on top of that: modest file size reduction and some
     latency benefit from ONNX Runtime's optimized int8 kernels — but the
     gain is smaller and less guaranteed than in deep learning. That's why
     this script actually MEASURES the effect instead of assuming it.

Three versions compared:
  - native XGBoost (.json)
  - ONNX fp32
  - ONNX int8 (dynamically quantized)

Metrics reported for each:
  - file size on disk
  - single-row inference latency (the realistic edge scenario: one sensor
    reading arriving at a time, not a batch)
  - REAL test-set RMSE (using your actual held-out CMAPSS test engines,
    not just prediction-difference sanity checks) — so you can say
    "quantization changed test RMSE by X" with a number, not a guess.

Install (if not already):
    pip install onnxmltools skl2onnx onnxruntime onnx

Run:
    python scripts/quantize_and_benchmark.py
"""

from onnx import helper
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Reuse Week 3's real feature engineering — same reasoning as evaluate_test.py
from signal_features import (
    add_rolling_features,
    add_thermal_stress_features,
    add_fft_feature,
    KEY_SENSORS_FOR_ROLLING,
)

# ---- Config ------------------------------------------------------------
MODEL_DIR = Path("models")
NATIVE_MODEL_PATH = MODEL_DIR / "xgb_rul_model.json"
META_PATH = MODEL_DIR / "best_params.json"

ONNX_FP32_PATH = MODEL_DIR / "xgb_rul_model_fp32.onnx"
ONNX_INT8_PATH = MODEL_DIR / "xgb_rul_model_int8.onnx"

TEST_DATA_PATH = Path("data/cmapss/processed/test_FD001.csv")
TRUE_RUL_PATH = Path("data/cmapss/RUL_FD001.txt")

UNIT_COL = "unit_number"
TIME_COL = "time_cycles"

N_LATENCY_RUNS = 200  # repeated single-row predictions to get a stable average


def load_test_set(feature_cols, rul_clip):
    """Rebuild the same held-out test evaluation set used in evaluate_test.py,
    so accuracy comparisons here are on genuine unseen engines, not train data."""
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df = add_rolling_features(test_df, KEY_SENSORS_FOR_ROLLING)
    test_df = add_thermal_stress_features(test_df)
    test_df = add_fft_feature(test_df)

    last_cycle = (
        test_df.sort_values([UNIT_COL, TIME_COL])
        .groupby(UNIT_COL)
        .tail(1)
        .reset_index(drop=True)
    )

    true_rul = pd.read_csv(TRUE_RUL_PATH, header=None, names=["true_RUL"])
    true_rul[UNIT_COL] = np.arange(1, len(true_rul) + 1)

    merged = last_cycle.merge(true_rul, on=UNIT_COL, how="inner")
    X_test = merged[feature_cols].astype(np.float32)
    y_test = merged["true_RUL"].clip(upper=rul_clip).values
    return X_test, y_test


def time_single_row_predictions(predict_fn, X, n_runs=N_LATENCY_RUNS):
    """Time repeated single-row predictions — the realistic edge scenario
    of one sensor reading arriving at a time, not a batch."""
    row = X.iloc[[0]] if isinstance(X, pd.DataFrame) else X[:1]
    # warm-up (excludes one-time setup cost like session init)
    for _ in range(5):
        predict_fn(row)

    start = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(row)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per prediction


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    rul_clip = meta["rul_clip"]
    n_features = len(feature_cols)

    print("Loading native XGBoost model and real held-out test set...")
    native_model = xgb.XGBRegressor()
    native_model.load_model(NATIVE_MODEL_PATH)
    X_test, y_test = load_test_set(feature_cols, rul_clip)
    print(f"Test engines: {len(X_test)}")

    results = []

    # ---- 1. Native XGBoost -------------------------------------------------
    native_preds = native_model.predict(X_test)
    native_rmse = np.sqrt(mean_squared_error(y_test, np.clip(native_preds, 0, rul_clip)))
    native_latency = time_single_row_predictions(
        lambda row: native_model.predict(row), X_test
    )
    native_size_kb = NATIVE_MODEL_PATH.stat().st_size / 1024

    results.append({
        "version": "Native XGBoost",
        "file_size_kb": round(native_size_kb, 1),
        "single_row_latency_ms": round(native_latency, 3),
        "test_rmse": round(native_rmse, 3),
    })

    # ---- 2. Convert to ONNX (fp32) -----------------------------------------
    print("\nConverting to ONNX (fp32)...")
    # onnxmltools' XGBoost converter expects generic 'f0','f1',... feature
    # names internally, but this model was trained on a named DataFrame, so
    # the booster stored real column names (e.g. 'sensor_4_roll_mean5').
    # Reset them to XGBoost's default numbered scheme just for export — this
    # does not change the model's behavior, only how splits are labeled
    # internally. Column ORDER (not names) is what actually matters for
    # correctness, and that's preserved since we never reordered feature_cols.
    booster = native_model.get_booster()
    booster.feature_names = None

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(native_model, initial_types=initial_type)

    # Fix: onnxmltools only sets the 'ai.onnx.ml' opset domain (for the tree
    # ensemble op), but onnxruntime's quantizer requires a standard 'ai.onnx'
    # domain entry to exist too, to check op compatibility. Add it if missing.
    existing_domains = {opset.domain for opset in onnx_model.opset_import}
    if "" not in existing_domains:
        onnx_model.opset_import.append(helper.make_opsetid("", 13))

    with open(ONNX_FP32_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH))
    input_name = sess_fp32.get_inputs()[0].name

    def onnx_fp32_predict(row):
        return sess_fp32.run(None, {input_name: row.values.astype(np.float32)})[0]

    onnx_fp32_preds = np.array([onnx_fp32_predict(X_test.iloc[[i]])[0][0]
                                 for i in range(len(X_test))])
    onnx_fp32_rmse = np.sqrt(mean_squared_error(y_test, np.clip(onnx_fp32_preds, 0, rul_clip)))
    onnx_fp32_latency = time_single_row_predictions(onnx_fp32_predict, X_test)
    onnx_fp32_size_kb = ONNX_FP32_PATH.stat().st_size / 1024

    max_diff_fp32 = np.abs(native_preds - onnx_fp32_preds).max()
    print(f"Max prediction diff, native vs ONNX fp32: {max_diff_fp32:.6f} "
          f"(should be ~0 — confirms conversion preserved the model exactly)")

    results.append({
        "version": "ONNX fp32",
        "file_size_kb": round(onnx_fp32_size_kb, 1),
        "single_row_latency_ms": round(onnx_fp32_latency, 3),
        "test_rmse": round(onnx_fp32_rmse, 3),
    })

    # ---- 3. Dynamic INT8 quantization ---------------------------------------
    print("\nApplying dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
    )

    sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH))

    def onnx_int8_predict(row):
        return sess_int8.run(None, {input_name: row.values.astype(np.float32)})[0]

    onnx_int8_preds = np.array([onnx_int8_predict(X_test.iloc[[i]])[0][0]
                                 for i in range(len(X_test))])
    onnx_int8_rmse = np.sqrt(mean_squared_error(y_test, np.clip(onnx_int8_preds, 0, rul_clip)))
    onnx_int8_latency = time_single_row_predictions(onnx_int8_predict, X_test)
    onnx_int8_size_kb = ONNX_INT8_PATH.stat().st_size / 1024

    max_diff_int8 = np.abs(native_preds - onnx_int8_preds).max()
    print(f"Max prediction diff, native vs ONNX int8: {max_diff_int8:.4f} "
          f"(this WILL be nonzero — quantifies the precision/size tradeoff)")

    results.append({
        "version": "ONNX int8 (quantized)",
        "file_size_kb": round(onnx_int8_size_kb, 1),
        "single_row_latency_ms": round(onnx_int8_latency, 3),
        "test_rmse": round(onnx_int8_rmse, 3),
    })

    # ---- Summary -------------------------------------------------------------
    summary = pd.DataFrame(results)
    print("\n=== Compression comparison (real held-out test set) ===")
    print(summary.to_string(index=False))

    size_reduction = (1 - onnx_int8_size_kb / native_size_kb) * 100
    rmse_change = onnx_int8_rmse - native_rmse
    print(f"\nFile size: {size_reduction:+.1f}% vs native "
          f"({'smaller' if size_reduction > 0 else 'larger'})")
    print(f"Test RMSE change from quantization: {rmse_change:+.3f} "
          f"({'negligible — safe to deploy quantized' if abs(rmse_change) < 0.5 else 'noticeable — weigh size/latency gain against this accuracy cost'})")

    summary_path = MODEL_DIR / "compression_comparison.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved comparison table to {summary_path}")


if __name__ == "__main__":
    main()