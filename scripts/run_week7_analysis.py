"""
run_week7_analysis.py
----------------------
Week 7 deliverable: runs drift monitoring, per-RUL-bucket error analysis,
and adversarial robustness testing (including zero vs. median imputation
comparison) against the REAL trained XGBoost model and REAL CMAPSS test set.

Run:
    python scripts/run_week7_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from evaluate_test import (
    engineer_test_features,
    load_true_rul,
    TEST_DATA_PATH,
    TRUE_RUL_PATH,
    MODEL_PATH,
    META_PATH,
    UNIT_COL,
    TIME_COL,
)
from drift_monitor import simulate_calibration_decay, compute_drift_report
from error_analysis import regression_error_by_rul_bucket
from robustness_test import noise_robustness_curve, compare_imputation_strategies

TRAIN_DATA_PATH = Path("data/cmapss/processed/train_FD001.csv")


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    rul_clip = meta["rul_clip"]

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    print("Loading + engineering train data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df = engineer_test_features(train_df)

    print("Loading + engineering test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df = engineer_test_features(test_df)

    last_cycle = (
        test_df.sort_values([UNIT_COL, TIME_COL])
        .groupby(UNIT_COL)
        .tail(1)
        .reset_index(drop=True)
    )
    true_rul = load_true_rul(TRUE_RUL_PATH)
    merged = last_cycle.merge(true_rul, on=UNIT_COL, how="inner")

    X_test = merged[feature_cols]
    y_test = merged["true_RUL"].clip(upper=rul_clip).to_numpy()

    print("\n" + "=" * 60)
    print("1. DRIFT REPORT: train distribution vs real held-out test engines")
    print("=" * 60)
    drift_report = compute_drift_report(train_df, test_df, feature_cols)
    print(drift_report.to_string(index=False))

    print("\n" + "=" * 60)
    print("1b. DRIFT DETECTION SANITY CHECK: synthetic calibration decay")
    print("=" * 60)
    drifted_test = simulate_calibration_decay(test_df, sensor_cols=feature_cols)
    synthetic_drift_report = compute_drift_report(test_df, drifted_test, feature_cols)
    print(synthetic_drift_report.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("2. ERROR ANALYSIS BY RUL BUCKET")
    print("=" * 60)
    preds = np.clip(model.predict(X_test), 0, rul_clip)
    bucket_report = regression_error_by_rul_bucket(y_test, preds)
    print(bucket_report.to_string(index=False))

    print("\n" + "=" * 60)
    print("3. NOISE ROBUSTNESS")
    print("=" * 60)
    X_test_arr = X_test.to_numpy()

    def predict_fn(X_in):
        return np.clip(model.predict(X_in), 0, rul_clip)

    noise_report = noise_robustness_curve(predict_fn, X_test_arr, y_test)
    print(noise_report.to_string(index=False))

    print("\n" + "=" * 60)
    print("3b. MISSING-DATA ROBUSTNESS -- zero vs. median imputation")
    print("=" * 60)
    missing_report = compare_imputation_strategies(predict_fn, X_test_arr, y_test)
    print(missing_report.to_string(index=False))

    out_dir = Path("models")
    drift_report.to_csv(out_dir / "week7_drift_report.csv", index=False)
    bucket_report.to_csv(out_dir / "week7_error_by_rul_bucket.csv", index=False)
    noise_report.to_csv(out_dir / "week7_noise_robustness.csv", index=False)
    missing_report.to_csv(out_dir / "week7_missing_robustness.csv", index=False)
    print("\nAll Week 7 reports saved to models/week7_*.csv")


if __name__ == "__main__":
    main()
