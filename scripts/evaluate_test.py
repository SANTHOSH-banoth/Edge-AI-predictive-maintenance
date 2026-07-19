"""
evaluate_test.py
-----------------
The missing piece from Week 4's first pass: everything so far (CV RMSE
14.42) was measured on splits of the TRAINING data. CMAPSS ships a genuine
held-out test set specifically so people can report real generalization
performance — this is the number that actually validates the pipeline,
and the one most likely to get asked about directly.

CMAPSS test protocol (Saxena & Goebel, 2008):
  - Each test engine's trajectory is truncated at some cycle BEFORE failure
    (not run to failure like training engines).
  - RUL_FD001.txt gives the TRUE remaining life at that truncation point,
    one value per engine, in engine order.
  - The standard evaluation: predict RUL using only the LAST recorded
    cycle of each test engine, then compare against RUL_FD001.txt.

This script:
  1. Loads raw test_FD001.csv
  2. Re-applies the exact same feature engineering as signal_features.py
     (imported directly, not re-implemented, so there's no drift between
     train-time and test-time feature logic)
  3. Takes the last cycle per engine, predicts RUL with the saved model
  4. Clips predictions at the same RUL_CLIP used in training, for a fair
     apples-to-apples comparison
  5. Reports true test RMSE + per-engine error table

Run:
    python scripts/evaluate_test.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Reuse Week 3's actual feature engineering functions — do not duplicate logic
from signal_features import (
    add_rolling_features,
    add_thermal_stress_features,
    add_fft_feature,
    KEY_SENSORS_FOR_ROLLING,
)

# ---- Config ------------------------------------------------------------
# Adjust these paths if your raw test files live somewhere else — CMAPSS
# ships test_FD001.txt and RUL_FD001.txt as whitespace-delimited, no header.
TEST_DATA_PATH = Path("data/cmapss/processed/test_FD001.csv")
TRUE_RUL_PATH = Path("data/cmapss/RUL_FD001.txt")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_rul_model.json"
META_PATH = MODEL_DIR / "best_params.json"

UNIT_COL = "unit_number"
TIME_COL = "time_cycles"


def cmapss_score(y_true, y_pred):
    """
    Official CMAPSS asymmetric scoring function (Saxena & Goebel, 2008) --
    identical formula used in cnn_rul.py and lstm_rul.py, so this number is
    directly comparable across all three models, not just RMSE.

    Late predictions (predicted RUL > actual RUL, i.e. the model said "more
    life left" than there really was) are penalized far more harshly than
    early ones, because that's the dangerous error in real maintenance
    scheduling -- missing a failure window costs much more than an early,
    over-cautious maintenance call.

    d = y_pred - y_true
        d < 0 (early/conservative prediction): score += exp(-d/13) - 1
        d >= 0 (late/optimistic prediction):   score += exp(d/10) - 1
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    early = d < 0
    late = ~early
    score = np.zeros_like(d, dtype=np.float64)
    score[early] = np.exp(-d[early] / 13.0) - 1.0
    score[late] = np.exp(d[late] / 10.0) - 1.0
    return float(np.sum(score))


def load_true_rul(path):
    """RUL_FD001.txt: one integer per line, in engine order (1, 2, 3, ...)."""
    true_rul = pd.read_csv(path, header=None, names=["true_RUL"])
    true_rul["unit_number"] = np.arange(1, len(true_rul) + 1)
    return true_rul


def engineer_test_features(df):
    """Apply the identical Week 3 pipeline to the raw test set."""
    print("Applying Week 3 feature engineering to test data...")
    df = add_rolling_features(df, KEY_SENSORS_FOR_ROLLING)
    df = add_thermal_stress_features(df)
    df = add_fft_feature(df)
    return df


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    rul_clip = meta["rul_clip"]
    cv_rmse = meta.get("cv_rmse")
    baseline_rmse = meta.get("baseline_rmse")

    print(f"Loading raw test data from {TEST_DATA_PATH}...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"Test shape (raw): {test_df.shape}")

    test_df = engineer_test_features(test_df)

    missing_feats = set(feature_cols) - set(test_df.columns)
    if missing_feats:
        raise ValueError(
            f"Engineered test data is missing features the model expects: {missing_feats}. "
            "Check that signal_features.py functions match what was used at train time."
        )

    # Standard CMAPSS protocol: take only the LAST recorded cycle per engine
    last_cycle = (
        test_df.sort_values([UNIT_COL, TIME_COL])
        .groupby(UNIT_COL)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"Engines in test set: {last_cycle[UNIT_COL].nunique()}")

    true_rul = load_true_rul(TRUE_RUL_PATH)
    merged = last_cycle.merge(true_rul, on=UNIT_COL, how="inner")
    if len(merged) != len(last_cycle):
        print(
            f"WARNING: {len(last_cycle) - len(merged)} engines couldn't be matched "
            f"to RUL_FD001.txt — check engine ID alignment."
        )

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    X_test = merged[feature_cols]
    preds = model.predict(X_test)
    preds_clipped = np.clip(preds, 0, rul_clip)

    true_vals = merged["true_RUL"].clip(upper=rul_clip)  # same clip, fair comparison

    test_rmse = np.sqrt(mean_squared_error(true_vals, preds_clipped))
    test_mae = mean_absolute_error(true_vals, preds_clipped)
    test_score = cmapss_score(true_vals.values, preds_clipped)

    print("\n=== Test-set evaluation (real CMAPSS held-out engines) ===")
    print(f"Engines evaluated: {len(merged)}")
    print(f"Test RMSE:    {test_rmse:.3f}")
    print(f"Test MAE:     {test_mae:.3f}")
    print(f"CMAPSS Score: {test_score:.1f}  (lower is better; same formula as cnn_rul.py/lstm_rul.py)")
    print(f"\nFor direct comparison against your other models (fill in from their printed output):")
    print(f"  XGBoost   -> RMSE {test_rmse:.3f}, CMAPSS Score {test_score:.1f}")
    print(f"  LSTM      -> RMSE 12.803, CMAPSS Score 267.0")
    print(f"  CNN       -> RMSE 18.063, CMAPSS Score 649.7")
    if cv_rmse is not None:
        print(f"\nFor comparison:")
        print(f"  Training CV RMSE:        {cv_rmse:.3f}")
        if baseline_rmse is not None:
            print(f"  LinearRegression baseline CV RMSE: {baseline_rmse:.3f}")
        gap = test_rmse - cv_rmse
        print(f"  Test vs CV gap: {gap:+.3f} "
              f"({'larger error on unseen engines — some overfit to watch' if gap > 1.5 else 'close to CV — model generalizes well'})")

    # Per-engine error table, worst offenders first — useful for spotting
    # whether errors are randomly distributed or concentrated in specific
    # engines (which would suggest a systematic issue, not just noise)
    merged["predicted_RUL"] = preds_clipped
    merged["abs_error"] = (merged["true_RUL"].clip(upper=rul_clip) - merged["predicted_RUL"]).abs()
    report = merged[[UNIT_COL, TIME_COL, "true_RUL", "predicted_RUL", "abs_error"]].sort_values(
        "abs_error", ascending=False
    )
    print("\nTop 10 largest errors (engine, last cycle, true vs predicted RUL):")
    print(report.head(10).to_string(index=False))

    report_path = MODEL_DIR / "test_set_evaluation.csv"
    report.to_csv(report_path, index=False)
    print(f"\nFull per-engine report saved to {report_path}")


if __name__ == "__main__":
    main()