"""
shap_analysis.py
-------------------
Week 4, Days 10-11: SHAP (SHapley Additive exPlanations) explainability
for the XGBoost RUL model.

Why this matters more than just "nice to have":
Every model so far answers "what's the predicted RUL?" SHAP answers the
much more useful question for a maintenance engineer: "WHY does the model
think this engine has 12 cycles left?" -- which sensor readings pushed the
prediction down (toward failure) and which pushed it up (toward healthy).

For interviews specifically, this is where your mechanical engineering
background pays off: if SHAP shows thermal_stress_index and
cumulative_thermal_stress as top drivers, that's not a coincidence to
just report -- it's a chance to say "this matches known turbine failure
physics: sustained thermal stress is a primary driver of blade fatigue
and creep damage, so the model rediscovering that from data alone is a
genuine sanity check on whether it learned something real, not spurious
correlations."

What this script produces:
  1. A global feature importance bar chart (mean |SHAP value| per feature)
     -> "which sensors matter most, on average, across all engines"
  2. A summary beeswarm-style plot -> shows not just importance but
     DIRECTION (does high sensor_4 push RUL up or down?)
  3. A waterfall plot for one specific high-risk engine -> shows exactly
     how that one prediction was built up, feature by feature -- this is
     the "explain this specific prediction" story, not just aggregate stats
  4. A printed top-10 ranking with physical interpretation

Uses the SAME feature engineering (signal_features.py) and test-set
construction as quantize_and_benchmark.py / evaluate_test.py, so the
explanations are computed on genuine held-out engines.

Run:
    python scripts/shap_analysis.py
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, avoids display issues
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from signal_features import (
    add_rolling_features, add_thermal_stress_features, add_fft_feature,
    KEY_SENSORS_FOR_ROLLING,
)

MODEL_DIR = "models"
DOCS_DIR = "docs"
NATIVE_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_rul_model.json")
META_PATH = os.path.join(MODEL_DIR, "best_params.json")

TEST_DATA_PATH = os.path.join("data", "cmapss", "processed", "test_FD001.csv")
TRUE_RUL_PATH = os.path.join("data", "cmapss", "RUL_FD001.txt")

UNIT_COL, TIME_COL = "unit_number", "time_cycles"

# Physical interpretation notes for the most likely top features -- used to
# annotate the printed ranking with WHY a feature matters, not just its rank.
PHYSICAL_NOTES = {
    "thermal_stress_index": "deviation of T24/T30/T50 from this engine's healthy baseline -- direct fatigue driver",
    "cumulative_thermal_stress": "running total of thermal stress -- mirrors cumulative fatigue damage (Miner's rule)",
    "sensor_4_roll_std5": "rising instability in T50 (turbine outlet temp) over recent cycles",
    "sensor_11_fft_dominant_energy": "oscillatory pattern strength in HPC outlet pressure over the recent window",
    "sensor_4": "T50, turbine outlet temperature -- the hottest, most safety-critical measured stage",
    "sensor_11": "Ps30, static pressure at HPC outlet",
}


def load_test_set(feature_cols, rul_clip):
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df = add_rolling_features(test_df, KEY_SENSORS_FOR_ROLLING)
    test_df = add_thermal_stress_features(test_df)
    test_df = add_fft_feature(test_df)

    last_cycle = (
        test_df.sort_values([UNIT_COL, TIME_COL])
        .groupby(UNIT_COL).tail(1).reset_index(drop=True)
    )
    true_rul = pd.read_csv(TRUE_RUL_PATH, header=None, names=["true_RUL"])
    true_rul[UNIT_COL] = np.arange(1, len(true_rul) + 1)

    merged = last_cycle.merge(true_rul, on=UNIT_COL, how="inner")
    X_test = merged[feature_cols].astype(np.float32)
    y_test = merged["true_RUL"].clip(upper=rul_clip).values
    unit_ids = merged[UNIT_COL].values
    return X_test, y_test, unit_ids


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)

    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    rul_clip = meta["rul_clip"]

    print("Loading model and real held-out test set...")
    model = xgb.XGBRegressor()
    model.load_model(NATIVE_MODEL_PATH)
    X_test, y_test, unit_ids = load_test_set(feature_cols, rul_clip)
    print(f"Test engines: {len(X_test)}, features: {len(feature_cols)}")

    # ---- Compute SHAP values (TreeExplainer -- exact and fast for XGBoost) ----
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # ---- 1. Global feature importance bar chart ----
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    bar_path = os.path.join(DOCS_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved global importance chart to {bar_path}")

    # ---- 2. Summary beeswarm plot (importance + direction) ----
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    beeswarm_path = os.path.join(DOCS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved summary beeswarm plot to {beeswarm_path}")

    # ---- 3. Waterfall for the single highest-risk engine (lowest predicted RUL) ----
    preds = model.predict(X_test)
    riskiest_idx = int(np.argmin(preds))
    riskiest_unit = unit_ids[riskiest_idx]

    plt.figure()
    shap.plots.waterfall(shap_values[riskiest_idx], show=False, max_display=12)
    plt.tight_layout()
    waterfall_path = os.path.join(DOCS_DIR, f"shap_waterfall_engine_{riskiest_unit}.png")
    plt.savefig(waterfall_path, dpi=150)
    plt.close()
    print(f"Saved waterfall plot for highest-risk engine (#{riskiest_unit}, "
          f"predicted RUL={preds[riskiest_idx]:.1f}) to {waterfall_path}")

    # ---- 4. Printed top-10 ranking with physical interpretation ----
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n=== Top 10 features driving RUL predictions (real held-out test set) ===")
    for i, row in importance_df.head(10).iterrows():
        note = PHYSICAL_NOTES.get(row["feature"], "")
        note_str = f"  -- {note}" if note else ""
        print(f"{i+1:>2}. {row['feature']:<35} mean|SHAP|={row['mean_abs_shap']:.3f}{note_str}")

    importance_df.to_csv(os.path.join(MODEL_DIR, "shap_feature_importance.csv"), index=False)
    print(f"\nFull ranking saved to {MODEL_DIR}/shap_feature_importance.csv")
    print(f"Plots saved to {DOCS_DIR}/ -- include these directly in your README/portfolio.")


if __name__ == "__main__":
    main()
