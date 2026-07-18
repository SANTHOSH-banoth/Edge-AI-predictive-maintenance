"""
Week 4 (part 2): SHAP explainability for the trained XGBoost RUL model.

Run this after scripts/train_xgboost.py has produced:
  - models/xgb_rul_model.json
  - models/best_params.json

Generates:
  - models/shap_summary.png       (global feature importance / direction)
  - models/shap_dependence_<feature>.png  (top 3 features)
  - models/shap_values.npy        (raw SHAP values, for further analysis)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

DATA_PATH = Path("data/cmapss/processed/train_FD001_engineered.csv")
MODEL_DIR = Path("models")

UNIT_COL = "unit_number"
TIME_COL = "time_cycles"
TARGET_COL = "RUL"

# Keep the sample size manageable for SHAP (TreeExplainer is fast, but
# plotting 20k+ points gets noisy and slow to render). Adjust as needed.
SAMPLE_SIZE = 3000
RANDOM_STATE = 42


def main():
    with open(MODEL_DIR / "best_params.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    rul_clip = meta["rul_clip"]

    df = pd.read_csv(DATA_PATH)
    X = df[feature_cols].copy()
    y = df[TARGET_COL].clip(upper=rul_clip)

    if len(X) > SAMPLE_SIZE:
        sample_idx = X.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).index
        X_sample = X.loc[sample_idx]
    else:
        X_sample = X

    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "xgb_rul_model.json")

    print(f"Computing SHAP values on {len(X_sample)} sampled rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    np.save(MODEL_DIR / "shap_values.npy", shap_values)

    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved {MODEL_DIR / 'shap_summary.png'}")

    # Dependence plots for the top 3 most impactful features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = X_sample.columns[np.argsort(mean_abs_shap)[::-1][:3]]

    for feat in top_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X_sample, show=False)
        plt.tight_layout()
        out_path = MODEL_DIR / f"shap_dependence_{feat}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print("\nTop 3 features by mean |SHAP value|:")
    for feat, val in zip(top_features, sorted(mean_abs_shap)[::-1][:3]):
        print(f"  {feat}: {val:.3f}")


if __name__ == "__main__":
    main()
