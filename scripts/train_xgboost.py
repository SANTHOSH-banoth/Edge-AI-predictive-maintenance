"""
Week 4: XGBoost RUL regression with Optuna hyperparameter tuning + SHAP explainability.

Pipeline:
  1. Load engineered training data (from Week 3's signal_features.py output)
  2. Cross-validate with GroupKFold, grouped by engine unit_number
     (prevents leakage — cycles from the same engine never split across folds)
  3. Optuna searches hyperparameters to minimize mean CV RMSE
  4. Refit best model on full training data
  5. SHAP summary + dependence plots for explainability
  6. Save model + feature list + SHAP values

Run:
    python scripts/train_xgboost.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---- Config ----------------------------------------------------------------
DATA_PATH = Path("data/cmapss/processed/train_FD001_engineered.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

UNIT_COL = "unit_number"      # engine id column — adjust if yours is named differently
TIME_COL = "time_cycles"
TARGET_COL = "RUL"

N_FOLDS = 5
N_TRIALS = 30
TIMEOUT_SECONDS = None         # set e.g. 900 to cap total tuning time
RANDOM_STATE = 42

# RUL is often clipped in CMAPSS work (piecewise linear degradation model):
# early-life cycles get a flat max RUL cap since degradation hasn't started yet.
RUL_CLIP = 125


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH} — shape {df.shape}")

    missing = {UNIT_COL, TIME_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    drop_cols = {UNIT_COL, TIME_COL, TARGET_COL}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].clip(upper=RUL_CLIP)
    groups = df[UNIT_COL]

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target clipped at RUL={RUL_CLIP} (piecewise linear degradation assumption)")
    return X, y, groups, feature_cols


def cv_rmse(params, X, y, groups, n_folds=N_FOLDS, return_iterations=False):
    gkf = GroupKFold(n_splits=n_folds)
    rmses = []
    best_iterations = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            **params,
            n_estimators=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        best_iterations.append(model.best_iteration)

    mean_rmse = float(np.mean(rmses))
    if return_iterations:
        return mean_rmse, best_iterations
    return mean_rmse


def baseline_cv_rmse(X, y, groups, n_folds=N_FOLDS):
    """Simple LinearRegression through the same GroupKFold CV, purely so the
    tuned XGBoost RMSE has an honest reference point ('naive baseline got X,
    engineered features + tuned XGBoost got Y') rather than floating alone."""
    gkf = GroupKFold(n_splits=n_folds)
    rmses = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return float(np.mean(rmses))


def objective(trial, X, y, groups):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }
    return cv_rmse(params, X, y, groups)


def time_single_fit(X, y, groups):
    """Time one 5-fold CV evaluation to estimate total Optuna runtime."""
    default_params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.1,
    }
    start = time.time()
    rmse = cv_rmse(default_params, X, y, groups)
    elapsed = time.time() - start
    est_total = elapsed * N_TRIALS
    print(f"\nTiming check: one 5-fold CV eval took {elapsed:.1f}s (baseline RMSE={rmse:.2f})")
    print(f"Estimated total time for {N_TRIALS} trials: ~{est_total/60:.1f} min")
    return elapsed


def main():
    X, y, groups, feature_cols = load_data()

    print("\nRunning LinearRegression baseline (same CV splits) for reference...")
    baseline_rmse = baseline_cv_rmse(X, y, groups)
    print(f"Baseline (LinearRegression) CV RMSE: {baseline_rmse:.3f}")

    time_single_fit(X, y, groups)
    proceed = input("\nProceed with full Optuna search? [y/n]: ").strip().lower()
    if proceed != "y":
        print("Stopped before tuning. Adjust N_TRIALS/N_FOLDS in the config and re-run.")
        return

    print(f"\nStarting Optuna search: {N_TRIALS} trials x {N_FOLDS}-fold CV")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(
        lambda trial: objective(trial, X, y, groups),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        show_progress_bar=True,
    )

    print(f"\nBest CV RMSE: {study.best_value:.3f}")
    print(f"Improvement over LinearRegression baseline: {baseline_rmse - study.best_value:.3f} RMSE "
          f"({(1 - study.best_value / baseline_rmse) * 100:.1f}% lower error)")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Re-run CV once more with the winning params purely to read off each
    # fold's best_iteration (found via early stopping) — averaging these
    # gives the final refit a grounded tree count instead of training all
    # 1000 trees blind with no early stopping signal at all.
    best_params = study.best_params
    _, best_iterations = cv_rmse(best_params, X, y, groups, return_iterations=True)
    avg_best_iteration = int(round(np.mean(best_iterations)))
    print(f"\nPer-fold best iterations (early stopping): {best_iterations}")
    print(f"Using averaged best iteration for final refit: {avg_best_iteration} trees "
          f"(previously: fixed 1000 trees, no early stopping — a real gap in the first pass)")

    final_model = xgb.XGBRegressor(
        **best_params,
        n_estimators=avg_best_iteration,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    # Save artifacts
    model_path = MODEL_DIR / "xgb_rul_model.json"
    final_model.save_model(model_path)

    with open(MODEL_DIR / "best_params.json", "w") as f:
        json.dump({
            "params": best_params,
            "cv_rmse": study.best_value,
            "baseline_rmse": baseline_rmse,
            "n_estimators_used": avg_best_iteration,
            "feature_cols": feature_cols,
            "rul_clip": RUL_CLIP,
        }, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved params/metadata to {MODEL_DIR / 'best_params.json'}")
    print("\nNext: run scripts/explain_shap.py to generate SHAP explainability plots.")


if __name__ == "__main__":
    main()