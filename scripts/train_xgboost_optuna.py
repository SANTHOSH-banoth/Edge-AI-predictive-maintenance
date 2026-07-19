"""
train_xgboost_optuna.py
--------------------------
Week 4, Days 8-9 + Week 5 MLflow tracking: proper hyperparameter tuning +
cross-validation for the XGBoost RUL model, with every trial and the final
model logged to MLflow.

Why GroupKFold instead of plain KFold:
Your data has multiple rows per engine (one per cycle/window). A plain
random K-fold split would let rows from the SAME engine land in both train
and validation -- the model would partially "see" that engine already,
making validation scores look artificially good. GroupKFold guarantees
every row from a given engine stays entirely within one fold. This is the
same leakage class of bug that was already fixed elsewhere in this project
(cnn_rul.py / lstm_rul.py / autoencoder_anomaly.py train/val splits).

Why Optuna instead of GridSearchCV:
GridSearchCV requires guessing a fixed grid of values up front and tries
every combination -- wasteful. Optuna's TPE sampler looks at results so
far and picks the next hyperparameters more intelligently, so you get
better results in fewer total trials.

Why MLflow on top of that:
Without it, comparing trials means scrolling terminal output and copying
numbers by hand -- error-prone and doesn't scale past a handful of runs.
MLflow logs every trial's hyperparameters + CV RMSE automatically, plus
the final tuned model as a versioned artifact, all browsable in a local
web UI (`mlflow ui`) with sortable columns and comparison charts. This is
local-only (no server/account needed) -- it writes to an `mlruns/` folder
in your project, which should be gitignored (it's regenerable, not source).

Output (matches existing project conventions so evaluate_test.py,
quantize_and_benchmark.py, and shap_analysis.py all keep working unchanged):
    models/xgb_rul_model.json   -- the tuned model
    models/best_params.json     -- feature_cols, rul_clip, AND now also
                                    the tuned hyperparameters + CV results
    mlruns/                     -- MLflow's local tracking store (new)

Run:
    python scripts/train_xgboost_optuna.py

Then view results:
    mlflow ui
    (opens http://localhost:5000 -- browse and compare every trial)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
from signal_features import (
    add_rolling_features, add_thermal_stress_features, add_fft_feature,
    KEY_SENSORS_FOR_ROLLING,
)

TRAIN_PATH = os.path.join("data", "cmapss", "processed", "train_FD001.csv")
MODEL_DIR = "models"
NATIVE_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_rul_model.json")
META_PATH = os.path.join(MODEL_DIR, "best_params.json")

RUL_CLIP = 125
N_SPLITS = 5
N_OPTUNA_TRIALS = 20  # ~15-30 min depending on your machine; raise for a more thorough
                       # search if you have time, lower (e.g. 10) for a quicker pass
SEED = 42

MLFLOW_EXPERIMENT_NAME = "edge-ai-pm-xgboost-rul"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_features(df):
    df = add_rolling_features(df, KEY_SENSORS_FOR_ROLLING)
    df = add_thermal_stress_features(df)
    df = add_fft_feature(df)
    return df


def main():
    # Explicit tracking URI -- the default local './mlruns' folder backend
    # is in MLflow's "maintenance mode" and caused a run to silently fail
    # to write its metadata earlier. Pointing both training and `mlflow ui`
    # at the same SQLite file avoids that mismatch entirely.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Loading and engineering features from training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CLIP)
    train_df = build_features(train_df)

    drop_cols = {"unit_number", "time_cycles", "RUL"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    print(f"Using {len(feature_cols)} features")

    X = train_df[feature_cols].values
    y = train_df["RUL"].values
    groups = train_df["unit_number"].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": SEED,
            "n_jobs": -1,
        }
        gkf = GroupKFold(n_splits=N_SPLITS)
        fold_rmses = []
        for train_idx, val_idx in gkf.split(X, y, groups):
            model = xgb.XGBRegressor(**params)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            fold_rmses.append(float(np.sqrt(mean_squared_error(y[val_idx], preds))))
        trial.set_user_attr("fold_rmses", fold_rmses)
        cv_rmse_mean = float(np.mean(fold_rmses))

        # Log this trial as its own nested MLflow run -- shows up as a child
        # of the parent "optuna_search" run, browsable/sortable in the UI.
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", float(np.std(fold_rmses)))

        return cv_rmse_mean

    print(f"\nRunning Optuna search ({N_OPTUNA_TRIALS} trials x {N_SPLITS}-fold GroupKFold CV)...")
    print("This trains 200 models total -- expect several minutes.\n")

    with mlflow.start_run(run_name="optuna_search") as parent_run:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))

        def progress_callback(study, trial):
            if trial.number % 5 == 0 or trial.number == N_OPTUNA_TRIALS - 1:
                print(f"  Trial {trial.number + 1}/{N_OPTUNA_TRIALS} | "
                      f"this trial CV RMSE: {trial.value:.3f} | best so far: {study.best_value:.3f}")

        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[progress_callback])

        best_fold_rmses = study.best_trial.user_attrs["fold_rmses"]
        print(f"\n=== Best hyperparameters (CV RMSE: {study.best_value:.3f}) ===")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print(f"\nPer-fold RMSE: {[round(r, 3) for r in best_fold_rmses]}")
        print(f"Fold RMSE std dev: {np.std(best_fold_rmses):.3f} "
              f"(lower = more consistent across different engines -- a measure of how much to trust the CV mean)")

        # ---- Train final model on ALL training data with best hyperparameters ----
        best_params = {**study.best_params, "random_state": SEED, "n_jobs": -1}
        final_model = xgb.XGBRegressor(**best_params)
        final_model.fit(X, y)

        os.makedirs(MODEL_DIR, exist_ok=True)
        final_model.save_model(NATIVE_MODEL_PATH)

        meta = {
            "feature_cols": feature_cols,
            "rul_clip": RUL_CLIP,
            "tuned_hyperparameters": study.best_params,
            "cv_rmse_mean": round(study.best_value, 3),
            "cv_rmse_std": round(float(np.std(best_fold_rmses)), 3),
            "cv_fold_rmses": [round(r, 3) for r in best_fold_rmses],
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_cv_folds": N_SPLITS,
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

        # ---- Log the winning run on the PARENT run (best params + final model) ----
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_rmse_mean", study.best_value)
        mlflow.log_metric("best_cv_rmse_std", float(np.std(best_fold_rmses)))
        mlflow.log_param("n_optuna_trials", N_OPTUNA_TRIALS)
        mlflow.log_param("n_cv_folds", N_SPLITS)

        # mlflow.xgboost.log_model() infers the current environment's pip
        # package versions to bundle a conda/pip env spec alongside the
        # model -- on some Windows/venv setups this crashes on a broken
        # pip metadata entry unrelated to this project. That packaging is
        # a nice-to-have (lets MLflow reconstruct your exact environment
        # elsewhere), not essential -- the model file itself is already
        # saved to disk via final_model.save_model() above. Try the full
        # version first; if it fails, fall back to a plain artifact copy
        # (no environment introspection, can't fail the same way) so a
        # local packaging quirk never kills the actual tuning results.
        try:
            mlflow.xgboost.log_model(final_model, name="model")
        except Exception as e:
            print(f"\n(Note: mlflow.xgboost.log_model() failed on this environment -- "
                  f"{type(e).__name__}. Falling back to a plain artifact copy instead, "
                  f"which skips the environment-introspection step that failed.)")
            mlflow.log_artifact(NATIVE_MODEL_PATH, artifact_path="model")

        mlflow.log_artifact(META_PATH)

        print(f"\nSaved tuned model to {NATIVE_MODEL_PATH}")
        print(f"Saved metadata (incl. tuned hyperparameters + CV results) to {META_PATH}")
        print(f"Logged {N_OPTUNA_TRIALS} trials + final model to MLflow "
              f"(experiment: '{MLFLOW_EXPERIMENT_NAME}', run: {parent_run.info.run_id})")
        print(f"\nView results with:  mlflow ui")
        print(f"\nNext: re-run evaluate_test.py, quantize_and_benchmark.py, and shap_analysis.py")
        print(f"to refresh all downstream results against this newly-tuned model.")


if __name__ == "__main__":
    main()