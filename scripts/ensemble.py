"""
Week 8 — Ensembling
=====================
Combines predictions from multiple trained models (LSTM + CNN + XGBoost,
or any subset) and checks whether the ensemble actually beats every
single model — it doesn't always, and reporting that honestly (like
your Week 6 quantization write-up did) is more credible than assuming
it must help.

Three combination strategies:
  - simple average
  - weighted average (weights inversely proportional to each model's
    individual RMSE, so more accurate models count more)
  - stacked ensemble (a linear meta-learner trained on the base models'
    predictions — learns its own weights instead of using an RMSE heuristic)

Usage
-----
    from ensemble import compare_ensembles

    # predictions: dict of {model_name: np.ndarray of predictions on same y_test}
    predictions = {"xgboost": xgb_preds, "lstm": lstm_preds, "cnn": cnn_preds}
    report = compare_ensembles(predictions, y_test)
    print(report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def simple_average(predictions: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.column_stack(list(predictions.values()))
    return stacked.mean(axis=1)


def weighted_average(
    predictions: dict[str, np.ndarray], y_true: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Weight = 1/RMSE for each model, normalized to sum to 1. More accurate
    models (lower individual RMSE) get more say in the final prediction.
    """
    individual_rmse = {name: _rmse(y_true, preds) for name, preds in predictions.items()}
    inv = {name: 1.0 / r for name, r in individual_rmse.items()}
    total = sum(inv.values())
    weights = {name: v / total for name, v in inv.items()}

    combined = np.zeros_like(y_true, dtype=float)
    for name, preds in predictions.items():
        combined += weights[name] * preds
    return combined, weights


def stacked_ensemble(
    predictions: dict[str, np.ndarray], y_true: np.ndarray, n_splits: int = 5, seed: int = 42
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Out-of-fold stacking: trains a Ridge meta-learner on the base models'
    predictions to learn combination weights, using K-fold CV so the
    meta-learner's own reported performance isn't inflated by fitting on
    the same rows it's evaluated on.

    Returns the out-of-fold stacked predictions plus the meta-learner's
    final coefficients (fit on all data, for interpretability — which
    model does the meta-learner trust most?).
    """
    X = np.column_stack(list(predictions.values()))
    names = list(predictions.keys())
    n = len(y_true)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros(n)

    for train_idx, val_idx in kf.split(X):
        meta = Ridge(alpha=1.0, positive=True)  # positive=True: no model gets a negative weight
        meta.fit(X[train_idx], y_true[train_idx])
        oof_preds[val_idx] = meta.predict(X[val_idx])

    # Fit once more on everything just to report interpretable final weights
    final_meta = Ridge(alpha=1.0, positive=True)
    final_meta.fit(X, y_true)
    coefs = dict(zip(names, final_meta.coef_))

    return oof_preds, coefs


def compare_ensembles(
    predictions: dict[str, np.ndarray], y_true: np.ndarray
) -> pd.DataFrame:
    """
    Builds a comparison table: each individual model's RMSE, plus all
    three ensemble strategies' RMSE, sorted best-first. Use this to
    honestly answer "does ensembling actually help here?"
    """
    rows = []
    for name, preds in predictions.items():
        rows.append({"method": name, "rmse": round(_rmse(y_true, preds), 3), "type": "individual"})

    avg_preds = simple_average(predictions)
    rows.append({"method": "ensemble_simple_avg", "rmse": round(_rmse(y_true, avg_preds), 3), "type": "ensemble"})

    weighted_preds, weights = weighted_average(predictions, y_true)
    rows.append({"method": "ensemble_weighted_avg", "rmse": round(_rmse(y_true, weighted_preds), 3), "type": "ensemble"})

    stacked_preds, coefs = stacked_ensemble(predictions, y_true)
    rows.append({"method": "ensemble_stacked", "rmse": round(_rmse(y_true, stacked_preds), 3), "type": "ensemble"})

    report = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    report.attrs["weighted_avg_weights"] = weights
    report.attrs["stacked_coefs"] = coefs
    return report


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    y_true = rng.uniform(0, 130, n)

    # simulate 3 models with different error characteristics
    xgb_preds = y_true + rng.normal(0, 8, n)
    lstm_preds = y_true + rng.normal(1, 6, n)   # slightly biased but tighter
    cnn_preds = y_true + rng.normal(-2, 11, n)  # noisier

    predictions = {"xgboost": xgb_preds, "lstm": lstm_preds, "cnn": cnn_preds}
    report = compare_ensembles(predictions, y_true)
    print(report.to_string(index=False))
    print("\nWeighted-avg weights:", report.attrs["weighted_avg_weights"])
    print("Stacked meta-learner coefficients:", report.attrs["stacked_coefs"])
