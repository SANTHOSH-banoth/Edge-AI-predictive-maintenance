"""
Week 7 — Adversarial / Robustness Testing
==========================================
Feeds a trained model deliberately corrupted inputs (Gaussian noise at
increasing severity, and missing readings via masking) and measures how
gracefully it degrades. A model that falls off a cliff at 5% missing
data is not edge-deployment-ready, even if its clean-data RMSE is great.

This is model-agnostic: pass in any `predict_fn(X) -> np.ndarray`, so it
works the same for your RandomForest, XGBoost, LSTM, or CNN wrapped in
whatever predict interface you already have.

Usage
-----
    from robustness_test import noise_robustness_curve, missingness_robustness_curve

    noise_report = noise_robustness_curve(model.predict, X_test, y_test)
    missing_report = missingness_robustness_curve(model.predict, X_test, y_test)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def _rmse(y_true, y_pred) -> float:
    # guard against NaN predictions (a model that fails ungracefully
    # should show up as NaN/inf here, not silently get skipped)
    if np.any(~np.isfinite(y_pred)):
        return float("inf")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def noise_robustness_curve(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    noise_levels: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add Gaussian noise scaled to each feature's std, at increasing
    severity levels (as a fraction of that std), and measure RMSE decay.

    noise_levels: e.g. [0, 0.05, 0.1, 0.25, 0.5, 1.0]
        0    -> clean baseline
        0.5  -> noise with std = 50% of each feature's own std
    """
    noise_levels = noise_levels if noise_levels is not None else [0, 0.05, 0.1, 0.25, 0.5, 1.0]
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    feature_std = X.std(axis=0)
    baseline_rmse = None

    rows = []
    for level in noise_levels:
        noise = rng.normal(0, feature_std * level, size=X.shape)
        X_noisy = X + noise
        y_pred = predict_fn(X_noisy)
        rmse = _rmse(y_true, y_pred)
        if level == 0:
            baseline_rmse = rmse
        rows.append(
            {
                "noise_level": level,
                "rmse": round(rmse, 3),
                "rmse_increase_pct": round(
                    100 * (rmse - baseline_rmse) / baseline_rmse, 1
                )
                if baseline_rmse
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def missingness_robustness_curve(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y_true: np.ndarray,
    missing_fractions: list[float] | None = None,
    impute_value: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly mask an increasing fraction of feature values (per-cell,
    not per-row) and impute with `impute_value` (defaults to 0, i.e.
    "sensor stopped reporting"). Measures RMSE and prediction-failure
    rate (NaN/inf outputs) as missingness increases.

    missing_fractions: e.g. [0, 0.01, 0.05, 0.1, 0.2, 0.4]
    """
    missing_fractions = (
        missing_fractions if missing_fractions is not None else [0, 0.01, 0.05, 0.1, 0.2, 0.4]
    )
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    baseline_rmse = None

    rows = []
    for frac in missing_fractions:
        mask = rng.random(X.shape) < frac
        X_missing = X.copy()
        X_missing[mask] = impute_value
        y_pred = predict_fn(X_missing)
        rmse = _rmse(y_true, y_pred)
        failure_rate = float(np.mean(~np.isfinite(y_pred))) if y_pred is not None else 1.0
        if frac == 0:
            baseline_rmse = rmse
        rows.append(
            {
                "missing_fraction": frac,
                "rmse": round(rmse, 3),
                "rmse_increase_pct": round(
                    100 * (rmse - baseline_rmse) / baseline_rmse, 1
                )
                if baseline_rmse and np.isfinite(rmse)
                else (0.0 if frac == 0 else float("inf")),
                "prediction_failure_rate": failure_rate,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test with a toy linear model so this runs standalone.
    rng = np.random.default_rng(0)
    n, d = 1000, 5
    X = rng.normal(0, 1, (n, d))
    true_w = rng.normal(0, 2, d)
    y = X @ true_w + rng.normal(0, 0.5, n)

    def toy_predict(X_in):
        return X_in @ true_w  # a "perfect" linear model on clean data

    print(noise_robustness_curve(toy_predict, X, y))
    print()
    print(missingness_robustness_curve(toy_predict, X, y))
