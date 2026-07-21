"""
Week 7 — Per-Failure-Mode Error Analysis
=========================================
Aggregate metrics hide *where* a model fails. This breaks predictions
down by failure mode / operating regime so you can say something like
"the LSTM is great on bearing wear but falls apart on thermal faults"
instead of just "RMSE = 12.3".

Works for both:
  - RUL regression (continuous target): buckets by RUL range
  - Classification (failure_mode label): breaks down by class

Usage
-----
    from error_analysis import regression_error_by_mode, classification_error_by_mode

    report = regression_error_by_mode(y_true, y_pred, failure_mode=df["failure_mode"])
    report_by_rul_bucket = regression_error_by_rul_bucket(y_true, y_pred)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_recall_fscore_support,
)


def regression_error_by_mode(
    y_true: np.ndarray, y_pred: np.ndarray, failure_mode: pd.Series
) -> pd.DataFrame:
    """RMSE/MAE/bias broken down per failure mode, for RUL regression models."""
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "failure_mode": failure_mode.to_numpy()}
    )
    rows = []
    for mode, group in df.groupby("failure_mode"):
        err = group["y_pred"] - group["y_true"]
        rows.append(
            {
                "failure_mode": mode,
                "n": len(group),
                "rmse": round(float(np.sqrt(mean_squared_error(group.y_true, group.y_pred))), 3),
                "mae": round(float(mean_absolute_error(group.y_true, group.y_pred)), 3),
                "mean_bias": round(float(err.mean()), 3),  # +ve = overpredicts RUL (dangerous: late warning)
            }
        )
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def regression_error_by_rul_bucket(
    y_true: np.ndarray, y_pred: np.ndarray, bins: list[int] | None = None
) -> pd.DataFrame:
    """
    Error broken down by true-RUL bucket. The end-of-life buckets (low RUL)
    matter most operationally — that's where a late/optimistic prediction
    costs you an actual failure instead of a maintenance window.
    """
    bins = bins or [0, 15, 30, 60, 100, 10_000]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bucket"] = pd.cut(df["y_true"], bins=bins, labels=labels, right=False)

    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        if len(group) == 0:
            continue
        err = group["y_pred"] - group["y_true"]
        rows.append(
            {
                "rul_bucket": bucket,
                "n": len(group),
                "rmse": round(float(np.sqrt(mean_squared_error(group.y_true, group.y_pred))), 3),
                "mean_bias": round(float(err.mean()), 3),
                "pct_dangerously_late": round(
                    float((err > 5).mean() * 100), 1
                ),  # predicted RUL > 5 cycles more than truth
            }
        )
    return pd.DataFrame(rows)


def classification_error_by_mode(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> pd.DataFrame:
    """Precision/recall/F1 per class, for a failure-mode classifier / anomaly detector."""
    labels = sorted(set(y_true) | set(y_pred))
    names = class_names or [str(l) for l in labels]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": names,
            "precision": np.round(precision, 3),
            "recall": np.round(recall, 3),
            "f1": np.round(f1, 3),
            "support": support,
        }
    ).sort_values("f1")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(1, 130, n)
    modes = rng.choice(["bearing_wear", "thermal_fault", "corrosion"], n)
    # simulate the model being worse on thermal faults
    noise = np.where(modes == "thermal_fault", rng.normal(8, 6, n), rng.normal(0, 3, n))
    y_pred = y_true + noise

    print(regression_error_by_mode(y_true, y_pred, pd.Series(modes)))
    print()
    print(regression_error_by_rul_bucket(y_true, y_pred))
