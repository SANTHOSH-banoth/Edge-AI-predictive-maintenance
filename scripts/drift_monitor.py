"""
Week 7 — Drift Monitoring
=========================
Two things live here:

1. drift simulation: inject gradual "sensor calibration decay" into a
   feature dataframe, mimicking what happens when a physical sensor
   slowly loses accuracy over months of operation.
2. drift detection: compare an incoming (possibly drifted) distribution
   against the training distribution using Population Stability Index
   (PSI) and the Kolmogorov-Smirnov test, per-feature.

Usage
-----
    from drift_monitor import simulate_calibration_decay, compute_drift_report

    drifted_df = simulate_calibration_decay(test_df, sensor_cols=[...], max_bias=0.15, max_scale=1.2)
    report = compute_drift_report(train_df, drifted_df, feature_cols=[...])
    print(report.sort_values("psi", ascending=False).head(10))
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Drift simulation
# ---------------------------------------------------------------------------

def simulate_calibration_decay(
    df: pd.DataFrame,
    sensor_cols: list[str],
    max_bias: float = 0.10,
    max_scale: float = 1.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate gradual sensor calibration decay across the rows of df,
    treating row order as a time axis (e.g. sorted by cycle/timestamp).

    For each sensor, drift ramps linearly from 0 -> 1 across the rows:
        drifted = raw * (1 + t * (scale_factor - 1)) + t * bias_offset

    where t in [0, 1] is the fractional position in the dataframe.
    Each sensor gets its own randomly-drawn bias/scale severity so the
    decay isn't uniform across channels — this mirrors real hardware,
    where some sensors degrade faster than others.

    Parameters
    ----------
    df : the clean dataframe to corrupt (a copy is returned, df is untouched)
    sensor_cols : which columns to drift
    max_bias : max additive offset at t=1, as a fraction of each column's std
    max_scale : max multiplicative scale factor at t=1 (e.g. 1.15 = +15%)
    seed : rng seed for reproducibility

    Returns
    -------
    A new dataframe with drifted sensor columns plus a `drift_t` column
    recording the simulated time fraction (useful for validating you
    detect drift progressively, not just at the end).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    t = np.linspace(0, 1, n)
    out["drift_t"] = t

    for col in sensor_cols:
        col_std = out[col].std() or 1.0
        bias_severity = rng.uniform(0.3, 1.0) * max_bias * col_std
        scale_severity = 1 + rng.uniform(0.3, 1.0) * (max_scale - 1)
        scale_factor = 1 + t * (scale_severity - 1)
        bias_offset = t * bias_severity
        out[col] = out[col] * scale_factor + bias_offset

    return out


# ---------------------------------------------------------------------------
# 2. Drift detection
# ---------------------------------------------------------------------------

def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """
    Classic PSI. Rule of thumb thresholds:
        < 0.1  : no significant drift
        0.1-0.25 : moderate drift, worth watching
        > 0.25 : significant drift, retrain/investigate
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # avoid divide-by-zero / log(0)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_drift_report(
    train_df: pd.DataFrame,
    incoming_df: pd.DataFrame,
    feature_cols: list[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """
    Per-feature drift report combining PSI and a two-sample KS test.

    Returns a dataframe sorted by PSI descending, with columns:
        feature, psi, psi_flag, ks_stat, ks_pvalue, ks_flag
    """
    rows = []
    for col in feature_cols:
        expected = train_df[col].dropna().to_numpy()
        actual = incoming_df[col].dropna().to_numpy()

        psi = population_stability_index(expected, actual, bins=psi_bins)
        ks_stat, ks_p = stats.ks_2samp(expected, actual)

        rows.append(
            {
                "feature": col,
                "psi": round(psi, 4),
                "psi_flag": _psi_flag(psi),
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 6),
                "ks_flag": "DRIFT" if ks_p < 0.01 else "ok",
            }
        )

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def _psi_flag(psi: float) -> str:
    if psi < 0.1:
        return "ok"
    if psi < 0.25:
        return "moderate"
    return "significant"


if __name__ == "__main__":
    # Minimal smoke test with synthetic data so this runs standalone.
    rng = np.random.default_rng(0)
    n = 2000
    train = pd.DataFrame(
        {
            "sensor_1": rng.normal(50, 5, n),
            "sensor_2": rng.normal(100, 10, n),
            "sensor_3": rng.uniform(0, 1, n),
        }
    )
    test_clean = pd.DataFrame(
        {
            "sensor_1": rng.normal(50, 5, n),
            "sensor_2": rng.normal(100, 10, n),
            "sensor_3": rng.uniform(0, 1, n),
        }
    )

    drifted = simulate_calibration_decay(
        test_clean, sensor_cols=["sensor_1", "sensor_2", "sensor_3"]
    )

    report = compute_drift_report(
        train, drifted, feature_cols=["sensor_1", "sensor_2", "sensor_3"]
    )
    print(report)
