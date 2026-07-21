"""
Week 8 — Cost-Based Maintenance Decision Layer
================================================
Translates a raw RUL prediction (+ optional uncertainty) into an actual
recommended maintenance action, and estimates $ saved vs a naive
reactive-maintenance baseline (run to failure). This is the piece that
ties ML output to a business decision — the thing that gets asked about
in an interview, not just "what's your RMSE."

Decision logic (threshold-based, tunable):
    RUL <= critical_threshold      -> "IMMEDIATE: schedule maintenance now"
    RUL <= warning_threshold       -> "SOON: schedule within next window"
    RUL <= watch_threshold         -> "MONITOR: increase inspection frequency"
    RUL >  watch_threshold         -> "OK: no action needed"

Cost model (simple, transparent, tunable — swap in your own $ figures):
    cost_unplanned_failure : cost if the asset fails before maintenance
                              happens (lost production, expedited parts,
                              possible safety incident)
    cost_planned_maintenance : cost of proactively maintaining on schedule
    cost_early_maintenance_waste : cost of maintaining "too early" (wasted
                                    remaining useful life on the part)

Usage
-----
    from maintenance_decision import recommend_action, estimate_savings

    action = recommend_action(predicted_rul=12)
    savings_report = estimate_savings(y_true, y_pred, action_thresholds=...)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ActionThresholds:
    critical: float = 15
    warning: float = 30
    watch: float = 60


@dataclass
class CostModel:
    cost_unplanned_failure: float = 50_000
    cost_planned_maintenance: float = 5_000
    cost_early_maintenance_waste_per_cycle: float = 40  # $ wasted per cycle of unused RUL


def recommend_action(predicted_rul: float, thresholds: ActionThresholds = ActionThresholds()) -> str:
    if predicted_rul <= thresholds.critical:
        return "IMMEDIATE: schedule maintenance now"
    if predicted_rul <= thresholds.warning:
        return "SOON: schedule within next maintenance window"
    if predicted_rul <= thresholds.watch:
        return "MONITOR: increase inspection frequency"
    return "OK: no action needed"


def recommend_actions_batch(
    predicted_rul: np.ndarray, thresholds: ActionThresholds = ActionThresholds()
) -> pd.Series:
    return pd.Series([recommend_action(r, thresholds) for r in predicted_rul])


def estimate_savings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: ActionThresholds = ActionThresholds(),
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    """
    Per-engine estimate of maintenance cost under the model's recommendation,
    vs. a "reactive" baseline where maintenance only happens after failure
    (i.e. every engine incurs cost_unplanned_failure).

    Cases, based on comparing true RUL to the model's action trigger point:
      - Model recommends action AND true RUL was low (asset really was near
        failure): planned maintenance succeeds -> cost_planned_maintenance,
        avoiding cost_unplanned_failure. This is the main savings driver.
      - Model recommends action too early (true RUL was still high): planned
        maintenance happens, but some useful life is wasted ->
        cost_planned_maintenance + waste cost for the unused cycles.
      - Model recommends "OK" (no action) but true RUL was actually low
        (a dangerously late/missed prediction): unplanned failure happens ->
        full cost_unplanned_failure. This is the costly failure mode Week 7's
        error analysis flagged (positive mean_bias = late predictions).
    """
    rows = []
    for true_rul, pred_rul in zip(y_true, y_pred):
        action = recommend_action(pred_rul, thresholds)
        acted = action != "OK: no action needed"

        if acted and true_rul <= thresholds.warning:
            # planned maintenance, and it was warranted -> avoided a failure
            cost = costs.cost_planned_maintenance
            reactive_cost = costs.cost_unplanned_failure
            outcome = "planned_maintenance_success"
        elif acted:
            # planned maintenance, but true RUL was still high -> some waste
            wasted_cycles = max(true_rul - thresholds.warning, 0)
            cost = costs.cost_planned_maintenance + wasted_cycles * costs.cost_early_maintenance_waste_per_cycle
            reactive_cost = costs.cost_unplanned_failure
            outcome = "planned_maintenance_early"
        elif true_rul <= thresholds.critical:
            # model said "OK" but asset was actually near failure -> missed it
            cost = costs.cost_unplanned_failure
            reactive_cost = costs.cost_unplanned_failure
            outcome = "missed_failure_no_savings"
        else:
            # model said "OK" and asset genuinely had plenty of life -> correct, no cost yet
            cost = 0
            reactive_cost = 0
            outcome = "correct_no_action"

        rows.append(
            {
                "true_rul": true_rul,
                "predicted_rul": pred_rul,
                "action": action,
                "outcome": outcome,
                "model_cost": cost,
                "reactive_baseline_cost": reactive_cost,
                "savings": reactive_cost - cost,
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarize_savings(savings_df: pd.DataFrame) -> dict:
    return {
        "total_model_cost": float(savings_df["model_cost"].sum()),
        "total_reactive_baseline_cost": float(savings_df["reactive_baseline_cost"].sum()),
        "total_savings": float(savings_df["savings"].sum()),
        "missed_failures": int((savings_df["outcome"] == "missed_failure_no_savings").sum()),
        "outcome_breakdown": savings_df["outcome"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.uniform(0, 130, n)
    y_pred = y_true + rng.normal(2, 8, n)  # slightly optimistic (late) on average

    thresholds = ActionThresholds(critical=15, warning=30, watch=60)
    costs = CostModel()

    savings_df = estimate_savings(y_true, y_pred, thresholds, costs)
    summary = summarize_savings(savings_df)

    print(savings_df.head(10).to_string(index=False))
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
