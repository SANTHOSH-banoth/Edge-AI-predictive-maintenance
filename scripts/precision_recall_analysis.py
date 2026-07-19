"""
precision_recall_analysis.py
-------------------------------
Open thread #2: a real, quantitative precision/recall tradeoff analysis
for the ORIGINAL edge-MLP model from Week 1 (the AI4I-style synthetic
predictive maintenance pipeline -- the one deployed via ONNX with a
1,276x size reduction vs the cloud RandomForest).

Why this matters: the original project reported precision 57.6% / recall
93.0% at the DEFAULT classification threshold (0.5) without examining
whether that's actually the right operating point. A classifier's
threshold is a free dial -- moving it trades recall for precision in
either direction. This script sweeps that dial and makes the choice
explicit and defensible, instead of silently accepting sklearn's default.

The business reasoning (stated once here, not repeated per threshold):
  - False negative (missed real failure): unplanned downtime, potential
    safety issue, most expensive outcome by far.
  - False positive (false alarm): someone inspects a machine that turns
    out to be fine. Costs an inspection, nothing more.
  Given that asymmetry, recall should be weighted much more heavily than
  precision -- which is exactly why the model was tuned toward recall in
  the first place (via SMOTE + threshold choice). This script quantifies
  HOW MUCH precision that recall is costing, across the whole threshold
  range, so the choice is backed by numbers instead of stated by feel.

Run:
    python scripts/precision_recall_analysis.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

MODEL_DIR = "models"
DATA_PATH = "data/machine_sensor_data.csv"

THRESHOLDS_TO_REPORT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def rebuild_test_set():
    """Reconstruct the EXACT same test split used in train_model.py
    (same feature engineering, same random_state, same stratify) so this
    analysis evaluates on the real held-out test data, not a new split."""
    df = pd.read_csv(DATA_PATH)

    df["Temp_diff_K"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["Power_W"] = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)
    df["Wear_Torque_Product"] = df["Tool_wear_min"] * df["Torque_Nm"]

    type_encoder = joblib.load(f"{MODEL_DIR}/type_encoder.pkl")
    df["Type_encoded"] = type_encoder.transform(df["Type"])

    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_cols = json.load(f)

    X = df[feature_cols].values
    y = df["Machine_failure"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test


def main():
    print("Rebuilding the original held-out test set...")
    X_test, y_test = rebuild_test_set()
    print(f"Test set: {len(X_test)} samples, {y_test.sum()} real failures ({y_test.mean():.1%})")

    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    edge_model = joblib.load(f"{MODEL_DIR}/edge_model.pkl")

    X_test_scaled = scaler.transform(X_test)
    probs = edge_model.predict_proba(X_test_scaled)[:, 1]

    # ---- Full precision-recall curve ----
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, probs)
    avg_precision = average_precision_score(y_test, probs)

    print(f"\nAverage Precision (area under PR curve): {avg_precision:.3f}")

    # ---- Report at specific, interpretable thresholds ----
    print(f"\n{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}{'Missed failures':<18}{'False alarms':<15}")
    rows = []
    for t in THRESHOLDS_TO_REPORT:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_test, preds, zero_division=0)
        rows.append({"threshold": t, "precision": precision, "recall": recall,
                      "f1": f1, "missed_failures": fn, "false_alarms": fp})
        print(f"{t:<12}{precision:<12.3f}{recall:<12.3f}{f1:<10.3f}{fn:<18}{fp:<15}")

    df_thresh = pd.DataFrame(rows)
    df_thresh.to_csv(f"{MODEL_DIR}/precision_recall_thresholds.csv", index=False)

    # ---- Explicit recommendation, grounded in the numbers above ----
    default_row = df_thresh[df_thresh["threshold"] == 0.5].iloc[0]
    lower_row = df_thresh[df_thresh["threshold"] == 0.3].iloc[0]

    print(f"\n=== Recommendation ===")
    print(f"Current default (0.5): {int(default_row['missed_failures'])} missed failures, "
          f"{int(default_row['false_alarms'])} false alarms, recall {default_row['recall']:.1%}")
    print(f"Lower threshold (0.3): {int(lower_row['missed_failures'])} missed failures, "
          f"{int(lower_row['false_alarms'])} false alarms, recall {lower_row['recall']:.1%}")

    extra_alarms = int(lower_row["false_alarms"] - default_row["false_alarms"])
    fewer_misses = int(default_row["missed_failures"] - lower_row["missed_failures"])

    if fewer_misses > 0:
        print(f"\nLowering the threshold to 0.3 catches {fewer_misses} more real failure(s), "
              f"at the cost of {extra_alarms} additional false alarm(s).")
        print("Given a missed failure costs far more than an unnecessary inspection, "
              "this is a reasonable trade to make -- recommend threshold 0.3 over the default 0.5.")
    else:
        print(f"\nAt this test set size, threshold 0.3 doesn't catch additional failures "
              f"over 0.5 (small test set -- {int(y_test.sum())} total failures means single-sample "
              f"differences can look like 'no change'). Default 0.5 remains a reasonable choice; "
              f"in a larger production dataset this threshold sweep should be re-run to confirm.")

    print(f"\nFull threshold table saved to {MODEL_DIR}/precision_recall_thresholds.csv")


if __name__ == "__main__":
    main()
