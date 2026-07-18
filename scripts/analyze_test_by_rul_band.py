"""
analyze_test_by_rul_band.py
-----------------------------
Follow-up to evaluate_test.py: breaks test RMSE down by true-RUL band, to
verify (rather than just claim) that error concentrates in the high-RUL,
low-urgency region rather than near end-of-life where it would actually
matter operationally.

Reads models/test_set_evaluation.csv (already produced by evaluate_test.py)
— no need to re-run the model.

Run:
    python scripts/analyze_test_by_rul_band.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPORT_PATH = Path("models/test_set_evaluation.csv")

# Bands chosen around operational meaning, not arbitrary quartiles:
#   <30   = near-term / urgent maintenance window
#   30-70 = mid-range, planning horizon
#   70+   = low urgency, RUL_CLIP region where labels were flattened
BANDS = [(0, 30, "urgent (<30)"), (30, 70, "mid-range (30-70)"), (70, 1000, "low-urgency (70+)")]


def main():
    df = pd.read_csv(REPORT_PATH)

    print("=== Test RMSE by true-RUL band ===\n")
    rows = []
    for low, high, label in BANDS:
        subset = df[(df["true_RUL"] >= low) & (df["true_RUL"] < high)]
        if len(subset) == 0:
            continue
        rmse = np.sqrt((subset["abs_error"] ** 2).mean())
        mae = subset["abs_error"].mean()
        rows.append({
            "band": label,
            "n_engines": len(subset),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    overall_rmse = np.sqrt((df["abs_error"] ** 2).mean())
    print(f"\nOverall test RMSE (all {len(df)} engines): {overall_rmse:.2f}")

    urgent = result[result["band"].str.contains("urgent")]
    low_urgency = result[result["band"].str.contains("low-urgency")]
    if len(urgent) and len(low_urgency):
        u_rmse = urgent["rmse"].iloc[0]
        lu_rmse = low_urgency["rmse"].iloc[0]
        print(f"\nUrgent-band RMSE ({u_rmse:.2f}) vs low-urgency-band RMSE ({lu_rmse:.2f}):")
        if u_rmse < lu_rmse:
            print("Confirmed — the model is most accurate exactly where it matters most "
                  "(near end-of-life), and least precise in the low-urgency region where "
                  "RUL_CLIP intentionally flattened resolution.")
        else:
            print("Not confirmed by this test set — urgent-band error is not lower than "
                  "low-urgency-band error. Worth investigating before repeating that claim "
                  "in an interview.")


if __name__ == "__main__":
    main()
