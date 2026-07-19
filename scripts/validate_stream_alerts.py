"""
validate_stream_alerts.py
----------------------------
Open thread #3: sanity-check the alert stream from simulate_edge_stream.py.

Why this matters: simulate_edge_stream.py reported "30 alerts out of 120
timesteps" -- a raw count, with no check on WHEN those alerts fired. A demo
that raises alerts randomly throughout a machine's healthy early life would
be misleading (or worse, would mean the underlying trajectory/model
relationship is broken) even if the total count looks reasonable. A
trustworthy predictive maintenance system should raise alerts that
concentrate near end-of-life, correlate with the degradation signal
(tool wear), and ideally NOT fire during the clearly-healthy early period.

This script checks all three of those properties explicitly, the same
pass/fail style as validate_data.py, rather than just eyeballing the
dashboard.

Run (after simulate_edge_stream.py has produced dashboard/stream_data.json):
    python scripts/validate_stream_alerts.py
"""

import json
import numpy as np

STREAM_PATH = "dashboard/stream_data.json"

EARLY_LIFE_FRACTION = 0.2  # first 20% of the simulated timeline = "should be healthy"


def main():
    with open(STREAM_PATH) as f:
        stream = json.load(f)

    n = len(stream)
    t = np.array([r["t"] for r in stream])
    tool_wear = np.array([r["tool_wear_min"] for r in stream])
    prob = np.array([r["failure_probability"] for r in stream])
    is_alert = np.array([r["prediction"] == "FAILURE RISK" for r in stream])

    print(f"Loaded stream: {n} timesteps, {is_alert.sum()} alerts ({is_alert.mean():.1%} of timeline)\n")

    issues = []

    # --- Check 1: correlation between tool wear (degradation signal) and failure probability ---
    correlation = np.corrcoef(tool_wear, prob)[0, 1]
    print(f"Check 1 -- Correlation(tool_wear, failure_probability): {correlation:.3f}")
    if correlation > 0.5:
        print("  PASS: failure probability rises meaningfully with wear, as expected physically.")
    else:
        issues.append(f"Weak correlation ({correlation:.3f}) between tool wear and predicted failure "
                       f"probability -- expected a strong positive relationship.")
        print(f"  FAIL: {issues[-1]}")

    # --- Check 2: no (or very few) alerts during the clearly-healthy early period ---
    early_cutoff = int(n * EARLY_LIFE_FRACTION)
    early_alerts = is_alert[:early_cutoff].sum()
    print(f"\nCheck 2 -- Alerts in first {EARLY_LIFE_FRACTION:.0%} of timeline (t=0 to {early_cutoff}): {early_alerts}")
    if early_alerts == 0:
        print("  PASS: no false alarms during the clearly-healthy early period.")
    else:
        issues.append(f"{early_alerts} alert(s) fired during the early-life period (t<{early_cutoff}), "
                       f"where the machine should still read as healthy.")
        print(f"  FLAG (not necessarily a bug): {issues[-1]}")
        print("  Note: given this model's real recall/precision tradeoff (see precision_recall_analysis.py),")
        print("  occasional early false alarms are expected behavior, not an error -- but worth naming, not hiding.")

    # --- Check 3: alerts concentrate in the back half of the timeline ---
    first_half_alerts = is_alert[:n // 2].sum()
    second_half_alerts = is_alert[n // 2:].sum()
    print(f"\nCheck 3 -- Alert distribution: {first_half_alerts} in first half, {second_half_alerts} in second half")
    if second_half_alerts >= first_half_alerts:
        print("  PASS: alerts concentrate later in the machine's life, matching real degradation behavior.")
    else:
        issues.append(f"More alerts in the first half ({first_half_alerts}) than the second half "
                       f"({second_half_alerts}) -- alerts should concentrate near end-of-life.")
        print(f"  FAIL: {issues[-1]}")

    # --- Check 4: first alert timing ---
    if is_alert.any():
        first_alert_t = int(t[is_alert][0])
        print(f"\nCheck 4 -- First alert at t={first_alert_t} (of {n} total timesteps, "
              f"{first_alert_t/n:.0%} through the simulated lifecycle)")
    else:
        print("\nCheck 4 -- No alerts raised in this simulation.")

    # --- Summary ---
    print(f"\n=== Summary ===")
    if not issues:
        print("All checks passed. The alert stream behaves coherently: it tracks the degradation")
        print("signal, avoids early false alarms, and concentrates near end-of-life -- consistent")
        print("with a model that's actually learned the failure pattern, not just noise.")
    else:
        print(f"{len(issues)} item(s) worth noting (see above) -- not necessarily bugs, but worth")
        print("being able to explain if asked, rather than only reporting the raw alert count.")


if __name__ == "__main__":
    main()
