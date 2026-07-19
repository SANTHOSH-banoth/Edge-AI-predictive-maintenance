"""
tests/test_feature_engineering.py
------------------------------------
Tests signal_features.py's core functions against small, hand-computable
synthetic data -- so the expected values in these tests were worked out
by hand, not copied from the function's own output (which would just
test "does the function agree with itself").

Focus: the per-engine grouping correctness that was manually verified
earlier in this project by code review. These tests turn that one-time
manual check into something that runs automatically and can't silently
break if the functions are ever edited.
"""

import numpy as np
import pandas as pd
import pytest

from signal_features import add_rolling_features, add_thermal_stress_features


@pytest.fixture
def two_engine_df():
    """Two tiny synthetic engines, values chosen so rolling stats and
    thermal stress can be checked by hand. Engine 2's values are
    deliberately very different from engine 1's, so any cross-engine
    leakage in rolling/baseline computation would be obvious.

    6 cycles per engine (not 5): the baseline is computed from the first
    5 cycles, so a jump needs to happen at cycle 6 -- AFTER the baseline
    window closes -- for "cycles within the baseline window show ~0
    stress" to be a valid, uncontaminated assertion. Putting the jump at
    cycle 5 (as an earlier version of this fixture did) means the jump
    itself gets folded into the baseline mean, which is a different,
    equally valid behavior but not what that assertion was checking."""
    return pd.DataFrame({
        "unit_number": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "time_cycles": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
        "sensor_2": [10.0] * 6 + [1000.0] * 6,
        "sensor_3": [20.0] * 6 + [2000.0] * 6,
        "sensor_4": [30.0, 30.0, 30.0, 30.0, 30.0, 40.0] + [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3010.0],
    })


def test_rolling_mean_does_not_leak_across_engines(two_engine_df):
    """The core leakage check: engine 2's rolling mean for its FIRST row
    must be computed using ONLY engine 2's own data (with min_periods=1,
    that's just its own value, 1000.0) -- never influenced by engine 1's
    trailing values (10.0), even though engine 1's rows come immediately
    before engine 2's in the raw dataframe."""
    result = add_rolling_features(two_engine_df, ["sensor_2"], window=5)

    engine2_first_row = result[(result["unit_number"] == 2) & (result["time_cycles"] == 1)]
    rolling_mean = engine2_first_row["sensor_2_roll_mean5"].iloc[0]

    assert rolling_mean == pytest.approx(1000.0), (
        f"Engine 2's first-row rolling mean should be exactly 1000.0 (its own "
        f"value with min_periods=1), but got {rolling_mean} -- this suggests "
        f"engine 1's data leaked into engine 2's window."
    )


def test_thermal_stress_baseline_is_per_engine(two_engine_df):
    """Each engine's thermal stress baseline must come from ITS OWN first
    5 cycles, not a global baseline. Engine 1 and engine 2 have wildly
    different sensor scales (10s vs 1000s) -- if the baseline were global
    or leaked across engines, thermal stress values would be enormous
    and roughly equal for both engines instead of near-zero for cycles
    1-5 (within each engine's own flat baseline window)."""
    result = add_thermal_stress_features(two_engine_df)

    # Engine 1's cycles 1-5 are all identical to its own baseline (the
    # jump happens at cycle 6, outside the baseline window), so thermal
    # stress should be ~0 for cycles 1-5.
    engine1_baseline_period = result[(result["unit_number"] == 1) & (result["time_cycles"] <= 5)]
    assert engine1_baseline_period["thermal_stress_index"].abs().max() < 1e-6, (
        "Engine 1's thermal stress should be ~0 within its own baseline "
        "window (cycles 1-5, before the cycle-6 jump)."
    )

    # Engine 2's cycles 1-5 are ALSO flat relative to ITS OWN baseline
    # (1000s vs 1000s), so thermal stress should also be ~0 -- NOT some
    # huge number reflecting a mismatch with engine 1's much smaller scale.
    engine2_baseline_period = result[(result["unit_number"] == 2) & (result["time_cycles"] <= 5)]
    assert engine2_baseline_period["thermal_stress_index"].abs().max() < 1e-6, (
        "Engine 2's thermal stress should be ~0 relative to ITS OWN "
        "baseline -- a large value here would mean the baseline leaked "
        "from engine 1's much smaller sensor scale."
    )


def test_thermal_stress_baseline_uses_only_first_five_cycles(two_engine_df):
    """Engine 1's sensor_4 jumps from 30.0 to 40.0 at cycle 6 -- AFTER the
    5-cycle baseline window closes. This confirms the baseline is fixed
    from early-life data and does NOT drift to include later cycles
    (which would be a different, and wrong, definition of 'healthy
    baseline' -- one that chases the engine's current state instead of
    anchoring to its original healthy condition)."""
    result = add_thermal_stress_features(two_engine_df)
    engine1 = result[result["unit_number"] == 1].sort_values("time_cycles")

    # Baseline for sensor_4 = mean of cycles 1-5 = 30.0 (flat, jump not
    # included). Cycle 6's deviation = 40 - 30 = 10, weighted by
    # THERMAL_WEIGHTS["sensor_4"] = 0.5 -> 5.0, since sensor_2/sensor_3
    # contribute exactly 0 (flat, no deviation, all 6 cycles).
    cycle6_stress = engine1[engine1["time_cycles"] == 6]["thermal_stress_index"].iloc[0]
    assert cycle6_stress == pytest.approx(5.0, abs=1e-6)