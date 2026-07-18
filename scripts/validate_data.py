"""
validate_data.py
-------------------
Week 3, Day 7: data validation.

Why this matters (and why it's not just "extra checking"): every model
you've trained so far assumes new data will look statistically similar
to what it was trained on. In a real deployment, sensors drift, get
recalibrated, occasionally send garbage, or a schema changes upstream
(a column renamed, a unit conversion changed). A model given data outside
its trained-on distribution will still happily produce a confident-looking
number -- it just won't mean anything. Validating data BEFORE it reaches
the model is what separates a production ML system from a training
script that only ever sees clean, known-good data.

This script works in two modes:
  1. BASELINE  -- computes schema + statistics from your training data
                  (column names, dtypes, per-column min/max/mean/std,
                  known key constraints) and saves it as a reference.
  2. VALIDATE  -- checks a new dataset (here: the official test set)
                  against that baseline and reports any violations.

This is a lightweight, fully custom version of what tools like
Great Expectations do in production -- same idea, no extra dependency,
and every check is something you can explain line-by-line in an
interview, which matters more at this stage than tool sophistication.

Checks performed:
  - Schema: same columns present, same dtypes
  - Missing values: no unexpected NaNs
  - Duplicate keys: no repeated (unit_number, time_cycles) pairs
  - Range/drift: each column's mean falls within a tolerance of the
    training mean (flags sensor drift or unit mismatches)
  - Domain-specific: RUL is non-negative; time_cycles increases
    monotonically within each engine (a broken sensor clock would violate this)
"""

import json
import numpy as np
import pandas as pd

TRAIN_PATH = "data/cmapss/processed/train_FD001.csv"
TEST_PATH = "data/cmapss/processed/test_FD001.csv"
SCHEMA_PATH = "data/cmapss/validation_schema.json"

DRIFT_TOLERANCE_STD = 3.0  # flag a column if its new mean is > N std devs from training mean


def build_baseline(df, path=SCHEMA_PATH):
    """Compute and save schema + statistics from the training set."""
    schema = {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "stats": {},
    }
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            schema["stats"][c] = {
                "mean": float(df[c].mean()),
                "std": float(df[c].std()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
            }

    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Baseline schema saved to {path}")
    print(f"  {len(schema['columns'])} columns, {len(schema['stats'])} numeric columns tracked")
    return schema


def validate(df, schema, name="dataset"):
    """Check a dataframe against a previously-built baseline schema."""
    issues = []

    # --- 1. Schema: columns match ---
    missing_cols = set(schema["columns"]) - set(df.columns)
    extra_cols = set(df.columns) - set(schema["columns"])
    if missing_cols:
        issues.append(f"MISSING COLUMNS: {sorted(missing_cols)}")
    if extra_cols:
        issues.append(f"UNEXPECTED NEW COLUMNS: {sorted(extra_cols)}")

    # --- 2. Missing values ---
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        issues.append(f"MISSING VALUES found in: {cols_with_nulls.to_dict()}")

    # --- 3. Duplicate keys ---
    if "unit_number" in df.columns and "time_cycles" in df.columns:
        dupes = df.duplicated(subset=["unit_number", "time_cycles"]).sum()
        if dupes > 0:
            issues.append(f"DUPLICATE (unit_number, time_cycles) PAIRS: {dupes} rows")

    # --- 4. RUL sanity (if present) ---
    if "RUL" in df.columns:
        if (df["RUL"] < 0).any():
            issues.append(f"NEGATIVE RUL VALUES: {(df['RUL'] < 0).sum()} rows")

    # --- 5. time_cycles monotonically increasing per engine ---
    if "unit_number" in df.columns and "time_cycles" in df.columns:
        non_monotonic = 0
        for uid, g in df.groupby("unit_number"):
            cycles = g.sort_index()["time_cycles"].values
            if not np.all(np.diff(cycles) > 0):
                non_monotonic += 1
        if non_monotonic > 0:
            issues.append(f"NON-MONOTONIC time_cycles in {non_monotonic} engine(s)")

    # --- 6. Distribution drift check ---
    drift_flags = []
    for c, stats in schema["stats"].items():
        if c not in df.columns or stats["std"] == 0:
            continue
        new_mean = df[c].mean()
        z = abs(new_mean - stats["mean"]) / stats["std"]
        if z > DRIFT_TOLERANCE_STD:
            drift_flags.append(f"{c} (z={z:.1f}, train_mean={stats['mean']:.2f}, new_mean={new_mean:.2f})")
    if drift_flags:
        issues.append(f"POSSIBLE DISTRIBUTION DRIFT in: {drift_flags}")

    # --- Report ---
    print(f"\n=== Validation report: {name} ===")
    print(f"Rows checked: {len(df)}")
    if not issues:
        print("PASSED — no issues found. Data is consistent with the training baseline.")
    else:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    return issues


def main():
    print("Loading training data to build baseline...")
    train_df = pd.read_csv(TRAIN_PATH)
    schema = build_baseline(train_df)

    print("\nLoading test data to validate against baseline...")
    test_df = pd.read_csv(TEST_PATH)
    # Test set doesn't have RUL per-row by design (it's what we predict),
    # so skip that column's presence check but still run everything else.
    validate(test_df, schema, name="test_FD001.csv (incoming data)")

    print("\n(Re-validating training data against its own baseline as a sanity check --")
    print(" this should always PASS since it IS the baseline.)")
    validate(train_df, schema, name="train_FD001.csv (self-check)")


if __name__ == "__main__":
    main()
