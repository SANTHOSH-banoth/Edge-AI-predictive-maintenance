"""
load_cmapss.py
----------------
Loads the NASA CMAPSS (FD001) turbofan degradation dataset,
assigns proper column names, computes RUL (Remaining Useful Life)
labels, and saves clean processed CSVs for downstream feature
engineering and deep learning models (LSTM / 1D-CNN).

Run from your project root:
    python scripts/load_cmapss.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------
RAW_DIR = os.path.join("data", "cmapss")
PROCESSED_DIR = os.path.join("data", "cmapss", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(RAW_DIR, "train_FD001.txt")
TEST_FILE = os.path.join(RAW_DIR, "test_FD001.txt")
RUL_FILE = os.path.join(RAW_DIR, "RUL_FD001.txt")

# ---------------------------------------------------------------
# 2. Column names
# CMAPSS files have NO headers. Each row is:
#   unit_number, time_cycles, op_setting_1..3, sensor_1..21
# All values are space-separated, with some trailing whitespace
# that creates extra empty columns - we drop those.
# ---------------------------------------------------------------
INDEX_COLS = ["unit_number", "time_cycles"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COL_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_raw_file(path):
    """Load a raw CMAPSS space-separated file into a clean DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Raw files sometimes parse with 2 extra trailing NaN columns
    df = df.iloc[:, : len(COL_NAMES)]
    df.columns = COL_NAMES
    return df


def add_rul_train(df):
    """
    For training data: each engine runs until failure, so RUL at
    each row = (max cycle for that engine) - (current cycle).
    """
    max_cycles = df.groupby("unit_number")["time_cycles"].max()
    max_cycles.name = "max_cycle"
    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df = df.drop(columns=["max_cycle"])
    return df


def add_rul_test(df, rul_file):
    """
    For test data: engines do NOT run to failure - they stop at some
    earlier point. The true RUL at that final cycle is given in
    RUL_FD001.txt (one value per engine, in unit_number order).
    We back-calculate RUL for every row the same way as training,
    but offset by the true final RUL.
    """
    true_rul = pd.read_csv(rul_file, sep=r"\s+", header=None)
    true_rul.columns = ["true_RUL"]
    true_rul["unit_number"] = true_rul.index + 1  # 1-indexed engines

    max_cycles = df.groupby("unit_number")["time_cycles"].max()
    max_cycles.name = "max_cycle_in_test"
    df = df.merge(max_cycles, on="unit_number", how="left")
    df = df.merge(true_rul, on="unit_number", how="left")

    # RUL at any row = (final_test_cycle - current_cycle) + true_RUL_at_end
    df["RUL"] = (df["max_cycle_in_test"] - df["time_cycles"]) + df["true_RUL"]
    df = df.drop(columns=["max_cycle_in_test", "true_RUL"])
    return df


def main():
    print("Loading raw CMAPSS FD001 files...")
    train_df = load_raw_file(TRAIN_FILE)
    test_df = load_raw_file(TEST_FILE)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Engines in train: {train_df['unit_number'].nunique()}")
    print(f"Engines in test:  {test_df['unit_number'].nunique()}")

    print("\nComputing RUL labels...")
    train_df = add_rul_train(train_df)
    test_df = add_rul_test(test_df, RUL_FILE)

    print("\nSample RUL values (train):")
    print(train_df[["unit_number", "time_cycles", "RUL"]].head())

    print("\nRUL distribution (train):")
    print(train_df["RUL"].describe())

    # -------------------------------------------------------------
    # Save processed CSVs
    # -------------------------------------------------------------
    train_out = os.path.join(PROCESSED_DIR, "train_FD001.csv")
    test_out = os.path.join(PROCESSED_DIR, "test_FD001.csv")
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"\nSaved processed train set to {train_out}")
    print(f"Saved processed test set to {test_out}")


if __name__ == "__main__":
    main()
