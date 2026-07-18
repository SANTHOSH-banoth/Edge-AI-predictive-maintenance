"""
build_sequences.py
--------------------
Converts the flat, processed CMAPSS train/test CSVs into 3D sliding-
window sequences suitable for deep learning models (LSTM / 1D-CNN).

Each sample becomes:
    X: (window_length, num_features)   -> a short history of sensor readings
    y: RUL value at the END of that window

NOTE on target scale: y is saved here in RAW cycle units (0-125, after
capping). This is intentional -- keeping this file's output units simple
and consistent for both the CNN and LSTM. The CNN's BatchNorm layers
already normalize internal activations regardless of target scale, so it
trains fine on raw RUL. The LSTM does NOT have that protection, so
lstm_rul.py normalizes y (divides by RUL_CAP) internally right before
training, and multiplies predictions back by RUL_CAP before evaluating.
If you build another sequence model later (e.g. a Transformer), check
whether it needs the same target-normalization treatment as the LSTM.

Run from your project root (after load_cmapss.py has been run once):
    python scripts/build_sequences.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ---------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "cmapss", "processed")
SEQ_DIR = os.path.join("data", "cmapss", "sequences")
os.makedirs(SEQ_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train_FD001.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test_FD001.csv")

WINDOW_LENGTH = 30      # look at last 30 cycles to predict RUL
RUL_CAP = 125           # standard CMAPSS practice: clip RUL so the model
                        # isn't asked to distinguish "300 cycles left" vs
                        # "280 cycles left" - both just mean "healthy"
                        # (lstm_rul.py's RUL_SCALE constant must match this)

SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]


def drop_flat_sensors(df, sensor_cols, threshold=1e-5):
    """
    Some CMAPSS sensors barely change across the whole dataset (dead/
    uninformative sensors). Drop any sensor whose std is near zero -
    it can't help the model and just adds noise.
    """
    stds = df[sensor_cols].std()
    keep_cols = stds[stds > threshold].index.tolist()
    dropped = [c for c in sensor_cols if c not in keep_cols]
    print(f"Dropping near-constant sensors: {dropped}")
    return keep_cols


def make_windows(df, feature_cols, window_length, unit_col="unit_number",
                  cycle_col="time_cycles", label_col="RUL"):
    """
    For each engine, slide a window of length `window_length` over its
    cycles. Each window's label is the RUL at the LAST cycle in that
    window. Engines shorter than window_length are skipped (can't form
    a full window) - this only affects a handful of edge-case units.
    """
    X_list, y_list, unit_list = [], [], []

    for unit_id, group in df.groupby(unit_col):
        group = group.sort_values(cycle_col)
        data = group[feature_cols].values
        labels = group[label_col].values

        n_cycles = len(group)
        if n_cycles < window_length:
            continue  # too short to form even one window

        for start in range(0, n_cycles - window_length + 1):
            end = start + window_length
            X_list.append(data[start:end])
            y_list.append(labels[end - 1])  # label at end of window
            unit_list.append(unit_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    units = np.array(unit_list)
    return X, y, units


def make_test_windows_last_only(df, feature_cols, window_length,
                                 unit_col="unit_number",
                                 cycle_col="time_cycles",
                                 label_col="RUL"):
    """
    For the TEST set we only need ONE window per engine: the most
    recent `window_length` cycles, used to predict RUL "right now".
    Engines with fewer than window_length cycles are padded at the
    front by repeating their earliest reading (simple, standard
    approach for short sequences).
    """
    X_list, y_list, unit_list = [], [], []

    for unit_id, group in df.groupby(unit_col):
        group = group.sort_values(cycle_col)
        data = group[feature_cols].values
        label = group[label_col].values[-1]  # RUL at final known cycle

        n_cycles = len(group)
        if n_cycles >= window_length:
            window = data[-window_length:]
        else:
            pad_len = window_length - n_cycles
            pad = np.repeat(data[0:1], pad_len, axis=0)
            window = np.vstack([pad, data])

        X_list.append(window)
        y_list.append(label)
        unit_list.append(unit_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    units = np.array(unit_list)
    return X, y, units


def main():
    print("Loading processed CMAPSS data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # -------------------------------------------------------------
    # Drop near-constant sensors (based on TRAIN stats only, then
    # apply same column set to test - never let test data influence
    # this decision)
    # -------------------------------------------------------------
    kept_sensors = drop_flat_sensors(train_df, SENSOR_COLS)
    feature_cols = SETTING_COLS + kept_sensors
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    # -------------------------------------------------------------
    # Cap RUL (standard CMAPSS practice)
    # -------------------------------------------------------------
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CAP)
    test_df["RUL"] = test_df["RUL"].clip(upper=RUL_CAP)

    # -------------------------------------------------------------
    # Scale features - fit ONLY on train, apply to both
    # -------------------------------------------------------------
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    scaler_path = os.path.join(SEQ_DIR, "sequence_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")

    # -------------------------------------------------------------
    # Build sliding windows
    # -------------------------------------------------------------
    print(f"\nBuilding training windows (length={WINDOW_LENGTH})...")
    X_train, y_train, units_train = make_windows(
        train_df, feature_cols, WINDOW_LENGTH
    )
    print(f"X_train shape: {X_train.shape}  (samples, window, features)")
    print(f"y_train shape: {y_train.shape}")

    print(f"\nBuilding test windows (last window per engine)...")
    X_test, y_test, units_test = make_test_windows_last_only(
        test_df, feature_cols, WINDOW_LENGTH
    )
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # -------------------------------------------------------------
    # Save everything as .npy for fast loading in training scripts
    # -------------------------------------------------------------
    np.save(os.path.join(SEQ_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(SEQ_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SEQ_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(SEQ_DIR, "y_test.npy"), y_test)

    # Save feature column order too - the LSTM/CNN scripts need this
    with open(os.path.join(SEQ_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nAll sequence arrays saved to {SEQ_DIR}/")
    print("Ready for LSTM / 1D-CNN training.")


if __name__ == "__main__":
    main()