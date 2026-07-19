"""
predict_train_fleet.py
-------------------------
Runs the same fleet-wide pipeline demo as predict.py's
demo_on_real_test_engines(), but over the TRAINING data instead of the
real held-out test set.

IMPORTANT HONESTY NOTE: most of X_train was used to actually TRAIN the
LSTM RUL model, the CNN, and the autoencoder. Predicting well on data a
model was trained on is expected and does NOT demonstrate generalization
-- that's what the real test-set evaluation (evaluate_test.py, and
predict.py against X_test) already established (correlation 0.939 vs
true RUL on genuinely unseen engines).

What THIS script adds that's still genuinely useful: it recomputes the
exact same engine-level train/val split used in lstm_rul.py (same seed,
same logic) to identify which ~15 engines were held out as validation
(never trained on) vs which ~85 were actually trained on. It then reports
metrics SEPARATELY for each group -- this shows the real size of the
generalization gap, rather than one blended number that hides it.

Run:
    python scripts/predict_train_fleet.py
"""

import os
import numpy as np
import pandas as pd

from predict import PredictiveMaintenancePipeline, SEQ_DIR
from lstm_rul import VAL_SPLIT, SEED


def recompute_lstm_val_engines(units_train):
    """Reproduces the exact split logic from lstm_rul.py's
    make_train_val_loaders -- same seed, same np.random.default_rng
    permutation -- to identify which engines were genuinely held out
    during training vs which were trained on."""
    unique_units = np.unique(units_train)
    rng = np.random.default_rng(SEED)
    shuffled_units = rng.permutation(unique_units)
    n_val_units = max(1, int(len(unique_units) * VAL_SPLIT))
    val_units = set(shuffled_units[:n_val_units])
    train_units = set(shuffled_units[n_val_units:])
    return train_units, val_units


def summarize_group(df, label):
    print(f"\n--- {label} ({df['engine_id'].nunique()} engines, {len(df)} windows) ---")
    print(df["alert_level"].value_counts().to_string())
    corr = df["predicted_rul"].corr(df["true_rul"])
    print(f"Correlation (predicted_rul vs true_rul): {corr:.4f}")
    rmse = np.sqrt(((df["predicted_rul"] - df["true_rul"]) ** 2).mean())
    print(f"RMSE (on these windows): {rmse:.3f}")


def main():
    print("Loading training sequence arrays...")
    X_train = np.load(os.path.join(SEQ_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(SEQ_DIR, "y_train.npy")).astype(np.float32)
    units_train = np.load(os.path.join(SEQ_DIR, "units_train.npy"))
    print(f"X_train: {X_train.shape}, {len(np.unique(units_train))} engines\n")

    trained_on_units, held_out_units = recompute_lstm_val_engines(units_train)
    print(f"Recomputed split: {len(trained_on_units)} engines actually trained on, "
          f"{len(held_out_units)} engines held out as validation (never trained on).")

    pipeline = PredictiveMaintenancePipeline()

    print("\nRunning pipeline over all training windows (this may take a minute)...")
    results = []
    for i in range(len(X_train)):
        result = pipeline.predict(X_train[i])
        result["engine_id"] = int(units_train[i])
        result["true_rul"] = float(y_train[i])
        result["group"] = "trained_on" if units_train[i] in trained_on_units else "held_out_val"
        results.append(result)

    df = pd.DataFrame(results)

    print("\n=== IMPORTANT: results below are split into two groups ===")
    print("'trained_on' engines: the model saw these during training --")
    print("  strong performance here is EXPECTED and does not prove generalization.")
    print("'held_out_val' engines: genuinely never trained on --")
    print("  this is the fairer comparison, methodologically similar to the")
    print("  real test-set result (correlation 0.939, computed in evaluate_test.py).")

    summarize_group(df[df["group"] == "trained_on"], "TRAINED-ON engines (in-sample)")
    summarize_group(df[df["group"] == "held_out_val"], "HELD-OUT VAL engines (genuinely unseen)")

    print("\n--- For reference: real TEST set result (from earlier) ---")
    print("Correlation (predicted_rul vs true_RUL): 0.939")
    print("Test RMSE: 12.803 (LSTM)")

    out_path = os.path.join("models", "train_fleet_prediction_report.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull report ({len(df)} windows) saved to {out_path}")


if __name__ == "__main__":
    main()
