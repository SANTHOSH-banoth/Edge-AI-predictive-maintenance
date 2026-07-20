"""
src/models/lstm_rul.py

LSTM-based Remaining Useful Life (RUL) regression model for CMAPSS turbofan data.

Loads pre-built sequence windows (from scripts/build_sequences.py), trains a
small 2-layer LSTM with dropout, evaluates with RMSE and the CMAPSS
asymmetric scoring function, and saves the trained model.

FIX APPLIED #1 (RUL scale / gradient stability): the original version
trained on raw RUL values (0-125) with no gradient clipping. For an LSTM
specifically, this combination is unstable -- gradients flowing back
through many recurrent time steps at that target scale tend to collapse
the model into just predicting the mean RUL for every input (which
trivially minimizes MSE without learning anything from the sensors). Two
small fixes solve it:
  1. Normalize the RUL target to [0, 1] during training (divide by
     RUL_SCALE), then multiply predictions back before evaluating/scoring.
  2. Clip gradients to a max norm of 1.0 after backward(), before the
     optimizer step -- standard practice for RNNs/LSTMs, prevents any
     single bad batch from throwing the weights into a bad region.
Neither the architecture nor the CNN needed this fix; the CNN's BatchNorm
layers already kept internal activations normalized regardless of target
scale, which is part of what BatchNorm is for.

FIX APPLIED #2 (train/val split leakage): the original make_train_val_loaders
split TRAINING WINDOWS randomly by index. Since consecutive sliding windows
overlap by up to (window_length - 1) timesteps, a random split put
near-duplicate windows from the SAME engine into both train and val --
validation loss was measuring memorization, not generalization, and early
stopping was triggering on a leaky signal. This version splits by ENGINE
(unit_number) instead, using units_train.npy (saved by build_sequences.py),
so no engine's windows appear in both sets. Same fix applied to cnn_rul.py.

All hyperparameters/metrics/artifacts also logged to MLflow (experiment:
"week2_deep_learning_rul") so this run is comparable side-by-side with
every other model trained in this project.

Usage:
    python src/models/lstm_rul.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_DIR = os.path.join("data", "cmapss", "sequences")
MODEL_OUT_DIR = os.path.join("models")
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "lstm_rul.pt")

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
VAL_SPLIT = 0.15
SEED = 42

RUL_SCALE = 125.0        # must match RUL_CAP in build_sequences.py
GRAD_CLIP_MAX_NORM = 1.0  # standard for LSTMs -- prevents exploding gradients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_EXPERIMENT_NAME = "week2_deep_learning_rul"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sequences(seq_dir=SEQ_DIR):
    print("Loading sequence arrays...")
    X_train = np.load(os.path.join(seq_dir, "X_train.npy"))
    y_train = np.load(os.path.join(seq_dir, "y_train.npy"))
    units_train = np.load(os.path.join(seq_dir, "units_train.npy"))
    X_test = np.load(os.path.join(seq_dir, "X_test.npy"))
    y_test = np.load(os.path.join(seq_dir, "y_test.npy"))

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train, y_train, units_train, X_test, y_test


def make_train_val_loaders(X_train, y_train, units_train, val_split=VAL_SPLIT,
                            batch_size=BATCH_SIZE, seed=SEED):
    # Split by ENGINE, not by window index -- see FIX APPLIED #2 note at
    # top of file. Overlapping windows from the same engine landing in
    # both train and val inflated validation performance and made early
    # stopping unreliable.
    unique_units = np.unique(units_train)
    rng = np.random.default_rng(seed)
    shuffled_units = rng.permutation(unique_units)
    n_val_units = max(1, int(len(unique_units) * val_split))
    val_units = set(shuffled_units[:n_val_units])
    train_units = set(shuffled_units[n_val_units:])

    train_mask = np.isin(units_train, list(train_units))
    val_mask = np.isin(units_train, list(val_units))

    print(f"Engine-level split: {len(train_units)} train engines, {len(val_units)} val engines "
          f"({train_mask.sum()} train windows, {val_mask.sum()} val windows)")

    X_tr = torch.tensor(X_train[train_mask], dtype=torch.float32)
    # Target normalized to [0, 1] -- see FIX APPLIED #1 note at top offile
    y_tr = torch.tensor(y_train[train_mask] / RUL_SCALE, dtype=torch.float32)
    X_val = torch.tensor(X_train[val_mask], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_mask] / RUL_SCALE, dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(train_units), len(val_units)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    """2-layer LSTM -> dropout -> linear head, predicts a single normalized RUL value."""

    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]              # (batch, hidden_size) -- final layer's last time step
        last_hidden = self.dropout(last_hidden)
        rul = self.head(last_hidden)       # (batch, 1) -- normalized [0,1] scale
        return rul.squeeze(-1)             # (batch,)


# ---------------------------------------------------------------------------
# CMAPSS scoring function
# ---------------------------------------------------------------------------
def cmapss_score(y_true, y_pred):
    """
    Official CMAPSS asymmetric scoring function.
    Late predictions (predicted RUL > actual RUL, i.e. we said "more life left"
    than there really was) are penalized far more harshly than early ones,
    because that's the dangerous error in real maintenance scheduling.

    d = y_pred - y_true
        d < 0 (early/conservative prediction): score += exp(-d/13) - 1
        d >= 0 (late/optimistic prediction):   score += exp(d/10) - 1

    NOTE: both y_true and y_pred must be in ORIGINAL RUL units (cycles),
    not the normalized [0,1] scale used during training.
    """
    d = y_pred - y_true
    early = d < 0
    late = ~early
    score = np.zeros_like(d, dtype=np.float64)
    score[early] = np.exp(-d[early] / 13.0) - 1.0
    score[late] = np.exp(d[late] / 10.0) - 1.0
    return float(np.sum(score))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                 lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    epochs_run = 0

    for epoch in range(1, num_epochs + 1):
        epochs_run = epoch
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            # Gradient clipping -- prevents the LSTM from taking an
            # unstable step that collapses it into predicting the mean.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # RMSE printed here is in normalized [0,1] units * RUL_SCALE =real cycles,
        # so it's directly comparable to the CNN's printed RMSE duringtraining.
        val_rmse_cycles = np.sqrt(val_loss) * RUL_SCALE
        print(f"Epoch {epoch:3d}/{num_epochs} | train_MSE={train_loss:.5f} | val_MSE={val_loss:.5f} "
              f"| val_RMSE(cycles)={val_rmse_cycles:.3f}")

        # ---- log per-epoch metrics so the MLflow UI shows a training curve ----
        mlflow.log_metrics({
            "train_mse": float(train_loss),
            "val_mse": float(val_loss),
            "val_rmse_cycles": float(val_rmse_cycles),
        }, step=epoch)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epochs_run, float(best_val_loss)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds_normalized = model(X_test_t).cpu().numpy()

    # De-normalize back to real RUL units (cycles) before scoring
    preds = preds_normalized * RUL_SCALE

    test_rmse = rmse(y_test, preds)
    test_score = cmapss_score(y_test, preds)

    print("\n=== Test Set Evaluation (LSTM) ===")
    print(f"RMSE:          {test_rmse:.3f}")
    print(f"CMAPSS Score:  {test_score:.1f}  (lower is better)")
    return preds, test_rmse, test_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lstm_rul"):

        X_train, y_train, units_train, X_test, y_test = load_sequences()
        n_features = X_train.shape[2]

        train_loader, val_loader, n_train_engines, n_val_engines = make_train_val_loaders(
            X_train, y_train, units_train
        )

        model = LSTMRegressor(n_features=n_features)
        print(f"\nModel: {model}\n")
        print(f"Training on device: {DEVICE}\n")

        # ---- log hyperparameters up front ----
        mlflow.log_params({
            "model_type": "LSTM",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "val_split": VAL_SPLIT,
            "rul_scale": RUL_SCALE,
            "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
            "n_features": n_features,
            "n_train_engines": n_train_engines,
            "n_val_engines": n_val_engines,
            "split_strategy": "engine-level (no window leakage)",
            "device": str(DEVICE),
            "seed": SEED,
        })

        model, epochs_run, best_val_loss = train_model(model, train_loader, val_loader)
        preds, test_rmse, test_score = evaluate(model, X_test, y_test)

        # ---- log final test metrics ----
        mlflow.log_metrics({
            "test_rmse": test_rmse,
            "test_cmapss_score": test_score,
            "best_val_mse": best_val_loss,
            "epochs_run": epochs_run,
        })

        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "rul_scale": RUL_SCALE,
        }, MODEL_OUT_PATH)
        print(f"\nModel saved to {MODEL_OUT_PATH}")

        # ---- log the saved model file as an MLflow artifact ----
        mlflow.log_artifact(MODEL_OUT_PATH)
        print(f"Run also logged to MLflow (experiment: {MLFLOW_EXPERIMENT_NAME})")


if __name__ == "__main__":
    main()

