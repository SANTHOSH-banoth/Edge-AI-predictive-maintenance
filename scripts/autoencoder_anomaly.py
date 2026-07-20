"""
src/models/autoencoder_anomaly.py

Unsupervised anomaly detection for CMAPSS turbofan sensors via a
reconstruction-error autoencoder.

Key idea: train ONLY on "healthy" windows (early-life engine data, farfrom
failure), then measure reconstruction error on all data. Windows the model
struggles to reconstruct are flagged as anomalous -- this can catch failure
modes a supervised RUL/classifier model never saw labeled examples of.

"Healthy" windows are defined here as windows whose true RUL (from the
sequence label) is above a threshold (default: top 70% of the RUL range,
i.e. we exclude the last ~30% of each engine's life, which is where
degradation dominates the signal).

FIX APPLIED #1 (train/val split leakage): the original make_train_val_loaders
split healthy windows randomly by index. Since consecutive sliding windows
overlap heavily, this leaked near-duplicate windows from the same engine
into both train and val -- same bug as cnn_rul.py / lstm_rul.py. Fixedby
splitting whole engines (via units_train.npy) into train/val first.

FIX APPLIED #2 (threshold contamination): the original computed the anomaly
threshold AND the "healthy flag rate" sanity check using reconstruction_error
on X_healthy -- the FULL healthy set, including the windows the model was
directly trained to reconstruct. That's not a real held-out test: a model
will trivially reconstruct data it was optimized on, so "~5% of healthy
windows flagged" was largely checking training-set fit, not generalization.
Fixed by computing both the threshold and the sanity-check flag rate using
ONLY the held-out healthy VALIDATION windows (from engines never trained on).

All hyperparameters/metrics/artifacts also logged to MLflow (experiment:
"week2_deep_learning_rul") so this run is comparable side-by-side with
the LSTM/CNN RUL models trained in this project.

Usage:
    python src/models/autoencoder_anomaly.py
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
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "autoencoder_anomaly.pt")

# A window is "healthy" if its RUL label is at or above this percentile of
# the training RUL distribution (i.e. plenty of life left / early in engine life).
HEALTHY_RUL_PERCENTILE = 70

LATENT_DIM = 16
HIDDEN_DIM = 64
DROPOUT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
VAL_SPLIT = 0.15
SEED = 42

# Flag a window as anomalous if its reconstruction error exceeds this
# percentile of the error distribution on held-out HEALTHY validation data.
ANOMALY_PERCENTILE = 95

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


def split_healthy_vs_all(X_train, y_train, units_train, healthy_percentile=HEALTHY_RUL_PERCENTILE):
    """Split training windows into 'healthy' (high RUL) vs the rest.
    Also carries units_train through so the healthy subset can later be
    split by engine, not by window index."""
    threshold = np.percentile(y_train, healthy_percentile)
    healthy_mask = y_train >= threshold
    print(f"Healthy RUL threshold (>= {healthy_percentile}th pct): {threshold:.1f}")
    print(f"Healthy windows: {healthy_mask.sum()} / {len(y_train)}")
    return (X_train[healthy_mask], units_train[healthy_mask],
            X_train[~healthy_mask], threshold)


def make_train_val_loaders(X_healthy, units_healthy, val_split=VAL_SPLIT,
                            batch_size=BATCH_SIZE, seed=SEED):
    # Split by ENGINE, not by window index -- see FIX APPLIED #1 note at
    # top of file.
    unique_units = np.unique(units_healthy)
    rng = np.random.default_rng(seed)
    shuffled_units = rng.permutation(unique_units)
    n_val_units = max(1, int(len(unique_units) * val_split))
    val_units = set(shuffled_units[:n_val_units])
    train_units = set(shuffled_units[n_val_units:])

    train_mask = np.isin(units_healthy, list(train_units))
    val_mask = np.isin(units_healthy, list(val_units))

    print(f"Engine-level split (healthy windows): {len(train_units)} train engines, "
          f"{len(val_units)} val engines ({train_mask.sum()} train windows, "
          f"{val_mask.sum()} val windows)")

    X_tr = torch.tensor(X_healthy[train_mask], dtype=torch.float32)
    X_val = torch.tensor(X_healthy[val_mask], dtype=torch.float32)

    # Autoencoder: input IS the target, so dataset just needs X twice.
    train_loader = DataLoader(TensorDataset(X_tr, X_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, X_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, X_healthy[val_mask], len(train_units), len(val_units)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder.
    Encoder LSTM compresses the window into a latent vector; decoder LSTM
    reconstructs the full window from that latent vector, repeated across
    time steps. Reconstruction error (MSE) per window is the anomaly score.
    """

    def __init__(self, n_features, seq_len, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.seq_len = seq_len

        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True, dropout=0.0)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.encoder_dropout = nn.Dropout(dropout)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_dropout(self.encoder_fc(h_n[-1]))   # (batch, latent_dim)

        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        dec_out, _ = self.decoder_lstm(dec_input)
        recon = self.output_layer(dec_out)  # (batch, seq_len, n_features)
        return recon


def reconstruction_error(model, X, batch_size=256):
    """Per-window MSE reconstruction error, returned as a 1D numpy array."""
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(DEVICE)
            recon = model(xb)
            err = ((recon - xb) ** 2).mean(dim=(1, 2))  # per-window MSE
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


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
        for xb, target in train_loader:
            xb, target = xb.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            recon = model(xb)
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, target in val_loader:
                xb, target = xb.to(DEVICE), target.to(DEVICE)
                val_losses.append(criterion(model(xb), target).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{num_epochs} | train_recon_MSE={train_loss:.5f} | val_recon_MSE={val_loss:.5f}")

        # ---- log per-epoch metrics so the MLflow UI shows a training curve ----
        mlflow.log_metrics({
            "train_recon_mse": float(train_loss),
            "val_recon_mse": float(val_loss),
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
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="autoencoder_anomaly"):

        X_train, y_train, units_train, X_test, y_test = load_sequences()
        n_features = X_train.shape[2]
        seq_len = X_train.shape[1]

        X_healthy, units_healthy, X_degraded, threshold = split_healthy_vs_all(
            X_train, y_train, units_train
        )
        train_loader, val_loader, X_healthy_val, n_train_engines, n_val_engines = make_train_val_loaders(
            X_healthy, units_healthy
        )

        model = LSTMAutoencoder(n_features=n_features, seq_len=seq_len)
        print(f"\nModel: {model}\n")
        print(f"Training on device: {DEVICE}\n")

        # ---- log hyperparameters up front ----
        mlflow.log_params({
            "model_type": "LSTM-Autoencoder",
            "healthy_rul_percentile": HEALTHY_RUL_PERCENTILE,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "max_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "val_split": VAL_SPLIT,
            "anomaly_percentile": ANOMALY_PERCENTILE,
            "n_features": n_features,
            "seq_len": seq_len,
            "n_train_engines": n_train_engines,
            "n_val_engines": n_val_engines,
            "split_strategy": "engine-level, healthy-only (no window/train leakage)",
            "device": str(DEVICE),
            "seed": SEED,
        })

        model, epochs_run, best_val_loss = train_model(model, train_loader, val_loader)

        # FIX APPLIED #2: establish the anomaly threshold using ONLY the
        # held-out healthy VALIDATION windows (from engines never trained on),
        # not the full X_healthy set which includes training data the model
        # was directly optimized to reconstruct.
        healthy_val_errors = reconstruction_error(model, X_healthy_val)
        anomaly_threshold = np.percentile(healthy_val_errors, ANOMALY_PERCENTILE)
        print(f"\nAnomaly score threshold ({ANOMALY_PERCENTILE}th pct of HELD-OUT healthy val errors): "
              f"{anomaly_threshold:.5f}")

        # Sanity check: degraded (near-failure) windows should show highererror
        # and a higher flagged-anomaly rate than healthy validation windows.
        degraded_errors = reconstruction_error(model, X_degraded)
        test_errors = reconstruction_error(model, X_test)

        healthy_flag_rate = (healthy_val_errors > anomaly_threshold).mean()
        degraded_flag_rate = (degraded_errors > anomaly_threshold).mean()
        test_flag_rate = (test_errors > anomaly_threshold).mean()

        print("\n=== Anomaly Detection Sanity Check (all on held-out data)===")
        print(f"Healthy VAL windows flagged anomalous: {healthy_flag_rate:.1%}  (expect ~{100 - ANOMALY_PERCENTILE}%)")
        print(f"Degraded windows    flagged anomalous: {degraded_flag_rate:.1%}  (expect notably higher)")
        print(f"Test windows        flagged anomalous: {test_flag_rate:.1%}")

        # ---- log final metrics, including the sanity-check flag rates --
        # these are the numbers worth watching across future retrains: a
        # healthy_flag_rate that drifts far from ~5% or a degraded_flag_rate
        # that stops being notably higher than healthy both signal the
        # anomaly detector has stopped discriminating well.
        mlflow.log_metrics({
            "best_val_recon_mse": best_val_loss,
            "epochs_run": epochs_run,
            "healthy_rul_threshold": float(threshold),
            "anomaly_error_threshold": float(anomaly_threshold),
            "healthy_val_flag_rate": float(healthy_flag_rate),
            "degraded_flag_rate": float(degraded_flag_rate),
            "test_flag_rate": float(test_flag_rate),
        })

        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "seq_len": seq_len,
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "healthy_rul_threshold": float(threshold),
            "anomaly_error_threshold": float(anomaly_threshold),
        }, MODEL_OUT_PATH)
        print(f"\nModel + thresholds saved to {MODEL_OUT_PATH}")

        # ---- log the saved model file as an MLflow artifact ----
        mlflow.log_artifact(MODEL_OUT_PATH)
        print(f"Run also logged to MLflow (experiment: {MLFLOW_EXPERIMENT_NAME})")


if __name__ == "__main__":
    main()

