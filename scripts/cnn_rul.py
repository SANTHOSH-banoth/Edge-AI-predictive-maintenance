"""
src/models/cnn_rul.py

1D-CNN Remaining Useful Life (RUL) regression model for CMAPSS turbofan data.

Same data pipeline as lstm_rul.py (loads the same .npy sequence windows) so
the two models are directly comparable on the same train/val/test split logic.

Usage:
    python src/models/cnn_rul.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_DIR = os.path.join("data", "cmapss", "sequences")
MODEL_OUT_DIR = os.path.join("models")
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "cnn_rul.pt")

NUM_CHANNELS = [32, 64, 64]   # output channels of each conv block
KERNEL_SIZE = 5
DROPOUT = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10
VAL_SPLIT = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data loading (identical to lstm_rul.py, kept self-contained on purpose)
# ---------------------------------------------------------------------------
def load_sequences(seq_dir=SEQ_DIR):
    print("Loading sequence arrays...")
    X_train = np.load(os.path.join(seq_dir, "X_train.npy"))
    y_train = np.load(os.path.join(seq_dir, "y_train.npy"))
    X_test = np.load(os.path.join(seq_dir, "X_test.npy"))
    y_test = np.load(os.path.join(seq_dir, "y_test.npy"))
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train, y_train, X_test, y_test


def make_train_val_loaders(X_train, y_train, val_split=VAL_SPLIT, batch_size=BATCH_SIZE, seed=SEED):
    n = X_train.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CNNRegressor(nn.Module):
    """
    Stack of 1D conv blocks (Conv1d -> BatchNorm -> ReLU -> Dropout) over the
    time dimension, followed by global average pooling and a linear head.
    Global pooling makes it robust to any window length without a fixed-size
    flatten layer.
    """

    def __init__(self, n_features, channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_ch = n_features
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x):
        # x arrives as (batch, seq_len, n_features) -> Conv1d wants (batch, n_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)              # (batch, channels, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        return self.head(x).squeeze(-1)      # (batch,)


# ---------------------------------------------------------------------------
# CMAPSS scoring function (identical to lstm_rul.py)
# ---------------------------------------------------------------------------
def cmapss_score(y_true, y_pred):
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

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{num_epochs} | train_MSE={train_loss:.3f} | val_MSE={val_loss:.3f} "
              f"| val_RMSE={np.sqrt(val_loss):.3f}")

        if val_loss < best_val_loss - 1e-4:
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
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test):
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()

    test_rmse = rmse(y_test, preds)
    test_score = cmapss_score(y_test, preds)

    print("\n=== Test Set Evaluation (CNN) ===")
    print(f"RMSE:          {test_rmse:.3f}")
    print(f"CMAPSS Score:  {test_score:.1f}  (lower is better)")
    return preds, test_rmse, test_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, y_train, X_test, y_test = load_sequences()
    n_features = X_train.shape[2]

    train_loader, val_loader = make_train_val_loaders(X_train, y_train)

    model = CNNRegressor(n_features=n_features)
    print(f"\nModel: {model}\n")
    print(f"Training on device: {DEVICE}\n")

    model = train_model(model, train_loader, val_loader)
    evaluate(model, X_test, y_test)

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "channels": NUM_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
    }, MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
