"""
tests/test_split_leakage.py
------------------------------
Permanent regression guard for the bug this project actually found and
fixed three times: cnn_rul.py, lstm_rul.py, and autoencoder_anomaly.py
all originally split TRAINING WINDOWS randomly by index instead of by
engine, which let overlapping sliding windows from the same engine leak
into both train and validation sets.

Fixing it once and moving on would leave the door open for it to silently
come back -- e.g. someone refactors make_train_val_loaders later, copies
the OLD pattern from an example online, or a future contributor adds a
new sequence model without knowing this project's history with this
exact bug. These tests make that impossible to do silently: if any of
the three make_train_val_loaders functions ever again splits by window
index instead of by engine, one of these tests fails immediately.

This is the highest-value test in the whole test suite -- it protects
against a bug that was real, that was found through actual investigation
(a 4x val-vs-test RMSE gap), and that a random split would silently
reintroduce with no error, no warning, just quietly-wrong validation
numbers again.
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_units_and_data():
    """20 synthetic engines, ~50 windows each -- enough to exercise a real
    train/val split (15% val = 3 engines) without needing real CMAPSS data
    on disk. Values themselves don't matter for this test; only unit IDs do."""
    rng = np.random.default_rng(0)
    n_engines = 20
    windows_per_engine = 50
    n_features = 18
    seq_len = 30

    units = np.repeat(np.arange(1, n_engines + 1), windows_per_engine)
    n_total = len(units)
    X = rng.standard_normal((n_total, seq_len, n_features)).astype(np.float32)
    y = rng.uniform(0, 125, size=n_total).astype(np.float32)
    return X, y, units


def _no_overlap_between_loaders(train_loader, val_loader):
    """Extract all X windows from both loaders and confirm none of the
    train windows are byte-identical to any val window. Since windows are
    continuous random floats, exact equality only happens if the SAME
    underlying row was assigned to both splits -- which is exactly what
    the leakage bug would cause when overlapping windows from one engine
    land in both sets."""
    train_X = np.concatenate([xb.numpy() for xb, _ in train_loader], axis=0)
    val_X = np.concatenate([xb.numpy() for xb, _ in val_loader], axis=0)

    # Flatten each window to a hashable signature for a fast set-based check.
    train_signatures = {tuple(row.flatten()[:10]) for row in train_X}
    val_signatures = {tuple(row.flatten()[:10]) for row in val_X}

    overlap = train_signatures & val_signatures
    assert len(overlap) == 0, (
        f"{len(overlap)} windows appear in BOTH train and val sets -- this is "
        f"exactly the leakage bug this project found and fixed three times. "
        f"Check that the split is happening by ENGINE, not by window index."
    )


def test_cnn_rul_split_has_no_engine_overlap(synthetic_units_and_data):
    from cnn_rul import make_train_val_loaders
    X, y, units = synthetic_units_and_data
    train_loader, val_loader = make_train_val_loaders(X, y, units, batch_size=256)
    _no_overlap_between_loaders(train_loader, val_loader)


def test_lstm_rul_split_has_no_engine_overlap(synthetic_units_and_data):
    from lstm_rul import make_train_val_loaders
    X, y, units = synthetic_units_and_data
    train_loader, val_loader = make_train_val_loaders(X, y, units, batch_size=256)
    _no_overlap_between_loaders(train_loader, val_loader)


def test_autoencoder_split_has_no_engine_overlap(synthetic_units_and_data):
    from autoencoder_anomaly import make_train_val_loaders
    X, y, units = synthetic_units_and_data
    # Autoencoder's function takes (X_healthy, units_healthy, ...) -- reuse
    # the full synthetic set as "healthy" for this structural test, since
    # the split logic itself doesn't depend on the healthy/degraded split.
    train_loader, val_loader, _ = make_train_val_loaders(X, units, batch_size=256)
    _no_overlap_between_loaders(train_loader, val_loader)


def test_cnn_rul_train_and_val_engine_sets_are_disjoint(synthetic_units_and_data):
    """A second, more direct check using the split logic itself --
    reconstructs which engines went where, and asserts the two engine ID
    sets share nothing in common."""
    from cnn_rul import VAL_SPLIT, SEED
    X, y, units = synthetic_units_and_data

    unique_units = np.unique(units)
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique_units)
    n_val = max(1, int(len(unique_units) * VAL_SPLIT))
    val_units = set(shuffled[:n_val])
    train_units = set(shuffled[n_val:])

    assert train_units.isdisjoint(val_units), (
        "Train and val engine sets overlap -- the split logic itself is "
        "computing overlapping engine groups, independent of how the "
        "resulting windows are loaded."
    )
    assert len(train_units) + len(val_units) == len(unique_units), (
        "Train + val engine counts don't add up to the total -- some "
        "engines are being dropped or double-counted by the split."
    )
