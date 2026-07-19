"""
tests/conftest.py
-------------------
Shared pytest fixtures. Adds scripts/ to sys.path so tests can import
project modules directly (signal_features, predict, cnn_rul, etc.) the
same way evaluate_test.py and api.py already do.
"""

import os
import sys

import numpy as np
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS_DIR)


@pytest.fixture(scope="session")
def pipeline():
    """Loads the real PredictiveMaintenancePipeline ONCE for the whole test
    session -- reused across every test that needs it, so model loading
    (ONNX session + PyTorch checkpoint) only happens once, not per-test.

    This requires the actual trained model files to exist on disk
    (models/lstm_rul.onnx, models/autoencoder_anomaly.pt) -- these tests
    are integration tests against your real trained models, not isolated
    unit tests with mocked models. That's a deliberate choice: testing
    decide_alert's LOGIC in isolation from real models would miss bugs
    in how the two connect.
    """
    from predict import PredictiveMaintenancePipeline
    return PredictiveMaintenancePipeline()
