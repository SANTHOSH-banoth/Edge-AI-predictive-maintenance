"""
predict.py
------------
Week 5/6: the unifying pipeline. Everything up to this point has been
separate experiments (LSTM, CNN, XGBoost, autoencoder, each with their own
training script). This is the one thing that turns those into an actual
SYSTEM: a single function that takes a 30-cycle sensor window and returns
a complete maintenance decision, the way a real deployed service would.

Pipeline stages, and which model handles each:
  1. ANOMALY CHECK   -> LSTMAutoencoder (native PyTorch)
   python -m http.server 8000  Unsupervised -- flags sensor patterns unlike anything in healthy
     training data, even failure modes never explicitly labeled.
  2. RUL ESTIMATE     -> LSTM regressor, deployed via ONNX Runtime
     Your best-performing, most efficient model (RMSE 12.8, CMAPSS score
     267 -- beats both XGBoost and the CNN -- and runs 2.79x faster via
     ONNX Runtime than native PyTorch (1.273ms -> 0.457ms single-row
     latency) with zero accuracy loss (RMSE/MAE/CMAPSS score identical
     to 3 decimal places), per the real benchmark in export_lstm_onnx.py).
  3. ALERT DECISION   -> simple rule combining both signals
     Real maintenance systems don't act on a single model's output alone;
     they combine multiple signals into an actionable decision with
     clear reasoning, which is what this stage does.

Why the autoencoder stays native PyTorch here (not ONNX) while the LSTM
regressor uses ONNX: this project's edge-deployment story centers on the
RUL model (the one you export/benchmark/compare most rigorously). Nothing
prevents exporting the autoencoder too -- it's a reasonable next step, but
wasn't part of this pipeline's initial scope. Worth saying exactly that if
asked, rather than implying every model here is edge-optimized.

Run as a demo over your real held-out test engines:
    python scripts/predict.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import pandas as pd

MODEL_DIR = "models"
SEQ_DIR = os.path.join("data", "cmapss", "sequences")

LSTM_ONNX_PATH = os.path.join(MODEL_DIR, "lstm_rul.onnx")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder_anomaly.pt")

RUL_SCALE = 125.0  # must match RUL_CAP in build_sequences.py / lstm_rul.py

# Alert thresholds -- these are decision-policy choices, not model outputs.
# In a real deployment these would be tuned against maintenance cost data
# (cost of unplanned downtime vs. cost of early/unnecessary maintenance).
RUL_URGENT_THRESHOLD = 20   # cycles remaining -- schedule maintenance now
RUL_WATCH_THRESHOLD = 50    # cycles remaining -- start monitoring closely


class LSTMAutoencoder(nn.Module):
    """Must match the architecture in autoencoder_anomaly.py exactly."""
    def __init__(self, n_features, seq_len, hidden_dim=64, latent_dim=16, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True, dropout=0.0)
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        self.encoder_dropout = nn.Dropout(dropout)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_dropout(self.encoder_fc(h_n[-1]))
        dec_input = self.decoder_fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.output_layer(dec_out)


class PredictiveMaintenancePipeline:
    """Loads all models once, then scores sensor windows through the
    full anomaly-check -> RUL-estimate -> alert-decision pipeline."""

    def __init__(self, model_dir=MODEL_DIR):
        # --- Load RUL model (ONNX Runtime -- the edge-deployed path) ---
        self.rul_session = ort.InferenceSession(
            os.path.join(model_dir, "lstm_rul.onnx"), providers=["CPUExecutionProvider"]
        )
        self.rul_input_name = self.rul_session.get_inputs()[0].name

        # --- Load autoencoder (native PyTorch) ---
        checkpoint = torch.load(AUTOENCODER_PATH, map_location="cpu", weights_only=False)
        self.autoencoder = LSTMAutoencoder(
            n_features=checkpoint["n_features"], seq_len=checkpoint["seq_len"],
            hidden_dim=checkpoint["hidden_dim"], latent_dim=checkpoint["latent_dim"],
            dropout=checkpoint["dropout"],
        )
        self.autoencoder.load_state_dict(checkpoint["model_state_dict"])
        self.autoencoder.eval()
        self.anomaly_threshold = checkpoint["anomaly_error_threshold"]

        print(f"Pipeline loaded. Anomaly threshold: {self.anomaly_threshold:.5f}")

    def predict_rul(self, window):
        """window: (seq_len, n_features) numpy array, already scaled.
        Returns predicted RUL in real cycle units."""
        batch = window[np.newaxis, :, :].astype(np.float32)  # add batch dim
        pred_normalized = self.rul_session.run(None, {self.rul_input_name: batch})[0]
        return float(pred_normalized.squeeze()) * RUL_SCALE

    def check_anomaly(self, window):
        """Returns (is_anomaly: bool, reconstruction_error: float)."""
        with torch.no_grad():
            x = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)
            recon = self.autoencoder(x)
            error = ((recon - x) ** 2).mean().item()
        return error > self.anomaly_threshold, error

    def decide_alert(self, predicted_rul, is_anomaly):
        """Combines both model outputs into one actionable decision.
        This is deliberately simple, explicit logic -- not another model --
        because the ALERT DECISION is a business/safety policy choice,
        and that should be transparent and auditable, not a black box."""
        if predicted_rul <= RUL_URGENT_THRESHOLD:
            return "URGENT", f"Predicted RUL ({predicted_rul:.1f} cycles) below urgent threshold ({RUL_URGENT_THRESHOLD}). Schedule maintenance now."
        if is_anomaly and predicted_rul <= RUL_WATCH_THRESHOLD:
            return "WARNING", f"Anomalous sensor pattern AND declining RUL ({predicted_rul:.1f} cycles). Recommend inspection soon."
        if is_anomaly:
            return "WATCH", f"Anomalous sensor pattern detected (RUL estimate {predicted_rul:.1f} still healthy). Monitor closely -- may be a failure mode the RUL model wasn't trained to recognize."
        if predicted_rul <= RUL_WATCH_THRESHOLD:
            return "WATCH", f"RUL estimate ({predicted_rul:.1f} cycles) entering monitoring window."
        return "HEALTHY", f"RUL estimate {predicted_rul:.1f} cycles, no anomaly detected."

    def predict(self, window):
        """Full pipeline for one sensor window. Returns a result dict."""
        predicted_rul = self.predict_rul(window)
        is_anomaly, recon_error = self.check_anomaly(window)
        alert_level, reason = self.decide_alert(predicted_rul, is_anomaly)
        return {
            "predicted_rul": round(predicted_rul, 1),
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": round(recon_error, 5),
            "alert_level": alert_level,
            "reason": reason,
        }


def demo_on_real_test_engines():
    """Runs the full pipeline on every real held-out CMAPSS test engine
    and prints/saves a summary -- the same kind of report a maintenance
    dashboard would show for a whole fleet."""
    print("Loading real held-out test sequences...")
    X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy")).astype(np.float32)
    print(f"X_test: {X_test.shape} ({X_test.shape[0]} engines)\n")

    pipeline = PredictiveMaintenancePipeline()

    results = []
    for i in range(len(X_test)):
        result = pipeline.predict(X_test[i])
        result["engine_id"] = i + 1
        results.append(result)

    df = pd.DataFrame(results)[["engine_id", "predicted_rul", "is_anomaly",
                                  "reconstruction_error", "alert_level", "reason"]]

    print("=== Fleet-wide prediction summary ===")
    print(df["alert_level"].value_counts().to_string())

    print("\n=== Sample: 5 most urgent engines ===")
    urgent = df.sort_values("predicted_rul").head(5)
    for _, row in urgent.iterrows():
        print(f"Engine {row['engine_id']:>3} | RUL={row['predicted_rul']:>6.1f} | "
              f"{row['alert_level']:<8} | {row['reason']}")

    out_path = os.path.join(MODEL_DIR, "fleet_prediction_report.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull fleet report saved to {out_path}")


if __name__ == "__main__":
    demo_on_real_test_engines()
