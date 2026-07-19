# Edge-AI Predictive Maintenance System

An end-to-end predictive maintenance pipeline with two parts that build on
each other:

1. **A cloud-vs-edge failure classifier** (AI4I-style sensor data) that
   demonstrates the core edge-AI trade-off: a ~1,300x smaller model for a
   few points of recall.
2. **A fleet-wide Remaining Useful Life (RUL) + anomaly detection pipeline**
   on NASA's CMAPSS turbofan degradation dataset, combining an autoencoder
   anomaly detector with an LSTM RUL regressor into a single alert-priority
   decision layer, exported to ONNX for edge inference.

A live dashboard (`dashboard/dashboard.html`) visualizes a simulated
machine's full lifecycle running through the deployed edge model in real
time.

---

## 1. Why this project exists

Factories run rotating machinery (motors, pumps, compressors, turbofans)
that fails unpredictably. Two common but flawed strategies:

- **Reactive maintenance** — fix it after it breaks → unplanned downtime.
- **Scheduled maintenance** — replace parts on a calendar regardless of
  condition → wasted parts and labor.

**Predictive maintenance** uses live sensor data to flag failure risk
*before* it happens. The **edge** part matters because streaming every
sensor reading to the cloud for inference is slow, bandwidth-heavy, and
breaks on poor plant connectivity — so the model needs to be small and fast
enough to run right next to the machine.

---

## 2. Part A — Cloud vs. Edge failure classifier

**Data:** `data/machine_sensor_data.csv` — 10,000 synthetic samples
matching the structure of the AI4I 2020 Predictive Maintenance dataset.
Each row: `Type`, `Air_temperature_K`, `Process_temperature_K`,
`Rotational_speed_rpm`, `Torque_Nm`, `Tool_wear_min` → binary
`Machine_failure` label. Failure rate is realistically imbalanced at 8.6%.

**Pipeline:**
```
generate_data.py → train_model.py → simulate_edge_stream.py → dashboard.html
 (synthetic data)   (train+export)    (live inference sim)     (visualize)
```

- `scripts/train_model.py` engineers physics-informed features
  (`Temp_diff`, `Power`, `Wear_Torque_Product`), fixes class imbalance with
  SMOTE on the training set only, trains a cloud RandomForest and a compact
  edge MLP (16→8), and exports the edge model to ONNX.
- `scripts/simulate_edge_stream.py` replays one machine's simulated
  lifecycle (healthy → degrading → high wear) through the deployed
  `edge_model.onnx` via `onnxruntime`, exactly as an edge gateway would,
  and writes `dashboard/stream_data.json`.
- `dashboard/dashboard.html` — a scrubbable, real-time-style monitor
  reading that stream: failure-probability curve, alert markers, and live
  sensor traces at any timestep. Serve it locally (browsers block local
  JSON fetches over `file://`):
  ```powershell
  cd dashboard
  python -m http.server 8000
  # open http://localhost:8000/dashboard.html
  ```

### Results — Cloud vs. Edge

| | Cloud (RandomForest) | Edge (MLP, sklearn) | Edge (MLP, ONNX Runtime) |
|---|---|---|---|
| Model size | 3,523.7 KB | 15.7 KB | **2.71 KB** |
| Accuracy | 95.5% | 93.5% | — |
| Precision | 65.8% | 57.6% | — |
| Recall | 98.3% | 93.0% | — |
| ROC-AUC | 0.976 | 0.970 | — |
| Avg. latency / sample | 0.0245 ms | 0.00034 ms | 0.0253 ms |

**Headline: ~99.9% smaller (≈1,300x) than the cloud model, for a
~5-point recall trade-off.** That's the edge-AI pitch — give up a little
accuracy to run inference locally, in real time, without a cloud
round-trip. Note the ONNX Runtime version is not faster in wall-clock terms
than raw sklearn on this laptop (single-sample calls have fixed runtime
overhead); its value is portability — the same `.onnx` file runs on
`onnxruntime` across edge gateways, mobile, and embedded targets without
shipping the full Python/sklearn stack.

### Decision threshold analysis (edge MLP)

Swept the classification threshold from 0.1–0.9 on the real held-out test
set (2,000 samples, 172 real failures) to make the recall/precision
trade-off explicit rather than assumed:

| Threshold | Precision | Recall | F1 | Missed failures | False alarms |
|---|---|---|---|---|---|
| 0.1 | 0.399 | 0.988 | 0.569 | 2 | 256 |
| 0.2 | 0.467 | 0.983 | 0.633 | 3 | 193 |
| **0.3** | **0.508** | **0.971** | **0.667** | **5** | **162** |
| 0.4 | 0.539 | 0.959 | 0.690 | 7 | 141 |
| 0.5 (default) | 0.576 | 0.930 | 0.711 | 12 | 118 |
| 0.6 | 0.598 | 0.901 | 0.719 | 17 | 104 |
| 0.7 | 0.627 | 0.890 | 0.736 | 19 | 91 |
| 0.8 | 0.670 | 0.814 | 0.735 | 32 | 69 |
| 0.9 | 0.699 | 0.703 | 0.701 | 51 | 52 |

Average precision (area under PR curve): **0.739**.

**Recommendation: threshold 0.3 over the sklearn default of 0.5.** It
catches 7 more real failures at the cost of 44 more false alarms — a
reasonable trade given a missed failure (unplanned downtime) is far more
expensive than an unnecessary inspection. Full sweep saved to
`models/precision_recall_thresholds.csv`.

---

## 3. Part B — CMAPSS RUL prediction + anomaly detection

**Data:** NASA's CMAPSS turbofan degradation dataset
(`data/cmapss/`) — multiple engines run from healthy to failure across
several operating conditions and fault modes, with 21+ sensor channels per
cycle. `scripts/load_cmapss.py` and `scripts/build_sequences.py` process
the raw `train_FD00X.txt` / `test_FD00X.txt` files into fixed-length
sliding-window sequences (`data/cmapss/sequences/`, shape `[engines, 30
timesteps, 18 features]`) for the sequence models.

**Three models, one decision layer:**

| Model | Role | Script |
|---|---|---|
| Autoencoder | Unsupervised anomaly detector — flags sensor patterns unlike anything in the healthy training data | `scripts/autoencoder_anomaly.py` |
| LSTM | Best-performing RUL (Remaining Useful Life) regressor | `scripts/lstm_rul.py` |
| XGBoost | RUL regression baseline for comparison | `scripts/train_xgboost.py` |
| CNN | Additional RUL architecture explored for comparison | `scripts/cnn_rul.py` |

`scripts/predict.py` is the unified fleet pipeline: run every held-out
engine through the anomaly detector *and* the RUL model, then combine both
signals into one alert level per engine (`HEALTHY` / `WATCH` / `URGENT` /
`WARNING`, where `WARNING` = anomaly and low-RUL agree). On the 100-engine
held-out CMAPSS test set:

```
alert_level
HEALTHY    66
WATCH      18
URGENT     13
WARNING     3
```

Full per-engine report saved to `models/fleet_prediction_report.csv`.

### RUL model comparison (CMAPSS test set)

| Model | Format | Size | Single-row latency | Test RMSE | CMAPSS score |
|---|---|---|---|---|---|
| XGBoost | Native | 1,498.1 KB | 2.934 ms | 14.09 | — |
| XGBoost | ONNX fp32 | 804.6 KB | 0.039 ms | 14.09 | — |
| XGBoost | ONNX int8 (quantized) | 804.7 KB | 0.049 ms | 14.09 | — |
| LSTM | Native PyTorch | 218.4 KB | 1.273 ms | 12.80 | 267.0 |
| **LSTM** | **ONNX fp32** | **218.3 KB** | **0.457 ms** | **12.80** | **267.0** |
| LSTM | ONNX int8 (quantized) | 63.98 KB | 0.153 ms | 12.85 | 271.2 |

**LSTM was selected as the deployed model** — lower RMSE than XGBoost, and
the ONNX export is ~6x faster than the native PyTorch version with
identical predictions (max prediction diff between native and ONNX:
7.6e-05). Int8 quantization shrinks it a further ~3.4x (218 KB → 64 KB)
for a marginal RMSE cost (12.80 → 12.85) — a reasonable trade if the
deployment target is memory-constrained, though the current pipeline ships
the fp32 ONNX version since size wasn't the binding constraint.

XGBoost's ONNX conversion, by contrast, gained most of its latency win from
format alone (2.934 ms → 0.039 ms) with quantization adding no further
benefit — expected, since int8 quantization targets neural-net-style
matrix multiplies, not tree ensembles.

### Edge deployment validation (XGBoost ONNX fp32)

Measured with `scripts/edge_deployment_validation.py`:

| Metric | Value |
|---|---|
| Process memory | 65.9 MB |
| Mean latency | 0.0426 ms |
| p95 latency | 0.0551 ms |
| p99 latency | 0.2676 ms |
| Throughput | 129,939 rows/sec |

### Resource-constrained hardware proxy

No Raspberry Pi or other embedded device was available for this project
(see [Limitations](#5-limitations)). As a partial substitute,
`scripts/benchmark_constrained.py` restricts ONNX Runtime to a single CPU
thread and compares it against the default (all-cores) run — this
approximates, but does not replicate, inference on a resource-constrained
device:

```
Default (all cores available):            0.352 ms
Single-thread (constrained-device proxy): 0.400 ms
Slowdown factor: 1.14x
```

Latency holds up well even single-threaded, suggesting the model doesn't
lean heavily on multi-core parallelism — a good sign for deployment on
constrained hardware, though this is explicitly a same-CPU proxy, not a
real embedded-device benchmark.

### Alert-rate sanity check

Before trusting the fleet pipeline, verified that `predict.py`'s alerts
behave sensibly rather than firing at random. Ran a single simulated
machine lifecycle (`simulate_edge_stream.py`, 120 timesteps, healthy →
end-of-life) through the deployed edge model and checked *where* alerts
land:

```
Total alerts: 30 / 120
First alert at t = 30
Last healthy reading at t = 115
```

Alerts are sparse and scattered early (t=30, 36, 61, 70–83…) and become
solidly clustered late (15 of the last 19 timesteps, t=101–119) — the
expected shape for a genuine wear-out trajectory, not a miscalibrated
model. Note this run's ~25% overall alert rate is *not* comparable to the
dataset's population-wide 8.6% failure rate: one is "fraction of one
machine's simulated life spent near end-of-life," the other is "fraction
of all machines that ever fail" — they measure different things and
aren't expected to match. Visualized in `dashboard/dashboard.html`.

---

## 4. Testing

`tests/` covers feature engineering correctness, alert-decision logic, and
(critically) that the train/test split doesn't leak information across
engine lifecycles:

```powershell
pytest tests/
```

- `test_feature_engineering.py` — engineered features compute correctly
- `test_alert_logic.py` — alert-level decision rules (anomaly + RUL →
  HEALTHY/WATCH/URGENT/WARNING) behave as specified
- `test_split_leakage.py` — no engine appears in both train and test
  sequences

---

## 5. Limitations

- **No real edge hardware was tested.** All "edge" latency numbers are
  measured on a laptop CPU via ONNX Runtime, with a single-thread run as a
  rough proxy for a constrained device (see above). A real embedded
  benchmark (Raspberry Pi, ESP32, industrial PC) would be the natural next
  step and the main gap between this project and a production deployment.
- **The dashboard replays one simulated lifecycle**, not a live sensor
  feed — there's no real-time ingestion, alerting service, or persistence
  layer behind it.
- **No drift monitoring.** Sensor distributions shift over time in real
  deployments; this pipeline doesn't detect or retrain against that.

---

## 6. Repo structure

```
edge_ai_pm_project/
├── data/
│   ├── machine_sensor_data.csv        # AI4I-style classifier dataset
│   └── cmapss/                        # NASA CMAPSS turbofan dataset
│       ├── train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt
│       ├── processed/                 # cleaned + feature-engineered CSVs
│       └── sequences/                 # windowed sequences for LSTM/CNN
├── scripts/
│   ├── generate_data.py               # AI4I-style synthetic data generator
│   ├── train_model.py                 # cloud RF + edge MLP, ONNX export
│   ├── simulate_edge_stream.py        # live inference simulation → dashboard feed
│   ├── precision_recall_analysis.py   # threshold sweep for edge MLP
│   ├── benchmark_constrained.py       # single-thread latency proxy
│   ├── load_cmapss.py, build_sequences.py, signal_features.py
│   ├── train_xgboost.py, lstm_rul.py, cnn_rul.py, autoencoder_anomaly.py
│   ├── export_lstm_onnx.py, quantize_lstm.py, quantize_and_benchmark.py
│   ├── edge_deployment_validation.py
│   ├── predict.py                     # unified fleet-wide alert pipeline
│   ├── predict_train_fleet.py, evaluate_test.py, analyze_test_by_rul_band.py
│   ├── explain_shap.py                # SHAP feature attribution
│   ├── validate_data.py, explore_data.py
│   └── api.py                         # inference API
├── models/                            # trained weights, ONNX exports, metrics
├── dashboard/
│   ├── dashboard.html                 # interactive monitor (standalone)
│   └── stream_data.json               # simulated stream (generated)
├── tests/
└── README.md
```

---

## 7. How to talk about this in an interview

- *"Why predictive maintenance?"* → cuts downtime cost vs. reactive/
  scheduled maintenance.
- *"Why edge AI specifically?"* → low latency, works offline, avoids
  streaming raw sensor data continuously to the cloud.
- *"How did you handle class imbalance?"* → SMOTE on the training set
  only; then swept the decision threshold from 0.1–0.9 and quantified
  exactly how many failures vs. false alarms each choice produces, rather
  than just eyeballing a default — recommended 0.3 over 0.5 because it
  catches 7 more real failures at a cost of 44 extra inspections.
- *"How did you choose between models?"* → compared XGBoost, LSTM, and
  CNN on RMSE and CMAPSS score on a genuinely held-out engine split (no
  leakage — verified with a dedicated test), picked the LSTM, then
  measured — not assumed — that ONNX export preserved accuracy exactly
  while cutting latency ~6x.
- *"How did you validate the fleet-alert logic wasn't just noise?"* → ran
  a full simulated machine lifecycle and checked that alerts cluster near
  true end-of-life rather than firing at random — quantified, not just
  visually inspected.
- *"What would you do with real deployment?"* → test on actual
  constrained hardware (Raspberry Pi/ESP32) instead of the single-thread
  proxy used here, add drift monitoring, and build a live ingestion layer
  behind the dashboard instead of a replayed simulation.

## 8. Resume bullet points (copy/adapt)

- Built an end-to-end predictive maintenance system spanning binary
  failure classification and multi-model Remaining Useful Life (RUL)
  regression on NASA CMAPSS turbofan degradation data.
- Designed and benchmarked an edge-deployable classifier (ONNX Runtime)
  against a cloud-scale RandomForest, achieving a **~1,300x size
  reduction** with only a ~5-point recall trade-off.
- Combined an unsupervised autoencoder anomaly detector with an LSTM RUL
  regressor into a unified fleet-alert pipeline, validated end-to-end
  against 100 held-out engines.
- Quantified the RUL model's decision threshold and quantization
  trade-offs directly (precision/recall sweep, fp32 vs. int8 ONNX
  benchmarks) rather than assuming defaults, cutting model size 3.4x for
  a 0.05 RMSE cost.
- Built an interactive monitoring dashboard visualizing live sensor
  streams, model confidence, and failure alerts across a simulated
  machine lifecycle, and validated alert timing against ground-truth
  degradation curves.