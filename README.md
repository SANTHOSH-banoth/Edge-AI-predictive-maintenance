# Edge-AI Predictive Maintenance System

Predicts machine failure from sensor data (temperature, RPM, torque, tool wear)
and compares a full-size **cloud model** against a compact **edge-deployable model**,
showing the classic edge-AI trade-off: small enough to run on the machine itself,
without sacrificing much predictive power.

---

## 1. Why this project exists (the problem)

Factories run rotating machinery (motors, pumps, compressors) that fail
unpredictably. Two bad options today:
- **Reactive maintenance**: fix it after it breaks → costly downtime.
- **Scheduled maintenance**: replace parts on a calendar, whether they need it
  or not → wasted parts and labor.

**Predictive maintenance** uses live sensor data to predict failure *before*
it happens. The **"Edge-AI"** part matters because sending every sensor
reading to the cloud for inference is slow, bandwidth-heavy, and doesn't work
if the plant has poor connectivity — so the model needs to be small enough to
run right next to the machine (a Raspberry Pi / industrial PC / microcontroller
acting as an edge gateway).

## 2. Dataset

`data/machine_sensor_data.csv` — 10,000 samples, generated to match the
structure and physics of the well-known **AI4I 2020 Predictive Maintenance**
dataset (UCI Machine Learning Repository). Each row is one machine reading:

| Column | Meaning |
|---|---|
| Type | Product quality variant (L/M/H) — affects wear rate |
| Air_temperature_K, Process_temperature_K | Ambient vs. internal temperature |
| Rotational_speed_rpm | Shaft speed |
| Torque_Nm | Load on the shaft |
| Tool_wear_min | Cumulative wear (minutes of use) |
| Machine_failure | Target: 1 = failure, 0 = healthy |
| Failure_type | Which failure mode fired (for explainability) |

Failure rate is **8.6%** — realistically imbalanced, which is why class
imbalance handling (SMOTE) matters (see below).

Five failure modes are modeled, each tied to a real physical cause:
1. **Heat Dissipation Failure** — temp difference too small at low speed
2. **Power Failure** — mechanical power (torque × angular velocity) out of safe range
3. **Overstrain Failure** — tool wear × torque exceeds a material limit
4. **Tool Wear Failure** — tool past end-of-life
5. **Random Failure** — rare unexplained failures (sensor/electrical noise)

## 3. Pipeline

```
generate_data.py  →  train_model.py  →  simulate_edge_stream.py  →  dashboard.html
 (synthetic data)     (train + export)     (live inference sim)      (visualize)
```

### `scripts/train_model.py`
- Feature engineering: derives `Temp_diff`, `Power`, `Wear_Torque_Product`
  from raw sensors (domain knowledge → features)
- **Cloud model**: RandomForest (200 trees) — high accuracy, heavier
- **Class imbalance fix**: SMOTE oversampling on the training set only
  (never on test data — that would leak information)
- **Edge model**: a compact MLP (2 hidden layers, 16→8 neurons)
- Exports the edge model to **ONNX** — the standard portable format for
  running models on edge devices, mobile, and embedded runtimes
  (`onnxruntime`), instead of shipping the full Python/sklearn stack
- Saves all metrics to `models/metrics.json`

### `scripts/simulate_edge_stream.py`
Simulates a machine over its lifecycle (healthy → degrading → high failure
risk) and runs every reading through the **deployed ONNX edge model**,
exactly as an edge gateway would. Produces `dashboard/stream_data.json`.

### `dashboard.html`
Interactive HMI-style monitor: live sensor readouts, a failure-probability
gauge, an alert log, and a scrubbable timeline showing the model's prediction
at every point in the machine's life — plus a side-by-side cloud vs. edge
comparison card.

## 4. Results

| | Cloud (RandomForest) | Edge (MLP → ONNX) |
|---|---|---|
| Model size | 3,448.8 KB | **2.7 KB** |
| Recall (catches real failures) | 96.5% | 93.0% |
| Precision | 65.6% | 57.6% |
| ROC-AUC | 0.975 | 0.970 |

**Headline number: ~1,276× smaller model for a 3.5-point recall trade-off.**
That's the core edge-AI pitch — you give up a little accuracy to gain the
ability to run inference locally, in real time, with no cloud round-trip.

**Recall was prioritized over precision** deliberately: a missed real failure
(false negative) causes unplanned downtime and is far more expensive than a
false alarm (false positive), which just triggers an unnecessary inspection.

## 5. How to talk about this in an interview (quick cheat sheet)

- *"Why predictive maintenance?"* → reduces downtime cost vs. reactive/scheduled maintenance.
- *"Why 'edge' AI specifically?"* → low latency, works offline, avoids streaming raw sensor data to the cloud continuously.
- *"How did you handle class imbalance?"* → SMOTE on the training set only; recall-focused metric choice.
- *"How did you make the model edge-deployable?"* → picked a small architecture, exported to ONNX (a hardware/runtime-agnostic format used across mobile & embedded devices), and measured the size/accuracy trade-off directly rather than assuming it.
- *"What would you do with real deployment?"* → quantize further (int8), test on actual constrained hardware (Raspberry Pi/ESP32), add drift monitoring since sensor distributions shift over time.

## 6. Files in this repo

```
edge_ai_pm/
├── data/machine_sensor_data.csv       # dataset
├── scripts/
│   ├── generate_data.py               # dataset generator
│   ├── train_model.py                 # training + ONNX export + benchmark
│   └── simulate_edge_stream.py        # live inference simulation
├── models/
│   ├── cloud_model.pkl, edge_model.pkl, edge_model.onnx
│   ├── scaler.pkl, type_encoder.pkl, feature_columns.json
│   └── metrics.json                   # all evaluation numbers
├── dashboard/stream_data.json         # simulated stream (raw)
├── dashboard.html                     # interactive monitor (standalone)
└── README.md
```

## 7. Resume bullet points (copy/adapt)

- Built an end-to-end predictive maintenance pipeline on industrial sensor
  data, engineering physics-informed features (thermal differential,
  mechanical power, wear-load product) to boost model signal.
- Addressed severe class imbalance (8.6% failure rate) using SMOTE, improving
  minority-class recall from 0% to 93% on a resource-constrained model.
- Designed and benchmarked an edge-deployable model (ONNX Runtime) against a
  cloud-scale RandomForest, achieving a **99.9% size reduction (1,276×
  smaller)** with only a 3.5-point recall trade-off — enabling real-time,
  offline inference at the machine.
- Built an interactive monitoring dashboard visualizing live sensor streams,
  model confidence, and failure alerts across a simulated machine lifecycle.
