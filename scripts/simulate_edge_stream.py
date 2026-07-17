"""
simulate_edge_stream.py
------------------------
Simulates a machine running over time, with sensor readings drifting from
healthy -> degrading -> failure risk (a realistic wear-out curve). Runs
each reading through the deployed edge model (ONNX Runtime, exactly as it
would run on an edge gateway next to the machine) and logs predictions.

This is the "inference" half of the pipeline: sensors -> edge model -> alert.

Output: dashboard/stream_data.json  (consumed by the dashboard artifact)
"""

import json
import joblib
import numpy as np
import onnxruntime as ort

MODEL_DIR = "models"
OUT_PATH = "dashboard/stream_data.json"
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
type_encoder = joblib.load(f"{MODEL_DIR}/type_encoder.pkl")
with open(f"{MODEL_DIR}/feature_columns.json") as f:
    feature_cols = json.load(f)

sess = ort.InferenceSession(f"{MODEL_DIR}/edge_model.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name

np.random.seed(7)
N_STEPS = 120  # simulate 120 timesteps (e.g. one reading every 5 min ~ 10 hours)

machine_type = "M"
type_encoded = type_encoder.transform([machine_type])[0]

records = []
tool_wear = 0.0

for t in range(N_STEPS):
    # Progressive wear-out: tool wear climbs, torque/temp get noisier as machine degrades
    degradation = t / N_STEPS  # 0 -> healthy, 1 -> near end of life
    tool_wear += np.random.uniform(1.5, 2.5)

    air_temp = np.random.normal(300, 1.5)
    process_temp = air_temp + np.random.normal(10 - 3 * degradation, 1 + degradation)
    rot_speed = np.random.normal(1500 - 150 * degradation, 150 + 80 * degradation)
    torque = np.random.normal(40 + 15 * degradation, 8 + 5 * degradation)

    temp_diff = process_temp - air_temp
    power = torque * (rot_speed * 2 * np.pi / 60)
    wear_torque = tool_wear * torque

    features = np.array([[type_encoded, air_temp, process_temp, rot_speed,
                           torque, tool_wear, temp_diff, power, wear_torque]], dtype=np.float32)
    features_scaled = scaler.transform(features).astype(np.float32)

    outputs = sess.run(None, {input_name: features_scaled})
    pred_label = int(outputs[0][0])
    # onnxruntime sklearn MLP output[1] is list of dicts with class probabilities
    proba_dict = outputs[1][0]
    failure_prob = float(proba_dict[1])

    records.append({
        "t": t,
        "air_temp_K": round(float(air_temp), 2),
        "process_temp_K": round(float(process_temp), 2),
        "rot_speed_rpm": round(float(rot_speed), 1),
        "torque_Nm": round(float(torque), 2),
        "tool_wear_min": round(float(tool_wear), 1),
        "failure_probability": round(failure_prob, 4),
        "prediction": "FAILURE RISK" if pred_label == 1 else "HEALTHY"
    })

with open(OUT_PATH, "w") as f:
    json.dump(records, f, indent=2)

n_alerts = sum(1 for r in records if r["prediction"] == "FAILURE RISK")
print(f"Simulated {N_STEPS} timesteps, {n_alerts} alert(s) raised.")
print(f"Saved stream to {OUT_PATH}")
