"""
generate_data.py
-----------------
Generates a synthetic industrial rotating-machinery sensor dataset that
mirrors the structure and physics of the well-known AI4I 2020 Predictive
Maintenance dataset (UCI). Used because raw plant/IoT sensor data is
hard to source quickly, but the relationships below (heat dissipation,
overstrain, tool wear) follow the same real engineering logic used in
that dataset and in industry.

Output: data/machine_sensor_data.csv  (10,000 rows)

Columns:
  UDI                 - unique row id
  Type                - product quality variant (L=Low,M=Medium,H=High)
  Air_temperature_K
  Process_temperature_K
  Rotational_speed_rpm
  Torque_Nm
  Tool_wear_min
  Machine_failure     - target (1 = failed, 0 = healthy)
  Failure_type        - which failure mode fired (for explainability)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ---- Product quality variant (affects tool wear rate) ----
type_choice = np.random.choice(["L", "M", "H"], size=N, p=[0.6, 0.3, 0.1])
wear_rate_bonus = {"L": 0, "M": 3, "H": 5}  # extra tool wear added, per AI4I logic

# ---- Base sensor readings ----
air_temp = np.random.normal(300, 2, N)                      # Kelvin
process_temp = air_temp + np.random.normal(10, 1, N)         # Kelvin, correlated with air temp
rot_speed = np.random.normal(1500, 180, N).clip(1000, 2900)  # rpm
torque = np.random.normal(40, 10, N).clip(3, 76)             # Nm
tool_wear = np.random.uniform(0, 250, N)
tool_wear = tool_wear + np.array([wear_rate_bonus[t] for t in type_choice])

df = pd.DataFrame({
    "UDI": np.arange(1, N + 1),
    "Type": type_choice,
    "Air_temperature_K": air_temp.round(2),
    "Process_temperature_K": process_temp.round(2),
    "Rotational_speed_rpm": rot_speed.round(0).astype(int),
    "Torque_Nm": torque.round(2),
    "Tool_wear_min": tool_wear.round(1),
})

# ---- Failure modes (physics-inspired rules + noise, same logic family as AI4I) ----
# 1. Heat Dissipation Failure (HDF): temp difference too small AND low rotational speed
temp_diff = df["Process_temperature_K"] - df["Air_temperature_K"]
hdf = (temp_diff < 7.7) & (df["Rotational_speed_rpm"] < 1300)

# 2. Power Failure (PWF): power = torque * rot_speed (rad/s) outside safe band
power = df["Torque_Nm"] * (df["Rotational_speed_rpm"] * 2 * np.pi / 60)
pwf = (power < 2500) | (power > 10500)

# 3. Overstrain Failure (OSF): tool wear * torque exceeds a type-dependent threshold
osf_threshold = df["Type"].map({"L": 13500, "M": 14500, "H": 15500})
osf = (df["Tool_wear_min"] * df["Torque_Nm"]) > osf_threshold

# 4. Tool Wear Failure (TWF): tool wear beyond end-of-life window (with randomness)
twf = (df["Tool_wear_min"] > 225) & (np.random.rand(N) < 0.55)

# 5. Random Failures (RNF): rare unexplained failures (sensor/electrical noise)
rnf = np.random.rand(N) < 0.001

machine_failure = (hdf | pwf | osf | twf | rnf).astype(int)

failure_type = np.select(
    [twf, hdf, pwf, osf, rnf],
    ["Tool_Wear_Failure", "Heat_Dissipation_Failure", "Power_Failure",
     "Overstrain_Failure", "Random_Failure"],
    default="No_Failure"
)

df["Machine_failure"] = machine_failure
df["Failure_type"] = failure_type

print("Failure rate: %.2f%%" % (df["Machine_failure"].mean() * 100))
print(df["Failure_type"].value_counts())

df.to_csv("data/machine_sensor_data.csv", index=False)
print("\nSaved to data/machine_sensor_data.csv, shape:", df.shape)
