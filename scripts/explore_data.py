"""
explore_data.py
----------------
Quick look at how sensor readings differ between healthy machines
and ones that failed. Run this after generate_data.py.
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

df = pd.read_csv("data/machine_sensor_data.csv")

print("Total rows:", len(df))
print("Failures:", df["Machine_failure"].sum(), "\n")

# Average sensor value, grouped by whether the machine failed (1) or not (0)
comparison = df.groupby("Machine_failure")[[
    "Air_temperature_K", "Process_temperature_K",
    "Rotational_speed_rpm", "Torque_Nm", "Tool_wear_min"
]].mean().round(2)

print("Average sensor values: healthy (0) vs failed (1)")
print(comparison)

print("\nFailure breakdown by type:")
print(df["Failure_type"].value_counts())