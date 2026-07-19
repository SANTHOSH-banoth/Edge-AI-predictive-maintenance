"""
vibration_features.py
----------------------
Week 3: physics-informed vibration / bearing-fault-frequency features for
the AI4I-style edge classifier (data/machine_sensor_data.csv).

This is the piece the CMAPSS side of the project can't do: CMAPSS is
turbofan data with no raw high-frequency channel, so the FFT/wavelet
features built on it operate on a slow cycle-to-cycle signal (see
signal_features.py's honesty notes). The AI4I data, by contrast, is
exactly the classic mechanical-engineering rotating-shaft scenario
(motor/pump/compressor on rolling-element bearings), where bearing fault
frequencies are a real, standard vibration-analysis tool.

Background you need to know (for interviews):
Rolling-element bearings fail at specific, predictable frequencies
determined by their geometry and shaft speed -- NOT at the shaft's own
rotating frequency. The four classic fault frequencies:
  BPFO  Ball Pass Frequency, Outer race  -- outer race defect
  BPFI  Ball Pass Frequency, Inner race  -- inner race defect
  BSF   Ball Spin Frequency              -- defect on a rolling element itself
  FTF   Fundamental Train Frequency      -- cage defect
Standard formulas (contact angle theta=0 for a radial-load deep-groove
bearing), given shaft frequency fr, n balls, ball diameter Bd, pitch
diameter Pd:
  BPFO = (n/2) * fr * (1 - Bd/Pd)
  BPFI = (n/2) * fr * (1 + Bd/Pd)
  BSF  = (Pd / (2*Bd)) * fr * (1 - (Bd/Pd)**2)
  FTF  = (fr/2) * (1 - Bd/Pd)
This project uses a generic deep-groove ball bearing geometry (9 balls,
7.94mm ball diameter, 39.04mm pitch diameter -- representative of a
common small industrial bearing family) since AI4I doesn't specify a real
bearing part number.

IMPORTANT HONESTY NOTE (read before quoting this in an interview):
AI4I has no raw vibration channel. This script SIMULATES a short vibration
waveform per reading and injects energy at the bearing fault frequencies,
with injected amplitude scaled by that row's tool_wear (as a stand-in for
bearing wear, since AI4I doesn't separately track bearing condition). That
coupling (wear -> fault-frequency amplitude) is a MODELED relationship
for demonstration, not a measured one -- be precise about that distinction
if asked, exactly like the FFT feature on the CMAPSS side. What IS real
and correctly applied here: the fault-frequency formulas themselves, and
the FFT-based feature extraction technique (RMS, crest factor, band
energy around each fault frequency) -- that part is exactly how a real
vibration analyst would process an actual accelerometer signal.

Output: data/machine_sensor_data_engineered.csv
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/machine_sensor_data.csv"
OUT_PATH = "data/machine_sensor_data_engineered.csv"

# Simulated accelerometer sampling parameters
FS = 2000            # Hz -- comfortably resolves fault frequencies up to a few hundred Hz
DURATION = 0.5        # seconds per simulated reading
N_SAMPLES = int(FS * DURATION)

# Generic deep-groove ball bearing geometry (representative, not a specific
# real part number -- AI4I doesn't specify one)
N_BALLS = 9
BALL_DIAMETER_MM = 7.94
PITCH_DIAMETER_MM = 39.04
BD_PD_RATIO = BALL_DIAMETER_MM / PITCH_DIAMETER_MM

BAND_HALF_WIDTH_HZ = 3.0   # +/- window around each fault frequency when extracting band energy


def bearing_fault_frequencies(rpm):
    """Standard bearing fault frequency formulas (contact angle = 0)."""
    fr = rpm / 60.0  # shaft rotational frequency, Hz
    bpfo = (N_BALLS / 2) * fr * (1 - BD_PD_RATIO)
    bpfi = (N_BALLS / 2) * fr * (1 + BD_PD_RATIO)
    bsf = (PITCH_DIAMETER_MM / (2 * BALL_DIAMETER_MM)) * fr * (1 - BD_PD_RATIO ** 2)
    ftf = (fr / 2) * (1 - BD_PD_RATIO)
    return {"fr": fr, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def simulate_vibration_signal(rpm, tool_wear_min, max_tool_wear, rng):
    """
    Synthesize a short vibration waveform for one machine reading.

    Always present: broadband noise + a small 1x-shaft-frequency
    component (baseline mechanical unbalance every real machine has).
    Wear-scaled: BPFO/BPFI/BSF amplitude grows with degradation_factor,
    the row's tool wear normalized to [0, 1] against the fleet's max
    observed tool wear, roughly quadratically -- modeling how bearing
    defect severity (and therefore vibration energy at its fault
    frequency) tends to accelerate as a defect propagates, not grow
    linearly with running time.
    """
    t = np.arange(N_SAMPLES) / FS
    freqs = bearing_fault_frequencies(rpm)
    degradation = np.clip(tool_wear_min / max(max_tool_wear, 1e-6), 0, 1)

    signal = rng.normal(0, 0.05, size=N_SAMPLES)  # baseline sensor noise
    signal += 0.03 * np.sin(2 * np.pi * freqs["fr"] * t)  # always-present shaft unbalance

    bpfo_amp = 0.02 + 0.35 * degradation ** 2
    bpfi_amp = 0.015 + 0.25 * degradation ** 2
    bsf_amp = 0.01 + 0.15 * degradation ** 2

    signal += bpfo_amp * np.sin(2 * np.pi * freqs["bpfo"] * t + rng.uniform(0, 2 * np.pi))
    signal += bpfi_amp * np.sin(2 * np.pi * freqs["bpfi"] * t + rng.uniform(0, 2 * np.pi))
    signal += bsf_amp * np.sin(2 * np.pi * freqs["bsf"] * t + rng.uniform(0, 2 * np.pi))

    return signal, freqs


def _band_energy(spectrum, fft_freqs, center_hz, half_width=BAND_HALF_WIDTH_HZ):
    """Sum of FFT magnitude in a narrow band around a target frequency --
    how a real vibration analyst reads energy at a specific fault
    frequency off a spectrum, rather than requiring an exact bin match."""
    mask = (fft_freqs >= center_hz - half_width) & (fft_freqs <= center_hz + half_width)
    if not mask.any():
        return 0.0
    return float(spectrum[mask].sum())


def extract_vibration_features(signal, freqs):
    """RMS and crest factor (standard, model-agnostic vibration health
    indicators) plus band energy at each bearing fault frequency."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    crest_factor = peak / rms if rms > 0 else 0.0

    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    fft_freqs = np.fft.rfftfreq(len(signal), d=1.0 / FS)

    return {
        "vib_rms": rms,
        "vib_crest_factor": crest_factor,
        "vib_bpfo_energy": _band_energy(spectrum, fft_freqs, freqs["bpfo"]),
        "vib_bpfi_energy": _band_energy(spectrum, fft_freqs, freqs["bpfi"]),
        "vib_bsf_energy": _band_energy(spectrum, fft_freqs, freqs["bsf"]),
    }


def add_vibration_features(df, rpm_col="Rotational_speed_rpm", wear_col="Tool_wear_min", seed=11):
    """Simulate a vibration signal per row and extract fault-frequency
    features from it. Deterministic given `seed` so re-runs reproduce
    the same engineered CSV."""
    rng = np.random.default_rng(seed)
    max_wear = df[wear_col].max()

    records = []
    for rpm, wear in zip(df[rpm_col].values, df[wear_col].values):
        signal, freqs = simulate_vibration_signal(rpm, wear, max_wear, rng)
        records.append(extract_vibration_features(signal, freqs))

    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)


def main():
    print("Loading AI4I-style sensor data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Input shape: {df.shape}")

    print(f"\nSimulating vibration signal ({N_SAMPLES} samples @ {FS} Hz per reading) "
          f"and extracting bearing fault-frequency features for {len(df)} rows...")
    df = add_vibration_features(df)

    new_cols = ["vib_rms", "vib_crest_factor", "vib_bpfo_energy", "vib_bpfi_energy", "vib_bsf_energy"]
    print(f"\nNew engineered columns added ({len(new_cols)}):")
    for c in new_cols:
        print(" -", c)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nOutput shape: {df.shape}")
    print(f"Saved to {OUT_PATH}")

    print("\nSanity check -- mean fault-frequency energy, failures vs. healthy rows "
          "(BPFO/BPFI/BSF energy should be higher for Machine_failure=1, since it's "
          "wear-scaled by construction):")
    check = df.groupby("Machine_failure")[["vib_bpfo_energy", "vib_bpfi_energy", "vib_bsf_energy"]].mean()
    print(check.to_string())


if __name__ == "__main__":
    main()
