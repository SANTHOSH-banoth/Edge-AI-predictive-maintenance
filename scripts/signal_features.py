"""
signal_features.py
---------------------
Week 3, Days 3-4: physics-informed signal processing features on top of
the raw CMAPSS sensor readings. This is the mechanical-engineering
differentiator for this project -- most DA/DS candidates building a
predictive maintenance project stop at "feed raw sensors to a model."
This step encodes real turbine physics into the features themselves.

Background you need to know (for interviews):
NASA documents exactly what each CMAPSS sensor physically measures:
  sensor_2  = T24  Total temperature at LPC (low pressure compressor) outlet
  sensor_3  = T30  Total temperature at HPC (high pressure compressor) outlet
  sensor_4  = T50  Total temperature at LPT (low pressure turbine) outlet
  sensor_11 = Ps30 Static pressure at HPC outlet
  sensor_7  = P30  Total pressure at HPC outlet
(Full sensor list: NASA CMAPSS documentation / Saxena & Goebel, 2008.)

Features added, and the physical reasoning behind each:

1. Rolling mean / rolling std (window=5 cycles) on key sensors
   -> Smooths noise, and a RISING rolling std signals the engine's
      behavior is becoming less stable cycle-to-cycle -- an early
      degradation signature before any single reading looks abnormal.

2. Rate of change (1-cycle diff) on key sensors
   -> A sudden jump in T50 (turbine outlet temp) between consecutive
      cycles is a much stronger warning sign than the absolute value --
      real jet engines are monitored for exactly this kind of trend,
      not just threshold breaches.

3. Thermal Stress Index
   -> A weighted combination of how far T24/T30/T50 have drifted from
      that ENGINE's own early-life baseline (its first 5 cycles).
      Physically: sustained operation above a component's design
      temperature accelerates material fatigue and creep -- this is
      literally why turbine blade coatings and cooling systems exist.

4. Cumulative Thermal Stress (fatigue-style integral)
   -> Running cumulative sum of the Thermal Stress Index over an
      engine's life: approximates integral(stress dt). This mirrors
      real fatigue analysis, where cumulative damage (not instantaneous
      stress) predicts remaining life -- the mechanical engineering
      concept of Miner's rule for cumulative fatigue damage.

5. Frequency-domain feature (FFT-based)
   -> For each 30-cycle window, we run an FFT across the cycle axis of
      a key sensor (Ps30) and extract the dominant non-zero frequency's
      energy. IMPORTANT HONESTY NOTE for interviews: this is FFT applied
      to the discrete cycle-to-cycle signal, not a raw high-frequency
      vibration waveform (CMAPSS doesn't include one). It captures
      OSCILLATORY / periodic degradation patterns across cycles, not
      literal bearing vibration harmonics. Don't oversell it as
      vibration analysis -- it's legitimate frequency-domain feature
      engineering on a slower-timescale signal, and it's worth being
      precise about that distinction if asked.

6. Wavelet transform features (multi-resolution energy)
   -> FFT assumes the frequency content of a window is stationary across
      that whole window -- one spectrum describing all 30 cycles equally.
      Real degradation isn't stationary: a developing fault's frequency
      signature changes AS it progresses, even within one window. A
      discrete wavelet transform (DWT) decomposes the signal into both
      time and frequency simultaneously, so it can localize WHEN a
      frequency-domain shift happened inside the window, not just that
      one exists somewhere in it. This is why wavelet-based features are
      the more common real-world choice for non-stationary condition-
      monitoring signals (vibration, acoustic emission) versus plain FFT.
      SAME HONESTY NOTE as the FFT feature: this decomposes the slower
      cycle-to-cycle signal CMAPSS provides, not a raw vibration waveform.

Output: data/cmapss/train_FD001_engineered.csv
"""

import numpy as np
import pandas as pd
import pywt

DATA_PATH = "data/cmapss/processed/train_FD001.csv"
OUT_PATH = "data/cmapss/processed/train_FD001_engineered.csv"

ROLL_WINDOW = 5
FFT_WINDOW = 30
WAVELET_WINDOW = 64      # db4 @ level 3 needs a longer window than the FFT feature
WAVELET_NAME = "db4"
WAVELET_LEVEL = 3

# Sensors with known, documented physical meaning -- used for physics features
TEMP_SENSORS = ["sensor_2", "sensor_3", "sensor_4"]   # T24, T30, T50
PRESSURE_SENSOR = "sensor_11"     # Ps30
KEY_SENSORS_FOR_ROLLING = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12"]

# Weights reflect that turbine outlet temp (T50) is the most safety-critical
# of the three -- it's closest to the hottest, most stressed component.
THERMAL_WEIGHTS = {"sensor_2": 0.2, "sensor_3": 0.3, "sensor_4": 0.5}


def add_rolling_features(df, sensor_cols, window=ROLL_WINDOW):
    """Rolling mean/std and rate-of-change, computed PER ENGINE so one
    engine's history never leaks into another's rolling window."""
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    grouped = df.groupby("unit_number")

    for col in sensor_cols:
        df[f"{col}_roll_mean{window}"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{col}_roll_std{window}"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"{col}_rate_of_change"] = grouped[col].transform(lambda s: s.diff().fillna(0))

    return df


def add_thermal_stress_features(df, temp_sensors=TEMP_SENSORS, weights=THERMAL_WEIGHTS):
    """Thermal Stress Index (deviation from each engine's own early-life
    baseline) and its cumulative sum (fatigue-style integral)."""
    df = df.copy()
    stress_series = pd.Series(index=df.index, dtype=float)
    cum_series = pd.Series(index=df.index, dtype=float)

    for uid, group in df.groupby("unit_number"):
        group_sorted = group.sort_values("time_cycles")
        baseline = group_sorted[temp_sensors].iloc[:5].mean()  # first 5 cycles = "healthy baseline"
        stress = sum(weights[s] * (group_sorted[s] - baseline[s]) for s in temp_sensors)
        stress_series.loc[group_sorted.index] = stress.values
        cum_series.loc[group_sorted.index] = stress.cumsum().values

    df["thermal_stress_index"] = stress_series
    df["cumulative_thermal_stress"] = cum_series
    return df


def add_fft_feature(df, sensor=PRESSURE_SENSOR, window=FFT_WINDOW):
    """
    For each cycle, look back `window` cycles of `sensor` (per engine),
    run an FFT, and extract the energy of the strongest non-DC frequency
    component. Cycles with fewer than `window` prior readings get 0
    (not enough history yet for a meaningful spectrum).
    """
    df = df.copy()
    fft_energy = np.zeros(len(df))

    for uid, group in df.groupby("unit_number"):
        idx = group.sort_values("time_cycles").index
        values = df.loc[idx, sensor].values
        for i in range(len(values)):
            if i < window - 1:
                continue
            segment = values[i - window + 1:i + 1]
            segment = segment - segment.mean()  # remove DC component
            spectrum = np.abs(np.fft.rfft(segment))
            if len(spectrum) > 1:
                fft_energy[idx[i]] = spectrum[1:].max()  # strongest non-DC frequency

    df[f"{sensor}_fft_dominant_energy"] = fft_energy
    return df


def add_wavelet_features(df, sensor=PRESSURE_SENSOR, window=WAVELET_WINDOW,
                          wavelet=WAVELET_NAME, level=WAVELET_LEVEL):
    """
    For each cycle, take the trailing `window` cycles of `sensor` (per
    engine), run a discrete wavelet transform (DWT), and extract the
    energy in each decomposition level as a feature. Level 0 is the
    coarsest approximation (slow trend); higher levels are progressively
    finer detail coefficients (faster-changing structure within the
    window). A rising energy in the finer detail levels, specifically,
    is the multi-resolution equivalent of "this sensor got noisier" --
    but localized in time, unlike the single FFT feature above.

    Cycles with fewer than `window` prior readings get all-zero wavelet
    energies (not enough history yet). Auto-caps the decomposition level
    if `window` is too short for the requested level/wavelet combination
    (pywt raises on that rather than silently truncating).
    """
    df = df.copy()

    # pywt errors if `level` is too deep for `window`, given the wavelet's
    # filter length -- cap it up front and warn once, rather than crash
    # partway through the loop on the very first engine.
    filter_len = pywt.Wavelet(wavelet).dec_len
    max_level = pywt.dwt_max_level(window, filter_len)
    if level > max_level:
        print(f"WARNING: requested wavelet level={level} too deep for "
              f"window={window} with '{wavelet}' (filter_len={filter_len}). "
              f"Capping to level={max_level}.")
        level = max_level

    energy_cols = [f"{sensor}_wavelet_energy_L{i}" for i in range(level + 1)]
    energies = {c: np.zeros(len(df)) for c in energy_cols}

    for uid, group in df.groupby("unit_number"):
        idx = group.sort_values("time_cycles").index
        values = df.loc[idx, sensor].values
        for i in range(len(values)):
            if i < window - 1:
                continue
            segment = values[i - window + 1:i + 1]
            segment = segment - segment.mean()
            coeffs = pywt.wavedec(segment, wavelet, level=level)
            # coeffs[0] = coarsest approximation, coeffs[1:] = detail
            # coefficients from coarsest to finest
            for lvl, c in enumerate(coeffs):
                energies[energy_cols[lvl]][idx[i]] = float(np.sum(np.asarray(c) ** 2))

    for c in energy_cols:
        df[c] = energies[c]
    return df


def main():
    print("Loading labeled CMAPSS training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Input shape: {df.shape}")

    print(f"\nAdding rolling mean/std + rate-of-change (window={ROLL_WINDOW}) for {len(KEY_SENSORS_FOR_ROLLING)} sensors...")
    df = add_rolling_features(df, KEY_SENSORS_FOR_ROLLING)

    print("Adding thermal stress index + cumulative thermal stress...")
    df = add_thermal_stress_features(df)

    print(f"Adding FFT-based frequency feature (window={FFT_WINDOW}) on {PRESSURE_SENSOR}...")
    df = add_fft_feature(df)

    print(f"Adding wavelet-based multi-resolution energy features (window={WAVELET_WINDOW}, "
          f"wavelet={WAVELET_NAME}, level={WAVELET_LEVEL}) on {PRESSURE_SENSOR}...")
    df = add_wavelet_features(df)

    original_cols = pd.read_csv(DATA_PATH, nrows=1).columns
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"\nNew engineered columns added ({len(new_cols)}):")
    for c in new_cols:
        print(" -", c)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nOutput shape: {df.shape}")
    print(f"Saved to {OUT_PATH}")

    print("\nSanity check -- engine #1, cycles near end of life (thermal stress "
          "and wavelet detail energy should trend upward):")
    eng1 = df[df["unit_number"] == 1].tail(8)
    wavelet_check_col = f"{PRESSURE_SENSOR}_wavelet_energy_L{WAVELET_LEVEL}"
    cols_to_show = ["time_cycles", "RUL", "thermal_stress_index",
                     "cumulative_thermal_stress", wavelet_check_col]
    print(eng1[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
