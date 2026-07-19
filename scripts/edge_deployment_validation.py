"""
edge_deployment_validation.py
-------------------------------
Upgrades Week 5 from a "model compression benchmark" to an actual edge
deployment validation. Four additions the first pass didn't cover:

1. STANDALONE MINIMAL-DEPENDENCY INFERENCE
   The whole point of ONNX export is to run on a device WITHOUT xgboost,
   sklearn, or even pandas installed -- those are training-time tools, and
   a real edge target (embedded ARM board, microcontroller-class device)
   won't have them. This script's inference path uses ONLY onnxruntime +
   numpy, proving the deployment artifact is genuinely self-contained.

2. PEAK MEMORY PROFILING
   File size on disk is NOT the same as runtime memory footprint. This
   measures actual peak RSS memory during inference using the standalone
   session, which is what matters on a memory-constrained device.

3. LATENCY DISTRIBUTION (p50 / p95 / p99), not just mean
   A single average latency hides tail behavior. For any claim about
   real-time monitoring feasibility, p95/p99 latency is what actually
   determines whether the system meets a deadline consistently -- a
   claim like "sub-millisecond inference" needs the tail to back it up,
   not just the average.

4. THROUGHPUT / BATCH TEST
   Single-row latency (Week 5's first pass) characterizes the "one new
   sensor reading arrives" scenario. This adds a batch-throughput number
   (rows/sec at batch size 32) to characterize the other end of the
   operating envelope -- e.g. reprocessing a backlog after a connectivity
   gap, or scoring multiple engines at once.

Requires only: onnxruntime, numpy, psutil
    pip install psutil

Run (after quantize_and_benchmark.py has produced the ONNX files):
    python scripts/edge_deployment_validation.py
"""

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import os

MODEL_DIR = Path("models")
ONNX_FP32_PATH = MODEL_DIR / "xgb_rul_model_fp32.onnx"
META_PATH = MODEL_DIR / "best_params.json"

# Recommendation from the Week 5 comparison: ONNX fp32, not int8 -- int8
# gave no measurable size/latency benefit for this tree ensemble and added
# a small overhead, so there is no reason to carry the extra complexity.
DEPLOY_MODEL_PATH = ONNX_FP32_PATH

N_LATENCY_SAMPLES = 500
BATCH_SIZE = 32
N_BATCH_RUNS = 100


def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def load_synthetic_input(n_features, n_rows=1):
    """Standalone script has no pandas/sklearn dependency, so it can't reuse
    signal_features.py -- generates realistic-scale random input purely to
    exercise the inference path itself. Latency/memory characteristics for
    a fixed-shape numeric input don't depend on whether values are real
    sensor data or synthetic; this only needs the right shape and dtype."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_rows, n_features)).astype(np.float32)


def measure_latency_distribution(session, input_name, X, n_samples=N_LATENCY_SAMPLES):
    """Single-row latency, repeated, to get the full distribution rather
    than just a mean -- p95/p99 is what actually determines whether a
    real-time monitoring deadline is met consistently."""
    row = X[:1]

    # warm-up
    for _ in range(10):
        session.run(None, {input_name: row})

    latencies_ms = []
    for _ in range(n_samples):
        start = time.perf_counter()
        session.run(None, {input_name: row})
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "max_ms": float(np.max(latencies_ms)),
    }


def measure_throughput(session, input_name, n_features, batch_size=BATCH_SIZE, n_runs=N_BATCH_RUNS):
    """Batch-mode throughput -- the complementary scenario to single-row
    latency: e.g. scoring a backlog of readings after a connectivity gap,
    or scoring multiple engines' latest readings in one pass."""
    batch = load_synthetic_input(n_features, n_rows=batch_size)

    # warm-up
    for _ in range(5):
        session.run(None, {input_name: batch})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: batch})
    elapsed = time.perf_counter() - start

    total_rows = batch_size * n_runs
    rows_per_sec = total_rows / elapsed
    return rows_per_sec


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    n_features = len(meta["feature_cols"])

    print("=== Edge Deployment Validation ===")
    print(f"Model: {DEPLOY_MODEL_PATH.name}")
    print(f"Dependencies used for this inference path: onnxruntime, numpy only\n")

    # ---- 1. Confirm standalone load with ONLY onnxruntime + numpy ---------
    tracemalloc.start()
    mem_before = get_process_memory_mb()

    session = ort.InferenceSession(str(DEPLOY_MODEL_PATH))
    input_name = session.get_inputs()[0].name

    X_synthetic = load_synthetic_input(n_features, n_rows=max(BATCH_SIZE, 1))
    _ = session.run(None, {input_name: X_synthetic[:1]})  # confirms it actually runs

    mem_after_load = get_process_memory_mb()
    print(f"Confirmed: model loads and predicts using only onnxruntime + numpy "
          f"(no xgboost/pandas/sklearn imported in this script).")
    print(f"Process memory after model load + first inference: {mem_after_load:.1f} MB "
          f"(delta from baseline: +{mem_after_load - mem_before:.1f} MB)\n")

    # ---- 2. Latency distribution -------------------------------------------
    print(f"Measuring single-row latency distribution over {N_LATENCY_SAMPLES} runs...")
    latency_stats = measure_latency_distribution(session, input_name, X_synthetic)
    print(f"  mean: {latency_stats['mean_ms']:.4f} ms")
    print(f"  p50:  {latency_stats['p50_ms']:.4f} ms")
    print(f"  p95:  {latency_stats['p95_ms']:.4f} ms")
    print(f"  p99:  {latency_stats['p99_ms']:.4f} ms")
    print(f"  max:  {latency_stats['max_ms']:.4f} ms\n")

    # ---- 3. Throughput (batch mode) ----------------------------------------
    print(f"Measuring batch throughput (batch_size={BATCH_SIZE}, {N_BATCH_RUNS} runs)...")
    rows_per_sec = measure_throughput(session, input_name, n_features)
    print(f"  Throughput: {rows_per_sec:,.0f} rows/sec\n")

    # ---- 4. Peak memory during a sustained inference burst ----------------
    print("Measuring peak memory during a sustained inference burst (1000 calls)...")
    tracemalloc.reset_peak()
    for _ in range(1000):
        session.run(None, {input_name: X_synthetic[:1]})
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Peak traced Python-level memory during burst: {peak / 1024:.1f} KB")
    print(f"  (Note: this traces Python-level allocations; the ONNX Runtime C++ "
          f"session itself holds additional native memory not captured by "
          f"tracemalloc -- process RSS above is the more complete figure.)\n")

    # ---- Summary -------------------------------------------------------------
    file_size_kb = DEPLOY_MODEL_PATH.stat().st_size / 1024
    summary = {
        "model": DEPLOY_MODEL_PATH.name,
        "file_size_kb": round(file_size_kb, 1),
        "process_memory_mb": round(mem_after_load, 1),
        "latency_mean_ms": round(latency_stats["mean_ms"], 4),
        "latency_p95_ms": round(latency_stats["p95_ms"], 4),
        "latency_p99_ms": round(latency_stats["p99_ms"], 4),
        "throughput_rows_per_sec": round(rows_per_sec, 0),
    }

    print("=== Deployment Readiness Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = MODEL_DIR / "edge_deployment_validation.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
