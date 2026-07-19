"""
benchmark_constrained.py
---------------------------
Open thread #1: no physical Raspberry Pi available, so this is an honest
PROXY rather than a real hardware test -- restricting ONNX Runtime to a
single CPU thread. This does NOT replicate a Pi (different CPU architecture,
clock speed, memory bandwidth, cache sizes all matter and aren't simulated
here). What it DOES tell you: how much your model's speed currently depends
on multi-core parallelism. A model that only looks fast because it's using
all 8 laptop cores would be a bad surprise on a single/dual-core embedded
device -- this test catches that kind of hidden assumption.

Be precise about this distinction if it comes up in an interview: this is
"single-thread CPU behavior," not "verified on embedded hardware." Real
hardware testing remains a documented limitation of this project (that's a
fine, honest thing to say -- most student projects don't test on physical
edge devices, and saying so directly is better than implying otherwise).

Run:
    python scripts/benchmark_constrained.py
"""

import os
import time
import numpy as np
import onnxruntime as ort

MODEL_DIR = "models"
LSTM_ONNX_PATH = os.path.join(MODEL_DIR, "lstm_rul.onnx")
SEQ_DIR = os.path.join("data", "cmapss", "sequences")
N_RUNS = 300


def benchmark(session, X, n_runs=N_RUNS):
    input_name = session.get_inputs()[0].name
    row = X[:1]
    for _ in range(10):
        session.run(None, {input_name: row})  # warm-up
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: row})
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms


def main():
    print("Loading real test sequences...")
    X_test = np.load(os.path.join(SEQ_DIR, "X_test.npy")).astype(np.float32)
    print(f"X_test: {X_test.shape}\n")

    # ---- Default (unrestricted -- uses all available CPU cores/threads) ----
    default_opts = ort.SessionOptions()
    default_session = ort.InferenceSession(
        LSTM_ONNX_PATH, sess_options=default_opts, providers=["CPUExecutionProvider"]
    )
    default_latency = benchmark(default_session, X_test)

    # ---- Single-thread (proxy for a resource-constrained device) ----
    constrained_opts = ort.SessionOptions()
    constrained_opts.intra_op_num_threads = 1
    constrained_opts.inter_op_num_threads = 1
    constrained_session = ort.InferenceSession(
        LSTM_ONNX_PATH, sess_options=constrained_opts, providers=["CPUExecutionProvider"]
    )
    constrained_latency = benchmark(constrained_session, X_test)

    print("=== Single-row inference latency ===")
    print(f"Default (all cores available): {default_latency:.3f} ms")
    print(f"Single-thread (constrained-device proxy): {constrained_latency:.3f} ms")

    slowdown = constrained_latency / default_latency
    print(f"\nSlowdown factor: {slowdown:.2f}x")
    if slowdown < 2.0:
        verdict = ("Latency holds up well even single-threaded -- this model doesn't "
                    "appear to lean heavily on multi-core parallelism, a good sign for "
                    "deployment on constrained hardware.")
    else:
        verdict = ("Meaningful slowdown single-threaded -- worth keeping in mind if "
                    "targeting genuinely single-core hardware; real-world latency on "
                    "such a device would need actual on-device testing to confirm.")
    print(verdict)

    print(f"\nNOTE: this is a same-CPU thread-count proxy, not a real embedded-device")
    print(f"benchmark. Documented as a limitation -- see project README.")


if __name__ == "__main__":
    main()
