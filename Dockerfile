# Edge-AI Predictive Maintenance -- API service container
#
# Builds a container that serves scripts/api_server.py: the FastAPI
# service exposing /predict (RUL + anomaly detection, LSTM+autoencoder)
# and /predict/failure_risk (cloud/edge model-selector, RandomForest+MLP).
#
# Build:
#   docker build -t edge-ai-pm-api .
#
# Run:
#   docker run -p 8000:8000 edge-ai-pm-api
#
# Then open http://localhost:8000/docs

FROM python:3.12-slim

WORKDIR /app

# System deps needed by scientific Python packages (scipy/onnxruntime
# wheels sometimes need these at import time even with prebuilt wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# torch was pinned as a CPU-only build (2.13.0+cpu) on the original
# training machine. Regular PyPI doesn't host CPU-only wheels under that
# exact tag, so torch is installed separately from PyTorch's CPU wheel
# index, and excluded from the main requirements.txt install below.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Only what the API actually needs at runtime: the serving code, the
# trained model artifacts it loads at startup, and predict.py (the
# pipeline logic api_server.py imports from). Training scripts, tests,
# and raw/processed data are intentionally NOT copied -- they're not
# needed to serve predictions and would bloat the image for no benefit.
COPY scripts/api_server.py scripts/predict.py ./scripts/
COPY models/ ./models/

# Only the pre-built sequence arrays, not the raw/intermediate CMAPSS
# files -- this is what /demo/engine/{id} reads to serve predictions on
# real held-out test engines without a request body.
COPY data/cmapss/sequences/ ./data/cmapss/sequences/

EXPOSE 8000

# api_server.py's on_event("startup") handler loads models from the
# "models" directory using a relative path, so the container's working
# directory must be /app (set by WORKDIR above) for that to resolve.
CMD uvicorn scripts.api_server:app --host 0.0.0.0 --port ${PORT:-8000}
