# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — RTV Field Image Classifier API
# Task 4: Containerised deployment
#
# This Dockerfile creates a self-contained image that includes:
# - Python 3.11 runtime
# - All required libraries
# - The API source code
# - The trained model checkpoint
#
# Build:  docker build -t rtv-classifier .
# Run:    docker run -p 8000:8000 rtv-classifier
# Test:   curl -X POST http://localhost:8000/predict -F "file=@image.jpg"
# ─────────────────────────────────────────────────────────────────────────────

# We use the official Python slim image — full Python without unnecessary system packages
# "slim" keeps the image size reasonable (~200MB vs ~900MB for the full image)
FROM python:3.11-slim

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 is needed by OpenCV (used internally by Albumentations for some operations)
# We install and clean up in one RUN to keep image layers small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer
# If only the source code changes, Docker won't re-install packages
COPY api/requirements_api.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
# src/ contains dataset.py and model.py — needed by the API
COPY src/ ./src/
COPY api/main.py ./api/main.py

# ── Copy trained model checkpoint ─────────────────────────────────────────────
# This must exist in outputs/best_model.pth before you build the container
# Run python src/train.py first to generate it
COPY outputs/best_model.pth ./outputs/best_model.pth

# ── Environment variables ──────────────────────────────────────────────────────
# Tell the API where to find the model checkpoint inside the container
ENV MODEL_CHECKPOINT_PATH=/app/outputs/best_model.pth

# Prevent Python from writing .pyc files (keeps the container clean)
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output is sent directly to the terminal (for logging)
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will periodically call this to check if the API is alive
# If it fails 3 times, the container is marked as unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Run the API ────────────────────────────────────────────────────────────────
# uvicorn is the ASGI server that runs FastAPI
# --host 0.0.0.0 makes it accessible from outside the container
# --workers 1 is safe for CPU inference; increase for GPU multi-core
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
