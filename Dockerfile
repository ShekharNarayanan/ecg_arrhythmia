FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./ README.md ./

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY validate_detected_peaks.py .
COPY visualize_features.py .

# Install dependencies + local package in editable mode
RUN uv sync --frozen

# Folders mounted at runtime:
# - Data/raw (input data)
# - mlruns (mlflow tracking)
# - peak_detection_plots (output plots)
# - feature_plots (output plots)
