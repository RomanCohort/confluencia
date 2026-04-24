# Confluencia Dockerfile
# Multi-stage build for drug + epitope modules

FROM python:3.11-slim AS base

# System dependencies for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    librdkit-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first (for layer caching)
COPY requirements-shared-full.txt .
RUN pip install --no-cache-dir -r requirements-shared-full.txt

# Copy source code
COPY confluencia_shared/ ./confluencia_shared/
COPY confluencia-2.0-drug/ ./confluencia-2.0-drug/
COPY confluencia-2.0-epitope/ ./confluencia-2.0-epitope/
COPY benchmarks/ ./benchmarks/
COPY data/ ./data/
COPY pyproject.toml .
COPY LICENSE .
COPY CITATION.cff .

# Install package
RUN pip install --no-cache-dir -e .

# Expose Streamlit ports
EXPOSE 8501 8502

# Default: run drug frontend
CMD ["streamlit", "run", "confluencia-2.0-drug/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
