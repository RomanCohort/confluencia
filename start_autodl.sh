#!/bin/bash
# Confluencia — AutoDL GPU startup script
# Usage: bash start_autodl.sh

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo "=== Confluencia v2.3 — AutoDL GPU ==="
echo "Working dir: $(pwd)"
echo "Python: $(which python3 2>/dev/null || which python)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo ""

# Check if dependencies are installed
python3 -c "import streamlit" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements-shared-full.txt
    pip install -e .
}

echo "Starting Confluencia Joint Frontend on port 8501..."
echo "Access via: http://localhost:8501"
echo ""

streamlit run confluencia_joint/joint_streamlit.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true
