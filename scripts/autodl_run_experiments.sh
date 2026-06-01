#!/bin/bash
# AutoDL Experiment Runner
# ========================
# Run on AutoDL GPU instance for Confluencia reviewer experiments
#
# Prerequisites:
#   1. AutoDL instance with GPU (RTX 3090 / A100 recommended)
#   2. conda/miniconda installed
#
# Usage:
#   chmod +x autodl_run_experiments.sh
#   ./autodl_run_experiments.sh

set -e

echo "============================================"
echo "Confluencia AutoDL Experiment Runner"
echo "============================================"

# --- Configuration ---
PROJECT_DIR="/root/IGEM"  # Change to your project path on AutoDL
HF_MIRROR="https://hf-mirror.com"  # AutoDL HuggingFace mirror

# --- Environment Setup ---
echo ""
echo "[1/5] Setting up Python environment..."

# Use AutoDL's base conda
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true

# Create or activate environment
if conda env list | grep -q "confluencia"; then
    echo "  Activating existing confluencia environment..."
    conda activate confluencia
else
    echo "  Creating new confluencia environment..."
    conda create -n confluencia python=3.10 -y
    conda activate confluencia
fi

# --- Install Dependencies ---
echo ""
echo "[2/5] Installing dependencies..."

pip install numpy pandas scikit-learn scipy matplotlib fair-esm 2>/dev/null || \
pip install numpy pandas scikit-learn scipy matplotlib

# fair-esm for ESM-2 model (optional, needed for benchmark)
pip install fair-esm 2>/dev/null && echo "  fair-esm installed" || echo "  fair-esm not available, ESM-2 benchmark may use cached embeddings"

# --- HuggingFace Mirror ---
echo ""
echo "[3/5] Configuring HuggingFace mirror for AutoDL..."
export HF_ENDPOINT="${HF_MIRROR}"
echo "  HF_ENDPOINT=${HF_ENDPOINT}"

# --- Run k_uptake Sensitivity Analysis ---
echo ""
echo "[4/5] Running RNACTM k_uptake sensitivity analysis..."
echo "  (Pure Python, no GPU needed)"

cd "${PROJECT_DIR}/scripts"

python benchmark_k_uptake_sensitivity.py

echo ""
echo "  k_uptake sensitivity results:"
cat benchmark_results/k_uptake_sensitivity_all.csv 2>/dev/null || echo "  [Warning] No CSV output found"

# --- Run ESM-2 Pooling Benchmark ---
echo ""
echo "[5/5] Running ESM-2 pooling benchmark..."
echo "  This requires GPU and will download ESM-2 model weights"

# First try with 650M model (paper-relevant)
ESM2_MODEL_SIZE=650M ESM2_BENCHMARK_N=2000 \
  HF_ENDPOINT="${HF_MIRROR}" \
  python benchmark_esm2_pooling.py 2>/dev/null || {
    echo ""
    echo "  [Fallback] 650M failed, trying 35M model (CPU-compatible)..."
    ESM2_MODEL_SIZE=35M ESM2_BENCHMARK_N=500 \
      HF_ENDPOINT="${HF_MIRROR}" \
      python benchmark_esm2_pooling.py
}

echo ""
echo "  ESM-2 pooling results:"
ls benchmark_results/esm2_pooling_benchmark_*.json 2>/dev/null || echo "  [Warning] No JSON output found"

# --- Summary ---
echo ""
echo "============================================"
echo "Experiment Complete!"
echo "============================================"
echo ""
echo "Results saved to: ${PROJECT_DIR}/scripts/benchmark_results/"
echo ""
echo "Key files:"
echo "  - k_uptake_sensitivity_all.csv    (PK sensitivity data)"
echo "  - k_uptake_sensitivity_summary.json"
echo "  - k_uptake_sensitivity_IV.png     (IV route plot)"
echo "  - k_uptake_sensitivity_IM.png     (IM route plot)"
echo "  - k_uptake_sensitivity_SC.png     (SC route plot)"
echo "  - esm2_pooling_benchmark_*.json   (ESM-2 results)"
echo ""
echo "Copy results back to local machine:"
echo "  scp -r root@<AUTODL_IP>:${PROJECT_DIR}/scripts/benchmark_results/ ./benchmark_results/"
