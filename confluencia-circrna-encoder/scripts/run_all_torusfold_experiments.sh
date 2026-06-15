#!/bin/bash
# run_all_torusfold_experiments.sh
# TorusFold 完整实验验证脚本 — AutoDL GPU
#
# 用法：
#   chmod +x run_all_torusfold_experiments.sh
#   ./run_all_torusfold_experiments.sh
#
# 输出：
#   models/torusfold_best.pt         — 最佳模型
#   models/torusfold_history.json    — 训练历史
#   models/torusfold_external_results.json — 外部验证结果
#   output/benchmark/benchmark_results.json — Benchmark 结果
#
# 预计运行时间：~6h (A100 GPU)

set -e  # 遇到错误立即退出

# ─── 配置 ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/../data/circrna"

# 数据路径（AutoDL 环境）
TRAIN_DATA="${DATA_DIR}/unified_training_data.csv"
EXTERNAL_DATA="${DATA_DIR}/sequences_enhanced.csv"

# 如果数据在 AutoDL 标准路径
if [ ! -f "${TRAIN_DATA}" ]; then
    TRAIN_DATA="/root/autodl-tmp/IGEM集成方案/data/circrna/unified_training_data.csv"
fi
if [ ! -f "${EXTERNAL_DATA}" ]; then
    EXTERNAL_DATA="/root/autodl-tmp/IGEM集成方案/data/circrna/sequences_enhanced.csv"
fi

# GPU 配置
DEVICE="cuda"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA not available, falling back to CPU"
    DEVICE="cpu"
fi

# ─── 打印配置 ───────────────────────────────────────────────────────────────────

echo "=============================================="
echo "TorusFold Complete Experiment Validation"
echo "=============================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Training data: ${TRAIN_DATA}"
echo "External data: ${EXTERNAL_DATA}"
echo "Device: ${DEVICE}"
echo "Timestamp: $(date)"
echo "=============================================="
echo ""

cd "${PROJECT_ROOT}"

# ─── Step 1: Unit Tests ───────────────────────────────────────────────────────────

echo "[Step 1] Running unit tests..."
echo "=============================================="

python tests/test_torusfold_v2.py

echo ""
echo "✓ Unit tests passed"
echo ""

# ─── Step 2: Training (Full ESM-2 Backbone) ───────────────────────────────────────

echo "[Step 2] Training TorusFold..."
echo "=============================================="

python scripts/train_torusfold.py \
    --data "${TRAIN_DATA}" \
    --external-data "${EXTERNAL_DATA}" \
    --epochs 20 \
    --batch-size 8 \
    --lr 1e-3 \
    --device "${DEVICE}" \
    --max-seq-len 200 \
    --n-harmonics 8 \
    --hidden-dim 256 \
    --seq-aware-split \
    --esm-model esm2_t12_35M_UR50D

echo ""
echo "✓ Training complete"
echo ""

# ─── Step 3: Benchmark Experiments ───────────────────────────────────────────────

echo "[Step 3] Running benchmark experiments..."
echo "=============================================="

python scripts/benchmark_torusfold.py \
    --data "${EXTERNAL_DATA}" \
    --backbone rna-fm \
    --epochs 30 \
    --batch-size 8 \
    --device "${DEVICE}" \
    --c-z 128 \
    --n-physics-samples 20 \
    --output-dir output/benchmark

echo ""
echo "✓ Benchmark complete"
echo ""

# ─── Step 4: External Validation ───────────────────────────────────────────────

echo "[Step 4] Running external validation..."
echo "=============================================="

python scripts/train_torusfold.py \
    --external-only \
    --device "${DEVICE}" \
    --data "${TRAIN_DATA}" \
    --external-data "${EXTERNAL_DATA}"

echo ""
echo "✓ External validation complete"
echo ""

# ─── Step 5: Summary ───────────────────────────────────────────────────────────

echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - ${PROJECT_ROOT}/models/torusfold_best.pt"
echo "  - ${PROJECT_ROOT}/models/torusfold_history.json"
echo "  - ${PROJECT_ROOT}/models/torusfold_external_results.json"
echo "  - ${PROJECT_ROOT}/output/benchmark/benchmark_results.json"
echo "  - ${PROJECT_ROOT}/output/benchmark/history_circular.csv"
echo "  - ${PROJECT_ROOT}/output/benchmark/history_linear.csv"
echo "  - ${PROJECT_ROOT}/output/benchmark/physics_validation.csv"
echo "  - ${PROJECT_ROOT}/output/benchmark/viennarna_comparison.csv"
echo ""

# 打印关键结果
if [ -f "${PROJECT_ROOT}/models/torusfold_external_results.json" ]; then
    echo "External Validation Results:"
    cat "${PROJECT_ROOT}/models/torusfold_external_results.json"
    echo ""
fi

if [ -f "${PROJECT_ROOT}/output/benchmark/benchmark_results.json" ]; then
    echo "Benchmark Summary:"
    python -c "
import json
with open('${PROJECT_ROOT}/output/benchmark/benchmark_results.json') as f:
    r = json.load(f)
    b1 = r.get('benchmark1_classification', {})
    print(f\"  Circular F1: {b1.get('circular_topology', {}).get('pathway_f1_macro', 'N/A'):.4f}\")
    print(f\"  Linear F1:   {b1.get('linear_baseline', {}).get('pathway_f1_macro', 'N/A'):.4f}\")
    b2 = r.get('benchmark2_physics', {})
    print(f\"  Closure distance: {b2.get('mean_closure_distance', 'N/A'):.3f} Å\")
    b3 = r.get('benchmark3_viennarna', {})
    print(f\"  ViennaRNA MFE diff: {b3.get('mean_mfe_diff', 'N/A'):+.1f} kcal/mol\")
"
    echo ""
fi

echo "=============================================="
echo "Experiment finished at: $(date)"
echo "=============================================="