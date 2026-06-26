#!/bin/bash
# ============================================================
# DGX Spark - 高质量批量circRNA结构生成脚本
# ============================================================
# 适用场景：算力充足，追求最高质量训练数据
# 硬件需求：DGX Spark (8× A100/H100 80GB)
# 产出格式：可直接导入TorusFold训练
# ============================================================

set -euo pipefail

# ============================================================
# 配置
# ============================================================
NUM_WORKERS=8                    # GPU数量
CONFIG_FILE="config_quality.yaml" # 高质量配置
MODE="quality"                   # 质量模式（20ns MD）

# 时间戳输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/dgx/output/circrna_batch_${TIMESTAMP}"

# 输入数据（需要预先准备）
INPUT_FASTA="${1:-/dgx/data/circrna_sequences.fasta}"

echo "============================================================"
echo "  circRNA 3D 批量生成 Pipeline — DGX Spark (质量模式)"
echo "============================================================"
echo "  输入: $INPUT_FASTA"
echo "  输出: $OUTPUT_DIR"
echo "  模式: $MODE (20ns MD)"
echo "  GPU数: $NUM_WORKERS"
echo "  日期: $(date)"
echo "============================================================"

# ============================================================
# Step 0: 环境检测
# ============================================================
echo ""
echo "[Step 0] 检测环境..."

check_env() {
    local failed=0
    python -c "import RNA; print('  ✓ ViennaRNA', RNA.__file__)" || { echo "  ✗ ViennaRNA 缺失"; failed=1; }
    python -c "import openmm; print('  ✓ OpenMM', openmm.__file__)" || { echo "  ✗ OpenMM 缺失"; failed=1; }
    python -c "import ray; print('  ✓ Ray', ray.__file__)" || { echo "  ✗ Ray 缺失"; failed=1; }
    python -c "import torch; print('  ✓ PyTorch', torch.__file__)" || { echo "  ✗ PyTorch 缺失"; failed=1; }
    python -c "import numpy; print('  ✓ NumPy', numpy.__file__)" || { echo "  ✗ NumPy 缺失"; failed=1; }
    python -c "import yaml; print('  ✓ PyYAML', yaml.__file__)" || { echo "  ✗ PyYAML 缺失"; failed=1; }

    # 检查RoseTTAFold2NA
    if [ -d "RoseTTAFold2NA" ]; then
        echo "  ✓ RoseTTAFold2NA: $(ls RoseTTAFold2NA/run_infer.py 2>/dev/null || echo '缺少run_infer.py')"
    else
        echo "  ! RoseTTAFold2NA 未安装"
        echo "    将使用 dummy 模式（仅测试用）"
    fi

    # 检查GPU
    echo ""
    echo "  GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo "  (未检测到GPU)"

    return $failed
}

check_env || {
    echo ""
    echo "⚠ 环境不完整，请安装缺失依赖"
    echo "  pip install openmm ray pyyaml"
    echo "  conda install -c bioconda viennarna"
    exit 1
}

# ============================================================
# Step 1: 预过滤（快速筛掉不稳定的序列）
# ============================================================
echo ""
echo "[Step 1] 预过滤序列..."
echo "  输入: $INPUT_FASTA"

PREFILTERED="/dgx/data/prefiltered_${TIMESTAMP}.fasta"
PREFILTER_REPORT="/dgx/data/prefilter_report_${TIMESTAMP}.json"

python prefilter.py \
    --fasta "$INPUT_FASTA" \
    --output "$PREFILTERED" \
    --config "$CONFIG_FILE" \
    --report "$PREFILTER_REPORT"

# 读取过滤后的序列数
N_SEQS=$(grep -c "^>" "$PREFILTERED" 2>/dev/null || echo "0")
echo "  过滤后保留: $N_SEQS 条序列"

if [ "$N_SEQS" -eq 0 ]; then
    echo "  ! 无序列通过过滤，请调整prefilter阈值"
    exit 1
fi

# ============================================================
# Step 2: 运行主pipeline（Ray并行，8 GPU）
# ============================================================
echo ""
echo "[Step 2] 运行3D结构生成 Pipeline..."
echo "  开始时间: $(date)"
echo ""

START_TIME=$(date +%s)

python parallel_worker.py \
    --config "$CONFIG_FILE" \
    --fasta "$PREFILTERED" \
    --num-workers "$NUM_WORKERS" \
    --output "$OUTPUT_DIR" \
    --export-torusfold

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "============================================================"
echo "  生成完成！"
echo "  耗时: ${ELAPSED_HOURS}h (${ELAPSED}s)"
echo "  平均: $(echo "scale=1; $ELAPSED / $N_SEQS" | bc)s/seq"
echo "============================================================"

# ============================================================
# Step 3: 质量报告汇总
# ============================================================
echo ""
echo "[Step 3] 生成质量报告..."

REPORT_PATH="$OUTPUT_DIR/dataset_report.json"
if [ -f "$REPORT_PATH" ]; then
    echo ""
    echo "============================================================"
    echo "  最终质量报告"
    echo "============================================================"
    python -c "
import json
with open('$REPORT_PATH') as f:
    r = json.load(f)
print(f'  总结构数: {r[\"total_structures\"]}')
print(f'  置信度:   {r[\"confidence\"][\"mean\"]:.3f} ± {r[\"confidence\"][\"std\"]:.3f}')
print(f'  能量:     {r[\"energy\"][\"mean\"]:.1f} kJ/mol')
print(f'  BSJ距离:  {r[\"bsj_distance\"][\"mean\"]:.2f} Å')
if 'distribution' in r.get('confidence', {}):
    print(f'  分布: {r[\"confidence\"][\"distribution\"]}')
print(f'')
print(f'  分量分数:')
for k, v in r.get('component_scores', {}).items():
    print(f'    {k}: {v[\"mean\"]:.3f}')
"
fi

# ============================================================
# Step 4: 复制到TorusFold训练目录
# ============================================================
echo ""
echo "[Step 4] 准备TorusFold训练数据..."

TORUSFOLD_DATA="/dgx/data/torusfold_training_${TIMESTAMP}/"
if [ -d "$OUTPUT_DIR/torusfold_format" ]; then
    cp -r "$OUTPUT_DIR/torusfold_format" "$TORUSFOLD_DATA"
    echo "  ✓ 已复制到: $TORUSFOLD_DATA"
    echo "    文件:"
    ls -lh "$TORUSFOLD_DATA"
else
    echo "  ! TorusFold格式数据未生成"
fi

# ============================================================
# Step 5: 输出汇总
# ============================================================
echo ""
echo "============================================================"
echo "  DGX Spark Pipeline 全部完成"
echo "============================================================"
echo "  输入序列数: $N_SEQS"
echo "  总耗时: ${ELAPSED_HOURS}h"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练数据: $TORUSFOLD_DATA"
echo ""
echo "  下一步:"
echo "    1. 检查质量: python -c \"import json; print(json.load(open('${REPORT_PATH}')))\""
echo "    2. 训练TorusFold: cd /workspace/torusfold && python train_curriculum.py --labels ${TORUSFOLD_DATA}"
echo "    3. 低置信度补充: 对 < 0.8 的结构重新模拟"
echo "============================================================"