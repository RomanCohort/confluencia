#!/bin/bash
# ============================================================
# 大规模CircRNA 3D结构生成 - 严格Quality模式
# 数据: 140,790条序列 (全部CircBase)
# 模式: Quality (严格质量筛选)
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  CircRNA 3D结构生成 - 大规模Quality模式"
echo "============================================================"
echo ""
echo "输入数据: circbase_seqs.fa.gz"
echo "总序列数: 140,790"
echo "处理模式: Quality (严格)"
echo "预计时间: ~$((140790 * 5 / 60 / 8))小时 (8 GPU)"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# Step 1: 检查trRosettaRNA2权重
echo "[Step 1] 检查trRosettaRNA2权重..."
if [ ! -d "weights/params" ]; then
    echo "✗ weights/params目录不存在"
    exit 1
fi

ls -lh weights/params/models/*.pth.tar
echo "✓ 权重文件已就绪"

# Step 2: 验证CircBase数据
echo ""
echo "[Step 2] 验证CircBase数据..."
SEQ_COUNT=$(zcat circbase_seqs.fa.gz | grep -c "^>")
echo "✓ 序列总数: $SEQ_COUNT"

# Step 3: 创建输出目录
echo ""
echo "[Step 3] 创建输出目录..."
mkdir -p output_quality_full
OUTPUT_DIR="output_quality_full"

# Step 4: 显示运行选项
echo ""
echo "============================================================"
echo "  运行选项"
echo "============================================================"
echo ""
echo "由于数据量大(14万条)，建议分批运行："
echo ""
echo "[选项A] 全量Quality模式（推荐先测试小批量）"
echo "  # 先用前1000条测试"
echo "  head -2000 circbase_seqs.fa.gz > test_1k.fa"
echo "  ./run_quality_pipeline.sh test_1k.fa --mode quality"
echo ""
echo "[选项B] 全量Quality模式（直接运行全部）"
echo "  ./run_quality_pipeline.sh circbase_seqs.fa.gz --mode quality"
echo ""
echo "[选项C] Fast模式（快速生成）+ Quality筛选"
echo "  ./run_quality_pipeline.sh circbase_seqs.fa.gz --mode fast --postfilter"
echo ""
echo "============================================================"

# Step 5: 询问用户选择
read -p "请选择运行选项 (A/B/C): " CHOICE
case $CHOICE in
    A|a)
        echo ""
        echo "开始测试运行(前1000条)..."
        echo "head -2000 circbase_seqs.fa.gz > test_1k.fa"
        head -2000 circbase_seqs.fa.gz > test_1k.fa
        TOTAL=$(grep -c "^>" test_1k.fa)
        echo "✓ 测试集: $TOTAL 条序列"
        ./run_quality_pipeline.sh test_1k.fa --mode quality
        ;;
    B|b)
        echo ""
        echo "开始全量Quality模式运行..."
        ./run_quality_pipeline.sh circbase_seqs.fa.gz --mode quality
        ;;
    C|c)
        echo ""
        echo "开始Fast模式 + 后处理筛选..."
        ./run_quality_pipeline.sh circbase_seqs.fa.gz --mode fast --postfilter
        ;;
    *)
        echo "无效选项，退出"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  运行完成！"
echo "============================================================"