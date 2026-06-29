#!/bin/bash
# ============================================================
# 清理deploy_package目录下的旧版本pipeline文件
# ============================================================
#
# 问题：这些文件是旧版本的复制，没有被git跟踪
# 当运行 python pipeline.py 时，Python优先加载这些旧文件
# 而不是 circrna_3d_pipeline/ 目录下的新文件
#
# 这导致测试脚本通过但pipeline运行失败：
# - 测试脚本导入 circrna_3d_pipeline/stage2_trrosetta.py (新代码)
# - pipeline.py 导入 stage2_trrosetta.py (旧代码)
#
# 解决：删除这些旧文件，确保使用新版本

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

echo "============================================================"
echo "  清理旧版本pipeline文件"
echo "============================================================"

# 列出要删除的文件
OLD_FILES=(
    "pipeline.py"
    "stage1_vienna.py"
    "stage2_trrosetta.py"
    "stage2_rosetta.py"
    "stage3_cyclize.py"
    "stage3_cyclize_optimized.py"
    "stage4_md.py"
    "stage5_quality.py"
    "prefilter.py"
    "conformer_clustering.py"
    "circbase_converter.py"
    "parallel_worker.py"
    "parallel_worker_trrosetta.py"
    "__init__.py"
)

echo ""
echo "检查旧文件："
for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ 找到旧文件: $file"
    fi
done

echo ""
echo "删除旧文件："
for file in "${OLD_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已删除: $file"
    fi
done

echo ""
echo "============================================================"
echo "  清理完成！"
echo "============================================================"
echo ""
echo "现在运行正确的pipeline："
echo "  python circrna_3d_pipeline/pipeline.py --config config_quality.yaml --fasta circbase_filtered_5000.fa --output circbase_3d_full_output --export-torusfold"
echo ""