#!/bin/bash
# 并行运行 Stage 2 进程（自动检测 chunk 数量）
# 用法: bash run_stage2_parallel.sh [num_chunks] [input_prefix] [output_prefix]
# 默认: 30 份, stage1_chunk, stage2_output

NUM=${1:-30}
INPUT_PREFIX=${2:-stage1_chunk}
OUTPUT_PREFIX=${3:-stage2_output}

echo "Starting ${NUM} parallel Stage 2 processes..."
echo "Input prefix: ${INPUT_PREFIX}_XX"
echo "Output prefix: ${OUTPUT_PREFIX}_XX"
echo ""

# 检查是否有 chunk 目录
found=0
for i in $(seq -w 0 $((NUM-1))); do
    input_dir="${INPUT_PREFIX}_${i}"
    if [ -d "$input_dir" ]; then
        found=$((found+1))
    fi
done

if [ $found -eq 0 ]; then
    echo "错误: 找不到 ${INPUT_PREFIX}_XX 目录"
    echo "请先运行: python split_stage1_for_parallel.py stage1_merged ${NUM}"
    exit 1
fi

echo "找到 ${found} 个输入目录，启动并行处理..."

# 启动并行进程
for i in $(seq -w 0 $((NUM-1))); do
    input_dir="${INPUT_PREFIX}_${i}"
    output_dir="${OUTPUT_PREFIX}_${i}"

    if [ -d "$input_dir" ]; then
        echo "Launching Part $i: $input_dir -> $output_dir"
        python run_stage2_only.py \
            --input "$input_dir" \
            --output "$output_dir" &
    fi
done

echo ""
echo "All processes launched. Waiting for completion..."
wait

echo ""
echo "✓ All Stage 2 processes completed!"

# 合并所有输出
echo "Merging outputs..."
mkdir -p ${OUTPUT_PREFIX}_all

for i in $(seq -w 0 $((NUM-1))); do
    src_dir="${OUTPUT_PREFIX}_${i}"
    if [ -d "$src_dir" ]; then
        cp -r "$src_dir"/* ${OUTPUT_PREFIX}_all/ 2>/dev/null
        echo "  Merged: $src_dir ($(ls $src_dir | wc -l) sequences)"
    fi
done

echo ""
echo "✓ All outputs merged to ${OUTPUT_PREFIX}_all/"
echo "Total: $(ls ${OUTPUT_PREFIX}_all | wc -l) sequences"
