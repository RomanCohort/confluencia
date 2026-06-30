#!/bin/bash
# 并行运行10个Stage 1进程

echo "Starting 10 parallel Stage 1 processes..."

for i in {0..9}; do
    part_file="fasta_parts/part_$(printf '%02d' $i).fa"
    output_dir="stage1_output_part$(printf '%02d' $i)"

    echo "Launching Part $i: $part_file"

    # 在后台运行每个部分
    python run_stage1_only.py \
        --fasta "$part_file" \
        --output "$output_dir" &
done

echo "All 10 processes launched. Waiting for completion..."
wait

echo "✓ All Stage 1 processes completed!"

# 合并所有输出
echo "Merging outputs..."
mkdir -p stage1_output_all

for i in {0..9}; do
    src_dir="stage1_output_part$(printf '%02d' $i)"
    if [ -d "$src_dir" ]; then
        cp -r "$src_dir"/* stage1_output_all/
    fi
done

echo "✓ All outputs merged to stage1_output_all/"