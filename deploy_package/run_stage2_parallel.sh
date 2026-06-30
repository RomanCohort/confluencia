#!/bin/bash
# 并行运行10个Stage 2进程

echo "Starting 10 parallel Stage 2 processes..."

for i in {0..9}; do
    input_dir="stage1_output_part$(printf '%02d' $i)"
    output_dir="stage2_output_part$(printf '%02d' $i)"

    if [ -d "$input_dir" ]; then
        echo "Launching Part $i: $input_dir"

        # 在后台运行每个部分
        python run_stage2_only.py \
            --input "$input_dir" \
            --output "$output_dir" \
            --config config_quality.yaml &
    else
        echo "Warning: $input_dir not found, skipping Part $i"
    fi
done

echo "All processes launched. Waiting for completion..."
wait

echo "✓ All Stage 2 processes completed!"

# 合并所有输出
echo "Merging outputs..."
mkdir -p stage2_output_all

for i in {0..9}; do
    src_dir="stage2_output_part$(printf '%02d' $i)"
    if [ -d "$src_dir" ]; then
        cp -r "$src_dir"/* stage2_output_all/
    fi
done

echo "✓ All outputs merged to stage2_output_all/"