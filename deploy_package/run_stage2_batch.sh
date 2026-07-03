#!/bin/bash
# Stage 2 分批并行执行（控制 GPU 内存）
# 用法: bash run_stage2_batch.sh [并发数] [输入前缀] [输出前缀]
# 默认: 并发 5, stage1_chunk, stage2_output

CONCURRENCY=${1:-5}
INPUT_PREFIX=${2:-stage1_chunk}
OUTPUT_PREFIX=${3:-stage2_output}

echo "=========================================="
echo "Stage 2 分批并行执行"
echo "并发数: $CONCURRENCY"
echo "=========================================="
echo ""

# 收集所有任务
TASKS=()
for i in $(seq -w 0 29); do
    input_dir="${INPUT_PREFIX}_${i}"
    output_dir="${OUTPUT_PREFIX}_${i}"
    if [ -d "$input_dir" ]; then
        # 检查是否已完成（跳过已完成的）
        if [ -d "$output_dir" ] && [ "$(ls $output_dir 2>/dev/null | wc -l)" -gt 0 ]; then
            echo "跳过已完成的: $output_dir ($(ls $output_dir | wc -l) sequences)"
        else
            TASKS+=("$i:$input_dir:$output_dir")
        fi
    fi
done

echo "待处理任务: ${#TASKS[@]} 个"
echo ""

# 分批执行
BATCH=0
for task in "${TASKS[@]}"; do
    IFS=':' read -r idx input_dir output_dir <<< "$task"

    # 启动后台任务
    echo "[$(date +%H:%M:%S)] 启动 chunk $idx: $input_dir -> $output_dir"
    python run_stage2_only.py --input "$input_dir" --output "$output_dir" &

    # 控制并发数
    while [ $(jobs -r | wc -l) -ge $CONCURRENCY ]; do
        sleep 2
    done

    BATCH=$((BATCH+1))
    if [ $((BATCH % CONCURRENCY)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] 等待当前批次完成..."
        wait
        echo "[$(date +%H:%M:%S)] 批次完成，继续下一批"
        echo ""
    fi
done

# 等待剩余任务
echo "[$(date +%H:%M:%S)] 等待最后批次完成..."
wait

echo ""
echo "=========================================="
echo "✓ Stage 2 全部完成！"
echo "=========================================="

# 合并结果
echo "合并结果..."
mkdir -p ${OUTPUT_PREFIX}_all
for i in $(seq -w 0 29); do
    src_dir="${OUTPUT_PREFIX}_${i}"
    if [ -d "$src_dir" ]; then
        cp -r "$src_dir"/* ${OUTPUT_PREFIX}_all/ 2>/dev/null
    fi
done

echo "✓ 合并完成: ${OUTPUT_PREFIX}_all/"
echo "总数: $(ls ${OUTPUT_PREFIX}_all 2>/dev/null | wc -l) sequences"
