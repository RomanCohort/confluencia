#!/bin/bash
# 完整的并行处理流程：Stage 1 + Stage 2

echo "=========================================="
echo "Parallel Pipeline: Stage 1 + Stage 2"
echo "=========================================="

# Stage 1并行
echo ""
echo "[Stage 1] Splitting FASTA..."
python split_fasta.py \
  --fasta circbase_filtered_5000.fa \
  --num-parts 10 \
  --output fasta_parts

echo ""
echo "[Stage 1] Running 10 parallel processes..."
bash run_stage1_parallel.sh

# Stage 2并行
echo ""
echo "[Stage 2] Running 10 parallel processes..."
bash run_stage2_parallel.sh

echo ""
echo "=========================================="
echo "Stage 1 + Stage 2 COMPLETED!"
echo "=========================================="
echo "Stage 1 output: stage1_output_all/"
echo "Stage 2 output: stage2_output_all/"
echo "=========================================="