#!/bin/bash
# 两阶段处理方案：GPU预处理 + CPU集群处理

# 阶段1：GPU加速trRosettaRNA2预测（所有序列）
echo "阶段1：GPU预处理（trRosettaRNA2 Stage 1-2）"
python circrna_3d_pipeline/run_stages_1_2.py \
  --fasta circbase_filtered_5000.fa \
  --output stage1_stage2_output

# 阶段2：CPU集群处理Stage 3-5
# 分成13个批次，每批10,000条序列
echo "阶段2：CPU集群处理（OpenMM Stage 3-5）"

for i in {0..12}; do
  batch_start=$((i * 10000))
  batch_end=$(( (i+1) * 10000 ))

  if [ $batch_end -gt 130530 ]; then
    batch_end=130530
  fi

  echo "启动批次 $i: 序列 $batch_start - $batch_end"

  # 在后台运行每个批次
  python circrna_3d_pipeline/run_stages_3_5.py \
    --input stage1_stage2_output \
    --output stage3_stage5_output \
    --batch-start $batch_start \
    --batch-end $batch_end &
done

# 等待所有批次完成
wait

echo "所有批次完成！"