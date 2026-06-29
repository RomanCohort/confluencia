#!/bin/bash
# ============================================================
# 从完整CircBase数据准备circRNA序列用于3D结构生成
# ============================================================

echo "============================================================"
echo "  CircBase数据处理 - AutoDL"
echo "============================================================"

# Step 1: 解压CircBase数据
echo "[Step 1] 解压CircBase数据..."
cd /root/autodl-tmp/confluencia/confluencia/deploy_package

if [ -f "circbase_seqs.fa.gz" ]; then
    echo "  解压 circbase_seqs.fa.gz..."
    gunzip -c circbase_seqs.fa.gz > circbase_seqs.fa

    # 统计序列数
    TOTAL_SEQS=$(grep -c "^>" circbase_seqs.fa)
    echo "  ✓ 总序列数: $TOTAL_SEQS"
else
    echo "  ✗ 未找到 circbase_seqs.fa.gz"
    echo "  请上传CircBase数据到当前目录"
    exit 1
fi

# Step 2: 过滤序列（选择合适的长度和BSJ信息）
echo ""
echo "[Step 2] 过滤序列..."

python << 'PYEOF'
import sys

# 过滤标准：
# - 长度：50-500 nt (适合trRosettaRNA2和环化)
# - 有明确的BSJ信息（circRNA的环化位点）
# - GC含量：30-70% (稳定性)

filtered_seqs = []
total_count = 0
filtered_count = 0

with open("circbase_seqs.fa", 'r') as f:
    seq_id = None
    sequence = ""

    for line in f:
        if line.startswith('>'):
            # 处理上一个序列
            if seq_id and sequence:
                total_count += 1

                # 过滤条件
                length = len(sequence)
                gc_count = sum(1 for b in sequence if b in 'GC')
                gc_content = gc_count / length

                # 长度过滤
                if 50 <= length <= 500:
                    # GC含量过滤
                    if 0.30 <= gc_content <= 0.70:
                        # 只保留A/C/G/U
                        if all(b in 'ACGU' for b in sequence):
                            # circRNA的BSJ：首尾相连
                            # 添加BSJ信息到header
                            bsj_start = 0
                            bsj_end = length

                            # 写入过滤后的序列
                            filtered_seqs.append((seq_id, sequence, bsj_start, bsj_end))
                            filtered_count += 1

            # 解析新的header
            seq_id = line.strip()
            sequence = ""
        else:
            sequence += line.strip().upper()

# 处理最后一个序列
if seq_id and sequence:
    total_count += 1
    length = len(sequence)
    gc_count = sum(1 for b in sequence if b in 'GC')
    gc_content = gc_count / length

    if 50 <= length <= 500 and 0.30 <= gc_content <= 0.70:
        if all(b in 'ACGU' for b in sequence):
            filtered_seqs.append((seq_id, sequence, 0, length))
            filtered_count += 1

# 写入过滤后的序列（全部）
print(f"总序列数: {total_count}")
print(f"过滤后保留: {filtered_count}")
print(f"过滤率: {filtered_count/total_count*100:.1f}%")

with open("circbase_filtered_all.fa", 'w') as f:
    for i, (seq_id, seq, bsj_start, bsj_end) in enumerate(filtered_seqs):
        # 简化header，添加BSJ信息
        new_id = f"circ_{i:06d} length={len(seq)} bsj_start={bsj_start} bsj_end={bsj_end}"
        f.write(f"{new_id}\n{seq}\n")

print(f"✓ 已写入 circbase_filtered_all.fa ({filtered_count} 条序列)")

# 统计长度分布
import numpy as np
lengths = [len(s[1]) for s in filtered_seqs]
print(f"长度分布:")
print(f"  最小: {min(lengths)} nt")
print(f"  最大: {max(lengths)} nt")
print(f"  平均: {np.mean(lengths):.1f} nt")
print(f"  中位数: {np.median(lengths):.1f} nt")
PYEOF

# 验证过滤后的数据
FILTERED_COUNT=$(grep -c "^>" circbase_filtered_all.fa 2>/dev/null || echo "0")
echo ""
echo "  ✓ 过滤后序列数: $FILTERED_COUNT"

# Step 3: 准备配置文件（大规模生成）
echo ""
echo "[Step 3] 准备大规模生成配置..."

cat > config_large_scale.yaml << 'YAML_EOF'
# circRNA 3D Pipeline - 大规模生成配置（10000+序列）

vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"
  output_bp_probs: true

trrosetta:
  model_path: "./trRosettaRNA2/"
  weights_path: "./weights/params/"
  num_samples: 5
  device: "cuda:0"
  max_seq_length: 500
  use_msa: false  # 关闭MSA加速
  use_gpu: true

cyclize:
  bsj_restraint_k: 2000.0
  bsj_target_distance: 3.5
  ss_restraint_k: 100.0
  max_iterations: 2000
  minimization_tolerance: 5.0

md:
  forcefield:
    protein: "amber14-all.xml"
    water: "amber14/tip3pfb.xml"

  # Fast模式（快速生成）
  fast:
    duration_ns: 2.0
    temperature_k: 300
    timestep_fs: 2.0
    snapshot_interval_ps: 100
    bsj_restraint_k: 500.0
    padding_nm: 0.8
    minimize_only: false

  # Quality模式（高质量，用于关键验证集）
  quality:
    duration_ns: 10.0  # 稍微缩短以加速
    temperature_k: 300
    timestep_fs: 1.0
    snapshot_interval_ps: 50
    bsj_restraint_k: 1000.0
    padding_nm: 1.0
    minimize_only: false

quality:
  energy_threshold_kjmol: 500.0
  bsj_target_angstrom: 3.5
  bsj_max_distance_a: 4.0
  bp_rmsd_max_a: 1.5
  rmsd_variance_max: 0.3

  min_confidence_threshold: 0.60  # 降低阈值以保留更多结构
  require_all_metrics_pass: false

parallel:
  num_workers: 8  # 8 GPU并行
  ray: true
  timeout_per_sequence_s: 600  # 10分钟超时

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  save_trajectories: false  # 不保存轨迹节省空间
  output_dir: "circbase_output/"
YAML_EOF

echo "  ✓ 配置文件已创建: config_large_scale.yaml"

# Step 4: 显示运行指南
echo ""
echo "============================================================"
echo "  数据准备完成！"
echo "============================================================"
echo ""
echo "输入数据: circbase_filtered_all.fa"
echo "序列数量: $FILTERED_COUNT"
echo ""
echo "运行选项:"
echo ""
echo "  [选项A] 快速模式（推荐大规模生成）"
echo "    ./deploy_full_pipeline.sh circbase_filtered_all.fa --config config_large_scale.yaml --mode fast"
echo "    预计时间: ~$((FILTERED_COUNT * 2 / 60 / 8))小时 (8 GPU)"
echo ""
echo "  [选项B] 质量模式（关键验证集）"
echo "    ./deploy_full_pipeline.sh circbase_filtered_all.fa --config config_large_scale.yaml --mode quality"
echo "    预计时间: ~$((FILTERED_COUNT * 5 / 60 / 8))小时 (8 GPU)"
echo ""
echo "  [选项C] 分批运行（推荐）"
echo "    # 先运行前1000条测试"
echo "    head -2000 circbase_filtered_all.fa > test_1000.fa"
echo "    ./deploy_full_pipeline.sh test_1000.fa --mode fast"
echo ""
echo "============================================================"