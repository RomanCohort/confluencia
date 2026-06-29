#!/bin/bash
# ============================================================
# CircRNA 3D Pipeline - Quality模式（大规模生成）
# 支持: Fast和Quality两种模式
# ============================================================

set -euo pipefail

# 解析参数
INPUT_FASTA="${1:-circbase_seqs.fa.gz}"
MODE="${2:-quality}"  # quality | fast
POSTFILTER="${3:-}"   # --postfilter 或 空

echo "============================================================"
echo "  CircRNA 3D Pipeline - Quality Mode"
echo "============================================================"
echo ""
echo "输入: $INPUT_FASTA"
echo "模式: $MODE"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# Step 1: 解压数据
echo "[Step 1] 解压输入数据..."
if [[ "$INPUT_FASTA" == *.gz ]]; then
    gunzip -c "$INPUT_FASTA" > input.fasta
else
    cp "$INPUT_FASTA" input.fasta
fi

SEQ_COUNT=$(grep -c "^>" input.fasta)
echo "✓ 序列数: $SEQ_COUNT"

# Step 2: 过滤序列
echo ""
echo "[Step 2] 严格序列过滤..."
python << 'PYEOF'
import sys
import re

filtered = []
total = 0

with open("input.fasta", 'r') as f:
    seq_id = None
    seq = ""
    for line in f:
        if line.startswith('>'):
            if seq_id and seq:
                total += 1
                length = len(seq)
                gc = sum(1 for b in seq if b in 'GC') / length
                if 50 <= length <= 500 and 0.30 <= gc <= 0.70:
                    if all(b in 'ACGUT' for b in seq):
                        max_repeat = max(len(m) for m in re.findall(r'(A+|C+|G+|U+|T+)', seq))
                        if max_repeat < 20:
                            filtered.append((seq_id, seq, 0, length))
            seq_id = line.strip()
            seq = ""
        else:
            seq += line.strip().upper()

if seq_id and seq:
    total += 1
    length = len(seq)
    gc = sum(1 for b in seq if b in 'GC') / length
    if 50 <= length <= 500 and 0.30 <= gc <= 0.70:
        if all(b in 'ACGUT' for b in seq):
            max_repeat = max(len(m) for m in re.findall(r'(A+|C+|G+|U+|T+)', seq))
            if max_repeat < 20:
                filtered.append((seq_id, seq, 0, length))

with open("filtered.fasta", 'w') as f:
    for i, (sid, seq, bsj_start, bsj_end) in enumerate(filtered):
        f.write(f">circ_{i:06d} length={len(seq)} bsj={bsj_start}-{bsj_end}\n{seq}\n")

print(f"总序列数: {total}")
print(f"过滤后: {len(filtered)} ({len(filtered)/total*100:.1f}%)")
PYEOF

FILTERED_COUNT=$(grep -c "^>" filtered.fasta)
echo "✓ 过滤后保留: $FILTERED_COUNT 条"

# Step 3: 配置Pipeline
echo ""
echo "[Step 3] 配置Pipeline..."

if [ "$MODE" == "quality" ]; then
    cat > config_quality_strict.yaml << 'YAML_EOF'
vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"

trrosetta:
  model_path: "./trRosettaRNA2/"
  weights_path: "./weights/params/"
  num_samples: 10
  device: "cuda:0"
  use_gpu: true

cyclize:
  bsj_restraint_k: 3000.0
  bsj_target_distance: 3.5
  max_iterations: 3000
  minimization_tolerance: 3.0

md:
  quality:
    duration_ns: 20.0
    temperature_k: 300
    timestep_fs: 1.0
    snapshot_interval_ps: 20
    bsj_restraint_k: 1500.0
    padding_nm: 1.2
    equilibration_steps: 100000
    production_steps: 10000000

quality:
  min_confidence_threshold: 0.85
  require_all_metrics_pass: true

parallel:
  num_workers: 8
  ray: true
  timeout_per_sequence_s: 1800

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  output_dir: "output_quality_full/"
YAML_EOF
    CONFIG="config_quality_strict.yaml"
else
    cat > config_fast.yaml << 'YAML_EOF'
vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"

trrosetta:
  model_path: "./trRosettaRNA2/"
  num_samples: 5
  device: "cuda:0"
  use_gpu: true

cyclize:
  bsj_restraint_k: 2000.0
  bsj_target_distance: 3.5
  max_iterations: 2000

md:
  fast:
    duration_ns: 2.0
    temperature_k: 300
    timestep_fs: 2.0
    snapshot_interval_ps: 100
    bsj_restraint_k: 500.0
    padding_nm: 0.8
    minimize_only: false

quality:
  min_confidence_threshold: 0.60
  bsj_max_distance_a: 4.0
  require_all_metrics_pass: false

parallel:
  num_workers: 8
  ray: true
  timeout_per_sequence_s: 600

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  output_dir: "output_fast/"
YAML_EOF
    CONFIG="config_fast.yaml"
fi

echo "✓ 配置文件: $CONFIG"

# Step 4: 运行Pipeline
echo ""
echo "[Step 4] 启动Pipeline..."
START_TIME=$(date +%s)

python parallel_worker.py \
    --config $CONFIG \
    --fasta filtered.fasta \
    --num-workers 8 \
    --output output_quality_full \
    --mode $MODE \
    --export-torusfold 2>&1 | tee pipeline_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "============================================================"
echo "  Pipeline完成！"
echo "============================================================"
echo "  耗时: ${ELAPSED_HOURS}小时"
echo "============================================================"