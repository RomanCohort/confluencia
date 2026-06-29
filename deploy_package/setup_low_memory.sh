#!/bin/bash
# ============================================================
# CircRNA Pipeline - 低显存优化配置
# ============================================================

echo "============================================================"
echo "  CircRNA Pipeline - 显存优化配置"
echo "============================================================"
echo ""

cd /root/autodl-tmp/confluencia/confluencia/deploy_package

# Step 1: 检查当前GPU显存
echo "[Step 1] 检查GPU显存..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

# Step 2: 根据显存选择配置
read -p "你的GPU显存大小 (例如: 12): " MEMORY_SIZE

if [ "$MEMORY_SIZE" -ge 24 ]; then
    echo ""
    echo "✓ 高显存模式 (≥24GB)"
    echo "  使用Quality模式 + 8 GPU并行"
    CONFIG="quality"
    WORKERS=8
elif [ "$MEMORY_SIZE" -ge 16 ]; then
    echo ""
    echo "✓ 中显存模式 (16-23GB)"
    echo "  使用Fast模式 + 4 GPU并行"
    CONFIG="fast"
    WORKERS=4
elif [ "$MEMORY_SIZE" -ge 8 ]; then
    echo ""
    echo "⚠ 低显存模式 (8-15GB)"
    echo "  使用Fast模式 + 2 GPU并行"
    echo "  每序列时间约增加2倍"
    CONFIG="fast"
    WORKERS=2
else
    echo ""
    echo "✗ 显存不足！无法运行此Pipeline"
    echo ""
    echo "建议："
    echo "  1. 升级到至少8GB显存的GPU"
    echo "  2. 或减少序列数测试"
    exit 1
fi

# Step 3: 创建配置文件
echo ""
echo "[Step 3] 创建配置文件..."

cat > config_low_mem.yaml << 'YAML_EOF'
vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"

trrosetta:
  model_path: "./trRosettaRNA2/"
  num_samples: 3  # 减少采样
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
  num_workers: ${WORKERS}  # 根据显存调整
  ray: true
  timeout_per_sequence_s: 600

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  save_trajectories: false
  output_dir: "output_low_mem/"
YAML_EOF

echo "  ✓ 配置文件已创建: config_low_mem.yaml"
echo "  并行度: $WORKERS GPU"
echo "  模式: $CONFIG"

# Step 4: 显示运行命令
echo ""
echo "============================================================"
echo "  运行命令"
echo "============================================================"
echo ""
echo "./run_quality_pipeline.sh circbase_seqs.fa.gz $CONFIG"
echo ""
echo "注意："
echo "  - 如果显存不足，会OOM崩溃"
echo "  - 建议先运行小批量测试"
echo "  - 可随时修改config_low_mem.yaml调整参数"
echo "============================================================"