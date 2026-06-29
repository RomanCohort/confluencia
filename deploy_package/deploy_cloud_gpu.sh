#!/bin/bash
# ============================================================
# 云GPU集群部署脚本 - circRNA环化3D结构生成Pipeline
# ============================================================
# 适用场景：AutoDL、阿里云GPU、腾讯云GPU等
# 硬件需求：≥1 GPU (推荐8× A100/V100)
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  circRNA 3D环化Pipeline - 云GPU集群部署"
echo "============================================================"

# ============================================================
# Step 1: 环境检测与安装
# ============================================================
echo ""
echo "[Step 1] 检测环境并安装依赖..."

# Python版本检查
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "  Python版本: $PYTHON_VERSION"

# 安装核心依赖（如果缺失）
install_if_missing() {
    python -c "import $1" 2>/dev/null || {
        echo "  安装 $1..."
        pip install $1 -q
    }
}

install_if_missing "openmm"
install_if_missing "yaml"
install_if_missing "numpy"
install_if_missing "pandas"

# ViennaRNA安装（需要conda）
if ! python -c "import RNA" 2>/dev/null; then
    echo ""
    echo "  ⚠ ViennaRNA未安装"
    echo "    安装方法:"
    echo "      conda install -c bioconda viennarna"
    echo "    或使用Linux包管理器:"
    echo "      apt-get install vienna-rna (Ubuntu/Debian)"
    echo "      yum install vienna-rna (CentOS/RHEL)"
    echo ""
    echo "    如无法安装，Pipeline将使用简化模式"
fi

# GPU检测
echo ""
echo "  GPU信息:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
else
    echo "    (未检测到NVIDIA GPU)"
fi

# ============================================================
# Step 2: 准备测试数据
# ============================================================
echo ""
echo "[Step 2] 生成测试数据..."

TEST_DATA_DIR="./test_data"
mkdir -p $TEST_DATA_DIR

# 生成测试FASTA文件（10个circRNA序列）
cat > $TEST_DATA_DIR/test_circrna.fasta << 'EOF'
>circ_test_001 length=80 bsj_start=0 bsj_end=80
ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU
>circ_test_002 length=100 bsj_start=0 bsj_end=100
GCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA
>circ_test_003 length=120 bsj_start=0 bsj_end=120
AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC
>circ_test_004 length=60 bsj_start=0 bsj_end=60
GGGGCCCCAAAAUUUUUUUUAAAACCCCGGGGGGGGCCCCAAAAUUUUAAAACCCCGGGG
>circ_test_005 length=90 bsj_start=0 bsj_end=90
ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU
EOF

echo "  ✓ 已生成测试数据: $TEST_DATA_DIR/test_circrna.fasta"
echo "    序列数: 5"
echo "    长度范围: 60-120 nt"

# ============================================================
# Step 3: 配置Pipeline
# ============================================================
echo ""
echo "[Step 3] 配置Pipeline参数..."

CONFIG_FILE="config_fast.yaml"

# 创建快速配置（适合测试）
cat > $CONFIG_FILE << 'YAML_EOF'
# circRNA 3D Pipeline - 快速配置（适合云GPU测试）

vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"
  output_bp_probs: true

rosetta:
  model_path: "models/rosettafold2na/"
  num_samples: 3
  batch_size: 1
  device: "cuda:0"
  max_seq_length: 500

cyclize:
  bsj_restraint_k: 1500.0
  bsj_target_distance: 3.5
  ss_restraint_k: 50.0
  max_iterations: 1000
  minimization_tolerance: 10.0

md:
  forcefield:
    protein: "amber14-all.xml"
    water: "amber14/tip3pfb.xml"

  fast:
    duration_ns: 2.0
    temperature_k: 300
    timestep_fs: 2.0
    snapshot_interval_ps: 100
    bsj_restraint_k: 500.0
    padding_nm: 0.8
    minimize_only: false

  prefilter:
    minimize_only: true
    max_iterations: 500
    bsj_restraint_k: 1000.0

quality:
  energy_threshold_kjmol: 500.0
  bsj_target_angstrom: 3.5
  bsj_max_distance_a: 4.0
  bp_rmsd_max_a: 1.5
  rmsd_variance_max: 0.3

  min_confidence_threshold: 0.60
  require_all_metrics_pass: false

  confidence_weights:
    energy: 0.25
    rmsd_plateau: 0.25
    bsj: 0.30
    ss_preservation: 0.20

parallel:
  num_workers: 1
  ray: false
  timeout_per_sequence_s: 600

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  save_trajectories: false
  output_dir: "test_output/"
YAML_EOF

echo "  ✓ 配置文件已创建: $CONFIG_FILE"
echo "    模式: fast (2ns MD)"
echo "    预期耗时: ~1分钟/序列"

# ============================================================
# Step 4: 运行Pipeline
# ============================================================
echo ""
echo "[Step 4] 启动Pipeline..."

OUTPUT_DIR="test_output_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "  输出目录: $OUTPUT_DIR"
echo "  开始时间: $(date)"
echo ""

# 运行主脚本（单进程模式，适合测试）
python ../confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/pipeline.py \
    --config $CONFIG_FILE \
    --fasta $TEST_DATA_DIR/test_circrna.fasta \
    --output $OUTPUT_DIR \
    --mode fast \
    --num-workers 1 \
    2>&1 | tee $OUTPUT_DIR/pipeline_log.txt

# ============================================================
# Step 5: 检查结果
# ============================================================
echo ""
echo "[Step 5] 检查生成结果..."

if [ -f "$OUTPUT_DIR/dataset_report.json" ]; then
    echo ""
    echo "============================================================"
    echo "  生成完成！"
    echo "============================================================"
    echo ""
    python -c "
import json
with open('$OUTPUT_DIR/dataset_report.json') as f:
    r = json.load(f)
print(f'  总结构数: {r.get(\"total_structures\", \"N/A\")}')
print(f'  平均置信度: {r.get(\"confidence\", {}).get(\"mean\", \"N/A\")}')
print(f'  平均能量: {r.get(\"energy\", {}).get(\"mean\", \"N/A\")} kJ/mol')
print(f'  BSJ距离: {r.get(\"bsj_distance\", {}).get(\"mean\", \"N/A\")} Å')
"
    echo ""
    echo "  输出文件:"
    ls -lh $OUTPUT_DIR/ | head -20
else
    echo "  ⚠ Pipeline可能未完全成功"
    echo "    请检查日志: $OUTPUT_DIR/pipeline_log.txt"
fi

# ============================================================
# Step 6: 后续步骤
# ============================================================
echo ""
echo "============================================================"
echo "  下一步操作"
echo "============================================================"
echo ""
echo "  1. 扩大测试规模:"
echo "     编辑 $TEST_DATA_DIR/test_circrna.fasta 添加更多序列"
echo ""
echo "  2. 提高质量模式:"
echo "     将 $CONFIG_FILE 中 duration_ns 改为 20.0 (质量模式)"
echo ""
echo "  3. 扩展到完整数据集:"
echo "     准备完整FASTA文件 (如 circbase_10k.fasta)"
echo "     运行: ./deploy_cloud_gpu.sh circbase_10k.fasta"
echo ""
echo "  4. 导入TorusFold训练:"
echo "     cp $OUTPUT_DIR/torusfold_format/ /path/to/torusfold/data/"
echo ""
echo "============================================================"