#!/bin/bash
# ============================================================
# circRNA环化Pipeline - DGX/云GPU完整版部署脚本
# ============================================================
# 包含全部5个Stage:
#   Stage 1: ViennaRNA (二级结构)
#   Stage 2: RoseTTAFold2NA (3D预测)
#   Stage 3: OpenMM BSJ环化
#   Stage 4: OpenMM MD弛豫
#   Stage 5: 质量过滤
# ============================================================

set -euo pipefail

echo "============================================================"
echo "  circRNA环化Pipeline - 完整版部署"
echo "============================================================"
echo "  日期: $(date)"
echo "  主机: $(hostname)"
echo "============================================================"

# ============================================================
# Step 1: 系统信息检测
# ============================================================
echo ""
echo "[Step 1] 系统信息检测..."

# CPU信息
echo "  CPU:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket" || echo "    (无法获取CPU信息)"

# 内存信息
echo ""
echo "  内存:"
free -h | head -5 || echo "    (无法获取内存信息)"

# GPU信息
echo ""
echo "  GPU:"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "    GPU数量: $GPU_COUNT"
else
    echo "    (未检测到NVIDIA GPU)"
    GPU_COUNT=0
fi

# 硬盘空间
echo ""
echo "  硬盘空间:"
df -h / | head -5

# ============================================================
# Step 2: 安装核心依赖
# ============================================================
echo ""
echo "[Step 2] 安装核心依赖..."

# Python版本检查
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python版本: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11) ]]; then
    echo "  [ERROR] 需要Python 3.9/3.10/3.11"
    echo "    当前版本: $PYTHON_VERSION"
    echo "    安装方法: conda create -n circrna python=3.10"
    exit 1
fi

# 检查conda
if ! command -v conda &>/dev/null; then
    echo "  [ERROR] conda未安装"
    echo "    安装Miniconda:"
    echo "      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "      bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 安装ViennaRNA（Stage 1必需）
echo ""
echo "  安装ViennaRNA..."
if ! python3 -c "import RNA" 2>/dev/null; then
    echo "    conda install -c bioconda viennarna -y"
    conda install -c bioconda viennarna -y
    echo "    [OK] ViennaRNA已安装"
else
    echo "    [OK] ViennaRNA已存在"
fi

# 安装OpenMM（Stage 3-4必需）
echo ""
echo "  安装OpenMM..."
if ! python3 -c "import openmm" 2>/dev/null; then
    echo "    conda install -c conda-forge openmm -y"
    conda install -c conda-forge openmm -y
    echo "    [OK] OpenMM已安装"
else
    echo "    [OK] OpenMM已存在"
fi

# 安装Ray（并行必需）
echo ""
echo "  安装Ray..."
if ! python3 -c "import ray" 2>/dev/null; then
    echo "    pip install ray"
    pip install ray -q
    echo "    [OK] Ray已安装"
else
    echo "    [OK] Ray已存在"
fi

# 安装其他依赖
echo ""
echo "  安装其他依赖..."
pip install pyyaml numpy pandas scipy matplotlib seaborn tqdm -q

# 验证依赖
echo ""
echo "  验证依赖安装:"
python3 -c "import RNA; print('    ViennaRNA:', RNA.__file__)" || echo "    [ERROR] ViennaRNA"
python3 -c "import openmm; print('    OpenMM:', openmm.__version__)" || echo "    [ERROR] OpenMM"
python3 -c "import ray; print('    Ray:', ray.__version__)" || echo "    [ERROR] Ray"
python3 -c "import torch; print('    PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())" || echo "    [WARN] PyTorch未安装"

# ============================================================
# Step 3: 安装RoseTTAFold2NA（Stage 2必需）
# ============================================================
echo ""
echo "[Step 3] 安装RoseTTAFold2NA..."

ROSETTA_DIR="./RoseTTAFold2NA"

if [ ! -d "$ROSETTA_DIR" ]; then
    echo "  克隆RoseTTAFold2NA..."
    git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git $ROSETTA_DIR

    echo ""
    echo "  下载预训练权重..."
    cd $ROSETTA_DIR

    # 下载权重（约5GB）
    if [ ! -f "weights/RoseTTAFold2NA_weights.tar.gz" ]; then
        wget -O weights/RoseTTAFold2NA_weights.tar.gz \
            https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz

        tar -xzf weights/RoseTTAFold2NA_weights.tar.gz -C weights/
    fi

    cd ..

    echo "  [OK] RoseTTAFold2NA已安装"
else
    echo "  [OK] RoseTTAFold2NA已存在"
fi

# ============================================================
# Step 4: 准备Pipeline代码
# ============================================================
echo ""
echo "[Step 4] 准备Pipeline代码..."

PIPELINE_DIR="./circrna_3d_pipeline"

if [ -d "../confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline" ]; then
    echo "  复制Pipeline代码..."
    cp -r ../confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline $PIPELINE_DIR
    echo "  [OK] Pipeline代码已复制"
else
    echo "  [WARN] Pipeline代码未找到，需要手动复制"
    echo "    从IGEM集成方案复制:"
    echo "      cp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline ./"
fi

# 复制配置文件
echo ""
echo "  复制配置文件..."
cp $PIPELINE_DIR/config_quality.yaml ./config.yaml || {
    echo "  [WARN] 配置文件未找到，使用默认配置"
    cat > config.yaml << 'YAML_EOF'
# circRNA 3D Pipeline - 默认配置

vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"
  output_bp_probs: true

rosetta:
  model_path: "./RoseTTAFold2NA/"
  num_samples: 5
  batch_size: 1
  device: "cuda:0"
  max_seq_length: 500

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

  quality:
    duration_ns: 20.0
    temperature_k: 300
    timestep_fs: 1.0
    snapshot_interval_ps: 20
    bsj_restraint_k: 1000.0
    padding_nm: 1.2
    equilibration_steps: 50000
    production_steps: 10000000

quality:
  min_confidence_threshold: 0.80
  bsj_max_distance_a: 3.8
  require_all_metrics_pass: true

parallel:
  num_workers: ${GPU_COUNT}
  ray: true
  timeout_per_sequence_s: 1800

output:
  format: "json"
  save_pdbs: true
  save_energies: true
  save_trajectories: true
YAML_EOF
}

# ============================================================
# Step 5: 准备输入数据
# ============================================================
echo ""
echo "[Step 5] 准备输入数据..."

DATA_DIR="./data"
mkdir -p $DATA_DIR

# 检查用户是否提供FASTA
if [ -f "${1:-}" ]; then
    INPUT_FASTA="$1"
    echo "  使用用户提供的FASTA: $INPUT_FASTA"
else
    echo "  生成测试数据..."

    # 从CircBase下载真实circRNA序列（如果可用）
    if [ -f "../data/circrna/circbase_seqs.fa.gz" ]; then
        echo "    解压CircBase序列..."
        gunzip -c ../data/circrna/circbase_seqs.fa.gz | head -100 > $DATA_DIR/test_100.fasta
        INPUT_FASTA="$DATA_DIR/test_100.fasta"
        echo "    [OK] 使用CircBase前100序列"
    else
        # 生成合成测试序列
        echo "    生成合成测试序列..."
        python3 << 'PYTHON_EOF'
import random
import sys

sequences = []
bases = ['A', 'C', 'G', 'U']

for i in range(10):
    length = random.randint(80, 150)
    seq = ''.join(random.choices(bases, k=length))
    sequences.append((f"circ_test_{i:03d}", seq))

with open("./data/test_10.fasta", 'w') as f:
    for name, seq in sequences:
        f.write(f">{name} length={len(seq)} bsj_start=0 bsj_end={len(seq)}\n")
        f.write(f"{seq}\n")

print("生成10条测试序列")
PYTHON_EOF

        INPUT_FASTA="$DATA_DIR/test_10.fasta"
        echo "    [OK] 生成10条合成序列"
    fi
fi

# 统计序列
SEQ_COUNT=$(grep -c "^>" $INPUT_FASTA 2>/dev/null || echo "0")
echo "  输入序列数: $SEQ_COUNT"

# ============================================================
# Step 6: 运行预过滤（可选）
# ============================================================
echo ""
echo "[Step 6] 预过滤序列..."

if [ -f "$PIPELINE_DIR/prefilter.py" ]; then
    PREFILTERED="$DATA_DIR/prefiltered.fasta"

    python3 $PIPELINE_DIR/prefilter.py \
        --fasta $INPUT_FASTA \
        --output $PREFILTERED \
        --config config.yaml

    FILTERED_COUNT=$(grep -c "^>" $PREFILTERED 2>/dev/null || echo "0")
    echo "  过滤后保留: $FILTERED_COUNT 条序列"

    INPUT_FASTA=$PREFILTERED
else
    echo "  [SKIP] 预过滤脚本未找到"
fi

# ============================================================
# Step 7: 运行主Pipeline
# ============================================================
echo ""
echo "[Step 7] 启动完整Pipeline..."

OUTPUT_DIR="./output_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "  输出目录: $OUTPUT_DIR"
echo "  配置文件: config.yaml"
echo "  GPU数量: ${GPU_COUNT:-1}"
echo "  开始时间: $(date)"
echo ""

START_TIME=$(date +%s)

# 运行Pipeline（根据GPU数量选择并行模式）
if [ "${GPU_COUNT:-0}" -ge 8 ]; then
    echo "  使用DGX Spark模式（8 GPU并行）..."

    python3 $PIPELINE_DIR/parallel_worker.py \
        --config config.yaml \
        --fasta $INPUT_FASTA \
        --num-workers ${GPU_COUNT} \
        --output $OUTPUT_DIR \
        --mode quality \
        --export-torusfold \
        2>&1 | tee $OUTPUT_DIR/pipeline_log.txt

elif [ "${GPU_COUNT:-0}" -ge 1 ]; then
    echo "  使用单GPU模式..."

    python3 $PIPELINE_DIR/pipeline.py \
        --config config.yaml \
        --fasta $INPUT_FASTA \
        --output $OUTPUT_DIR \
        --mode fast \
        2>&1 | tee $OUTPUT_DIR/pipeline_log.txt

else
    echo "  [ERROR] 未检测到GPU"
    echo "    Pipeline需要GPU运行RoseTTAFold2NA"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "============================================================"
echo "  Pipeline完成！"
echo "============================================================"
echo "  总耗时: ${ELAPSED_HOURS}h (${ELAPSED}s)"
echo "  平均: $(echo "scale=1; $ELAPSED / $SEQ_COUNT" | bc)s/序列"
echo "============================================================"

# ============================================================
# Step 8: 检查结果
# ============================================================
echo ""
echo "[Step 8] 检查生成结果..."

if [ -f "$OUTPUT_DIR/dataset_report.json" ]; then
    echo ""
    echo "============================================================"
    echo "  最终质量报告"
    echo "============================================================"

    python3 -c "
import json
with open('$OUTPUT_DIR/dataset_report.json') as f:
    r = json.load(f)

print(f'  总结构数: {r.get(\"total_structures\", \"N/A\")}')
print(f'  成功数: {r.get(\"successful\", \"N/A\")}')
print(f'  平均置信度: {r.get(\"confidence\", {}).get(\"mean\", \"N/A\"):.3f}')
print(f'  平均能量: {r.get(\"energy\", {}).get(\"mean\", \"N/A\"):.1f} kJ/mol')
print(f'  BSJ距离: {r.get(\"bsj_distance\", {}).get(\"mean\", \"N/A\"):.2f} A')

print(f'')
print(f'  质量分布:')
dist = r.get('confidence', {}).get('distribution', {})
for k, v in dist.items():
    print(f'    {k}: {v}')

print(f'')
print(f'  分量评分:')
for k, v in r.get('component_scores', {}).items():
    print(f'    {k}: {v.get(\"mean\", \"N/A\"):.3f}')
"

    echo ""
    echo "  [OK] Pipeline成功完成"
else
    echo "  [WARN] 未找到dataset_report.json"
    echo "    Pipeline可能未完全成功"
fi

# ============================================================
# Step 9: 准备TorusFold训练数据
# ============================================================
echo ""
echo "[Step 9] 准备TorusFold训练数据..."

TORUSFOLD_DIR="./torusfold_training_$(date +%Y%m%d_%H%M%S)"

if [ -d "$OUTPUT_DIR/torusfold_format" ]; then
    cp -r $OUTPUT_DIR/torusfold_format $TORUSFOLD_DIR

    echo "  [OK] TorusFold训练数据已复制"
    echo "    位置: $TORUSFOLD_DIR"
    echo "    文件:"
    ls -lh $TORUSFOLD_DIR/
else
    echo "  [WARN] torusfold_format未生成"
fi

# ============================================================
# Step 10: 后续步骤指南
# ============================================================
echo ""
echo "============================================================"
echo "  后续步骤"
echo "============================================================"
echo ""
echo "  1. 查看质量报告:"
echo "     cat $OUTPUT_DIR/dataset_report.json"
echo ""
echo "  2. 检查PDB文件:"
echo "     ls $OUTPUT_DIR/seq_*/cyclized/*.pdb"
echo ""
echo "  3. 可视化结构:"
echo "     PyMOL: load $OUTPUT_DIR/seq_0/cyclized/cyclized_0.pdb"
echo ""
echo "  4. 训练TorusFold:"
echo "     cd ../confluencia_3_0/core/circrna/torusfold"
echo "     python train_curriculum.py --labels $TORUSFOLD_DIR"
echo ""
echo "  5. 扩展数据集:"
echo "     准备更多FASTA序列并重新运行:"
echo "     ./deploy_full_pipeline.sh new_sequences.fasta"
echo ""
echo "============================================================"
echo "  部署完成！"
echo "============================================================"