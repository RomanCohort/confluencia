#!/bin/bash
# ============================================================
# AutoDL云GPU - circRNA环化Pipeline快速修复版
# 解决RoseTTAFold2NA克隆失败问题
# ============================================================

echo "============================================================"
echo "  circRNA环化Pipeline - 快速修复版"
echo "============================================================"
echo ""

# 检查当前目录
cd /root/autodl-tmp/confluencia/confluencia/deploy_package
echo "当前目录: $(pwd)"
echo ""

# 方案A: 删除有问题的RoseTTAFold2NA目录并重新克隆（无凭证）
echo "[方案A] 重新克隆RoseTTAFold2NA（不使用凭证）..."
if [ -d "./RoseTTAFold2NA" ]; then
    rm -rf ./RoseTTAFold2NA
fi

# 清除可能存在的git凭证缓存
git config --global --unset credential.helper
git config --global --unset url."https://RomanCohort@github.com/".insteadOf

# 直接克隆（不带任何凭证）
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git

if [ ! -d "./RoseTTAFold2NA/run_infer.py" ]; then
    echo "[ERROR] RoseTTAFold2NA克隆失败或文件缺失"
    echo ""
    echo "=== 方案B: 跳过RoseTTAFold2NA，使用简化模式 ==="
    echo ""
    echo "修改配置文件，注释掉Stage 2（RoseTTAFold2NA）..."
    cat > config.yaml << 'YAML_EOF'
# circRNA 3D Pipeline - 简化配置（跳过RoseTTAFold2NA）
# 注意：此配置仅用于测试OpenMM环化功能
# 完整版需要安装RoseTTAFold2NA才能进行3D预测

vienna:
  max_bp_span: -1
  temperature: 37.0
  model: "rna"
  output_bp_probs: true

# rosetta:  # 注释掉RoseTTAFold2NA
#   model_path: "models/rosettafold2na/"
#   num_samples: 5
#   batch_size: 1
#   device: "cuda:0"
#   max_seq_length: 500

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

  fast:
    duration_ns: 2.0
    temperature_k: 300
    timestep_fs: 2.0
    snapshot_interval_ps: 100
    bsj_restraint_k: 500.0
    padding_nm: 0.8
    minimize_only: false

quality:
  energy_threshold_kjmol: 500.0
  bsj_target_angstrom: 3.5
  bsj_max_distance_a: 4.0
  bp_rmsd_max_a: 1.5
  rmsd_variance_max: 0.3

  min_confidence_threshold: 0.60
  require_all_metrics_pass: false

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

    echo "已创建config.yaml（简化模式）"
    echo ""
    echo "继续执行以下步骤："
    echo ""
else
    echo "[OK] RoseTTAFold2NA克隆成功"
    echo "继续执行后续步骤..."
    echo ""
fi

# 生成测试数据
echo "[Step 1] 生成测试数据..."
python << 'PYEOF'
import random
bases = ['A', 'C', 'G', 'U']
with open("test.fasta", 'w') as f:
    for i in range(10):
        L = random.randint(80, 150)
        seq = ''.join(random.choices(bases, k=L))
        f.write(f">circ_test_{i:03d} length={L} bsj_start=0 bsj_end={L}\n{seq}\n")
print("生成10条测试序列到 test.fasta")
PYEOF

# 验证序列数
SEQ_COUNT=$(grep -c "^>" test.fasta 2>/dev/null || echo "0")
echo "测试序列数: $SEQ_COUNT"
echo ""

# 显示下一步命令
echo "============================================================"
echo "  下一步操作"
echo "============================================================"
echo ""
echo "由于RoseTTAFold2NA克隆失败，现在有两种选择："
echo ""
echo "选项1: 等待手动修复后继续（推荐）"
echo "  1. 在AutoDL终端手动执行："
echo "     git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git"
echo "     cd RoseTTAFold2NA"
echo "     wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz"
echo "     tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/"
echo "     cd /root/autodl-tmp/confluencia/confluencia/deploy_package"
echo "     ./deploy_full_pipeline.sh test.fasta"
echo ""
echo "选项2: 使用简化模式测试OpenMM环化功能"
echo "  1. 运行简化测试脚本："
echo "     python simple_pipeline_test.py"
echo ""
echo "  2. 查看生成的PDB文件："
echo "     ls test_output/*.pdb"
echo ""
echo "注意：简化模式无法进行3D预测，只能测试BSJ环化"
echo "============================================================"