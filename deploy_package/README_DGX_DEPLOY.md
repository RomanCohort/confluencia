# ============================================================
# circRNA环化Pipeline - DGX Spark完整部署指南
# ============================================================
# 适用环境: DGX Spark / AutoDL / 阿里云GPU / 腾讯云GPU
# 硬件要求: ≥1 GPU (推荐8× A100/H100)
# ============================================================

# ============================================================
# 一、环境准备（首次部署）
# ============================================================

## 1.1 创建Python环境（推荐conda）

```bash
# 创建Python 3.10环境
conda create -n circrna python=3.10 -y
conda activate circrna
```

## 1.2 安装Stage 1依赖 - ViennaRNA

```bash
# ViennaRNA用于二级结构预测（必需）
conda install -c bioconda viennarna -y

# 验证安装
python -c "import RNA; print('ViennaRNA OK:', RNA.__file__)"
```

## 1.3 安装Stage 2依赖 - RoseTTAFold2NA

```bash
# 克隆仓库
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
cd RoseTTAFold2NA

# 下载预训练权重（~5GB）
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/

# 安装PyTorch依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd ..
```

## 1.4 安装Stage 3-4依赖 - OpenMM

```bash
# OpenMM用于BSJ环化和MD弛豫（必需）
conda install -c conda-forge openmm -y

# 验证安装
python -c "import openmm; print('OpenMM OK:', openmm.__version__)"
```

## 1.5 安装并行依赖 - Ray

```bash
# Ray用于多GPU并行（DGX Spark必需）
pip install ray

# 验证安装
python -c "import ray; print('Ray OK:', ray.__version__)"
```

## 1.6 安装其他依赖

```bash
pip install pyyaml numpy pandas scipy matplotlib seaborn tqdm
```

## 1.7 验证完整依赖

```bash
# 运行验证脚本
python verify_dependencies.py

# 预期输出: [SUCCESS] 所有关键依赖已就绪！
```

# ============================================================
# 二、准备Pipeline代码
# ============================================================

## 2.1 复制Pipeline代码

```bash
# 从IGEM集成方案复制完整Pipeline
cp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline ./circrna_3d_pipeline

# 检查关键文件
ls circrna_3d_pipeline/
# 预期文件:
#   pipeline.py               (主Pipeline)
#   stage1_vienna.py          (二级结构)
#   stage2_rosetta.py         (3D预测)
#   stage3_cyclize.py         (BSJ环化)
#   stage4_md.py              (MD弛豫)
#   stage5_quality.py         (质量过滤)
#   parallel_worker.py        (Ray并行)
#   config_quality.yaml       (高质量配置)
#   prefilter.py              (预过滤)
```

## 2.2 配置文件说明

**config_quality.yaml关键参数**:

```yaml
# Stage 3环化参数
cyclize:
  bsj_restraint_k: 2000.0      # BSJ约束强度
  bsj_target_distance: 3.5     # 理想BSJ距离（Å）
  ss_restraint_k: 100.0        # 二级结构保持
  max_iterations: 2000         # 最小化迭代

# Stage 4 MD参数（质量模式）
md:
  quality:
    duration_ns: 20.0          # 20ns生产相
    temperature_k: 300         # 温度
    timestep_fs: 1.0           # 时间步长
    bsj_restraint_k: 1000.0    # 生产相BSJ约束

# Stage 5质量阈值
quality:
  min_confidence_threshold: 0.80  # 置信度阈值
  bsj_max_distance_a: 3.8         # BSJ距离上限
```

# ============================================================
# 三、准备输入数据
# ============================================================

## 3.1 输入FASTA格式

```fasta
>circ_001 length=120 bsj_start=0 bsj_end=120
ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU...
>circ_002 length=85 bsj_start=0 bsj_end=85
GCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA...
```

**要求**:
- 每条序列需标注bsj_start和bsj_end（环化位点）
- 序列长度: 30-500 nt（推荐80-150）
- GC含量: 30-70%

## 3.2 数据来源

**选项A: 使用CircBase真实数据**

```bash
# 解压CircBase序列（140K circRNA）
gunzip -c data/circrna/circbase_seqs.fa.gz | head -1000 > input_1k.fasta
```

**选项B: 使用合成测试数据**

```bash
# 生成10条测试序列
python << 'EOF'
import random
bases = ['A', 'C', 'G', 'U']
with open("test.fasta", 'w') as f:
    for i in range(10):
        L = random.randint(80, 150)
        seq = ''.join(random.choices(bases, k=L))
        f.write(f">circ_test_{i:03d} length={L} bsj_start=0 bsj_end={L}\n{seq}\n")
EOF
```

# ============================================================
# 四、运行Pipeline（完整版）
# ============================================================

## 4.1 DGX Spark模式（8 GPU并行）

```bash
# 使用完整部署脚本
chmod +x deploy_full_pipeline.sh
./deploy_full_pipeline.sh input.fasta

# 脚本将自动:
#   1. 检测系统环境
#   2. 验证依赖安装
#   3. 运行预过滤
#   4. 启动8 GPU并行Pipeline
#   5. 生成质量报告
#   6. 导出TorusFold格式
```

## 4.2 手动运行（分步控制）

```bash
# Step 1: 预过滤（可选）
python circrna_3d_pipeline/prefilter.py \
    --fasta input.fasta \
    --output prefiltered.fasta \
    --config config_quality.yaml

# Step 2: 运行主Pipeline（质量模式）
python circrna_3d_pipeline/parallel_worker.py \
    --config config_quality.yaml \
    --fasta prefiltered.fasta \
    --num-workers 8 \
    --output output_quality \
    --mode quality \
    --export-torusfold

# Step 3: 检查结果
cat output_quality/dataset_report.json
```

## 4.3 快速模式（测试用）

```bash
# 2ns MD，约1分钟/序列
python circrna_3d_pipeline/parallel_worker.py \
    --config config.yaml \
    --fasta test.fasta \
    --num-workers 1 \
    --output output_fast \
    --mode fast
```

# ============================================================
# 五、输出结果解读
# ============================================================

## 5.1 输出目录结构

```
output_quality/
├── dataset_report.json          # 总体质量报告
├── torusfold_format/            # TorusFold训练数据
│   ├── coords.npy               # (N, L, 3) 坐标
│   ├── confidences.npy          # (N,) 置信度
│   └── metadata.json            # 元信息
├── seq_0/                       # 单序列详情
│   ├── stage1_vienna.json       # 二级结构
│   ├── linear/                  # 线性3D预测
│   │   └── sample_0.pdb
│   ├── cyclized/                # 环化结构
│   │   └── cyclized_0.pdb       # ★ 关键输出
│   ├── md/                      # MD轨迹
│   │   ├── snapshots/
│   │   └── trajectory.dcd
│   └── quality_report.json      # 质量评估
├── seq_1/
└ ...
```

## 5.2 dataset_report.json解读

```json
{
  "total_structures": 500,
  "successful": 420,
  "confidence": {
    "mean": 0.87,
    "std": 0.05,
    "distribution": {
      "high": 250,      // ≥ 0.85
      "medium": 120,    // 0.5-0.85
      "low": 50         // < 0.5
    }
  },
  "bsj_distance": {
    "mean": 3.62,       // ★ 理想值: 3.5 Å
    "std": 0.15,
    "within_threshold": 400  // < 3.8 Å
  },
  "energy": {
    "mean": -850.2      // 热力学稳定性
  },
  "component_scores": {
    "bsj": {"mean": 0.91},      // 环化质量
    "rmsd_plateau": {"mean": 0.85}, // 收敛性
    "ss_preservation": {"mean": 0.79} // 二级结构保持
  }
}
```

**关键指标**:
- **BSJ距离**: 平均值应接近3.5 Å（磷酸二酯键长度）
- **置信度**: ≥ 0.80的结构占比应 > 50%
- **成功率**: successful / total应 > 80%

## 5.3 质量分布评估

| 置信度范围 | TorusFold训练权重 | 处理建议 |
|------------|------------------|---------|
| ≥ 0.85 (High) | 2.0 | 直接使用 |
| 0.5-0.85 (Medium) | 1.0 | 可用，建议补充验证 |
| < 0.5 (Low) | 0.1 | 建议重新生成 |

# ============================================================
# 六、性能与时间估算
# ============================================================

| 模式 | MD时长 | 每序列耗时 | 10K序列总时间（8 GPU） |
|------|--------|-----------|----------------------|
| **prefilter** | 仅最小化 | ~15秒 | ~3小时 |
| **fast** | 2ns | ~1分钟 | ~42小时 |
| **quality** | 20ns | ~5-15分钟 | **~8.7天** |
| **ultra_quality** | 50ns | ~25分钟 | ~21天 |

**推荐策略**:
1. 先用prefilter快速筛选（过滤不稳定序列）
2. 对通过筛选的序列运行quality模式
3. 对关键验证集少量样本运行ultra_quality

# ============================================================
# 七、TorusFold训练集成
# ============================================================

## 7.1 导入训练数据

```bash
# 复制TorusFold格式数据
cp -r output_quality/torusfold_format /path/to/torusfold/data/circrna_generated/

# 训练TorusFold
cd confluencia_3_0/core/circrna/torusfold
python train_curriculum.py \
    --labels data/circrna_generated \
    --epochs 100 \
    --batch-size 32
```

## 7.2 数据格式转换

TorusFold训练数据格式:
```
coords.npy:      (N, L, 3)  # C3'原子坐标
confidences.npy: (N,)       # 置信度分数（作为训练权重）
metadata.json:   {sequences, lengths, bsj_positions}
```

# ============================================================
# 八、常见问题与解决
# ============================================================

## Q1: ViennaRNA安装失败

```bash
# 解决方案: 使用Linux包管理器
apt-get install vienna-rna  # Ubuntu/Debian
yum install vienna-rna      # CentOS/RHEL

# 或使用bioconda（推荐）
conda install -c bioconda viennarna
```

## Q2: RoseTTAFold2NA权重下载慢

```bash
# 解决方案: 使用国内镜像或离线下载
# 离线下载后上传到服务器:
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
# 或使用镜像源（如有）
```

## Q3: OpenMM力场模板错误

```bash
# 错误: No template found for residue
# 解决方案: 确保PDB文件包含完整RNA原子

# 检查PDB质量:
python -c "
from openmm.app import PDBFile
pdb = PDBFile('test.pdb')
print('Topology:', pdb.topology)
for res in pdb.topology.residues():
    print(f'Residue {res.id}: {res.name}, atoms={list(res.atoms())}')
"
```

## Q4: BSJ环化距离过大

```yaml
# 调整config_quality.yaml
cyclize:
  bsj_restraint_k: 3000.0      # 增强约束（原2000）
  max_iterations: 5000         # 更多迭代（原2000）
```

## Q5: GPU内存不足

```bash
# 检查GPU内存
nvidia-smi

# 解决方案: 减少batch_size
# config.yaml:
rosetta:
  batch_size: 1          # 单样本处理
  max_seq_length: 300    # 缩短序列上限
```

## Q6: Ray初始化失败

```bash
# 错误: Ray failed to start
# 解决方案: 检查GPU配置

# 测试Ray GPU分配
python -c "
import ray
ray.init(num_gpus=8)
print('Available GPUs:', ray.available_resources())
ray.shutdown()
"
```

# ============================================================
# 九、完整部署检查清单
# ============================================================

```bash
# 1. 系统环境
conda activate circrna           # ✓
python --version                 # ✓ Python 3.10
nvidia-smi                       # ✓ GPU可用

# 2. 核心依赖
python -c "import RNA"           # ✓ ViennaRNA
python -c "import openmm"        # ✓ OpenMM
python -c "import ray"           # ✓ Ray
python -c "import torch"         # ✓ PyTorch CUDA

# 3. RoseTTAFold2NA
ls RoseTTAFold2NA/run_infer.py   # ✓
ls RoseTTAFold2NA/weights/*.pt   # ✓ 权重文件

# 4. Pipeline代码
ls circrna_3d_pipeline/pipeline.py          # ✓
ls circrna_3d_pipeline/config_quality.yaml  # ✓

# 5. 输入数据
grep -c "^>" input.fasta         # ✓ 序列数

# 6. 运行Pipeline
./deploy_full_pipeline.sh input.fasta  # ✓

# 7. 检查输出
cat output_quality/dataset_report.json  # ✓
ls output_quality/torusfold_format/     # ✓
```

# ============================================================
# 十、联系与支持
# ============================================================

**Pipeline位置**:
- Windows本地: D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/
- DGX Spark: /path/to/circrna_3d_pipeline/

**关键脚本**:
- 部署脚本: deploy_full_pipeline.sh
- 验证脚本: verify_dependencies.py
- 主Pipeline: pipeline.py
- 并行Worker: parallel_worker.py
- 配置文件: config_quality.yaml

**参考文献**:
- RoseTTAFold2NA: https://github.com/baker-laboratory/RoseTTAFold2NA
- ViennaRNA: http://www.tbi.univie.ac.at/~ronny/RNA/
- OpenMM: http://openmm.org/

**论文引用**:
- TorusFold: manuscripts/torusfold_paper/torusfold.tex
- Confluencia: manuscripts/confluencia_3.0_COMPLETE_LATEX.tex

# ============================================================