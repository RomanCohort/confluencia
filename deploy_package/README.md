# 🚀 circRNA环化Pipeline - 云GPU部署包

## 📦 部署包内容

```
deploy_package.tar.gz (20KB) - 完整部署脚本
├── deploy_full_pipeline.sh      # 全自动部署脚本（全部5个Stage）
├── verify_dependencies.py       # 依赖验证脚本
├── README_DGX_DEPLOY.md         # 完整部署指南（详细）
├── QUICK_START.md               # 快速开始指南（推荐阅读）
└── simple_pipeline_test.py      # 简化测试脚本（本地用）

pipeline_code.tar.gz (37KB) - Pipeline核心代码
├── pipeline.py                  # 主Pipeline（全部5个Stage）
├── stage1_vienna.py             # Stage 1: 二级结构预测
├── stage2_rosetta.py            # Stage 2: RoseTTAFold2NA 3D预测
├── stage3_cyclize.py            # Stage 3: BSJ环化（关键）
├── stage4_md.py                 # Stage 4: MD弛豫（20ns）
├── stage5_quality.py            # Stage 5: 质量过滤
├── parallel_worker.py           # Ray并行worker
├── prefilter.py                 # 预过滤脚本
└── config_quality.yaml          # 高质量配置（20ns MD）
```

---

## 🎯 三种部署方式

### 方式A: 快速部署（推荐）

**适用场景**: AutoDL、阿里云GPU、腾讯云GPU等

```bash
# 1. 上传部署包到云GPU（在本地Windows执行）
scp deploy_package.tar.gz pipeline_code.tar.gz root@your-server:/root/

# 2. 连接到云GPU
ssh root@your-server

# 3. 解压并运行
cd /root
tar -xzf deploy_package.tar.gz
tar -xzf pipeline_code.tar.gz

# 4. 验证依赖
python deploy_package/verify_dependencies.py

# 5. 运行Pipeline
cd deploy_package
./deploy_full_pipeline.sh input.fasta
```

---

### 方式B: 本地测试（简化版）

**适用场景**: Windows本地快速测试（无GPU）

```bash
# 1. 安装核心依赖
pip install openmm pyyaml numpy pandas

# 2. 运行简化测试
cd D:/IGEM集成方案/deploy_package
python simple_pipeline_test.py

# 输出: test_output/*.pdb（环化结构）
```

---

### 方式C: DGX Spark集群（完整版）

**适用场景**: DGX Spark (8× A100/H100)

```bash
# 1. 上传到DGX Spark
scp -r deploy_package pipeline_code.tar.gz user@dgx-spark:/workspace/

# 2. 连接DGX Spark
ssh user@dgx-spark

# 3. 准备环境
conda create -n circrna python=3.10 -y
conda activate circrna

# 4. 安装全部依赖
conda install -c bioconda viennarna -y
conda install -c conda-forge openmm -y
pip install ray pyyaml numpy pandas

# 5. 安装RoseTTAFold2NA
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
cd RoseTTAFold2NA
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/
cd ..

# 6. 运行完整Pipeline（8 GPU并行）
cd /workspace/deploy_package
./deploy_full_pipeline.sh input_10k.fasta
```

---

## ⏱️ 时间估算

| 模式 | MD时长 | 每序列耗时 | 1000序列（8 GPU） |
|------|--------|-----------|-----------------|
| **fast** | 2ns | ~1分钟 | ~2小时 |
| **quality** | 20ns | ~5-15分钟 | **~8-10小时** |
| **ultra** | 50ns | ~25分钟 | ~2天 |

---

## 📋 关键文件说明

### deploy_full_pipeline.sh
**完整版部署脚本**，包含：
- ✅ 自动检测系统环境
- ✅ 验证所有依赖
- ✅ 预过滤序列
- ✅ 运行RoseTTAFold2NA 3D预测
- ✅ OpenMM BSJ环化
- ✅ 20ns MD弛豫
- ✅ 质量过滤
- ✅ 导出TorusFold训练格式

### config_quality.yaml
**高质量配置**，关键参数：
```yaml
cyclize:
  bsj_restraint_k: 2000.0      # BSJ约束强度
  bsj_target_distance: 3.5     # 理想BSJ距离（Å）

md:
  quality:
    duration_ns: 20.0          # 20ns生产相
    temperature_k: 300         # 温度

quality:
  min_confidence_threshold: 0.80  # 置信度阈值
  bsj_max_distance_a: 3.8         # BSJ距离上限
```

---

## 🔍 验证检查清单

### Step 1: 依赖验证
```bash
python verify_dependencies.py

# 预期输出:
# [OK] Python: 3.10
# [OK] ViennaRNA: 2.7.x
# [OK] OpenMM: 8.x
# [OK] Ray: 2.x
# [OK] RoseTTAFold2NA
# [OK] Pipeline代码
# [SUCCESS] 所有关键依赖已就绪！
```

### Step 2: 测试数据运行
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

# 运行测试
./deploy_full_pipeline.sh test.fasta

# 检查结果
cat output_*/dataset_report.json
```

### Step 3: 质量检查
```bash
# 检查BSJ距离（应接近3.5 Å）
# 检查置信度（应 ≥ 0.80）
# 检查成功率（应 > 80%）

python -c "
import json
with open('output_*/dataset_report.json') as f:
    r = json.load(f)
    print(f'BSJ距离: {r[\"bsj_distance\"][\"mean\"]:.2f} Å')
    print(f'置信度: {r[\"confidence\"][\"mean\"]:.3f}')
    print(f'成功率: {r[\"successful\"]/r[\"total_structures\"]*100:.1f}%')
"
```

---

## 🎉 预期输出

### 成功标志
```
output_quality/
├── dataset_report.json          # ✅ 质量报告
├── torusfold_format/            # ✅ 训练数据
│   ├── coords.npy               # (N, L, 3)
│   ├── confidences.npy          # (N,)
│   └── metadata.json
├── seq_0/cyclized/*.pdb         # ✅ 环化结构
├── seq_1/
└ ...
```

### 质量报告示例
```json
{
  "total_structures": 500,
  "successful": 420,
  "confidence": {"mean": 0.87},
  "bsj_distance": {"mean": 3.62, "within_threshold": 400},
  "energy": {"mean": -850.2}
}
```

---

## 📚 文档索引

1. **QUICK_START.md** - 快速开始（推荐先读）
2. **README_DGX_DEPLOY.md** - 完整部署指南（详细）
3. **verify_dependencies.py** - 依赖验证脚本
4. **deploy_full_pipeline.sh** - 主部署脚本

---

## 🛠️ 故障排查

### 问题: RoseTTAFold2NA安装失败
```bash
# 解决方案: 使用国内镜像
# 或本地下载后上传
```

### 问题: GPU内存不足
```bash
# 解决方案: 调整config
# batch_size: 1
# max_seq_length: 300
```

### 问题: BSJ距离过大
```bash
# 解决方案: 增强约束
# bsj_restraint_k: 3000.0
# max_iterations: 5000
```

---

## 💡 优化建议

1. **先运行预过滤**（prefilter模式）
2. **对通过筛选的序列运行quality模式**
3. **关键验证集使用ultra_quality**

---

## 📞 下一步

1. ✅ 阅读QUICK_START.md
2. ✅ 上传deploy_package.tar.gz到云GPU
3. ✅ 运行verify_dependencies.py
4. ✅ 准备输入FASTA
5. ✅ 运行./deploy_full_pipeline.sh

---

**完整Pipeline代码位置**:
- Windows: D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/
- 云GPU: /workspace/circrna_3d_pipeline/

**已打包文件**:
- deploy_package.tar.gz (20KB)
- pipeline_code.tar.gz (37KB)

🎉 部署包已准备完毕，可以上传到云GPU开始生成数据！