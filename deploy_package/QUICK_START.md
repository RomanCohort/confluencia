# circRNA环化Pipeline - 云GPU快速部署指南

## 🎯 立即行动清单

### ✅ 已完成（本地环境）
- **ViennaRNA 2.7.2** - 已安装 ✓
- **OpenMM 8.5.2** - 已安装 ✓
- **PyTorch 2.10.0** - 已安装（CPU版）✓

### ⚠️ 需要在云GPU上安装
1. **RoseTTAFold2NA** - Stage 2 3D预测
2. **Ray** - 多GPU并行
3. **Pipeline代码** - 完整5-Stage流程

---

## 🚀 Step 1: 连接云GPU集群

```bash
# 通过SSH连接（以AutoDL为例）
ssh root@your-autodl-server

# 或在云控制台打开Web Terminal
```

---

## 🚀 Step 2: 创建环境并安装依赖

```bash
# 创建Python 3.10环境（推荐conda）
conda create -n circrna python=3.10 -y
conda activate circrna

# 安装核心依赖
conda install -c bioconda viennarna -y
conda install -c conda-forge openmm -y
pip install ray pyyaml numpy pandas scipy matplotlib tqdm

# 验证安装
python -c "import RNA; print('ViennaRNA OK')"
python -c "import openmm; print('OpenMM OK')"
python -c "import ray; print('Ray OK')"
```

---

## 🚀 Step 3: 安装RoseTTAFold2NA

```bash
# 克隆仓库
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
cd RoseTTAFold2NA

# 下载预训练权重（~5GB）
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/

cd ..

# 验证
ls RoseTTAFold2NA/weights/*.pt
```

---

## 🚀 Step 4: 准备Pipeline代码

```bash
# 创建Pipeline目录
mkdir -p circrna_3d_pipeline

# 从IGEM集成方案复制（或从Windows上传）
# 关键文件：
# - pipeline.py
# - stage1_vienna.py
# - stage2_rosetta.py
# - stage3_cyclize.py
# - stage4_md.py
# - stage5_quality.py
# - parallel_worker.py
# - config_quality.yaml
# - prefilter.py

# 如果无法直接复制，可以创建符号链接或上传
cp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/* ./circrna_3d_pipeline/
```

---

## 🚀 Step 5: 准备输入数据

```bash
# 创建数据目录
mkdir -p data

# 选项A: 使用CircBase真实数据（推荐）
gunzip -c data/circrna/circbase_seqs.fa.gz | head -1000 > data/input_1k.fasta

# 选项B: 生成合成测试数据（10条序列）
python << 'EOF'
import random
bases = ['A', 'C', 'G', 'U']
with open("data/test_10.fasta", 'w') as f:
    for i in range(10):
        L = random.randint(80, 150)
        seq = ''.join(random.choices(bases, k=L))
        f.write(f">circ_test_{i:03d} length={L} bsj_start=0 bsj_end={L}\n{seq}\n")
EOF

# 验证序列数
grep -c "^>" data/test_10.fasta  # 应输出10
```

---

## 🚀 Step 6: 运行Pipeline（完整版）

```bash
# 赋予执行权限
chmod +x deploy_full_pipeline.sh

# 运行完整版（全部5个Stage）
./deploy_full_pipeline.sh data/test_10.fasta

# 脚本将自动：
#   1. 检测系统环境
#   2. 验证依赖
#   3. 预过滤序列
#   4. 运行RoseTTAFold2NA 3D预测
#   5. OpenMM BSJ环化
#   6. 20ns MD弛豫
#   7. 质量过滤
#   8. 导出TorusFold格式
```

---

## 🚀 Step 7: 检查结果

```bash
# 查看质量报告
cat output_*/dataset_report.json

# 关键指标：
# - BSJ距离：应接近3.5 Å
# - 置信度：平均值应 ≥ 0.80
# - 成功率：应 > 80%

# 查看生成的PDB文件
ls output_*/seq_*/cyclized/*.pdb

# 可视化（可选）
pymol output_*/seq_0/cyclized/cyclized_0.pdb
```

---

## 🚀 Step 8: 扩展到完整数据集

```bash
# 准备完整FASTA（1000-10000序列）
# ... 准备你的数据 ...

# 运行（8 GPU并行）
./deploy_full_pipeline.sh data/input_1k.fasta

# 预计时间（质量模式，20ns MD）：
# - 1000序列：~8-10小时（8 GPU）
# - 10000序列：~3-4天（8 GPU）
```

---

## 📊 预期输出

```
output_quality/
├── dataset_report.json          # 质量报告
├── torusfold_format/            # TorusFold训练数据
│   ├── coords.npy               # (N, L, 3) 坐标
│   ├── confidences.npy          # (N,) 置信度
│   └── metadata.json
├── seq_0/
│   ├── cyclized/cyclized_0.pdb  # ★ 环化结构
│   ├── md/snapshots/            # MD快照
│   └── quality_report.json
└ ...
```

---

## 🔧 故障排查

### 问题1: RoseTTAFold2NA权重下载慢

```bash
# 解决方案: 本地下载后上传
# Windows:
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
scp RoseTTAFold2NA_weights.tar.gz root@your-server:/path/to/RoseTTAFold2NA/weights/
```

### 问题2: Pipeline代码复制失败

```bash
# 解决方案: 手动上传关键文件
# 从Windows上传到云GPU：
scp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/*.py root@server:/path/to/
scp -r D:/IGEM集成方案/confluencia_3_0/core/circrna/torusfold/circrna_3d_pipeline/*.yaml root@server:/path/to/
```

### 问题3: GPU内存不足

```bash
# 调整config_quality.yaml
# 减少batch_size和序列长度上限

# config_quality.yaml:
rosetta:
  batch_size: 1          # 单样本处理
  max_seq_length: 300    # 缩短上限（原500）
```

---

## 📚 完整文档

- **详细部署指南**: README_DGX_DEPLOY.md
- **验证脚本**: verify_dependencies.py
- **部署脚本**: deploy_full_pipeline.sh

---

## 💡 快速测试（无需RoseTTAFold2NA）

如果RoseTTAFold2NA安装困难，可以先用简化模式测试：

```bash
# 编辑config_quality.yaml，注释掉Stage 2
# 或使用本地测试数据

python simple_pipeline_test.py  # Windows本地简化测试
```

---

## 🎉 下一步

1. **上传到云GPU**: 将整个`deploy_package`目录上传
2. **安装依赖**: 按照Step 1-3安装
3. **运行测试**: 先用10条序列测试
4. **扩展到完整数据集**: 运行1000-10000序列
5. **导入TorusFold训练**: 使用生成的数据

---

**需要帮助?** 查看README_DGX_DEPLOY.md获取完整故障排查指南