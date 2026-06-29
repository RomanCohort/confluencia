# 🚀 AutoDL云GPU - circRNA环化Pipeline运行指南

## ✅ 你已完成
- ✓ 克隆GitHub仓库成功
- ✓ 进入deploy_package目录

## 🔧 解决权限问题

```bash
# 赋予执行权限
chmod +x deploy_full_pipeline.sh

# 验证权限
ls -lh deploy_full_pipeline.sh
# 应显示: -rwxr-xr-x (有x执行权限)

# 运行Pipeline
./deploy_full_pipeline.sh input.fasta
```

---

## 📋 完整运行步骤

### Step 1: 验证依赖

```bash
# 先验证依赖是否齐全
python verify_dependencies.py

# 预期输出:
# [OK] Python: 3.10
# [OK] ViennaRNA
# [OK] OpenMM
# [OK] Ray
# [OK] RoseTTAFold2NA (需要安装)
# [OK] Pipeline代码
```

### Step 2: 安装缺失依赖（如果有）

```bash
# 安装ViennaRNA（Stage 1必需）
conda install -c bioconda viennarna -y

# 安装OpenMM（Stage 3-4必需）
conda install -c conda-forge openmm -y

# 安装Ray（并行必需）
pip install ray

# 安装其他依赖
pip install pyyaml numpy pandas scipy matplotlib tqdm
```

### Step 3: 安装RoseTTAFold2NA（Stage 2必需）

```bash
# 克隆RoseTTAFold2NA
cd /root/autodl-tmp
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
cd RoseTTAFold2NA

# 下载预训练权重（~5GB）
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/

# 返回部署目录
cd /root/autodl-tmp/confluencia/confluencia/deploy_package
```

### Step 4: 准备输入数据

```bash
# 生成测试序列（10条）
python << 'EOF'
import random
bases = ['A', 'C', 'G', 'U']
with open("test.fasta", 'w') as f:
    for i in range(10):
        L = random.randint(80, 150)
        seq = ''.join(random.choices(bases, k=L))
        f.write(f">circ_test_{i:03d} length={L} bsj_start=0 bsj_end={L}\n{seq}\n")
print("生成10条测试序列")
EOF

# 验证序列数
grep -c "^>" test.fasta  # 应输出: 10
```

### Step 5: 运行Pipeline

```bash
# 赋予执行权限
chmod +x deploy_full_pipeline.sh

# 运行（测试模式，10序列）
./deploy_full_pipeline.sh test.fasta

# 或运行快速模式（2ns MD）
# 编辑config.yaml，设置mode: fast
# ./deploy_full_pipeline.sh test.fasta
```

### Step 6: 检查结果

```bash
# 查看质量报告
cat output_*/dataset_report.json

# 查看生成的PDB文件
ls output_*/seq_*/cyclized/*.pdb

# 查看TorusFold训练数据
ls output_*/torusfold_format/
```

---

## 🎯 快速命令汇总（复制粘贴）

```bash
# 一键解决权限问题
chmod +x deploy_full_pipeline.sh verify_dependencies.py

# 验证依赖
python verify_dependencies.py

# 生成测试数据
python << 'EOF'
import random
bases = ['A', 'C', 'G', 'U']
with open("test.fasta", 'w') as f:
    for i in range(10):
        L = random.randint(80, 150)
        seq = ''.join(random.choices(bases, k=L))
        f.write(f">circ_test_{i:03d} length={L} bsj_start=0 bsj_end={L}\n{seq}\n")
EOF

# 运行Pipeline
./deploy_full_pipeline.sh test.fasta
```

---

## ⚠️ 可能遇到的问题

### 问题1: RoseTTAFold2NA未安装

**症状**: verify_dependencies.py显示 `[FAIL] RoseTTAFold2NA`

**解决**:
```bash
cd /root/autodl-tmp
git clone https://github.com/baker-laboratory/RoseTTAFold2NA.git
cd RoseTTAFold2NA
wget https://files.ipd.uw.edu/public/RoseTTAFold2NA/RoseTTAFold2NA_weights.tar.gz
tar -xzf RoseTTAFold2NA_weights.tar.gz -C weights/
cd /root/autodl-tmp/confluencia/confluencia/deploy_package
```

### 问题2: ViennaRNA未安装

**症状**: `ModuleNotFoundError: No module named 'RNA'`

**解决**:
```bash
conda install -c bioconda viennarna -y
```

### 问题3: OpenMM未安装

**症状**: `ModuleNotFoundError: No module named 'openmm'`

**解决**:
```bash
conda install -c conda-forge openmm -y
```

### 问题4: Ray未安装

**症状**: `ModuleNotFoundError: No module named 'ray'`

**解决**:
```bash
pip install ray
```

### 问题5: GPU未检测到

**症状**: `GPU数量: 0`

**解决**:
```bash
# 检查NVIDIA驱动
nvidia-smi

# 如果无输出，联系AutoDL客服
```

---

## 📊 预期输出

成功运行后，你会看到：

```
============================================================
  circRNA环化Pipeline - 完整版部署
============================================================

[Step 1] 检测环境并安装依赖...
  Python版本: 3.10.x
  ✓ ViennaRNA
  ✓ OpenMM
  ✓ Ray

[Step 2] 安装RoseTTAFold2NA...
  ✓ RoseTTAFold2NA已安装

[Step 3] 准备Pipeline代码...
  ✓ Pipeline代码已复制

[Step 4] 准备输入数据...
  输入序列数: 10

[Step 5] 预过滤序列...
  过滤后保留: 9 条序列

[Step 6] 启动完整Pipeline...
  输出目录: output_20260629_230000
  开始时间: Sun Jun 29 23:00:00 CST 2026

  [Stage 1] Predicting secondary structure for seq_0...
  [Stage 2] Predicting 3D structure for seq_0...
  [Stage 3] Cyclizing BSJ for seq_0...
    ✓ BSJ约束已添加 (k=2000.0, r0=3.5 Å)
    ✓ 环化完成
      BSJ距离: 3.62 Å
      能量: -850.2 kJ/mol
  [Stage 4] Running MD relaxation for seq_0...
  [Stage 5] Filtering and scoring for seq_0...

============================================================
  Pipeline完成！
============================================================
  总耗时: 0.25h (900s)
  平均: 100.0s/序列
============================================================

最终质量报告
  总结构数: 45
  平均置信度: 0.870
  平均能量: -850.2 kJ/mol
  BSJ距离: 3.62 Å

  质量分布:
    high: 25
    medium: 15
    low: 5

  ✓ Pipeline成功完成
```

---

## 🎉 成功标志

如果看到以下内容，说明Pipeline运行成功：

1. ✓ `dataset_report.json` 生成
2. ✓ `output_*/seq_*/cyclized/*.pdb` 文件存在
3. ✓ BSJ距离平均值接近3.5 Å
4. ✓ 置信度平均值 ≥ 0.80

---

## 📚 文档链接

- **快速开始**: `cat QUICK_START.md`
- **完整指南**: `cat README_DGX_DEPLOY.md`
- **GitHub仓库**: https://github.com/RomanCohort/confluencia

---

## 💡 下一步

运行成功后：

1. **查看PDB文件**:
   ```bash
   ls output_*/seq_*/cyclized/*.pdb
   ```

2. **下载到本地查看**（可选）:
   ```bash
   # 在本地Windows执行
   scp root@autodl-server:/root/autodl-tmp/confluencia/confluencia/deploy_package/output_*/seq_*/cyclized/*.pdb ./
   ```

3. **扩展到完整数据集**:
   ```bash
   # 准备1000序列的FASTA文件
   ./deploy_full_pipeline.sh input_1k.fasta
   ```

---

**祝运行顺利！如有问题，查看README_DGX_DEPLOY.md**