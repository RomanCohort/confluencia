# TorusFold 代理实验运行说明

## 实验目的
验证 Torus Positional Encoding (TPE) 相比标准 Positional Encoding 在预测 BSJ（back-splice junction）附近配对概率上的性能优势。

## 运行实验

### 1. 安装依赖

**必需：**
```bash
pip install torch numpy matplotlib tqdm scipy
```

**可选（推荐用于生成更真实的伪标签）：**
```bash
conda install -c bioconda viennarna
pip install viennarna  # 或使用 ViennaRNA 的 Python bindings
```

### 2. 运行命令

**快速测试（20个序列，50 epochs）：**
```bash
python torusfold_proxy_experiment.py \
    --n-sequences 20 \
    --n-epochs 50 \
    --n-seeds 2 \
    --output-dir results
```

**完整实验（50个序列，100 epochs，3次重复）：**
```bash
python torusfold_proxy_experiment.py \
    --n-sequences 50 \
    --n-epochs 100 \
    --n-seeds 3 \
    --output-dir results
```

**使用 ViennaRNA 生成真实配对概率：**
```bash
python torusfold_proxy_experiment.py \
    --n-sequences 50 \
    --n-epochs 100 \
    --n-seeds 3 \
    --viennarna
```

## 预期输出

### 1. 控制台输出

实验会实时显示：
- 设备信息（CPU/GCUDA）
- 生成的序列长度范围
- TPE 周期性验证结果
- 训练进度（每个 epoch 的 loss）
- 每个种子训练后的 MSE 值
- 统计检验结果（paired t-test）
- 与 Manuscript TBD 值对应的占位符

最关键的输出：
```
BSJ Region (±20nt from BSJ):
  Standard PE MSE: 0.0XXXX ± 0.0XXX
  TPE MSE:         0.0XXXX ± 0.0XXX
  ΔMSE:            +XX.X%
  Paired t-test:   t=X.XXX, p=0.XXXX
  Significant:     Yes / No
```

### 2. 输出文件

- `results/proxy_experiment_results.json`: 结果摘要
  - `bsj_region.mse`: BSJ区域平均MSE
  - `bsj_region.delta_percent`: TPE相对标准PE的改进百分比
  - `bsj_region.p_value`: 统计显著性p值
  - `config`: 实验配置参数

- `results/proxy_experiment_comparison.png`: 三张图的对比
  - 每个序列的MSE散点图（BSJ vs 全序列）
  - BSJ区域和全序列的平均MSE柱状图

- `results/proxy_experiment_comparison.pdf`: 同上，PDF格式

### 3. Manuscript 占位符值

实验结束后，控制台会打印 Manuscript 需要的 TBD 值：

```
MANUSCRIPT TBD VALUES (copy to paper):
Abstract:  TPE reduces BSJ region error by +XX%
           (MSE X.XXX vs X.XXX, p=0.XX)
```

将 `+XX%` 和 p 值复制到 manuscripts 文件的相应位置。

## 实验原理详解

### 数据准备

1. **序列生成**：生成50条circRNA序列，跨越长度100-500 nt
   - GC含量：30-70%
   - 有些序列添加Alu-like倒置重复结构

2. **伪标签生成**：
   - **方法1（默认）**：合成配对概率
     - BSJ附近区域的配对概率较高（模拟环形拓扑）
     - 添加stem-loop结构
   - **方法2（ViennaRNA）**：使用 `RNA.fold_compound(..., md.circ=True)` 生成真实配对概率

### 模型训练

- **标准PE Model**：使用 Vaswani (2017) 的正弦位置编码
- **TPE Model**：使用 Torus Positional Encoding
  - **周期性保证**：TPE(i) = TPE(i+L) 数学成立
  - **验证结果**：max |TPE(i) - TPE(i+L)| < 1e-6

### 评估指标

- **MSE (Mean Squared Error)**：配对概率预测的平均误差
- **关注区域**：
  - BSJ ±20nt：环形拓扑影响最大的区域
  - 全序列：整体预测性能

### 统计检验

- **Paired t-test**：比较标准PE和TPE在各序列上的表现
- **阈值**：p < 0.05 为统计显著

## 如何解读结果

### 成功的标志

- TPE 在 BSJ 区域的 MSE 显著低于标准PE（例如 p < 0.05）
- 改进幅度：5-15%（预期范围内）
- 全序列的 MSE 不应明显劣于标准PE

### 失败的迹象

- p > 0.05（无显著差异）
- TPE MSE 远高于标准PE
- 这是**正常的**，说明代理任务尚需优化

### 调试建议

如果结果不理想，可以尝试：
1. **增加 harmonics 数量**：`--n-harmonics 32`（默认16）
2. **增加训练epochs**：`--n-epochs 200`
3. **调整学习率**：`--lr 5e-5`
4. **增大序列数量**：`--n-sequences 100`

## Manuscript 文件修改

实验完成后，从控制台复制以下信息到 `confluencia_3.0_bioinformatics_original.md`：

**位置1 - Abstract（第11行）：**
```markdown
TPE reduces BSJ-flanking region prediction error by **[你的XX%]** relative to standard positional encoding (**[MSE values]**, **[你的p-value]**)
```

**位置2 - Results Table（第113-114行）：**
```markdown
| Standard PE | **[your_MSE]** | **[your_MSE]** | — |
| TPE (TorusFold) | **[your_MSE]** | **[your_MSE]** | **[your_delta%]** |
```

**位置3 - Results 正文（第148行左右）：**
```markdown
TPE reduces prediction error in the BSJ-flanking region by **[你的XX%]** relative to standard PE (**[你的p-value]**)
```

## 注意事项

1. **计算资源**：
   - CPU 可运行，但较慢
   - 建议在 AutoDL 或 GPU 上运行以节省时间

2. **随机种子**：固定 seed (42 + seed_idx*17)，确保结果可重复

3. **数据量**：当前使用合成数据。如需真实数据，需先从 circBase 下载50条序列和它们在BSJ附近的配对概率（可使用ViennaRNA或参考文献中的伪标签）

4. **实验时间**：
   - 20序列+50epochs：约10-30分钟（CPU）
   - 50序列+100epochs：约1-3小时（CPU）

5. **结果重复性**：运行3次不同seed，报告中报告平均值±标准差

## 文件清单

```
D:\IGEM集成方案\manuscripts\
├── scripts/
│   └── torusfold_proxy_experiment.py   # 实验主脚本
├── figures/
│   ├── confluencia_3.0_bioinformatics_original.md  # 论文（待填入实验数据）
│   ├── confluencia_3.0_bioinformatics_original.docx # 论文Word文档
│   └── results/
│       ├── proxy_experiment_results.json          # 结果摘要
│       ├── proxy_experiment_comparison.png        # 对比图
│       └── proxy_experiment_comparison.pdf        # 对比图（PDF）
└── README_torusfold_experiment.md                # 本文件
```

## 常见问题

### Q: 为什么初始提示"Warnings"但脚本还在运行？
A: 这些是可选依赖的警告（如matplotlib, tqdm），不影响核心功能。脚本会用简化的替代实现。

### Q: 训练过程中loss不下降？
A: 可能原因：
- 学习率过大/过小
- 数据量太少
- 模型容量不足

**解决**：
1. 尝试降低学习率：`--lr 1e-5`
2. 增加epochs：`--n-epochs 200`
3. 增加数据：运行3次replace用不同seed，然后取平均

### Q: 如何在AutoDL上运行？
A: 
```bash
# 1. 上传脚本到AutoDL
# 2. 安装依赖
pip install torch numpy matplotlib tqdm scipy
conda install -c bioconda viennarna  # 如果要用

# 3. 运行
python torusfold_proxy_experiment.py \
    --n-sequences 50 \
    --n-epochs 100 \
    --n-seeds 3 \
    --output-dir results
```

### Q: 结果保存在哪里？
A: 当前目录的 `results/` 文件夹。复制JSON和图片文件到 manuscripts 目录。

### Q: 如果想使用真实的 circBase 数据怎么办？
A: 需要修改 `generate_diverse_circrna_sequences()` 函数，从网络下载circBase FASTA文件，然后：
1. 识别Back-splice junction位置（从circBase注释或文献）
2. 使用 ViennaRNA circ 模式生成配对概率伪标签
3. 现有脚本已经准备好处理这种输入格式