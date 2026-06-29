# TorusFold 训练策略对比与使用指南

## 策略对比表

| 策略 | 数据需求 | 核心思想 | 优势 | 劣势 | 推荐场景 |
|------|---------|---------|------|------|---------|
| **物理蒸馏** | 无真实结构 | ViennaRNA 作为教师 | ✅ 物理保证<br>✅ 无需实验数据<br>✅ 碱基配对确定性高 | ⚠️ 依赖 ViennaRNA 准确性<br>⚠️ 无法捕捉新配对 | **最推荐**<br>基础方案 |
| **对比学习** | 无标签 | 序列扰动→几何一致 | ✅ 完全自监督<br>✅ 学习内在几何<br>✅ 无负样本风险（BYOL） | ⚠️ 训练时间长<br>⚠️ 需调整温度参数 | 无 ViennaRNA 环境<br>序列多样性高 |
| **FixMatch** | 无标签 | 弱增强→强监督 | ✅ 强鲁棒性<br>✅ 自适应置信度<br>✅ 简单高效 | ⚠️ 需设计增强策略<br>⚠️ 置信阈值敏感 | 噪声数据<br>训练不稳定 |
| **迁移学习** | PDB 线性 RNA | 局部锚点+渐进解冻 | ✅ 利用现有数据<br>✅ 物理保证局部<br>✅ 降低不确定性 | ⚠️ 最冒险<br>⚠️ 需预训练模型 | 有 PDB RNA 数据<br>治疗级 circRNA |

---

## 策略 1: 物理蒸馏 (Physics Distillation)

### 核心原理

```
ViennaRNA 教师模型（物理软件）
         ↓
  二级结构预测（能量最低原理）
         ↓
  接触图 + 配对概率 + MFE
         ↓
  作为"虚拟真实标签"
         ↓
  距离学习模型（学生）
         ↓
  损失 = 学生预测 - 教师标签
```

### 关键设计

#### 1. 接触图蒸馏

```python
# ViennaRNA circ 模式
md = RNA.md()
md.circ = True  # 🔑 关键：环状模式
fc = RNA.fold_compound(seq, md)

# MFE 结构 + 配分函数概率
structure, mfe = fc.mfe()
fc.pf()  # 配分函数

# 构建接触图
for i, j in pairs:
    prob = fc.get_pair_probs(i+1, j+1)
    contact_map[i, j] = prob
```

#### 2. 损失函数设计

```python
# MSE 损失（接触图）
L_contact = MSE(P_model, P_vienna)

# KL 散度（配对概率分布）
L_kl = KL(P_vienna || P_model)

# 能量一致性
L_energy = |E_model - E_vienna|

# BSJ 闭合
L_closure = |‖x[0] - x[-1]‖ - 5.9Å|²

# 总损失
L_total = w_contact × L_contact
        + w_kl × L_kl
        + w_energy × L_energy
        + w_closure × L_closure
```

#### 3. 置信度加权

```python
# 高置信配对权重大
weight = where(
    teacher_contact > 0.9,
    teacher_contact × 2.0,  # 高置信加权
    0.1,                     # 低置信降权
)

loss = MSE × weight
```

### 使用方法

```bash
# 基础物理蒸馏
python unified_training_strategy.py --strategy physics_distillation --epochs 50

# 调整权重
python unified_training_strategy.py \
    --strategy physics_distillation \
    --w-contact 1.5 \
    --w-kl 0.7 \
    --w-closure 3.0
```

### 预期效果

- **训练稳定性**：高（物理约束明确）
- **收敛速度**：快（~20-30 epoch）
- **BSJ 闭合精度**：< 1Å
- **配对准确率**：依赖 ViennaRNA

---

## 策略 2: 对比学习 (Contrastive Learning)

### 核心原理

```
同一 circRNA 序列
         ↓
    ┌─────┴─────┐
    │           │
  视角1        视角2
  (突变)      (Dropout)
    │           │
    ↓           ↓
  编码器      编码器
    │           │
    ↓           ↓
  投影头      投影头
    │           │
    ↓           ↓
   z_i        z_j
    │           │
    └─────┬─────┘
          ↓
    一致性损失
    (即使扰动，几何特征必须一致)
```

### 关键设计

#### 1. SimCLR（InfoNCE 损失）

```python
# 正样本对：(z_i, z_j) —— 同序列不同扰动
# 负样本对：所有其他组合

sim = z @ z.T / temperature  # 相似度矩阵

# 正样本位置
labels = [B, B+1, ..., 2B-1]  # z_i[b] 与 z_j[b] 是正样本

loss = CrossEntropy(sim, labels)
```

**优势**：简单，无需额外网络
**劣势**：需要大量负样本，依赖批次大小

#### 2. BYOL（自蒸馏，无负样本）

```python
# 在线网络：编码器 → 投影头 → 预测头
p_online = predictor(project(encoder(seq_i)))

# 目标网络：编码器 → 投影头（动量更新）
z_target = projector_target(encoder_target(seq_j))

# 损失：余弦相似度
loss = 2 - 2 × cos_similarity(p_online, z_target)

# 动量更新目标网络
θ_target = 0.996 × θ_target + 0.004 × θ_online
```

**优势**：无负样本，训练稳定
**劣势**：需要动量更新，架构复杂

#### 3. 几何一致性对比

```python
# 同序列两个增强版本 → 预测的接触图必须相似

contact_i = exp(-dist(coords_i) / T)
contact_j = exp(-dist(coords_j) / T)

loss = KL(contact_i || contact_j)
```

### 使用方法

```bash
# SimCLR 框架
python unified_training_strategy.py \
    --strategy contrastive \
    --contrastive-framework simclr \
    --temperature 0.07

# BYOL 框架（推荐）
python unified_training_strategy.py \
    --strategy contrastive \
    --contrastive-framework byol \
    --temperature 0.1
```

### 预期效果

- **训练稳定性**：中（BYOL > SimCLR）
- **收敛速度**：慢（~50-100 epoch）
- **几何一致性**：高
- **无需外部数据**：✅

---

## 策略 3: FixMatch 一致性训练

### 核心原理

```
原始序列
    ↓
┌───┴───┐
│       │
弱增强  强增强
│       │
↓       ↓
保守   激进
预测   预测
│       │
↓       ↓
硬伪   软预测
标签   ↓
│       │
↓       ↓
置信度 > 0.9？
│       │
└───┬───┘
    ↓
用弱增强的硬标签监督强增强
（强迫模型学习"不管怎么变，本质不变"）
```

### 关键设计

#### 1. 弱增强 → 硬伪标签

```python
# 弱增强：极保守（突变率 1%，噪声 0.5Å）
seq_weak = augment(seq, mutation=0.01, noise=0.5)

# 模型预测
pred_weak = model(seq_weak)

# 只保留高置信（> 0.9）作为硬标签
pseudo_label = where(pred_weak > 0.9, 1.0, 0.0)
```

#### 2. 强增强 → 一致性训练

```python
# 强增强：激进（突变率 15%，噪声 3Å + 旋转）
seq_strong = augment(seq, mutation=0.15, noise=3.0, rotation=True)

# 模型预测
pred_strong = model(seq_strong)

# 损失：只在高置信区域计算
loss = where(
    confidence_mask,
    MSE(pred_strong, pseudo_label),
    0.0  # 低置信区域不参与训练
)
```

#### 3. 分层监督

```python
# 局部结构（高置信）→ 硬监督
# 全局拓扑（低置信）→ 软监督

loss = hard_loss × confidence_high
     + soft_loss × (1 - confidence_high)
```

### 使用方法

```bash
# FixMatch 基础
python unified_training_strategy.py \
    --strategy fixmatch \
    --confidence-threshold 0.9 \
    --weak-aug 0.1 \
    --strong-aug 1.0

# ReMixMatch（分布对齐）
python unified_training_strategy.py \
    --strategy fixmatch \
    --preset advanced
```

### 预期效果

- **训练稳定性**：高（置信度过滤）
- **收敛速度**：中（~30-50 epoch）
- **对噪声鲁棒**：✅
- **置信度自适应**：✅

---

## 策略 4: 迁移学习 (Transfer Learning)

### 核心原理

```
PDB 线性 RNA（数千个）
    ↓
预训练基础模型
    ↓
预测 circRNA
    ↓
筛选高置信局部片段（> 0.99）
    ↓
冻结锚点参数
    ↓
只训练 BSJ 连接区
    ↓
渐进解冻
    ↓
全量微调
```

### 关键设计

#### 1. 预训练（PDB 线性 RNA）

```python
# PDB RNA 数据源
- tRNA (数百个)
- ribozyme (数百个)
- mRNA fragments
- ribosomal RNA fragments

# 预训练任务：标准 3D 结构预测
loss = MSE(pred_coords, pdb_coords)
```

#### 2. 锚点生成

```python
# 预训练模型预测 circRNA
pred_coords = pretrained_model(circRNA_seq)

# 识别高置信局部区域
for region in sliding_window(seq):
    conf = confidence[region].mean()
    
    if conf > 0.99:
        # 高置信锚点（冻结）
        anchors.append({
            'region': region,
            'coords': pred_coords[region],
            'frozen': True,
        })
    elif conf > 0.7:
        # 中置信（可微调）
        anchors.append({
            'region': region,
            'coords': pred_coords[region],
            'frozen': False,
        })
```

#### 3. 渐进式训练

```python
# Stage 0（Epoch 0-10）：锚定阶段
freeze_except_bsj()  # 只训练 BSJ 相关层
loss = closure_loss  # 只计算 BSJ 闭合

# Stage 1（Epoch 10-30）：微调阶段
unfreeze_medium_confidence()  # 解冻中置信区域
loss = closure_loss + anchor_loss

# Stage 2（Epoch 30+）：全量阶段
unfreeze_all()  # 全部可训练
loss = full_training_loss
```

### 使用方法

```bash
# 需要预训练模型
python unified_training_strategy.py \
    --strategy transfer \
    --pretrained path/to/pretrained.pt \
    --epochs 100

# 渐进解冻
python unified_training_strategy.py \
    --strategy transfer \
    --pretrained pretrained.pt \
    --preset advanced
```

### 预期效果

- **利用现有数据**：✅（PDB 数千个线性 RNA）
- **物理保证**：✅（局部结构从已知数据迁移）
- **降低不确定性**：✅
- **训练稳定性**：中（依赖预训练质量）
- **收敛速度**：快（锚点已接近最优）

---

## 推荐组合策略

### 组合 1: 基础方案（最推荐）

```bash
python unified_training_strategy.py --preset standard
```

**配置**：
- 物理蒸馏（ViennaRNA 教师）
- FixMatch 一致性训练

**适用场景**：
- 无真实 circRNA 结构数据
- 有 ViennaRNA 环境
- 快速原型验证

**预期效果**：
- 训练稳定
- BSJ 闭合精度 < 1Å
- ~50 epoch 收敛

---

### 组合 2: 进阶方案

```bash
python unified_training_strategy.py --preset advanced
```

**配置**：
- 物理蒸馏
- 对比学习（BYOL）
- 渐进解冻

**适用场景**：
- 序列多样性高
- 需要几何一致性
- 训练时间充足

**预期效果**：
- 几何一致性高
- 对扰动鲁棒
- ~100 epoch 收敛

---

### 组合 3: 生产级方案（最高质量）

```bash
python unified_training_strategy.py --preset production
```

**配置**：
- 预训练（PDB 线性 RNA）
- 物理蒸馏（锐化 + 置信度加权）
- FixMatch（分布对齐）
- 对比学习（BYOL + 几何一致性）
- 渐进解冻

**适用场景**：
- 有 PDB RNA 数据
- 治疗级 circRNA 设计
- 最高质量要求

**预期效果**：
- 利用现有 RNA 结构数据
- 多重物理保证
- 几何一致性强
- ~150 epoch 收敛

---

## 训练监控指标

### 关键指标

| 指标 | 物理蒸馏 | 对比学习 | FixMatch | 迁移学习 |
|------|---------|---------|---------|---------|
| **训练损失** | L_total | InfoNCE/余弦 | 一致性损失 | 锚点损失 |
| **置信度比率** | - | - | ✅ 关键 | ✅ 关键 |
| **BSJ 闭合距离** | ✅ 关键 | ✅ | ✅ 关键 | ✅ 关键 |
| **接触图准确率** | ✅ | ✅ | ✅ | ✅ |
| **几何一致性** | - | ✅ 关键 | ✅ | ✅ |

### 停止条件

```python
# 物理蒸馏
if closure_dist < 1Å and contact_acc > 0.8:
    stop()

# FixMatch
if confidence_ratio > 0.9 and closure_dist < 1Å:
    stop()

# 对比学习
if geometric_consistency < 0.1 and closure_dist < 1Å:
    stop()
```

---

## 调参建议

### 物理蒸馏

```python
# 推荐参数
w_contact = 1.0-1.5
w_kl = 0.5-0.7
w_energy = 0.3-0.5
w_closure = 2.0-3.0

temperature_sharpen = 0.5  # 锐化温度
confidence_threshold = 0.7  # 置信阈值
```

### 对比学习

```python
# SimCLR
temperature = 0.07  # 低温度 = 高区分度
batch_size = 64+    # 需要足够负样本

# BYOL
momentum = 0.996-0.998  # 目标网络更新速度
temperature = 0.1
```

### FixMatch

```python
confidence_threshold_weak = 0.9-0.95  # 弱增强伪标签阈值
weak_mutation_rate = 0.01-0.05
strong_mutation_rate = 0.1-0.2
```

---

## 总结

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| **快速验证** | 物理蒸馏 | 无需外部数据，物理保证 |
| **无 ViennaRNA** | 对比学习 | 完全自监督 |
| **噪声数据** | FixMatch | 自适应置信度过滤 |
| **治疗级 circRNA** | 迁移学习 + 组合 | 利用 PDB 数据，多重保证 |
| **最高质量** | 生产级组合 | 全策略整合 |

**核心思想**：在无真实 circRNA 结构数据的情况下，利用物理规律（ViennaRNA）、几何一致性（对比学习）、训练鲁棒性（FixMatch）和现有数据（迁移学习）作为"虚拟监督"，通过多重约束确保模型收敛到物理合理的 3D 结构。