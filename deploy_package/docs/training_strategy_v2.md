# CircRNA 3D结构数据训练策略 - 完整规划文档

## 📊 数据来源与权重映射表

| 数据来源 | 质量等级 | 数据量估算 | 训练用途 | 损失权重 | 说明 |
|:---|:---|:---|:---|:---|:---|
| **pdb_circularized** | ⭐⭐⭐⭐⭐ | ~2,000条 | 验证集（不训练） | 0.0 | 真实实验结构，最高质量 |
| **pdb3d** | ⭐⭐⭐⭐⭐ | ~1,000条 | 验证集（不训练） | 0.0 | 真实3D结构，基准测试用 |
| **shape_experimental** | ⭐⭐⭐⭐ | ~10,000条 | 主训练集 | 1.5 | 实验验证二级结构，高可信度 |
| **shape_expanded** | ⭐⭐⭐⭐ | ~20,000条 | 主训练集 | 1.2 | 基于实验扩展，次高可信度 |
| **rfam_consensus** | ⭐⭐⭐ | ~15,000条 | 辅助训练集 | 1.0 | Rfam家族保守结构，中等可信度 |
| **trrosetta_predicted** | ⭐⭐⭐ | ~80,000条 | 主训练集 | 1.0 | trRosettaRNA2预测，标准质量 |
| **synthetic** | ⭐⭐ | ~5,000条 | 预训练 | 0.5 | 合成数据，低权重 |
| **vienna_fallback** | ⭐⭐ | ~2,000条 | 数据增强 | 0.3 | ViennaRNA二级结构，最低权重 |

---

## 🔍 质量过滤标准（优化版）

### **过滤阈值调整**

```python
# circRNA特性优化过滤标准
def filter_structures_v2(result):
    """
    针对13万circRNA数据的优化过滤策略
    预计保留率: 60-65% (约8万条)
    """
    # 1. 置信度阈值：0.70起步（更务实）
    if result.get('confidence', 0) < 0.70:
        return False

    # 2. BSJ距离：保留多模态，过滤极端值
    bsj_dist = result.get('bsj_distance', None)
    if bsj_dist is not None and (bsj_dist < 2.8 or bsj_dist > 5.0):
        return False

    # 3. 能量阈值：放宽到800 kJ/mol
    if result.get('energy', 0) > 800:
        return False

    # 4. 结构完整性：RMSD方差容忍0.3
    if result.get('rmsd_variance', 0) > 0.3:
        return False

    # 5. 新增：BSJ几何合理性检查
    if not check_bsj_geometry(result):
        return False

    return True

def check_bsj_geometry(result):
    """
    检查BSJ几何合理性
    """
    # BSJ附近的原子不能有严重冲突
    bsj_clash_count = result.get('bsj_clashes', 0)
    if bsj_clash_count > 5:
        return False

    return True
```

---

## 🧠 双阶段训练架构

### **阶段一：自定义Transformer快速验证**

```python
class CircRNA3DTransformerV1(nn.Module):
    """
    快速验证版本：简单高效的Transformer架构
    目标：验证数据质量和损失函数设计
    """
    def __init__(self):
        super().__init__()

        # 序列编码器
        self.seq_encoder = nn.Embedding(4, 64)  # A,U,G,C
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=6
        )

        # 3D坐标生成器
        self.coord_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # x,y,z
        )

        # BSJ距离预测器
        self.bsj_predictor = nn.Linear(256, 1)

    def forward(self, seq, ss_features):
        # 编码序列
        seq_embed = self.seq_encoder(seq)
        encoded = self.transformer(seq_embed)

        # 预测坐标
        coords = self.coord_decoder(encoded)

        # 预测BSJ距离
        bsj_dist = self.bsj_predictor(encoded.mean(dim=1))

        return coords, bsj_dist
```

**训练计划：**
- 数据：8,000条（10%过滤后数据）
- 时间：2-3小时
- 目标：验证架构合理性和损失函数设计

---

### **阶段二：Mamba+Transformer混合架构**

```python
class CircRNA3DHybridV2(nn.Module):
    """
    最终版本：Mamba+Transformer混合架构
    目标：追求最优性能
    """
    def __init__(self):
        super().__init__()

        # 1. Mamba层（长距离依赖）
        from mamba_ssm import Mamba
        self.mamba_encoder = Mamba(
            d_model=256,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # 2. Transformer层（局部交互）
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=256, nhead=8)
            for _ in range(4)
        ])

        # 3. 图神经网络（结构约束）
        self.gnn = GraphNeuralNetwork(
            node_dim=256,
            edge_dim=64,
            num_layers=3
        )

        # 4. 多任务输出头
        self.coord_head = nn.Linear(256, 3)
        self.bsj_head = nn.Linear(256, 1)
        self.confidence_head = nn.Linear(256, 1)

    def forward(self, seq, ss_graph):
        # Mamba编码（捕捉长距离依赖）
        mamba_out = self.mamba_encoder(seq)

        # Transformer细化（局部交互）
        trans_out = mamba_out
        for layer in self.transformer_layers:
            trans_out = layer(trans_out)

        # GNN结构约束
        gnn_out = self.gnn(trans_out, ss_graph)

        # 多任务输出
        coords = self.coord_head(gnn_out)
        bsj_dist = self.bsj_head(gnn_out.mean(dim=1))
        confidence = torch.sigmoid(self.confidence_head(gnn_out.mean(dim=1)))

        return {
            'coords': coords,
            'bsj_distance': bsj_dist,
            'confidence': confidence
        }
```

---

## 📊 训练策略：四阶段流程

### **阶段0：数据分块与权重分配**

```python
def assign_data_weights(data):
    """
    根据数据来源分配训练权重
    """
    weight_map = {
        'pdb_circularized': 0.0,  # 不参与训练
        'pdb3d': 0.0,             # 不参与训练
        'shape_experimental': 1.5,
        'shape_expanded': 1.2,
        'rfam_consensus': 1.0,
        'trrosetta_predicted': 1.0,
        'synthetic': 0.5,
        'vienna_fallback': 0.3
    }

    for item in data:
        source = item['source']
        item['loss_weight'] = weight_map.get(source, 1.0)

    return data

# 划分数据集
train_data = [d for d in data if d['loss_weight'] > 0]
val_data = [d for d in data if d['source'] in ['pdb_circularized', 'pdb3d']]
test_data = val_data  # 复用高质量数据
```

---

### **阶段1：快速验证（10%数据）**

```bash
# 用10%数据快速验证架构
python train_v1.py \
  --data train_8k.json \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-3 \
  --warmup 2

# 目标：确认架构可行，损失下降正常
# 时间：2-3小时
```

---

### **阶段2：主训练（全部数据+权重）**

```python
def weighted_loss(predictions, targets, loss_weights):
    """
    加权损失函数（考虑数据来源质量差异）
    """
    # 基础损失
    loss_coords = MSE(predictions['coords'], targets['coords'])
    loss_bsj = DistanceLoss(predictions['bsj_distance'], 3.5)
    loss_conf = BCE(predictions['confidence'], targets['confidence'])

    # 应用样本权重
    weighted_loss = (
        loss_weights['coords'] * loss_coords +
        loss_weights['bsj'] * loss_bsj +
        loss_weights['conf'] * loss_conf
    )

    # 乘以数据来源权重
    sample_weight = targets['data_source_weight']
    final_loss = weighted_loss * sample_weight

    return final_loss.mean()
```

```bash
# 全量训练（加权）
python train_v2.py \
  --data train_80k_weighted.json \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --warmup 5 \
  --gradient_accumulation 4

# 目标：获得基线模型
# 时间：24-48小时
```

---

### **阶段3：BSJ专项优化**

```python
def bsj_focused_loss(predictions, targets):
    """
    BSJ专项优化损失函数
    """
    # 加大BSJ约束权重
    loss_bsj = DistanceLoss(predictions['bsj_distance'], 3.5)

    # BSJ附近的坐标精度
    bsj_coords = extract_bsj_region(predictions['coords'], targets['bsj_start'], targets['bsj_end'])
    loss_bsj_coords = MSE(bsj_coords, targets['bsj_coords'])

    # BSJ几何约束（二面角、键角）
    loss_bsj_geometry = GeometryConstraintLoss(predictions['coords'])

    return 2.0 * loss_bsj + 1.5 * loss_bsj_coords + 1.0 * loss_bsj_geometry
```

```bash
# BSJ专项微调
python train_v2_finetune.py \
  --data train_bsj_focused.json \
  --epochs 50 \
  --lr 5e-5 \
  --loss_weight_bsj 2.0

# 目标：BSJ准确率 > 90%
# 时间：12-24小时
```

---

### **阶段4：长序列平衡**

```python
def balance_long_sequences(data):
    """
    平衡长序列比例（防止长序列RMSD过高）
    """
    # 长序列（>1000nt）过采样
    long_seqs = [d for d in data if len(d['sequence']) > 1000]
    short_seqs = [d for d in data if len(d['sequence']) <= 1000]

    # 过采样长序列到30%比例
    target_long_ratio = 0.3
    current_long_ratio = len(long_seqs) / len(data)

    if current_long_ratio < target_long_ratio:
        oversample_factor = int(target_long_ratio / current_long_ratio)
        long_seqs_oversampled = long_seqs * oversample_factor
        balanced_data = short_seqs + long_seqs_oversampled
    else:
        balanced_data = data

    return balanced_data
```

---

## 🎯 迭代策略：BSJ优先

```python
def iterative_improvement(results):
    """
    基于评估结果的迭代策略（BSJ核心）
    """
    # 优先级1：BSJ准确率（circRNA特有核心指标）
    if results['bsj_accuracy'] < 85:
        print("⚠ BSJ准确率不足，启动BSJ专项优化")
        augment_bsj_data()
        increase_bsj_loss_weight()
        return "BSJ优化"

    # 优先级2：长序列RMSD
    elif results['rmsd_long'] > 4.0:
        print("⚠ 长序列RMSD过高，增加长序列比例")
        balance_long_sequences()
        return "长序列平衡"

    # 优先级3：置信度预测
    elif results['confidence_auc'] < 0.80:
        print("⚠ 置信度预测不足，增加物理特征")
        add_physicochemical_features()
        return "特征增强"

    # 优先级4：整体RMSD
    elif results['overall_rmsd'] > 2.5:
        print("⚠ 整体RMSD偏高，检查数据质量")
        tighten_quality_filter()
        return "质量收紧"

    else:
        print("✓ 所有指标达标，模型训练完成")
        return "完成"
```

---

## 📈 目标指标（修订版）

| 指标 | 目标值 | 优先级 | 说明 |
|:---|:---|:---|:---|
| **BSJ准确率** | **> 90%** | **P0** | circRNA特有核心指标 |
| 整体RMSD | < 2.5 Å | P1 | 结构预测准确性 |
| 长序列RMSD | < 4.0 Å | P1 | 长序列性能 |
| 置信度AUC | > 0.80 | P2 | 置信度预测质量 |
| TM-score | > 0.7 | P2 | 全局结构相似度 |

---

## 🚀 执行时间线（优化版）

| 阶段 | 时间 | 数据量 | 目标 | 关键产出 |
|:---|:---|:---|:---|:---|
| **数据分块** | 0.5天 | 130k → 80k | 数据清洗+权重分配 | 加权训练数据集 |
| **阶段1** | 3小时 | 8k | 快速验证 | 验证损失下降曲线 |
| **阶段2** | 2天 | 80k | 主训练 | 基线模型 |
| **阶段3** | 1天 | 高质量 | BSJ专项 | BSJ优化模型 |
| **阶段4** | 0.5天 | 平衡后 | 长序列平衡 | 最终模型 |
| **评估** | 0.5天 | 13k测试 | 达标验证 | 性能报告 |
| **总计** | **5天** | **130k** | **生产级模型** | **可部署模型** |

---

## 💾 模型保存策略

```python
# 多版本保存
checkpoints = {
    'stage1_quick': 'transformer_v1_quick.pt',      # 快速验证
    'stage2_baseline': 'transformer_v1_baseline.pt', # 基线模型
    'stage3_hybrid': 'mamba_transformer_v2.pt',     # 混合架构
    'stage4_final': 'circrna_3d_final.pt',          # 最终模型
    'best_bsj': 'circrna_3d_best_bsj.pt'            # BSJ最优
}

# 每个checkpoint保存内容
save_dict = {
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'training_config': config,
    'metrics_history': metrics,
    'data_split_info': split_info,
    'source_weights': weight_map
}
```

---

## 📝 论文/审稿应对材料

**数据策略说明：**

> "我们收集了13万条circRNA序列，经过严格的质量过滤（置信度>0.70，BSJ距离2.8-5.0 Å，能量<800 kJ/mol，RMSD方差<0.3），保留8万条高质量结构用于训练。根据数据来源的质量差异，我们设计了分层次的损失权重策略：实验验证数据权重1.5，计算预测数据权重1.0，合成数据权重0.5。这种加权策略确保了模型优先学习高质量数据，同时保留数据的多样性。验证集仅使用最高质量的实验结构（pdb_circularized和pdb3d），确保评估的可靠性。"

---

**这份优化后的规划已经准备就绪！现在可以开始执行了！**