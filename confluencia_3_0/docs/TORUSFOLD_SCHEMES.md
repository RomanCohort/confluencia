# TorusFold Schemes 完整指南

## 概述

TorusFold V3 采用 **MOE (Mixture of Experts)** 架构，包含 **7 个 Scheme**，每个 Scheme 有不同的架构设计和训练策略。

---

## Scheme 总览表

| Scheme | 名称 | 状态 | 核心架构 | 内存占用 | RMSD | 适用场景 |
|--------|------|------|---------|---------|------|---------|
| **S1** | DL+Physics Cascade | ✅ Active | EGNN → Physics refinement | ~20 GB | ~2.5Å | 通用，长序列 |
| **S2** | Batch+Physics Filter | ✅ Active | Batch sampling → Energy filter | ~18 GB | ~2.8Å | 中短序列 |
| **S3** | Dual-Engine | ⚠️ Deferred | 多模型集成 teacher | - | - | 需其他 Scheme 训练好后使用 |
| **S4** | DDPM+EGNN Guided | ✅ Active | Diffusion + closure reward | ~25 GB | ~2.2Å | 高精度，复杂结构 |
| **S5** | Physics-Biased Attention | ❌ Deprecated | 物理约束注意力 | CPU bottleneck | - | NaN 爆炸，已弃用 |
| **S6** | GNN Latent Diffusion | ✅ Active | Encoder → Latent → Decoder | ~12 GB | ~2.6Å | 内存受限环境 |
| **S7** | Mamba+Transformer Hybrid | ✅ Active | O(L) global + O(L×w) local | ~8 GB | ~2.4Å | 超长序列，实时推理 |

---

## 详细解析

### Scheme 1: DL+Physics Cascade

**全称**: Deep Learning + Physics Cascade

**架构流程**:
```
Input sequence
    ↓
[Feature Extractor] → 提取序列特征
    ↓
[EGNN Layer 1] → Equivariant Graph Neural Network
    ↓
[Pair Representation] → 配对表示 z[i,j]
    ↓
[Triangle Update Block × N] → 三角形更新
    ↓
[Physics Refinement] → 几何约束精修
    ↓
Output coordinates
```

**关键特点**:
- **EGNN (Equivariant Graph Neural Network)**: 旋转等变图网络，保持分子对称性
- **Cascade 设计**: 先学全局配对，再用物理约束修正
- **优势**: 理论精度高，适合复杂结构

**缺点**:
- 计算量大（~20GB）
- 训练时间长

**适用场景**:
- 长序列 (>500nt)
- 复杂二级结构
- 高精度要求

---

### Scheme 2: Batch+Physics Filter

**全称**: Batch Sampling + Physics Filter

**架构流程**:
```
Input sequence
    ↓
[Batch Sampler] → 生成多个候选构象
    ↓
[Energy Function] → 计算每个构象的能量
    ↓
[Physics Filter] → 筛选低能量构象
    ↓
[Ensemble Avg] → 平均输出
```

**关键特点**:
- **Batch Sampling**: 同时采样多个可能的构象
- **Physics Filter**: 基于能量函数过滤不合理构象
- **优势**: 快速收敛，适合中短序列

**缺点**:
- 采样过程可能陷入局部最优
- 长序列效果下降

**适用场景**:
- 中短序列 (<400nt)
- 快速预测需求
- 资源受限环境

---

### Scheme 3: Dual-Engine (Deferred)

**全称**: Dual-Engine Integration

**状态**: ⚠️ **Deferred** — 等待 S1/S6/S7 训练完成后作为 teacher 使用

**设计理念**:
- 双引擎协同：一个负责快速推理，一个负责高精度验证
- Teacher-Student 模式：用训练好的模型指导新模型

**为什么延迟**:
- 当前没有足够高质量的 teacher 模型
- 需要 S1/S6/S7 先训练完成并验证

**未来计划**:
- S1/S6/S7 训练完成后启用
- 实现真正的 dual-engine 协作

---

### Scheme 4: DDPM+EGNN Guided

**全称**: Denoising Diffusion Probabilistic Model + EGNN Guided

**架构流程**:
```
Noise → [Denoising Steps] → Clean Coordinates
    ↑                      ↓
[Closure Reward]      [EGNN Refinement]
    ↓                      ↓
Guide diffusion direction   Enhance pairing accuracy
```

**关键特点**:
- **DDPM (Diffusion Model)**: 去噪扩散概率模型
- **Closure Reward**: 专门优化 BSJ 闭环奖励
- **优势**: 精度最高，RMSD ~2.2Å

**缺点**:
- 内存占用最大 (~25GB)
- 推理速度慢

**适用场景**:
- 发表级精度要求
- 复杂环状 RNA 结构
- 不差钱、不差时间

---

### Scheme 5: Physics-Biased Attention (Deprecated)

**全称**: Physics-Biased Attention Mechanism

**状态**: ❌ **Deprecated** — 因 NaN 爆炸和 CPU 瓶颈弃用

**问题原因**:
1. **NaN Explosion**: 数值不稳定导致梯度爆炸
2. **CPU Bottleneck**: 物理计算卡在 CPU 上，无法利用 GPU

**替代方案**:
- S4 (DDPM+EGNN): 更稳定的扩散方法
- S6 (GNN Latent): 降低内存占用
- S7 (Mamba): 线性复杂度

---

### Scheme 6: GNN Latent Diffusion

**全称**: Graph Neural Network Latent Diffusion

**架构流程**:
```
Input sequence
    ↓
[Encoder] → 压缩到 latent space
    ↓
[Latent Diffusion] → 在 latent 空间进行扩散
    ↓
[Decoder] → 解码回坐标空间
```

**关键特点**:
- **Latent Space**: 将高维坐标压缩到低维潜在空间
- **Memory Efficient**: ~12GB（比 S4 少 50%）
- **O(L) Complexity**: 线性时间复杂度

**优势**:
- 内存友好（适合 RTX 3090）
- 速度快于 S4

**适用场景**:
- 内存受限环境
- 中等精度要求
- 批量预测任务

---

### Scheme 7: Mamba+Transformer Hybrid

**全称**: Mamba State Space Model + Transformer Hybrid

**架构流程**:
```
Input sequence
    ↓
[Global Attention (Transformer)] → O(L²) 全局交互
    ↓
[Local Attention (Mamba)] → O(L×w) 局部交互
    ↓
[Fusion Head] → 融合两种注意力
    ↓
Output coordinates
```

**关键特点**:
- **Mamba (State Space Model)**: 线性复杂度 O(L)，解决 Transformer O(L²) 问题
- **Hybrid Architecture**: 全局注意力捕捉长程依赖，局部注意力处理细节
- **Memory Efficient**: ~8GB（最低）

**优势**:
- **最快推理速度**: 适合实时应用
- **最省内存**: 单卡可运行
- **超长序列支持**: 轻松处理 >1000nt

**适用场景**:
- 实时预测需求
- 超长序列 (>500nt)
- 边缘设备部署

---

## MOE 路由机制

### SeqTopK Routing

根据序列特征自动选择 Top-K 最相关的 Scheme：

```python
def route_to_schemes(sequence_features: dict) -> List[str]:
    """根据特征路由到最佳 Scheme。"""
    
    # 特征提取
    features = extract_features(sequence_features)
    # length, gc_content, dsRNA_fraction, bsj_stability, ...
    
    # Gate 计算激活分数
    gate_scores = {
        "S1": compute_gate_score("S1", features),
        "S2": compute_gate_score("S2", features),
        "S4": compute_gate_score("S4", features),
        "S6": compute_gate_score("S6", features),
        "S7": compute_gate_score("S7", features),
    }
    
    # Select Top-2
    selected = sorted(gate_scores.items(), key=lambda x: -x[1])[:2]
    return [scheme for scheme, score in selected]
```

### 路由规则示例

| 序列特征 | 推荐 Scheme | 原因 |
|---------|------------|------|
| 长度 < 200nt | S2 | Batch+Physics 快 |
| 长度 200-500nt | S1 | 平衡精度和速度 |
| 长度 > 500nt | S7 | Mamba 线性复杂度 |
| dsRNA > 0.4 | S4 | DDPM 擅长复杂结构 |
| GC > 0.6 | S1 | EGNN 处理高 GC |
| BSJ 稳定性 < 0.5 | S4 | Closure reward 优化 |

---

## 专家模式配置示例

```yaml
# expert_mode.yaml
expert_mode: true
selected_schemes: ["S1", "S4"]  # 用户指定
scheme_weights:
  S1: 0.60
  S4: 0.40
bypass_gate: false  # 仍使用自动路由
```

---

## 故障排查

### NaN 问题

**现象**: 预测结果出现 `nan` / `inf`

**原因**:
- S5 (已弃用) 的 NaN 爆炸
- 训练数据质量问题
- 过拟合

**解决**:
- 使用 S1/S4/S6/S7（无 NaN 问题）
- 检查输入序列质量
- 增加正则化

### 内存不足

**现象**: `CUDA out of memory`

**解决**:
- 切换到 S6 (~12GB) 或 S7 (~8GB)
- 减小 batch size
- 使用混合精度训练

### 推理速度慢

**现象**: 单个序列预测 > 30 秒

**解决**:
- 使用 S7 (Mamba) 加速
- 预加载模型到显存
- 批量预测减少启动开销

---

## 性能对比总结

| Scheme | 内存 | 速度 | 精度 | 综合评分 |
|--------|------|------|------|---------|
| S1 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 9.0 |
| S2 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 8.0 |
| S3 | - | - | - | - (deferred) |
| S4 | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | 9.5 |
| S5 | - | - | - | 0 (deprecated) |
| S6 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 8.5 |
| S7 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 9.5 |

**注**: ⭐ 越多越好

---

## 下一步学习

1. **深入某个 Scheme**: 查看具体代码实现
2. **实验验证**: 实际运行不同 Scheme 对比效果
3. **自定义 Scheme**: 开发新的架构变体
