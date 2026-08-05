# Confluencia — circRNA 3D 结构预测

## 项目概述

circRNA 3D 结构预测系统，基于 TorusFold 架构，集成 RhoFold+ RNA FM 编码器和 S10 等变解码器。

## 核心架构

### 三层层次化预测系统

```
circRNA 序列
    ↓
Level 0: MSA 聚类
  - Embedding K-means: N 条序列 → M 条代表性序列 (M=64)
  - 支持 MMseqs2 / CD-HIT CLI 回退
    ↓
Level 1: Chunk 独立预测
  - 切分: 500nt/chunk, 重叠 50nt (10%)
  - 每个 chunk: RhoFold+ RNA FM → PairTrack → CG Decoder
  - 显存: O(chunk^2) per chunk, 而非 O(L^2) 全局
    ↓
Level 2: 特征提取
  - coord_feat: MLP(3→64→256) 平均池化
  - contact_feat: AdaptiveAvgPool2d(8) → Linear(64→256)
  - node_feat: mean(node_repr) → (256,)
    ↓
Level 3: 全链融合 GNN
  - 全局注意力: chunk 间交互
  - 变换头: offset (3) + rotation (3) + scale (1) per chunk
  - 坐标精炼: MLP(3+256→128→3)
    ↓
重叠一致性损失
  - coord: Kabsch 对齐后 MSE
  - bond: (d - 5.9A)^2
  - tangent: 1 - cos(angle)
  - bsj_closure: (||P0-PL|| - 5.9)^2
```

### 串联 vs 并联

RhoFold+ 和 S10 是**串联**关系:
- RhoFold+ RNA FM 作为**编码器** (冻结, 99.5M 参数)
- S10 等变 GNN 作为**解码器** (可训练, 每个 chunk 独立)

### 关键设计决策

1. **Chunk 边界**: 环状 RNA 的 BSJ 位置必须是 chunk 边界
2. **MSA 聚类**: 先做再用, 避免 O(N×L²) 计算
3. **显存优化**: O(chunk^2) per chunk, 而非 O(L^2) 全局
4. **端到端可微**: 所有组件可梯度回传

## 文件结构

```
core/circrna/torusfold/
  msa_cluster.py          — MSA 聚类 + 分层 MSA 处理
  chunk_splitter.py       — 长序列 chunk 切分
  chunk_predictor.py      — Level 1 chunk 独立预测 + Level 2 特征提取
  chunk_fusion.py         — Level 3 全链融合 GNN
  overlap_loss.py         — 重叠区域一致性损失
  hierarchical_predictor.py — 完整管线集成
  train_hierarchical.py   — 训练脚本
  rhofold_backbone.py     — RhoFold+ RNA FM 编码器
  pair_track.py           — 三角一致性更新
  cg_decoder.py           — CG 3-bead 坐标解码
  bsj_fape.py             — BSJ FAPE 损失
```

## 数据

- `data/circrna_3d_all_consolidated.npz`: 82106 个样本, CG 3-bead 坐标
- 长度范围: 151-4984nt
- 训练时 max_len=500 过滤后 45066 个样本

## 训练

```bash
python core/circrna/torusfold/train_hierarchical.py \
  --labels data/circrna_3d_all_consolidated.npz \
  --epochs 30 \
  --device cuda \
  --chunk_size 200 \
  --max_len 500
```

## 测试

```bash
# Smoke test
python core/circrna/torusfold/tests/smoke_hierarchical.py

# 端到端测试
python core/circrna/torusfold/tests/test_hierarchical_e2e.py
```

## 依赖

- PyTorch 2.0+
- CUDA (AMD ROCm 或 NVIDIA)
- numpy, scipy
- (可选) MMseqs2 / CD-HIT CLI

## 注意事项

- Kabsch 对齐: 避免 inplace 操作 `Vt[-1, :] *= -1`, 用 `clone()` 替代
- SVG 架构图: XML 中 `<` 需转义为 `&lt;`
- GPU 训练: AMD GPU 有 experimental attention 警告, 无害
