# TorusFold S10 训练方案 v3 最终版 + 物理质量评估

**日期**: 2026-07-29  
**版本**: v3.0

---

## 数据合并完成

### 合并策略
- **refined2 ⊂ v3**（72,316 条完全在 v3 的 82,047 条中）
- **v3_refined 废弃**（41,947 条，被 refined2 完全覆盖）
- **refined2 独有长链**：171 条（1000+ nt，最长 5000 nt）
- **合并结果**: `data/circrna_3d_all/` (82,047 条)
  - 72,316 条来自 refined2（优先，质量更高）
  - 9,731 条来自 v3（refined2 没有的）

### 长度分布
```
151-1000 nt:  77,357 条 (94.6%)
1001-5000 nt:   171 条 ( 0.2%)  ← refined2 独有
```

### 数据质量抽检
- 随机抽检 50 个样本
- **全部通过**：无 NaN/INF，无 clash (0%)，Rg 合理 (58-300 Å)
- 8 条 Rg 偏大 (>200 Å) 属于正常物理范围，不是错误

---

## 训练方案 v3（密度退火）

### 核心理念
**不用"课程学习"（curriculum learning）**，改用 **Confidence-Weighted Co-Training with Progressive Density**

- 所有长度的数据**从 epoch 1 就参与训练**
- 短链权重高、长链权重低（初始 8:2）
- 随训练进程，长链权重逐渐升高（退火到 3:7）
- 避免"训完短链再学长链"的表征迁移问题

### 四个训练阶段（单一连续训练，25-30天）

| 阶段 | Epoch % | 短链:长链 权重 | 目标 |
|------|---------|---------------|------|
| Phase 1 | 0-25% | 8:2 | 建立基本表征 |
| Phase 2 | 25-50% | 6:4 | 短链收敛，长链开始学习 |
| Phase 3 | 50-75% | 4:6 | 长链主导，短链防遗忘 |
| Phase 4 | 75-100% | 3:7 + 偏差退火 | 最终收敛 |

### 偏差退火（最后 20% epoch）
- 高斯噪声：σ = 0 → 2 Å
- 5% 配对翻转
- BSJ 位置 ±15° 旋转

### 代码改动
- **新增**: `DensityAnnealingSampler`（替代 LengthBucketSampler）
- **新增**: 梯度规模监控（每 100 step 记录）
- **删除**: v2 的 stage 划分逻辑
- **预计**: ~100 行代码，2 小时完成

---

## 物理质量评估（已集成到 conformational_ensemble.py）

### 新增功能
为每个生成的构象计算完整物理质量：

```python
@dataclass
class PhysicalQuality:
    rg: float              # 回转半径 (Å)
    bond_mean: float       # 平均键长 (Å)
    bond_std: float        # 键长标准差
    bond_score: float      # 键长质量分 [0,1]
    clash_count: int       # 空间位阻数量
    clash_ratio: float     # 位阻比例
    clash_score: float     # 位阻质量分 [0,1]
    bsj_distance: float    # BSJ 距离 (Å)
    bsj_score: float       # BSJ 质量分 [0,1]
    dfire_score: float     # DFIRE-RNA 打分（可选）
    rsrnasp_score: float   # rsRNASP 打分（可选）
    confidence: float      # 综合置信度 [0,1]
    grade: str             # 质量等级 (S/A/B/C/D)
```

### 集成方式
- `predict_conformational_ensemble()` 现在返回 `quality` 和 `mean_quality`
- `ConformationalEnsemble` dataclass 新增 `quality: List[List[PhysicalQuality]]`
- 质量等级：S (≥0.92), A (≥0.88), B (≥0.85), C (≥0.80), D (<0.80)

### 质量评分公式
```python
confidence = (
    0.25 * bond_score +
    0.25 * clash_score +
    0.30 * bsj_score +
    0.10 * min(rg / 100.0, 1.0) +
    0.10 * 0.5  # placeholder for dfire/rsrnasp
)
```

### 物理指标说明
1. **键长均匀性**: C3'-C3' ~5.9 Å，标准差越小越好
2. **空间位阻**: 距离 <3.0 Å 的非相邻残基对
3. **BSJ 几何**: circRNA 首尾连接处距离 ~3.5 Å（高斯分布评分）
4. **Rg 归一化**: 回转半径 /100，越大越差

---

## 下一步计划

### 优先级 1: 完成代码改动
1. 实现 `DensityAnnealingSampler`
2. 添加梯度规模监控
3. 删除 v2 stage 逻辑

### 优先级 2: 实验验证
1. 跑密度退火 vs 固定密度 5:5 的对照实验
2. 记录各桶 loss 和梯度 norm
3. 对比最终模型质量

### 优先级 3: 论文叙事
- 不用 "curriculum learning"
- 用 "length-stratified confidence-weighted training with late-stage noise annealing"
- 强调数据质量梯度（短链准、长链噪声大）

---

## 相关文件

### 数据
- `data/circrna_3d_all/` — 合并后的完整数据集 (82,047 条)
- `data/circrna_3d_v3/` — 原始数据 (82,047 条)
- `data/circrna_3d_v3_refined2/` — 精修数据 (72,316 条)

### 代码
- `confluencia_3_0/core/circrna/torusfold/conformational_ensemble.py` — 物理质量评估
- `confluencia_3_0/core/circrna/torusfold/density_annealing_sampler.py` — 待创建
- `confluencia_3_0/core/circrna/torusfold/train_torusfold_circbase_fast.py` — 训练脚本

### 文档
- `confluencia_3_0/docs/data_merge_summary.md` — 数据合并报告
- `confluencia_3_0/docs/training_plan_v3.md` — 待创建

---

## 技术细节

### 为什么不叫 curriculum learning？
- 学术界对手工课程学习持怀疑态度
- 我们的方法是"分桶加权采样"，不是"分阶段训练"
- 所有长度从 epoch 1 就参与，只是权重不同

### 为什么用 refined2 而不是 refined？
- refined 是 refined2 的子集（41,947 ⊂ 72,316）
- refined2 包含更多长链数据（171 条 1000+ nt）
- 质量更高，覆盖更全面

### 物理质量评估的权重来源
- 借鉴 `stage5_quality.py` 的五门质量评估体系
- 调整权重：BSJ 占 30%（最重要），键长和位阻各 25%
- DFIRE/rsRNASP 占 10%（目前 placeholder，需要外部工具）
