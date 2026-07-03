# 动态置信度系统文档

## 核心改进

### V1/V2 系统的问题

```python
# 硬编码置信度（V1/V2）
DEFAULT_CONFIDENCE = {
    "pdb_circularized": 1.0,       # 为什么是 1.0？
    "isrnacirc": 0.7,             # 为什么是 0.7？
    "synthetic": 0.3,             # 为什么是 0.3？
}
```

**问题：**
- ❌ 主观判断（"拍脑袋"）
- ❌ 无法验证
- ❌ 不考虑具体样本的物理质量
- ❌ 所有同一来源的数据置信度相同

---

### V3 动态计算公式

```python
confidence = base_conf(source) * 0.50
           + energy_score * 0.20
           + bsj_score * 0.15
           + clash_score * 0.10
           + convergence_score * 0.05
```

**改进：**
- ✅ 结合来源质量 + 物理指标
- ✅ 每个样本有独立置信度
- ✅ 可计算、可验证
- ✅ 物理意义明确

---

## 计算公式详解

### 1. 来源评分（50%权重）

```python
base_confidence = BASE_CONFIDENCE_BY_SOURCE[source]

BASE_CONFIDENCE_BY_SOURCE = {
    "pdb_circularized": 0.95,    # 真实 PDB 环化
    "isrnacirc": 0.70,           # IsRNAcirc 物理模拟
    "synthetic": 0.40,           # ViennaRNA 合成
}
```

**原理：来源是置信度的基础，不同来源的质量差异显著。**

---

### 2. 能量评分（20%权重）

```python
energy_score = 1.0 - clip(energy / energy_max, 0.0, 1.0)

# 示例：
energy = 350 kJ/mol → energy_score = 1 - 350/1000 = 0.65
energy = 750 kJ/mol → energy_score = 1 - 750/1000 = 0.25
```

**原理：能量越低越稳定，置信度越高。**

---

### 3. BSJ 评分（15%权重）

```python
bsj_score = 1.0 - |bsj_distance - target| / tolerance

# 示例：
bsj_distance = 3.5Å → bsj_score = 1.0
bsj_distance = 6.0Å → bsj_score = 0.0 (超出范围)
```

**原理：BSJ 闭环是 circRNA 的核心特征，距离越接近目标越好。**

---

### 4. Clash 评分（10%权重）

```python
clash_score = 1.0 - clip(clash_count / clash_max, 0.0, 1.0)

# 示例：
clash_count = 1 → clash_score = 0.9
clash_count = 8 → clash_score = 0.2
```

**原理：空间冲突越少，结构越合理。**

---

### 5. 收敛性评分（5%权重）

```python
convergence_score = 1.0 - clip(rmsd_variance / rmsd_var_max, 0.0, 1.0)

# 示例：
rmsd_variance = 0.1 → convergence_score = 0.8
rmsd_variance = 0.4 → convergence_score = 0.2
```

**原理：RMSD 方差越小，结构越稳定。**

---

## 使用示例

### 基本使用

```python
from dynamic_confidence import compute_dynamic_confidence

breakdown = compute_dynamic_confidence(
    source="pdb_circularized",
    energy=350.0,
    bsj_distance=3.5,
    clash_count=1,
    rmsd_variance=0.1,
)

print(f"置信度: {breakdown.overall:.3f}")
print(f"来源: {breakdown.source}")
print(f"能量评分: {breakdown.energy_score:.3f}")
print(f"BSJ评分: {breakdown.bsj_score:.3f}")
```

---

### 批量处理

```python
from dynamic_confidence import compute_confidence_for_dataset

records = [
    {
        'source': 'pdb_circularized',
        'energy': 350.0,
        'bsj_distance': 3.5,
        'clash_count': 1,
        'rmsd_variance': 0.1,
    },
    {
        'source': 'synthetic',
        'energy': 750.0,
        'bsj_distance': 6.0,
        'clash_count': 8,
        'rmsd_variance': 0.4,
    },
]

confidences, breakdowns = compute_confidence_for_dataset(records)

print(f"平均置信度: {np.mean(confidences):.3f}")
```

---

### 质量报告

```python
from dynamic_confidence import generate_quality_report

report = generate_quality_report(breakdowns)

print(f"平均置信度: {report['mean_confidence']:.3f}")
print(f"低质量样本: {report['n_low_quality']}/{report['n_total']}")
print(f"高质量样本: {report['n_high_quality']}/{report['n_total']}")
```

---

## 自定义配置

### 自定义权重

```python
custom_weights = {
    'source': 0.30,      # 降低来源权重
    'energy': 0.30,      # 提高能量权重
    'bsj': 0.25,         # 提高 BSJ 权重
    'clash': 0.10,
    'convergence': 0.05,
}

breakdown = compute_dynamic_confidence(..., weights=custom_weights)
```

---

### 自定义阈值

```python
custom_thresholds = {
    'energy_max': 500,      # 更严格的能量阈值
    'bsj_min': 3.0,         # 更严格的 BSJ 范围
    'bsj_max': 4.0,
    'clash_max': 5,
}

breakdown = compute_dynamic_confidence(..., thresholds=custom_thresholds)
```

---

## 对比分析

| 维度 | V1/V2（硬编码） | V3（动态） |
|------|---------------|----------|
| **计算方式** | 主观判断 | **客观计算** |
| **验证性** | ❌ 无法验证 | ✅ **物理指标可验证** |
| **样本差异** | ❌ 所有样本相同 | ✅ **每个样本独立** |
| **可解释性** | ❌ 黑盒 | ✅ **五维度分解** |
| **灵活性** | ❌ 固定值 | ✅ **权重/阈值可调** |

---

## 理论依据

### 为什么需要动态置信度？

```
科学哲学问题：

  Q1: 什么是"置信度"？
      → "我们对数据的信任程度"
      
  Q2: 如何量化？
      → V1/V2: 硬编码（主观）
      → V3: 物理指标（客观）
      
  Q3: 物理指标能代表"信任度"吗？
      → 不能完全代表，但比"拍脑袋"更科学
      → 至少可以验证、可解释
```

---

### 物理指标的合理性

| 指标 | 物理意义 | 与置信度的关系 |
|------|---------|---------------|
| **能量** | 结构稳定性 | 低能量 → 稳定 → 高置信度 |
| **BSJ距离** | circRNA核心特征 | 接近目标 → 合理 → 高置信度 |
| **Clash数** | 空间合理性 | 少冲突 → 合理 → 高置信度 |
| **RMSD方差** | 结构收敛性 | 低方差 → 稳定 → 高置信度 |

---

## 下一步改进

### 可能的改进方向

1. **机器学习权重**
   - 用真实数据训练权重
   - 优化权重配置

2. **更多物理指标**
   - 键长方差
   - 键角异常
   - 二面角分布

3. **自适应阈值**
   - 根据数据来源调整阈值
   - 动态优化阈值范围

4. **置信度传播**
   - 在训练过程中更新置信度
   - 基于模型反馈调整

---

## 参考

- IsRNAcirc: Jiang et al., PLOS Comp Biol 2024
- ViennaRNA: Lorenz et al., 2011
- AlphaFold3 stereochemistry issues: Stein et al., 2024