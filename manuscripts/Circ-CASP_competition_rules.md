# Circ-CASP: circRNA 3D Structure Prediction Challenge

## 竞赛概述

Circ-CASP（Critical Assessment of circRNA Structure Prediction）是首个专注于 circRNA 三维结构预测的公开竞赛。参照 CASP 竞赛规则，旨在推动 circRNA 结构预测方法的发展。

### 背景
circRNA（环状 RNA）是一种通过反向剪接形成的共价闭合环状 RNA 分子。由于其独特的拓扑结构（5'-3' 端连接形成环），传统的线性 RNA 结构预测方法难以直接应用。Circ-CASP 旨在建立 circRNA 3D 结构预测的评估标准和基准数据集。

---

## 竞赛规则

### 1. 数据集

| 数据类型 | 数量 | 来源 | 用途 |
|----------|------|------|------|
| 训练集 | 10,000 | 合成 + IsRNAcirc 扩增 | 公开提供 |
| 测试集 | 30 | IsRNAcirc 真实结构 | 预测时公开序列，结果保密 |

**训练集特征：**
- 序列长度范围：50-500 nt（合成）+ 1000-2000 nt（IsRNAcirc 扩增）
- 包含：序列、二级结构（部分）、伪标签 3D 坐标
- 格式：JSON + NPY（兼容 PyTorch/TensorFlow）

**测试集特征：**
- 30 个真实 circRNA（来自 IsRNAcirc 预测的高质量结构）
- 预测阶段：仅公开序列信息
- 评估阶段：公开真实 3D 结构和评分

### 2. 预测目标

每个 circRNA 包含以下预测任务：

| 任务编号 | 任务名称 | 描述 | 权重 |
|----------|----------|------|------|
| T1 | **整体结构** | 全原子 RMSD | 40% |
| T2 | **BSJ 闭合** | 5'-3' 端距离误差 | 20% |
| T3 | **骨架构象** | 相邻核苷酸距离一致性 | 15% |
| T4 | **二级结构** | 配对碱基预测准确性 | 15% |
| T5 | **构象多样性** | 提供多个候选构象 | 10% |

### 3. 评分标准

#### T1: 整体结构 RMSD
$$\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}||p_i - t_i||^2}$$

评分规则：
- RMSD < 5 Å: 100 分
- RMSD < 10 Å: 80 分
- RMSD < 15 Å: 60 分
- RMSD < 20 Å: 40 分
- RMSD < 30 Å: 20 分
- RMSD ≥ 30 Å: 0 分

#### T2: BSJ 闭合距离
$$d_{\text{BSJ}} = ||p_0 - p_{N-1}||$$

真实距离约 5.9 Å（磷酸二酯键长度）。

评分规则：
- |d - 5.9| < 1 Å: 100 分
- |d - 5.9| < 2 Å: 80 分
- |d - 5.9| < 5 Å: 60 分
- |d - 5.9| < 10 Å: 40 分
- |d - 5.9| ≥ 10 Å: 0 分

#### T3: 骨架距离一致性
相邻核苷酸 C3' 原子距离应接近 5.9 Å（A-form RNA）。

$$\text{Bond\_Score} = \frac{1}{N}\sum_{i=1}^{N-1} \max(0, 100 - 20 \times |d_i - 5.9|)$$

#### T4: 二级结构配对
碱基配对预测准确性（AU/GC/GU 配对）。

$$\text{Pair\_F1} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}$$

#### T5: 构象多样性（可选）
提交多个候选构象（最多 5 个），取最佳评分。

### 4. 总分计算

$$\text{Total\_Score} = \sum_{i=1}^{5} w_i \times \text{Score}_i$$

总分范围：0-100 分。

---

## 提交格式

### 文件结构
```
submission/
├── team_info.json         # 队伍信息
├── predictions/
│   ├── circ_001_coords.npy  # 预测坐标 (N, 3)
│   ├── circ_001_pairs.json  # 预测配对
│   ├── circ_002_coords.npy
│   ├── ...
│   └── circ_030_coords.npy
└── method_description.md   # 方法描述
```

### team_info.json
```json
{
  "team_name": "Your Team Name",
  "contact_email": "email@example.com",
  "members": ["Member 1", "Member 2"],
  "method_category": "DL / Physics / Hybrid",
  "method_description": "Brief description"
}
```

### predictions/circ_XXX_coords.npy
- NumPy array: shape (N, 3), dtype float32
- N = 序列长度
- 坐标单位：Ångstrom
- 原子类型：C3' 原子（可选用 P 原子）

---

## 时间安排

| 时间节点 | 事项 |
|----------|------|
| 第 0 周 | 公布训练集 + 竞赛规则 |
| 第 1-3 周 | 模型训练阶段 |
| 第 4 周 | 公布测试集序列 |
| 第 5 周 | 提交预测结果 |
| 第 6 周 | 公布评分结果 |

---

## 基准方法

主办方提供 6 种基准方法供参考：

| 方法编号 | 方法名称 | 类型 | 预计性能 |
|----------|----------|------|----------|
| M1 | Helical Baseline | Physics | RMSD ~25 Å |
| M2 | EGNN + Physics | DL + Physics | RMSD ~15 Å |
| M3 | Dual-Engine Iterative | Hybrid | RMSD ~12 Å |
| M4 | DDPM Guided Diffusion | DL | RMSD ~10 Å |
| M5 | Physics-Biased Transformer | DL + Physics | RMSD ~8 Å |
| M6 | GNN Latent Diffusion | DL | RMSD ~10 Å |

参赛者可以：
1. 使用基准方法作为起点改进
2. 开发全新方法
3. 组合多种方法

---

## 竞赛奖项

| 奖项 | 条件 | 数量 |
|------|------|------|
| 🏆 冠军 | 总分最高 | 1 队 |
| 🥈 亚军 | 总分第二 | 1 队 |
| 🥉 季军 | 总分第三 | 1 队 |
| ⭐ 创新奖 | 方法新颖性 | 2 队 |
| 📊 最佳单项 | 单任务满分 | 5 队 |

---

## 报名方式

1. 发送邮件至：circ-casp@example.com
2. 邮件内容：
   - 队伍名称
   - 成员信息
   - 联系邮箱
3. 收到确认后获得训练集下载链接

---

## 联系方式

- 竞赛主页：https://github.com/RomanCohort/confluencia
- 邮箱：circ-casp@example.com
- 微信群：扫码加入（海报中提供）

---

## 附录：评估脚本示例

```python
import numpy as np

def evaluate_prediction(pred_coords, true_coords, pairs_pred, pairs_true):
    """计算预测评分。"""

    N = len(pred_coords)

    # T1: RMSD
    rmsd = np.sqrt(np.mean(np.sum((pred_coords - true_coords) ** 2, axis=1)))
    t1_score = max(0, 100 - 3.33 * rmsd)  # RMSD=30Å 时为0分

    # T2: BSJ closure
    bsj_pred = np.linalg.norm(pred_coords[0] - pred_coords[-1])
    bsj_true = np.linalg.norm(true_coords[0] - true_coords[-1])
    bsj_error = abs(bsj_pred - bsj_true)
    t2_score = max(0, 100 - 20 * bsj_error)

    # T3: Bond consistency
    bond_pred = np.linalg.norm(pred_coords[1:] - pred_coords[:-1], axis=1)
    bond_true = np.linalg.norm(true_coords[1:] - true_coords[:-1], axis=1)
    bond_errors = np.abs(bond_pred - bond_true)
    t3_score = np.mean(np.maximum(0, 100 - 20 * bond_errors))

    # T4: Pair F1
    # ... pair prediction evaluation

    # Total
    weights = [0.4, 0.2, 0.15, 0.15, 0.1]
    total = weights[0]*t1_score + weights[1]*t2_score + weights[2]*t3_score

    return {
        'rmsd': rmsd,
        'bsj_error': bsj_error,
        'bond_error': np.mean(bond_errors),
        't1': t1_score,
        't2': t2_score,
        't3': t3_score,
        'total': total,
    }
```

---

**Circ-CASP 组织委员会**
吉林大学 XX 学院
2026 年 X 月