"""dynamic_confidence.py — 动态置信度计算模块。

核心理念：
  - 置信度不再是硬编码常量，而是根据物理指标动态计算
  - 结合来源质量 + 物理约束 + 结构稳定性

计算公式：
  confidence = base_conf(source) * 0.50
             + energy_score * 0.20
             + bsj_score * 0.15
             + clash_score * 0.10
             + convergence_score * 0.05

改进对照：
  V1/V2:    硬编码 {"pdb": 1.0, "synthetic": 0.3}
  V3:       动态计算 + 物理指标 + 可验证

作者：Confluencia Team
日期：2026-07-01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════════════════════════

# 来源基础置信度（来自 DEFAULT_CONFIDENCE）
BASE_CONFIDENCE_BY_SOURCE = {
    "pdb_circularized": 0.95,       # 真实 PDB 环化
    "pdb_circularized_aug": 0.50,   # 增强后降低
    "shape_experimental": 0.85,     # SHAPE 实验验证
    "isrnacirc": 0.70,              # IsRNAcirc 物理模拟
    "isrnacirc_aug": 0.35,          # 增强后降低
    "circbase_real": 0.50,          # circBase 数据库
    "synthetic": 0.40,              # ViennaRNA 合成
    "af3_predicted": 0.80,          # AlphaFold3 预测
    "unknown": 0.30,                # 未知来源
}

# 物理指标阈值
PHYSICS_THRESHOLDS = {
    # 能量阈值（kJ/mol）
    "energy_max": 1000.0,           # 最大能量
    "energy_good": 400.0,           # 理想能量

    # BSJ 距离阈值（Å）
    "bsj_min": 2.8,                 # 最小 BSJ 距离
    "bsj_max": 5.0,                 # 最大 BSJ 距离
    "bsj_target": 3.5,              # 理想 BSJ 距离

    # Clash 阈值
    "clash_max": 10,                # 最大冲突数
    "clash_good": 2,                # 理想冲突数

    # RMSD 方差阈值
    "rmsd_var_max": 0.5,            # 最大方差
    "rmsd_var_good": 0.1,           # 理想方差

    # Bond length variance
    "bond_var_max": 2.0,            # 最大键长方差
    "bond_var_good": 0.5,           # 理想键长方差
}

# 权重配置
CONFIDENCE_WEIGHTS = {
    "source": 0.50,                 # 来源权重（最重要）
    "energy": 0.20,                 # 能量权重
    "bsj": 0.15,                    # BSJ 闭环权重
    "clash": 0.10,                  # 空间冲突权重
    "convergence": 0.05,            # 收敛性权重
}


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConfidenceBreakdown:
    """置信度分解结果（可解释）。"""

    # === 综合评分 ===
    overall: float                  # 综合置信度 [0, 1]

    # === 来源评分 ===
    source: str                     # 数据来源
    base_confidence: float          # 基础置信度

    # === 物理指标评分 ===
    energy_score: float             # 能量评分
    bsj_score: float                # BSJ 闭环评分
    clash_score: float              # 空间冲突评分
    convergence_score: float        # 收敛性评分

    # === 物理指标原始值 ===
    energy_kj: Optional[float] = None
    bsj_distance: Optional[float] = None
    clash_count: Optional[int] = None
    rmsd_variance: Optional[float] = None
    bond_variance: Optional[float] = None

    # === 警告 ===
    warnings: List[str] = field(default_factory=list)

    # === 建议 ===
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "overall": self.overall,
            "source": self.source,
            "base_confidence": self.base_confidence,
            "energy_score": self.energy_score,
            "bsj_score": self.bsj_score,
            "clash_score": self.clash_score,
            "convergence_score": self.convergence_score,
            "energy_kj": self.energy_kj,
            "bsj_distance": self.bsj_distance,
            "clash_count": self.clash_count,
            "rmsd_variance": self.rmsd_variance,
            "bond_variance": self.bond_variance,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }

    def summary(self) -> str:
        """生成摘要报告。"""
        lines = []
        lines.append("=" * 60)
        lines.append("动态置信度报告")
        lines.append("=" * 60)

        # 综合评分
        lines.append(f"\n综合置信度: {self.overall:.3f}")

        # 来源
        lines.append(f"\n来源评分:")
        lines.append(f"  来源: {self.source}")
        lines.append(f"  基础置信度: {self.base_confidence:.3f}")

        # 物理指标
        lines.append(f"\n物理指标评分:")
        lines.append(f"  能量: {self.energy_score:.3f} (能量: {self.energy_kj:.1f} kJ/mol)")
        lines.append(f"  BSJ: {self.bsj_score:.3f} (距离: {self.bsj_distance:.2f}Å)")
        lines.append(f"  Clash: {self.clash_score:.3f} (冲突数: {self.clash_count})")
        lines.append(f"  收敛: {self.convergence_score:.3f} (RMSD方差: {self.rmsd_variance:.3f})")

        # 警告
        if self.warnings:
            lines.append("\n⚠️ 警告:")
            for w in self.warnings:
                lines.append(f"  {w}")

        # 建议
        if self.recommendations:
            lines.append("\n💡 建议:")
            for r in self.recommendations:
                lines.append(f"  {r}")

        lines.append("=" * 60)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 核心计算函数
# ═══════════════════════════════════════════════════════════════

def compute_dynamic_confidence(
    source: str,
    energy: Optional[float] = None,
    bsj_distance: Optional[float] = None,
    clash_count: Optional[int] = None,
    rmsd_variance: Optional[float] = None,
    bond_variance: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> ConfidenceBreakdown:
    """动态计算置信度。

    Args:
        source: 数据来源（如 "pdb_circularized", "synthetic"）
        energy: 能量值（kJ/mol）
        bsj_distance: BSJ 闭环距离（Å）
        clash_count: 空间冲突数
        rmsd_variance: RMSD 方差（收敛性）
        bond_variance: 键长方差
        weights: 自定义权重
        thresholds: 自定义阈值

    Returns:
        ConfidenceBreakdown: 置信度分解结果
    """
    w = weights or CONFIDENCE_WEIGHTS
    t = thresholds or PHYSICS_THRESHOLDS

    # === 1. 来源评分 ===
    base_confidence = BASE_CONFIDENCE_BY_SOURCE.get(source, 0.30)
    source_score = base_confidence

    # === 2. 能量评分 ===
    if energy is not None:
        # 能量越低越好（归一化到 [0, 1]）
        energy_norm = np.clip(energy / t["energy_max"], 0.0, 1.0)
        energy_score = 1.0 - energy_norm  # 低能量 → 高评分
        energy_score = float(np.clip(energy_score, 0.0, 1.0))
    else:
        energy_score = 0.5  # 无数据 → 中等评分

    # === 3. BSJ 评分 ===
    if bsj_distance is not None:
        # BSJ 距离越接近目标越好
        if bsj_distance < t["bsj_min"] or bsj_distance > t["bsj_max"]:
            bsj_score = 0.0  # 超出范围 → 0
        else:
            # 距离目标越近越好
            dist_to_target = abs(bsj_distance - t["bsj_target"])
            tolerance = (t["bsj_max"] - t["bsj_min"]) / 2
            bsj_score = 1.0 - dist_to_target / tolerance
            bsj_score = float(np.clip(bsj_score, 0.0, 1.0))
    else:
        bsj_score = 0.5

    # === 4. Clash 评分 ===
    if clash_count is not None:
        # 冲突越少越好
        clash_norm = np.clip(clash_count / t["clash_max"], 0.0, 1.0)
        clash_score = 1.0 - clash_norm
        clash_score = float(np.clip(clash_score, 0.0, 1.0))
    else:
        clash_score = 0.5

    # === 5. 收敛性评分 ===
    if rmsd_variance is not None:
        # RMSD 方差越小越好（结构稳定）
        rmsd_norm = np.clip(rmsd_variance / t["rmsd_var_max"], 0.0, 1.0)
        convergence_score = 1.0 - rmsd_norm
        convergence_score = float(np.clip(convergence_score, 0.0, 1.0))
    elif bond_variance is not None:
        # 用键长方差替代
        bond_norm = np.clip(bond_variance / t["bond_var_max"], 0.0, 1.0)
        convergence_score = 1.0 - bond_norm
        convergence_score = float(np.clip(convergence_score, 0.0, 1.0))
    else:
        convergence_score = 0.5

    # === 6. 综合评分 ===
    overall = (
        w["source"] * source_score +
        w["energy"] * energy_score +
        w["bsj"] * bsj_score +
        w["clash"] * clash_score +
        w["convergence"] * convergence_score
    )
    overall = float(np.clip(overall, 0.0, 1.0))

    # === 7. 警告 ===
    warnings = []

    if energy is not None and energy > t["energy_good"]:
        warnings.append(f"⚠️ 能量偏高: {energy:.1f} kJ/mol (> {t['energy_good']:.1f})")

    if bsj_distance is not None:
        if bsj_distance < t["bsj_min"]:
            warnings.append(f"⚠️ BSJ 距离过小: {bsj_distance:.2f}Å (< {t['bsj_min']:.1f}Å)")
        elif bsj_distance > t["bsj_max"]:
            warnings.append(f"⚠️ BSJ 距离过大: {bsj_distance:.2f}Å (> {t['bsj_max']:.1f}Å)")

    if clash_count is not None and clash_count > t["clash_good"]:
        warnings.append(f"⚠️ 空间冲突较多: {clash_count} (> {t['clash_good']})")

    if rmsd_variance is not None and rmsd_variance > t["rmsd_var_good"]:
        warnings.append(f"⚠️ 结构收敛性差: RMSD方差 {rmsd_variance:.3f} (> {t['rmsd_var_good']:.2f})")

    if overall < 0.5:
        warnings.append(f"⚠️ 综合置信度过低: {overall:.3f}")

    # === 8. 建议 ===
    recommendations = []

    if overall < 0.3:
        recommendations.append("💡 建议：提高数据来源质量或改进生成方法")

    if energy is not None and energy > 600:
        recommendations.append("💡 建议：优化结构能量（可能存在不合理的几何约束）")

    if clash_count is not None and clash_count > 5:
        recommendations.append("💡 建议：减少空间冲突（可能需要调整坐标生成策略）")

    # === 9. 构建结果 ===
    return ConfidenceBreakdown(
        overall=overall,
        source=source,
        base_confidence=base_confidence,
        energy_score=energy_score,
        bsj_score=bsj_score,
        clash_score=clash_score,
        convergence_score=convergence_score,
        energy_kj=energy,
        bsj_distance=bsj_distance,
        clash_count=clash_count,
        rmsd_variance=rmsd_variance,
        bond_variance=bond_variance,
        warnings=warnings,
        recommendations=recommendations,
    )


# ═══════════════════════════════════════════════════════════════
# 批量计算函数
# ═══════════════════════════════════════════════════════════════

def compute_confidence_for_dataset(
    records: List[Dict],
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[ConfidenceBreakdown]]:
    """为整个数据集计算置信度。

    Args:
        records: 数据记录列表，每个记录应包含：
            - 'source': 数据来源
            - 'energy': 能量（可选）
            - 'bsj_distance': BSJ 距离（可选）
            - 'bsj_clashes': 冲突数（可选）
            - 'rmsd_variance': RMSD 方差（可选）
        weights: 自定义权重
        thresholds: 自定义阈值

    Returns:
        confidences: 置信度列表
        breakdowns: 分解报告列表
    """
    confidences = []
    breakdowns = []

    for rec in records:
        source = rec.get('source', 'unknown')
        energy = rec.get('energy')
        bsj_distance = rec.get('bsj_distance')
        clash_count = rec.get('bsj_clashes')
        rmsd_variance = rec.get('rmsd_variance')

        breakdown = compute_dynamic_confidence(
            source=source,
            energy=energy,
            bsj_distance=bsj_distance,
            clash_count=clash_count,
            rmsd_variance=rmsd_variance,
            weights=weights,
            thresholds=thresholds,
        )

        confidences.append(breakdown.overall)
        breakdowns.append(breakdown)

    return confidences, breakdowns


# ═══════════════════════════════════════════════════════════════
# 数据质量报告
# ═══════════════════════════════════════════════════════════════

def generate_quality_report(
    breakdowns: List[ConfidenceBreakdown],
) -> Dict:
    """生成数据集质量报告。

    Returns:
        {
            'mean_confidence': float,
            'std_confidence': float,
            'n_low_quality': int,
            'n_high_quality': int,
            'source_breakdown': Dict[str, float],
            'warning_stats': Dict[str, int],
        }
    """
    confidences = [b.overall for b in breakdowns]

    # 统计
    mean_conf = float(np.mean(confidences))
    std_conf = float(np.std(confidences))
    n_low = sum(1 for c in confidences if c < 0.5)
    n_high = sum(1 for c in confidences if c >= 0.7)

    # 按来源分组
    source_groups = {}
    for b in breakdowns:
        source = b.source
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(b.overall)

    source_breakdown = {
        source: float(np.mean(conf_list))
        for source, conf_list in source_groups.items()
    }

    # 警告统计
    warning_stats = {}
    for b in breakdowns:
        for w in b.warnings:
            # 提取警告类型
            warning_type = w.split(" ")[0] if w else "unknown"
            warning_stats[warning_type] = warning_stats.get(warning_type, 0) + 1

    return {
        'mean_confidence': mean_conf,
        'std_confidence': std_conf,
        'n_low_quality': n_low,
        'n_high_quality': n_high,
        'n_total': len(breakdowns),
        'source_breakdown': source_breakdown,
        'warning_stats': warning_stats,
    }


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("动态置信度计算模块测试")
    print("=" * 60)

    # 测试不同来源
    test_cases = [
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
        {
            'source': 'isrnacirc',
            'energy': 500.0,
            'bsj_distance': 4.0,
            'clash_count': 3,
            'rmsd_variance': 0.2,
        },
    ]

    print("\n单个样本测试:")
    for i, case in enumerate(test_cases):
        breakdown = compute_dynamic_confidence(**case)
        print(f"\n样本 {i+1}:")
        print(breakdown.summary())

    # 批量测试
    print("\n批量测试:")
    confidences, breakdowns = compute_confidence_for_dataset(test_cases)
    print(f"  置信度列表: {[f'{c:.3f}' for c in confidences]}")

    # 质量报告
    report = generate_quality_report(breakdowns)
    print("\n质量报告:")
    print(f"  平均置信度: {report['mean_confidence']:.3f}")
    print(f"  标准差: {report['std_confidence']:.3f}")
    print(f"  低质量样本: {report['n_low_quality']}/{report['n_total']}")
    print(f"  高质量样本: {report['n_high_quality']}/{report['n_total']}")
    print(f"  按来源分组: {report['source_breakdown']}")

    print("=" * 60)