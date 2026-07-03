"""example_dynamic_confidence.py — 动态置信度使用示例。

演示如何使用 dynamic_confidence.py 替代硬编码置信度。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic_confidence import (
    compute_dynamic_confidence,
    BASE_CONFIDENCE_BY_SOURCE,
    CONFIDENCE_WEIGHTS,
    PHYSICS_THRESHOLDS,
)
import numpy as np


def example_basic_usage():
    """示例1：基本使用。"""
    print("\n" + "=" * 60)
    print("示例1：基本使用")
    print("=" * 60)

    # 单个样本
    breakdown = compute_dynamic_confidence(
        source="pdb_circularized",
        energy=350.0,        # kJ/mol
        bsj_distance=3.5,     # Å
        clash_count=1,
        rmsd_variance=0.1,
    )

    print(breakdown.summary())


def example_compare_hardcoded_vs_dynamic():
    """示例2：对比硬编码 vs 动态计算。"""
    print("\n" + "=" * 60)
    print("示例2：对比硬编码 vs 动态计算")
    print("=" * 60)

    # 硬编码版本（V1/V2）
    hardcoded = {
        "pdb_circularized": 1.0,
        "synthetic": 0.3,
    }

    # 动态计算版本
    cases = [
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

    print("\n对比结果:")
    print("-" * 60)
    print(f"{'来源':<25} {'硬编码':>8} {'动态计算':>10} {'改进幅度':>10}")
    print("-" * 60)

    for case in cases:
        # 硬编码
        hardcoded_val = hardcoded.get(case['source'], 0.5)

        # 动态计算
        dynamic_val = compute_dynamic_confidence(**case).overall

        # 改进幅度
        improvement = (dynamic_val - hardcoded_val) / hardcoded_val if hardcoded_val > 0 else 0

        print(f"{case['source']:<25} {hardcoded_val:>8.2f} {dynamic_val:>10.3f} {improvement*100:>9.1f}%")


def example_custom_weights():
    """示例3：自定义权重。"""
    print("\n" + "=" * 60)
    print("示例3：自定义权重")
    print("=" * 60)

    # 默认权重
    print("\n默认权重:")
    print(f"  来源: {CONFIDENCE_WEIGHTS['source']:.2f}")
    print(f"  能量: {CONFIDENCE_WEIGHTS['energy']:.2f}")
    print(f"  BSJ: {CONFIDENCE_WEIGHTS['bsj']:.2f}")
    print(f"  Clash: {CONFIDENCE_WEIGHTS['clash']:.2f}")
    print(f"  收敛性: {CONFIDENCE_WEIGHTS['convergence']:.2f}")

    # 自定义权重（强调物理指标）
    custom_weights = {
        'source': 0.30,      # 降低来源权重
        'energy': 0.30,      # 提高能量权重
        'bsj': 0.25,         # 提高 BSJ 权重
        'clash': 0.10,       # 保持
        'convergence': 0.05, # 保持
    }

    print("\n自定义权重（强调物理指标）：")
    for key, val in custom_weights.items():
        label = {
            'source': '来源',
            'energy': '能量',
            'bsj': 'BSJ',
            'clash': 'Clash',
            'convergence': '收敛性',
        }[key]
        print(f"  {label}: {val:.2f}")

    # 测试不同权重的影响
    case = {
        'source': 'isrnacirc',
        'energy': 500.0,
        'bsj_distance': 4.0,
        'clash_count': 3,
        'rmsd_variance': 0.2,
    }

    print("\n同一数据在不同权重下的结果:")
    print("-" * 60)

    # 默认权重
    b1 = compute_dynamic_confidence(**case, weights=CONFIDENCE_WEIGHTS)
    print(f"默认权重: {b1.overall:.3f}")

    # 自定义权重
    b2 = compute_dynamic_confidence(**case, weights=custom_weights)
    print(f"自定义权重: {b2.overall:.3f}")

    print("\n差异分析:")
    print(f"  来源评分: {b2.base_confidence:.3f} vs {b1.base_confidence:.3f}")
    print(f"  能量评分: {b2.energy_score:.3f} vs {b1.energy_score:.3f}")
    print(f"  综合得分: {b2.overall:.3f} vs {b1.overall:.3f}")


def example_threshold_sensitivity():
    """示例4：阈值敏感性分析。"""
    print("\n" + "=" * 60)
    print("示例4：阈值敏感性分析")
    print("=" * 60)

    case = {
        'source': 'synthetic',
        'energy': 750.0,
        'bsj_distance': 6.0,
        'clash_count': 8,
        'rmsd_variance': 0.4,
    }

    print("\n不同阈值下的置信度变化:")
    print("-" * 60)

    thresholds = [
        {"name": "宽松", "energy_max": 1500, "bsj_min": 2.0, "bsj_max": 8.0, "clash_max": 20},
        {"name": "默认", "energy_max": 1000, "bsj_min": 2.8, "bsj_max": 5.0, "clash_max": 10},
        {"name": "严格", "energy_max": 500, "bsj_min": 3.0, "bsj_max": 4.0, "clash_max": 5},
    ]

    for t in thresholds:
        result = compute_dynamic_confidence(
            **case,
            thresholds={
                'energy_max': t["energy_max"],
                'bsj_min': t["bsj_min"],
                'bsj_max': t["bsj_max"],
                'clash_max': t["clash_max"],
            }
        )
        print(f"{t['name']:<10} {result.overall:.3f}")


def example_batch_processing():
    """示例5：批量处理。"""
    print("\n" + "=" * 60)
    print("示例5：批量处理")
    print("=" * 60)

    # 模拟数据集
    dataset = []
    sources = ["pdb_circularized", "isrnacirc", "synthetic"]

    for i in range(10):
        dataset.append({
            'source': np.random.choice(sources),
            'energy': np.random.uniform(200, 800),
            'bsj_distance': np.random.uniform(2.5, 5.5),
            'clash_count': np.random.randint(0, 10),
            'rmsd_variance': np.random.uniform(0.05, 0.5),
        })

    # 批量计算
    from dynamic_confidence import compute_confidence_for_dataset

    confidences, breakdowns = compute_confidence_for_dataset(dataset)

    print(f"\n数据集大小: {len(dataset)}")
    print(f"平均置信度: {np.mean(confidences):.3f}")
    print(f"标准差: {np.std(confidences):.3f}")

    # 按来源分组统计
    source_groups = {}
    for b in breakdowns:
        source = b.source
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(b.overall)

    print("\n按来源分组:")
    for source, confs in source_groups.items():
        print(f"  {source:<20} {np.mean(confs):.3f} ± {np.std(confs):.3f}")

    # 质量报告
    from dynamic_confidence import generate_quality_report
    report = generate_quality_report(breakdowns)

    print("\n质量报告:")
    print(f"  低质量样本 (<0.5): {report['n_low_quality']}/{report['n_total']}")
    print(f"  高质量样本 (≥0.7): {report['n_high_quality']}/{report['n_total']}")
    print(f"  警告总数: {sum(report['warning_stats'].values())}")


if __name__ == "__main__":
    print("=" * 60)
    print("动态置信度使用示例")
    print("=" * 60)

    example_basic_usage()
    example_compare_hardcoded_vs_dynamic()
    example_custom_weights()
    example_threshold_sensitivity()
    example_batch_processing()

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)