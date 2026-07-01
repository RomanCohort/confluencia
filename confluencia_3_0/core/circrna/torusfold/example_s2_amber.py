"""
example_s2_amber.py — Scheme 2 Amber 升级版使用示例

演示如何使用 Amber RNA OL3 力场增强的 S2 求解器。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time


def example_basic_usage():
    """示例1：基本使用。"""
    print("\n" + "=" * 60)
    print("示例1：基本使用")
    print("=" * 60)

    from constraint_solver_amber import AmberEnhancedSolver, AmberSolverConfig

    # 配置
    config = AmberSolverConfig(
        n_samples=20,
        use_amber_forcefield=True,
        minimize_steps=500,
    )

    # 创建求解器
    solver = AmberEnhancedSolver(config)

    # 模拟约束集合
    class MockConstraintSet:
        def __init__(self, seq_len):
            self.seq_len = seq_len
            self.pair_constraints = []

    constraints = MockConstraintSet(seq_len=100)

    # 求解
    start = time.time()
    results = solver.solve(constraints)
    elapsed = time.time() - start

    print(f"\n求解完成：")
    print(f"  序列长度: 100 nt")
    print(f"  候选数量: {len(results)}")
    print(f"  耗时: {elapsed:.2f}s")

    if results:
        best = results[0]
        closure = np.linalg.norm(best[0] - best[-1])
        print(f"  最佳闭环: {closure:.2f}Å (目标: 5.9Å)")


def example_compare_original_vs_amber():
    """示例2：对比原始 S2 vs Amber 升级版。"""
    print("\n" + "=" * 60)
    print("示例2：对比原始 S2 vs Amber 升级版")
    print("=" * 60)

    from constraint_solver import GeometricConstraintSolver, SolverConfig
    from constraint_solver_amber import AmberEnhancedSolver, AmberSolverConfig

    class MockConstraintSet:
        def __init__(self, seq_len):
            self.seq_len = seq_len
            self.pair_constraints = [
                (10, 30, 10.6, 1.0),  # 模拟配对
                (20, 40, 10.6, 1.0),
            ]

    constraints = MockConstraintSet(seq_len=50)

    # 原始 S2
    print("\n原始 S2 (粗粒化能量)：")
    original_solver = GeometricConstraintSolver(SolverConfig(n_samples=10))
    start = time.time()
    original_results = original_solver.solve(constraints)
    original_time = time.time() - start
    print(f"  耗时: {original_time:.2f}s")

    # Amber 升级版
    print("\nAmber 升级版：")
    amber_solver = AmberEnhancedSolver(AmberSolverConfig(n_samples=10))
    start = time.time()
    amber_results = amber_solver.solve(constraints)
    amber_time = time.time() - start
    print(f"  耗时: {amber_time:.2f}s")

    # 对比
    print("\n性能对比：")
    print(f"  时间成本: {amber_time / original_time:.1f}x")
    print(f"  预期精度提升: ~3x (25Å → 8Å)")


def example_openmm_usage():
    """示例3：使用 OpenMM 精确最小化。"""
    print("\n" + "=" * 60)
    print("示例3：OpenMM 精确最小化")
    print("=" * 60)

    from constraint_solver_amber import AmberEnhancedSolver, AmberSolverConfig, HAS_OPENMM

    if not HAS_OPENMM:
        print("  OpenMM 未安装，跳过此示例")
        return

    config = AmberSolverConfig(
        use_openmm=True,
        openmm_platform="CPU",
        minimize_steps=1000,
    )

    solver = AmberEnhancedSolver(config)

    class MockConstraintSet:
        def __init__(self, seq_len):
            self.seq_len = seq_len
            self.pair_constraints = []

    constraints = MockConstraintSet(seq_len=30)

    start = time.time()
    results = solver.solve(constraints)
    elapsed = time.time() - start

    print(f"\nOpenMM 最小化完成：")
    print(f"  耗时: {elapsed:.2f}s")
    print(f"  精度: 预期 < 5Å (最高精度)")


def example_energy_components():
    """示例4：Amber 能量组成分析。"""
    print("\n" + "=" * 60)
    print("示例4：Amber 能量组成分析")
    print("=" * 60)

    from constraint_solver_amber import AMBER_RNA_OL3_PARAMS

    print("\nAmber RNA OL3 参数：")
    print(f"  键长力常数: {AMBER_RNA_OL3_PARAMS['bond_force_constants']}")
    print(f"  WC 配对距离: {AMBER_RNA_OL3_PARAMS['wc_pair']['distance']}Å")
    print(f"  WC 配对力常数: {AMBER_RNA_OL3_PARAMS['wc_pair']['force_constant']} kJ/mol/Å²")
    print(f"  范德华参数: {AMBER_RNA_OL3_PARAMS['vdw_params']}")
    print(f"  静电电荷: {AMBER_RNA_OL3_PARAMS['charges']}")
    print(f"  基堆积距离: {AMBER_RNA_OL3_PARAMS['stacking_distance']}Å")


if __name__ == "__main__":
    print("=" * 60)
    print("Scheme 2 Amber 升级版示例")
    print("=" * 60)

    example_basic_usage()
    example_compare_original_vs_amber()
    example_openmm_usage()
    example_energy_components()

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)