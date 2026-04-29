#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vpc_validation.py - VPC (Visual Predictive Check) 验证

VPC 是群体 PK 模型验证的核心方法，通过模拟数据与观测数据的分布比较来评估模型性能。

参考文献:
- Karlsson KE, et al. J Pharmacokinet Pharmacodyn. 2015
- Nguyen TH, et al. CPT Pharmacometrics Syst Pharmacol. 2017
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from scipy import stats

# =============================================================================
# VPC 核心函数
# =============================================================================

@dataclass
class VPCResult:
    """VPC 结果"""
    time_bins: np.ndarray
    observed_percentiles: Dict[str, np.ndarray]
    simulated_percentiles: Dict[str, np.ndarray]
    coverage: np.ndarray  # 90% PI 覆盖率
    n_observations: int
    n_subjects: int
    pi_percentiles: Tuple[int, int] = (5, 95)


def stratify_by_covariate(df: pd.DataFrame, covariate: str) -> Dict[str, pd.DataFrame]:
    """按协变量分层"""
    return {str(v): g for v, g in df.groupby(covariate)}


def compute_time_bins(
    times: np.ndarray,
    n_bins: int = 10,
    method: str = 'percentile'
) -> np.ndarray:
    """
    计算时间分层

    Args:
        times: 时间点数组
        n_bins: 分层数量
        method: 'percentile' (分位数) 或 'linear' (均匀)

    Returns:
        分层边界数组
    """
    if method == 'percentile':
        # 使用分位数分层
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(times, percentiles)
    else:
        # 均匀分层
        bins = np.linspace(times.min(), times.max(), n_bins + 1)

    return bins


def compute_percentiles_by_bin(
    df: pd.DataFrame,
    time_col: str,
    dv_col: str,
    bins: np.ndarray,
    percentiles: List[float] = [5, 50, 95]
) -> Dict[str, np.ndarray]:
    """
    计算每个时间分层的百分位数

    Args:
        df: 数据框
        time_col: 时间列名
        dv_col: 观测值列名
        bins: 分层边界
        percentiles: 要计算的百分位数

    Returns:
        {percentile_name: values_by_bin}
    """
    results = {f'p{p}': [] for p in percentiles}

    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]

        if i == 0:
            mask = (df[time_col] >= lower) & (df[time_col] <= upper)
        else:
            mask = (df[time_col] > lower) & (df[time_col] <= upper)

        values = df.loc[mask, dv_col].values

        if len(values) < 3:
            # 数据不足，使用 NaN
            for p in percentiles:
                results[f'p{p}'].append(np.nan)
        else:
            for p in percentiles:
                results[f'p{p}'].append(np.percentile(values, p))

    return {k: np.array(v) for k, v in results.items()}


def simulate_vpc_data(
    df: pd.DataFrame,
    params: Dict,
    n_simulations: int = 100,
    seed: int = 42
) -> List[pd.DataFrame]:
    """
    模拟 VPC 数据

    Args:
        df: 原始观测数据
        params: PK 参数 {'tv_ke', 'omega_ke', 'sigma_prop', ...}
        n_simulations: 每个时间点的模拟次数
        seed: 随机种子

    Returns:
        模拟数据列表 (每个模拟一次)
    """
    np.random.seed(seed)

    simulations = []
    subjects = df['ID'].unique()

    for sim_idx in range(n_simulations):
        sim_rows = []

        for subj in subjects:
            subj_data = df[df['ID'] == subj].copy()

            if len(subj_data) == 0:
                continue

            # 获取该受试者的观测时间点
            times = subj_data['TIME'].values
            doses = subj_data['DOSE'].values
            weights = subj_data['WT'].values

            # 个体参数 (对数正态分布)
            eta_ke = np.random.normal(0, params.get('omega_ke', 0.3))
            eta_v = np.random.normal(0, params.get('omega_v', 0.25))
            ke = params['tv_ke'] * np.exp(eta_ke)
            v = params['tv_v'] * np.exp(eta_v)

            # 模拟浓度
            for t, dose, weight in zip(times, doses, weights):
                v_actual = v * weight
                conc = (dose / v_actual) * np.exp(-ke * t)

                # 添加残差
                sigma = params.get('sigma_prop', 0.1)
                residual = np.random.normal(0, sigma)
                conc_obs = conc * np.exp(residual)

                sim_rows.append({
                    'ID': subj,
                    'TIME': t,
                    'SIM_ID': sim_idx,
                    'DV': conc_obs,
                })

        if sim_rows:
            simulations.append(pd.DataFrame(sim_rows))

    return simulations


def compute_vpc(
    df: pd.DataFrame,
    params: Dict,
    time_col: str = 'TIME',
    dv_col: str = 'DV',
    n_bins: int = 10,
    n_simulations: int = 100,
    pi_percentiles: Tuple[int, int] = (5, 95),
    seed: int = 42,
) -> VPCResult:
    """
    执行 VPC 分析

    Args:
        df: 观测数据
        params: PK 参数
        time_col: 时间列
        dv_col: 观测值列
        n_bins: 时间分层数量
        n_simulations: 模拟次数
        pi_percentiles: 预测区间百分位
        seed: 随机种子

    Returns:
        VPCResult 对象
    """
    times = df[time_col].values
    dvs = df[dv_col].values

    # 计算分层
    bins = compute_time_bins(times, n_bins=n_bins, method='percentile')

    # 观测数据百分位数
    obs_percentiles = compute_percentiles_by_bin(
        df, time_col, dv_col, bins,
        percentiles=[pi_percentiles[0], 50, pi_percentiles[1]]
    )

    # 模拟数据
    simulations = simulate_vpc_data(df, params, n_simulations=n_simulations, seed=seed)

    # 计算模拟数据的百分位数
    sim_p5 = []
    sim_p50 = []
    sim_p95 = []

    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]

        sim_values_at_bin = []
        for sim_df in simulations:
            if i == 0:
                mask = (sim_df[time_col] >= lower) & (sim_df[time_col] <= upper)
            else:
                mask = (sim_df[time_col] > lower) & (sim_df[time_col] <= upper)

            sim_values_at_bin.extend(sim_df.loc[mask, dv_col].values)

        if len(sim_values_at_bin) >= 3:
            sim_values_at_bin = np.array(sim_values_at_bin)
            sim_p5.append(np.percentile(sim_values_at_bin, pi_percentiles[0]))
            sim_p50.append(np.percentile(sim_values_at_bin, 50))
            sim_p95.append(np.percentile(sim_values_at_bin, pi_percentiles[1]))
        else:
            sim_p5.append(np.nan)
            sim_p50.append(np.nan)
            sim_p95.append(np.nan)

    simulated_percentiles = {
        f'p{pi_percentiles[0]}': np.array(sim_p5),
        'p50': np.array(sim_p50),
        f'p{pi_percentiles[1]}': np.array(sim_p95),
    }

    # 计算覆盖率 (观测值落在模拟区间内的比例)
    coverage = compute_coverage(obs_percentiles, simulated_percentiles)

    return VPCResult(
        time_bins=bins,
        observed_percentiles=obs_percentiles,
        simulated_percentiles=simulated_percentiles,
        coverage=coverage,
        n_observations=len(df),
        n_subjects=df['ID'].nunique(),
        pi_percentiles=pi_percentiles,
    )


def compute_coverage(
    obs_percentiles: Dict[str, np.ndarray],
    sim_percentiles: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    计算 90% PI 覆盖率

    观测值的中位数落在模拟的预测区间内的比例
    """
    p5_sim = sim_percentiles.get('p5', None)
    p95_sim = sim_percentiles.get('p95', None)
    p50_obs = obs_percentiles.get('p50', None)

    if p5_sim is None or p95_sim is None or p50_obs is None:
        return np.array([])

    # 检查观测中位数是否在模拟区间内
    in_interval = (p50_obs >= p5_sim) & (p50_obs <= p95_sim)
    coverage = np.mean(in_interval)

    return np.array([coverage])


def compute_npc(
    df: pd.DataFrame,
    params: Dict,
    n_simulations: int = 100,
    seed: int = 42
) -> Dict:
    """
    NPC (Numerical Predictive Check) 分析

    参考文献: Harling K. et al. PAGE. 2009
    """
    np.random.seed(seed)

    # 获取所有观测值
    times = df['TIME'].values
    dvs = df['DV'].values
    subjects = df['ID'].unique()

    # 计算每个观测点的模拟分布
    all_percentiles = []

    for t, dv in zip(times, dvs):
        sim_values = []

        for subj in subjects:
            subj_data = df[df['ID'] == subj]
            if len(subj_data) == 0:
                continue

            dose = subj_data['DOSE'].iloc[0]
            weight = subj_data['WT'].iloc[0]

            # 模拟多次
            for _ in range(n_simulations // len(subjects) + 1):
                eta_ke = np.random.normal(0, params.get('omega_ke', 0.3))
                eta_v = np.random.normal(0, params.get('omega_v', 0.25))
                ke = params['tv_ke'] * np.exp(eta_ke)
                v = params['tv_v'] * np.exp(eta_v)

                v_actual = v * weight
                conc = (dose / v_actual) * np.exp(-ke * t)

                sigma = params.get('sigma_prop', 0.1)
                residual = np.random.normal(0, sigma)
                conc_obs = conc * np.exp(residual)

                sim_values.append(conc_obs)

        if len(sim_values) >= 10:
            percentile = stats.percentileofscore(sim_values, dv)
            all_percentiles.append(percentile)

    all_percentiles = np.array(all_percentiles)

    # NPC 统计
    npc_result = {
        'observed_percentiles': all_percentiles,
        'n_observations': len(all_percentiles),
        'expected_median': 50,
        'observed_median': np.median(all_percentiles),
        'bias': np.median(all_percentiles) - 50,
        'coverage_90': np.mean((all_percentiles >= 5) & (all_percentiles <= 95)) * 100,
        'coverage_95': np.mean((all_percentiles >= 2.5) & (all_percentiles <= 97.5)) * 100,
    }

    return npc_result


def plot_vpc(
    vpc_result: VPCResult,
    output_path: str = None,
    title: str = "Visual Predictive Check",
    log_scale: bool = True
):
    """
    绘制 VPC 图

    Args:
        vpc_result: VPC 结果
        output_path: 输出路径 (可选)
        title: 图表标题
        log_scale: 是否使用对数坐标
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # 计算每个 bin 的中点时间
        bin_centers = (vpc_result.time_bins[:-1] + vpc_result.time_bins[1:]) / 2

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. VPC 主图
        ax1 = axes[0]

        p5_sim = vpc_result.simulated_percentiles.get('p5', np.array([]))
        p95_sim = vpc_result.simulated_percentiles.get('p95', np.array([]))
        p50_sim = vpc_result.simulated_percentiles.get('p50', np.array([]))
        p50_obs = vpc_result.observed_percentiles.get('p50', np.array([]))

        # 填充 90% PI 区域
        ax1.fill_between(bin_centers, p5_sim, p95_sim, alpha=0.3, color='steelblue',
                        label=f'{vpc_result.pi_percentiles[0]}-{vpc_result.pi_percentiles[1]}% PI')

        # 模拟中位数线
        ax1.plot(bin_centers, p50_sim, 'b-', linewidth=2, label='Simulated median')

        # 观测中位数点
        valid = ~np.isnan(p50_obs)
        ax1.plot(bin_centers[valid], p50_obs[valid], 'ro', markersize=8,
                label='Observed median', zorder=5)

        # 观测数据点 (散点)
        # 注意: 这里不传观测数据，使用已计算的百分位数

        ax1.set_xlabel('Time (h)', fontsize=12)
        ax1.set_ylabel('Concentration', fontsize=12)
        ax1.set_title(f'{title}\n(90% PI coverage: {vpc_result.coverage.mean()*100:.1f}%)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        if log_scale:
            ax1.set_yscale('log')

        # 2. 覆盖率柱状图
        ax2 = axes[1]

        coverage = vpc_result.coverage.mean() * 100
        expected = 90  # 90% PI 期望覆盖率

        colors = ['green' if abs(coverage - expected) < 10 else 'orange' if abs(coverage - expected) < 20 else 'red']

        bars = ax2.bar(['90% PI\nCoverage'], [coverage], color=colors[0], alpha=0.7)
        ax2.axhline(y=expected, color='black', linestyle='--', linewidth=2, label=f'Expected: {expected}%')
        ax2.axhline(y=expected - 10, color='gray', linestyle=':', linewidth=1)
        ax2.axhline(y=expected + 10, color='gray', linestyle=':', linewidth=1)

        ax2.set_ylabel('Coverage (%)', fontsize=12)
        ax2.set_title('Prediction Interval Coverage', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, [coverage]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"VPC figure saved: {output_path}")

        plt.show()

    except ImportError:
        print("Warning: matplotlib not installed, skipping plot")


def npc_pvalue(npcr: Dict) -> float:
    """
    计算 NPC p 值

    使用卡方检验比较观测百分位分布与期望分布
    """
    obs_perc = npcr['observed_percentiles']
    n = len(obs_perc)

    # 期望: 均匀分布 [0, 100]
    expected_perc = np.linspace(0, 100, n)

    # Kolmogorov-Smirnov 检验
    ks_stat, pvalue = stats.kstest(obs_perc / 100, 'uniform')

    return pvalue


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("VPC (Visual Predictive Check) Validation")
    print("=" * 60)

    # 1. 加载数据
    data_path = Path(__file__).parent.parent / 'data' / 'real_poppk_dataset.csv'
    df = pd.read_csv(data_path)

    print(f"\nData loaded: {len(df)} observations, {df['ID'].nunique()} subjects")

    # 2. 加载拟合参数
    result_path = Path(__file__).parent.parent / 'benchmarks' / 'results' / 'rnactm_poppk_fit_results.json'
    with open(result_path, 'r', encoding='utf-8') as f:
        fit_results = json.load(f)

    params = fit_results['parameters']
    print(f"Parameters loaded from fit results")

    # 3. VPC 分析
    print("\n3. VPC Analysis...")

    # 按修饰类型分层
    modifications = df['MODIFICATION'].unique()

    vpc_results_by_mod = {}
    npc_results_by_mod = {}

    for mod in modifications:
        df_mod = df[df['MODIFICATION'] == mod]

        if len(df_mod) < 30:  # 需要足够的数据
            print(f"  Skipping {mod}: insufficient data ({len(df_mod)} obs)")
            continue

        print(f"\n  VPC for {mod} ({len(df_mod)} obs)...")

        vpc_res = compute_vpc(
            df_mod,
            params,
            time_col='TIME',
            dv_col='DV',
            n_bins=8,
            n_simulations=50,
            seed=42,
        )

        vpc_results_by_mod[mod] = vpc_res

        print(f"    90% PI coverage: {vpc_res.coverage.mean()*100:.1f}%")
        print(f"    Median obs: {np.nanmean(vpc_res.observed_percentiles.get('p50', [])):.2f}")
        print(f"    Median sim: {np.nanmean(vpc_res.simulated_percentiles.get('p50', [])):.2f}")

        # NPC 分析
        npc_res = compute_npc(df_mod, params, n_simulations=50, seed=42)
        npc_results_by_mod[mod] = npc_res

        print(f"    NPC bias: {npc_res['bias']:.2f}")
        print(f"    NPC 90% coverage: {npc_res['coverage_90']:.1f}%")

    # 4. 总体 VPC
    print("\n4. Overall VPC...")

    overall_vpc = compute_vpc(
        df,
        params,
        time_col='TIME',
        dv_col='DV',
        n_bins=10,
        n_simulations=50,
        seed=42,
    )

    print(f"\n  Overall 90% PI coverage: {overall_vpc.coverage.mean()*100:.1f}%")

    # 5. 保存结果
    output_dir = Path(__file__).parent.parent / 'benchmarks' / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    # VPC 结果 JSON
    vpc_report = {
        'analysis_type': 'Visual Predictive Check (VPC)',
        'n_observations': int(overall_vpc.n_observations),
        'n_subjects': int(overall_vpc.n_subjects),
        'overall': {
            'pi_90_coverage': round(float(overall_vpc.coverage.mean()), 4),
            'pi_percentiles': list(overall_vpc.pi_percentiles),
        },
        'by_modification': {},
        'validation_status': 'PASS' if overall_vpc.coverage.mean() > 0.5 else 'FAIL',
    }

    for mod, vpc_res in vpc_results_by_mod.items():
        vpc_report['by_modification'][mod] = {
            'n_observations': int(vpc_res.n_observations),
            'n_subjects': int(vpc_res.n_subjects),
            'pi_90_coverage': round(float(vpc_res.coverage.mean()), 4),
        }

        if mod in npc_results_by_mod:
            npc_res = npc_results_by_mod[mod]
            vpc_report['by_modification'][mod]['npc'] = {
                'bias': round(npc_res['bias'], 2),
                'coverage_90': round(npc_res['coverage_90'], 1),
                'coverage_95': round(npc_res['coverage_95'], 1),
            }

    with open(output_dir / 'vpc_validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(vpc_report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_dir / 'vpc_validation_results.json'}")

    # 6. 绘图
    print("\n5. Generating VPC plot...")

    plot_output = output_dir / 'vpc_plot.png'
    plot_vpc(overall_vpc, output_path=str(plot_output), title="RNACTM VPC (All Modifications)")

    print(f"  VPC figure saved: {plot_output}")

    # 7. 验证总结
    print("\n" + "=" * 60)
    print("VPC Validation Summary")
    print("=" * 60)

    coverage = overall_vpc.coverage.mean() * 100
    status = "PASS" if coverage > 50 else "FAIL"
    print(f"\n  Status: {status}")
    print(f"  90% PI Coverage: {coverage:.1f}% (target: >50%)")

    if coverage >= 80:
        print(f"  Rating: Excellent (>80% coverage)")
    elif coverage >= 60:
        print(f"  Rating: Good (60-80% coverage)")
    elif coverage >= 50:
        print(f"  Rating: Acceptable (50-60% coverage)")
    else:
        print(f"  Rating: Poor (<50% coverage)")

    print("\n  By Modification:")
    for mod, vpc_res in vpc_results_by_mod.items():
        mod_cov = vpc_res.coverage.mean() * 100
        mod_status = "PASS" if mod_cov > 50 else "FAIL"
        print(f"    {mod_status} {mod:10s}: {mod_cov:.1f}% coverage")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return vpc_report, vpc_results_by_mod


if __name__ == '__main__':
    vpc_report, vpc_results = main()
