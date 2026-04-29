#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_real_poppk.py - 用真实数据拟合 RNACTM PopPK 模型

基于文献参数生成的 PK 数据进行参数拟合，并与文献参考值比较。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import json
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# PK 参数类
# =============================================================================

@dataclass
class PKParameters:
    """PK 参数"""
    tv_ka: float = 0.1    # 吸收速率 (1/h)
    tv_ke: float = 0.12    # 消除速率 (1/h)
    tv_v: float = 2.0      # 分布容积 (L/kg)
    tv_f: float = 0.02     # 生物利用度

    omega_ka: float = 0.3  # IIV on ka (CV)
    omega_ke: float = 0.3   # IIV on ke (CV)
    omega_v: float = 0.25  # IIV on V (CV)
    omega_f: float = 0.5   # IIV on F (CV)

    sigma_prop: float = 0.1  # 比例残差
    sigma_add: float = 0.05  # 加法残差

    def to_dict(self) -> Dict:
        return {
            'tv_ka': self.tv_ka, 'tv_ke': self.tv_ke,
            'tv_v': self.tv_v, 'tv_f': self.tv_f,
            'omega_ka': self.omega_ka, 'omega_ke': self.omega_ke,
            'omega_v': self.omega_v, 'omega_f': self.omega_f,
            'sigma_prop': self.sigma_prop, 'sigma_add': self.sigma_add,
        }


# =============================================================================
# PK 模型
# =============================================================================

class OneCompartmentModel:
    """单室模型"""

    def __init__(self):
        self.name = "OneCompartmentModel"

    def predict(
        self,
        params: PKParameters,
        times: np.ndarray,
        doses: np.ndarray,
        routes: np.ndarray,
        weights: np.ndarray,
        eta_ka: np.ndarray = None,
        eta_ke: np.ndarray = None,
        eta_v: np.ndarray = None,
        eta_f: np.ndarray = None,
    ) -> np.ndarray:
        """
        预测浓度

        Args:
            params: PK 参数
            times: 时间点
            doses: 剂量
            routes: 给药途径 (0=IV, 1=IM, 2=SC)
            weights: 体重
            eta_*: 个体偏差

        Returns:
            预测浓度数组
        """
        n = len(times)
        predictions = np.zeros(n)

        # TV 参数
        tv_ka = params.tv_ka
        tv_ke = params.tv_ke
        tv_v = params.tv_v
        tv_f = params.tv_f

        for i in range(n):
            dose = doses[i]
            t = times[i]
            route = routes[i]
            w = weights[i]

            # 个体参数
            ka = tv_ka * np.exp(eta_ka[i] if eta_ka is not None else 0)
            ke = tv_ke * np.exp(eta_ke[i] if eta_ke is not None else 0)
            v = tv_v * w * np.exp(eta_v[i] if eta_v is not None else 0)
            f = tv_f * np.exp(eta_f[i] if eta_f is not None else 0)

            if route == 0:  # IV
                # C = (Dose/V) × exp(-ke × t)
                conc = (dose * f / v) * np.exp(-ke * t) if t > 0 else (dose * f / v)
            else:
                # 血管外给药
                ka_rate = tv_ka if route == 1 else tv_ka * 0.8  # IM vs SC
                if ka_rate <= ke:
                    ka_rate = ke * 1.1

                # C = (f × ka × Dose / V) / (ka - ke) × [exp(-ke×t) - exp(-ka×t)]
                numerator = f * ka_rate * dose
                denominator = v * (ka_rate - ke)
                if denominator == 0:
                    conc = 0
                else:
                    conc = (numerator / denominator) * (np.exp(-ke * t) - np.exp(-ka_rate * t))
                    if t == 0:
                        conc = 0

            predictions[i] = max(conc, 1e-10)

        return predictions


class RNACTMModel(OneCompartmentModel):
    """RNACTM 六室模型（简化为单室用于参数估计）"""

    def __init__(self, extended: bool = False):
        super().__init__()
        self.name = "RNACTMModel"
        self.extended = extended

        # 文献参考参数
        self.REFERENCE_PARAMS = {
            'none': {'ke': 0.1155, 'half_life': 6.0, 'cv': 0.25},
            'm6A': {'ke': 0.0642, 'half_life': 10.8, 'cv': 0.22},
            'psi': {'ke': 0.0462, 'half_life': 15.0, 'cv': 0.20},
            '5mC': {'ke': 0.0555, 'half_life': 12.5, 'cv': 0.22},
            'ms2m6A': {'ke': 0.0347, 'half_life': 20.0, 'cv': 0.18},
        }


# =============================================================================
# PopPK 拟合器
# =============================================================================

class PopPKFitter:
    """群体 PK 拟合器"""

    def __init__(self, model: RNACTMModel):
        self.model = model
        self.result = None

    def objective(self, params_flat: np.ndarray, df: pd.DataFrame) -> float:
        """
        目标函数 (FOCE)

        最小化条件估计的 OFV
        """
        # 解包参数
        tv_ka, tv_ke, tv_v, tv_f = params_flat[:4]
        omega_ka, omega_ke, omega_v, omega_f = params_flat[4:8]
        sigma_prop, sigma_add = params_flat[8:10]

        # 确保参数合理
        if any(p <= 0 for p in [tv_ke, tv_v, sigma_prop]):
            return 1e10
        if any(p < 0 for p in [omega_ka, omega_ke, omega_v, omega_f]):
            return 1e10

        params = PKParameters(
            tv_ka=tv_ka, tv_ke=tv_ke, tv_v=tv_v, tv_f=tv_f,
            omega_ka=omega_ka, omega_ke=omega_ke,
            omega_v=omega_v, omega_f=omega_f,
            sigma_prop=sigma_prop, sigma_add=sigma_add,
        )

        # 获取数据
        times = df['TIME'].values
        doses = df['DOSE'].values
        routes = df['ROUTE'].map({'IV': 0, 'IM': 1, 'SC': 2}).values
        weights = df['WT'].values
        observed = df['DV'].values

        # 获取每个受试者的 ETA
        subjects = df['ID'].unique()
        n_subjects = len(subjects)

        # 初始化 ETA
        eta_ka = np.zeros(len(df))
        eta_ke = np.zeros(len(df))
        eta_v = np.zeros(len(df))

        # E步：估计 ETA
        for subj in subjects:
            mask = df['ID'] == subj
            subj_obs = observed[mask]
            subj_times = times[mask]

            if len(subj_obs) < 2:
                continue

            # 简单估计：使用残差
            pred_mean = self.model.predict(params, subj_times, doses[mask], routes[mask], weights[mask])

            # 计算残差并估计 ETA
            residual = np.log(subj_obs + 1e-10) - np.log(pred_mean + 1e-10)

            # 简单近似：eta = residual * sigma
            eta_val = np.median(residual) * omega_ke
            eta_ke[mask] = eta_val

        # 预测
        predicted = self.model.predict(params, times, doses, routes, weights, eta_ka, eta_ke, eta_v)

        # 计算加权残差
        log_obs = np.log(observed + 1e-10)
        log_pred = np.log(predicted + 1e-10)
        wres = (log_obs - log_pred) / sigma_prop

        # OFV (目标函数值)
        ofv = np.sum(wres ** 2) + n_subjects * np.log(sigma_prop ** 2)

        return ofv

    def fit(
        self,
        df: pd.DataFrame,
        initial_params: PKParameters = None,
        max_iter: int = 100,
        method: str = 'L-BFGS-B',
    ) -> 'FitResult':
        """
        拟合 PopPK 模型
        """
        if initial_params is None:
            # 默认初始值（基于文献）
            initial_params = PKParameters(
                tv_ka=2.82, tv_ke=0.12, tv_v=6.89, tv_f=0.001,
                omega_ka=0.85, omega_ke=0.78, omega_v=0.25, omega_f=0.5,
                sigma_prop=0.10, sigma_add=0.05,
            )

        # 扁平化初始参数
        x0 = np.array([
            initial_params.tv_ka, initial_params.tv_ke,
            initial_params.tv_v, initial_params.tv_f,
            initial_params.omega_ka, initial_params.omega_ke,
            initial_params.omega_v, initial_params.omega_f,
            initial_params.sigma_prop, initial_params.sigma_add,
        ])

        # 优化
        result = minimize(
            self.objective,
            x0,
            args=(df,),
            method=method,
            options={'maxiter': max_iter, 'disp': False},
        )

        # 解包结果
        final_params = PKParameters(
            tv_ka=result.x[0], tv_ke=result.x[1],
            tv_v=result.x[2], tv_f=result.x[3],
            omega_ka=result.x[4], omega_ke=result.x[5],
            omega_v=result.x[6], omega_f=result.x[7],
            sigma_prop=result.x[8], sigma_add=result.x[9],
        )

        # 计算拟合优度
        times = df['TIME'].values
        doses = df['DOSE'].values
        routes = df['ROUTE'].map({'IV': 0, 'IM': 1, 'SC': 2}).values
        weights = df['WT'].values
        observed = df['DV'].values

        predicted = self.model.predict(final_params, times, doses, routes, weights)

        # R²
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - observed.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean((observed - predicted) ** 2))

        # Pearson r
        pearson_r = np.corrcoef(observed, predicted)[0, 1]

        self.result = FitResult(
            final_params=final_params,
            r_squared=r_squared,
            rmse=rmse,
            pearson_r=pearson_r,
            ofv=result.fun,
            n_observations=len(df),
            n_subjects=df['ID'].nunique(),
            success=result.success,
        )

        return self.result


@dataclass
class FitResult:
    """拟合结果"""
    final_params: PKParameters
    r_squared: float
    rmse: float
    pearson_r: float
    ofv: float
    n_observations: int
    n_subjects: int
    success: bool


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_fit(
    df: pd.DataFrame,
    model: RNACTMModel,
    initial_params: PKParameters,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Bootstrap 参数估计

    返回参数分布和统计量
    """
    np.random.seed(seed)

    subjects = df['ID'].unique()
    param_names = ['tv_ka', 'tv_ke', 'tv_v', 'tv_f', 'omega_ka', 'omega_ke', 'omega_v', 'omega_f', 'sigma_prop']

    results = []
    successful = 0

    for i in range(n_bootstrap):
        # 重采样受试者
        sampled_subjects = np.random.choice(subjects, size=len(subjects), replace=True)
        df_boot = df[df['ID'].isin(sampled_subjects)].copy()

        # 如果采样后无数据，跳过
        if len(df_boot) < 10:
            continue

        try:
            fitter = PopPKFitter(model)
            fit_result = fitter.fit(df_boot, initial_params, max_iter=20)

            if fit_result.success:
                results.append({
                    'iteration': i,
                    **{name: getattr(fit_result.final_params, name) for name in param_names},
                })
                successful += 1
        except Exception:
            pass

    df_boot_results = pd.DataFrame(results)

    # 统计量
    stats = {}
    for name in param_names:
        if name in df_boot_results.columns:
            vals = df_boot_results[name]
            stats[name] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'median': vals.median(),
                'ci_lower': vals.quantile(0.025),
                'ci_upper': vals.quantile(0.975),
                'cv_percent': vals.std() / vals.mean() * 100 if vals.mean() != 0 else np.nan,
            }

    summary = {
        'n_successful': successful,
        'n_total': n_bootstrap,
        'success_rate': successful / n_bootstrap,
        'param_stats': stats,
    }

    return df_boot_results, summary


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("RNACTM PopPK 参数拟合")
    print("=" * 60)

    # 加载数据
    data_path = Path(__file__).parent.parent / 'data' / 'real_poppk_dataset.csv'
    df = pd.read_csv(data_path)

    print(f"\n数据加载: {len(df)} 条记录, {df['ID'].nunique()} 个受试者")

    # 初始化模型
    model = RNACTMModel()

    # 初始参数（基于文献）
    initial_params = PKParameters(
        tv_ka=2.82, tv_ke=0.12, tv_v=6.89, tv_f=0.001,
        omega_ka=0.85, omega_ke=0.78, omega_v=0.25, omega_f=0.5,
        sigma_prop=0.10, sigma_add=0.05,
    )

    # Fit
    print("\n1. Parameter fitting (FOCE)...")
    fitter = PopPKFitter(model)
    fit_result = fitter.fit(df, initial_params, max_iter=100)

    print(f"\n   Fit result:")
    print(f"   R2 = {fit_result.r_squared:.4f}")
    print(f"   RMSE = {fit_result.rmse:.4f}")
    print(f"   Pearson r = {fit_result.pearson_r:.4f}")
    print(f"   OFV = {fit_result.ofv:.2f}")
    print(f"   Success = {fit_result.success}")

    print("\n   Parameter estimates:")
    params = fit_result.final_params
    print(f"   tv_ka = {params.tv_ka:.4f} 1/h")
    print(f"   tv_ke = {params.tv_ke:.4f} 1/h (half-life {np.log(2)/params.tv_ke:.1f}h)")
    print(f"   tv_V = {params.tv_v:.4f} L/kg")
    print(f"   tv_F = {params.tv_f:.6f}")
    print(f"   omega_ke = {params.omega_ke:.4f} CV%")
    print(f"   sigma_prop = {params.sigma_prop:.4f}")

    # 2. Bootstrap (50 iterations for speed)
    print("\n2. Bootstrap (50 iterations)...")
    df_boot, boot_summary = bootstrap_fit(
        df, model, initial_params, n_bootstrap=50, seed=42
    )

    print(f"\n   Bootstrap success: {boot_summary['n_successful']}/{boot_summary['n_total']}")
    print(f"   Success rate: {boot_summary['success_rate']*100:.1f}%")

    print("\n   Parameter Bootstrap CI:")
    for name, stats in boot_summary['param_stats'].items():
        print(f"   {name:12s}: {stats['median']:.4f} [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] CV={stats['cv_percent']:.1f}%")

    # 3. Literature comparison
    print("\n3. Literature comparison:")
    print("-" * 60)

    reference_ke = {
        'none': 0.1155,
        'm6A': 0.0642,
        'psi': 0.0462,
        '5mC': 0.0555,
        'ms2m6A': 0.0347,
    }

    reference_hl = {
        'none': 6.0,
        'm6A': 10.8,
        'psi': 15.0,
        '5mC': 12.5,
        'ms2m6A': 20.0,
    }

    fitted_ke = params.tv_ke
    fitted_hl = np.log(2) / fitted_ke

    for mod, ref_k in reference_ke.items():
        ref_h = reference_hl[mod]
        error = abs(fitted_ke - ref_k) / ref_k * 100
        status = "PASS" if error < 30 else "FAIL"
        print(f"   {status} {mod:10s}: ke={fitted_ke:.4f} vs ref={ref_k:.4f} (error {error:.1f}%) | HL={fitted_hl:.1f}h vs ref={ref_h:.1f}h")

    # 4. Save results
    print("\n4. Saving results...")

    output_dir = Path(__file__).parent.parent / 'benchmarks' / 'results'
    output_dir.mkdir(exist_ok=True, parents=True)

    # JSON report
    report = {
        'model': 'RNACTM One-Compartment PopPK',
        'fit_quality': {
            'r_squared': round(fit_result.r_squared, 4),
            'rmse': round(fit_result.rmse, 4),
            'pearson_r': round(fit_result.pearson_r, 4),
            'ofv': round(fit_result.ofv, 2),
        },
        'parameters': {
            'tv_ka': round(params.tv_ka, 4),
            'tv_ke': round(params.tv_ke, 4),
            'tv_v': round(params.tv_v, 4),
            'tv_f': round(params.tv_f, 6),
            'omega_ka': round(params.omega_ka, 4),
            'omega_ke': round(params.omega_ke, 4),
            'omega_v': round(params.omega_v, 4),
            'omega_f': round(params.omega_f, 4),
            'sigma_prop': round(params.sigma_prop, 4),
        },
        'bootstrap': {
            'n_successful': boot_summary['n_successful'],
            'n_total': boot_summary['n_total'],
            'success_rate': round(boot_summary['success_rate'], 3),
            'ci_95': {name: {
                'median': round(stats['median'], 4),
                'lower': round(stats['ci_lower'], 4),
                'upper': round(stats['ci_upper'], 4),
                'cv': round(stats['cv_percent'], 1),
            } for name, stats in boot_summary['param_stats'].items()},
        },
        'literature_comparison': {
            mod: {
                'fitted_ke': round(fitted_ke, 4),
                'fitted_hl': round(fitted_hl, 2),
                'ref_ke': round(ref_k, 4),
                'ref_hl': ref_h,
                'error_ke_pct': round(abs(fitted_ke - ref_k) / ref_k * 100, 1),
            }
            for mod, ref_k in reference_ke.items()
        },
        'data_summary': {
            'n_observations': fit_result.n_observations,
            'n_subjects': fit_result.n_subjects,
            'modifications': df['MODIFICATION'].unique().tolist(),
        },
    }

    with open(output_dir / 'rnactm_poppk_fit_results.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"   Saved: {output_dir / 'rnactm_poppk_fit_results.json'}")

    # Bootstrap results CSV
    df_boot.to_csv(output_dir / 'rnactm_bootstrap_results.csv', index=False)
    print(f"   Saved: {output_dir / 'rnactm_bootstrap_results.csv'}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return fit_result, boot_summary


if __name__ == '__main__':
    fit_result, boot_summary = main()
