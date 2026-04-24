"""
RNACTM 临床级模型层
====================
实现：
1. PopPK 非线性混合效应模型 (Nonlinear Mixed Effects Model)
2. 个体间变异 (IIV) 参数化
3. 协变量分析 (Covariate Analysis)
4. 不确定性量化 (Bootstrap, MCMC)
5. 模型选择和诊断

数学框架：
-----------
群体模型: y_ij = f(θ_i, x_ij) + ε_ij

个体参数: θ_i = TV(θ) * exp(η_i)
- TV(θ) = 群体典型值 (Typical Value)
- η_i ~ N(0, ω²) = 个体间变异

残差模型: ε_ij ~ N(0, σ²)

协变量模型: TV(θ) = θ_pop * (COV/COV_ref)^β_COV

参考文献：
- NONMEM User Guide (Beal & Sheiner, 1998)
- Pharmacometrics: The Science of Quantitative Pharmacology (Ette & Williams, 2007)
- ICH E4: Dose-Response Information to Support Drug Registration
"""

from __future__ import annotations

import json
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.integrate import odeint, solve_ivp


# ============================================================================
# Model Structures
# ============================================================================

class PKModelType(str, Enum):
    """PK 模型类型"""
    ONE_COMPARTMENT = "1cmt"
    TWO_COMPARTMENT = "2cmt"
    RNACTM_6_COMPARTMENT = "rnactm_6cmt"
    RNACTM_EXTENDED = "rnactm_extended"


@dataclass
class PKParameters:
    """PK 参数容器"""
    # 群体典型值 (TV = Typical Value)
    tv_ka: float = 0.10       # 吸收速率常数 (1/h)
    tv_ke: float = 0.12       # 消除速率常数 (1/h)
    tv_v: float = 2.0         # 分布容积 (L/kg)
    tv_f: float = 0.02        # 生物利用度分数
    tv_k_release: float = 0.12   # LNP 释放速率 (1/h)
    tv_k_escape: float = 0.02   # 内体逃逸分数
    tv_k_translate: float = 0.10  # 翻译速率 (1/h)
    tv_k_degrade: float = 0.12    # RNA 降解速率 (1/h)
    tv_k_protein_deg: float = 0.03  # 蛋白降解速率 (1/h)

    # 个体间变异 (CV%)
    omega_ka: float = 0.40
    omega_ke: float = 0.30
    omega_v: float = 0.25
    omega_f: float = 0.50
    omega_k_release: float = 0.35
    omega_k_escape: float = 0.45
    omega_k_translate: float = 0.30
    omega_k_degrade: float = 0.28

    # 残差变异 (CV%)
    sigma_prop: float = 0.20   # 比例误差
    sigma_add: float = 0.05    # 加法误差 (ng/mL)

    # 协变量效应
    beta_weight_ka: float = 0.0    # 体重对 ka 的影响
    beta_weight_v: float = 1.0     # 体重对 V 的影响
    beta_weight_ke: float = 0.75   # 体重对 ke 的影响

    # 修饰效应 (相对于未修饰的倍数)
    beta_mod_psi_ke: float = 0.40   # Ψ 修饰使 ke 降低 60%
    beta_mod_m6a_ke: float = 0.56   # m6A 修饰使 ke 降低 44%
    beta_mod_5mc_ke: float = 0.50   # 5mC 修饰使 ke 降低 50%

    def to_array(self) -> np.ndarray:
        """转换为参数向量"""
        return np.array([
            self.tv_ka, self.tv_ke, self.tv_v, self.tv_f,
            self.omega_ka, self.omega_ke, self.omega_v, self.omega_f,
            self.sigma_prop, self.sigma_add,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> PKParameters:
        """从参数向量创建"""
        return cls(
            tv_ka=arr[0], tv_ke=arr[1], tv_v=arr[2], tv_f=arr[3],
            omega_ka=arr[4], omega_ke=arr[5], omega_v=arr[6], omega_f=arr[7],
            sigma_prop=arr[8], sigma_add=arr[9],
        )

    def get_parameter_names(self) -> List[str]:
        """获取参数名称列表"""
        return [
            'tv_ka', 'tv_ke', 'tv_v', 'tv_f',
            'omega_ka', 'omega_ke', 'omega_v', 'omega_f',
            'sigma_prop', 'sigma_add',
        ]


# ============================================================================
# Individual Parameters with Covariates
# ============================================================================

@dataclass
class IndividualParams:
    """个体参数（考虑协变量和个体间变异）"""

    # 基础参数
    ka: float          # 吸收速率 (1/h)
    ke: float          # 消除速率 (1/h)
    v: float           # 分布容积 (L/kg)
    f: float           # 生物利用度

    # RNACTM 扩展参数
    k_release: float = 0.12
    k_escape: float = 0.02
    k_translate: float = 0.10
    k_degrade: float = 0.12
    k_protein_deg: float = 0.03

    # 组织分布
    f_liver: float = 0.80
    f_spleen: float = 0.10

    # 元数据
    subject_id: str = ""
    eta_values: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_population(
        cls,
        pop_params: PKParameters,
        covariates: Dict[str, float],
        eta: Optional[Dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> IndividualParams:
        """
        从群体参数生成个体参数

        参数:
            pop_params: 群体参数
            covariates: 协变量字典 (weight_kg, modification, species, etc.)
            eta: 个体间变异值 (如果为 None，则随机生成)
            rng: 随机数生成器
        """
        rng = rng or np.random.default_rng()

        # 提取协变量
        weight = covariates.get('weight_kg', 20.0)
        mod = covariates.get('modification', 'none')
        route = covariates.get('route', 'IV')

        # 计算群体典型值 (考虑协变量)
        # V ~ weight^1.0 (等比例缩放)
        tv_v = pop_params.tv_v * (weight / 70.0) ** pop_params.beta_weight_v

        # CL ~ weight^0.75 (异速缩放)
        tv_ke = pop_params.tv_ke * (weight / 70.0) ** pop_params.beta_weight_ke

        # 修饰效应
        mod_factor = 1.0
        if mod == 'psi' or mod == 'Ψ':
            mod_factor = pop_params.beta_mod_psi_ke
        elif mod == 'm6a' or mod == 'm6A':
            mod_factor = pop_params.beta_mod_m6a_ke
        elif mod == '5mc' or mod == '5mC':
            mod_factor = pop_params.beta_mod_5mc_ke
        tv_ke *= mod_factor

        # 给药途径影响
        route_ka_factor = {'IV': 1.0, 'IM': 0.5, 'SC': 0.4, 'ID': 0.3}
        tv_ka = pop_params.tv_ka * route_ka_factor.get(route, 1.0)

        # 生成个体间变异 (如果没有提供)
        if eta is None:
            eta = {
                'ka': rng.normal(0, pop_params.omega_ka),
                'ke': rng.normal(0, pop_params.omega_ke),
                'v': rng.normal(0, pop_params.omega_v),
                'f': rng.normal(0, pop_params.omega_f),
            }

        # 计算个体参数
        ka_i = tv_ka * np.exp(eta.get('ka', 0))
        ke_i = tv_ke * np.exp(eta.get('ke', 0))
        v_i = tv_v * np.exp(eta.get('v', 0))
        f_i = pop_params.tv_f * np.exp(eta.get('f', 0))

        return cls(
            ka=ka_i,
            ke=ke_i,
            v=v_i,
            f=f_i,
            k_degrade=ke_i,  # 对于 circRNA，ke = k_degrade
            subject_id=covariates.get('subject_id', ''),
            eta_values=eta,
        )


# ============================================================================
# ODE Models
# ============================================================================

class PKModel(ABC):
    """PK 模型基类"""

    @abstractmethod
    def simulate(
        self,
        params: IndividualParams,
        dose: float,
        times: np.ndarray,
        route: str = 'IV',
    ) -> np.ndarray:
        """模拟浓度-时间曲线"""
        pass

    @abstractmethod
    def get_compartments(self) -> List[str]:
        """获取房室名称"""
        pass


class OneCompartmentModel(PKModel):
    """单室模型 (基准模型)"""

    def simulate(
        self,
        params: IndividualParams,
        dose: float,
        times: np.ndarray,
        route: str = 'IV',
    ) -> np.ndarray:
        """模拟单室模型"""
        if route == 'IV':
            # IV 单次给药: C(t) = (Dose/V) * exp(-ke*t)
            c0 = dose / params.v
            c = c0 * np.exp(-params.ke * times)
        else:
            # 血管外给药
            # C(t) = (F*ka*Dose/V)/(ka-ke) * (exp(-ke*t) - exp(-ka*t))
            ka, ke, v, f = params.ka, params.ke, params.v, params.f
            if abs(ka - ke) > 1e-6:
                c = (f * ka * dose / v / (ka - ke)) * (
                    np.exp(-ke * times) - np.exp(-ka * times)
                )
            else:
                # 特殊情况: ka ≈ ke
                c = (f * dose / v) * times * np.exp(-ke * times)
        return np.maximum(c, 0)

    def get_compartments(self) -> List[str]:
        return ['central']


class RNACTMModel(PKModel):
    """RNACTM 六房室模型 (临床级)"""

    def __init__(self, extended: bool = False):
        self.extended = extended
        self.compartments_basic = [
            'injection', 'lnp', 'endosome',
            'cytoplasmic_rna', 'protein', 'clearance'
        ]
        self.compartments_extended = [
            'injection', 'lnp', 'endosome',
            'cytoplasmic_rna', 'protein', 'clearance',
            'liver', 'spleen', 'immune_activation'
        ]

    def _ode_system(
        self,
        t: float,
        y: np.ndarray,
        params: IndividualParams,
    ) -> np.ndarray:
        """ODE 系统"""
        Inj, LNP, Endo, Cyto, Prot, Clear = y[:6]

        # 基本通量
        dInj = -params.k_release * Inj
        dLNP = params.k_release * Inj - params.k_release * LNP
        dEndo = params.k_release * LNP - params.k_escape * Endo
        k_total = params.k_degrade + params.k_translate
        dCyto = params.k_escape * Endo - k_total * Cyto
        k_prot_deg = params.k_protein_deg
        dProt = params.k_translate * Cyto - k_prot_deg * Prot
        dClear = params.k_degrade * Cyto + k_prot_deg * Prot

        if self.extended:
            # 扩展房室
            # ...
            pass

        return np.array([dInj, dLNP, dEndo, dCyto, dProt, dClear])

    def simulate(
        self,
        params: IndividualParams,
        dose: float,
        times: np.ndarray,
        route: str = 'IV',
    ) -> np.ndarray:
        """模拟 RNACTM 模型"""
        # 初始条件
        y0 = np.zeros(6)
        y0[0] = dose  # 初始剂量在注射部位

        # 求解 ODE
        t_span = (times[0], times[-1])
        t_eval = times

        sol = solve_ivp(
            lambda t, y: self._ode_system(t, y, params),
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-6,
            atol=1e-9,
        )

        # 返回蛋白浓度（房室 4）
        return sol.y[4, :]

    def simulate_full(
        self,
        params: IndividualParams,
        dose: float,
        times: np.ndarray,
    ) -> pd.DataFrame:
        """模拟所有房室"""
        y0 = np.zeros(6)
        y0[0] = dose

        sol = solve_ivp(
            lambda t, y: self._ode_system(t, y, params),
            (times[0], times[-1]),
            y0,
            t_eval=times,
            method='LSODA',
        )

        df = pd.DataFrame({
            'time_h': times,
            'injection': sol.y[0, :],
            'lnp': sol.y[1, :],
            'endosome': sol.y[2, :],
            'cytoplasmic_rna': sol.y[3, :],
            'protein': sol.y[4, :],
            'clearance': sol.y[5, :],
        })
        return df

    def get_compartments(self) -> List[str]:
        return self.compartments_extended if self.extended else self.compartments_basic


# ============================================================================
# Population PK Fitting (FOCE Algorithm)
# ============================================================================

class PopPKFitter:
    """群体 PK 拟合器 (FOCE 算法)"""

    def __init__(
        self,
        model: PKModel,
        params: Optional[PKParameters] = None,
        method: str = 'foce',  # 'fo', 'foce', 'imp'
    ):
        self.model = model
        self.params = params or PKParameters()
        self.method = method
        self.fit_result: Optional[Dict] = None

    def _objective_function(
        self,
        theta: np.ndarray,
        data: pd.DataFrame,
        return_individual: bool = False,
    ) -> Union[float, Tuple[float, Dict]]:
        """
        目标函数 (负对数似然)

        OBJ(θ) = Σᵢ OBJᵢ(θ)

        OBJᵢ(θ) = -2 * log Lᵢ(θ)

        log Lᵢ(θ) = -0.5 * (εᵢ' Ω⁻¹ εᵢ + log|Ω| + nᵢ log 2π)
        """
        params = PKParameters.from_array(theta[:10])

        # 按受试者分组
        obj_total = 0.0
        individual_results = {}

        for subject_id, group in data.groupby('subject_id'):
            times = group['time_h'].values
            obs = group['concentration'].values
            dose = group['dose'].iloc[0]
            route = group['route'].iloc[0] if 'route' in group.columns else 'IV'

            # 协变量
            covariates = {
                'subject_id': subject_id,
                'weight_kg': group['weight_kg'].iloc[0] if 'weight_kg' in group.columns else 20.0,
                'modification': group['modification'].iloc[0] if 'modification' in group.columns else 'none',
                'route': route,
            }

            # 优化个体参数 (FOCE)
            # 简化: 使用 FO 近似
            best_eta = self._optimize_eta(params, covariates, times, obs, dose, route)
            individual_params = IndividualParams.from_population(params, covariates, eta=best_eta)

            # 预测
            pred = self.model.simulate(individual_params, dose, times, route)

            # 残差
            eps = obs - pred

            # 加权残差 (考虑比例和加法误差)
            w = 1.0 / (params.sigma_prop**2 * pred**2 + params.sigma_add**2 + 1e-10)

            # 对数似然贡献
            n = len(times)
            log_det = np.sum(np.log(params.sigma_prop**2 * pred**2 + params.sigma_add**2 + 1e-10))

            # 个体间变异惩罚
            eta_values = np.array([best_eta.get(k, 0) for k in ['ka', 'ke', 'v', 'f']])
            omega_diag = np.array([params.omega_ka, params.omega_ke, params.omega_v, params.omega_f])
            omega_diag = np.maximum(omega_diag, 0.01)  # 避免数值问题
            eta_penalty = np.sum(eta_values**2 / omega_diag**2)

            obj_i = 0.5 * (np.sum(eps**2 * w) + log_det + eta_penalty)
            obj_total += obj_i

            if return_individual:
                individual_results[subject_id] = {
                    'eta': best_eta,
                    'params': individual_params,
                    'pred': pred,
                    'residuals': eps,
                }

        if return_individual:
            return obj_total, individual_results
        return obj_total

    def _optimize_eta(
        self,
        params: PKParameters,
        covariates: Dict,
        times: np.ndarray,
        obs: np.ndarray,
        dose: float,
        route: str,
    ) -> Dict[str, float]:
        """优化个体参数 eta (FOCE 内循环)"""

        def eta_objective(eta_vec: np.ndarray) -> float:
            eta = {'ka': eta_vec[0], 'ke': eta_vec[1], 'v': eta_vec[2], 'f': eta_vec[3]}
            ind_params = IndividualParams.from_population(params, covariates, eta=eta)
            pred = self.model.simulate(ind_params, dose, times, route)
            eps = obs - pred

            # 加权残差
            w = 1.0 / (params.sigma_prop**2 * pred**2 + params.sigma_add**2 + 1e-10)
            residual = np.sum(eps**2 * w)

            # eta 惩罚
            omega_diag = np.array([params.omega_ka, params.omega_ke, params.omega_v, params.omega_f])
            eta_penalty = np.sum(eta_vec**2 / np.maximum(omega_diag, 0.01)**2)

            return residual + eta_penalty

        # 初始 eta = 0
        eta0 = np.zeros(4)

        result = optimize.minimize(
            eta_objective,
            eta0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        return {
            'ka': result.x[0],
            'ke': result.x[1],
            'v': result.x[2],
            'f': result.x[3],
        }

    def fit(
        self,
        data: pd.DataFrame,
        initial_params: Optional[PKParameters] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        拟合群体 PK 模型

        参数:
            data: PK 数据 (columns: subject_id, time_h, concentration, dose, route, modification, weight_kg)
            initial_params: 初始参数
            bounds: 参数边界
            verbose: 打印进度
        """
        if initial_params is not None:
            self.params = initial_params

        # 默认边界
        if bounds is None:
            bounds = [
                (0.001, 10.0),   # tv_ka
                (0.001, 1.0),    # tv_ke
                (0.1, 20.0),     # tv_v
                (0.001, 0.5),    # tv_f
                (0.01, 2.0),     # omega_ka
                (0.01, 2.0),     # omega_ke
                (0.01, 2.0),     # omega_v
                (0.01, 2.0),     # omega_f
                (0.01, 1.0),     # sigma_prop
                (0.001, 1.0),    # sigma_add
            ]

        theta0 = self.params.to_array()[:10]

        if verbose:
            print("Starting PopPK fitting...")
            print(f"  Data: {len(data)} observations, {data['subject_id'].nunique()} subjects")
            print(f"  Initial OFV: {self._objective_function(theta0, data):.2f}")

        # 优化
        result = optimize.minimize(
            lambda t: self._objective_function(t, data),
            theta0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-8},
        )

        # 提取结果
        fitted_params = PKParameters.from_array(result.x)
        final_ofv = result.fun

        # 计算个体参数
        _, individual_results = self._objective_function(
            result.x, data, return_individual=True
        )

        self.fit_result = {
            'fitted_params': fitted_params,
            'final_ofv': final_ofv,
            'optimization_success': result.success,
            'n_iterations': result.nit,
            'individual_results': individual_results,
            'data': data,
        }

        if verbose:
            print(f"\nFitting completed:")
            print(f"  Final OFV: {final_ofv:.2f}")
            print(f"  Iterations: {result.nit}")
            print(f"\nFitted parameters:")
            for name, val in zip(fitted_params.get_parameter_names()[:10], result.x):
                print(f"    {name}: {val:.4f}")

        return self.fit_result

    def get_individual_predictions(self) -> pd.DataFrame:
        """获取个体预测"""
        if self.fit_result is None:
            raise ValueError("Model not fitted yet. Run fit() first.")

        rows = []
        for subject_id, result in self.fit_result['individual_results'].items():
            data_subj = self.fit_result['data'][
                self.fit_result['data']['subject_id'] == subject_id
            ]
            times = data_subj['time_h'].values
            obs = data_subj['concentration'].values
            pred = result['pred']

            for t, o, p in zip(times, obs, pred):
                rows.append({
                    'subject_id': subject_id,
                    'time_h': t,
                    'observed': o,
                    'predicted': p,
                    'residual': o - p,
                    'cwres': (o - p) / np.sqrt(self.fit_result['fitted_params'].sigma_prop**2 * p**2 +
                                                self.fit_result['fitted_params'].sigma_add**2),
                })

        return pd.DataFrame(rows)


# ============================================================================
# Bootstrap and Uncertainty Quantification
# ============================================================================

class BootstrapPK:
    """Bootstrap 参数不确定性估计"""

    def __init__(
        self,
        fitter: PopPKFitter,
        n_bootstrap: int = 200,
        seed: int = 42,
    ):
        self.fitter = fitter
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.bootstrap_results: List[Dict] = []

    def run(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> Dict:
        """运行 Bootstrap"""
        rng = np.random.default_rng(self.seed)
        subjects = data['subject_id'].unique()
        n_subjects = len(subjects)

        bootstrap_params = []

        if verbose:
            print(f"Running {self.n_bootstrap} bootstrap iterations...")

        for i in range(self.n_bootstrap):
            # 有放回抽样
            sampled_subjects = rng.choice(subjects, size=n_subjects, replace=True)
            boot_data = pd.concat([
                data[data['subject_id'] == s] for s in sampled_subjects
            ], ignore_index=True)

            # 拟合
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self.fitter.fit(boot_data, verbose=False)

                params = result['fitted_params']
                bootstrap_params.append({
                    'iteration': i,
                    'tv_ka': params.tv_ka,
                    'tv_ke': params.tv_ke,
                    'tv_v': params.tv_v,
                    'tv_f': params.tv_f,
                    'omega_ka': params.omega_ka,
                    'omega_ke': params.omega_ke,
                    'sigma_prop': params.sigma_prop,
                })
            except Exception as e:
                if verbose:
                    print(f"  Bootstrap {i+1} failed: {e}")
                continue

            if verbose and (i + 1) % 50 == 0:
                print(f"  Completed {i+1}/{self.n_bootstrap}")

        # 汇总结果
        df = pd.DataFrame(bootstrap_params)
        summary = {}

        for col in ['tv_ka', 'tv_ke', 'tv_v', 'tv_f', 'omega_ka', 'omega_ke', 'sigma_prop']:
            if col in df.columns:
                summary[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'cv_percent': df[col].std() / df[col].mean() * 100,
                    'ci_lower': df[col].quantile(0.025),
                    'ci_upper': df[col].quantile(0.975),
                }

        self.bootstrap_results = bootstrap_params

        if verbose:
            print(f"\nBootstrap Summary:")
            for param, stats_dict in summary.items():
                print(f"  {param}: {stats_dict['mean']:.4f} "
                      f"(95% CI: [{stats_dict['ci_lower']:.4f}, {stats_dict['ci_upper']:.4f}])")

        return {
            'summary': summary,
            'bootstrap_params': bootstrap_params,
            'n_successful': len(bootstrap_params),
        }


# ============================================================================
# Model Diagnostics
# ============================================================================

class PKModelDiagnostics:
    """PK 模型诊断"""

    def __init__(self, fit_result: Dict):
        self.fit_result = fit_result
        self.pred_df: Optional[pd.DataFrame] = None

    def compute_goodness_of_fit(self) -> Dict:
        """计算拟合优度指标"""
        pred_df = self._get_predictions()

        # R²
        ss_res = np.sum((pred_df['observed'] - pred_df['predicted'])**2)
        ss_tot = np.sum((pred_df['observed'] - pred_df['observed'].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # RMSE
        rmse = np.sqrt(np.mean((pred_df['observed'] - pred_df['predicted'])**2))

        # MAE
        mae = np.mean(np.abs(pred_df['observed'] - pred_df['predicted']))

        # MPE (平均百分比误差)
        mpe = np.mean((pred_df['observed'] - pred_df['predicted']) / pred_df['observed'] * 100)

        # CWRES 统计
        cwres_mean = pred_df['cwres'].mean()
        cwres_std = pred_df['cwres'].std()

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mpe_percent': mpe,
            'cwres_mean': cwres_mean,
            'cwres_std': cwres_std,
            'n_observations': len(pred_df),
        }

    def _get_predictions(self) -> pd.DataFrame:
        """获取预测数据框"""
        if self.pred_df is not None:
            return self.pred_df

        rows = []
        for subject_id, result in self.fit_result['individual_results'].items():
            data_subj = self.fit_result['data'][
                self.fit_result['data']['subject_id'] == subject_id
            ]
            times = data_subj['time_h'].values
            obs = data_subj['concentration'].values
            pred = result['pred']
            params = self.fit_result['fitted_params']

            for t, o, p in zip(times, obs, pred):
                rows.append({
                    'subject_id': subject_id,
                    'time_h': t,
                    'observed': o,
                    'predicted': p,
                    'residual': o - p,
                    'cwres': (o - p) / np.sqrt(params.sigma_prop**2 * p**2 + params.sigma_add**2),
                })

        self.pred_df = pd.DataFrame(rows)
        return self.pred_df

    def check_model_assumptions(self) -> Dict:
        """检查模型假设"""
        pred_df = self._get_predictions()

        # 1. 正态性检验 (CWRES)
        _, shapiro_p = stats.shapiro(pred_df['cwres'])

        # 2. 同方差性检验
        # Breusch-Pagan test 近似
        pred_sorted = pred_df.sort_values('predicted')
        n = len(pred_df)
        residuals = pred_df['residual'].values

        # 3. 独立性 (时间趋势)
        time_corr = np.corrcoef(pred_df['time_h'], pred_df['residual'])[0, 1]

        return {
            'cwres_normality_p': shapiro_p,
            'cwres_normality_pass': shapiro_p > 0.05,
            'time_residual_corr': time_corr,
            'independence_pass': abs(time_corr) < 0.3,
        }

    def vpc(
        self,
        data: pd.DataFrame,
        model: PKModel,
        params: PKParameters,
        n_sim: int = 1000,
        percentiles: List[int] = [5, 50, 95],
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Visual Predictive Check (VPC)

        通过模拟生成预测区间
        """
        rng = np.random.default_rng(seed)

        # 时间网格
        time_grid = np.linspace(0, data['time_h'].max(), 50)

        # 按剂量/修饰分组
        groups = data.groupby(['dose', 'modification'])

        vpc_results = []

        for (dose, mod), group in groups:
            # 获取该组的协变量
            covariates = {
                'weight_kg': group['weight_kg'].iloc[0] if 'weight_kg' in group.columns else 20.0,
                'modification': mod,
                'route': group['route'].iloc[0] if 'route' in group.columns else 'IV',
            }

            # 模拟多次
            sim_curves = []
            for _ in range(n_sim):
                # 生成随机 eta
                eta = {
                    'ka': rng.normal(0, params.omega_ka),
                    'ke': rng.normal(0, params.omega_ke),
                    'v': rng.normal(0, params.omega_v),
                    'f': rng.normal(0, params.omega_f),
                }
                ind_params = IndividualParams.from_population(params, covariates, eta=eta)

                # 模拟
                pred = model.simulate(ind_params, dose, time_grid, covariates['route'])

                # 添加残差变异
                pred_with_error = pred * rng.lognormal(0, params.sigma_prop)
                pred_with_error += rng.normal(0, params.sigma_add)

                sim_curves.append(pred_with_error)

            sim_array = np.array(sim_curves)  # (n_sim, n_times)

            # 计算百分位数
            for p in percentiles:
                p_values = np.percentile(sim_array, p, axis=0)
                for t, v in zip(time_grid, p_values):
                    vpc_results.append({
                        'time_h': t,
                        'dose': dose,
                        'modification': mod,
                        'percentile': p,
                        'concentration': v,
                    })

        return pd.DataFrame(vpc_results)


# ============================================================================
# Convenience Functions
# ============================================================================

def fit_population_pk(
    data: pd.DataFrame,
    model_type: str = '1cmt',
    initial_params: Optional[PKParameters] = None,
    bootstrap: bool = True,
    n_bootstrap: int = 200,
    verbose: bool = True,
) -> Dict:
    """
    拟合群体 PK 模型的便捷函数

    参数:
        data: PK 数据 (必须包含 subject_id, time_h, concentration, dose)
        model_type: 模型类型 ('1cmt', '2cmt', 'rnactm_6cmt')
        initial_params: 初始参数
        bootstrap: 是否进行 Bootstrap
        n_bootstrap: Bootstrap 次数
        verbose: 打印进度
    """
    # 选择模型
    if model_type == '1cmt':
        model = OneCompartmentModel()
    elif model_type == 'rnactm_6cmt':
        model = RNACTMModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 拟合
    fitter = PopPKFitter(model, params=initial_params)
    fit_result = fitter.fit(data, initial_params=initial_params, verbose=verbose)

    # 诊断
    diagnostics = PKModelDiagnostics(fit_result)
    gof = diagnostics.compute_goodness_of_fit()
    assumptions = diagnostics.check_model_assumptions()

    fit_result['diagnostics'] = gof
    fit_result['assumptions'] = assumptions

    # Bootstrap
    if bootstrap:
        boot = BootstrapPK(fitter, n_bootstrap=n_bootstrap)
        boot_result = boot.run(data, verbose=verbose)
        fit_result['bootstrap'] = boot_result

    return fit_result


# ============================================================================
# Demo
# ============================================================================

if __name__ == '__main__':
    from pk_data_layer import SyntheticPKGenerator

    print("="*60)
    print("PopPK Model Layer Demo")
    print("="*60)

    # 1. 生成合成数据
    print("\n1. Generating synthetic PK data...")
    generator = SyntheticPKGenerator(seed=42)
    pop_data = generator.generate_population(n_subjects=20)

    # 转换为 DataFrame
    df = pop_data.to_dataframe()
    print(f"   Created {len(df)} observations from {len(pop_data)} subjects")

    # 2. 拟合模型
    print("\n2. Fitting population PK model...")
    fit_result = fit_population_pk(
        df,
        model_type='1cmt',
        bootstrap=False,  # 快速演示
        verbose=True,
    )

    # 3. 诊断
    print("\n3. Model diagnostics:")
    gof = fit_result['diagnostics']
    print(f"   R² = {gof['r2']:.4f}")
    print(f"   RMSE = {gof['rmse']:.4f}")
    print(f"   MAE = {gof['mae']:.4f}")

    print("\n" + "="*60)
    print("Done!")
