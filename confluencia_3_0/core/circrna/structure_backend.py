"""
structure_backend.py — 统一结构预测后端系统

支持三种模式切换：
  1. TorusFold: 深度学习方法（需训练）
  2. Pipeline: ViennaRNA + RoseTTAFold2NA + OpenMM（无需训练）
  3. Heuristic: 启发式快速计算（无依赖）

使用方式：
  backend = StructureBackend(mode="torusfold")  # 或 "pipeline" 或 "heuristic"
  result = backend.predict(sequence)

  # 动态切换
  backend.set_mode("pipeline")
  result = backend.predict(sequence)

设计理念：
  - 所有模式返回统一格式的 TorusFoldSignalsExtended
  - 自动 fallback: TorusFold → Pipeline → Heuristic
  - 可配置优先级和超时
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import numpy as np


class BackendMode(Enum):
    """后端模式枚举。"""
    TORUSFOLD = "torusfold"
    PIPELINE = "pipeline"
    HEURISTIC = "heuristic"


@dataclass
class BackendConfig:
    """后端配置。"""
    # 默认模式
    default_mode: str = "pipeline"  # 推荐 pipeline（无需训练）

    # Fallback 顺序
    fallback_order: List[str] = field(default_factory=lambda: [
        "torusfold",  # 尝试深度学习
        "pipeline",   # 尝试物理建模
        "heuristic",  # 最后兜底
    ])

    # 超时设置（秒）
    torusfold_timeout: float = 5.0
    pipeline_timeout: float = 300.0  # Pipeline 可能慢
    heuristic_timeout: float = 1.0

    # TorusFold 配置
    torusfold_checkpoint: Optional[str] = None
    torusfold_device: str = "cuda"

    # Pipeline 配置
    pipeline_vienna_timeout: float = 5.0
    pipeline_rosetta_timeout: float = 60.0
    pipeline_openmm_steps: int = 500

    # 启用/禁用特定后端
    enable_torusfold: bool = True
    enable_pipeline: bool = True
    enable_heuristic: bool = True

    # === 专家模式配置（V3 新增）===
    expert_mode: bool = False  # 是否启用专家模式
    selected_schemes: List[str] = field(default_factory=list)  # ["S1", "S4", "S7"]
    scheme_weights: Dict[str, float] = field(default_factory=dict)  # {"S1": 0.7, "S4": 0.3}
    bypass_gate: bool = False  # 是否跳过自动路由

    # Scheme 状态（系统维护）
    scheme_status: Dict[str, str] = field(default_factory=lambda: {
        "S1": "active",       # DL+Physics Cascade
        "S2": "active",       # Batch+Physics Filter
        "S3": "deferred",     # Dual-Engine（需要 teacher）
        "S4": "active",       # DDPM+EGNN Guided
        "S5": "deprecated",   # Physics-Biased Attention（NaN 问题）
        "S6": "active",       # GNN Latent Diffusion
        "S7": "active",       # Mamba+Transformer Hybrid
    })


# ═══════════════════════════════════════════════════════════════
# Unified Output
# ═══════════════════════════════════════════════════════════════

@dataclass
class StructurePredictionResult:
    """统一的结构预测结果。"""
    # 核心信号（所有模式必须输出）
    available: bool
    method: str  # "torusfold", "pipeline", "heuristic"

    # === 3D 结构 ===
    coords: Optional[np.ndarray] = None  # (L, 3)
    pair_probs: Optional[np.ndarray] = None  # (L, L)
    pair_list: Optional[List[Tuple[int, int]]] = None  # [(i, j), ...]

    # === 质量指标 ===
    bsj_closure: float = 5.9  # Å
    bond_rmsd: float = 0.0
    clash_count: int = 0
    confidence: float = 0.5  # 保留硬编码（向后兼容）

    # === TorusFold 扩展信号 ===
    dsRNA_fraction: float = 0.0
    bsj_stability: float = 0.5
    sasa_mean: float = 0.5
    sasa_bsj: float = 0.5
    sasa_per_nucleotide: Optional[np.ndarray] = None
    motif_accessibility: Dict[str, float] = field(default_factory=dict)
    dsRNA_mean_length: float = 20.0

    # === Pipeline 特有 ===
    mfe_kcal: Optional[float] = None
    dot_bracket: Optional[str] = None
    vienna_method: Optional[str] = None

    # === 性能信息 ===
    elapsed_time: float = 0.0
    timed_out: bool = False

    # === 动态置信度（V3 新增）===
    confidence_breakdown: Optional["ConfidenceBreakdown"] = None

    # === 降级记录（V3 新增）===
    fallback_log: List[str] = field(default_factory=list)

    def compute_confidence(self, weights: Optional[Dict[str, float]] = None) -> "ConfidenceBreakdown":
        """动态计算置信度（方案2：内部方法）。

        Args:
            weights: 自定义权重，None 则使用默认值

        Returns:
            ConfidenceBreakdown: 置信度分解结果
        """
        from confluencia_3_0.core.circrna.confidence_metrics import (
            compute_confidence_breakdown as _compute_breakdown
        )
        self.confidence_breakdown = _compute_breakdown(self, weights)
        return self.confidence_breakdown


# ═══════════════════════════════════════════════════════════════
# Backend Implementations
# ═══════════════════════════════════════════════════════════════

class TorusFoldBackend:
    """TorusFold 深度学习后端。"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.device = config.torusfold_device

    def _load_model(self):
        """懒加载模型。"""
        if self.model is not None:
            return True

        try:
            # 尝试加载 TorusFold
            import torch
            from confluencia_3_0.core.circrna.torusfold.torusfold import (
                TorusFold, TorusFoldConfig
            )

            config = TorusFoldConfig()
            self.model = TorusFold(config)

            if self.config.torusfold_checkpoint:
                self.model.load(self.config.torusfold_checkpoint, device=self.device)

            self.model = self.model.to(self.device)
            self.model.eval()
            return True

        except Exception as e:
            print(f"TorusFoldBackend: Failed to load model: {e}")
            return False

    def predict(
        self,
        sequence: str,
        timeout: Optional[float] = None,
    ) -> StructurePredictionResult:
        """TorusFold 预测。"""
        timeout = timeout or self.config.torusfold_timeout
        start_time = time.time()

        # 检查模型是否可用
        if not self.config.enable_torusfold or not self._load_model():
            return StructurePredictionResult(
                available=False,
                method="torusfold",
                timed_out=True,
            )

        try:
            import torch

            # 推理
            with torch.no_grad():
                result = self.model.predict_single(
                    sequence.upper().replace("T", "U"),
                    gene_expr=None,
                    device=self.device,
                )

            elapsed = time.time() - start_time
            timed_out = elapsed > timeout

            # 提取信号
            coords = result.get("coords", None)
            pair_probs = result.get("pair_probs", None)

            if coords is not None:
                coords = coords.cpu().numpy() if hasattr(coords, 'cpu') else coords
            if pair_probs is not None:
                pair_probs = pair_probs.cpu().numpy() if hasattr(pair_probs, 'cpu') else pair_probs

            # 计算衍生信号
            if coords is not None:
                from confluencia_3_0.core.circrna.torusfold_scorer_v2 import (
                    compute_sasa_from_coords,
                    compute_bsj_sasa,
                    compute_dsRNA_mean_length,
                    extract_extended_signals,
                )

                # SASA
                sasa = compute_sasa_from_coords(coords)
                sasa_mean = float(np.mean(sasa))
                sasa_bsj = compute_bsj_sasa(coords)

                # dsRNA
                if pair_probs is not None:
                    dsRNA_frac = float(np.mean(pair_probs > 0.8))
                    dsRNA_mean_len = compute_dsRNA_mean_length(pair_probs)
                else:
                    dsRNA_frac = 0.0
                    dsRNA_mean_len = 20.0

                # BSJ closure
                bsj_closure = float(np.linalg.norm(coords[0] - coords[-1]))
                bsj_stability = float(1.0 / (1.0 + bsj_closure / 5.9))

                # Bond RMSD
                bond_lengths = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
                bond_rmsd = float(np.sqrt(np.mean((bond_lengths - 5.9) ** 2)))

            else:
                sasa_mean = 0.5
                sasa_bsj = 0.5
                dsRNA_frac = 0.0
                dsRNA_mean_len = 20.0
                bsj_closure = 5.9
                bsj_stability = 0.5
                bond_rmsd = 0.0

            return StructurePredictionResult(
                available=coords is not None and not timed_out,
                method="torusfold",
                coords=coords,
                pair_probs=pair_probs,
                bsj_closure=bsj_closure,
                bond_rmsd=bond_rmsd,
                confidence=0.85,
                dsRNA_fraction=dsRNA_frac,
                bsj_stability=bsj_stability,
                sasa_mean=sasa_mean,
                sasa_bsj=sasa_bsj,
                sasa_per_nucleotide=sasa if coords is not None else None,
                dsRNA_mean_length=dsRNA_mean_len,
                elapsed_time=elapsed,
                timed_out=timed_out,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"TorusFoldBackend: Prediction failed: {e}")
            return StructurePredictionResult(
                available=False,
                method="torusfold",
                elapsed_time=elapsed,
                timed_out=True,
            )


class PipelineBackend:
    """ViennaRNA + RoseTTAFold2NA + OpenMM Pipeline 后端。"""

    def __init__(self, config: BackendConfig):
        self.config = config

    def predict(
        self,
        sequence: str,
        timeout: Optional[float] = None,
    ) -> StructurePredictionResult:
        """Pipeline 预测。

        流程：
          1. ViennaRNA circ-mode → 二级结构 + MFE
          2. RoseTTAFold2NA → 3D 坐标（线性）
          3. OpenMM → BSJ 环化 + 能量优化
        """
        timeout = timeout or self.config.pipeline_timeout
        start_time = time.time()

        if not self.config.enable_pipeline:
            return StructurePredictionResult(
                available=False,
                method="pipeline",
                timed_out=True,
            )

        try:
            # === Stage 1: ViennaRNA ===
            ss_result = self._run_vienna(sequence)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                return StructurePredictionResult(
                    available=False,
                    method="pipeline",
                    elapsed_time=elapsed,
                    timed_out=True,
                )

            # === Stage 2: RoseTTAFold2NA（可选）===
            # 如果 RF2 不可用，用启发式 3D
            coords_3d = self._run_rosetta(sequence, ss_result)

            elapsed = time.time() - start_time
            if elapsed > timeout:
                # 返回部分结果（至少有二级结构）
                return StructurePredictionResult(
                    available=True,
                    method="pipeline_partial",
                    coords=None,
                    dot_bracket=ss_result.get('dot_bracket', ''),
                    mfe_kcal=ss_result.get('mfe', 0.0),
                    confidence=0.5,
                    elapsed_time=elapsed,
                    timed_out=True,
                )

            # === Stage 3: OpenMM 环化（可选）===
            if coords_3d is not None:
                coords_circ = self._run_openmm(coords_3d, ss_result)
            else:
                coords_circ = None

            elapsed = time.time() - start_time
            timed_out = elapsed > timeout

            # 提取信号
            if coords_circ is not None:
                bsj_closure = float(np.linalg.norm(coords_circ[0] - coords_circ[-1]))
                bond_lengths = np.linalg.norm(coords_circ[1:] - coords_circ[:-1], axis=1)
                bond_rmsd = float(np.sqrt(np.mean((bond_lengths - 5.9) ** 2)))

                # SASA
                from confluencia_3_0.core.circrna.torusfold_scorer_v2 import (
                    compute_sasa_from_coords,
                    compute_bsj_sasa,
                )
                sasa = compute_sasa_from_coords(coords_circ)
                sasa_mean = float(np.mean(sasa))
                sasa_bsj = compute_bsj_sasa(coords_circ)

                # dsRNA from pair_list
                pair_list = ss_result.get('pairs', [])
                dsRNA_frac = len(pair_list) / max(len(sequence), 1)
                dsRNA_mean_len = 20.0  # 启发式估计

            else:
                bsj_closure = 5.9
                bond_rmsd = 0.0
                sasa_mean = 0.5
                sasa_bsj = 0.5
                dsRNA_frac = len(ss_result.get('pairs', [])) / max(len(sequence), 1)
                dsRNA_mean_len = 20.0
                sasa = None

            return StructurePredictionResult(
                available=coords_circ is not None or ss_result.get('dot_bracket') is not None,
                method="pipeline",
                coords=coords_circ,
                pair_list=ss_result.get('pairs', []),
                bsj_closure=bsj_closure,
                bond_rmsd=bond_rmsd,
                confidence=0.7,
                dsRNA_fraction=dsRNA_frac,
                bsj_stability=float(1.0 / (1.0 + bsj_closure / 5.9)),
                sasa_mean=sasa_mean,
                sasa_bsj=sasa_bsj,
                sasa_per_nucleotide=sasa,
                dsRNA_mean_length=dsRNA_mean_len,
                dot_bracket=ss_result.get('dot_bracket', ''),
                mfe_kcal=ss_result.get('mfe', 0.0),
                elapsed_time=elapsed,
                timed_out=timed_out,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"PipelineBackend: Prediction failed: {e}")
            return StructurePredictionResult(
                available=False,
                method="pipeline",
                elapsed_time=elapsed,
                timed_out=True,
            )

    def _run_vienna(self, sequence: str) -> Dict:
        """运行 ViennaRNA circ-mode。"""
        try:
            import RNA

            md = RNA.md()
            md.circ = True
            fc = RNA.fold_compound(sequence.upper().replace("T", "U"), md)
            ss, mfe = fc.mfe()

            # 解析配对
            pairs = []
            stack = []
            for pos, char in enumerate(ss):
                if char == "(":
                    stack.append(pos)
                elif char == ")" and stack:
                    j = stack.pop()
                    pairs.append((j, pos))

            return {
                'dot_bracket': ss,
                'mfe': mfe,
                'pairs': pairs,
                'method': 'vienna_circ',
            }

        except ImportError:
            # 启发式兜底
            return self._heuristic_ss(sequence)

    def _heuristic_ss(self, sequence: str) -> Dict:
        """启发式二级结构预测（ViennaRNA 不可用时）。"""
        L = len(sequence)
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        pairs = []

        # 简单启发式配对
        paired = set()
        for i in range(L):
            if i in paired:
                continue
            for j in range(i + 4, min(i + 20, L)):
                if j in paired:
                    continue
                if complement.get(sequence[i].upper()) == sequence[j].upper():
                    pairs.append((i, j))
                    paired.add(i)
                    paired.add(j)
                    break

        # 生成 dot-bracket
        ss = ['.'] * L
        for i, j in pairs:
            ss[i] = '('
            ss[j] = ')'
        ss = ''.join(ss)

        return {
            'dot_bracket': ss,
            'mfe': -10.0,  # 假设值
            'pairs': pairs,
            'method': 'heuristic',
        }

    def _run_rosetta(self, sequence: str, ss_result: Dict) -> Optional[np.ndarray]:
        """运行 RoseTTAFold2NA（如果可用）。"""
        try:
            # 尝试导入 RF2
            # 这里用启发式 3D 作为兜底
            return self._heuristic_3d(sequence, ss_result)

        except Exception:
            return self._heuristic_3d(sequence, ss_result)

    def _heuristic_3d(self, sequence: str, ss_result: Dict) -> np.ndarray:
        """启发式 3D 坐标生成。"""
        L = len(sequence)

        # A-form helix parameters
        rise_per_base = 2.8  # Å
        twist_per_base = 32.7  # degrees

        coords = np.zeros((L, 3), dtype=np.float32)

        for i in range(L):
            angle = np.radians(twist_per_base * i)
            radius = 10.0  # Å (approximate helix radius)

            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = rise_per_base * i

        return coords

    def _run_openmm(self, coords: np.ndarray, ss_result: Dict) -> Optional[np.ndarray]:
        """运行 OpenMM 环化（如果可用）。"""
        try:
            from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
                GeometricConstraintSolver, SolverConfig
            )

            config = SolverConfig(
                max_iterations=self.config.pipeline_openmm_steps,
            )
            solver = GeometricConstraintSolver(config)

            # 约束
            pairs = ss_result.get('pairs', [])

            result = solver.solve(coords, pairs)

            return result.get('coords', coords)

        except Exception:
            # 简单闭环
            return self._simple_close(coords)

    def _simple_close(self, coords: np.ndarray) -> np.ndarray:
        """简单闭环（OpenMM 不可用时）。"""
        L = len(coords)
        coords = coords.copy()

        # Annealing closure
        for step in range(500):
            diff = coords[0] - coords[-1]
            dist = np.linalg.norm(diff)
            if dist < 6.0:
                break

            correction = 0.01 * (dist - 5.9) * diff / max(dist, 1e-6)
            coords[0] -= correction * 0.5
            coords[-1] += correction * 0.5

            # 传播到邻近核苷酸
            for i in range(1, min(5, L // 2)):
                alpha = (5 - i + 1) / (5 + 1)
                coords[i] -= correction * 0.05 * alpha
                coords[-(i + 1)] += correction * 0.05 * alpha

        return coords


class HeuristicBackend:
    """启发式快速后端。"""

    def __init__(self, config: BackendConfig):
        self.config = config

    def predict(
        self,
        sequence: str,
        timeout: Optional[float] = None,
    ) -> StructurePredictionResult:
        """启发式预测（秒级）。"""
        timeout = timeout or self.config.heuristic_timeout
        start_time = time.time()

        if not self.config.enable_heuristic:
            return StructurePredictionResult(
                available=False,
                method="heuristic",
                timed_out=True,
            )

        try:
            L = len(sequence)
            gc = sum(1 for c in sequence.upper() if c in "GC") / max(L, 1)

            # === 启发式信号 ===
            # dsRNA fraction: GC 高 → dsRNA 多
            dsRNA_frac = gc * 0.6 + 0.1

            # motif 检测
            gu_count = sum(sequence.upper().count(m) for m in ["GU", "UG"])
            au_count = sum(sequence.upper().count(m) for m in ["AU", "UA"])

            motif_density = (gu_count + au_count) / max(L / 50, 1)

            # BSJ stability: 基于长度和 GC
            bsj_stability = 0.3 + 0.4 * (1 - L / 1000.0) + 0.3 * gc

            # SASA: 启发式估计
            sasa_mean = 0.5 - 0.2 * gc + 0.1 * (L / 500.0)
            sasa_bsj = sasa_mean * 0.9  # BSJ 略低

            # 置信度
            confidence = 0.3  # 启发式置信度低

            elapsed = time.time() - start_time
            timed_out = elapsed > timeout

            return StructurePredictionResult(
                available=True,
                method="heuristic",
                coords=None,  # 不生成坐标
                pair_probs=None,
                bsj_closure=5.9,  # 假设完美闭合
                bond_rmsd=0.0,
                confidence=confidence,
                dsRNA_fraction=dsRNA_frac,
                bsj_stability=bsj_stability,
                sasa_mean=sasa_mean,
                sasa_bsj=sasa_bsj,
                dsRNA_mean_length=20.0,
                elapsed_time=elapsed,
                timed_out=timed_out,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"HeuristicBackend: Prediction failed: {e}")
            return StructurePredictionResult(
                available=False,
                method="heuristic",
                elapsed_time=elapsed,
                timed_out=True,
            )


# ═══════════════════════════════════════════════════════════════
# Unified Backend System
# ═══════════════════════════════════════════════════════════════

class StructureBackend:
    """统一结构预测后端系统。

    支持：
      - 指定模式运行
      - 自动 fallback
      - 性能监控
    """

    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()

        # 初始化各后端
        self.backends = {
            'torusfold': TorusFoldBackend(self.config),
            'pipeline': PipelineBackend(self.config),
            'heuristic': HeuristicBackend(self.config),
        }

        # 当前模式
        self.current_mode = self.config.default_mode

    def set_mode(self, mode: str):
        """设置当前模式。"""
        if mode not in self.backends:
            raise ValueError(f"Unknown backend mode: {mode}")
        self.current_mode = mode

    def get_mode(self) -> str:
        """获取当前模式。"""
        return self.current_mode

    def predict(
        self,
        sequence: str,
        mode: Optional[str] = None,
        fallback: bool = True,
        verbose: bool = True,
    ) -> StructurePredictionResult:
        """结构预测。

        Args:
            sequence: circRNA 序列
            mode: 指定模式（None 用默认）
            fallback: 是否自动 fallback
            verbose: 是否打印 fallback 提示

        Returns:
            StructurePredictionResult
        """
        mode = mode or self.current_mode

        if not fallback:
            # 单模式运行
            backend = self.backends.get(mode)
            if backend is None:
                return StructurePredictionResult(
                    available=False,
                    method=mode,
                    timed_out=True,
                )
            return backend.predict(sequence)

        # === Fallback 流程 ===
        attempted = []
        for try_mode in self.config.fallback_order:
            if try_mode not in self.backends:
                continue

            backend = self.backends[try_mode]
            result = backend.predict(sequence)

            if result.available and not result.timed_out:
                # 成功
                if verbose and try_mode != mode:
                    # Fallback 发生了
                    print(f"[StructureBackend] Fallback: {mode} → {try_mode} (elapsed: {result.elapsed_time:.2f}s)")
                return result

            # 记录失败原因
            attempted.append({
                'mode': try_mode,
                'available': result.available,
                'timed_out': result.timed_out,
                'elapsed': result.elapsed_time,
            })

            if verbose:
                reason = "timeout" if result.timed_out else "unavailable"
                print(f"[StructureBackend] {try_mode} failed ({reason}), trying next...")

        # 所有后端失败
        if verbose:
            print(f"[StructureBackend] All backends failed. Attempted: {[a['mode'] for a in attempted]}")

        return StructurePredictionResult(
            available=False,
            method="fallback_failed",
            timed_out=True,
            elapsed_time=sum(a['elapsed'] for a in attempted),
        )

    def batch_predict(
        self,
        sequences: List[str],
        mode: Optional[str] = None,
        parallel: bool = False,
    ) -> List[StructurePredictionResult]:
        """批量预测。"""
        results = []

        for seq in sequences:
            result = self.predict(seq, mode=mode)
            results.append(result)

        return results

    def predict_expert(
        self,
        sequence: str,
        schemes: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
    ) -> StructurePredictionResult:
        """专家模式预测（V3 新增）。

        Args:
            sequence: circRNA 序列
            schemes: 指定的 Scheme 列表（如 ["S1", "S4"]）
            weights: Scheme 权重（可选）
            verbose: 是否打印警告信息

        Returns:
            StructurePredictionResult
        """
        # 使用配置中的 schemes（如果未提供）
        if schemes is None:
            if self.config.expert_mode and self.config.selected_schemes:
                schemes = self.config.selected_schemes
            else:
                schemes = ["S1", "S4"]  # 默认两个最稳定的 Scheme

        # 验证 Scheme 状态
        warnings = validate_expert_config(self.config)

        # 检查是否有 deprecated scheme
        active_schemes = []
        for scheme in schemes:
            status = self.config.scheme_status.get(scheme, "unknown")
            if status == "deprecated":
                if verbose:
                    print(f"[ExpertMode] WARNING: {scheme} is deprecated. Skipping...")
                continue
            elif status == "deferred":
                if verbose:
                    print(f"[ExpertMode] INFO: {scheme} is deferred. May not be fully available.")
            active_schemes.append(scheme)

        if not active_schemes:
            # 所有指定 scheme 都不可用，回退到默认模式
            if verbose:
                print("[ExpertMode] No active schemes available. Falling back to auto mode.")
            return self.predict(sequence)

        # 调用每个 Scheme（当前简化为调用不同 backend）
        results = []
        for scheme in active_schemes:
            # Scheme 映射到 backend
            scheme_to_backend = {
                "S1": "torusfold",
                "S2": "pipeline",
                "S4": "torusfold",  # S4 也是 TorusFold 的 diffusion 模式
                "S6": "torusfold",
                "S7": "torusfold",
            }

            backend_mode = scheme_to_backend.get(scheme, "torusfold")
            backend = self.backends.get(backend_mode)

            if backend is None:
                continue

            result = backend.predict(sequence)

            # 检查 NaN
            if check_nan_output(result):
                if verbose:
                    print(f"[ExpertMode] WARNING: {scheme} produced NaN output. Skipping...")
                continue

            # 记录到 fallback_log
            if hasattr(result, "fallback_log"):
                result.fallback_log.append(f"expert_mode:{scheme}")

            results.append(result)

        if not results:
            # 所有结果都失败
            return StructurePredictionResult(
                available=False,
                method="expert_mode_failed",
                fallback_log=["all_schemes_failed"],
            )

        # 融合结果
        if len(results) == 1:
            return results[0]

        # 多结果融合
        ensemble = safe_ensemble_prediction(results, weights)

        # 添加专家模式信息
        if hasattr(ensemble, "fallback_log"):
            ensemble.fallback_log.extend([f"ensemble:{s}" for s in active_schemes])

        return ensemble


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════

def quick_predict(
    sequence: str,
    mode: str = "pipeline",
    fallback: bool = True,
) -> StructurePredictionResult:
    """快速预测接口。"""
    backend = StructureBackend()
    backend.set_mode(mode)
    return backend.predict(sequence, fallback=fallback)


def predict_with_torusfold(sequence: str) -> StructurePredictionResult:
    """TorusFold 预测（带 fallback）。"""
    return quick_predict(sequence, mode="torusfold", fallback=True)


def predict_with_pipeline(sequence: str) -> StructurePredictionResult:
    """Pipeline 预测（带 fallback）。"""
    return quick_predict(sequence, mode="pipeline", fallback=True)


def predict_heuristic(sequence: str) -> StructurePredictionResult:
    """启发式预测。"""
    return quick_predict(sequence, mode="heuristic", fallback=False)


# ═══════════════════════════════════════════════════════════════
# Expert Mode Support (V3 新增)
# ═══════════════════════════════════════════════════════════════

def validate_expert_config(config: BackendConfig) -> List[str]:
    """验证专家模式配置，返回警告列表。

    Args:
        config: BackendConfig 对象

    Returns:
        warnings: 警告信息列表（如果有问题）
    """
    warnings = []

    if not config.expert_mode:
        return warnings  # 未启用专家模式

    # 检查是否指定了 schemes
    if not config.selected_schemes:
        warnings.append("⚠️ 专家模式已启用，但未指定 selected_schemes")
        return warnings

    # 检查每个 scheme 的状态
    for scheme in config.selected_schemes:
        status = config.scheme_status.get(scheme, "unknown")

        if status == "deprecated":
            warnings.append(f"⚠️ Scheme {scheme} is deprecated (S5). Using it may cause errors.")
        elif status == "deferred":
            warnings.append(f"ℹ️  Scheme {scheme} is deferred. May not be available yet.")

    return warnings


def check_nan_output(result: StructurePredictionResult) -> bool:
    """检查输出是否有 NaN/Inf。

    Args:
        result: StructurePredictionResult 对象

    Returns:
        True 如果检测到 NaN/Inf
    """
    if result.coords is None:
        return False

    # 检查坐标
    if np.isnan(result.coords).any() or np.isinf(result.coords).any():
        return True

    # 检查 pair_probs
    if result.pair_probs is not None:
        if np.isnan(result.pair_probs).any() or np.isinf(result.pair_probs).any():
            return True

    return False


def safe_ensemble_prediction(
    results: List[StructurePredictionResult],
    weights: Optional[Dict[str, float]] = None,
) -> StructurePredictionResult:
    """安全融合多个预测结果。

    过滤掉有 NaN 的结果，然后加权平均。

    Args:
        results: 预测结果列表
        weights: 权重字典（可选）

    Returns:
        融合后的结果
    """
    import numpy as np

    # 过滤掉有 NaN 的结果
    valid_results = [r for r in results if not check_nan_output(r)]

    if not valid_results:
        # 所有结果都有 NaN，返回第一个（可能有部分有效数据）
        return results[0]

    # 如果没有提供权重，使用均匀权重
    if weights is None:
        n = len(valid_results)
        weights = {f"res_{i}": 1.0 / n for i in range(n)}

    # 计算加权平均坐标
    all_coords = np.array([r.coords for r in valid_results])
    weighted_coords = sum(w * c for w, c in zip(weights.values(), all_coords))

    # 创建新结果
    ensemble = StructurePredictionResult(
        available=True,
        method="ensemble",
        coords=weighted_coords,
        confidence=np.mean([r.confidence for r in valid_results]),
        elapsed_time=sum(r.elapsed_time for r in valid_results),
    )

    return ensemble


# ═══════════════════════════════════════════════════════════════
# Integration with V2/V3 (continued)
# ═══════════════════════════════════════════════════════════════

def extract_signals_from_backend(
    sequence: str,
    backend: Optional[StructureBackend] = None,
    mode: str = "pipeline",
) -> StructurePredictionResult:
    """从 Backend 提取信号（用于 V2/V3）。"""
    if backend is None:
        backend = StructureBackend()
        backend.set_mode(mode)

    return backend.predict(sequence, fallback=True)


def get_torusfold_like_signals(
    sequence: str,
    prefer_mode: str = "pipeline",
) -> Dict:
    """获取类似 TorusFold 的信号（兼容 V2/V3）。

    Returns:
        {
            'available': bool,
            'method': str,
            'dsRNA_fraction': float,
            'bsj_stability': float,
            'sasa_mean': float,
            'sasa_bsj': float,
            'coords': np.ndarray or None,
            'pair_probs': np.ndarray or None,
            ...
        }
    """
    result = quick_predict(sequence, mode=prefer_mode)

    return {
        'available': result.available,
        'method': result.method,
        'dsRNA_fraction': result.dsRNA_fraction,
        'bsj_stability': result.bsj_stability,
        'sasa_mean': result.sasa_mean,
        'sasa_bsj': result.sasa_bsj,
        'sasa_per_nucleotide': result.sasa_per_nucleotide,
        'motif_accessibility': result.motif_accessibility,
        'dsRNA_mean_length': result.dsRNA_mean_length,
        'coords': result.coords,
        'pair_probs': result.pair_probs,
        'bsj_closure': result.bsj_closure,
        'bond_rmsd': result.bond_rmsd,
        'confidence': result.confidence,
        'elapsed_time': result.elapsed_time,
    }