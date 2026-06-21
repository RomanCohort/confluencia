"""
torusfold_scorer.py — TorusFold 输出 → 评分链桥接

将 TorusFold 的深度学习预测信号接入四维目标向量:
  stability      ← 闭合约束损失 + BSJ 稳定性 + circ_stability_head
  translation    ← BSJ 配对概率 + pair_map dsRNA fraction + translation_head
  immune_evasion ← pair_map dsRNA + immune_pathway_head (替代启发式)
  delivery       ← 不变 (序列长度/GC/修饰)

使用方式:
  # 模型训练完成后，启用结构预测
  scorer = TorusFoldScorer(device="cuda", use_structure_prediction=True)
  signals = scorer.extract_signals("AUGCGCUAU...", gene_expr={...})
  # → {"closure_score": 0.85, "bsj_stability": 0.72, "dsRNA_fraction": 0.35, ...}

  # 模型未训练时，使用启发式fallback
  scorer = TorusFoldScorer(device="cpu", use_structure_prediction=False)
  signals = scorer.extract_signals("AUGCGCUAU...")
  # → TorusFoldSignals(available=False, method="heuristic_fallback")

  objectives = scorer.compute_objectives("AUGCGCUAU...", modification="m6A",
                                          immune_scores=..., torusfold_signals=signals)
  # → np.array([stability, translation, immune_evasion, delivery])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch

# Lazy import TorusFold to avoid import errors when model not trained
_TorusFold = None
_TorusFoldConfig = None


def _get_torusfold_classes():
    """Lazy import TorusFold classes."""
    global _TorusFold, _TorusFoldConfig
    if _TorusFold is None:
        from .torusfold import TorusFold, TorusFoldConfig
        _TorusFold = TorusFold
        _TorusFoldConfig = TorusFoldConfig
    return _TorusFold, _TorusFoldConfig


@dataclass
class TorusFoldSignals:
    """TorusFold 提取的结构信号，用于修正评分链。"""
    # 闭合约束
    closure_distance: float = 0.0        # 首-末核苷酸 3D 距离 (Å, 越小越好)
    closure_loss: float = 0.0            # Diffusion 闭合损失 (越小越好)
    closure_score: float = 0.5           # 归一化闭合评分 [0,1] (越高越好)

    # BSJ 配对
    bsj_stability: float = 0.5           # BSJ 位点稳定性 [0,1]
    bsj_confidence: float = 0.5          # BSJ 预测置信度 [0,1]
    bsj_pair_count: float = 0.0          # 跨 BSJ 配对数

    # pair_map 衍生
    dsRNA_fraction: float = 0.0          # 配对碱基比例 (pair_prob > 0.5)
    mean_pair_prob: float = 0.0          # 平均配对概率
    long_range_pair_fraction: float = 0.0  # 长程配对比例 (circ_dist > L/4)

    # TorusFold 多任务头
    translation_efficiency: float = 0.5  # 翻译效率 [0,1]
    circ_stability: float = 0.5          # 环稳定性 [0,1]
    immune_rig_i: float = 0.3            # RIG-I 激活概率
    immune_tlr: float = 0.2              # TLR 激活概率
    immune_pkr: float = 0.3              # PKR 激活概率

    # 物理约束信号 (physics_b / physics_ba 模式)
    energy_score: float = 0.0            # CG 能量 (kcal/mol, 越低越好)
    bond_rmsd: float = 0.0               # 键长 RMSD (Å)
    pair_satisfaction: float = 0.0        # 配对距离满足率 [0,1]
    clash_count: int = 0                  # 空间碰撞数
    n_conformations: int = 0              # 采样构象数

    # 3D 结构可及性信号 (训练完成后可用)
    motif_accessibility: Dict[str, float] = None  # motif_id → 3D可及性 [0,1]
    ires_3d_accessibility: float = 0.0    # IRES 区域3D可及性 [0,1]
    bsj_3d_closure_tightness: float = 0.0 # BSJ区域3D闭合紧密度 [0,1]
    surface_exposed_fraction: float = 0.0  # 表面暴露碱基比例 [0,1]
    buried_motif_count: int = 0           # 被埋藏的免疫motif数量

    # 元信息
    available: bool = False              # TorusFold 是否成功运行
    method: str = "none"                 # "torusfold" | "physics_b" | "physics_ba" | "heuristic_fallback"


class TorusFoldScorer:
    """TorusFold 信号提取 + 评分修正桥接。

    作用:
    1. extract_signals(): 运行 TorusFold，提取结构信号
    2. compute_objectives(): 将信号融入四维目标向量
    3. compute_immune_override(): 用 TorusFold 免疫头替代启发式评分
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        config=None,
        structure_mode: str = "simple",
        diffusion_steps: int = 100,
        solver_samples: int = 20,
        openmm_minimize_steps: int = 500,
        openmm_md_steps: int = 5000,
        use_structure_prediction: bool = False,
    ):
        """
        Args:
            use_structure_prediction: If True, run TorusFold model to get 3D
                structure signals. If False (default), use heuristic fallback
                until the model is trained. Set to True after training completes.
        """
        self.device = device
        self.structure_mode = structure_mode
        self.use_structure_prediction = use_structure_prediction

        # 如果未提供 config，根据 structure_mode 构造
        if config is not None:
            self.config = config
        else:
            TorusFold, TorusFoldConfig = _get_torusfold_classes()
            self.config = TorusFoldConfig(
                structure_mode=structure_mode,
                n_diffusion_steps=diffusion_steps,
                n_solver_samples=solver_samples,
                n_minimize_steps=openmm_minimize_steps,
                n_md_steps=openmm_md_steps,
            )
        self._model = None
        self._model_path = model_path

    @property
    def model(self):
        if not self.use_structure_prediction:
            return None
        if self._model is not None:
            return self._model
        try:
            TorusFold, _ = _get_torusfold_classes()
            model = TorusFold(self.config)
            if self._model_path:
                model.load(self._model_path, device=self.device)
            model = model.to(self.device)
            model.eval()
            self._model = model
            return model
        except Exception:
            return None

    def extract_signals(
        self,
        sequence: str,
        gene_expr: Optional[Dict[str, float]] = None,
        structure_mode: Optional[str] = None,
    ) -> TorusFoldSignals:
        """运行 TorusFold 提取结构信号。

        Args:
            sequence: circRNA 序列
            gene_expr: 基因表达字典
            structure_mode: 可选覆盖结构模式 (simple/diffusion/physics_b/physics_ba)
                           如为 None，使用初始化时的 self.structure_mode

        Returns:
            TorusFoldSignals. If use_structure_prediction=False or model
            unavailable, returns heuristic fallback signals.
        """
        # If structure prediction is disabled, return heuristic fallback
        if not self.use_structure_prediction:
            return self._heuristic_fallback_signals(sequence)

        # 如果传入了 structure_mode，临时更新 config
        effective_mode = structure_mode or self.structure_mode
        if effective_mode != self.config.structure_mode:
            self.config.structure_mode = effective_mode
            # 需要重新创建模型
            self._model = None

        model = self.model
        if model is None:
            return self._heuristic_fallback_signals(sequence)

        seq = sequence.upper().replace("T", "U")
        if gene_expr is None:
            gene_expr = {g: 0.5 for g in self.config.gene_cols}

        try:
            with torch.no_grad():
                result = model.predict_single(seq, gene_expr, device=self.device)

            # 提取 pair_map 衍生信号
            dsRNA_frac = 0.0
            mean_pair_prob = 0.0
            long_range_frac = 0.0

            if "pair_probs" in result and isinstance(result.get("pair_probs"), torch.Tensor):
                pp = result["pair_probs"]  # (1, L, L) or (L, L)
                if pp.dim() == 3:
                    pp = pp[0]
                L = pp.size(0)
                # dsRNA fraction: 配对概率 > 0.5 的比例 (取上三角)
                upper = torch.triu(pp, diagonal=1)
                n_pairs = (upper > 0.5).float().sum().item()
                total = max(L * (L - 1) / 2, 1)
                dsRNA_frac = n_pairs / total
                mean_pair_prob = upper.mean().item()
                # 长程配对: circular_distance > L/4
                if L > 4:
                    cdist = circular_distance_matrix(L, pp.device)
                    long_range_mask = (cdist > L / 4).float()
                    long_range_pairs = (upper * long_range_mask).sum().item()
                    long_range_total = long_range_mask.sum().item()
                    long_range_frac = long_range_pairs / max(long_range_total, 1)

            # 闭合评分: closure_distance → [0,1]
            closure_dist = result.get("closure_distance", 0.0)
            if isinstance(closure_dist, torch.Tensor):
                closure_dist = closure_dist.item()
            closure_score = max(0.0, min(1.0, 1.0 - closure_dist / 20.0))  # 20Å → 0

            # 闭合损失 → 归一化
            closure_loss = result.get("closure_loss", 0.0)
            if isinstance(closure_loss, torch.Tensor):
                closure_loss = closure_loss.item()
            closure_loss_score = max(0.0, min(1.0, 1.0 - closure_loss))

            # 合并两个闭合信号
            combined_closure = 0.5 * closure_score + 0.5 * closure_loss_score

            # 物理约束信号 (physics_b / physics_ba)
            energy_score = 0.0
            bond_rmsd = 0.0
            pair_satisfaction = 0.0
            clash_count = 0
            n_conformations = 0
            structure_method = "torusfold"

            if "energy_score" in result:
                es = result["energy_score"]
                energy_score = es.item() if isinstance(es, torch.Tensor) else float(es)
            if "bond_rmsd" in result:
                br = result["bond_rmsd"]
                bond_rmsd = br.item() if isinstance(br, torch.Tensor) else float(br)
            if "pair_satisfaction" in result:
                ps = result["pair_satisfaction"]
                pair_satisfaction = ps.item() if isinstance(ps, torch.Tensor) else float(ps)
            if "structure_method" in result:
                sm = result["structure_method"]
                structure_method = sm if isinstance(sm, str) else str(sm)

            return TorusFoldSignals(
                closure_distance=closure_dist,
                closure_loss=closure_loss,
                closure_score=combined_closure,
                bsj_stability=result.get("bsj_stability", 0.5),
                bsj_confidence=result.get("bsj_confidence", 0.5),
                bsj_pair_count=result.get("bsj_pair_count", 0.0),
                dsRNA_fraction=dsRNA_frac,
                mean_pair_prob=mean_pair_prob,
                long_range_pair_fraction=long_range_frac,
                translation_efficiency=result.get("translation_efficiency", 0.5),
                circ_stability=result.get("circ_stability", 0.5),
                immune_rig_i=result.get("immune_pathway_RIG-I", 0.3),
                immune_tlr=result.get("immune_pathway_TLR", 0.2),
                immune_pkr=result.get("immune_pathway_PKR", 0.3),
                energy_score=energy_score,
                bond_rmsd=bond_rmsd,
                pair_satisfaction=pair_satisfaction,
                clash_count=clash_count,
                n_conformations=n_conformations,
                available=True,
                method=structure_method or effective_mode,
            )

    def _heuristic_fallback_signals(self, sequence: str) -> TorusFoldSignals:
        """启发式回退信号（模型未训练时使用）。"""
        seq = sequence.upper().replace("T", "U")
        L = len(seq)

        # 闭合评分：序列越长，闭环越难
        closure_score = max(0.1, 1.0 - L / 2000.0)

        # BSJ稳定性：基于BSJ区域的GC含量
        bsj_region = seq[:20] + seq[-20:] if L > 40 else seq
        gc_bsj = sum(1 for c in bsj_region if c in "GC") / len(bsj_region) if bsj_region else 0.3
        bsj_stability = 0.3 + gc_bsj * 0.5

        # IRES 3D可及性（假设部分IRES在表面）
        ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"]
        ires_count = sum(1 for m in ires_motifs if m in seq)
        ires_3d_accessibility = min(0.9, 0.3 + ires_count * 0.15)

        return TorusFoldSignals(
            closure_distance=L * 0.5,
            closure_loss=1.0 - closure_score,
            closure_score=closure_score,
            bsj_stability=bsj_stability,
            bsj_confidence=0.5,
            bsj_pair_count=0.0,
            dsRNA_fraction=0.0,
            mean_pair_prob=0.0,
            long_range_pair_fraction=0.0,
            translation_efficiency=0.5,
            circ_stability=closure_score,
            immune_rig_i=0.3,
            immune_tlr=0.2,
            immune_pkr=0.3,
            energy_score=100.0,
            bond_rmsd=2.0,
            pair_satisfaction=0.0,
            clash_count=0,
            n_conformations=0,
            # 3D结构可及性信号
            motif_accessibility={},
            ires_3d_accessibility=ires_3d_accessibility,
            bsj_3d_closure_tightness=bsj_stability,
            surface_exposed_fraction=0.5,
            buried_motif_count=0,
            available=False,
            method="heuristic_fallback",
        )

        except Exception:
            return TorusFoldSignals(available=False, method="heuristic_fallback")

    def compute_objectives(
        self,
        seq: str,
        modification: str = "none",
        immune_scores: Optional[Dict[str, float]] = None,
        torusfold_signals: Optional[TorusFoldSignals] = None,
        viennarna_mfe: Optional[float] = None,
    ) -> np.ndarray:
        """计算修正后的四维目标向量 [stability, translation, immune_evasion, delivery]。

        当 TorusFold 信号可用时，用 DL 预测修正启发式评分;
        否则退化为纯启发式 (与原始 compute_cirrna_objectives 一致)。
        """
        seq = seq.upper().replace("T", "U")
        length = len(seq)

        if length < 50:
            return np.array([0.3, 0.3, 0.5, 0.3], dtype=np.float32)

        gc = sum(1 for c in seq if c in "GC") / length

        # ─── stability ───
        stability = 0.3 + gc * 0.5
        mod_bonus = {
            "m6A": 0.1, "Psi": 0.15, "5mC": 0.08, "ms2m6A": 0.12,
            "2OMeA": 0.1, "2OMeU": 0.1, "m5U": 0.05, "s2U": 0.05,
        }
        stability += mod_bonus.get(modification, 0.0)

        if torusfold_signals and torusfold_signals.available:
            # TorusFold 修正: 闭合约束 + circ_stability_head + BSJ 3D闭合
            stability = 0.5 * stability + 0.2 * torusfold_signals.closure_score + 0.15 * torusfold_signals.circ_stability

            # 3D结构信号: BSJ区域3D闭合紧密度
            if torusfold_signals.bsj_3d_closure_tightness > 0:
                stability = 0.7 * stability + 0.3 * torusfold_signals.bsj_3d_closure_tightness

            # 物理约束修正 (仅 physics_b / physics_ba 模式)
            if torusfold_signals.energy_score > 0:
                # CG 能量越低 = 结构越稳定
                energy_norm = max(0.0, min(1.0, 1.0 - torusfold_signals.energy_score / 500.0))
                stability = 0.5 * stability + 0.3 * energy_norm + 0.2 * torusfold_signals.closure_score

        if viennarna_mfe is not None:
            # ViennaRNA MFE 修正: 更负 = 更稳定
            mfe_norm = max(0.0, min(1.0, (-viennarna_mfe) / 300.0))  # -300 kcal → 1.0
            stability = 0.7 * stability + 0.3 * mfe_norm

        obj0 = np.clip(stability, 0.0, 1.0)

        # ─── translation ───
        ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"]
        ires_count = sum(1 for m in ires_motifs if m in seq)
        translation = 0.2 + ires_count * 0.12
        aug_count = seq.count("AUG")
        translation += min(aug_count * 0.05, 0.2)
        if 0.4 <= gc <= 0.55:
            translation += 0.1

        if torusfold_signals and torusfold_signals.available:
            # TorusFold 修正: BSJ 稳定性 + translation_head + IRES 3D可及性
            translation = (0.4 * translation
                           + 0.2 * torusfold_signals.bsj_stability
                           + 0.2 * torusfold_signals.translation_efficiency
                           + 0.2 * torusfold_signals.ires_3d_accessibility)

        obj1 = np.clip(translation, 0.0, 1.0)

        # ─── immune_evasion ───
        if torusfold_signals and torusfold_signals.available:
            # TorusFold 修正: 用 DL 免疫头 + pair_map dsRNA + 3D motif可及性
            # 替代启发式 dsRNA 估计
            pkr = torusfold_signals.immune_pkr
            rig_i = torusfold_signals.immune_rig_i
            tlr = torusfold_signals.immune_tlr

            # pair_map dsRNA 额外修正 PKR (更准确)
            dsRNA_from_pairmap = torusfold_signals.dsRNA_fraction
            pkr_adjusted = 0.6 * pkr + 0.4 * dsRNA_from_pairmap

            # 3D结构可及性修正: 被埋藏的motif不会触发免疫
            # surface_exposed_fraction越高 = 更多碱基暴露 = 更易被免疫传感器检测
            exposure_penalty = torusfold_signals.surface_exposed_fraction
            buried_bonus = 1.0 - exposure_penalty * 0.3

            immune_evasion = (
                (1.0 - pkr_adjusted) * 0.35
                + (1.0 - abs(rig_i - 0.35)) * 0.25
                + (1.0 - tlr) * 0.2
                + buried_bonus * 0.2
            )

            # 如果有具体的motif可及性数据，进一步修正
            if torusfold_signals.motif_accessibility:
                # 免疫motif被埋藏 = 更好的免疫逃逸
                avg_motif_exposure = sum(torusfold_signals.motif_accessibility.values()) / max(len(torusfold_signals.motif_accessibility), 1)
                immune_evasion = 0.8 * immune_evasion + 0.2 * (1.0 - avg_motif_exposure)
        elif immune_scores:
            pkr = immune_scores.get("pkr_score", 0.3)
            rig_i = immune_scores.get("rig_i_score", 0.3)
            tlr = immune_scores.get("tlr_score", immune_scores.get("tlr7_score", 0.2))
            immune_evasion = (1.0 - pkr) * 0.4 + (1.0 - abs(rig_i - 0.35)) * 0.3 + (1.0 - tlr) * 0.3
        else:
            dsRNA_potential = gc * 0.7 * (length > 500)
            gu_content = sum(1 for c in seq if c in "GU") / length
            rig_i_estimate = gu_content * 0.5
            immune_evasion = (1.0 - dsRNA_potential) * 0.5 + (1.0 - abs(rig_i_estimate - 0.35)) * 0.5

        obj2 = np.clip(immune_evasion, 0.0, 1.0)

        # ─── delivery ───
        # delivery 不受 TorusFold 影响 (序列长度/GC/修饰决定)
        delivery = 0.3
        if length < 2000:
            delivery += 0.25
        elif length < 5000:
            delivery += 0.15
        if 0.35 < gc < 0.55:
            delivery += 0.2
        if modification in ["m6A", "Psi", "2OMeA", "2OMeU"]:
            delivery += 0.15
        obj3 = np.clip(delivery, 0.0, 1.0)

        return np.array([obj0, obj1, obj2, obj3], dtype=np.float32)

    def compute_immune_override(
        self,
        torusfold_signals: TorusFoldSignals,
    ) -> Optional[Dict[str, float]]:
        """用 TorusFold 免疫头替代启发式免疫评分。

        Returns:
            替代后的免疫评分 dict, 或 None (信号不可用时)
        """
        if not torusfold_signals or not torusfold_signals.available:
            return None

        return {
            "rig_i_score": torusfold_signals.immune_rig_i,
            "tlr7_score": torusfold_signals.immune_tlr * 0.6,
            "tlr8_score": torusfold_signals.immune_tlr * 0.4,
            "pkr_score": max(torusfold_signals.immune_pkr,
                             torusfold_signals.dsRNA_fraction),
            "overall_immunogenicity": (
                0.35 * torusfold_signals.immune_rig_i
                + 0.20 * torusfold_signals.immune_tlr * 0.6
                + 0.15 * torusfold_signals.immune_tlr * 0.4
                + 0.30 * max(torusfold_signals.immune_pkr,
                             torusfold_signals.dsRNA_fraction)
            ),
            "sensing_method": "torusfold_dl_override",
        }


def quick_score(
    sequence: str,
    modification: str = "none",
    use_structure_prediction: bool = False,
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """快速评分入口，用于 RL-ABM 或搜索。

    Args:
        use_structure_prediction: 启用TorusFold 3D结构预测。
            设为True需先完成模型训练。
    """
    scorer = TorusFoldScorer(
        device=device,
        model_path=model_path,
        use_structure_prediction=use_structure_prediction,
    )
    signals = scorer.extract_signals(sequence)
    objectives = scorer.compute_objectives(
        sequence, modification, None, signals, None,
    )
    result = {
        "stability": float(objectives[0]),
        "translation": float(objectives[1]),
        "immune_evasion": float(objectives[2]),
        "delivery": float(objectives[3]),
        "scoring_method": signals.method,
    }
    if signals.available:
        result.update({
            "tf_closure_score": signals.closure_score,
            "tf_bsj_stability": signals.bsj_stability,
            "tf_dsRNA_fraction": signals.dsRNA_fraction,
            "tf_translation_efficiency": signals.translation_efficiency,
            "tf_circ_stability": signals.circ_stability,
        })
    return result
