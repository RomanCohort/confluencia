"""circRNA 序列进化 — REINFORCE + Pareto 多目标优化

移动自 core/circrna/cirrna_evolution.py → core/evolution/cirrna_evolution.py。
保留对 circrna/bsj_features 的交叉引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd

from .pareto import (
    softmax,
    normalize_cols,
    pareto_front_mask,
    select_weights_with_pareto,
    reward_from_weights,
    pick_actions,
)
from .actions import CIRCRNA_ACTIONS

# Lazy import TorusFoldScorer to avoid hard dependency on torch/torusfold
_TorusFoldScorer = None
_TorusFoldSignals = None


def _get_torusfold_scorer():
    """Lazy import TorusFoldScorer and TorusFoldSignals."""
    global _TorusFoldScorer, _TorusFoldSignals
    if _TorusFoldScorer is None:
        try:
            from ..circrna.torusfold_scorer import TorusFoldScorer, TorusFoldSignals
            _TorusFoldScorer = TorusFoldScorer
            _TorusFoldSignals = TorusFoldSignals
        except Exception:
            _TorusFoldScorer = None
            _TorusFoldSignals = None
    return _TorusFoldScorer, _TorusFoldSignals


@dataclass
class CircRNAEvolutionConfig:
    """circRNA 序列进化配置。"""
    rounds: int = 5
    top_k: int = 8
    candidates_per_round: int = 24
    epsilon: float = 0.15
    lr: float = 0.06
    seed_seq: str = "AUGCGCUAUGGCUAGCUAUGCGCUAUGGCUAGCUAUGCGCUAUGGC"
    modification: str = "m6A"
    delivery_vector: str = "LNP_liver"
    route: str = "IV"
    ires_type: str = "EMCV"
    use_pareto_search: bool = True
    pareto_weight_samples: int = 32
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    seed: int = 42
    weight_stability: float = 0.35
    weight_translation: float = 0.30
    weight_immune_evasion: float = 0.25
    weight_delivery: float = 0.10
    # TorusFold 结构评分配置
    use_torusfold_scoring: bool = False          # 启用 TorusFold 3D 结构信号
    torusfold_model_path: Optional[str] = None   # TorusFold 模型权重路径
    torusfold_device: str = "cpu"                # TorusFold 推理设备 (cuda/cpu)
    torusfold_use_structure: bool = False        # 是否运行 3D 结构预测 (需先训练模型)


@dataclass
class CircRNAEvolutionArtifacts:
    """进化运行产物。"""
    final_policy_logits: Dict[str, float]
    reflections: List[str]
    rounds_ran: int
    best_reward: float
    per_round_best: List[float]
    selected_weights: Dict[str, float]
    best_sequence: str
    best_modification: str


# ===================================================================
# circRNA 突变算子
# ===================================================================

def mutate_backbone(seq: str, rng: np.random.Generator, n_mutations: int = 3) -> str:
    """circRNA 骨架点突变 (保护 BSJ 区域)。"""
    s = list(seq.upper().replace("T", "U"))
    if len(s) < 10:
        return seq

    try:
        from ..circrna.bsj_features import extract_bsj_features
        bsj_features = extract_bsj_features(seq)
        protected_start, protected_end = bsj_features.protected_region
        if bsj_features.circularization_score > 0.7:
            protected_start = max(protected_start, 15)
            protected_end = max(protected_end, 15)
    except Exception:
        protected_start = min(5, len(s) // 4)
        protected_end = min(5, len(s) // 4)

    mutable_range = list(range(protected_start, len(s) - protected_end))
    if len(mutable_range) < 3:
        mutable_range = list(range(len(s)))

    nt_choices = ["A", "U", "G", "C"]
    n_mut = min(n_mutations, len(mutable_range))

    indices = rng.choice(mutable_range, size=n_mut, replace=False)
    for idx in indices:
        current = s[idx]
        alternatives = [nt for nt in nt_choices if nt != current]
        s[idx] = str(rng.choice(alternatives))

    return "".join(s)


def optimize_ires(seq: str, rng: np.random.Generator) -> str:
    """优化 IRES 区域 (翻译增强)。"""
    s = list(seq.upper().replace("T", "U"))
    if len(s) < 30:
        return seq

    strong_motifs = ["GCGCC", "CCUG", "GGGG", "UUGU", "AUGG", "GGAAGG", "CCCUUU"]

    if rng.random() < 0.5 and len(s) > 40:
        pos = int(rng.integers(10, len(s) - 10))
        motif = str(rng.choice(strong_motifs))
        s[pos:pos + len(motif)] = list(motif)
    else:
        pos = int(rng.integers(5, max(6, len(s) - 10)))
        motif = str(rng.choice(strong_motifs))
        replace_len = min(len(motif), len(s) - pos - 5)
        if replace_len > 2:
            s[pos:pos + replace_len] = list(motif[:replace_len])

    return "".join(s)


def shuffle_ires_flanking(seq: str, rng: np.random.Generator) -> str:
    """打乱 IRES 近端和 BSJ 侧翼区域 (保留 BSJ)。"""
    s = seq.upper().replace("T", "U")
    if len(s) < 30:
        return seq

    first_aug = s.find("AUG")
    if first_aug < 0:
        first_aug = len(s) // 3

    stop_codons = ["UAA", "UAG", "UGA"]
    last_stop = -1
    for sc in stop_codons:
        idx = s.rfind(sc)
        if idx > last_stop:
            last_stop = idx + 3

    if last_stop <= first_aug:
        last_stop = len(s)

    ires_proximal_end = min(first_aug, len(s) - 1)
    if ires_proximal_end > 3:
        ires_proximal = list(s[:ires_proximal_end])
        rng.shuffle(ires_proximal)
        s = "".join(ires_proximal) + s[ires_proximal_end:]

    if last_stop < len(s) - 3:
        bsj_flanking = list(s[last_stop:])
        rng.shuffle(bsj_flanking)
        s = s[:last_stop] + "".join(bsj_flanking)

    return s


def _modification_pool() -> List[str]:
    """可用 circRNA 修饰。"""
    return ["none", "m6A", "Psi", "5mC", "ms2m6A", "2OMeA", "2OMeU", "m5U", "s2U"]


# ===================================================================
# 目标计算
# ===================================================================

def compute_cirrna_objectives(
    seq: str,
    modification: str,
    immune_scores: Optional[Dict[str, float]] = None,
    torusfold_signals: Optional[Any] = None,
) -> np.ndarray:
    """计算 circRNA 四维目标向量 (全部最大化)。

    当 torusfold_signals 提供且 available=True 时，使用 TorusFold DL
    预测信号修正启发式评分 (与 TorusFoldScorer.compute_objectives 相同
    混合逻辑); 否则退化为纯启发式。

    Returns: [stability, translation, immune_evasion, delivery]
    """
    seq = seq.upper().replace("T", "U")
    length = len(seq)

    if length < 50:
        return np.array([0.3, 0.3, 0.5, 0.3], dtype=np.float32)

    gc = sum(1 for c in seq if c in "GC") / length

    # 稳定性
    stability = 0.3 + gc * 0.5
    mod_stability_bonus = {
        "m6A": 0.1, "Psi": 0.15, "5mC": 0.08, "ms2m6A": 0.12,
        "2OMeA": 0.1, "2OMeU": 0.1, "m5U": 0.05, "s2U": 0.05,
    }
    stability += mod_stability_bonus.get(modification, 0.0)

    if torusfold_signals and getattr(torusfold_signals, "available", False):
        # TorusFold 修正: 闭合约束 + circ_stability_head + BSJ 3D闭合
        stability = 0.5 * stability + 0.2 * torusfold_signals.closure_score + 0.15 * torusfold_signals.circ_stability
        if getattr(torusfold_signals, "bsj_3d_closure_tightness", 0) > 0:
            stability = 0.7 * stability + 0.3 * torusfold_signals.bsj_3d_closure_tightness
        if getattr(torusfold_signals, "energy_score", 0) > 0:
            energy_norm = max(0.0, min(1.0, 1.0 - torusfold_signals.energy_score / 500.0))
            stability = 0.5 * stability + 0.3 * energy_norm + 0.2 * torusfold_signals.closure_score

    obj0 = np.clip(stability, 0.0, 1.0)

    # 翻译潜力
    ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG", "GGAAGG"]
    ires_count = sum(1 for m in ires_motifs if m in seq)
    translation = 0.2 + ires_count * 0.12
    aug_count = seq.count("AUG")
    translation += min(aug_count * 0.05, 0.2)
    if 0.4 <= gc <= 0.55:
        translation += 0.1

    if torusfold_signals and getattr(torusfold_signals, "available", False):
        translation = (0.4 * translation
                       + 0.2 * torusfold_signals.bsj_stability
                       + 0.2 * torusfold_signals.translation_efficiency
                       + 0.2 * getattr(torusfold_signals, "ires_3d_accessibility", 0.5))

    obj1 = np.clip(translation, 0.0, 1.0)

    # 免疫逃逸
    if torusfold_signals and getattr(torusfold_signals, "available", False):
        # TorusFold DL 免疫头 + pair_map dsRNA + 3D motif可及性
        pkr = torusfold_signals.immune_pkr
        rig_i = torusfold_signals.immune_rig_i
        tlr = torusfold_signals.immune_tlr
        dsRNA_from_pairmap = torusfold_signals.dsRNA_fraction
        pkr_adjusted = 0.6 * pkr + 0.4 * dsRNA_from_pairmap
        exposure_penalty = getattr(torusfold_signals, "surface_exposed_fraction", 0.5)
        buried_bonus = 1.0 - exposure_penalty * 0.3
        immune_evasion = (
            (1.0 - pkr_adjusted) * 0.35
            + (1.0 - abs(rig_i - 0.35)) * 0.25
            + (1.0 - tlr) * 0.2
            + buried_bonus * 0.2
        )
        motif_acc = getattr(torusfold_signals, "motif_accessibility", None)
        if motif_acc:
            avg_motif_exposure = sum(motif_acc.values()) / max(len(motif_acc), 1)
            immune_evasion = 0.8 * immune_evasion + 0.2 * (1.0 - avg_motif_exposure)
    elif immune_scores:
        pkr = immune_scores.get("pkr_score", 0.3)
        rig_i = immune_scores.get("rig_i_score", 0.3)
        tlr = immune_scores.get("tlr_score", 0.2)
        immune_evasion = (1.0 - pkr) * 0.4 + (1.0 - abs(rig_i - 0.35)) * 0.3 + (1.0 - tlr) * 0.3
    else:
        dsRNA_potential = gc * 0.7 * (length > 500)
        gu_content = sum(1 for c in seq if c in "GU") / length
        rig_i_estimate = gu_content * 0.5
        immune_evasion = (1.0 - dsRNA_potential) * 0.5 + (1.0 - abs(rig_i_estimate - 0.35)) * 0.5
    obj2 = np.clip(immune_evasion, 0.0, 1.0)

    # 递送兼容性 (不受 TorusFold 影响)
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


# ===================================================================
# 主进化函数
# ===================================================================

def evolve_cirrna(
    cfg: CircRNAEvolutionConfig,
    immune_score_fn: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, CircRNAEvolutionArtifacts]:
    """反思式 RL circRNA 序列进化。

    Args:
        cfg: 进化配置
        immune_score_fn: 可选免疫评分函数 (seq -> Dict)

    Returns:
        (结果 DataFrame, 进化产物)
    """
    rng = np.random.default_rng(cfg.seed)
    logits = np.zeros((len(CIRCRNA_ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[Dict] = []

    current_pool: List[str] = [cfg.seed_seq]
    current_mods: List[str] = [cfg.modification]
    best_reward_global = -1e9
    best_seq_global = cfg.seed_seq
    best_mod_global = cfg.modification
    no_improve = 0
    per_round_best: List[float] = []
    rounds_ran = 0

    # 创建 TorusFoldScorer (如果启用)
    torusfold_scorer = None
    if cfg.use_torusfold_scoring:
        TorusFoldScorer, _ = _get_torusfold_scorer()
        if TorusFoldScorer is not None:
            try:
                torusfold_scorer = TorusFoldScorer(
                    model_path=cfg.torusfold_model_path,
                    device=cfg.torusfold_device,
                    use_structure_prediction=cfg.torusfold_use_structure,
                )
                reflections.append(f"TorusFoldScorer initialized (device={cfg.torusfold_device}, use_structure={cfg.torusfold_use_structure})")
            except Exception as e:
                reflections.append(f"TorusFoldScorer init failed: {e}, falling back to heuristic")
                torusfold_scorer = None
        else:
            reflections.append("TorusFoldScorer unavailable, using heuristic scoring")

    prior_w = np.array([
        cfg.weight_stability, cfg.weight_translation,
        cfg.weight_immune_evasion, cfg.weight_delivery,
    ], dtype=np.float32)

    for rd in range(max(cfg.rounds, 1)):
        rounds_ran = rd + 1
        n = max(cfg.candidates_per_round, 4)
        action_idx = pick_actions(logits, n=n, eps=cfg.epsilon,
                                  n_actions=len(CIRCRNA_ACTIONS), rng=rng)

        candidates: List[str] = []
        mod_candidates: List[str] = []
        actions: List[str] = []

        for ai in action_idx.tolist():
            base_seq = str(rng.choice(current_pool))
            base_mod = str(rng.choice(current_mods))

            if ai == 0:
                candidates.append(mutate_backbone(base_seq, rng, n_mutations=int(rng.integers(1, 5))))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[0])
            elif ai == 1:
                candidates.append(optimize_ires(base_seq, rng))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[1])
            elif ai == 2:
                candidates.append(shuffle_ires_flanking(base_seq, rng))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[2])
            else:
                candidates.append(base_seq)
                mod_pool = _modification_pool()
                new_mod = str(rng.choice([m for m in mod_pool if m != base_mod] or mod_pool))
                mod_candidates.append(new_mod)
                actions.append(CIRCRNA_ACTIONS[3])

        # 计算目标
        obj_matrix = np.zeros((len(candidates), 4), dtype=np.float32)
        for i, (seq, mod) in enumerate(zip(candidates, mod_candidates)):
            immune_scores = None
            if immune_score_fn:
                try:
                    immune_scores = immune_score_fn(seq)
                except Exception:
                    immune_scores = None

            # 提取 TorusFold 信号 (如果启用)
            torusfold_signals = None
            if torusfold_scorer is not None:
                try:
                    torusfold_signals = torusfold_scorer.extract_signals(seq)
                except Exception:
                    torusfold_signals = None

            obj_matrix[i] = compute_cirrna_objectives(
                seq, mod, immune_scores, torusfold_signals,
            )

        obj_norm = normalize_cols(obj_matrix)

        selected_w = prior_w.copy()
        if cfg.use_pareto_search and obj_norm.shape[0] >= 2:
            selected_w = select_weights_with_pareto(
                X_obj_norm=obj_norm, top_k=cfg.top_k,
                n_samples=cfg.pareto_weight_samples, rng=rng, prior=prior_w,
            )
            if selected_w.shape[0] != 4:
                selected_w = prior_w.copy()

        rewards = reward_from_weights(obj_norm, selected_w)
        p_mask = pareto_front_mask(obj_norm)

        for i, (seq, mod, act) in enumerate(zip(candidates, mod_candidates, actions)):
            row = {
                "round": rd + 1, "action": act,
                "circrna_seq": seq, "seq_length": len(seq),
                "modification": mod, "delivery_vector": cfg.delivery_vector,
                "route": cfg.route, "ires_type": cfg.ires_type,
                "reward": float(rewards[i]),
                "obj_stability": float(obj_matrix[i, 0]),
                "obj_translation": float(obj_matrix[i, 1]),
                "obj_immune_evasion": float(obj_matrix[i, 2]),
                "obj_delivery": float(obj_matrix[i, 3]),
                "pareto_front": bool(p_mask[i]),
            }
            all_rows.append(row)

        # REINFORCE 策略更新
        r_center = rewards - rewards.mean()
        for i, act in enumerate(actions):
            aidx = CIRCRNA_ACTIONS.index(act)
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        reflections.append(
            f"Round {rd+1}: reward_mean={float(rewards.mean()):.4f}, pareto={int(p_mask.sum())}"
        )

        best_reward = float(rewards.max())
        best_idx = int(np.argmax(rewards))
        per_round_best.append(best_reward)

        if best_reward > best_reward_global + cfg.min_improve:
            best_reward_global = best_reward
            best_seq_global = candidates[best_idx]
            best_mod_global = mod_candidates[best_idx]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max(cfg.early_stop_patience, 1):
            reflections.append(f"Early-stop at round {rd+1}")
            break

        top_idx = np.argsort(-rewards)[:max(cfg.top_k, 2)]
        current_pool = [candidates[i] for i in top_idx]
        current_mods = [mod_candidates[i] for i in top_idx]

    result_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    logits_dict = {k: float(v) for k, v in zip(CIRCRNA_ACTIONS, logits.tolist())}

    artifacts = CircRNAEvolutionArtifacts(
        final_policy_logits=logits_dict,
        reflections=reflections,
        rounds_ran=rounds_ran,
        best_reward=float(best_reward_global),
        per_round_best=per_round_best,
        selected_weights={
            "stability": float(selected_w[0]),
            "translation": float(selected_w[1]),
            "immune_evasion": float(selected_w[2]),
            "delivery": float(selected_w[3]),
        },
        best_sequence=best_seq_global,
        best_modification=best_mod_global,
    )
    return result_df, artifacts


# ===================================================================
# 便捷函数
# ===================================================================

def run_cirrna_evolution(
    seed_seq: str,
    rounds: int = 5,
    modification: str = "m6A",
) -> Tuple[pd.DataFrame, CircRNAEvolutionArtifacts]:
    """快速 circRNA 进化 (默认配置)。"""
    cfg = CircRNAEvolutionConfig(rounds=rounds, seed_seq=seed_seq, modification=modification)
    return evolve_cirrna(cfg)


def optimize_for_translation(seq: str, rounds: int = 3) -> str:
    """优化 circRNA 翻译效率。"""
    cfg = CircRNAEvolutionConfig(
        rounds=rounds, seed_seq=seq,
        weight_translation=0.5, weight_stability=0.25,
        weight_immune_evasion=0.15, weight_delivery=0.10,
    )
    _, artifacts = evolve_cirrna(cfg)
    return artifacts.best_sequence


def optimize_for_stability(seq: str, rounds: int = 3) -> str:
    """优化 circRNA 稳定性。"""
    cfg = CircRNAEvolutionConfig(
        rounds=rounds, seed_seq=seq,
        weight_stability=0.5, weight_translation=0.20,
        weight_immune_evasion=0.20, weight_delivery=0.10,
    )
    _, artifacts = evolve_cirrna(cfg)
    return artifacts.best_sequence


def optimize_for_immune_safety(seq: str, rounds: int = 3) -> str:
    """优化 circRNA 免疫安全性。"""
    cfg = CircRNAEvolutionConfig(
        rounds=rounds, seed_seq=seq,
        weight_immune_evasion=0.5, weight_stability=0.25,
        weight_translation=0.15, weight_delivery=0.10,
    )
    _, artifacts = evolve_cirrna(cfg)
    return artifacts.best_sequence
