"""药物分子进化 — REINFORCE + Pareto 多目标优化

内化自 confluencia-2.0-drug/core/evolution.py 的分子进化部分。

算法: epsilon-greedy 策略选择动作 (ed2mol/mutate_light/mutate_heavy)，
Pareto 导向多目标权重搜索，REINFORCE 策略梯度更新，自适应风险门控。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
from .actions import MOLECULE_ACTIONS


@dataclass
class EvolutionConfig:
    """药物分子进化配置。"""
    rounds: int = 5
    top_k: int = 12
    candidates_per_round: int = 48
    epsilon: float = 0.15
    lr: float = 0.06
    dose: float = 2.0
    freq: float = 1.0
    treatment_time: float = 24.0
    group_id: str = "EVO"
    epitope_seq: str = "SLYNTVATL"
    compute_mode: str = "low"
    use_pareto_search: bool = True
    pareto_weight_samples: int = 64
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    adaptive_enabled: bool = False
    adaptive_strength: float = 0.2
    use_adaptive_gate_penalty: bool = True
    risk_gate_threshold: float = 0.70
    risk_gate_penalty: float = 0.20
    risk_gate_threshold_mode: str = "fixed"
    risk_gate_threshold_quantile: float = 0.80


@dataclass
class EvolutionArtifacts:
    """分子进化输出产物。"""
    final_policy_logits: Dict[str, float]
    reflections: List[str]
    used_ed2mol: bool
    selected_objective_weights: Dict[str, float]
    rounds_ran: int
    best_reward: float
    per_round_best: List[float]


def _mutate_smiles(smiles: str, heavy: bool, rng: np.random.Generator) -> str:
    """SMILES 字符级随机突变。"""
    s = str(smiles or "").strip()
    if not s:
        return "CCO"
    atoms_light = ["C", "N", "O", "F"]
    atoms_heavy = ["C", "N", "O", "F", "Cl", "Br", "S"]
    atoms = atoms_heavy if heavy else atoms_light

    mode = int(rng.integers(0, 3 if heavy else 2))
    if mode == 0 and len(s) > 1:
        i = int(rng.integers(0, len(s)))
        return s[:i] + str(rng.choice(atoms)) + s[i + 1:]
    if mode == 1:
        return s + str(rng.choice(atoms))
    i = int(rng.integers(0, len(s)))
    return s[:i] + "=" + s[i:]


def _objective_matrix(df: pd.DataFrame, risk_gate_threshold: float) -> np.ndarray:
    """构建 7 目标矩阵 (全部最大化方向)。"""
    def _series(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros((len(df),), dtype=np.float32), index=df.index)

    tox = _series("toxicity_risk_pred").to_numpy(dtype=np.float32)
    infl = _series("inflammation_risk_pred").to_numpy(dtype=np.float32)
    risk_gate = np.maximum(tox, infl)
    thr = float(np.clip(risk_gate_threshold, 0.1, 0.99))
    gate_excess = np.clip((risk_gate - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)

    return np.column_stack([
        df["efficacy_pred"].to_numpy(dtype=np.float32),
        df["target_binding_pred"].to_numpy(dtype=np.float32),
        df["immune_cell_activation_pred"].to_numpy(dtype=np.float32),
        -infl,
        -tox,
        -df["ctm_peak_toxicity"].to_numpy(dtype=np.float32),
        -gate_excess.astype(np.float32),
    ]).astype(np.float32)


def _resolve_risk_gate_threshold(risk_gate: np.ndarray, cfg: EvolutionConfig) -> float:
    """解析风险门控阈值 (固定/分位数模式)。"""
    mode = str(getattr(cfg, "risk_gate_threshold_mode", "fixed")).strip().lower()
    if mode == "quantile":
        q = float(np.clip(getattr(cfg, "risk_gate_threshold_quantile", 0.80), 0.50, 0.99))
        if risk_gate.size > 0:
            thr = float(np.quantile(risk_gate.astype(np.float32), q))
            return float(np.clip(thr, 0.10, 0.99))
    return float(np.clip(getattr(cfg, "risk_gate_threshold", 0.70), 0.10, 0.99))


def evolve_molecules_with_reflection(
    seed_smiles: List[str],
    cfg: EvolutionConfig,
    pipeline_fn=None,
    ed2mol_adapter=None,
) -> Tuple[pd.DataFrame, EvolutionArtifacts]:
    """反思式 RL 分子进化。

    Args:
        seed_smiles: 种子 SMILES 列表
        cfg: 进化配置
        pipeline_fn: 预测管线函数 (替代 2.0 的 run_pipeline)
        ed2mol_adapter: ED2Mol 适配器实例 (可选)

    Returns:
        (结果 DataFrame, 进化产物)
    """
    rng = np.random.default_rng(42)
    seeds = [str(x).strip() for x in seed_smiles if str(x).strip()]
    if not seeds:
        seeds = ["CCO", "CCN(CC)CC"]

    logits = np.zeros((len(MOLECULE_ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[pd.DataFrame] = []
    used_ed2mol_any = False
    selected_w = np.array([1.0, 0.50, 0.45, 0.50, 0.35, 0.20, 0.40], dtype=np.float32)
    best_reward_global = -1e9
    no_improve_rounds = 0
    per_round_best: List[float] = []

    current_pool = seeds.copy()
    rounds_ran = 0

    for rd in range(int(max(cfg.rounds, 1))):
        rounds_ran = rd + 1
        n = int(max(cfg.candidates_per_round, 6))
        action_idx = pick_actions(logits, n=n, eps=cfg.epsilon,
                                  n_actions=len(MOLECULE_ACTIONS), rng=rng)

        candidates: List[str] = []
        actions: List[str] = []

        # ED2Mol 分支
        ed_slots = int(np.sum(action_idx == 0))
        ed_generated: List[str] = []
        if ed_slots > 0 and ed2mol_adapter is not None:
            try:
                ed_ret = ed2mol_adapter.generate(max_count=max(ed_slots * 2, 16), timeout_sec=300)
                ed_generated = ed_ret.smiles
                used_ed2mol_any = used_ed2mol_any or (not getattr(ed_ret, 'used_fallback', True) and len(ed_generated) > 0)
                reflections.append(f"Round {rd+1}: ED2Mol generated={len(ed_generated)}")
            except Exception as e:
                reflections.append(f"Round {rd+1}: ED2Mol failed: {e}")

        for ai in action_idx.tolist():
            base = str(rng.choice(current_pool))
            if ai == 0:
                if ed_generated:
                    candidates.append(str(rng.choice(ed_generated)))
                else:
                    candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(MOLECULE_ACTIONS[0])
            elif ai == 1:
                candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(MOLECULE_ACTIONS[1])
            else:
                candidates.append(_mutate_smiles(base, heavy=True, rng=rng))
                actions.append(MOLECULE_ACTIONS[2])

        # 评估候选分子
        if pipeline_fn is not None:
            try:
                cand_df = pd.DataFrame({
                    "smiles": candidates,
                    "epitope_seq": [cfg.epitope_seq] * len(candidates),
                    "dose": [cfg.dose] * len(candidates),
                    "freq": [cfg.freq] * len(candidates),
                    "treatment_time": [cfg.treatment_time] * len(candidates),
                    "group_id": [cfg.group_id] * len(candidates),
                })
                pred_df = pipeline_fn(cand_df, compute_mode=cfg.compute_mode)
            except Exception:
                pred_df = pd.DataFrame({"smiles": candidates})
                for col in ["efficacy_pred", "target_binding_pred", "immune_cell_activation_pred",
                            "inflammation_risk_pred", "toxicity_risk_pred", "ctm_peak_toxicity"]:
                    pred_df[col] = 0.0
        else:
            # 无管线时使用随机评分
            pred_df = pd.DataFrame({"smiles": candidates})
            pred_df["efficacy_pred"] = rng.random(len(candidates)).astype(np.float32)
            pred_df["target_binding_pred"] = rng.random(len(candidates)).astype(np.float32)
            pred_df["immune_cell_activation_pred"] = rng.random(len(candidates)).astype(np.float32)
            pred_df["inflammation_risk_pred"] = rng.random(len(candidates)).astype(np.float32) * 0.3
            pred_df["toxicity_risk_pred"] = rng.random(len(candidates)).astype(np.float32) * 0.3
            pred_df["ctm_peak_toxicity"] = rng.random(len(candidates)).astype(np.float32) * 0.2

        pred_df = pred_df.copy()
        pred_df["round"] = int(rd + 1)
        pred_df["action"] = actions

        # 风险门控
        tox_arr = pd.to_numeric(pred_df.get("toxicity_risk_pred", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        infl_arr = pd.to_numeric(pred_df.get("inflammation_risk_pred", 0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        risk_gate_arr = np.maximum(tox_arr, infl_arr)
        gate_thr = _resolve_risk_gate_threshold(risk_gate_arr, cfg)

        obj = _objective_matrix(pred_df, risk_gate_threshold=float(gate_thr))
        obj_norm = normalize_cols(obj)

        if cfg.use_pareto_search:
            prior = np.array([1.0, 0.50, 0.45, 0.50, 0.35, 0.20, 0.40], dtype=np.float32)
            selected_w = select_weights_with_pareto(
                X_obj_norm=obj_norm, top_k=cfg.top_k,
                n_samples=cfg.pareto_weight_samples, rng=rng, prior=prior,
            )

        pred_df["reward"] = reward_from_weights(obj_norm, selected_w)
        pred_df["pareto_front"] = pareto_front_mask(obj)

        # 自适应风险门控惩罚
        if bool(cfg.use_adaptive_gate_penalty):
            thr = float(gate_thr)
            pcoef = float(max(cfg.risk_gate_penalty, 0.0))
            over = np.clip((risk_gate_arr - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
            gate_penalty = (pcoef * over).astype(np.float32)
            pred_df["reward"] = (pred_df["reward"].to_numpy(dtype=np.float32) - gate_penalty).astype(np.float32)

        # REINFORCE 策略更新
        r = pred_df["reward"].to_numpy(dtype=np.float32)
        r_center = r - float(r.mean())
        for i, act in enumerate(actions):
            aidx = MOLECULE_ACTIONS.index(str(act))
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        # 反思日志
        reflections.append(
            f"Round {rd+1}: reward_mean={float(r.mean()):.4f}, pareto={int(pred_df['pareto_front'].sum())}"
        )

        round_best = float(pred_df["reward"].max()) if not pred_df.empty else -1e9
        per_round_best.append(round_best)
        if round_best > best_reward_global + float(cfg.min_improve):
            best_reward_global = round_best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= int(max(cfg.early_stop_patience, 1)):
            reflections.append(f"Round {rd+1}: early-stop (no improvement for {no_improve_rounds} rounds)")
            all_rows.append(pred_df)
            break

        top_df = pred_df.sort_values("reward", ascending=False).head(int(max(cfg.top_k, 2))).copy()
        current_pool = top_df["smiles"].astype(str).tolist()
        all_rows.append(pred_df)

    out = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    logits_dict = {k: float(v) for k, v in zip(MOLECULE_ACTIONS, logits.tolist())}
    w_names = ["efficacy", "binding", "immune_cell", "low_inflammation", "low_toxicity", "low_ctm_toxicity", "low_gate_excess"]
    w_dict = {k: float(v) for k, v in zip(w_names, selected_w.tolist())}
    art = EvolutionArtifacts(
        final_policy_logits=logits_dict,
        reflections=reflections,
        used_ed2mol=used_ed2mol_any,
        selected_objective_weights=w_dict,
        rounds_ran=int(rounds_ran),
        best_reward=float(best_reward_global if best_reward_global > -1e8 else 0.0),
        per_round_best=per_round_best,
    )
    return out, art
