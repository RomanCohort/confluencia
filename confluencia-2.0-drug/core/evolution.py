from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .ed2mol_adapter import ED2MolAdapter
from .pipeline import run_pipeline


ACTIONS = ["ed2mol", "mutate_light", "mutate_heavy"]

# circRNA-specific evolution actions
CIRCRNA_ACTIONS = ["mutate_backbone", "optimize_ires", "shuffle_utr", "add_modification"]


@dataclass
class EvolutionConfig:
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
    final_policy_logits: Dict[str, float]
    reflections: List[str]
    used_ed2mol: bool
    selected_objective_weights: Dict[str, float]
    rounds_ran: int
    best_reward: float
    per_round_best: List[float]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _mutate_smiles(smiles: str, heavy: bool, rng: np.random.Generator) -> str:
    s = str(smiles or "").strip()
    if not s:
        return "CCO"
    atoms_light = ["C", "N", "O", "F"]
    atoms_heavy = ["C", "N", "O", "F", "Cl", "Br", "S"]
    atoms = atoms_heavy if heavy else atoms_light

    mode = int(rng.integers(0, 3 if heavy else 2))
    if mode == 0 and len(s) > 1:
        i = int(rng.integers(0, len(s)))
        return s[:i] + str(rng.choice(atoms)) + s[i + 1 :]
    if mode == 1:
        return s + str(rng.choice(atoms))
    i = int(rng.integers(0, len(s)))
    return s[:i] + "=" + s[i:]


def _pick_actions(logits: np.ndarray, n: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    probs = _softmax(logits)
    acts = []
    for _ in range(n):
        if float(rng.random()) < float(eps):
            acts.append(int(rng.integers(0, len(ACTIONS))))
        else:
            acts.append(int(rng.choice(np.arange(len(ACTIONS)), p=probs)))
    return np.array(acts, dtype=int)


def _objective_matrix(df: pd.DataFrame, risk_gate_threshold: float) -> np.ndarray:
    # Convert all objectives to maximization direction.
    def _series(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(np.zeros((len(df),), dtype=np.float32), index=df.index)

    tox = _series("toxicity_risk_pred").to_numpy(dtype=np.float32)
    infl = _series("inflammation_risk_pred").to_numpy(dtype=np.float32)
    risk_gate = np.maximum(tox, infl)
    thr = float(np.clip(risk_gate_threshold, 0.1, 0.99))
    gate_excess = np.clip((risk_gate - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)

    return np.column_stack(
        [
            df["efficacy_pred"].to_numpy(dtype=np.float32),
            df["target_binding_pred"].to_numpy(dtype=np.float32),
            df["immune_cell_activation_pred"].to_numpy(dtype=np.float32),
            -infl,
            -tox,
            -df["ctm_peak_toxicity"].to_numpy(dtype=np.float32),
            -gate_excess.astype(np.float32),
        ]
    ).astype(np.float32)


def _resolve_risk_gate_threshold(risk_gate: np.ndarray, cfg: EvolutionConfig) -> float:
    mode = str(getattr(cfg, "risk_gate_threshold_mode", "fixed")).strip().lower()
    if mode == "quantile":
        q = float(np.clip(getattr(cfg, "risk_gate_threshold_quantile", 0.80), 0.50, 0.99))
        if risk_gate.size > 0:
            thr = float(np.quantile(risk_gate.astype(np.float32), q))
            return float(np.clip(thr, 0.10, 0.99))
    return float(np.clip(getattr(cfg, "risk_gate_threshold", 0.70), 0.10, 0.99))


def _normalize_cols(X: np.ndarray) -> np.ndarray:
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    den = np.maximum(mx - mn, 1e-6)
    return (X - mn) / den


def _pareto_front_mask(X: np.ndarray) -> np.ndarray:
    # True if point is non-dominated for a maximization objective matrix.
    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def _select_weights_with_pareto(
    X_obj_norm: np.ndarray,
    top_k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    d = X_obj_norm.shape[1]
    # Include a prior hand-tuned vector then random dirichlet candidates.
    prior = np.array([1.0, 0.50, 0.45, 0.50, 0.35, 0.20, 0.40], dtype=np.float32)
    if int(prior.size) < int(d):
        prior = np.pad(prior, (0, int(d) - int(prior.size)), constant_values=0.20)
    if int(prior.size) > int(d):
        prior = prior[: int(d)]
    bank = [prior.astype(np.float32)]
    for _ in range(max(int(n_samples), 4) - 1):
        bank.append(rng.dirichlet(np.ones(d, dtype=np.float32)).astype(np.float32))

    best_w = bank[0]
    best_score = -1e9
    for w in bank:
        w = w / np.maximum(w.sum(), 1e-8)
        r = (X_obj_norm @ w).astype(np.float32)
        top = np.sort(r)[-max(int(top_k), 2):]
        score = float(top.mean())
        if score > best_score:
            best_score = score
            best_w = w
    return best_w.astype(np.float32)


def _reward_from_weights(X_obj_norm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.maximum(w.sum(), 1e-8)
    return (X_obj_norm @ w).astype(np.float32)


def evolve_molecules_with_reflection(
    seed_smiles: List[str],
    cfg: EvolutionConfig,
    ed2mol_repo_dir: str,
    ed2mol_config_path: str,
    ed2mol_python_cmd: str = "python",
) -> Tuple[pd.DataFrame, EvolutionArtifacts]:
    rng = np.random.default_rng(42)
    seeds = [str(x).strip() for x in seed_smiles if str(x).strip()]
    if not seeds:
        seeds = ["CCO", "CCN(CC)CC"]

    adapter = ED2MolAdapter(repo_dir=ed2mol_repo_dir, python_cmd=ed2mol_python_cmd)
    logits = np.zeros((len(ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[pd.DataFrame] = []
    used_ed2mol_any = False
    selected_w = np.array([1.0, 0.50, 0.45, 0.50, 0.35, 0.20, 0.40], dtype=np.float32)
    best_reward_global = -1e9
    no_improve_rounds = 0
    per_round_best: List[float] = []
    pred_cache: Dict[Tuple[str, str, float, float, float, str], Dict[str, float | str]] = {}

    current_pool = seeds.copy()
    rounds_ran = 0
    for rd in range(int(max(cfg.rounds, 1))):
        rounds_ran = rd + 1
        n = int(max(cfg.candidates_per_round, 6))
        action_idx = _pick_actions(logits, n=n, eps=cfg.epsilon, rng=rng)

        candidates: List[str] = []
        actions: List[str] = []

        # ED2Mol branch (shared generation, then sampled into slots)
        ed_slots = int(np.sum(action_idx == 0))
        ed_generated: List[str] = []
        if ed_slots > 0:
            ed_ret = adapter.generate(config_path=ed2mol_config_path, max_count=max(ed_slots * 2, 16), timeout_sec=300)
            ed_generated = ed_ret.smiles
            used_ed2mol_any = used_ed2mol_any or (not ed_ret.used_fallback and len(ed_generated) > 0)
            reflections.append(f"Round {rd+1}: ED2Mol -> {ed_ret.message}, generated={len(ed_generated)}")

        for ai in action_idx.tolist():
            base = str(rng.choice(current_pool))
            if ai == 0:
                if ed_generated:
                    candidates.append(str(rng.choice(ed_generated)))
                else:
                    candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(ACTIONS[0])
            elif ai == 1:
                candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(ACTIONS[1])
            else:
                candidates.append(_mutate_smiles(base, heavy=True, rng=rng))
                actions.append(ACTIONS[2])

        cache_rows: List[Dict[str, float | str]] = []
        miss_rows: List[Dict[str, float | str]] = []
        for smi in candidates:
            key = (
                str(smi),
                str(cfg.epitope_seq),
                float(cfg.dose),
                float(cfg.freq),
                float(cfg.treatment_time),
                str(cfg.compute_mode),
            )
            if key in pred_cache:
                cached = {
                    **pred_cache[key],
                    "smiles": str(smi),
                    "epitope_seq": str(cfg.epitope_seq),
                    "dose": float(cfg.dose),
                    "freq": float(cfg.freq),
                    "treatment_time": float(cfg.treatment_time),
                    "group_id": str(cfg.group_id),
                }
                cache_rows.append(cached)
            else:
                miss_rows.append(
                    {
                        "smiles": str(smi),
                        "epitope_seq": str(cfg.epitope_seq),
                        "dose": float(cfg.dose),
                        "freq": float(cfg.freq),
                        "treatment_time": float(cfg.treatment_time),
                        "group_id": str(cfg.group_id),
                    }
                )

        miss_df = pd.DataFrame(miss_rows)
        pred_parts: List[pd.DataFrame] = []
        if not miss_df.empty:
            pred_miss, _, _ = run_pipeline(
                miss_df,
                compute_mode=cfg.compute_mode,
                adaptive_enabled=bool(cfg.adaptive_enabled),
                adaptive_strength=float(cfg.adaptive_strength),
            )
            pred_parts.append(pred_miss)
            for _, r in pred_miss.iterrows():
                key = (
                    str(r.get("smiles", "")),
                    str(r.get("epitope_seq", "")),
                    float(r.get("dose", 0.0)),
                    float(r.get("freq", 0.0)),
                    float(r.get("treatment_time", 0.0)),
                    str(cfg.compute_mode),
                )
                pred_cache[key] = {
                    "efficacy_pred": float(r.get("efficacy_pred", 0.0)),
                    "target_binding_pred": float(r.get("target_binding_pred", 0.0)),
                    "immune_activation_pred": float(r.get("immune_activation_pred", 0.0)),
                    "immune_cell_activation_pred": float(r.get("immune_cell_activation_pred", 0.0)),
                    "inflammation_risk_pred": float(r.get("inflammation_risk_pred", 0.0)),
                    "toxicity_risk_pred": float(r.get("toxicity_risk_pred", 0.0)),
                    "ctm_peak_toxicity": float(r.get("ctm_peak_toxicity", 0.0)),
                }

        if cache_rows:
            pred_parts.append(pd.DataFrame(cache_rows))

        pred_df = pd.concat(pred_parts, axis=0, ignore_index=True) if pred_parts else pd.DataFrame()
        pred_df = pred_df.copy()
        pred_df["round"] = int(rd + 1)
        pred_df["action"] = actions

        def _risk_series(name: str) -> pd.Series:
            if name in pred_df.columns:
                return pd.to_numeric(pred_df[name], errors="coerce").fillna(0.0)
            return pd.Series(np.zeros((len(pred_df),), dtype=np.float32), index=pred_df.index)

        tox_arr = _risk_series("toxicity_risk_pred").to_numpy(dtype=np.float32)
        infl_arr = _risk_series("inflammation_risk_pred").to_numpy(dtype=np.float32)
        risk_gate_arr = np.maximum(tox_arr, infl_arr)
        gate_thr = _resolve_risk_gate_threshold(risk_gate_arr, cfg)

        obj = _objective_matrix(pred_df, risk_gate_threshold=float(gate_thr))
        obj_norm = _normalize_cols(obj)
        if cfg.use_pareto_search:
            selected_w = _select_weights_with_pareto(
                X_obj_norm=obj_norm,
                top_k=cfg.top_k,
                n_samples=cfg.pareto_weight_samples,
                rng=rng,
            )
        pred_df["reward"] = _reward_from_weights(obj_norm, selected_w)
        pred_df["reward_raw"] = pred_df["reward"].astype(np.float32)
        pred_df["obj_efficacy"] = obj[:, 0]
        pred_df["obj_binding"] = obj[:, 1]
        pred_df["obj_immune_cell"] = obj[:, 2]
        pred_df["obj_low_inflammation"] = obj[:, 3]
        pred_df["obj_low_toxicity"] = obj[:, 4]
        pred_df["obj_low_ctm_toxicity"] = obj[:, 5]
        pred_df["obj_low_gate_excess"] = obj[:, 6]
        pred_df["pareto_front"] = _pareto_front_mask(obj)
        pred_df["risk_gate_threshold_used"] = float(gate_thr)

        review_ratio = 0.0
        mean_penalty = 0.0
        penalty_shift_l1 = 0.0
        penalty_shift_max_action = "n/a"
        if bool(cfg.use_adaptive_gate_penalty) and all(
            c in pred_df.columns for c in ["toxicity_risk_pred", "inflammation_risk_pred"]
        ):
            risk_gate = risk_gate_arr
            thr = float(gate_thr)
            pcoef = float(max(cfg.risk_gate_penalty, 0.0))
            over = np.clip((risk_gate - thr) / max(1.0 - thr, 1e-6), 0.0, 1.0)
            gate_penalty = (pcoef * over).astype(np.float32)

            pred_df["adaptive_gate_flag"] = np.where(risk_gate >= thr, "review", "ok")
            pred_df["reward_penalty_gate"] = gate_penalty
            pred_df["reward"] = (pred_df["reward"].to_numpy(dtype=np.float32) - gate_penalty).astype(np.float32)

            review_ratio = float(np.mean(risk_gate >= thr)) if len(risk_gate) > 0 else 0.0
            mean_penalty = float(np.mean(gate_penalty)) if len(gate_penalty) > 0 else 0.0

            # Quantify policy update shift introduced by risk-gate penalty.
            raw_r = pred_df["reward_raw"].to_numpy(dtype=np.float32)
            pen_r = pred_df["reward"].to_numpy(dtype=np.float32)
            raw_c = raw_r - float(raw_r.mean())
            pen_c = pen_r - float(pen_r.mean())
            shift_per_action = np.zeros((len(ACTIONS),), dtype=np.float32)
            for i, act in enumerate(actions):
                aidx = ACTIONS.index(str(act))
                shift_per_action[aidx] += float(cfg.lr) * float(pen_c[i] - raw_c[i])
            penalty_shift_l1 = float(np.sum(np.abs(shift_per_action)))
            penalty_shift_max_action = str(ACTIONS[int(np.argmax(np.abs(shift_per_action)))])
        else:
            pred_df["adaptive_gate_flag"] = "n/a"
            pred_df["reward_penalty_gate"] = np.zeros((len(pred_df),), dtype=np.float32)

        # Pure RL (REINFORCE-like): update action logits by centered reward.
        r = pred_df["reward"].to_numpy(dtype=np.float32)
        r_center = r - float(r.mean())
        for i, act in enumerate(actions):
            aidx = ACTIONS.index(str(act))
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        # Reflection note from action-level outcomes.
        act_means = pred_df.groupby("action", as_index=False).agg(reward=("reward", "mean"))
        act_means = act_means.sort_values(by="reward", ascending=False)
        best_act = str(act_means.iloc[0]["action"])
        worst_act = str(act_means.iloc[-1]["action"])
        reflections.append(
            f"Round {rd+1}: best_action={best_act}, worst_action={worst_act}, reward_mean={float(r.mean()):.4f}, pareto={int(pred_df['pareto_front'].sum())}, gate_thr={gate_thr:.3f}, review_ratio={review_ratio:.2%}, gate_penalty_mean={mean_penalty:.4f}, policy_shift_l1={penalty_shift_l1:.4f}, shift_peak_action={penalty_shift_max_action}"
        )

        round_best = float(pred_df["reward"].max()) if not pred_df.empty else -1e9
        per_round_best.append(round_best)
        if round_best > best_reward_global + float(cfg.min_improve):
            best_reward_global = round_best
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= int(max(cfg.early_stop_patience, 1)):
            reflections.append(
                f"Round {rd+1}: early-stop triggered (no improvement for {no_improve_rounds} rounds)"
            )
            all_rows.append(pred_df)
            break

        # Keep top molecules for next round.
        top_df = pred_df.sort_values("reward", ascending=False).head(int(max(cfg.top_k, 2))).copy()
        current_pool = top_df["smiles"].astype(str).tolist()
        all_rows.append(pred_df)

    out = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    logits_dict = {k: float(v) for k, v in zip(ACTIONS, logits.tolist())}
    w_names = [
        "efficacy",
        "binding",
        "immune_cell",
        "low_inflammation",
        "low_toxicity",
        "low_ctm_toxicity",
        "low_gate_excess",
    ]
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


# ===================================================================
# circRNA sequence evolution
# ===================================================================

@dataclass
class CircRNAEvolutionConfig:
    """Configuration for circRNA sequence evolution."""
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
    dose: float = 2.0
    freq: float = 1.0
    treatment_time: float = 24.0
    use_pareto_search: bool = True
    pareto_weight_samples: int = 32
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    seed: int = 42


def _mutate_cirrna_backbone(seq: str, rng: np.random.Generator, n_mutations: int = 3) -> str:
    """Introduce point mutations into circRNA backbone sequence.

    Preserves the overall length and avoids changing the first/last
    nucleotides (near the backsplice junction).
    """
    s = list(seq.upper())
    if len(s) < 10:
        return seq

    # Protect first 5 and last 5 nt (backsplice junction region)
    protected_start = min(5, len(s) // 4)
    protected_end = min(5, len(s) // 4)
    mutable_range = range(protected_start, len(s) - protected_end)
    if len(mutable_range) < 3:
        mutable_range = range(len(s))

    nt_choices = ["A", "U", "G", "C"]
    n_mut = min(n_mutations, len(mutable_range))

    indices = rng.choice(list(mutable_range), size=n_mut, replace=False)
    for idx in indices:
        current = s[idx]
        alternatives = [nt for nt in nt_choices if nt != current]
        s[idx] = str(rng.choice(alternatives))

    return "".join(s)


def _optimize_ires_region(seq: str, rng: np.random.Generator) -> str:
    """Optimize IRES region for stronger translation initiation.

    Inserts or replaces subsequences with known IRES-enhancing motifs.
    """
    s = list(seq.upper())
    if len(s) < 30:
        return seq

    # Strong IRES motifs to insert/replace
    strong_motifs = ["GCGCC", "CCUG", "GGGG", "UUGU", "AUGG"]

    # With 50% chance, insert a motif; with 50%, replace a region
    if rng.random() < 0.5 and len(s) > 40:
        # Insert a strong motif at a random position (avoid first/last 10)
        pos = int(rng.integers(10, len(s) - 10))
        motif = str(rng.choice(strong_motifs))
        s[pos:pos + len(motif)] = list(motif)
    else:
        # Replace a short region with a strong motif
        pos = int(rng.integers(5, max(6, len(s) - 10)))
        motif = str(rng.choice(strong_motifs))
        # Only replace if the region is similar length
        replace_len = min(len(motif), len(s) - pos - 5)
        if replace_len > 2:
            s[pos:pos + replace_len] = list(motif[:replace_len])

    return "".join(s)


def _shuffle_utr(seq: str, rng: np.random.Generator) -> str:
    """Shuffle UTR-like regions (regions before first AUG and after last stop codon)."""
    s = seq.upper()
    if len(s) < 30:
        return seq

    # Find first AUG (start codon)
    first_aug = s.find("AUG")
    if first_aug < 0:
        first_aug = len(s) // 3

    # Find last stop codon
    stop_codons = ["UAA", "UAG", "UGA"]
    last_stop = -1
    for sc in stop_codons:
        idx = s.rfind(sc)
        if idx > last_stop:
            last_stop = idx + 3

    if last_stop <= first_aug:
        last_stop = len(s)

    # 5' UTR: before first AUG
    utr5_end = min(first_aug, len(s) - 1)
    if utr5_end > 3:
        utr5 = list(s[:utr5_end])
        rng.shuffle(utr5)
        s = "".join(utr5) + s[utr5_end:]

    # 3' UTR: after last stop codon
    if last_stop < len(s) - 3:
        utr3 = list(s[last_stop:])
        rng.shuffle(utr3)
        s = s[:last_stop] + "".join(utr3)

    return s


def _pick_cirrna_actions(
    logits: np.ndarray, n: int, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Epsilon-greedy action selection for circRNA evolution."""
    probs = _softmax(logits)
    acts = []
    for _ in range(n):
        if float(rng.random()) < float(eps):
            acts.append(int(rng.integers(0, len(CIRCRNA_ACTIONS))))
        else:
            acts.append(int(rng.choice(np.arange(len(CIRCRNA_ACTIONS)), p=probs)))
    return np.array(acts, dtype=int)


def _cirrna_objective_matrix(
    features: np.ndarray,
    feature_names: List[str],
    innate_scores: List[Dict[str, float]],
) -> np.ndarray:
    """Build objective matrix for circRNA evolution.

    Objectives (all maximization):
      - stability_score
      - translation_efficiency (IRES + ORF)
      - immune_evasion (safety)
      - delivery_compatibility
    """
    n = len(features)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Extract feature indices
    def _feat_idx(name: str) -> int:
        if name in feature_names:
            return feature_names.index(name)
        return -1

    stab_idx = _feat_idx("crna_struct_stability")
    mod_stab_idx = _feat_idx("crna_mod_stability_factor")
    ires_idx = _feat_idx("crna_ires_score")
    orf_idx = _feat_idx("crna_orf_length")
    kozak_idx = _feat_idx("crna_kozak_score")
    bio_idx = _feat_idx("crna_bioavailability")
    escape_idx = _feat_idx("crna_endosomal_escape")

    objs = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        # Stability: struct_stability * mod_stability_factor
        stab = 0.5
        if stab_idx >= 0:
            stab *= features[i, stab_idx]
        if mod_stab_idx >= 0:
            stab *= min(float(features[i, mod_stab_idx]), 3.0) / 3.0
        objs[i, 0] = float(np.clip(stab, 0.0, 1.0))

        # Translation: IRES * ORF * Kozak
        trans = 0.3
        if ires_idx >= 0:
            trans += 0.4 * features[i, ires_idx]
        if orf_idx >= 0:
            trans += 0.2 * features[i, orf_idx]
        if kozak_idx >= 0:
            trans += 0.1 * features[i, kozak_idx]
        objs[i, 1] = float(np.clip(trans, 0.0, 1.0))

        # Immune evasion: from innate immune module
        if i < len(innate_scores):
            objs[i, 2] = float(np.clip(innate_scores[i].get("innate_safety_score", 0.5), 0.0, 1.0))
        else:
            objs[i, 2] = 0.5

        # Delivery compatibility: bioavailability * endosomal_escape
        delivery = 0.3
        if bio_idx >= 0:
            delivery += 0.5 * features[i, bio_idx]
        if escape_idx >= 0:
            delivery += 0.2 * features[i, escape_idx]
        objs[i, 3] = float(np.clip(delivery, 0.0, 1.0))

    return objs


def _cirrna_modification_pool() -> List[str]:
    return ["none", "m6A", "Psi", "5mC", "ms2m6A"]


def evolve_cirrna_sequences(cfg: CircRNAEvolutionConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run circRNA sequence evolution with reflection-based RL.

    Evolves circRNA backbone sequences and modification strategies to optimize:
    stability, translation efficiency, immune evasion, and delivery compatibility.

    Returns:
        (results_df, artifacts_dict)
    """
    from .features import build_cirrna_feature_matrix, build_cirrna_feature_vector
    from .innate_immune import assess_innate_immune, innate_immune_result_to_dict

    rng = np.random.default_rng(cfg.seed)
    logits = np.zeros((len(CIRCRNA_ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[Dict[str, Any]] = []

    current_pool: List[str] = [cfg.seed_seq]
    current_mods: List[str] = [cfg.modification]
    best_reward_global = -1e9
    no_improve = 0
    per_round_best: List[float] = []
    rounds_ran = 0

    for rd in range(max(cfg.rounds, 1)):
        rounds_ran = rd + 1
        n = max(cfg.candidates_per_round, 4)
        action_idx = _pick_cirrna_actions(logits, n=n, eps=cfg.epsilon, rng=rng)

        candidates: List[str] = []
        mod_candidates: List[str] = []
        actions: List[str] = []

        for ai in action_idx.tolist():
            base_seq = str(rng.choice(current_pool))
            base_mod = str(rng.choice(current_mods))

            if ai == 0:  # mutate_backbone
                candidates.append(_mutate_cirrna_backbone(base_seq, rng, n_mutations=rng.integers(1, 5)))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[0])
            elif ai == 1:  # optimize_ires
                candidates.append(_optimize_ires_region(base_seq, rng))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[1])
            elif ai == 2:  # shuffle_utr
                candidates.append(_shuffle_utr(base_seq, rng))
                mod_candidates.append(base_mod)
                actions.append(CIRCRNA_ACTIONS[2])
            else:  # add_modification
                candidates.append(base_seq)
                mod_pool = _cirrna_modification_pool()
                new_mod = str(rng.choice([m for m in mod_pool if m != base_mod] or mod_pool))
                mod_candidates.append(new_mod)
                actions.append(CIRCRNA_ACTIONS[3])

        # Compute features and innate immune scores for all candidates
        feat_vecs = []
        innate_scores = []
        for seq, mod in zip(candidates, mod_candidates):
            vec = build_cirrna_feature_vector(
                seq=seq, modification=mod,
                delivery_vector=cfg.delivery_vector, route=cfg.route,
                ires_type=cfg.ires_type,
            )
            feat_vecs.append(vec)
            ir = assess_innate_immune(seq, mod, cfg.delivery_vector)
            innate_scores.append(innate_immune_result_to_dict(ir))

        features = np.stack(feat_vecs, axis=0) if feat_vecs else np.zeros((0, 1), dtype=np.float32)
        feature_names = [
            "crna_struct_stability", "crna_mod_stability_factor",
            "crna_ires_score", "crna_orf_length", "crna_kozak_score",
            "crna_bioavailability", "crna_endosomal_escape",
        ]

        obj = _cirrna_objective_matrix(features, feature_names, innate_scores)
        obj_norm = _normalize_cols(obj)

        # Pareto weight selection
        selected_w = np.array([0.35, 0.30, 0.25, 0.10], dtype=np.float32)
        if cfg.use_pareto_search and obj_norm.shape[0] >= 2:
            selected_w = _select_weights_with_pareto(
                X_obj_norm=obj_norm, top_k=cfg.top_k,
                n_samples=cfg.pareto_weight_samples, rng=rng,
            )
            if selected_w.shape[0] != 4:
                selected_w = np.array([0.35, 0.30, 0.25, 0.10], dtype=np.float32)

        rewards = _reward_from_weights(obj_norm, selected_w)
        pareto_mask = _pareto_front_mask(obj_norm)

        # Store results
        for i, (seq, mod, act) in enumerate(zip(candidates, mod_candidates, actions)):
            inn = innate_scores[i]
            row = {
                "round": rd + 1,
                "action": act,
                "circrna_seq": seq,
                "modification": mod,
                "delivery_vector": cfg.delivery_vector,
                "route": cfg.route,
                "ires_type": cfg.ires_type,
                "reward": float(rewards[i]),
                "obj_stability": float(obj[i, 0]),
                "obj_translation": float(obj[i, 1]),
                "obj_immune_evasion": float(obj[i, 2]),
                "obj_delivery": float(obj[i, 3]),
                "pareto_front": bool(pareto_mask[i]),
                "innate_immune_score": inn["innate_immune_score"],
                "innate_safety_score": inn["innate_safety_score"],
                "innate_ifn_storm_risk": inn["innate_ifn_storm_risk"],
                "innate_ifn_storm_level": inn["innate_ifn_storm_level"],
                "innate_mod_evasion": inn["innate_mod_evasion"],
            }
            all_rows.append(row)

        # RL policy update (REINFORCE-like)
        r_center = rewards - rewards.mean()
        for i, act in enumerate(actions):
            aidx = CIRCRNA_ACTIONS.index(act)
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        # Reflection
        act_rewards = {}
        for act, rew in zip(actions, rewards.tolist()):
            if act not in act_rewards:
                act_rewards[act] = []
            act_rewards[act].append(rew)
        act_means = {a: float(np.mean(rs)) for a, rs in act_rewards.items()}
        best_act = max(act_means, key=act_means.get) if act_means else "n/a"
        worst_act = min(act_means, key=act_means.get) if act_means else "n/a"
        reflections.append(
            "Round %d: best=%s(%.4f), worst=%s(%.4f), "
            "pareto=%d, reward_mean=%.4f" % (
                rd + 1, best_act, act_means.get(best_act, 0),
                worst_act, act_means.get(worst_act, 0),
                int(pareto_mask.sum()), float(rewards.mean()),
            )
        )

        # Early stopping
        round_best = float(rewards.max()) if len(rewards) > 0 else -1e9
        per_round_best.append(round_best)
        if round_best > best_reward_global + cfg.min_improve:
            best_reward_global = round_best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max(cfg.early_stop_patience, 1):
            reflections.append(
                "Round %d: early-stop (no improve for %d rounds)" % (rd + 1, no_improve)
            )
            break

        # Update pool: top-k sequences
        reward_order = np.argsort(-rewards)
        top_indices = reward_order[:max(cfg.top_k, 2)]
        current_pool = [candidates[int(idx)] for idx in top_indices]
        current_mods = [mod_candidates[int(idx)] for idx in top_indices]

    result_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    logits_dict = {k: float(v) for k, v in zip(CIRCRNA_ACTIONS, logits.tolist())}

    artifacts = {
        "final_policy_logits": logits_dict,
        "reflections": reflections,
        "rounds_ran": rounds_ran,
        "best_reward": float(best_reward_global if best_reward_global > -1e8 else 0.0),
        "per_round_best": per_round_best,
        "selected_weights": {
            "stability": float(selected_w[0]),
            "translation": float(selected_w[1]),
            "immune_evasion": float(selected_w[2]),
            "delivery": float(selected_w[3]),
        },
    }
    return result_df, artifacts
