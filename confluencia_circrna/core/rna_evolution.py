"""
rna_evolution.py — circRNA and Molecule Evolution

Evolutionary optimization for:
1. circRNA sequences (backbone, IRES, UTR, modifications)
2. Small molecules (ED2Mol integration)
3. Pareto-optimal selection
4. Reflection-based reinforcement learning

Literature basis:
- Yang et al., 2017: circRNA optimization for translation
- Liu et al., 2022: m6A modification optimization
- Zhong et al., 2018: circRNA sequence design
- pineappleK/ED2Mol: Structure-based molecule generation

Key concepts:
- Mutation operators: backbone, IRES, UTR shuffle, modification
- Multi-objective optimization: stability, translation, immune, delivery
- Pareto front selection for trade-offs
- RL policy learning from rewards
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from confluencia_circrna.core.ed2mol_adapter import ED2MolAdapter, ED2MolRunResult
from confluencia_circrna.core.ed2mol_templates import build_ed2mol_config_text, write_ed2mol_config

# Evolution actions for molecules
MOLECULE_ACTIONS = ["ed2mol", "mutate_light", "mutate_heavy"]

# Evolution actions for circRNA
CIRCRNA_ACTIONS = ["mutate_backbone", "optimize_ires", "shuffle_utr", "add_modification"]


@dataclass
class EvolutionConfig:
    """Configuration for molecule evolution."""
    rounds: int = 5
    top_k: int = 12
    candidates_per_round: int = 48
    epsilon: float = 0.15          # Exploration rate
    lr: float = 0.06               # Learning rate for policy
    use_pareto_search: bool = True
    pareto_weight_samples: int = 64
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    seed: int = 42


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
    use_pareto_search: bool = True
    pareto_weight_samples: int = 32
    early_stop_patience: int = 3
    min_improve: float = 1e-4
    seed: int = 42


@dataclass
class EvolutionArtifacts:
    """Evolution run artifacts."""
    final_policy_logits: Dict[str, float]
    reflections: List[str]
    used_ed2mol: bool
    selected_objective_weights: Dict[str, float]
    rounds_ran: int
    best_reward: float
    per_round_best: List[float]


# ===================================================================
# Utility functions
# ===================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def _normalize_cols(X: np.ndarray) -> np.ndarray:
    """Normalize columns to [0, 1]."""
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    den = np.maximum(mx - mn, 1e-6)
    return (X - mn) / den


def _pareto_front_mask(X: np.ndarray) -> np.ndarray:
    """Identify Pareto-optimal points."""
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


def _reward_from_weights(X_obj_norm: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted reward."""
    w = np.asarray(weights, dtype=np.float32)
    w = w / np.maximum(w.sum(), 1e-8)
    return (X_obj_norm @ w).astype(np.float32)


def _select_weights_with_pareto(
    X_obj_norm: np.ndarray,
    top_k: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select optimal weights using Pareto search."""
    d = X_obj_norm.shape[1]

    # Prior weights
    prior = np.ones(d, dtype=np.float32) / d
    bank = [prior]

    # Random Dirichlet samples
    for _ in range(max(n_samples, 4) - 1):
        bank.append(rng.dirichlet(np.ones(d, dtype=np.float32)).astype(np.float32))

    best_w = bank[0]
    best_score = -1e9

    for w in bank:
        w = w / np.maximum(w.sum(), 1e-8)
        r = (X_obj_norm @ w).astype(np.float32)
        top = np.sort(r)[-max(top_k, 2):]
        score = float(top.mean())
        if score > best_score:
            best_score = score
            best_w = w

    return best_w.astype(np.float32)


# ===================================================================
# Molecule mutation operators
# ===================================================================

def _mutate_smiles(smiles: str, heavy: bool, rng: np.random.Generator) -> str:
    """Mutate SMILES string."""
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


def _pick_actions(logits: np.ndarray, n: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Epsilon-greedy action selection."""
    probs = _softmax(logits)
    acts = []
    for _ in range(n):
        if float(rng.random()) < float(eps):
            acts.append(int(rng.integers(0, len(MOLECULE_ACTIONS))))
        else:
            acts.append(int(rng.choice(np.arange(len(MOLECULE_ACTIONS)), p=probs)))
    return np.array(acts, dtype=int)


# ===================================================================
# circRNA mutation operators
# ===================================================================

def _mutate_cirrna_backbone(seq: str, rng: np.random.Generator, n_mutations: int = 3) -> str:
    """Introduce point mutations into circRNA backbone."""
    s = list(seq.upper().replace("T", "U"))
    if len(s) < 10:
        return seq

    # Protect backsplice junction region (first/last 5 nt)
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
    """Optimize IRES region for translation."""
    s = list(seq.upper().replace("T", "U"))
    if len(s) < 30:
        return seq

    # Strong IRES motifs
    strong_motifs = ["GCGCC", "CCUG", "GGGG", "UUGU", "AUGG"]

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


def _shuffle_utr(seq: str, rng: np.random.Generator) -> str:
    """Shuffle UTR-like regions."""
    s = seq.upper().replace("T", "U")
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

    # Shuffle 5' UTR
    utr5_end = min(first_aug, len(s) - 1)
    if utr5_end > 3:
        utr5 = list(s[:utr5_end])
        rng.shuffle(utr5)
        s = "".join(utr5) + s[utr5_end:]

    # Shuffle 3' UTR
    if last_stop < len(s) - 3:
        utr3 = list(s[last_stop:])
        rng.shuffle(utr3)
        s = s[:last_stop] + "".join(utr3)

    return s


def _pick_cirrna_actions(logits: np.ndarray, n: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Action selection for circRNA evolution."""
    probs = _softmax(logits)
    acts = []
    for _ in range(n):
        if float(rng.random()) < float(eps):
            acts.append(int(rng.integers(0, len(CIRCRNA_ACTIONS))))
        else:
            acts.append(int(rng.choice(np.arange(len(CIRCRNA_ACTIONS)), p=probs)))
    return np.array(acts, dtype=int)


def _cirrna_modification_pool() -> List[str]:
    """Available modifications for circRNA."""
    return ["none", "m6A", "Psi", "5mC", "ms2m6A", "2OMeA", "2OMeU"]


# ===================================================================
# circRNA objective computation
# ===================================================================

def _compute_cirrna_objectives(
    seq: str,
    modification: str,
    immune_scores: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Compute objective vector for circRNA.

    Objectives (all maximization):
    - Stability (GC content, modification)
    - Translation potential (IRES-like motifs)
    - Immune evasion (low PKR, appropriate RIG-I)
    - Delivery compatibility
    """
    seq = seq.upper().replace("T", "U")
    length = len(seq)

    if length < 50:
        return np.array([0.3, 0.3, 0.5, 0.3], dtype=np.float32)

    gc = sum(1 for c in seq if c in "GC") / length

    # Stability
    stability = 0.3 + gc * 0.5
    if modification in ["m6A", "Psi", "2OMeA"]:
        stability += 0.1  # Modification bonus
    obj0 = np.clip(stability, 0.0, 1.0)

    # Translation potential (IRES-like motifs)
    ires_motifs = ["GCGCC", "GGGG", "UUGU", "AUGG", "CCUG"]
    ires_count = sum(1 for m in ires_motifs if m in seq)
    translation = 0.2 + ires_count * 0.15 + (gc > 0.4) * 0.1
    obj1 = np.clip(translation, 0.0, 1.0)

    # Immune evasion
    if immune_scores:
        pkr = immune_scores.get("pkr_score", 0.3)
        rig_i = immune_scores.get("rig_i_score", 0.3)
        # Lower PKR = better evasion, moderate RIG-I = optimal
        immune_evasion = 1.0 - pkr * 0.6 + (0.2 < rig_i < 0.5) * 0.2
    else:
        # Estimate from sequence
        dsRNA_potential = gc * 0.7 * (length > 500)  # Long GC-rich = more dsRNA
        immune_evasion = 1.0 - dsRNA_potential * 0.5
    obj2 = np.clip(immune_evasion, 0.0, 1.0)

    # Delivery compatibility (shorter, moderate GC better)
    delivery = 0.4
    if length < 2000:
        delivery += 0.2
    if 0.35 < gc < 0.55:
        delivery += 0.2
    if modification in ["m6A", "Psi"]:
        delivery += 0.1
    obj3 = np.clip(delivery, 0.0, 1.0)

    return np.array([obj0, obj1, obj2, obj3], dtype=np.float32)


# ===================================================================
# Main evolution functions
# ===================================================================

def evolve_molecules(
    seed_smiles: List[str],
    cfg: EvolutionConfig,
    ed2mol_repo_dir: Optional[str] = None,
    ed2mol_config_path: Optional[str] = None,
    objective_fn: Optional[callable] = None,
) -> Tuple[pd.DataFrame, EvolutionArtifacts]:
    """
    Evolve molecules using ED2Mol and mutation operators.

    Args:
        seed_smiles: Starting SMILES strings
        cfg: Evolution configuration
        ed2mol_repo_dir: ED2Mol repository path
        ed2mol_config_path: ED2Mol config file
        objective_fn: Function to compute objectives (efficacy, binding, etc.)

    Returns:
        (results_df, artifacts)
    """
    rng = np.random.default_rng(cfg.seed)
    seeds = [str(x).strip() for x in seed_smiles if str(x).strip()]
    if not seeds:
        seeds = ["CCO", "CCN(CC)CC"]

    adapter = None
    if ed2mol_repo_dir and ed2mol_config_path:
        adapter = ED2MolAdapter(repo_dir=ed2mol_repo_dir)

    logits = np.zeros((len(MOLECULE_ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[Dict] = []
    used_ed2mol = False

    current_pool = seeds.copy()
    best_reward_global = -1e9
    no_improve_rounds = 0
    per_round_best: List[float] = []
    rounds_ran = 0

    for rd in range(max(cfg.rounds, 1)):
        rounds_ran = rd + 1
        n = max(cfg.candidates_per_round, 6)
        action_idx = _pick_actions(logits, n=n, eps=cfg.epsilon, rng=rng)

        candidates: List[str] = []
        actions: List[str] = []

        # ED2Mol generation
        ed_slots = int(np.sum(action_idx == 0))
        ed_generated: List[str] = []

        if ed_slots > 0 and adapter and ed2mol_config_path:
            ed_ret = adapter.generate(config_path=ed2mol_config_path, max_count=max(ed_slots * 2, 16))
            ed_generated = ed_ret.smiles
            used_ed2mol = used_ed2mol or (not ed_ret.used_fallback and len(ed_generated) > 0)
            reflections.append(f"Round {rd+1}: ED2Mol -> {ed_ret.message}, generated={len(ed_generated)}")

        for ai in action_idx.tolist():
            base = str(rng.choice(current_pool))

            if ai == 0:  # ed2mol
                if ed_generated:
                    candidates.append(str(rng.choice(ed_generated)))
                else:
                    candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(MOLECULE_ACTIONS[0])

            elif ai == 1:  # mutate_light
                candidates.append(_mutate_smiles(base, heavy=False, rng=rng))
                actions.append(MOLECULE_ACTIONS[1])

            else:  # mutate_heavy
                candidates.append(_mutate_smiles(base, heavy=True, rng=rng))
                actions.append(MOLECULE_ACTIONS[2])

        # Compute objectives
        if objective_fn:
            obj_matrix = np.array([objective_fn(s) for s in candidates], dtype=np.float32)
        else:
            # Default objectives (placeholder)
            obj_matrix = np.zeros((len(candidates), 7), dtype=np.float32)
            for i, smi in enumerate(candidates):
                # Simple heuristic objectives
                mw = len(smi) * 10  # Approximate MW
                obj_matrix[i, 0] = np.clip(mw / 500, 0.0, 1.0)  # Size (moderate optimal)
                obj_matrix[i, 1] = np.clip((len(smi) - 5) / 20, 0.0, 1.0)  # Complexity
                obj_matrix[i, 2] = 0.5  # Placeholder binding
                obj_matrix[i, 3] = 0.5  # Placeholder safety
                obj_matrix[i, 4] = 0.5
                obj_matrix[i, 5] = 0.5
                obj_matrix[i, 6] = 0.5

        obj_norm = _normalize_cols(obj_matrix)

        # Pareto weight selection
        selected_w = np.ones(obj_matrix.shape[1], dtype=np.float32) / obj_matrix.shape[1]
        if cfg.use_pareto_search and obj_norm.shape[0] >= 2:
            selected_w = _select_weights_with_pareto(
                X_obj_norm=obj_norm, top_k=cfg.top_k,
                n_samples=cfg.pareto_weight_samples, rng=rng,
            )

        rewards = _reward_from_weights(obj_norm, selected_w)
        pareto_mask = _pareto_front_mask(obj_norm)

        # Store results
        for i, (smi, act) in enumerate(zip(candidates, actions)):
            all_rows.append({
                "round": rd + 1,
                "action": act,
                "smiles": smi,
                "reward": float(rewards[i]),
                "pareto_front": bool(pareto_mask[i]),
            })

        # RL update
        r_center = rewards - rewards.mean()
        for i, act in enumerate(actions):
            aidx = MOLECULE_ACTIONS.index(act)
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        # Reflection
        best_reward = float(rewards.max())
        reflections.append(f"Round {rd+1}: best_reward={best_reward:.4f}, pareto={int(pareto_mask.sum())}")

        # Early stopping
        per_round_best.append(best_reward)
        if best_reward > best_reward_global + cfg.min_improve:
            best_reward_global = best_reward
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        if no_improve_rounds >= max(cfg.early_stop_patience, 1):
            reflections.append(f"Early-stop at round {rd+1}")
            break

        # Update pool
        top_idx = np.argsort(-rewards)[:max(cfg.top_k, 2)]
        current_pool = [candidates[i] for i in top_idx]

    result_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    logits_dict = {k: float(v) for k, v in zip(MOLECULE_ACTIONS, logits.tolist())}
    w_names = ["efficacy", "binding", "immune_cell", "low_inflammation", "low_toxicity", "low_ctm", "low_gate"]
    w_dict = {k: float(v) for k, v in zip(w_names[:selected_w.shape[0]], selected_w.tolist())}

    artifacts = EvolutionArtifacts(
        final_policy_logits=logits_dict,
        reflections=reflections,
        used_ed2mol=used_ed2mol,
        selected_objective_weights=w_dict,
        rounds_ran=rounds_ran,
        best_reward=float(best_reward_global),
        per_round_best=per_round_best,
    )

    return result_df, artifacts


def evolve_cirrna(
    cfg: CircRNAEvolutionConfig,
    immune_score_fn: Optional[callable] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evolve circRNA sequences.

    Args:
        cfg: circRNA evolution configuration
        immune_score_fn: Function to compute immune scores

    Returns:
        (results_df, artifacts_dict)
    """
    rng = np.random.default_rng(cfg.seed)
    logits = np.zeros((len(CIRCRNA_ACTIONS),), dtype=np.float32)
    reflections: List[str] = []
    all_rows: List[Dict] = []

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
                candidates.append(_mutate_cirrna_backbone(base_seq, rng))
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

        # Compute objectives
        obj_matrix = np.zeros((len(candidates), 4), dtype=np.float32)

        for i, (seq, mod) in enumerate(zip(candidates, mod_candidates)):
            immune_scores = None
            if immune_score_fn:
                immune_scores = immune_score_fn(seq)

            obj_matrix[i] = _compute_cirrna_objectives(seq, mod, immune_scores)

        obj_norm = _normalize_cols(obj_matrix)

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
            all_rows.append({
                "round": rd + 1,
                "action": act,
                "circrna_seq": seq,
                "modification": mod,
                "delivery_vector": cfg.delivery_vector,
                "reward": float(rewards[i]),
                "obj_stability": float(obj_matrix[i, 0]),
                "obj_translation": float(obj_matrix[i, 1]),
                "obj_immune_evasion": float(obj_matrix[i, 2]),
                "obj_delivery": float(obj_matrix[i, 3]),
                "pareto_front": bool(pareto_mask[i]),
            })

        # RL update
        r_center = rewards - rewards.mean()
        for i, act in enumerate(actions):
            aidx = CIRCRNA_ACTIONS.index(act)
            logits[aidx] += float(cfg.lr) * float(r_center[i])

        # Reflection
        best_reward = float(rewards.max())
        reflections.append(f"Round {rd+1}: best_reward={best_reward:.4f}, pareto={int(pareto_mask.sum())}")

        # Early stopping
        per_round_best.append(best_reward)
        if best_reward > best_reward_global + cfg.min_improve:
            best_reward_global = best_reward
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max(cfg.early_stop_patience, 1):
            reflections.append(f"Early-stop at round {rd+1}")
            break

        # Update pool
        top_idx = np.argsort(-rewards)[:max(cfg.top_k, 2)]
        current_pool = [candidates[i] for i in top_idx]
        current_mods = [mod_candidates[i] for i in top_idx]

    result_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    logits_dict = {k: float(v) for k, v in zip(CIRCRNA_ACTIONS, logits.tolist())}

    artifacts = {
        "final_policy_logits": logits_dict,
        "reflections": reflections,
        "rounds_ran": rounds_ran,
        "best_reward": float(best_reward_global),
        "per_round_best": per_round_best,
        "selected_weights": {
            "stability": float(selected_w[0]),
            "translation": float(selected_w[1]),
            "immune_evasion": float(selected_w[2]),
            "delivery": float(selected_w[3]),
        },
    }

    return result_df, artifacts


# ===================================================================
# Convenience functions
# ===================================================================

def run_cirrna_evolution(seed_seq: str, rounds: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """Quick circRNA evolution."""
    cfg = CircRNAEvolutionConfig(rounds=rounds, seed_seq=seed_seq)
    return evolve_cirrna(cfg)


def run_molecule_evolution(seed_smiles: List[str], rounds: int = 5) -> Tuple[pd.DataFrame, EvolutionArtifacts]:
    """Quick molecule evolution."""
    cfg = EvolutionConfig(rounds=rounds)
    return evolve_molecules(seed_smiles, cfg)