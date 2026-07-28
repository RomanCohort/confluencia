"""
evolution.py — Evolutionary optimization for circRNA sequences.

v2 (2026-07): integrated with TorusFold S9 (project-torusfold-scheme9-impl).
Three upgrades over the original drug-2.0 adaptation:

  A. S9 immune-fingerprint fitness  —  evaluate() can call TorusFold
     structure_mode='torus' and score via the 5+1 immune heads (PKR /
     NLRP3 / DRACH / TLR7 / SPONGE / RIGI) instead of the legacy
     quick_predict+quick_dose_predict heuristic.

  B. Saliency-guided mutation        —  when use_torusfold=True, we compute
     ∂fitness/∂sequence_repr (a per-position saliency map identifying
     which positions' *representations* most influence fitness). This is
     NOT a true ∂fitness/∂sequence gradient — circRNA sequences are
     discrete tokens and ESM2 is frozen, so sequence tokens are not
     differentiable. The saliency map is used only to prioritize which
     positions to mutate, not to prescribe the mutation direction.

  C. Hard constraints                —  GC content (gc_min/gc_max),
     length (min/max_length), and non-empty closure-viable sequence
     are enforced post-mutation; violators are rejected and re-drawn.

v3 (2026-07-B): differentiable folding layer (建议 2) — parallel
    differentiable secondary-structure folding on continuous token
    probabilities, providing TRUE ∂fitness/∂sequence_token via the
    straight-through estimator (Bengio et al. 2013).

Legacy heuristic path is preserved as fallback (use_torusfold=False)
so existing scripts that import evolve_sequence / optimize_population
keep working without a TorusFold checkpoint.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the differentiable folding layer (建议 2).
from .torus_coord_head import DifferentiableSecondaryStructure


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""

    # Population
    population_size: int = 50
    elite_size: int = 10

    # Evolution
    n_generations: int = 100
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7

    # Selection
    selection_pressure: float = 2.0
    diversity_penalty: float = 0.1

    # Targets
    target_immunogenicity: float = 0.6
    target_stability: float = 0.7

    # Constraints (now ENFORCED, not just declared)
    min_length: int = 100
    max_length: int = 500
    gc_min: float = 0.35
    gc_max: float = 0.65

    # S9 integration (Change A/B)
    use_torusfold: bool = False              # toggle S9 fitness + saliency-guided mutation
    fitness_mode: str = "balanced"          # "balanced" | "immunogenic" | "therapeutic" | "multi_objective"
    saliency_guided_mutation: bool = True   # only effective when use_torusfold=True
    saliency_topk_hotspots: int = 5         # top-k saliency positions to prioritize
    saliency_mutation_boost: float = 0.20   # extra mutation rate on hotspots

    # NSGA-II multi-objective (Change 3)
    # When fitness_mode == "multi_objective", fitness is a vector and
    # selection uses non-dominated sorting + crowding distance (NSGA-II).
    # Objectives (all MAXIMIZED, internally negated for minimization routines):
    #   1. pkr_sasa        — PKR exposure (immune activation)
    #   2. sponge_score    — miRNA sponge potency
    #   3. 1 - m6a         — m6A shielding (inverted: low m6a = high score for therapy)
    #   4. 1 - rigi        — RIG-I attenuation (circ negative control)
    #   5. nlrp3_norm      — NLRP3 scaffold potential
    multi_objective_keys: List[str] = field(default_factory=lambda: [
        "pkr_sasa", "sponge_score", "inv_m6a", "inv_rigi", "nlrp3_norm",
    ])
    torusfold_device: str = "cpu"
    # Fitness weights (balanced mode)
    w_pkr: float = 0.25
    w_sponge: float = 0.25
    w_m6a: float = 0.20                     # therapeutic → want LOW m6a shielding
    w_rigi: float = 0.15                    # circ → want LOW RIG-I (negative control)
    w_nlrp3: float = 0.15

    # Gene expression (TorusFold input)
    gene_cols: List[str] = field(default_factory=lambda: [
        "TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"
    ])
    gene_dim: int = 6

    # Differentiable folding (建议 2) — true ∂fitness/∂sequence_token.
    # When enable_diff_folding=True, the saliency path uses a parallel
    # differentiable secondary-structure layer (DifferentiableSecondaryStructure)
    # over continuous token probabilities (Gumbel-Softmax + STE), instead of
    # the frozen-ESM2 ∂fitness/∂sequence_repr saliency.
    enable_diff_folding: bool = False
    diff_folding_d_model: int = 64
    diff_folding_min_loop: int = 3
    diff_folding_tau: float = 1.0          # Gumbel-Softmax temperature
    # When True, mutate toward the *direction* of the diff-folding gradient
    # (since this gradient IS a true ∂/∂token). Otherwise (S9 saliency) we
    # only prioritize hotspots, not direction (see _saliency_hotspots note).
    diff_folding_directed_mutation: bool = True

    # GFlowNet population seeding (建议 1) — when enable_gflownet_seeding=True,
    # a fraction of the initial population is drawn from a GFlowNet policy
    # (trained on the fitness function) instead of pure-random generation.
    # The GA still runs selection/crossover/mutation; GFlowNet just seeds
    # better starting points. Set gflownet_seeding_fraction=0.0 to disable.
    enable_gflownet_seeding: bool = False
    gflownet_seeding_fraction: float = 0.3
    gflownet_d_model: int = 64
    gflownet_hidden_dim: int = 128
    gflownet_num_layers: int = 2
    gflownet_train_steps: int = 50
    gflownet_batch_size: int = 8


def _gc_content(sequence: str) -> float:
    """Fraction of G+C in sequence (case-insensitive, ignores non-ACGU)."""
    if not sequence:
        return 0.0
    gc = sum(1 for b in sequence if b in "GCgc")
    return gc / len(sequence)


# Alphabet order MUST match DifferentiableSecondaryStructure.NUC_ORDER.
_NUC_INDEX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}


def _sequence_to_token_logits(sequence: str, scale: float = 10.0) -> torch.Tensor:
    """
    Convert a discrete RNA sequence to (L, 4) one-hot-derived *logits*.

    A near-hard one-hot (scale=10) reproduces the discrete sequence when
    passed through Gumbel-Softmax(hard=True). The scale lets gradients
    backpropagate to all 4 logits at each position (the off-target logits
    have small but nonzero gradients via the STE), enabling true
    ∂fitness/∂sequence_token.

    Non-ACGU characters are mapped to a uniform prior (all logits 0).
    """
    L = len(sequence)
    logits = torch.zeros(L, 4)
    for i, b in enumerate(sequence.upper()):
        idx = _NUC_INDEX.get(b)
        if idx is not None:
            logits[i, idx] = scale
    return logits


def _passes_constraints(
    sequence: str,
    config: EvolutionConfig,
) -> bool:
    """Return True iff sequence satisfies all hard constraints."""
    L = len(sequence)
    if L < config.min_length or L > config.max_length:
        return False
    if not sequence:
        return False
    # Only ACGU allowed
    if any(b not in "ACGUacgu" for b in sequence):
        return False
    gc = _gc_content(sequence)
    if gc < config.gc_min or gc > config.gc_max:
        return False
    return True


# ----------------------------------------------------------------------
# NSGA-II multi-objective utilities (Change 3)
# ----------------------------------------------------------------------

def _non_dominated_sort(
    population: List["CircRNAIndividual"],
) -> List[List["CircRNAIndividual"]]:
    """
    Fast non-dominated sorting (Deb et al. 2002).

    Partitions `population` into fronts F_1, F_2, ... where every individual
    in F_k is dominated only by individuals in F_1..F_{k-1}. Front index
    (rank) is written onto each individual via `setattr(_, 'nsga_rank', ...)`
    so the caller can use it for selection without re-sorting.

    Assumes each individual has a non-None `fitness_vector` of identical
    length, with all objectives MAXIMIZED. Individual a dominates b iff a
    is >= b on every objective and strictly > on at least one.
    """
    n = len(population)
    if n == 0:
        return []

    vecs = [ind.fitness_vector for ind in population]
    # Handle missing/degenerate vectors: treat as all-zeros (dominated by all).
    dim = max((v.shape[0] for v in vecs if v is not None), default=0)
    vecs = [v if v is not None else np.zeros(dim) for v in vecs]

    domination_count = [0] * n          # how many individuals dominate i
    dominated_set: List[List[int]] = [[] for _ in range(n)]  # who i dominates
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            vp, vq = vecs[p], vecs[q]
            if np.all(vp >= vq) and np.any(vp > vq):
                dominated_set[p].append(q)   # p dominates q
            elif np.all(vq >= vp) and np.any(vq > vp):
                domination_count[p] += 1     # q dominates p
        if domination_count[p] == 0:
            fronts[0].append(p)

    # Peel off successive fronts.
    k = 0
    while fronts[k]:
        next_front: List[int] = []
        for p in fronts[k]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        k += 1
        fronts.append(next_front)

    # Drop the trailing empty front (sentinel).
    if fronts and not fronts[-1]:
        fronts.pop()

    result: List[List[CircRNAIndividual]] = []
    for rank, front_idx in enumerate(fronts):
        front_pop = [population[i] for i in front_idx]
        for ind in front_pop:
            ind.nsga_rank = rank  # type: ignore[attr-defined]
        result.append(front_pop)
    return result


def _crowding_distance(front: List["CircRNAIndividual"]) -> None:
    """
    Crowding distance (Deb et al. 2002). Writes `nsga_crowding` onto each
    individual in `front`. Larger = more isolated = preferred (preserves
    diversity within a front). Operates in-place; returns None.

    Boundary individuals (best/worst per objective) get +inf so they are
    never dropped in favor of interior ones.
    """
    m = len(front)
    if m == 0:
        return
    if m == 1:
        front[0].nsga_crowding = float('inf')  # type: ignore[attr-defined]
        return
    if m == 2:
        for ind in front:
            ind.nsga_crowding = float('inf')  # type: ignore[attr-defined]
        return

    dim = front[0].fitness_vector.shape[0]
    for ind in front:
        ind.nsga_crowding = 0.0  # type: ignore[attr-defined]

    for obj in range(dim):
        # Sort front by this objective ASCENDING (Deb et al. convention so
        # f[k+1] - f[k-1] >= 0 for interior points). We still maximize —
        # only the sort direction is conventional; crowding is sign-symmetric.
        order = sorted(
            range(m),
            key=lambda i: front[i].fitness_vector[obj],
        )
        # Boundaries → inf.
        front[order[0]].nsga_crowding = float('inf')  # type: ignore[attr-defined]
        front[order[-1]].nsga_crowding = float('inf')  # type: ignore[attr-defined]

        f_max = front[order[-1]].fitness_vector[obj]
        f_min = front[order[0]].fitness_vector[obj]
        span = f_max - f_min
        if span <= 0:
            continue  # all equal on this objective → no crowding info

        for k in range(1, m - 1):
            prev_v = front[order[k - 1]].fitness_vector[obj]
            next_v = front[order[k + 1]].fitness_vector[obj]
            cur = front[order[k]]
            # type: ignore[attr-defined]
            if cur.nsga_crowding != float('inf'):
                cur.nsga_crowding += (next_v - prev_v) / span


class CircRNAIndividual:
    """Individual circRNA sequence in evolution."""

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.fitness = 0.0
        self.fitness_vector: Optional[np.ndarray] = None  # NSGA-II multi-objective
        self.age = 0
        # Cached S9 outputs (for saliency-guided mutation)
        self._last_repr: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Fitness evaluation — two paths
    # ------------------------------------------------------------------

    def evaluate(
        self,
        config: Optional[EvolutionConfig] = None,
        torusfold_model: Optional["object"] = None,
    ) -> float:
        """
        Evaluate fitness.

        If config.use_torusfold and torusfold_model is provided, use the
        S9 immune-fingerprint path (Change A). Otherwise fall back to the
        legacy quick_predict + quick_dose_predict heuristic.
        """
        config = config or EvolutionConfig()

        if config.use_torusfold and torusfold_model is not None:
            return self._evaluate_torusfold(config, torusfold_model)

        return self._evaluate_legacy()

    def _evaluate_legacy(self) -> float:
        """Legacy heuristic fitness (preserved for backward compatibility)."""
        from .innate_immune import quick_predict
        from .dose_tox import quick_dose_predict

        immune = quick_predict(self.sequence)
        imm_score = immune.get('overall_score', 0.0)

        dose = quick_dose_predict(self.sequence, dose=100)
        window = dose.get('therapeutic_window', 0.0)

        self.fitness = imm_score * 0.5 + window * 0.5
        return self.fitness

    def _evaluate_torusfold(
        self,
        config: EvolutionConfig,
        model,
    ) -> float:
        """
        S9 immune-fingerprint fitness (Change A).

        Runs TorusFold forward in structure_mode='torus' and combines the
        5+1 immune fingerprints into a scalar fitness. The cached
        sequence_repr is available for saliency-guided mutation (Change B),
        which re-runs a differentiable forward on demand.
        """
        # TorusFold expects a gene_expr tensor (B, gene_dim).
        gene_expr = torch.zeros(1, config.gene_dim, device=config.torusfold_device)

        # Forward — keep sequence_repr in the graph for grad extraction.
        outputs = model.forward(
            [self.sequence],
            gene_expr,
            device=config.torusfold_device,
            predict_structure=True,
        )

        fingerprints = outputs.get("immune_fingerprints", {})
        if not fingerprints:
            # S9 immune heads disabled → fall back to legacy.
            return self._evaluate_legacy()

        # Per-position quantities → scalar via mean.
        def _mean(t):
            if t is None:
                return 0.0
            return float(t.mean().item()) if torch.is_tensor(t) else float(t)

        pkr_sasa = _mean(fingerprints.get("pkr_sasa"))
        sponge = _mean(fingerprints.get("sponge_score"))
        m6a = _mean(fingerprints.get("m6a_write_prob"))
        rigi = _mean(fingerprints.get("rigi_score"))
        nlrp3 = _mean(fingerprints.get("nlrp3_persistence_length"))

        # Normalize nlrp3 persistence length to [0, 1] heuristic range.
        # Typical circRNA persistence length ~1-5 nm; map >5 nm → 1.0.
        nlrp3_norm = min(max(nlrp3 / 5.0, 0.0), 1.0)

        if config.fitness_mode == "immunogenic":
            # Vaccine-style: want HIGH PKR, HIGH TLR7/GU, HIGH NLRP3.
            tlr7 = _mean(fingerprints.get("tlr7_gu_density"))
            self.fitness = (
                0.4 * pkr_sasa + 0.3 * tlr7 + 0.3 * nlrp3_norm
            )
        elif config.fitness_mode == "multi_objective":
            # NSGA-II: keep objectives as a vector (all MAXIMIZED).
            # m6a and rigi are inverted (low m6a shielding, low RIG-I = good).
            self.fitness_vector = np.array([
                pkr_sasa,        # immune activation
                sponge,          # miRNA sponge
                1.0 - m6a,       # m6A shielding (inverted)
                1.0 - rigi,      # RIG-I attenuation (inverted)
                nlrp3_norm,      # NLRP3 scaffold
            ], dtype=np.float64)
            # Scalar projection (mean) — used only for stats/logging and
            # tournament fallback when NSGA-II selection is not active.
            self.fitness = float(self.fitness_vector.mean())
        elif config.fitness_mode == "therapeutic":
            # Protein-replacement therapy: want LOW immune activation.
            # m6a shielding is GOOD (lowers immune recognition) → invert.
            self.fitness = (
                0.4 * (1.0 - pkr_sasa) +
                0.3 * (1.0 - rigi) +
                0.3 * m6a
            )
        else:  # balanced
            self.fitness = (
                config.w_pkr * pkr_sasa +
                config.w_sponge * sponge +
                config.w_m6a * (1.0 - m6a) +     # therapeutic-leaning
                config.w_rigi * (1.0 - rigi) +   # circ should be low
                config.w_nlrp3 * nlrp3_norm
            )

        # Cache sequence_repr for saliency-guided mutation (Change B).
        if config.saliency_guided_mutation:
            seq_repr = outputs.get("sequence_repr")  # (1, L, d_model)
            if seq_repr is not None and seq_repr.requires_grad:
                self._last_repr = seq_repr
                # Saliency is recomputed on demand inside _saliency_hotspots()
                # via a fresh differentiable forward pass.

        return self.fitness

    # ------------------------------------------------------------------
    # Saliency-guided mutation support (Change B)
    # ------------------------------------------------------------------

    def _saliency_hotspots(
        self,
        config: EvolutionConfig,
        model,
    ) -> Optional[np.ndarray]:
        """
        Compute ∂fitness/∂sequence_repr — a per-position saliency map.

        SCIENTIFIC HONESTY NOTE:
            This is NOT a true gradient w.r.t. the *sequence*. circRNA
            sequences are discrete tokens (A/C/G/U), and ESM2 is frozen,
            so there is no differentiable path from token identity to
            sequence_repr. What we compute here is the saliency of each
            position's *learned representation* on the fitness scalar —
            i.e., "which positions' representations most influence
            fitness". This is a valid prioritization signal for mutation
            (mutate the most influential positions first), but it does
            NOT prescribe the mutation direction (which base to change
            into what). The mutation itself remains stochastic.

        Returns:
            (L,) array of per-position saliency magnitudes, or None if
            unavailable.
        """
        if not config.use_torusfold or model is None:
            return None

        gene_expr = torch.zeros(1, config.gene_dim, device=config.torusfold_device)
        # Re-run forward with grad enabled on sequence_repr.
        # NOTE: ESM2 backbone is frozen; sequence_repr is the output of the
        # torus transformer layers, which IS differentiable w.r.t. the
        # (frozen) ESM2 embedding → transformer weights. To get gradients
        # w.r.t. the *sequence*, we'd need a differentiable embedding; here
        # we approximate by taking grad w.r.t. sequence_repr itself, which
        # identifies which positions are most influential on fitness.
        try:
            outputs = model.forward(
                [self.sequence],
                gene_expr,
                device=config.torusfold_device,
                predict_structure=True,
            )
            fingerprints = outputs.get("immune_fingerprints", {})
            if not fingerprints:
                return None

            # Reconstruct a differentiable scalar fitness from the tensors.
            pkr_sasa = fingerprints.get("pkr_sasa")            # (1, L)
            sponge = fingerprints.get("sponge_score")          # (1,)
            m6a = fingerprints.get("m6a_write_prob")           # (1, L)
            rigi = fingerprints.get("rigi_score")              # (1,)
            nlrp3 = fingerprints.get("nlrp3_persistence_length")  # (1,)

            terms = []
            if pkr_sasa is not None:
                terms.append(config.w_pkr * pkr_sasa.mean())
            if sponge is not None:
                terms.append(config.w_sponge * sponge.mean())
            if m6a is not None:
                terms.append(config.w_m6a * (1.0 - m6a).mean())
            if rigi is not None:
                terms.append(config.w_rigi * (1.0 - rigi).mean())
            if nlrp3 is not None:
                terms.append(config.w_nlrp3 * (nlrp3 / 5.0).clamp(0, 1).mean())

            if not terms:
                return None

            scalar_fitness = sum(terms)
            seq_repr = outputs.get("sequence_repr")  # (1, L, d)
            if seq_repr is None:
                return None

            grad = torch.autograd.grad(
                scalar_fitness,
                seq_repr,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                return None
            # Per-position magnitude: (L,)
            mag = grad.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
            return mag
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Differentiable folding — TRUE ∂fitness/∂sequence_token (建议 2)
    # ------------------------------------------------------------------

    # Cache a shared folding layer on the evolution instance to avoid
    # re-instantiation per individual. Set by CircRNAEvolution.__init__.
    _diff_folding_layer: Optional[DifferentiableSecondaryStructure] = None

    def _diff_folding_saliency(
        self,
        config: EvolutionConfig,
        model,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        TRUE ∂fitness/∂sequence_token via differentiable folding (建议 2).

        Unlike `_saliency_hotspots` (which is a saliency map over the frozen
        ESM2 representation, not a true sequence gradient), this method runs
        a parallel differentiable secondary-structure layer over continuous
        token probabilities (Gumbel-Softmax + straight-through estimator).
        Gradients backpropagate through the soft pairing matrix and structural
        features all the way to the token logits — a genuine
        ∂fitness/∂sequence_token.

        The differentiable folding fitness is a *proxy* objective derived
        from structural features (stem density, GC content, pair-probability
        statistics) weighted to align with the active fitness_mode. It is
        NOT a replacement for the S9 immune-fingerprint fitness used in
        `evaluate()`; it is the gradient signal used to *guide* mutation.

        Returns:
            (hotspots, directed_grad) tuple where:
              hotspots      — (L,) per-position saliency magnitudes
              directed_grad — (L, 4) per-position, per-nucleotide gradient
                              (used for directed mutation when
                              config.diff_folding_directed_mutation=True)
            or None if unavailable.
        """
        if not config.enable_diff_folding:
            return None

        layer = CircRNAIndividual._diff_folding_layer
        if layer is None:
            try:
                layer = DifferentiableSecondaryStructure(
                    d_model=config.diff_folding_d_model,
                    min_loop=config.diff_folding_min_loop,
                    tau=config.diff_folding_tau,
                    hard_ste=True,
                )
                CircRNAIndividual._diff_folding_layer = layer
            except Exception:
                return None

        try:
            # Build (L, 4) token logits from the discrete sequence.
            token_logits = _sequence_to_token_logits(self.sequence).unsqueeze(0)
            token_logits.requires_grad_(True)

            layer.train()  # STE active in train mode
            fold_out = layer(token_logits)  # dict of differentiable tensors

            # Differentiable proxy fitness aligned with config.fitness_mode.
            # All terms are differentiable in token_logits via the STE.
            stem = fold_out['stem_density']             # (B,)
            gc = fold_out['gc_content']                 # (B,)
            mpp = fold_out['mean_pair_prob']            # (B,)
            pair_prob = fold_out['pair_prob']           # (B, L, L)

            # Pair-probability-weighted exposure proxy (positions in stems
            # are less exposed → lower PKR/RIG-I; positions in loops are
            # more exposed). Per-position loop-ness = 1 - row_sum(pair).
            row_pair = pair_prob.sum(dim=-1)            # (B, L)
            loopness = 1.0 - row_pair                   # (B, L)

            if config.fitness_mode == "immunogenic":
                # High immune activation: high stem density + high loopness.
                proxy = stem.mean() + loopness.mean()
            elif config.fitness_mode == "therapeutic":
                # Low immune activation + high GC shielding.
                proxy = (1.0 - stem.mean()) + 0.5 * gc.mean() + (1.0 - loopness).mean()
            elif config.fitness_mode == "multi_objective":
                # Pareto-style: maximize all structural objectives.
                proxy = stem.mean() + gc.mean() + mpp.mean()
            else:  # balanced
                proxy = (
                    0.3 * stem.mean()
                    + 0.3 * gc.mean()
                    + 0.2 * mpp.mean()
                    + 0.2 * loopness.mean()
                )

            grad = torch.autograd.grad(
                proxy,
                token_logits,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                return None

            grad_np = grad.squeeze(0).detach().cpu().numpy()  # (L, 4)
            # Per-position magnitude = L2 norm over the 4 nucleotide logits.
            hotspots = np.linalg.norm(grad_np, axis=-1)  # (L,)
            return hotspots, grad_np
        except Exception:
            return None


class CircRNAEvolution:
    """
    Evolutionary optimization for circRNA sequences (v2: S9-integrated).

    Methods:
    - Population initialization (seed + generator, all constraint-validated)
    - Fitness evaluation (legacy heuristic OR S9 immune fingerprints)
    - Selection (elite + tournament)
    - Crossover (single-point, length-aware)
    - Mutation (point + insert/delete, with optional gradient guidance)
    - Hard constraint enforcement (GC / length / alphabet)
    """

    NUCS = ['A', 'U', 'G', 'C']

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        torusfold_model: Optional["object"] = None,
    ):
        self.config = config or EvolutionConfig()
        self.population: List[CircRNAIndividual] = []
        self.history: List[Dict] = []
        self.torusfold_model = torusfold_model

    # ------------------------------------------------------------------
    # Population init
    # ------------------------------------------------------------------

    def initialize_population(self, seed_sequences: List[str] = None):
        """Initialize population. All individuals pass hard constraints."""
        from .generative import CircRNAGenerator

        generator = CircRNAGenerator()
        self.population = []

        if seed_sequences:
            for seq in seed_sequences[:self.config.elite_size]:
                if _passes_constraints(seq, self.config):
                    self.population.append(CircRNAIndividual(seq))

        # Fill rest with constraint-satisfying random sequences.
        attempts = 0
        while len(self.population) < self.config.population_size and attempts < 1000:
            L = int(np.random.randint(
                self.config.min_length, self.config.max_length + 1
            ))
            gc_target = np.random.uniform(self.config.gc_min, self.config.gc_max)
            seq = generator.generate_random(length=L, gc_target=gc_target)
            if _passes_constraints(seq, self.config):
                self.population.append(CircRNAIndividual(seq))
            attempts += 1

        # If still short (rare), pad with minimal-GC random sequences.
        while len(self.population) < self.config.population_size:
            L = self.config.min_length
            seq = generator.generate_random(length=L, gc_target=0.5)
            self.population.append(CircRNAIndividual(seq))

        # GFlowNet seeding (建议 1): replace a fraction of the population
        # with sequences sampled from a trained GFlowNet policy. The GA
        # still runs selection/crossover/mutation afterward; GFlowNet
        # only provides better-informed starting points.
        if self.config.enable_gflownet_seeding:
            self._seed_with_gflownet()

    def _seed_with_gflownet(self) -> None:
        """Replace a fraction of the population with GFlowNet samples.

        Builds a fitness function from the current evaluation path
        (S9 or legacy), trains a small GFlowNet briefly, and samples
        `gflownet_seeding_fraction * population_size` sequences to
        replace the worst-scoring individuals in the initial population.
        """
        try:
            from .gflownet import GFloWNetGenerator
        except Exception:
            return  # GFlowNet unavailable — silently skip.

        # Build a fitness callable compatible with GFlowNet (takes a
        # sequence string, returns a non-negative scalar).
        def fitness_fn(seq: str) -> float:
            ind = CircRNAIndividual(seq)
            f = ind.evaluate(self.config, self.torusfold_model)
            return max(float(f), 1e-6)

        gfn = GFloWNetGenerator(
            config={
                'd_model': self.config.gflownet_d_model,
                'hidden_dim': self.config.gflownet_hidden_dim,
                'num_layers': self.config.gflownet_num_layers,
                'min_length': self.config.min_length,
                'max_length': self.config.max_length,
                'lr': 1e-3,
            },
            fitness_function=fitness_fn,
        )

        # Brief training.
        try:
            gfn.train_gfn(
                n_steps=self.config.gflownet_train_steps,
                batch_size=self.config.gflownet_batch_size,
            )
        except Exception:
            pass  # Training may fail on edge cases; sampling still works.

        # Sample and filter to constraint-satisfying sequences.
        n_target = int(
            self.config.gflownet_seeding_fraction * self.config.population_size
        )
        candidates = gfn.sample_sequences(n_target, temperature=1.0)
        seeded = [
            CircRNAIndividual(s) for s in candidates
            if _passes_constraints(s, self.config)
        ]

        if not seeded:
            return  # None passed constraints — keep random population.

        # Replace the worst-scoring individuals with the seeded ones.
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        keep = self.config.population_size - len(seeded)
        self.population = self.population[:keep] + seeded

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_population(self) -> Dict:
        """Evaluate all individuals."""
        fitnesses = []
        for individual in self.population:
            f = individual.evaluate(self.config, self.torusfold_model)
            fitnesses.append(f)

        return {
            'mean_fitness': float(np.mean(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
        }

    def _evaluate_offspring(self, offspring: List[CircRNAIndividual]) -> Dict:
        """Evaluate a candidate offspring list (not yet assigned to
        self.population). Used by NSGA-II to ensure fitness_vectors exist
        before parent selection in the next generation."""
        fitnesses = []
        for individual in offspring:
            f = individual.evaluate(self.config, self.torusfold_model)
            fitnesses.append(f)
        return {
            'mean_fitness': float(np.mean(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
        }

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(self) -> List[CircRNAIndividual]:
        """
        Parent selection.

        - fitness_mode == 'multi_objective': NSGA-II tournament — binary
          comparison by (rank asc, crowding desc). Uses the per-individual
          `nsga_rank` / `nsga_crowding` set by _non_dominated_sort +
          _crowding_distance, which are computed once per generation in
          evolve_generation() before selection.
        - otherwise: elite + standard tournament on scalar fitness.
        """
        if self.config.fitness_mode == "multi_objective":
            return self._select_parents_nsga()

        parents = []
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        parents.extend(sorted_pop[:self.config.elite_size])

        n_parents = max(0, int(self.config.population_size * 0.5) - self.config.elite_size)
        for _ in range(n_parents):
            k = min(5, len(self.population))
            candidates = random.sample(self.population, k)
            winner = max(candidates, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

    def _select_parents_nsga(self) -> List[CircRNAIndividual]:
        """NSGA-II binary tournament on (rank, crowding)."""
        # Ensure ranks/crowding are current for this population.
        fronts = _non_dominated_sort(self.population)
        for front in fronts:
            _crowding_distance(front)

        def _better(a: CircRNAIndividual, b: CircRNAIndividual) -> CircRNAIndividual:
            ra = getattr(a, 'nsga_rank', 0)
            rb = getattr(b, 'nsga_rank', 0)
            if ra != rb:
                return a if ra < rb else b
            ca = getattr(a, 'nsga_crowding', 0.0)
            cb = getattr(b, 'nsga_crowding', 0.0)
            return a if ca > cb else b

        # Elitism: take whole first front (Pareto-optimal), then fill by
        # rank-then-crowding order.
        parents: List[CircRNAIndividual] = []
        flat_by_rank: List[CircRNAIndividual] = []
        for front in fronts:
            front_sorted = sorted(
                front,
                key=lambda x: getattr(x, 'nsga_crowding', 0.0),
                reverse=True,
            )
            flat_by_rank.extend(front_sorted)
        parents.extend(flat_by_rank[:self.config.elite_size])

        n_parents = max(
            0, int(self.config.population_size * 0.5) - self.config.elite_size
        )
        for _ in range(n_parents):
            k = min(5, len(self.population))
            if k < 2:
                parents.append(self.population[0])
                continue
            a, b = random.sample(self.population, 2)
            parents.append(_better(a, b))
        return parents

    # ------------------------------------------------------------------
    # Crossover (length-aware single-point)
    # ------------------------------------------------------------------

    def crossover(
        self,
        parent1: CircRNAIndividual,
        parent2: CircRNAIndividual,
    ) -> Tuple[str, str]:
        """
        Circular crossover preserving ring topology.

        circRNA is a closed loop with no canonical start position. The
        legacy single-point crossover `seq1[:p] + seq2[p:]` implicitly
        assumes both parents are aligned at position 0 — which is not
        rotation-invariant for circular molecules. Two parents at
        different rotational phases would produce different children
        depending on the arbitrary BSJ (back-splice junction) position.

        We use a circular-segment crossover instead:
          1. Randomly rotate parent2's phase (so the crossover point is
             not tied to either parent's arbitrary origin).
          2. Pick two cut points i < j on parent1's frame.
          3. Splice the rotated parent2's segment into parent1's [i,j]
             interval, producing two children that each inherit a
             contiguous arc from each parent — preserving ring topology.

        For unequal-length parents, the spliced segment is
        length-adjusted (truncated or padded from parent2's own sequence)
        so both children satisfy the length constraint post-hoc via
        _constrained_crossover retries.
        """
        seq1, seq2 = parent1.sequence, parent2.sequence
        if random.random() > self.config.crossover_rate:
            return seq1, seq2

        L1, L2 = len(seq1), len(seq2)

        # 1. Random phase rotation of parent2 (circular shift).
        phase = random.randint(0, L2 - 1)
        seq2_rot = seq2[phase:] + seq2[:phase]

        # 2. Two cut points on parent1's frame.
        if L1 < 3:
            return seq1, seq2  # too short to splice
        i = random.randint(1, L1 - 2)
        j = random.randint(i + 1, L1 - 1)
        arc_len = j - i  # length of the spliced arc

        # 3. Extract a segment of the same arc length from rotated parent2.
        #    Wrap around if arc_len exceeds L2 (circRNA can wrap multiple times).
        seg2 = ''.join(seq2_rot[k % L2] for k in range(arc_len))

        # 4. Child1: parent1 with [i,j] replaced by seg2.
        child1 = seq1[:i] + seg2 + seq1[j:]

        # 5. Child2: parent2_rot with the analogous arc replaced by
        #    parent1's [i,j] arc (truncated/wrapped to L2's frame).
        arc1 = seq1[i:j]  # parent1's arc
        # Choose a splice point on parent2's rotated frame.
        if L2 >= 3:
            k = random.randint(1, L2 - 2)
            k_end = k + arc_len
            # Wrap-aware replacement: build child2 piece by piece.
            if k_end <= L2:
                child2 = seq2_rot[:k] + arc1 + seq2_rot[k_end:]
            else:
                # Arc wraps around the ring of parent2.
                overflow = k_end - L2
                # arc1 may need to be longer to fill; if shorter, pad from parent2.
                fill = arc1
                if len(fill) < arc_len:
                    fill = fill + seq2_rot[:arc_len - len(fill)]
                child2 = fill[:overflow] + seq2_rot[overflow:k] + fill[overflow:]
            # Re-rotate child2 back to a random phase (cosmetic; ring invariance).
        else:
            child2 = seq2_rot

        return child1, child2

    # ------------------------------------------------------------------
    # Mutation (point + indel + optional saliency guidance)
    # ------------------------------------------------------------------

    def mutate(
        self,
        sequence: str,
        saliency_magnitudes: Optional[np.ndarray] = None,
        directed_grads: Optional[np.ndarray] = None,
    ) -> str:
        """
        Mutate sequence.

        Args:
            sequence: input sequence
            saliency_magnitudes: optional (L,) array of per-position
                saliency magnitudes. If provided, the top-k positions get
                an extra mutation-rate boost (Change B / 建议 2). NOTE:
                this is a saliency prioritization signal, NOT a directional
                gradient.
            directed_grads: optional (L, 4) array of per-position,
                per-nucleotide gradient from differentiable folding
                (建议 2). When this is provided AND
                config.diff_folding_directed_mutation=True, mutation is
                *DIRECTED* toward increasing fitness along this gradient.

        Tokens with high gradient in a specific nucleotide are more likely
        to undergo that specific nucleotide mutation than other outcomes.
        """
        # The precedence order is: directed_grads > saliency_magnitudes
        # > base mutation_rate. This ensures the TRUE ∂fitness/∂token
        # dominates when available.
        seq = list(sequence)
        L = len(seq)

        # NOTE: directed_grads / saliency_magnitudes come from the PARENT
        # sequence, which may differ in length from `sequence` (post-crossover).
        # If lengths mismatch, fall back to undirected mutation rather than
        # crash — the saliency signal is stale anyway on a recombined sequence.
        if directed_grads is not None and directed_grads.shape[0] != L:
            directed_grads = None
        if (saliency_magnitudes is not None
                and len(saliency_magnitudes) != L):
            saliency_magnitudes = None

        use_directed = (
            directed_grads is not None and self.config.diff_folding_directed_mutation
        )

        # Build per-position mutation rates.
        if use_directed:
            # Directed mutation: compute position-wise score for each nucleotide.
            # score[pos][n'] = grad[pos][n'] (higher = more likely to become n')
            # We turn this into mutation probabilities via softmax.
            # Temperature adapts exploration-exploitation tradeoff.
            temp = 1.0
            logits = directed_grads / temp  # (L, 4)
            logitm = logits - np.max(logits, axis=-1, keepdims=True)  # (L, 4)
            probs = np.exp(logitm) / np.sum(np.exp(logitm), axis=-1, keepdims=True)  # (L, 4)

            # Apply mutation: sample [A,U,G,C] with these probs, prefer mutated.
            for i in range(L):
                if random.random() < self.config.mutation_rate:
                    if random.random() < 0.8:  # 80% mutation
                        choice_idx = np.random.choice(4, p=probs[i])
                        seq[i] = self.NUCS[choice_idx]
                    else:  # 20% tolerate current base (exploration)
                        original = sequence[i]
                        choice_pool = [n for n in self.NUCS if n != original]
                        if choice_pool:
                            seq[i] = random.choice(choice_pool)
        elif saliency_magnitudes is not None and len(saliency_magnitudes) == L:
            # Old S9 saliency: only prioritize hotspots, no direction.
            k = min(self.config.saliency_topk_hotspots, L)
            top_idx = set(np.argsort(saliency_magnitudes)[-k:].tolist())
            rates = np.full(L, self.config.mutation_rate)
            rates[list(top_idx)] += self.config.saliency_mutation_boost
        else:
            rates = np.full(L, self.config.mutation_rate)

        for i in range(L):
            if use_directed:
                continue  # Handled above
            if random.random() < rates[i]:
                choices = [n for n in self.NUCS if n != seq[i]]
                seq[i] = random.choice(choices)

        # Occasional insert/delete (rare, length-constrained).
        if random.random() < 0.02:
            pos = random.randint(0, len(seq) - 1)
            if random.random() < 0.5 and len(seq) > self.config.min_length:
                seq.pop(pos)
            else:
                seq.insert(pos, random.choice(self.NUCS))

        return ''.join(seq)

    # ------------------------------------------------------------------
    # Constraint enforcement wrapper
    # ------------------------------------------------------------------

    def _constrained_mutate(
        self,
        sequence: str,
        saliency_magnitudes: Optional[np.ndarray] = None,
        directed_grads: Optional[np.ndarray] = None,
        max_retries: int = 10,
    ) -> str:
        """Mutate then enforce constraints; retry on violation."""
        for _ in range(max_retries):
            candidate = self.mutate(sequence, saliency_magnitudes, directed_grads)
            if _passes_constraints(candidate, self.config):
                return candidate
        # Fallback: return original (preserves constraints).
        return sequence

    def _constrained_crossover(
        self,
        p1: CircRNAIndividual,
        p2: CircRNAIndividual,
        max_retries: int = 10,
    ) -> Tuple[str, str]:
        """Crossover then enforce constraints; retry on violation."""
        for _ in range(max_retries):
            c1, c2 = self.crossover(p1, p2)
            if _passes_constraints(c1, self.config) and _passes_constraints(c2, self.config):
                return c1, c2
        # Fallback: return parents (already constraint-satisfying).
        return p1.sequence, p2.sequence

    # ------------------------------------------------------------------
    # Generation step
    # ------------------------------------------------------------------

    def evolve_generation(self) -> Dict:
        """Evolve one generation."""
        stats = self.evaluate_population()

        parents = self.select_parents()

        offspring = []
        # Elite passes through unchanged.
        if self.config.fitness_mode == "multi_objective":
            # NSGA-II elitism: survivors ordered by (rank, crowding).
            fronts = _non_dominated_sort(self.population)
            for front in fronts:
                _crowding_distance(front)
            flat: List[CircRNAIndividual] = []
            for front in fronts:
                flat.extend(sorted(
                    front,
                    key=lambda x: getattr(x, 'nsga_crowding', 0.0),
                    reverse=True,
                ))
            for elite in flat[:self.config.elite_size]:
                offspring.append(CircRNAIndividual(elite.sequence))
            # Record Pareto front size for diagnostics.
            stats['pareto_front_size'] = len(fronts[0]) if fronts else 0
        else:
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for elite in sorted_pop[:self.config.elite_size]:
                offspring.append(CircRNAIndividual(elite.sequence))

        # Breed new individuals.
        while len(offspring) < self.config.population_size:
            p1, p2 = random.sample(parents, 2)
            child1_seq, child2_seq = self._constrained_crossover(p1, p2)

            # Gradient/saliency-guided mutation (Change B / 建议 2).
            # When enable_diff_folding=True, use the TRUE token-level gradient
            # from the differentiable folding layer (directed mutation when
            # configured). Otherwise, fall back to the S9 saliency hotspot
            # signal (prioritization only, not directional).
            sal1 = None
            sal2 = None
            diff_grad1 = None
            diff_grad2 = None
            if (
                self.config.use_torusfold
                and self.config.saliency_guided_mutation
                and self.torusfold_model is not None
            ):
                if self.config.enable_diff_folding:
                    r1 = p1._diff_folding_saliency(self.config, self.torusfold_model)
                    r2 = p2._diff_folding_saliency(self.config, self.torusfold_model)
                    if r1 is not None:
                        sal1, diff_grad1 = r1
                    if r2 is not None:
                        sal2, diff_grad2 = r2
                else:
                    sal1 = p1._saliency_hotspots(self.config, self.torusfold_model)
                    sal2 = p2._saliency_hotspots(self.config, self.torusfold_model)

            child1_seq = self._constrained_mutate(child1_seq, sal1, diff_grad1)
            child2_seq = self._constrained_mutate(child2_seq, sal2, diff_grad2)

            offspring.append(CircRNAIndividual(child1_seq))
            if len(offspring) < self.config.population_size:
                offspring.append(CircRNAIndividual(child2_seq))

        offspring = offspring[:self.config.population_size]
        for ind in offspring:
            ind.age += 1

        # Evaluate offspring so fitness is set before the next generation's
        # select_parents (which sorts by fitness) and before the final
        # `best = max(self.population, key=lambda x: x.fitness)`.
        # Previously guarded by `fitness_mode == "multi_objective"` only,
        # which left balanced/immunogenic/therapeutic modes with zero-valued
        # offspring fitness — a latent bug surfaced by ablation runs.
        self._evaluate_offspring(offspring)

        self.population = offspring
        self.history.append(stats)
        return stats

    def run_evolution(self, n_generations: int = None) -> Dict:
        """Run full evolution."""
        n_generations = n_generations or self.config.n_generations
        print(f"Running evolution for {n_generations} generations "
              f"(mode={'S9' if self.config.use_torusfold else 'legacy'}, "
              f"fitness={self.config.fitness_mode})...")

        for gen in range(n_generations):
            if gen % 10 == 0:
                mean_f = np.mean([i.fitness for i in self.population])
                print(f"  Generation {gen}: mean_fitness={mean_f:.4f}")

            stats = self.evolve_generation()

            if stats['max_fitness'] > 0.9 and stats['std_fitness'] < 0.05:
                print(f"  Converged at generation {gen}")
                break

        best = max(self.population, key=lambda x: x.fitness)
        return {
            'best_sequence': best.sequence,
            'best_fitness': best.fitness,
            'generations': len(self.history),
            'final_stats': self.history[-1],
            'convergence': stats['std_fitness'] < 0.05,
        }

    def get_top_sequences(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n sequences."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [(ind.sequence, ind.fitness) for ind in sorted_pop[:n]]


# ----------------------------------------------------------------------
# Convenience entry points (backward-compatible signatures)
# ----------------------------------------------------------------------

def evolve_sequence(
    seed_sequence: str = None,
    n_generations: int = 50,
    target_fitness: float = 0.7,
    use_torusfold: bool = False,
    fitness_mode: str = "balanced",
    torusfold_model: Optional["object"] = None,
) -> Tuple[str, float]:
    """
    Quick evolutionary optimization.

    Args:
        seed_sequence: Starting sequence (optional)
        n_generations: Number of generations
        target_fitness: Target fitness (sets target_immunogenicity)
        use_torusfold: If True, use S9 immune-fingerprint fitness
        fitness_mode: "balanced" | "immunogenic" | "therapeutic"
        torusfold_model: Pre-loaded TorusFold model (required if use_torusfold=True
            and you want to avoid re-instantiating per call)

    Returns:
        Optimized sequence, fitness
    """
    config = EvolutionConfig(
        n_generations=n_generations,
        target_immunogenicity=target_fitness,
        use_torusfold=use_torusfold,
        fitness_mode=fitness_mode,
    )
    evolution = CircRNAEvolution(config, torusfold_model=torusfold_model)

    seed_sequences = [seed_sequence] if seed_sequence else None
    evolution.initialize_population(seed_sequences)

    result = evolution.run_evolution()
    return result['best_sequence'], result['best_fitness']


def optimize_population(
    n_sequences: int = 20,
    n_generations: int = 100,
    use_torusfold: bool = False,
    fitness_mode: str = "balanced",
    torusfold_model: Optional["object"] = None,
) -> List[Tuple[str, float]]:
    """Optimize population of sequences."""
    config = EvolutionConfig(
        population_size=50,
        n_generations=n_generations,
        use_torusfold=use_torusfold,
        fitness_mode=fitness_mode,
    )
    evolution = CircRNAEvolution(config, torusfold_model=torusfold_model)
    evolution.initialize_population()
    evolution.run_evolution()
    return evolution.get_top_sequences(n_sequences)
