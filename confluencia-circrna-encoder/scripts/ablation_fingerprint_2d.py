#!/usr/bin/env python3
"""
ablation_fingerprint_2d.py — Ablation: 2D vs 3D m6A exposure proxy.

QUICK SUMMARY: This script compares two evolution runs that differ ONLY in the
m6A *solvent-exposure* proxy:
  - 3D (default): exposure = sigmoid(torus_coords[..., 2])  (torus radius)
  - 2D (orbit C): exposure = 1 - mean(pair_probs)            (single-strandness)

IMPORTANT — What the switch does and does NOT do:

  - The `enable_fingerprint_2d` flag swaps ONLY the `exposure` term in:
        m6a_write_prob = is_drach * in_loop * exposure_proxy
    `is_drach` and `in_loop` are always predicted from the shared trunk feat
    (which contains torus_coords). The isolated question is:
    "Does the exposure sub-quantity need 3D, or is a 2D single-strandedness
    proxy sufficient?" — NOT "Does the whole m6A head need 3D?"

  - PKR/TLR7/NLRP3/sponge heads are UNCHANGED by this switch. They all consume
    the shared feat (sequence_repr + pair summary + torus_coords). NLRP3
    genuinely needs 3D (persistence length is a geometric property); the others
    empirically benefit from the shared trunk but are not 3D-critical.

RATIONALE FOR ISOLATING EXPOSURE: circRNA has no per-residue SASA labels. A
self-generated 3D SASA label would be circular (label and input share the same
forward pass). The 2D single-strandedness proxy (1 - pair_probs) is directly
supervised by the pair_probs signal, sidestepping that circularity.

Both runs share the same seed sequence, population, and evolution config.
Outputs a JSON + CSV of per-generation mean/max fitness for plotting.

By default uses a Mock backbone (sequence-sensitive random embeddings) so the
script runs on CPU without ESM2. For publishable numbers, use --real-backbone:
  python scripts/ablation_fingerprint_2d.py --real-backbone --esm-model esm2_t12_35M_UR50D

To initialize from a pretrained checkpoint:
  python scripts/ablation_fingerprint_2d.py --checkpoint models/torusfold_best.pt
NOTE: models/torusfold_best.pt is a TorusFoldTrainer checkpoint (no immune_heads).
For a full Scheme 9 checkpoint with immune_heads, retrain with structure_mode=torus
and save via TorusFold.save().
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evolution import CircRNAEvolution, EvolutionConfig
from core.torusfold import TorusFold, TorusFoldConfig


class MockBackbone(nn.Module):
    """Sequence-sensitive mock backbone (no ESM2 needed).

    Returns the dict shape TorusFold.forward expects:
        {"embedding": (B, d_model), "sequence_repr": (B, L, d_model)}

    Embeddings are seeded by nucleotide composition so that different
    sequences produce different (but deterministic) fingerprints — giving
    the evolution a signal to optimize against. This is NOT a learned
    representation; treat results as a pipeline sanity check, not a
    biological result.
    """

    def __init__(self, d_model: int = 64, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Per-nucleotide learnable embedding (A/U/G/C/N).
        self.embed = nn.Embedding(5, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _tokenize(seq: str) -> torch.Tensor:
        mapping = {"A": 0, "U": 1, "G": 2, "C": 3}
        return torch.tensor(
            [mapping.get(c, 4) for c in seq.upper()],
            dtype=torch.long,
        )

    def forward(self, sequences: List[str], device: str = "cpu", **kwargs) -> Dict:
        B = len(sequences)
        Ls = [min(len(s), self.max_len) for s in sequences]
        L = max(Ls) if Ls else 1

        # (B, L, d_model) padded token embeddings.
        reprs = torch.zeros(B, L, self.d_model, device=device)
        for i, seq in enumerate(sequences):
            tok = self._tokenize(seq[:L]).to(device)
            reprs[i, : len(tok)] = self.embed(tok)

        reprs = self.norm(self.proj(reprs))
        global_emb = reprs.mean(dim=1)  # (B, d_model)
        return {
            "embedding": global_emb,
            "sequence_repr": reprs,
            "rotation_augmented": False,
        }


def build_model(
    enable_fingerprint_2d: bool,
    device: str,
    use_real_backbone: bool = False,
    esm_model: str = "esm2_t12_35M_UR50D",
    checkpoint: str = None,
) -> TorusFold:
    """Construct a TorusFold with the given m6A routing.

    Args:
        enable_fingerprint_2d: True = 2D m6A proxy (1-pair_probs), False = 3D.
        device: 'cpu' or 'cuda'.
        use_real_backbone: If True, use CircEquivariantBackbone (ESM2/RNA-FM)
            which requires `fair-esm` installed and will download ~140 MB.
            If False (default), use MockBackbone (no download, CPU-only).
        esm_model: ESM2 model name (only used when use_real_backbone=True).
        checkpoint: Path to a checkpoint file. Two formats supported:
            1. TorusFold.save() format: has per-module keys (pair_init,
               pairformer, ..., immune_heads) + config dict. Loaded via
               model.load(path).
            2. TorusFoldTrainer format (models/torusfold_best.pt): has
               'model_state_dict' + 'config'. Loaded via partial weight
               transfer (pairformer/composite/etc., but NOT backbone or
               immune_heads since Trainer lacks them).
    """
    if use_real_backbone:
        # Real ESM2 backbone — d_model must match ESM2 output (480 or 640).
        d_model_map = {
            "esm2_t6_8M_UR50D": 320,
            "esm2_t12_35M_UR50D": 480,
            "esm2_t30_150M_UR50D": 640,
            "esm2_t33_650M_UR50D": 1280,
        }
        d_model = d_model_map.get(esm_model, 480)
    else:
        d_model = 64

    config = TorusFoldConfig(
        d_model=d_model,
        n_torus_layers=1,
        n_pairformer_blocks=2,
        n_heads_tri=2,
        c_z=32,
        structure_mode="torus",
        hidden_dim=64,
        dropout=0.1,
        n_rot_augments=0,
        enable_immune_fingerprints=True,
        enable_pkr_head=True,
        enable_nlrp3_head=True,
        enable_drach_head=True,
        enable_tlr7_head=True,
        enable_sponge_head=True,
        enable_rigi_head=False,
        enable_fingerprint_2d=enable_fingerprint_2d,
    )
    model = TorusFold(config)

    if use_real_backbone:
        # CircEquivariantBackbone will auto-load ESM2 on first forward.
        # (calls self.backbone.load_backbone() if not loaded yet.)
        pass
    else:
        model.backbone = MockBackbone(d_model=d_model).to(device)

    model = model.to(device)

    # --- Checkpoint loading ---
    if checkpoint is not None:
        import os
        if not os.path.isfile(checkpoint):
            print(f"  WARNING: checkpoint not found at {checkpoint}, "
                  f"using random weights.")
        else:
            state = torch.load(checkpoint, map_location=device,
                               weights_only=False)
            if "model_state_dict" in state:
                # TorusFoldTrainer format — partial transfer.
                sd = state["model_state_dict"]
                # Transfer shared modules (pair_init, pairformer, pair_head,
                # composite_head, report_head, response_head, bsj_analyzer).
                transferred, skipped = [], []
                for key in sd:
                    # Map Trainer keys to TorusFold module names.
                    # Trainer stores as "module.param", TorusFold as separate
                    # state dicts per module.
                    parts = key.split(".", 1)
                    module_name = parts[0]
                    if module_name in ("backbone",):
                        # Don't transfer backbone — dimensions may differ.
                        skipped.append(key)
                        continue
                    if hasattr(model, module_name):
                        target_module = getattr(model, module_name)
                        param_name = parts[1] if len(parts) > 1 else None
                        if param_name and hasattr(target_module, "state_dict"):
                            module_sd = target_module.state_dict()
                            if param_name in module_sd and \
                               module_sd[param_name].shape == sd[key].shape:
                                module_sd[param_name] = sd[key]
                                target_module.load_state_dict(module_sd)
                                transferred.append(key)
                            else:
                                skipped.append(key)
                        else:
                            skipped.append(key)
                    else:
                        skipped.append(key)
                print(f"  Checkpoint (Trainer format): transferred "
                      f"{len(transferred)}/{len(sd)} params, "
                      f"skipped {len(skipped)}")
            else:
                # TorusFold.save() format — direct load.
                try:
                    model.load(checkpoint)
                    print(f"  Checkpoint (TorusFold format) loaded from "
                          f"{checkpoint}")
                except Exception as e:
                    print(f"  WARNING: checkpoint load failed: {e}, "
                          f"using random weights.")

    model.eval()
    return model


def run_one(label: str, enable_2d: bool, args, device: str) -> Dict:
    """Run a single evolution and collect per-generation stats."""
    print("\n" + "=" * 60)
    print(f"  Run: {label} (enable_fingerprint_2d={enable_2d})")
    print("=" * 60)

    model = build_model(
        enable_fingerprint_2d=enable_2d,
        device=device,
        use_real_backbone=args.real_backbone,
        esm_model=args.esm_model,
        checkpoint=args.checkpoint,
    )

    evo_config = EvolutionConfig(
        population_size=args.population,
        elite_size=max(2, args.population // 5),
        n_generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        use_torusfold=True,
        fitness_mode=args.fitness_mode,
        saliency_guided_mutation=False,  # keep both runs identical; saliency off
        torusfold_device=device,
    )

    evolution = CircRNAEvolution(evo_config, torusfold_model=model)
    evolution.initialize_population(
        seed_sequences=[args.seed] if args.seed else None
    )
    result = evolution.run_evolution()

    # Per-generation convergence curve.
    curve = []
    for gen, stats in enumerate(evolution.history):
        curve.append({
            "gen": gen,
            "mean_fitness": stats["mean_fitness"],
            "max_fitness": stats["max_fitness"],
            "std_fitness": stats["std_fitness"],
        })

    print(f"  {label} best_fitness={result['best_fitness']:.4f} "
          f"gens={result['generations']}")

    return {
        "label": label,
        "enable_fingerprint_2d": enable_2d,
        "best_fitness": result["best_fitness"],
        "best_sequence": result["best_sequence"],
        "generations": result["generations"],
        "converged": result.get("convergence", False),
        "curve": curve,
    }


def write_outputs(results: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON: full curves.
    with open(out_dir / "ablation_fingerprint_2d.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # CSV: flat per-gen-per-run for easy plotting.
    with open(out_dir / "ablation_fingerprint_2d.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "enable_fingerprint_2d", "gen", "mean_fitness",
                    "max_fitness", "std_fitness"])
        for r in results:
            for row in r["curve"]:
                w.writerow([r["label"], r["enable_fingerprint_2d"], row["gen"],
                            row["mean_fitness"], row["max_fitness"],
                            row["std_fitness"]])

    print(f"\nWrote: {out_dir / 'ablation_fingerprint_2d.json'}")
    print(f"Wrote: {out_dir / 'ablation_fingerprint_2d.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description="Evolution ablation: 2D vs 3D m6A proxy (orbit C)"
    )
    parser.add_argument("--seed", type=str,
                        default="AUGCGACUCGUAGAUCCUAACGUCGUAGCUAAGCUAUUCGUA"
                                "AUGCGACUCGUAGAUCCUAACGUCGUAGCUAAGCUAUUCGUA",
                        help="Seed circRNA sequence")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--crossover-rate", type=float, default=0.7)
    parser.add_argument("--fitness-mode", type=str, default="balanced",
                        choices=["balanced", "immunogenic", "therapeutic",
                                 "multi_objective"])
    parser.add_argument("--output", type=str, default="results/ablation")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--real-backbone", action="store_true",
                        help="Use real ESM2 backbone (requires fair-esm, ~140MB "
                             "download). Default: MockBackbone (CPU, no download).")
    parser.add_argument("--esm-model", type=str, default="esm2_t12_35M_UR50D",
                        help="ESM2 model name (only used with --real-backbone). "
                             "Options: esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, "
                             "esm2_t30_150M_UR50D, esm2_t33_650M_UR50D")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a TorusFold or TorusFoldTrainer checkpoint "
                             "to initialize weights. Two formats supported:\n"
                             "  1. TorusFold.save() format (per-module state dicts) "
                             "→ loaded via model.load()\n"
                             "  2. TorusFoldTrainer format (models/torusfold_best.pt) "
                             "→ partial transfer of shared modules\n"
                             "NOTE: Current models/torusfold_best.pt is a Trainer "
                             "checkpoint — it lacks immune_heads, so those stay "
                             "random. For a full Scheme 9 checkpoint, train with "
                             "structure_mode=torus and save via TorusFold.save().")
    args = parser.parse_args()

    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else "cpu" if args.device == "auto" else args.device)
    print(f"Device: {device}")
    print(f"Seed length: {len(args.seed)} nt")
    print(f"Mode: {args.fitness_mode}, pop={args.population}, gens={args.generations}")

    results = [
        run_one("3D_torus_radius", enable_2d=False, args=args, device=device),
        run_one("2D_pair_probs",   enable_2d=True,  args=args, device=device),
    ]

    # Summary comparison.
    r3, r2 = results[0], results[1]
    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    print(f"  3D (torus radius)  best_fitness = {r3['best_fitness']:.4f}  "
          f"gens = {r3['generations']}")
    print(f"  2D (1-pair_probs)  best_fitness = {r2['best_fitness']:.4f}  "
          f"gens = {r2['generations']}")
    delta = r2["best_fitness"] - r3["best_fitness"]
    print(f"  Δ(2D - 3D) = {delta:+.4f}")
    if abs(delta) < 1e-6:
        print("  ⚠ Identical fitness — mock backbone may lack signal, or")
        print("    m6A weight is too low in this fitness_mode to matter.")

    write_outputs(results, Path(args.output))


if __name__ == "__main__":
    main()
