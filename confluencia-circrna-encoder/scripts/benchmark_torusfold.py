"""
benchmark_torusfold.py — Reproducible benchmark for TorusFold paper.

Produces quantitative results for:
1. Pathway classification (7-class) with/without circular topology
2. Physics module closure distance statistics
3. ViennaRNA circular vs linear MFE comparison
4. Computational performance (runtime, memory, parameters)

Usage:
    python benchmark_torusfold.py --backbone rna-fm --device cuda
    python benchmark_torusfold.py --backbone mock --device cpu  # quick test
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Data path
DATA_PATH = Path("/root/autodl-tmp/sequences_enhanced.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("D:/IGEM集成方案/data/circrna/sequences_enhanced.csv")
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_ROOT / "data" / "circrna" / "sequences_enhanced.csv"

PATHWAY_MAP = {
    'RIG-I': 0, 'MDA5': 1, 'NF-κB': 2, 'cGAS-STING': 3,
    'JAK-STAT': 4, 'TLR7/8': 5, 'PKR': 6,
}
PATHWAY_NAMES = list(PATHWAY_MAP.keys())
N_PATHWAYS = len(PATHWAY_NAMES)


# ═══════════════════════════════════════════════════════════════════
# Benchmark 1: Pathway Classification (with/without circular topology)
# ═══════════════════════════════════════════════════════════════════

class LinearBaselineClassifier(nn.Module):
    """Baseline: treats circRNA as linear sequence (no circular distance)."""

    def __init__(self, d_model=640, c_z=64, n_pathways=7, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.c_z = c_z

        self.left_proj = nn.Linear(d_model, c_z)
        self.right_proj = nn.Linear(d_model, c_z)
        # Linear distance instead of circular
        self.dist_embed = nn.Embedding(512, c_z)

        # Simple pairformer (no circular bias)
        self.pairformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=c_z, nhead=2, dim_feedforward=c_z*2,
                                        dropout=dropout, batch_first=True)
            for _ in range(2)
        ])
        self.ln = nn.LayerNorm(c_z)

        self.pair_head = nn.Sequential(
            nn.Linear(c_z, c_z), nn.GELU(), nn.Linear(c_z, 1), nn.Sigmoid(),
        )

        input_dim = d_model + c_z + 1
        self.pathway_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_pathways),
        )
        self.immunogenicity_head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, seq_repr, valid_mask, device='cpu'):
        B, L, _ = seq_repr.shape

        left = self.left_proj(seq_repr)
        right = self.right_proj(seq_repr)
        pair = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, c_z)

        # LINEAR distance (not circular!)
        pos = torch.arange(L, device=device)
        linear_dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().clamp(max=511).long()
        pair = pair + self.dist_embed(linear_dist).unsqueeze(0)

        pair_valid = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        pair = pair * pair_valid.unsqueeze(-1)

        # Simple transformer (no circular bias)
        pair_flat = pair.reshape(B * L, L, self.c_z)
        for layer in self.pairformer:
            pair_flat = layer(pair_flat)
        pair_repr = self.ln(pair_flat).reshape(B, L, L, self.c_z)

        pair_probs = self.pair_head(pair_repr).squeeze(-1)
        pair_probs = 0.5 * (pair_probs + pair_probs.transpose(-1, -2))
        pair_probs = pair_probs * pair_valid

        # NO BSJ analysis (linear baseline)
        # Use max pair prob as substitute feature
        bsj_strength = pair_probs.max(dim=-1)[0].mean(dim=-1)  # (B,)

        seq_emb = (seq_repr * valid_mask.unsqueeze(-1)).sum(dim=1) / \
                  valid_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pair_pooled = (pair_repr * pair_valid.unsqueeze(-1)).sum(dim=(1, 2)) / \
                      pair_valid.sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)

        class_input = torch.cat([seq_emb, pair_pooled, bsj_strength.unsqueeze(-1)], dim=-1)

        return {
            'pathway_logits': self.pathway_head(class_input),
            'immunogenicity_logits': self.immunogenicity_head(class_input).squeeze(-1),
            'pair_probs': pair_probs,
            'bsj_strength': bsj_strength,
        }


# ═══════════════════════════════════════════════════════════════════
# Benchmark 2: Physics Module Closure Distance Statistics
# ═══════════════════════════════════════════════════════════════════

def benchmark_physics_module(sequences, n_samples=20):
    """Test physics module closure distance across multiple sequences."""
    from core.physics_bridge import ConstraintExtractor, ConstraintSet
    from core.constraint_solver import GeometricConstraintSolver, SolverConfig
    from core.structure_validator import StructureValidator

    extractor = ConstraintExtractor(c_z=64)  # Use smaller c_z for speed
    solver = GeometricConstraintSolver(SolverConfig(n_samples=n_samples))
    validator = StructureValidator()

    results = []
    for seq in sequences:
        L = len(seq)
        if L < 10:
            continue

        # Create synthetic pair_probs (placeholder — in real use, from CircPairformer)
        pair_probs = torch.rand(1, L, L) * 0.3  # Low prob = mostly unpaired
        pair_repr = torch.randn(1, L, L, 64) * 0.1

        # Extract constraints
        constraint_set = extractor(pair_repr, pair_probs, seq)

        # Solve
        t0 = time.time()
        conformations = solver.solve(constraint_set)
        solve_time = time.time() - t0

        # Validate
        best_coords, metrics = validator.validate_best(conformations, constraint_set)

        results.append({
            'length': L,
            'closure_distance': metrics.closure_distance,
            'closure_score': metrics.closure_score,
            'bond_rmsd': metrics.bond_rmsd,
            'clash_count': metrics.clash_count,
            'pair_satisfaction': metrics.pair_satisfaction,
            'energy_score': metrics.energy_score,
            'stability_score': metrics.stability_score,
            'n_conformations': len(conformations),
            'solve_time_s': solve_time,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# Benchmark 3: ViennaRNA Circular vs Linear MFE
# ═══════════════════════════════════════════════════════════════════

def benchmark_viennarna(sequences, max_len=500):
    """Compare circular vs linear folding for circRNA sequences."""
    try:
        import RNA
    except ImportError:
        print("  ViennaRNA not available — skipping")
        return None

    results = []
    for seq in sequences:
        seq = seq.upper().replace('T', 'U')[:max_len]
        L = len(seq)
        if L < 20:
            continue

        # Linear folding
        md_l = RNA.md()
        fc_l = RNA.fold_compound(seq, md_l)
        ss_l, mfe_l = fc_l.mfe()

        # Circular folding
        md_c = RNA.md()
        md_c.circ = True
        fc_c = RNA.fold_compound(seq, md_c)
        ss_c, mfe_c = fc_c.mfe()

        # Count base pairs
        def count_pairs(ss):
            stack = []
            pairs = 0
            for c in ss:
                if c == '(':
                    stack.append(c)
                elif c == ')':
                    if stack:
                        stack.pop()
                        pairs += 1
            return pairs

        # Count BSJ-crossing pairs in circular structure
        def count_bsj_pairs(ss, L):
            pairs = 0
            stack = []
            pair_list = []
            for i, c in enumerate(ss):
                if c == '(':
                    stack.append(i)
                elif c == ')':
                    if stack:
                        j = stack.pop()
                        pair_list.append((j, i))
                        circ_dist = min(abs(i - j), L - abs(i - j))
                        if circ_dist >= L / 2:
                            pairs += 1
            return pairs, len(pair_list)

        bsj_pairs, total_pairs = count_bsj_pairs(ss_c, L)

        results.append({
            'length': L,
            'gc_content': sum(1 for c in seq if c in 'GC') / L,
            'mfe_linear': mfe_l,
            'mfe_circular': mfe_c,
            'mfe_diff': mfe_c - mfe_l,
            'pairs_linear': count_pairs(ss_l),
            'pairs_circular': total_pairs,
            'bsj_crossing_pairs': bsj_pairs,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# Dataset + Training Helpers
# ═══════════════════════════════════════════════════════════════════

class CircRNADataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'sequence': row['sequence'],
            'pathway': torch.tensor(PATHWAY_MAP.get(row.get('pathway', 'unknown'), 0), dtype=torch.long),
            'immunogenicity': torch.tensor(float(row.get('immunogenicity', 0)), dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'sequences': [b['sequence'] for b in batch],
        'pathway': torch.stack([b['pathway'] for b in batch]),
        'immunogenicity': torch.stack([b['immunogenicity'] for b in batch]),
    }


def train_and_evaluate(model, backbone, train_loader, test_loader, device, epochs=30, lr=5e-4):
    """Train model and return metrics history."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_f1 = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0

        for batch in train_loader:
            pw_target = batch['pathway'].to(device)
            imm_target = batch['immunogenicity'].to(device)

            seq_repr, valid_mask = backbone.encode(batch['sequences'], device, return_per_pos=True)
            out = model(seq_repr, valid_mask, device)

            pw_loss = F.cross_entropy(out['pathway_logits'], pw_target)
            imm_loss = F.binary_cross_entropy_with_logits(out['immunogenicity_logits'], imm_target)
            loss = pw_loss + 0.5 * imm_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_pw_true, all_pw_pred, all_imm_true, all_imm_pred = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                seq_repr, valid_mask = backbone.encode(batch['sequences'], device, return_per_pos=True)
                out = model(seq_repr, valid_mask, device)
                all_pw_true.extend(batch['pathway'].numpy())
                all_pw_pred.extend(out['pathway_logits'].argmax(dim=-1).cpu().numpy())
                all_imm_true.extend(batch['immunogenicity'].numpy())
                all_imm_pred.extend(torch.sigmoid(out['immunogenicity_logits']).cpu().numpy())

        pw_acc = accuracy_score(all_pw_true, all_pw_pred)
        pw_f1 = f1_score(all_pw_true, all_pw_pred, average='macro')
        try:
            imm_auc = roc_auc_score(all_imm_true, all_imm_pred)
        except:
            imm_auc = 0.5

        elapsed = time.time() - t0

        if pw_f1 > best_f1:
            best_f1 = pw_f1

        history.append({
            'epoch': epoch + 1,
            'loss': total_loss / len(train_loader),
            'pathway_acc': pw_acc,
            'pathway_f1': pw_f1,
            'imm_auc': imm_auc,
            'time_s': elapsed,
        })

    return history, best_f1


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DATA_PATH))
    parser.add_argument('--backbone', type=str, default='rna-fm',
                        choices=['rna-fm', 'rinalmo', 'mock'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-seq-len', type=int, default=200)
    parser.add_argument('--c-z', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-physics-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir or PROJECT_ROOT / 'output' / 'benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("TorusFold Benchmark — Bioinformatics Paper Results")
    print("=" * 70)

    # ─── Load Data ────────────────────────────────────────────────
    print("\n[Benchmark 0] Loading data...")
    df = pd.read_csv(args.data)
    df = df[df['sequence'].str.len().between(20, args.max_seq_len)].reset_index(drop=True)
    print(f"  {len(df)} samples (length 20-{args.max_seq_len})")
    print(f"  Pathway distribution: {df['pathway'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed,
                                          stratify=df['pathway'])
    train_loader = DataLoader(CircRNADataset(train_df), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CircRNADataset(test_df), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    # ─── Backbone ─────────────────────────────────────────────────
    print(f"\n[Benchmark 0] Loading backbone ({args.backbone})...")
    if args.backbone == 'rna-fm':
        from scripts.run_circrna_analysis import RNAFMBackbone
        backbone = RNAFMBackbone(freeze=True)
    elif args.backbone == 'rinalmo':
        from scripts.run_circrna_analysis import RiNALMoBackbone
        backbone = RiNALMoBackbone(freeze=True)
    else:
        from scripts.run_circrna_analysis import MockBackbone
        backbone = MockBackbone(d_model=128)

    d_model = backbone.d_model

    # ─── Benchmark 1: CircRNA Classifier (circular) vs Linear Baseline ─
    print("\n[Benchmark 1] Training CircRNA Classifier (circular topology)...")
    from scripts.run_circrna_analysis import CircRNAClassifier

    model_circ = CircRNAClassifier(
        d_model=d_model, c_z=args.c_z,
        n_pairformer_blocks=2, n_pathways=N_PATHWAYS,
    ).to(device)
    n_params_circ = sum(p.numel() for p in model_circ.parameters() if p.requires_grad)
    print(f"  Circular model: {n_params_circ:,} trainable params")

    history_circ, best_f1_circ = train_and_evaluate(
        model_circ, backbone, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr,
    )

    print(f"\n[Benchmark 1] Training Linear Baseline (no circular topology)...")
    model_linear = LinearBaselineClassifier(
        d_model=d_model, c_z=args.c_z, n_pathways=N_PATHWAYS,
    ).to(device)
    n_params_linear = sum(p.numel() for p in model_linear.parameters() if p.requires_grad)
    print(f"  Linear model: {n_params_linear:,} trainable params")

    history_linear, best_f1_linear = train_and_evaluate(
        model_linear, backbone, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr,
    )

    # Final evaluation for both
    model_circ.eval()
    model_linear.eval()

    def final_eval(model, loader, device):
        pw_true, pw_pred, imm_true, imm_pred = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                seq_repr, valid_mask = backbone.encode(batch['sequences'], device, return_per_pos=True)
                out = model(seq_repr, valid_mask, device)
                pw_true.extend(batch['pathway'].numpy())
                pw_pred.extend(out['pathway_logits'].argmax(dim=-1).cpu().numpy())
                imm_true.extend(batch['immunogenicity'].numpy())
                imm_pred.extend(torch.sigmoid(out['immunogenicity_logits']).cpu().numpy())
        return pw_true, pw_pred, imm_true, imm_pred

    pw_true_c, pw_pred_c, imm_true_c, imm_pred_c = final_eval(model_circ, test_loader, device)
    pw_true_l, pw_pred_l, imm_true_l, imm_pred_l = final_eval(model_linear, test_loader, device)

    bench1_results = {
        'circular_topology': {
            'n_params': n_params_circ,
            'pathway_accuracy': float(accuracy_score(pw_true_c, pw_pred_c)),
            'pathway_f1_macro': float(f1_score(pw_true_c, pw_pred_c, average='macro')),
            'immunogenicity_auc': float(roc_auc_score(imm_true_c, imm_pred_c)) if len(set(imm_true_c)) > 1 else 0.5,
            'best_f1': best_f1_circ,
        },
        'linear_baseline': {
            'n_params': n_params_linear,
            'pathway_accuracy': float(accuracy_score(pw_true_l, pw_pred_l)),
            'pathway_f1_macro': float(f1_score(pw_true_l, pw_pred_l, average='macro')),
            'immunogenicity_auc': float(roc_auc_score(imm_true_l, imm_pred_l)) if len(set(imm_true_l)) > 1 else 0.5,
            'best_f1': best_f1_linear,
        },
    }
    print(f"\n  Circular F1={bench1_results['circular_topology']['pathway_f1_macro']:.4f} "
          f"vs Linear F1={bench1_results['linear_baseline']['pathway_f1_macro']:.4f}")

    # ─── Benchmark 2: Physics Module ──────────────────────────────
    print(f"\n[Benchmark 2] Physics module closure distance test...")
    test_sequences = test_df['sequence'].tolist()[:args.n_physics_samples]
    physics_df = benchmark_physics_module(test_sequences, n_samples=20)

    if physics_df is not None and len(physics_df) > 0:
        bench2_results = {
            'n_sequences': len(physics_df),
            'mean_closure_distance': float(physics_df['closure_distance'].mean()),
            'std_closure_distance': float(physics_df['closure_distance'].std()),
            'max_closure_distance': float(physics_df['closure_distance'].max()),
            'mean_closure_score': float(physics_df['closure_score'].mean()),
            'mean_bond_rmsd': float(physics_df['bond_rmsd'].mean()),
            'mean_clash_count': float(physics_df['clash_count'].mean()),
            'mean_pair_satisfaction': float(physics_df['pair_satisfaction'].mean()),
            'mean_energy_score': float(physics_df['energy_score'].mean()),
            'mean_stability_score': float(physics_df['stability_score'].mean()),
            'mean_solve_time_s': float(physics_df['solve_time_s'].mean()),
            'length_range': [int(physics_df['length'].min()), int(physics_df['length'].max())],
        }
        print(f"  Closure distance: {bench2_results['mean_closure_distance']:.3f} ± "
              f"{bench2_results['std_closure_distance']:.3f} Å")
        print(f"  Bond RMSD: {bench2_results['mean_bond_rmsd']:.3f} Å")
        print(f"  Solve time: {bench2_results['mean_solve_time_s']:.2f} s/sequence")
    else:
        bench2_results = {'error': 'No results — check physics module imports'}

    # ─── Benchmark 3: ViennaRNA ───────────────────────────────────
    print(f"\n[Benchmark 3] ViennaRNA circular vs linear folding...")
    viennarna_df = benchmark_viennarna(test_df['sequence'].tolist()[:50])

    if viennarna_df is not None and len(viennarna_df) > 0:
        bench3_results = {
            'n_sequences': len(viennarna_df),
            'mean_mfe_linear': float(viennarna_df['mfe_linear'].mean()),
            'mean_mfe_circular': float(viennarna_df['mfe_circular'].mean()),
            'mean_mfe_diff': float(viennarna_df['mfe_diff'].mean()),
            'n_with_bsj_pairs': int((viennarna_df['bsj_crossing_pairs'] > 0).sum()),
            'mean_bsj_pairs': float(viennarna_df['bsj_crossing_pairs'].mean()),
        }
        print(f"  MFE: linear={bench3_results['mean_mfe_linear']:.1f}, "
              f"circular={bench3_results['mean_mfe_circular']:.1f}, "
              f"diff={bench3_results['mean_mfe_diff']:+.1f} kcal/mol")
        print(f"  Sequences with BSJ-crossing pairs: {bench3_results['n_with_bsj_pairs']}/{bench3_results['n_sequences']}")
    else:
        bench3_results = {'error': 'ViennaRNA not available'}

    # ─── Save All Results ─────────────────────────────────────────
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'backbone': args.backbone,
        'device': str(device),
        'seed': args.seed,
        'epochs': args.epochs,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'data_path': str(args.data),
        'benchmark1_classification': bench1_results,
        'benchmark2_physics': bench2_results,
        'benchmark3_viennarna': bench3_results,
    }

    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")

    # Also save detailed histories
    pd.DataFrame(history_circ).to_csv(output_dir / 'history_circular.csv', index=False)
    pd.DataFrame(history_linear).to_csv(output_dir / 'history_linear.csv', index=False)
    if physics_df is not None:
        physics_df.to_csv(output_dir / 'physics_validation.csv', index=False)
    if viennarna_df is not None:
        viennarna_df.to_csv(output_dir / 'viennarna_comparison.csv', index=False)

    # Print summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print(f"{'Metric':<30} {'Circular':>12} {'Linear':>12}")
    print("-" * 54)
    print(f"{'Pathway Accuracy':<30} {bench1_results['circular_topology']['pathway_accuracy']:>12.4f} "
          f"{bench1_results['linear_baseline']['pathway_accuracy']:>12.4f}")
    print(f"{'Pathway F1 (macro)':<30} {bench1_results['circular_topology']['pathway_f1_macro']:>12.4f} "
          f"{bench1_results['linear_baseline']['pathway_f1_macro']:>12.4f}")
    print(f"{'Immunogenicity AUC':<30} {bench1_results['circular_topology']['immunogenicity_auc']:>12.4f} "
          f"{bench1_results['linear_baseline']['immunogenicity_auc']:>12.4f}")
    print(f"{'Best F1':<30} {bench1_results['circular_topology']['best_f1']:>12.4f} "
          f"{bench1_results['linear_baseline']['best_f1']:>12.4f}")
    if 'mean_closure_distance' in bench2_results:
        print(f"\n{'Physics Module':<30}")
        print(f"{'  Closure distance (Å)':<30} {bench2_results['mean_closure_distance']:>12.3f}")
        print(f"{'  Bond RMSD (Å)':<30} {bench2_results['mean_bond_rmsd']:>12.3f}")
        print(f"{'  Solve time (s)':<30} {bench2_results['mean_solve_time_s']:>12.2f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
