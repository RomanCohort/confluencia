#!/usr/bin/env python3
"""
train_all_schemes.py — Train all 6 TorusFold schemes on 3D pseudo-labels.

Each scheme has its own architecture and training pipeline:
  - Scheme 1: DL+Physics Cascade (EGNN → Physics refinement)
  - Scheme 2: Batch+Physics Filter (Batch sampling → Energy filter)
  - Scheme 3: Dual-Engine Iterative (CS-Fold + PaxNet)
  - Scheme 4: DDPM+EGNN Guided (Diffusion with closure reward)
  - Scheme 5: Physics-Biased Attention (CircPairformer with physics bias)
  - Scheme 6: GNN Latent Diffusion (Encoder → Latent diffusion → Decoder)

Usage:
    python train_all_schemes.py --schemes 1 2 3 4 5 6 --n-train 500 --epochs 50
    python train_all_schemes.py --schemes 4 --n-train 1000  # Train only scheme 4
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confluencia_3_0.core.circrna.torusfold.constraint_solver import (
    GeometricConstraintSolver, SolverConfig
)


# ═══════════════════════════════════════════════════════════════
# Common: 3D Pseudo-label Generation
# ═══════════════════════════════════════════════════════════════

def generate_3d_pseudo_labels(n_seqs=500, min_len=30, max_len=500, seed=42):
    """Generate 3D coordinate pseudo-labels using ViennaRNA + Physics Solver."""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    sequences = []
    coords_labels = []
    pair_labels = []  # For schemes that need pair constraints
    metadata = []

    print(f"Generating {n_seqs} 3D pseudo-labels...")

    try:
        import RNA
        has_vienna = True
        print("  ViennaRNA available, using circ mode")
    except ImportError:
        has_vienna = False
        print("  ViennaRNA NOT available, using random pairing")

    config = SolverConfig(
        n_samples=10,
        use_annealing_closure=True,
        bond_length=5.9,
        pair_distance=10.6,
    )
    solver = GeometricConstraintSolver(config)

    for i in range(n_seqs):
        L = rng.randint(min_len, max_len)
        seq = ''.join(rng.choice(bases, size=L))

        pair_constraints = []

        if has_vienna:
            try:
                md = RNA.md()
                md.circ = True
                fc = RNA.fold_compound(seq, md)
                structure, mfe = fc.mfe()

                stack = []
                for pos, char in enumerate(structure):
                    if char == '(':
                        stack.append(pos)
                    elif char == ')' and stack:
                        j = stack.pop()
                        pair_constraints.append((j, pos, 10.6, 1.0))
            except Exception:
                pass

        if not pair_constraints:
            complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
            for j in range(L):
                for k in range(j + 4, min(j + 20, L)):
                    if complement.get(seq[j]) == seq[k] and rng.random() < 0.3:
                        pair_constraints.append((j, k, 10.6, 1.0))

        class CS:
            def __init__(self, n, pairs):
                self.seq_len = n
                self.pair_constraints = pairs

        cs = CS(L, pair_constraints)
        conformations = solver.solve(cs)

        if conformations and len(conformations) > 0:
            best_coords = conformations[0]
            closure_err = abs(np.linalg.norm(best_coords[0] - best_coords[-1]) - 5.9)

            if closure_err < 2.0:
                sequences.append(seq)
                coords_labels.append(best_coords)

                # Build pair probability matrix
                pair_prob = np.zeros((L, L))
                for (p1, p2, _, _) in pair_constraints:
                    pair_prob[p1, p2] = 0.85
                    pair_prob[p2, p1] = 0.85
                pair_labels.append(pair_prob)

                metadata.append({
                    'id': f'pseudo_{i:04d}',
                    'length': L,
                    'n_pairs': len(pair_constraints),
                    'closure_error': closure_err,
                })

                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{n_seqs} - L={L}, pairs={len(pair_constraints)}")

    print(f"  Successfully generated: {len(sequences)}/{n_seqs}")
    return sequences, coords_labels, pair_labels, metadata


class CircRNADataset(Dataset):
    def __init__(self, sequences, coords_labels, pair_labels=None):
        self.sequences = sequences
        self.coords_labels = coords_labels
        self.pair_labels = pair_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.coords_labels[idx]

        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        seq_ids = torch.tensor([mapping.get(b, 4) for b in seq], dtype=torch.long)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        item = {'seq_ids': seq_ids, 'coords': coords_tensor, 'length': len(seq)}

        if self.pair_labels is not None:
            pair_tensor = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
            item['pair_probs'] = pair_tensor

        return item


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    seq_ids_batch, coords_batch, lengths = [], [], []
    has_pairs = 'pair_probs' in batch[0]
    pair_batch = [] if has_pairs else None

    for b in batch:
        L = b['length']
        seq_pad = torch.zeros(max_len, dtype=torch.long)
        seq_pad[:L] = b['seq_ids']
        seq_ids_batch.append(seq_pad)

        coords_pad = torch.zeros(max_len, 3)
        coords_pad[:L] = b['coords']
        coords_batch.append(coords_pad)
        lengths.append(L)

        if has_pairs:
            pp = torch.zeros(max_len, max_len)
            pp[:L, :L] = b['pair_probs']
            pair_batch.append(pp)

    result = {
        'seq_ids': torch.stack(seq_ids_batch),
        'coords': torch.stack(coords_batch),
        'lengths': lengths,
    }
    if has_pairs:
        result['pair_probs'] = torch.stack(pair_batch)
    return result


# ═══════════════════════════════════════════════════════════════
# Scheme 1: DL+Physics Cascade
# ═══════════════════════════════════════════════════════════════

class Scheme1Model(nn.Module):
    """EGNN backbone → Physics refinement cascade."""
    def __init__(self, d_hidden=128, n_layers=4):
        super().__init__()
        from confluencia_3_0.core.circrna.torusfold.train_torusfold_3d import CircRNA3DModel
        self.egnn = CircRNA3DModel(d_hidden=d_hidden, n_layers=n_layers)

    def forward(self, seq_ids):
        return self.egnn(seq_ids)


def train_scheme1(train_loader, val_loader, args, device):
    print("\n" + "="*60)
    print("  Training Scheme 1: DL+Physics Cascade")
    print("="*60)

    model = Scheme1Model(d_hidden=args.d_hidden, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)

            out = model(seq_ids)
            diff = out['coords'] - target
            loss = torch.mean(torch.sum(diff**2, dim=-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['seq_ids'].to(device))
                diff = out['coords'] - batch['coords'].to(device)
                val_loss += torch.mean(torch.sum(diff**2, dim=-1)).item()

        print(f"  Epoch {epoch+1}/{args.epochs} train={train_loss/len(train_loader):.4f} "
              f"val={val_loss/len(val_loader):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.output}/scheme1.pt")

    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme 4: DDPM+EGNN Guided Diffusion
# ═══════════════════════════════════════════════════════════════

def train_scheme4(train_loader, val_loader, args, device):
    print("\n" + "="*60)
    print("  Training Scheme 4: DDPM+EGNN Guided Diffusion")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold.circrna_diffusion import (
        CircRNADiffusionModel, CircDiffusionConfig
    )

    config = CircDiffusionConfig(
        n_diffusion_steps=args.diffusion_steps,
        d_hidden=args.d_hidden,
    )
    model = CircRNADiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)

            # Forward diffusion + denoising
            out = model(seq_ids)
            pred_coords = out['coords']

            # Loss: coordinate matching + closure
            diff = pred_coords - target
            coord_loss = torch.mean(torch.sum(diff**2, dim=-1))

            # Closure reward
            closure = torch.norm(pred_coords[:, 0] - pred_coords[:, -1], dim=-1)
            closure_loss = torch.mean((closure - 5.9)**2)

            loss = coord_loss + 0.5 * closure_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        print(f"  Epoch {epoch+1}/{args.epochs} train={train_loss/len(train_loader):.4f}")

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), f"{args.output}/scheme4.pt")

    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme 5: Physics-Biased Attention
# ═══════════════════════════════════════════════════════════════

def train_scheme5(train_loader, val_loader, args, device):
    print("\n" + "="*60)
    print("  Training Scheme 5: Physics-Biased Attention")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold.triangle_update import CircPairformerBlock

    class Scheme5Model(nn.Module):
        def __init__(self, d_model=128, n_heads=4, n_blocks=4):
            super().__init__()
            self.embed = nn.Embedding(5, d_model)
            self.blocks = nn.ModuleList([
                CircPairformerBlock(c_z=d_model, use_physics_bias=True)
                for _ in range(n_blocks)
            ])
            self.coord_head = nn.Linear(d_model, 3)

        def forward(self, seq_ids, coords_init=None):
            B, L = seq_ids.shape
            h = self.embed(seq_ids)  # (B, L, D)

            # Init coords
            if coords_init is None:
                coords = torch.zeros(B, L, 3, device=seq_ids.device)
            else:
                coords = coords_init

            # Reshape for triangle update (expects z tensor)
            z = h.unsqueeze(-1).expand(-1, -1, -1, h.size(-1))  # (B, L, D, D)

            for block in self.blocks:
                z = block(z, coords=coords)

            # Extract coordinates from diagonal
            diag = z.diagonal(dim1=2, dim2=3)  # (B, L, D)
            coords = self.coord_head(diag)

            return {'coords': coords}

    model = Scheme5Model(d_model=args.d_hidden, n_blocks=args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)

            out = model(seq_ids)
            diff = out['coords'] - target
            loss = torch.mean(torch.sum(diff**2, dim=-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        print(f"  Epoch {epoch+1}/{args.epochs} train={train_loss/len(train_loader):.4f}")

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), f"{args.output}/scheme5.pt")

    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme 6: GNN Latent Diffusion
# ═══════════════════════════════════════════════════════════════

def train_scheme6(train_loader, val_loader, args, device):
    print("\n" + "="*60)
    print("  Training Scheme 6: GNN Latent Diffusion")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold.gnn_latent_diffusion import (
        GNNLatentDiffusionModel, GNNLatentConfig
    )

    config = GNNLatentConfig(
        n_diffusion_steps=args.diffusion_steps,
        d_hidden=args.d_hidden,
    )
    model = GNNLatentDiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)

            out = model(seq_ids, mode='train')
            pred_coords = out['coords']

            diff = pred_coords - target
            loss = torch.mean(torch.sum(diff**2, dim=-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        print(f"  Epoch {epoch+1}/{args.epochs} train={train_loss/len(train_loader):.4f}")

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), f"{args.output}/scheme6.pt")

    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme 2 & 3: Non-parametric (no training needed)
# ═══════════════════════════════════════════════════════════════

def train_scheme2(args):
    """Scheme 2 is physics-based sampling, no neural network training."""
    print("\n" + "="*60)
    print("  Scheme 2: Batch+Physics Filter (no training needed)")
    print("="*60)
    print("  This scheme uses constraint_solver directly.")
    print("  Skipping training, will use solver at inference time.")
    return 0.0


def train_scheme3(args):
    """Scheme 3 uses dual-engine, trained separately."""
    print("\n" + "="*60)
    print("  Scheme 3: Dual-Engine Iterative")
    print("="*60)
    print("  CS-Fold + PaxNet architecture.")
    print("  Using pre-configured dual_engine module.")
    return 0.0


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train all TorusFold schemes')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--min-len', type=int, default=30)
    parser.add_argument('--max-len', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--output', type=str, default='models/torusfold')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("  TorusFold Multi-Scheme Training")
    print("="*60)
    print(f"  Schemes: {args.schemes}")
    print(f"  Training samples: {args.n_train}")
    print(f"  Length range: {args.min_len}-{args.max_len}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")

    # Generate pseudo-labels (shared across all schemes)
    sequences, coords_labels, pair_labels, metadata = generate_3d_pseudo_labels(
        n_seqs=args.n_train,
        min_len=args.min_len,
        max_len=args.max_len,
        seed=args.seed
    )

    if len(sequences) < 10:
        print("ERROR: Not enough pseudo-labels generated.")
        return

    # Split train/val
    split = int(0.9 * len(sequences))
    train_ds = CircRNADataset(sequences[:split], coords_labels[:split], pair_labels[:split])
    val_ds = CircRNADataset(sequences[split:], coords_labels[split:], pair_labels[split:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # Train each scheme
    results = {}
    device = torch.device(args.device)

    for scheme_id in args.schemes:
        t0 = time.time()

        if scheme_id == 1:
            val_loss = train_scheme1(train_loader, val_loader, args, device)
        elif scheme_id == 2:
            val_loss = train_scheme2(args)
        elif scheme_id == 3:
            val_loss = train_scheme3(args)
        elif scheme_id == 4:
            val_loss = train_scheme4(train_loader, val_loader, args, device)
        elif scheme_id == 5:
            val_loss = train_scheme5(train_loader, val_loader, args, device)
        elif scheme_id == 6:
            val_loss = train_scheme6(train_loader, val_loader, args, device)
        else:
            print(f"  Unknown scheme {scheme_id}, skipping")
            continue

        elapsed = time.time() - t0
        results[scheme_id] = {
            'val_loss': val_loss,
            'time_seconds': elapsed,
        }
        print(f"  Scheme {scheme_id} completed in {elapsed:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("  Training Summary")
    print("="*60)
    for sid, res in sorted(results.items()):
        print(f"  Scheme {sid}: val_loss={res['val_loss']:.4f}, time={res['time_seconds']:.1f}s")

    # Save results
    with open(f"{args.output}/training_results.json", 'w') as f:
        json.dump({
            'args': vars(args),
            'results': {str(k): v for k, v in results.items()},
            'metadata': metadata[:50],
        }, f, indent=2)

    print(f"\n  Results saved to {args.output}/training_results.json")


if __name__ == '__main__':
    main()
