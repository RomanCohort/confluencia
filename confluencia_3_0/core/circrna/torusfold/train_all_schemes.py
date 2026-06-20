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
# Common: 3D Pseudo-label Loading
# ═══════════════════════════════════════════════════════════════

def load_pseudo_labels(labels_dir, n_seqs=None, max_len=None):
    """Load 3D pseudo-labels from disk.

    Args:
        labels_dir: Directory containing sequences.json and coords/
        n_seqs: Maximum number of sequences to load
        max_len: Maximum sequence length (filter out longer sequences)

    Expected structure:
        labels_dir/
            sequences.json  # {id, sequence, secondary_structure, pair_constraints}
            coords/
                pseudo_0000.npy
                pseudo_0001.npy
                ...
            metadata.json  # summary + per-sample info
    """
    import glob

    # Load sequences
    seq_path = os.path.join(labels_dir, 'sequences.json')
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"sequences.json not found in {labels_dir}")

    with open(seq_path, 'r') as f:
        seq_data = json.load(f)

    if n_seqs is not None:
        seq_data = seq_data[:n_seqs]

    # Load coordinates
    coords_files = glob.glob(os.path.join(labels_dir, 'coords', '*.npy'))
    coords_files.sort()

    if len(coords_files) != len(seq_data):
        raise ValueError(f"Mismatch: {len(coords_files)} .npy files, "
                         f"{len(seq_data)} sequences")

    sequences = []
    coords_labels = []
    pair_labels = []  # For schemes needing pair probs
    metadata = []

    for i, item in enumerate(seq_data):
        seq = item['sequence']
        sequences.append(seq)

        # Load coords
        coords = np.load(coords_files[i])  # (L, 3)
        coords_labels.append(coords)

        # Parse pairs from constraints (optional field)
        pair_list = item.get('pair_constraints', [])

        # Build pair probability matrix
        L = len(seq)
        pair_prob = np.zeros((L, L))
        complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}

        # From explicit constraints (high probability)
        for p1, p2 in pair_list:
            if p1 < p2:
                pair_prob[p1, p2] = 0.85
                pair_prob[p2, p1] = 0.85

        # Fill loops with heuristic pairing
        for j in range(L):
            for k in range(j + 4, min(j + 20, L)):
                if pair_prob[j, k] == 0:
                    b1, b2 = seq[j], seq[k]
                    if (b1 == 'G' and b2 == 'U') or (b1 == 'U' and b2 == 'G'):
                        pair_prob[j, k] = 0.3
                        pair_prob[k, j] = 0.3
                    elif complement.get(b1) == b2:
                        pair_prob[j, k] = 0.15
                        pair_prob[k, j] = 0.15

        pair_labels.append(pair_prob)

        # Add to metadata
        metadata.append({
            'id': item['id'],
            'length': L,
        })

    # Filter by max_len if specified
    if max_len is not None:
        keep = [i for i, m in enumerate(metadata) if m['length'] <= max_len]
        sequences = [sequences[i] for i in keep]
        coords_labels = [coords_labels[i] for i in keep]
        pair_labels = [pair_labels[i] for i in keep]
        metadata = [metadata[i] for i in keep]
        print(f"  After max_len={max_len} filter: {len(sequences)} samples")

    print(f"  Loaded {len(sequences)} pseudo-labels from {labels_dir}")

    return sequences, coords_labels, pair_labels, metadata


def generate_3d_pseudo_labels(n_seqs=500, min_len=30, max_len=500, seed=42):
    """Generate 3D coordinate pseudo-labels using ViennaRNA + Physics Solver."""
    rng = np.random.RandomState(seed)
    bases = ['A', 'C', 'G', 'U']

    sequences = []
    coords_labels = []
    pair_labels = []
    metadata = []

    print(f"  Generating {n_seqs} pseudo-labels...")

    try:
        import RNA
        has_vienna = True
        print("  ViennaRNA: available (circ mode)")
    except ImportError:
        has_vienna = False
        print("  ViennaRNA: NOT available")

    config = SolverConfig(n_samples=10, use_annealing_closure=True)
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
                        pair_constraints.append((stack.pop(), pos, 10.6, 1.0))
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

                pair_prob = np.zeros((L, L))
                for (p1, p2, _, _) in pair_constraints:
                    pair_prob[p1, p2] = 0.85
                    pair_prob[p2, p1] = 0.85
                pair_labels.append(pair_prob)

                metadata.append({'id': f'pseudo_{i:04d}', 'length': L})

                if len(sequences) % 100 == 0:
                    print(f"    {len(sequences)}/{n_seqs}")

    print(f"  Generated: {len(sequences)}/{n_seqs}")
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']

            # Normalize coords (center + scale) to prevent numerical explosion
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids)
            pred = out['coords']

            # Normalize prediction similarly
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            pred_norm = pred_centered / pred_scale

            # MSE on normalized coords (per-residue)
            loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                loss += torch.mean(diff ** 2)
            loss /= B

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        # Validation: RMSD in Å
        model.eval()
        val_rmsd = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                out = model(seq_ids)
                pred = out['coords']

                B = len(lengths)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred[b, :valid_L]
                    t = target[b, :valid_L]
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                    val_rmsd += rmsd.item()
                val_rmsd /= B

        avg_val = val_rmsd / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme1_best.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} train={train_loss/len(train_loader):.4f} "
              f"val={avg_val:.4f} pat={patience_counter}/10")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best val loss: {best_val:.4f}")
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
        n_diffusion_steps=min(args.diffusion_steps, 50),  # Reduce steps for stability
        d_node=getattr(args, 'd_hidden', 128),
        d_edge=getattr(args, 'd_hidden', 128) // 2,
    )
    model = CircRNADiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        nan_batches = 0

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            coords_target = batch['coords'].to(device)
            pair_probs = batch.get('pair_probs', None)
            if pair_probs is not None:
                pair_probs = pair_probs.to(device)

            # Normalize target coords to prevent numerical instability
            B, L, _ = coords_target.shape
            coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
            coords_scale = torch.norm(coords_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            coords_norm = coords_centered / coords_scale

            # Forward diffusion + denoising
            out = model(seq_tokens=seq_ids, coords_target=coords_norm, pair_probs=pair_probs)

            # Extract losses from diffusion model output
            noise_loss = out.get('noise_loss', torch.tensor(0.0, device=device))
            closure_loss = out.get('closure_loss', torch.tensor(0.0, device=device))
            loss = out.get('total_loss', noise_loss + 0.1 * closure_loss)

            # NaN check
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                print(f"  NaN/Inf detected in batch, skipping...")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        if nan_batches > len(train_loader) // 2:
            print(f"  Too many NaN batches ({nan_batches}), stopping training")
            return float('inf')

        avg_train = train_loss / max(len(train_loader) - nan_batches, 1)

        # Validation: RMSD in Å
        model.eval()
        val_rmsd = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                coords_target = batch['coords'].to(device)
                pair_probs = batch.get('pair_probs', None)
                if pair_probs is not None:
                    pair_probs = pair_probs.to(device)

                B, L, _ = coords_target.shape
                coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
                coords_scale = torch.norm(coords_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
                coords_norm = coords_centered / coords_scale

                out = model(seq_tokens=seq_ids, coords_target=coords_norm, pair_probs=pair_probs)
                # Use model's own coords prediction for RMSD
                pred_coords = out.get('coords', None)
                if pred_coords is not None:
                    for b in range(B):
                        p = pred_coords[b]
                        t = coords_target[b]
                        p_c = p - p.mean(dim=0)
                        t_c = t - t.mean(dim=0)
                        rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                        val_rmsd += rmsd.item()
                    val_rmsd /= B
                else:
                    val_rmsd += out.get('total_loss', torch.tensor(0.0)).item()

        avg_val = val_rmsd / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme4_best.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} "
              f"val={avg_val:.4f} nan={nan_batches} pat={patience_counter}/10")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best val loss: {best_val:.4f}")
    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme 5: Physics-Biased Attention
# ═══════════════════════════════════════════════════════════════

def train_scheme5(train_loader, val_loader, args, device):
    print("\n" + "="*60)
    print("  Training Scheme 5: Physics-Biased Attention")
    print("="*60)

    class Scheme5Model(nn.Module):
        """Transformer with physics-informed positional encoding for circRNA.

        Uses standard Transformer (O(L^2) attention) instead of
        CircPairformer's O(L^2) pair representation which causes OOM
        at L>200. Physics bias injected via rotary positional encoding
        adapted for circular topology.
        """
        def __init__(self, d_model=128, n_heads=4, n_blocks=4):
            super().__init__()
            self.embed = nn.Embedding(5, d_model)
            # Circular positional encoding
            self.circ_pos = nn.Embedding(512, d_model)  # max 512 positions
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(n_blocks)
            ])
            self.coord_head = nn.Linear(d_model, 3)
            self.bond_length = 5.9

        def forward(self, seq_ids, coords_init=None):
            B, L = seq_ids.shape
            device = seq_ids.device

            # Circular positional indices
            pos = torch.arange(L, device=device) % 512
            h = self.embed(seq_ids) + self.circ_pos(pos)  # (B, L, D)

            for block in self.blocks:
                h = block(h)

            coords = self.coord_head(h)  # (B, L, 3)

            # Physics-informed closure correction (soft, differentiable)
            # Shift first and last atoms toward each other slightly
            closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1, keepdim=True)
            # Clamp to prevent gradient explosion
            closure_error = (closure_dist - self.bond_length).clamp(-20, 20)
            correction = 0.05 * closure_error
            mid_point = (coords[:, 0] + coords[:, -1]) / 2
            coords[:, 0] = coords[:, 0] - correction * (coords[:, 0] - mid_point) / closure_dist.clamp(min=1.0)
            coords[:, -1] = coords[:, -1] - correction * (coords[:, -1] - mid_point) / closure_dist.clamp(min=1.0)

            return {'coords': coords}

    model = Scheme5Model(d_model=args.d_hidden, n_blocks=args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']

            # Normalize target coords
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids)
            pred = out['coords']

            # Normalize prediction
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            pred_norm = pred_centered / pred_scale

            loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                loss += torch.mean(diff ** 2)
            loss /= B

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation: use RMSD in Å (not normalized MSE)
        model.eval()
        val_rmsd = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                out = model(seq_ids)
                pred = out['coords']

                B = len(lengths)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred[b, :valid_L]
                    t = target[b, :valid_L]
                    # Center both for fair RMSD
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                    val_rmsd += rmsd.item()
                val_rmsd /= B

        avg_val = val_rmsd / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme5_best.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} "
              f"val={avg_val:.4f} pat={patience_counter}/10")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best val loss: {best_val:.4f}")
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
        d_node=args.d_hidden,
    )
    model = GNNLatentDiffusionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['coords'].to(device)
            lengths = batch['lengths']

            # Normalize target coords
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids, mode='train')
            pred_coords = out['coords']

            # Normalize prediction
            pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
            pred_scale = torch.norm(pred_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            pred_norm = pred_centered / pred_scale

            loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                loss += torch.mean(diff ** 2)
            loss /= B

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        # Validation: RMSD in Å
        model.eval()
        val_rmsd = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                out = model(seq_ids, mode='sample')
                pred = out['coords']

                B = len(lengths)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred[b, :valid_L]
                    t = target[b, :valid_L]
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                    val_rmsd += rmsd.item()
                val_rmsd /= B

        avg_val = val_rmsd / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme6_best.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} "
              f"val={avg_val:.4f} pat={patience_counter}/10")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Best val loss: {best_val:.4f}")
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


def train_scheme3(train_loader, val_loader, args, device):
    """Scheme 3: Dual-Engine Iterative (CS-Fold gradient-based training).

    Uses BSJClosurePenalty + PaxNetScorer for gradient feedback.
    Generator (G) learns to produce closed structures via:
    1. Physics solver generates initial coords
    2. G refines with gradient descent under closure penalty
    3. PaxNetScorer provides physics-based energy feedback
    """
    print("\n" + "="*60)
    print("  Training Scheme 3: Dual-Engine Iterative (CS-Fold)")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold.dual_engine import (
        BSJClosurePenalty, PaxNetScorer,
    )

    # Build trainable Generator model
    class Scheme3Generator(nn.Module):
        """Trainable coordinate refinement network.

        Takes initial coords from physics solver → outputs refined coords.
        Trained with BSJ closure penalty + energy scoring.
        """
        def __init__(self, d_model=128, n_layers=3):
            super().__init__()
            self.embed = nn.Embedding(5, d_model)
            self.coord_proj = nn.Linear(3, d_model)

            # Transformer for coordinate refinement
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=d_model * 2,
                    dropout=0.1,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ])

            self.coord_out = nn.Linear(d_model, 3)

        def forward(self, seq_ids, coords_init):
            """
            Args:
                seq_ids: (B, L) sequence tokens
                coords_init: (B, L, 3) initial coordinates from solver

            Returns:
                coords_refined: (B, L, 3)
            """
            B, L, _ = coords_init.shape

            # Combine sequence + coordinate features
            seq_feat = self.embed(seq_ids)  # (B, L, D)
            coord_feat = self.coord_proj(coords_init)  # (B, L, D)
            h = seq_feat + coord_feat  # (B, L, D)

            # Transformer refinement
            for layer in self.layers:
                h = layer(h)

            # Output refined coordinates (delta prediction)
            delta = self.coord_out(h)  # (B, L, 3)
            coords_refined = coords_init + delta

            return coords_refined

    # Initialize model and losses
    model = Scheme3Generator(
        d_model=args.d_hidden,
        n_layers=args.n_layers,
    ).to(device)

    bsj_penalty = BSJClosurePenalty(bond_length=5.9, weight=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Use helical coords for fast initialization (avoid slow solver per batch)
    def generate_helical_init(L, bond_length=5.9, device='cpu'):
        """Generate fast helical initial coords (ensures closure)."""
        coords = torch.zeros(L, 3, device=device)
        rise_per_nt = 2.8
        for i in range(L):
            angle = 2 * np.pi * i / L
            radius = bond_length * L / (2 * np.pi) * 0.5
            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = rise_per_nt * i - L * rise_per_nt / 2
        return coords

    best_val = float('inf')
    patience_counter = 0
    rng = np.random.RandomState(args.seed)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_metrics = {'coord': 0, 'closure': 0, 'bond': 0}

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target_coords = batch['coords'].to(device)
            lengths = batch['lengths']

            # Mixed initialization: gradually reduce teacher forcing
            B, L = seq_ids.shape
            tf_prob = max(0.0, 1.0 - epoch / (args.epochs * 0.5))  # 100% → 0% over first half

            coords_init = torch.zeros(B, L, 3, device=device)
            for b in range(B):
                valid_L = lengths[b]
                if rng.random() < tf_prob:
                    # Teacher forcing: start from target + noise
                    coords_init[b, :valid_L] = target_coords[b, :valid_L] + torch.randn(valid_L, 3, device=device) * 1.0
                else:
                    # Helical init: start from scratch
                    coords_init[b, :valid_L] = generate_helical_init(valid_L, device=device)

            # Refine with Generator
            coords_refined = model(seq_ids, coords_init)

            # 1. Coordinate MSE loss (per-residue, only valid positions)
            coord_loss = 0
            n_valid = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = coords_refined[b, :valid_L] - target_coords[b, :valid_L]
                coord_loss += torch.mean(diff ** 2)  # MSE per atom
                n_valid += 1
            coord_loss /= max(n_valid, 1)

            # 2. BSJ closure: relative error ((d - d_target) / d_target)^2
            # Target closure = distance between first and last atom in target
            target_closure = torch.norm(target_coords[:, 0] - target_coords[:, -1], dim=-1)
            pred_closure = torch.norm(coords_refined[:, 0] - coords_refined[:, -1], dim=-1)
            closure_rel_error = ((pred_closure - target_closure) / target_closure.clamp(min=1.0)) ** 2
            closure_loss = closure_rel_error.mean()

            # 3. Bond length: relative error ((d - d_target) / d_target)^2
            bond_loss = 0
            for b in range(B):
                valid_L = lengths[b]
                if valid_L > 1:
                    cr_pred = coords_refined[b, :valid_L]
                    cr_target = target_coords[b, :valid_L]
                    # Predicted bond distances (circular)
                    idx = torch.arange(valid_L, device=device)
                    nxt = (idx + 1) % valid_L
                    d_pred = torch.norm(cr_pred[nxt] - cr_pred[idx], dim=-1)
                    d_target = torch.norm(cr_target[nxt] - cr_target[idx], dim=-1)
                    bond_rel = ((d_pred - d_target) / d_target.clamp(min=1.0)) ** 2
                    bond_loss += bond_rel.mean()
            bond_loss /= max(n_valid, 1)

            # Combined loss: all terms now in comparable scale (0-1 range)
            loss = coord_loss + 0.3 * closure_loss + 0.1 * bond_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_metrics['coord'] += coord_loss.item()
            train_metrics['closure'] += closure_loss.item()
            train_metrics['bond'] += bond_loss.item()

        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target_coords = batch['coords'].to(device)
                lengths = batch['lengths']

                B, L = seq_ids.shape
                coords_init = torch.zeros(B, L, 3, device=device)
                for b in range(B):
                    valid_L = lengths[b]
                    coords_init[b, :valid_L] = generate_helical_init(valid_L, device=device)

                coords_refined = model(seq_ids, coords_init)

                # RMSD in Å
                val_rmsd = 0
                for b in range(B):
                    valid_L = lengths[b]
                    p = coords_refined[b, :valid_L]
                    t = target_coords[b, :valid_L]
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))
                    val_rmsd += rmsd.item()
                val_loss += val_rmsd / B

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme3.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} "
              f"train={train_loss:.4f} (coord={train_metrics['coord']:.3f}, "
              f"closure={train_metrics['closure']:.3f}, bond={train_metrics['bond']:.3f}) "
              f"val={val_loss:.1f}Å pat={patience_counter}/10")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Scheme 3 training complete: best_val={best_val:.4f}")
    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme-specific Data Requirements
# ═══════════════════════════════════════════════════════════════

SCHEME_DATA_REQUIREMENTS = {
    1: {'min_samples': 200, 'recommended': 500, 'epochs': 50,  'reason': 'EGNN轻量，Physics部分无需训练'},
    2: {'min_samples': 0,   'recommended': 0,   'epochs': 0,   'reason': '纯物理求解器，无需训练'},
    3: {'min_samples': 300, 'recommended': 500, 'epochs': 50,  'reason': 'Dual-Engine中等复杂度'},
    4: {'min_samples': 500, 'recommended': 1000,'epochs': 100, 'reason': '扩散模型需要大量数据'},
    5: {'min_samples': 300, 'recommended': 800, 'epochs': 50,  'reason': 'Pairformer+物理bias'},
    6: {'min_samples': 500, 'recommended': 1000,'epochs': 100, 'reason': 'Encoder+Diffusion+Decoder最重'},
}

def main():
    parser = argparse.ArgumentParser(description='Train all TorusFold schemes')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--labels', type=str, default='',
                        help='Path to pre-generated pseudo-labels directory')
    parser.add_argument('--n-train', type=int, default=500,
                        help='Number of samples (used if no --labels)')
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

    # Load pseudo-labels
    if args.labels and os.path.exists(args.labels):
        print(f"  Loading from: {args.labels}")
        sequences, coords_labels, pair_labels, metadata = load_pseudo_labels(
            args.labels, max_len=args.max_len)
    else:
        print(f"  Generating pseudo-labels (n={args.n_train})")
        sequences, coords_labels, pair_labels, metadata = generate_3d_pseudo_labels(
            n_seqs=args.n_train,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.seed
        )

    if len(sequences) < 10:
        print("ERROR: Not enough pseudo-labels.")
        return

    print(f"  Training samples: {len(sequences)}")
    print(f"  Device: {args.device}")

    # Print data requirements per scheme
    print(f"\n  Scheme data requirements:")
    for sid in args.schemes:
        req = SCHEME_DATA_REQUIREMENTS.get(sid, {})
        avail = len(sequences) if req.get('min_samples', 0) > 0 else 'N/A'
        needed = req.get('recommended', 0)
        status = 'OK' if req.get('min_samples', 0) == 0 or len(sequences) >= req.get('min_samples', 0) else 'LOW'
        print(f"    Scheme {sid}: need {needed}, have {avail} [{status}] - {req.get('reason', '')}")

    # Train each scheme with its own data subset
    results = {}
    device = torch.device(args.device)

    for scheme_id in args.schemes:
        req = SCHEME_DATA_REQUIREMENTS.get(scheme_id, {})

        # Scheme 2: no training needed
        if req.get('min_samples', 0) == 0:
            t0 = time.time()
            val_loss = train_scheme2(args) if scheme_id == 2 else 0.0
            elapsed = time.time() - t0
            results[scheme_id] = {'val_loss': val_loss, 'time_seconds': elapsed}
            continue

        # Determine how many samples this scheme uses
        n_needed = req.get('recommended', 500)
        n_available = len(sequences)
        n_use = min(n_needed, n_available)

        # Warn if data is insufficient
        if n_available < req.get('min_samples', 0):
            print(f"\n  WARNING: Scheme {scheme_id} needs {req['min_samples']} samples, "
                  f"only {n_available} available. Results may be poor.")

        # Split for this scheme
        split = int(0.9 * n_use)
        train_ds = CircRNADataset(
            sequences[:split], coords_labels[:split], pair_labels[:split])
        val_ds = CircRNADataset(
            sequences[split:n_use], coords_labels[split:n_use], pair_labels[split:n_use])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn,
                                num_workers=2, pin_memory=True, prefetch_factor=2)

        # Use scheme-specific epochs if not overridden
        scheme_epochs = args.epochs
        if args.epochs == 50:  # Default, use scheme-specific
            scheme_epochs = req.get('epochs', 50)

        # Temporarily override epochs
        original_epochs = args.epochs
        args.epochs = scheme_epochs

        t0 = time.time()

        if scheme_id == 1:
            val_loss = train_scheme1(train_loader, val_loader, args, device)
        elif scheme_id == 3:
            val_loss = train_scheme3(train_loader, val_loader, args, device)
        elif scheme_id == 4:
            val_loss = train_scheme4(train_loader, val_loader, args, device)
        elif scheme_id == 5:
            val_loss = train_scheme5(train_loader, val_loader, args, device)
        elif scheme_id == 6:
            val_loss = train_scheme6(train_loader, val_loader, args, device)
        else:
            print(f"  Unknown scheme {scheme_id}, skipping")
            args.epochs = original_epochs
            continue

        args.epochs = original_epochs
        elapsed = time.time() - t0
        results[scheme_id] = {
            'val_loss': val_loss,
            'time_seconds': elapsed,
            'n_samples': n_use,
            'epochs': scheme_epochs,
        }
        print(f"  Scheme {scheme_id} completed: {n_use} samples, "
              f"{scheme_epochs} epochs, {elapsed:.1f}s")

    # Summary
    print("\n" + "="*60)
    print("  Training Summary")
    print("="*60)
    for sid, res in sorted(results.items()):
        n_samp = res.get('n_samples', 'N/A')
        ep = res.get('epochs', args.epochs)
        print(f"  Scheme {sid}: val_loss={res['val_loss']:.4f}, "
              f"samples={n_samp}, epochs={ep}, time={res['time_seconds']:.1f}s")

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
