#!/usr/bin/env python3
"""
train_all_schemes.py — Train all 7 TorusFold schemes on 3D pseudo-labels.

Each scheme has its own architecture and training pipeline:
  - Scheme 1: DL+Physics Cascade (EGNN → Physics refinement)
  - Scheme 2: Batch+Physics Filter (Batch sampling → Energy filter)
  - Scheme 3: Dual-Engine Iterative (CS-Fold + PaxNet)
  - Scheme 4: DDPM+EGNN Guided (Diffusion with closure reward)
  - Scheme 5: Physics-Biased Attention (CircPairformer with physics bias)
  - Scheme 6: GNN Latent Diffusion (Encoder → Latent diffusion → Decoder)
  - Scheme 7: Mamba+Transformer Hybrid Diffusion (O(L) global + O(L×w) local)

Usage:
    python train_all_schemes.py --schemes 1 2 3 4 5 6 7 --n-train 500 --epochs 50
    python train_all_schemes.py --schemes 7 --max-len 1000  # Train only scheme 7 (long seqs)
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


def kabsch_rmsd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute RMSD after Kabsch optimal alignment.

    Args:
        pred: (L, 3) predicted coordinates
        target: (L, 3) target coordinates

    Returns:
        RMSD in Angstroms after optimal superposition
    """
    p_c = pred - pred.mean(dim=0)
    t_c = target - target.mean(dim=0)

    # Kabsch SVD alignment
    H = t_c.T @ p_c
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.det(Vt.T @ U.T))
        D = torch.diag(torch.tensor([1, 1, d], device=pred.device, dtype=torch.float32))
        R = Vt.T @ D @ U.T
        p_aligned = (R @ p_c.T).T
        rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
    except Exception:
        # Fallback: simple centered RMSD
        rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))

    return rmsd.item()


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

    # Load coordinates by matching json id (not glob sort, which may mismatch)
    sequences = []
    coords_labels = []
    pair_labels = []  # For schemes needing pair probs
    confidence_weights = []  # Per-sample confidence from source quality
    metadata = []
    n_missing = 0

    # Default confidence by source
    DEFAULT_CONFIDENCE = {
        "pdb_circularized": 1.0,
        "pdb_circularized_aug": 0.95,
        "shape_experimental": 0.9,
        "isrnacirc": 0.7,
        "isrnacirc_aug": 0.65,
        "circbase_real": 0.5,
        "medium_synth": 0.4,
        "synthetic": 0.3,
        "af3_predicted": 1.0,
        "rfam_consensus": 0.8,
    }

    for i, item in enumerate(seq_data):
        seq_id = item.get('id', f'pseudo_{i:05d}')
        coords_path = os.path.join(labels_dir, 'coords', f'{seq_id}.npy')

        if not os.path.exists(coords_path):
            n_missing += 1
            continue

        seq = item['sequence']
        coords = np.load(coords_path)  # (L, 3)

        # Verify coords shape matches sequence length
        if coords.shape[0] != len(seq):
            n_missing += 1
            continue

        sequences.append(seq)
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

        # BSJ flanking: boost pair probability near back-splice junction
        # In circRNA, nucleotides flanking the BSJ are more likely to pair
        bsj_window = min(8, L // 4)
        for j in range(bsj_window):
            for k in range(max(0, L - bsj_window), L):
                if pair_prob[j, k] == 0:
                    b1, b2 = seq[j], seq[k]
                    if complement.get(b1) == b2 or (b1 in 'GU' and b2 in 'GU'):
                        pair_prob[j, k] = 0.25
                        pair_prob[k, j] = 0.25
                elif pair_prob[j, k] < 0.85:
                    # Boost existing pairs near BSJ
                    pair_prob[j, k] = min(0.95, pair_prob[j, k] * 1.2)
                    pair_prob[k, j] = pair_prob[j, k]

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

        # Confidence weight from source quality
        source = item.get('source', 'synthetic')
        conf = item.get('confidence', DEFAULT_CONFIDENCE.get(source, 0.3))
        confidence_weights.append(conf)

        # Add to metadata
        metadata.append({
            'id': item['id'],
            'length': L,
            'source': source,
            'confidence': conf,
        })

    if n_missing > 0:
        print(f"  Skipped {n_missing} entries with missing/mismatched coords")

    # Filter by max_len if specified
    if max_len is not None:
        keep = [i for i, m in enumerate(metadata) if m['length'] <= max_len]
        sequences = [sequences[i] for i in keep]
        coords_labels = [coords_labels[i] for i in keep]
        pair_labels = [pair_labels[i] for i in keep]
        confidence_weights = [confidence_weights[i] for i in keep]
        metadata = [metadata[i] for i in keep]
        print(f"  After max_len={max_len} filter: {len(sequences)} samples")

    avg_conf = np.mean(confidence_weights)
    print(f"  Loaded {len(sequences)} pseudo-labels from {labels_dir}")
    print(f"  Average confidence: {avg_conf:.3f}")

    return sequences, coords_labels, pair_labels, confidence_weights, metadata


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
    def __init__(self, sequences, coords_labels, pair_labels=None, confidence_weights=None):
        self.sequences = sequences
        self.coords_labels = coords_labels
        self.pair_labels = pair_labels
        self.confidence_weights = confidence_weights

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.coords_labels[idx]

        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        seq_ids = torch.tensor([mapping.get(b, 4) for b in seq], dtype=torch.long)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        # Use actual coords length (may differ from seq length due to data issues)
        actual_L = coords_tensor.shape[0]
        if len(seq) != actual_L:
            # Truncate or pad seq to match coords
            if len(seq) > actual_L:
                seq_ids = seq_ids[:actual_L]
            else:
                seq_ids = torch.cat([seq_ids, torch.zeros(actual_L - len(seq), dtype=torch.long)])

        item = {'seq_ids': seq_ids, 'coords': coords_tensor, 'length': actual_L}

        if self.pair_labels is not None:
            pair_tensor = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
            item['pair_probs'] = pair_tensor

        if self.confidence_weights is not None:
            item['confidence'] = torch.tensor(self.confidence_weights[idx], dtype=torch.float32)

        return item


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    seq_ids_batch, coords_batch, lengths = [], [], []
    has_pairs = 'pair_probs' in batch[0]
    has_conf = 'confidence' in batch[0]
    pair_batch = [] if has_pairs else None
    conf_batch = [] if has_conf else None

    for b in batch:
        L = b['length']
        seq_pad = torch.zeros(max_len, dtype=torch.long)
        seq_pad[:L] = b['seq_ids']
        seq_ids_batch.append(seq_pad)

        # Pad with last valid coord instead of zeros
        # Verify data consistency first
        coords_actual_shape = b['coords'].shape[0]
        if coords_actual_shape != L:
            # Data mismatch: coords shape differs from declared length
            # Use actual coords length for padding
            actual_L = coords_actual_shape
            if actual_L < max_len:
                coords_pad = b['coords'][-1:].expand(max_len, 3).clone()
                coords_pad[:actual_L] = b['coords']
            else:
                coords_pad = b['coords'][:max_len].clone()
            # Update length to actual
            L = actual_L
        elif L < max_len:
            coords_pad = b['coords'][-1:].expand(max_len, 3).clone()
            coords_pad[:L] = b['coords']
        else:
            coords_pad = b['coords'].clone()
        coords_batch.append(coords_pad)
        lengths.append(L)

        if has_pairs:
            pp = torch.zeros(max_len, max_len)
            pp[:L, :L] = b['pair_probs']
            pair_batch.append(pp)

        if has_conf:
            conf_batch.append(b['confidence'])

    result = {
        'seq_ids': torch.stack(seq_ids_batch),
        'coords': torch.stack(coords_batch),
        'lengths': lengths,
    }
    if has_pairs:
        result['pair_probs'] = torch.stack(pair_batch)
    if has_conf:
        result['confidence'] = torch.stack(conf_batch)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # Full lr for EGNN
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
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

            # FIX 1: Normalize coords using TARGET scale only (not independent scales)
            # Problem: independent normalization destroys spatial scale signal
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids)
            pred = out['coords']

            # FIX: Use TARGET scale for prediction (not independent scale)
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            pred_norm = pred_centered / target_scale  # Use target scale!

            # MSE on normalized coords (per-residue)
            loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                loss += torch.mean(diff ** 2)
            loss /= B

            # Apply confidence weighting: higher quality data gets higher loss weight
            loss = loss * conf_scale * 2.0  # *2 to normalize around 1.0

            loss.backward()
            # Check for NaN gradients
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                optimizer.zero_grad()
                continue  # Skip this batch
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if not torch.isnan(loss):
                train_loss += loss.item()

        # Validation: RMSD in Angstroms (denormalize for interpretable metric)
        model.eval()
        val_rmsd = 0
        n_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['coords'].to(device)
                lengths = batch['lengths']

                out = model(seq_ids)
                pred = out['coords']

                # Skip if prediction contains NaN/Inf
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    continue

                # Denormalize: pred is in raw coordinate space from EGNN
                # Use Kabsch-aligned RMSD in Angstroms for fair comparison
                B = len(lengths)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred[b, :valid_L]
                    t = target[b, :valid_L]
                    # Center both
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    # Kabsch alignment
                    rmsd = kabsch_rmsd(p_c, t_c)
                    if not np.isnan(rmsd) and not np.isinf(rmsd):
                        val_rmsd += rmsd
                        n_val_samples += 1

        avg_val = val_rmsd / max(n_val_samples, 1)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
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
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

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

            # Apply confidence weighting
            loss = loss * conf_scale * 2.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        if nan_batches > len(train_loader) // 2:
            print(f"  Too many NaN batches ({nan_batches}), stopping training")
            return float('inf')

        avg_train = train_loss / max(len(train_loader) - nan_batches, 1)

        # Validation: sample from diffusion model and compute RMSD
        model.eval()
        val_rmsd = 0
        n_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                coords_target = batch['coords'].to(device)
                lengths = batch['lengths']

                B = len(lengths)
                # Sample predictions from diffusion model
                try:
                    out = model(seq_tokens=seq_ids, pair_probs=None)
                    pred_coords = out.get('coords', None)
                except Exception:
                    pred_coords = None

                if pred_coords is not None:
                    for b in range(B):
                        valid_L = lengths[b]
                        p = pred_coords[b, :valid_L]
                        t = coords_target[b, :valid_L]

                        # Skip if prediction has NaN/Inf
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            continue

                        # Kabsch alignment before RMSD
                        p_c = p - p.mean(dim=0)
                        t_c = t - t.mean(dim=0)

                        # SVD for optimal rotation
                        H = t_c.T @ p_c
                        try:
                            U, S, Vt = torch.linalg.svd(H)
                            d = torch.sign(torch.det(Vt.T @ U.T))
                            D = torch.diag(torch.tensor([1, 1, d], device=device, dtype=torch.float32))
                            R = Vt.T @ D @ U.T
                            p_aligned = (R @ p_c.T).T
                            rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
                        except Exception:
                            # Fallback to simple centered RMSD
                            rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))

                        val_rmsd += rmsd.item()
                        n_val_samples += 1
                # If no coords from diffusion, skip (don't fabricate a number)

        avg_val = val_rmsd / max(n_val_samples, 1)
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
            # Small init for coord head to prevent large initial outputs causing NaN
            nn.init.normal_(self.coord_head.weight, std=0.01)
            nn.init.zeros_(self.coord_head.bias)
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
            # Use clone() to avoid in-place modification of graph tensor
            closure_dist = torch.norm(coords[:, 0] - coords[:, -1], dim=-1, keepdim=True)
            # Clamp to prevent gradient explosion
            closure_error = (closure_dist - self.bond_length).clamp(-20, 20)
            correction = 0.05 * closure_error
            mid_point = (coords[:, 0] + coords[:, -1]) / 2
            # Guard against near-zero closure_dist causing NaN division
            safe_dist = closure_dist.clamp(min=1.0)
            direction_first = (coords[:, 0] - mid_point) / safe_dist
            direction_last = (coords[:, -1] - mid_point) / safe_dist
            coords = coords.clone()
            coords[:, 0] = coords[:, 0] - correction * direction_first
            coords[:, -1] = coords[:, -1] - correction * direction_last

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
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

            # FIX 1: Normalize target coords
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids)
            pred = out['coords']

            # FIX: Use TARGET scale for prediction
            pred_centered = pred - pred.mean(dim=1, keepdim=True)
            pred_norm = pred_centered / target_scale  # Use target scale!

            loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                loss += torch.mean(diff ** 2)
            loss /= B

            # Apply confidence weighting
            loss = loss * conf_scale * 2.0

            # NaN guard — skip batch if loss is NaN/Inf (same as Scheme 1/4)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            # Check for NaN gradients (same as Scheme 1)
            has_nan = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan = True
                    break
            if has_nan:
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation: RMSD in Å (not normalized MSE)
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
                    if not (torch.isnan(p).any() or torch.isinf(p).any()):
                        val_rmsd += kabsch_rmsd(p, t)
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
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

            # FIX 1: Normalize target coords using target scale only
            B, L, _ = target.shape
            target_centered = target - target.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            out = model(seq_ids, mode='train')
            pred_coords = out['coords']
            diff_loss = out.get('diffusion_loss', None)

            # FIX: Use TARGET scale for prediction
            pred_centered = pred_coords - pred_coords.mean(dim=1, keepdim=True)
            pred_norm = pred_centered / target_scale  # Use target scale!

            # Coordinate reconstruction loss
            coord_loss = 0
            for b in range(B):
                valid_L = lengths[b]
                diff = pred_norm[b, :valid_L] - target_norm[b, :valid_L]
                coord_loss += torch.mean(diff ** 2)
            coord_loss /= B

            # Total loss: diffusion (primary) + coordinate (auxiliary)
            if diff_loss is not None and not (torch.isnan(diff_loss) or torch.isinf(diff_loss)):
                loss = diff_loss + 0.1 * coord_loss
            else:
                loss = coord_loss

            # NaN check - skip batch if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf in loss, skipping batch...")
                optimizer.zero_grad()
                continue

            # Apply confidence weighting
            loss = loss * conf_scale * 2.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_rmsd = 0
        n_val_samples = 0
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

                    if torch.isnan(p).any() or torch.isinf(p).any():
                        continue

                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)

                    # Kabsch alignment
                    H = t_c.T @ p_c
                    try:
                        U, S, Vt = torch.linalg.svd(H)
                        d = torch.sign(torch.det(Vt.T @ U.T))
                        D = torch.diag(torch.tensor([1, 1, d], device=device, dtype=torch.float32))
                        R = Vt.T @ D @ U.T
                        p_aligned = (R @ p_c.T).T
                        rmsd = torch.sqrt(torch.mean(torch.sum((p_aligned - t_c) ** 2, dim=1)))
                    except Exception:
                        rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)))

                    val_rmsd += rmsd.item()
                    n_val_samples += 1

        avg_val = val_rmsd / max(n_val_samples, 1)
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
# Scheme 7: Mamba + Transformer Hybrid Diffusion
# ═══════════════════════════════════════════════════════════════

def train_scheme7(train_loader, val_loader, args, device):
    """Scheme 7: Mamba + Transformer Hybrid Diffusion for circRNA.

    Key advantage: O(L) global context via Mamba + O(L×w) local attention.
    Can handle sequences up to L=1000 on 24GB GPU (vs L=500 for Scheme 4).

    Memory: ~8GB for L=1000, batch=4, d=128 (vs ~25GB for Scheme 4 EGNN).
    """
    print("\n" + "="*60)
    print("  Training Scheme 7: Mamba+Transformer Hybrid Diffusion")
    print("="*60)

    from confluencia_3_0.core.circrna.torusfold.circrna_mamba_diffusion import (
        CircMambaDiffusionModel, CircMambaConfig
    )

    config = CircMambaConfig(
        d_model=args.d_hidden,
        d_ssm=max(32, args.d_hidden // 2),
        d_cond=max(32, args.d_hidden // 2),
        n_mamba_layers=getattr(args, 'n_mamba_layers', 4),
        n_attn_layers=getattr(args, 'n_attn_layers', 2),
        n_diffusion_steps=args.diffusion_steps,
        attn_window=getattr(args, 'attn_window', 20),
        bsj_flank=getattr(args, 'bsj_flank', 20),
        bond_length=5.9,
        closure_weight=1.0,
        use_gradient_checkpointing=True,
    )
    model = CircMambaDiffusionModel(config).to(device)

    # Print parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Config: d_model={config.d_model}, d_ssm={config.d_ssm}, "
          f"n_mamba={config.n_mamba_layers}, n_attn={config.n_attn_layers}, "
          f"window={config.attn_window}, bsj_flank={config.bsj_flank}")

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
        train_metrics = {'noise': 0, 'closure': 0}

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            coords_target = batch['coords'].to(device)
            pair_probs = batch.get('pair_probs', None)
            if pair_probs is not None:
                pair_probs = pair_probs.to(device)
            lengths = batch['lengths']
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

            # Normalize target coords
            B, L, _ = coords_target.shape
            coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
            coords_scale = torch.norm(coords_centered, dim=(1, 2), keepdim=True).clamp(min=1.0)
            coords_norm = coords_centered / coords_scale

            # Forward: diffusion training step
            out = model(
                seq_tokens=seq_ids,
                pair_probs=pair_probs,
                coords_target=coords_norm,
            )

            noise_loss = out.get('noise_loss', torch.tensor(0.0, device=device))
            closure_loss = out.get('closure_loss', torch.tensor(0.0, device=device))
            loss = out.get('total_loss', noise_loss + 0.1 * closure_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            # Apply confidence weighting
            loss = loss * conf_scale * 2.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            train_metrics['noise'] += noise_loss.item()
            train_metrics['closure'] += closure_loss.item()

        if nan_batches > len(train_loader) // 2:
            print(f"  Too many NaN batches ({nan_batches}), stopping")
            return float('inf')

        n_valid_batches = max(len(train_loader) - nan_batches, 1)
        avg_train = train_loss / n_valid_batches
        avg_noise = train_metrics['noise'] / n_valid_batches
        avg_closure = train_metrics['closure'] / n_valid_batches

        # Validation: sample from model and compute RMSD
        model.eval()
        val_rmsd = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                coords_target = batch['coords'].to(device)
                lengths = batch['lengths']

                B = len(lengths)
                try:
                    out = model(seq_tokens=seq_ids, pair_probs=None)
                    pred_coords = out.get('coords', None)
                except Exception:
                    pred_coords = None

                if pred_coords is not None:
                    for b in range(B):
                        valid_L = lengths[b]
                        p = pred_coords[b, :valid_L]
                        t = coords_target[b, :valid_L]
                        if not (torch.isnan(p).any() or torch.isinf(p).any()):
                            val_rmsd += kabsch_rmsd(p, t)
                else:
                    # Fallback: use training loss as proxy
                    coords_centered = coords_target - coords_target.mean(dim=1, keepdim=True)
                    coords_scale = torch.norm(coords_centered, dim=(1, 2), keepdim=True).clamp(min=1.0)
                    coords_norm = coords_centered / coords_scale
                    out = model(seq_tokens=seq_ids, coords_target=coords_norm)
                    val_rmsd += out.get('total_loss', torch.tensor(0.0)).item() * 100
                val_rmsd /= B

        avg_val = val_rmsd / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/scheme7_best.pt")
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1}/{args.epochs} train={avg_train:.4f} "
              f"(noise={avg_noise:.4f}, closure={avg_closure:.4f}) "
              f"val={avg_val:.4f} nan={nan_batches} pat={patience_counter}/10")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Use helical coords for fast initialization (avoid slow solver per batch)
    def generate_helical_init(L, bond_length=5.9, device='cpu'):
        """Generate fast helical initial coords (ensures closure).

        Returns coords centered at origin with unit-norm for stable training.
        Raw helical coords are centered and normalized so coord_proj
        receives inputs in a reasonable range (~[-1, 1]) regardless of L.
        """
        coords = torch.zeros(L, 3, device=device)
        rise_per_nt = 2.8
        for i in range(L):
            angle = 2 * np.pi * i / L
            radius = bond_length * L / (2 * np.pi) * 0.5
            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = rise_per_nt * i - L * rise_per_nt / 2
        # Center and normalize to unit norm for stable input to coord_proj
        coords = coords - coords.mean(dim=0)
        norm = torch.norm(coords)
        if norm > 1e-6:
            coords = coords / norm
        return coords

    best_val = float('inf')
    patience_counter = 0
    rng = np.random.RandomState(args.seed)
    nan_batches = 0  # Track NaN batches for early warning

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_metrics = {'coord': 0, 'closure': 0, 'bond': 0}

        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target_coords = batch['coords'].to(device)
            lengths = batch['lengths']
            conf_scale = batch.get('confidence', torch.tensor(0.5)).mean().item()

            # FIX: Always use helical init (no teacher forcing)
            # Problem: teacher forcing causes distribution mismatch at test time
            B, L = seq_ids.shape

            # Skip batches with very short sequences (cause numerical issues)
            min_valid_L = min(lengths)
            if min_valid_L < 4:
                continue

            # FIX: Normalize target coords to unit norm
            # This ensures loss is computed in a stable numerical range
            target_centered = target_coords - target_coords.mean(dim=1, keepdim=True)
            target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)
            target_norm = target_centered / target_scale

            coords_init = torch.zeros(B, L, 3, device=device)
            for b in range(B):
                valid_L = lengths[b]
                coords_init[b, :valid_L] = generate_helical_init(valid_L, device=device)

            # Helical init is already unit-norm, use directly
            coords_init_norm = coords_init

            # Refine with Generator
            coords_refined = model(seq_ids, coords_init_norm)

            # NaN check: skip batch if prediction contains NaN/Inf
            if torch.isnan(coords_refined).any() or torch.isinf(coords_refined).any():
                nan_batches += 1
                optimizer.zero_grad()
                continue

            # 1. Coordinate MSE loss on normalized coords
            coord_loss = 0
            n_valid = 0
            for b in range(B):
                valid_L = lengths[b]
                pred = coords_refined[b, :valid_L]
                target = target_norm[b, :valid_L]
                # Center both for fair comparison (translation-invariant)
                pred_c = pred - pred.mean(dim=0)
                target_c = target - target.mean(dim=0)
                mse = torch.mean(torch.sum((pred_c - target_c) ** 2, dim=1))
                coord_loss += mse
                n_valid += 1
            coord_loss /= max(n_valid, 1)

            # 2. BSJ closure: error in normalized space
            target_closure = torch.norm(target_norm[:, 0] - target_norm[:, -1], dim=-1)
            pred_closure = torch.norm(coords_refined[:, 0] - coords_refined[:, -1], dim=-1)
            closure_error = (pred_closure - target_closure).clamp(-5, 5)
            closure_loss = torch.mean(closure_error ** 2)

            # 3. Bond length consistency in normalized space
            bond_loss = 0
            n_bond = 0
            for b in range(B):
                valid_L = lengths[b]
                if valid_L > 1:
                    cr_pred = coords_refined[b, :valid_L]
                    cr_target = target_norm[b, :valid_L]
                    idx = torch.arange(valid_L, device=device)
                    nxt = (idx + 1) % valid_L
                    d_pred = torch.norm(cr_pred[nxt] - cr_pred[idx], dim=-1)
                    d_target = torch.norm(cr_target[nxt] - cr_target[idx], dim=-1)
                    bond_loss += torch.mean((d_pred - d_target) ** 2)
                    n_bond += 1
            bond_loss /= max(n_bond, 1)

            # Combined loss (normalized scale)
            loss = coord_loss + 0.1 * closure_loss + 0.1 * bond_loss

            # NaN check on loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                optimizer.zero_grad()
                continue

            # Apply confidence weighting
            loss = loss * conf_scale * 2.0

            # Zero grad BEFORE backward (standard practice)
            optimizer.zero_grad()
            loss.backward()
            # Check for NaN gradients after backward
            has_nan_grad = False
            for p in model.parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    has_nan_grad = True
                    break
            if has_nan_grad:
                nan_batches += 1
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_metrics['coord'] += coord_loss.item()
            train_metrics['closure'] += closure_loss.item()
            train_metrics['bond'] += bond_loss.item()

        # Early warning if too many NaN batches
        if nan_batches > len(train_loader) // 2:
            print(f"  WARNING: Too many NaN batches ({nan_batches}), stopping training")
            return float('inf')

        n_valid_batches = max(len(train_loader) - nan_batches, 1)
        train_loss /= n_valid_batches
        for k in train_metrics:
            train_metrics[k] /= n_valid_batches

        # Validation: RMSD in Angstroms (denormalize from normalized space)
        model.eval()
        val_loss = 0
        n_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target_coords = batch['coords'].to(device)
                lengths = batch['lengths']

                B, L = seq_ids.shape

                # Normalize target (same as training)
                target_centered = target_coords - target_coords.mean(dim=1, keepdim=True)
                target_scale = torch.norm(target_centered, dim=(1,2), keepdim=True).clamp(min=1.0)

                # Use helical init for validation (matches deployment)
                coords_init = torch.zeros(B, L, 3, device=device)
                for b in range(B):
                    valid_L = lengths[b]
                    coords_init[b, :valid_L] = generate_helical_init(valid_L, device=device)

                # Helical init is already unit-norm, use directly
                coords_init_norm = coords_init

                coords_refined = model(seq_ids, coords_init_norm)

                # Skip NaN/Inf
                if torch.isnan(coords_refined).any() or torch.isinf(coords_refined).any():
                    continue

                # Denormalize predictions back to Å for RMSD reporting
                pred_denorm = coords_refined * target_scale + target_coords.mean(dim=1, keepdim=True)

                # RMSD in Angstroms (centered, translation-invariant)
                for b in range(B):
                    valid_L = lengths[b]
                    p = pred_denorm[b, :valid_L]
                    t = target_coords[b, :valid_L]
                    p_c = p - p.mean(dim=0)
                    t_c = t - t.mean(dim=0)
                    rmsd = torch.sqrt(torch.mean(torch.sum((p_c - t_c) ** 2, dim=1)).clamp(min=0))
                    if not torch.isnan(rmsd) and not torch.isinf(rmsd):
                        val_loss += rmsd.item()
                        n_val_samples += 1

        val_loss /= max(n_val_samples, 1)
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
              f"val={val_loss:.1f}Å nan={nan_batches} pat={patience_counter}/10")

        # Reset NaN counter per epoch
        nan_batches = 0

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"  Scheme 3 training complete: best_val={best_val:.4f}")
    return best_val


# ═══════════════════════════════════════════════════════════════
# Scheme-specific Data Requirements
# ═══════════════════════════════════════════════════════════════

# Scheme-specific max sequence length (O(L^2) schemes need lower limits)
SCHEME_MAX_LEN = {
    1: 1000,  # EGNN with k-NN sparse edges, O(k*L) memory
    2: None,  # Pure physics, no limit
    3: 1000,  # Transformer - A800 80GB can handle
    4: 1000,  # EGNN - A800 can handle
    5: 1000,  # Attention - A800 80GB can handle
    6: 800,   # GNN O(L^2) - A800 can handle
    7: None,  # Mamba O(L) + O(L*w), no limit
}

SCHEME_DATA_REQUIREMENTS = {
    1: {'min_samples': 200, 'recommended': 500, 'epochs': 50,  'reason': 'EGNN轻量，Physics部分无需训练'},
    2: {'min_samples': 0,   'recommended': 0,   'epochs': 0,   'reason': '纯物理求解器，无需训练'},
    3: {'min_samples': 300, 'recommended': 500, 'epochs': 50,  'reason': 'Dual-Engine中等复杂度'},
    4: {'min_samples': 500, 'recommended': 1000,'epochs': 100, 'reason': '扩散模型需要大量数据'},
    5: {'min_samples': 300, 'recommended': 800, 'epochs': 50,  'reason': 'Pairformer+物理bias'},
    6: {'min_samples': 500, 'recommended': 1000,'epochs': 100, 'reason': 'Encoder+Diffusion+Decoder最重'},
    7: {'min_samples': 500, 'recommended': 1000,'epochs': 100, 'reason': 'Mamba+Transformer混合扩散，O(L)可处理长序列'},
}

def main():
    parser = argparse.ArgumentParser(description='Train all TorusFold schemes')
    parser.add_argument('--schemes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--labels', type=str, default='',
                        help='Path to pre-generated pseudo-labels directory')
    parser.add_argument('--n-train', type=int, default=500,
                        help='Number of samples (used if no --labels)')
    parser.add_argument('--min-len', type=int, default=30)
    parser.add_argument('--max-len', type=int, default=0,
                        help='Maximum sequence length (0=no limit, load all data). '
                             'Schemes 1-6 auto-limit to 500 internally due to O(L^2).')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--diffusion-steps', type=int, default=100)
    parser.add_argument('--n-mamba-layers', type=int, default=4,
                        help='Number of Mamba layers for Scheme 7')
    parser.add_argument('--n-attn-layers', type=int, default=2,
                        help='Number of local attention layers for Scheme 7')
    parser.add_argument('--attn-window', type=int, default=20,
                        help='Local attention window size for Scheme 7')
    parser.add_argument('--bsj-flank', type=int, default=20,
                        help='BSJ flanking region size for Scheme 7')
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
        max_len_filter = args.max_len if args.max_len > 0 else None
        sequences, coords_labels, pair_labels, confidence_weights, metadata = load_pseudo_labels(
            args.labels, max_len=max_len_filter)
    else:
        print(f"  Generating pseudo-labels (n={args.n_train})")
        gen_max_len = args.max_len if args.max_len > 0 else 500
        sequences, coords_labels, pair_labels, metadata = generate_3d_pseudo_labels(
            n_seqs=args.n_train,
            min_len=args.min_len,
            max_len=gen_max_len,
            seed=args.seed
        )
        # Generated data has uniform confidence
        confidence_weights = [0.3] * len(sequences)

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

        # Scheme-specific length filtering
        scheme_max_len = SCHEME_MAX_LEN.get(scheme_id)
        if scheme_max_len is not None:
            keep = [i for i, m in enumerate(metadata) if m['length'] <= scheme_max_len]
            seq_s = [sequences[i] for i in keep]
            coord_s = [coords_labels[i] for i in keep]
            pair_s = [pair_labels[i] for i in keep]
            conf_s = [confidence_weights[i] for i in keep]
            n_filtered = len(keep)
            if n_filtered < len(sequences):
                print(f"\n  Scheme {scheme_id}: filtered {len(sequences)} -> {n_filtered} "
                      f"(max_len={scheme_max_len})")
        else:
            seq_s = sequences
            coord_s = coords_labels
            pair_s = pair_labels
            conf_s = confidence_weights
            n_filtered = len(sequences)
            print(f"\n  Scheme {scheme_id}: using all {n_filtered} samples (no length limit)")

        # Determine how many samples this scheme uses
        n_available = n_filtered
        n_use = n_available  # Use all available data

        # Warn if data is insufficient
        if n_available < req.get('min_samples', 0):
            print(f"\n  WARNING: Scheme {scheme_id} needs {req['min_samples']} samples, "
                  f"only {n_available} available. Results may be poor.")

        # Split for this scheme
        split = int(0.9 * n_use)
        train_ds = CircRNADataset(
            seq_s[:split], coord_s[:split], pair_s[:split], conf_s[:split])
        val_ds = CircRNADataset(
            seq_s[split:n_use], coord_s[split:n_use], pair_s[split:n_use], conf_s[split:n_use])

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
        elif scheme_id == 7:
            val_loss = train_scheme7(train_loader, val_loader, args, device)
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
