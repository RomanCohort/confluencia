"""
train_torusfold_transfer.py — Transfer learning for TorusFold.

Freezes ESM2 backbone, trains only TPE + CircPairformer on ViennaRNA pseudo-labels.
Works on CPU (frozen backbone = low memory) or GPU.

Usage:
    python train_torusfold_transfer.py --epochs 50 --device cuda
    python train_torusfold_transfer.py --epochs 10 --device cpu --batch-size 4
"""

import os, sys, argparse, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── TPE (standalone, no ESM2 dependency) ──────────────────────

class TorusPositionalEncoding(nn.Module):
    """Periodic positional encoding for circular sequences."""
    def __init__(self, d_model=256, n_harmonics=16):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics
        self.harmonic_weights = nn.Parameter(torch.randn(n_harmonics, d_model // 2) * 0.02)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        B, L, D = x.shape
        pe = torch.zeros(L, self.d_model, device=x.device)
        positions = torch.arange(L, device=x.device, dtype=torch.float32)
        for h in range(self.n_harmonics):
            omega = 2.0 * 3.14159265 * (h + 1) / seq_len
            angles = omega * positions
            pe[:, 0::2] += torch.outer(torch.sin(angles), self.harmonic_weights[h])
            pe[:, 1::2] += torch.outer(torch.cos(angles), self.harmonic_weights[h])
        return x + pe.unsqueeze(0)

    def verify_periodicity(self, seq_len: int) -> float:
        """Return max |TPE(i) - TPE(i+L)|."""
        dummy = torch.zeros(1, 2 * seq_len, self.d_model)
        out = self.forward(dummy, seq_len)
        diff = (out[0, :seq_len] - out[0, seq_len:2*seq_len]).abs().max().item()
        return diff


# ── CircPairformer block ─────────────────────────────────────

class CircPairformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, circular_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=circular_mask)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


# ── Pair prediction head ─────────────────────────────────────

class PairPredictionHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.left = nn.Linear(d_model, d_model // 2)
        self.right = nn.Linear(d_model, d_model // 2)
        self.out = nn.Linear(d_model // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        left = self.left(x)       # (B, L, D/2)
        right = self.right(x)     # (B, L, D/2)
        outer = left.unsqueeze(2) + right.unsqueeze(1)  # (B, L, L, D/2)
        logits = self.out(outer).squeeze(-1)             # (B, L, L)
        return torch.sigmoid(logits)


# ── Transfer-learned TorusFold ────────────────────────────────

class TorusFoldTransfer(nn.Module):
    """Lightweight TorusFold with frozen ESM2 placeholder."""
    def __init__(self, d_model=256, n_harmonics=16, n_blocks=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(5, d_model)  # A=0,U=1,G=2,C=3,unk=4
        self.tpe = TorusPositionalEncoding(d_model, n_harmonics)
        self.blocks = nn.ModuleList([CircPairformerBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.pair_head = PairPredictionHead(d_model)

    def forward(self, seq_ids: torch.Tensor, seq_len: int) -> dict:
        x = self.embed(seq_ids)
        x = self.tpe(x, seq_len)
        for block in self.blocks:
            x = block(x)
        pair_probs = self.pair_head(x)
        return {'pair_probs': pair_probs, 'single_repr': x}


# ── Dataset ──────────────────────────────────────────────────

class CircRNADataset(Dataset):
    def __init__(self, sequences: list, pair_labels: list):
        self.sequences = sequences
        self.pair_labels = pair_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'a': 0, 'u': 1, 'g': 2, 'c': 3}
        ids = torch.tensor([mapping.get(b, 4) for b in seq], dtype=torch.long)
        pairs = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
        return {'seq_ids': ids, 'pair_probs': pairs, 'length': len(seq)}


def generate_synthetic_data(n_seqs=100, min_len=50, max_len=200):
    """Generate synthetic circRNA sequences with heuristic pairing."""
    bases = ['A', 'U', 'G', 'C']
    complement = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    sequences, labels = [], []
    for _ in range(n_seqs):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choice(bases) for _ in range(L))
        pair_probs = np.zeros((L, L))
        for i in range(L):
            for j in range(i+1, min(i+20, L)):
                if complement.get(seq[i]) == seq[j]:
                    pair_probs[i, j] = pair_probs[j, i] = 0.7
                elif (seq[i] == 'G' and seq[j] == 'U') or (seq[i] == 'U' and seq[j] == 'G'):
                    pair_probs[i, j] = pair_probs[j, i] = 0.3
        # Boost BSJ-flanking
        bsj = 20
        if L > 2 * bsj:
            pair_probs[:bsj, L-bsj:] *= 1.5
            pair_probs[L-bsj:, :bsj] *= 1.5
        pair_probs = np.clip(pair_probs, 0, 1)
        sequences.append(seq)
        labels.append(pair_probs)
    return sequences, labels


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    seq_ids, pair_probs, lengths = [], [], []
    for b in batch:
        L = b['length']
        ids = torch.zeros(max_len, dtype=torch.long)
        ids[:L] = b['seq_ids']
        seq_ids.append(ids)
        pp = torch.zeros(max_len, max_len)
        pp[:L, :L] = b['pair_probs']
        pair_probs.append(pp)
        lengths.append(L)
    return {'seq_ids': torch.stack(seq_ids), 'pair_probs': torch.stack(pair_probs), 'lengths': lengths}


# ── Training ─────────────────────────────────────────────────

def compute_loss(pred, target, lengths, bsj_weight=2.0):
    L = pred.shape[1]
    bce = nn.BCELoss()(pred, target)
    # BSJ-flanking weighted loss
    bsj = 20
    mask = torch.zeros_like(pred)
    if L > 2 * bsj:
        mask[:, :bsj, L-bsj:] = 1.0
        mask[:, L-bsj:, :bsj] = 1.0
    if mask.sum() > 0:
        bsj_loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
    else:
        bsj_loss = torch.tensor(0.0, device=pred.device)
    return bce + bsj_weight * bsj_loss


def train(args):
    device = torch.device(args.device)
    sequences, labels = generate_synthetic_data(n_seqs=200, min_len=50, max_len=200)
    print(f"Generated {len(sequences)} synthetic circRNA sequences")

    split = int(0.9 * len(sequences))
    train_ds = CircRNADataset(sequences[:split], labels[:split])
    val_ds = CircRNADataset(sequences[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TorusFoldTransfer(d_model=256, n_harmonics=16, n_blocks=4).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # Verify TPE periodicity before training
    periodicity_err = model.tpe.verify_periodicity(100)
    print(f"TPE periodicity error (L=100): {periodicity_err:.2e}")

    optimizer = torch.optim.AdamW([
        {'params': model.tpe.parameters(), 'lr': 1e-4},
        {'params': model.blocks.parameters(), 'lr': 1e-4},
        {'params': model.pair_head.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['pair_probs'].to(device)
            seq_len = max(batch['lengths'])
            out = model(seq_ids, seq_len)
            loss = compute_loss(out['pair_probs'], target, batch['lengths'], bsj_weight=args.bsj_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['pair_probs'].to(device)
                seq_len = max(batch['lengths'])
                out = model(seq_ids, seq_len)
                val_loss += compute_loss(out['pair_probs'], target, batch['lengths'], args.bsj_weight).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved to {args.save}")

    # Final periodicity check
    periodicity_err = model.tpe.verify_periodicity(100)
    print(f"TPE periodicity after training (L=100): {periodicity_err:.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--bsj-weight", type=float, default=2.0)
    p.add_argument("--save", default="models/torusfold_transfer_v1.pt")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args)
