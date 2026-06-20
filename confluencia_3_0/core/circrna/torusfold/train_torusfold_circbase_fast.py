"""
train_torusfold_circbase_fast.py — Train on pre-processed circBase labels.

Loads .npz labels from preprocess_circbase_labels.py, so no ViennaRNA needed at train time.

Usage:
    # After running preprocess_circbase_labels.py:
    python train_torusfold_circbase_fast.py --labels data/circrna/circbase_labels_2k.npz --epochs 100
    python train_torusfold_circbase_fast.py --labels data/circrna/circbase_labels_5k.npz --epochs 100 --device cuda
"""

import os, sys, random, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Model ─────────────────────────────────────────────────────

class TorusPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, n_harmonics=16):
        super().__init__()
        self.d_model = d_model
        self.n_harmonics = n_harmonics
        self.harmonic_weights = nn.Parameter(torch.randn(n_harmonics, d_model // 2) * 0.02)

    def forward(self, x, seq_len):
        B, L, D = x.shape
        pe = torch.zeros(L, self.d_model, device=x.device)
        positions = torch.arange(L, device=x.device, dtype=torch.float32)
        for h in range(self.n_harmonics):
            omega = 2.0 * 3.14159265 * (h + 1) / seq_len
            angles = omega * positions
            pe[:, 0::2] += torch.outer(torch.sin(angles), self.harmonic_weights[h])
            pe[:, 1::2] += torch.outer(torch.cos(angles), self.harmonic_weights[h])
        return x + pe.unsqueeze(0)


class CircPairformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


class PairPredictionHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.left = nn.Linear(d_model, d_model // 2)
        self.right = nn.Linear(d_model, d_model // 2)
        self.out = nn.Linear(d_model // 2, 1)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        outer = left.unsqueeze(2) + right.unsqueeze(1)
        return torch.sigmoid(self.out(outer).squeeze(-1))


class TorusFoldTransfer(nn.Module):
    def __init__(self, d_model=256, n_harmonics=16, n_blocks=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(5, d_model)
        self.tpe = TorusPositionalEncoding(d_model, n_harmonics)
        self.blocks = nn.ModuleList([CircPairformerBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.pair_head = PairPredictionHead(d_model)

    def forward(self, seq_ids, seq_len):
        x = self.embed(seq_ids)
        x = self.tpe(x, seq_len)
        for block in self.blocks:
            x = block(x)
        return self.pair_head(x)


# ── Data ──────────────────────────────────────────────────────

class CircBaseDataset(Dataset):
    def __init__(self, sequences, pair_labels, max_len=500):
        self.sequences = sequences
        self.pair_labels = pair_labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        L = min(len(seq), self.max_len)
        seq = seq[:L]
        mapping = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
        ids = torch.zeros(self.max_len, dtype=torch.long)
        for i, b in enumerate(seq):
            ids[i] = mapping.get(b, 4)
        pp = torch.zeros(self.max_len, self.max_len)
        label = self.pair_labels[idx]
        if label.shape[0] >= L and label.shape[1] >= L:
            pp[:L, :L] = torch.tensor(label[:L, :L], dtype=torch.float32)
        return {'seq_ids': ids, 'pair_probs': pp, 'length': L}


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    seq_ids, pair_probs, lengths = [], [], []
    for b in batch:
        L = b['length']
        seq_ids.append(b['seq_ids'][:max_len])
        pair_probs.append(b['pair_probs'][:max_len, :max_len])
        lengths.append(L)
    return {
        'seq_ids': torch.stack(seq_ids),
        'pair_probs': torch.stack(pair_probs),
        'lengths': lengths,
    }


def compute_loss(pred, target, lengths, bsj_weight=2.0):
    L = pred.shape[1]
    bce = F.binary_cross_entropy(pred, target)
    bsj = 20
    mask = torch.zeros_like(pred)
    if L > 2 * bsj:
        mask[:, :bsj, L-bsj:] = 1.0
        mask[:, L-bsj:, :bsj] = 1.0
    bsj_loss = ((pred - target) ** 2 * mask).sum() / max(mask.sum(), 1)
    return bce + bsj_weight * bsj_loss


def train(args):
    device = torch.device(args.device)

    # Load pre-processed data
    print(f"Loading pre-processed labels from {args.labels}...")
    data = np.load(args.labels, allow_pickle=True)
    sequences = list(data['sequences'])
    n_seqs = int(data['n_seqs'])
    pair_labels = [data[f'label_{i}'] for i in range(n_seqs)]
    print(f"Loaded {n_seqs} sequences with pre-computed labels")

    lengths = [len(s) for s in sequences]
    print(f"  Length: median={np.median(lengths):.0f}, mean={np.mean(lengths):.0f}")

    # Split
    n = len(sequences)
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx = indices[:int(0.8*n)]
    val_idx = indices[int(0.8*n):int(0.9*n)]
    test_idx = indices[int(0.9*n):]

    train_ds = CircBaseDataset([sequences[i] for i in train_idx], [pair_labels[i] for i in train_idx])
    val_ds = CircBaseDataset([sequences[i] for i in val_idx], [pair_labels[i] for i in val_idx])
    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_idx)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    model = TorusFoldTransfer(d_model=256, n_harmonics=16, n_blocks=4).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total:,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val = float('inf')
    patience_counter = 0
    patience = 15
    log_lines = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['pair_probs'].to(device)
            seq_len = max(batch['lengths'])
            pred = model(seq_ids, seq_len)
            loss = compute_loss(pred, target, batch['lengths'], args.bsj_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_ids = batch['seq_ids'].to(device)
                target = batch['pair_probs'].to(device)
                seq_len = max(batch['lengths'])
                pred = model(seq_ids, seq_len)
                val_loss += compute_loss(pred, target, batch['lengths'], args.bsj_weight).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), args.save.replace('.pt', '_best.pt'))
        else:
            patience_counter += 1

        line = f"Epoch {epoch+1}/{args.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  best={best_val:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  pat={patience_counter}/{patience}"
        print(line)
        log_lines.append(line)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Final: {args.save}")
    print(f"Best: {args.save.replace('.pt', '_best.pt')}")

    log_path = args.save.replace('.pt', '_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines))
    print(f"Log: {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Path to .npz from preprocess_circbase_labels.py")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bsj-weight", type=float, default=2.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--save", default="models/torusfold_circbase.pt")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args)
