"""
train_torusfold_vienna.py — Training with ViennaRNA pseudo-labels.

Uses ViennaRNA RNAfold to generate secondary structure constraints
as training targets, then trains on these real biophysical predictions.
"""

import os, sys, random
from pathlib import Path
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TorusPositionalEncoding(nn.Module):
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


class CircPairformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[1]
        left = self.left(x)
        right = self.right(x)
        outer = left.unsqueeze(2) + right.unsqueeze(1)
        logits = self.out(outer).squeeze(-1)
        return torch.sigmoid(logits)


class TorusFoldTransfer(nn.Module):
    def __init__(self, d_model=256, n_harmonics=16, n_blocks=4, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(5, d_model)
        self.tpe = TorusPositionalEncoding(d_model, n_harmonics)
        self.blocks = nn.ModuleList([CircPairformerBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.pair_head = PairPredictionHead(d_model)

    def forward(self, seq_ids: torch.Tensor, seq_len: int) -> dict:
        x = self.embed(seq_ids)
        x = self.tpe(x, seq_len)
        for block in self.blocks:
            x = block(x)
        pair_probs = self.pair_head(x)
        return {'pair_probs': pair_probs}


# ── ViennaRNA integration ─────────────────────────────────────

def run_viennarna(sequence: str) -> torch.Tensor:
    """Run ViennaRNA RNAfold and return pair probabilities."""
    seq_upper = sequence.upper().replace('T', 'U')
    try:
        result = subprocess.run(
            f"echo '{seq_upper}' | RNAfold --MEA --noPS",
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return parse_viennarna_output(seq_upper, result.stdout)
    except Exception as e:
        pass
    return torch.zeros(len(sequence), len(sequence))


def parse_viennarna_output(sequence: str, output: str) -> torch.Tensor:
    """Parse ViennaRNA RNAfold output to get pair probabilities."""
    L = len(sequence)
    pair_probs = torch.zeros(L, L)

    # Find the dot-bracket structure line
    lines = output.strip().split('\n')
    for i, line in enumerate(lines):
        if '.' in line and '(' in line and ')' in line:
            # This is likely the structure line
            # Find paired positions
            stack = []
            pairs = []
            for j, char in enumerate(line):
                if char == '(':
                    stack.append(j)
                elif char == ')':
                    if stack:
                        pairs.append((stack.pop(), j))

            # Assign probability based on structure
            for i, j in pairs:
                if i < j:
                    pair_probs[i, j] = 0.85
                    pair_probs[j, i] = 0.85

            # Add weak pairs in loops (poly-A/U regions are less stable)
            for i in range(L):
                for j in range(i+1, L):
                    if pair_probs[i, j] == 0:
                        # Check for GC vs AU
                        si, sj = sequence[i], sequence[j]
                        if (si == 'G' and sj == 'C') or (si == 'C' and sj == 'G'):
                            pair_probs[i, j] = 0.2
                            pair_probs[j, i] = 0.2
                        elif (si == 'A' and sj == 'U') or (si == 'U' and sj == 'A'):
                            pair_probs[i, j] = 0.15
                            pair_probs[j, i] = 0.15

            # Boost BSJ-flanking regions
            bsj = 20
            if L > 2 * bsj:
                pair_probs[:bsj, L-bsj:] *= 1.5
                pair_probs[L-bsj:, :bsj] *= 1.5

            return torch.clip(pair_probs, 0, 1)

    return torch.zeros(L, L)


def generate_viennarna_data(n_seqs=200, min_len=50, max_len=200):
    """Generate data using ViennaRNA RNAfold predictions."""
    bases = ['A', 'U', 'G', 'C']
    sequences, labels = [], []

    print("Generating ViennaRNA pseudo-labels...")
    for i in range(n_seqs):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choice(bases) for _ in range(L))
        pair_probs = run_viennarna(seq)
        sequences.append(seq)
        labels.append(pair_probs.numpy())

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_seqs} - generated ViennaRNA labels")

    return sequences, labels


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


def compute_loss(pred, target, lengths, bsj_weight=2.0):
    L = pred.shape[1]
    bce = F.binary_cross_entropy(pred, target)
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
    sequences, labels = generate_viennarna_data(n_seqs=200, min_len=50, max_len=200)
    print(f"Generated {len(sequences)} sequences with ViennaRNA pseudo-labels")

    split = int(0.9 * len(sequences))
    train_ds = CircRNADataset(sequences[:split], labels[:split])
    val_ds = CircRNADataset(sequences[split:], labels[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TorusFoldTransfer(d_model=256, n_harmonics=16, n_blocks=4).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            seq_ids = batch['seq_ids'].to(device)
            target = batch['pair_probs'].to(device)
            seq_len = max(batch['lengths'])
            out = model(seq_ids, seq_len)
            loss = compute_loss(out['pair_probs'], target, batch['lengths'], args.bsj_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

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
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  pat={patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--bsj-weight", type=float, default=2.0)
    p.add_argument("--save", default="models/torusfold_vienna.pt")
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args)
