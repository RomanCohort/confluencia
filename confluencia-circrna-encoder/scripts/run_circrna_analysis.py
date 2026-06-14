#!/usr/bin/env python3
"""
run_circrna_analysis.py — Full circRNA analysis pipeline.

Pipeline:
  1. circRNA sequences → FASTA
  2. FASTA → RNA-FM / RiNALMo embeddings (frozen backbone)
  3. CircPairformer refinement (circular topology)
  4. Pathway classification (7-class)
  5. Immunogenicity prediction (binary)
  6. BSJ-crossing pair visualization
  7. ViennaRNA circular structure comparison
  8. RhoFold+ 3D structure (if available on AutoDL)

Usage (AutoDL GPU):
    # RNA-FM backbone (default, easiest to install)
    python scripts/run_circrna_analysis.py --backbone rna-fm --device cuda

    # RiNALMo backbone (larger, better)
    python scripts/run_circrna_analysis.py --backbone rinalmo --device cuda

    # RhoFold+ 3D structure (separate step)
    python scripts/run_circrna_analysis.py --backbone rna-fm --device cuda --rhofold

    # Quick test
    python scripts/run_circrna_analysis.py --backbone mock --device cpu --epochs 3
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Data path — auto-detect
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
# 1. circRNA → FASTA
# ═══════════════════════════════════════════════════════════════════

def write_fasta(sequences, ids, output_path):
    """Write circRNA sequences to FASTA file."""
    with open(output_path, 'w') as f:
        for seq_id, seq in zip(ids, sequences):
            rna_seq = seq.upper().replace('T', 'U')
            f.write(f">{seq_id}\n{rna_seq}\n")
    return output_path


# ═══════════════════════════════════════════════════════════════════
# 2. Backbone: RNA-FM / RiNALMo / Mock
# ═══════════════════════════════════════════════════════════════════

class RNAFMBackbone(nn.Module):
    """RNA-FM: RNA foundation model (ml4bio). d_model=640, 12 layers, 23M ncRNA."""

    def __init__(self, freeze=True):
        super().__init__()
        import fm
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        self.d_model = self.model.embed_dim  # 640
        self.repr_layer = self.model.num_layers
        self.batch_converter = self.alphabet.get_batch_converter()
        print(f"  RNA-FM loaded: d_model={self.d_model}")

    def encode(self, sequences, device):
        self.model = self.model.to(device)
        seqs_rna = [s.upper().replace('T', 'U') for s in sequences]
        _, _, tokens = self.batch_converter([(f"s{i}", s) for i, s in enumerate(seqs_rna)])
        tokens = tokens.to(device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[self.repr_layer])
            emb = results["representations"][self.repr_layer]
            mask = (tokens[:, 1:-1] != self.alphabet.padding_idx).float().unsqueeze(-1)
            pooled = (emb[:, 1:-1, :] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled


class RiNALMoBackbone(nn.Module):
    """RiNALMo: largest RNA LM (650M params, 36M ncRNA).
    Requires GPU (uses flash_attn Triton kernels).
    """

    def __init__(self, model_name="giga", freeze=True, weights_path=None):
        super().__init__()
        import sys
        sys.path.insert(0, "/root/autodl-tmp/RiNALMo")

        from rinalmo.config import model_config
        from rinalmo.model.model import RiNALMo
        from rinalmo.data.alphabet import Alphabet

        self.model_name = model_name
        config = model_config(model_name)
        self.model = RiNALMo(config)

        # 加载权重
        if weights_path is None:
            weights_path = "/root/autodl-tmp/RiNALMo/weights/rinalmo_giga_pretrained.pt"

        weights_path = Path(weights_path)
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"  RiNALMo weights loaded from {weights_path}")
        else:
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        # 用官方 Alphabet
        self.alphabet = Alphabet(**config['alphabet'])
        self.d_model = config.get('d_model', 1280)
        print(f"  RiNALMo loaded: {model_name}, d_model={self.d_model}")
        print(f"  Alphabet tokens: {list(self.alphabet.tkn_to_idx.keys())}")

    def encode(self, sequences, device):
        # RiNALMo 必须在 GPU 上运行（flash_attn Triton kernel）
        self.model = self.model.to(device)
        self.model.eval()

        # RiNALMo 用 T 不用 U
        seqs_rna = [s.upper().replace('U', 'T') for s in sequences]
        token_lists = self.alphabet.batch_tokenize(seqs_rna)

        # Pad to same length (batch_tokenize returns variable-length lists)
        max_len = max(len(t) for t in token_lists)
        pad_idx = self.alphabet.tkn_to_idx.get('<pad>', 1)
        padded = [t + [pad_idx] * (max_len - len(t)) for t in token_lists]
        tokens = torch.tensor(padded, dtype=torch.int64, device=device)

        # RiNALMo flash_attn 要求 bfloat16
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model(tokens)

        repr = out.get("representation", out.get("embeddings"))

        # 检查 NaN
        if torch.isnan(repr).any():
            print(f"  WARNING: RiNALMo output has NaN, falling back to zeros")
            return torch.zeros(tokens.size(0), self.d_model, device=device)

        # Mean pool: exclude CLS, EOS, PAD tokens
        eos_idx = self.alphabet.tkn_to_idx.get('<eos>', 2)
        cls_idx = self.alphabet.tkn_to_idx.get('<cls>', 0)
        mask = (tokens != pad_idx) & (tokens != eos_idx) & (tokens != cls_idx)
        mask = mask.float().unsqueeze(-1)

        pooled = (repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled


class MockBackbone(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(5, d_model)

    def encode(self, sequences, device):
        B = len(sequences)
        emb = torch.zeros(B, self.d_model, device=device)
        for i, seq in enumerate(sequences):
            counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
            for c in seq.upper():
                if c in counts:
                    counts[c] += 1
            total = len(seq)
            feat = torch.tensor([counts['A']/total, counts['C']/total,
                                  counts['G']/total, counts['U']/total, total/500], device=device)
            emb[i] = self.embed.weight.mean(dim=0) + feat.mean() * 0.1
        return emb


# ═══════════════════════════════════════════════════════════════════
# 3. CircPairformer + Classifier (same as run_pathway_classification.py)
# ═══════════════════════════════════════════════════════════════════

class CircPairformerBlock(nn.Module):
    def __init__(self, c_z=64, n_heads=2, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(c_z)
        self.attn = nn.MultiheadAttention(c_z, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(c_z, c_z * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(c_z * 2, c_z), nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(c_z)

    def forward(self, z):
        B, L, _, c_z = z.shape
        z_flat = z.reshape(B * L, L, c_z)
        z_norm = self.ln(z_flat)
        z_attn, _ = self.attn(z_norm, z_norm, z_norm)
        z_flat = z_flat + z_attn
        z_flat = z_flat + self.ff(self.ln2(z_flat))
        return z_flat.reshape(B, L, L, c_z)


class CircPairformerStack(nn.Module):
    def __init__(self, n_blocks=2, c_z=64, n_heads=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            CircPairformerBlock(c_z, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(c_z)

    def forward(self, z):
        for block in self.blocks:
            z = block(z)
        return self.ln_out(z)


class CircRNAClassifier(nn.Module):
    def __init__(self, d_model=640, c_z=64, n_pairformer_blocks=2,
                 n_heads_tri=2, n_pathways=7, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.c_z = c_z

        self.left_proj = nn.Linear(d_model, c_z)
        self.right_proj = nn.Linear(d_model, c_z)
        self.dist_embed = nn.Embedding(256, c_z)

        self.pairformer = CircPairformerStack(n_pairformer_blocks, c_z, n_heads_tri, dropout)

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

    def _circ_dist_matrix(self, L, device):
        pos = torch.arange(L, device=device)
        diff = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
        return torch.min(diff, L - diff)

    def forward(self, seq_emb, device='cpu'):
        B = seq_emb.size(0)
        L = 64
        seq_repr = seq_emb.unsqueeze(1).expand(-1, L, -1)

        left = self.left_proj(seq_repr)
        right = self.right_proj(seq_repr)
        pair = left.unsqueeze(2) + right.unsqueeze(1)
        circ_dist = self._circ_dist_matrix(L, device).clamp(max=255).long()
        pair = pair + self.dist_embed(circ_dist).unsqueeze(0)

        pair_repr = self.pairformer(pair)
        pair_probs = self.pair_head(pair_repr).squeeze(-1)
        pair_probs = 0.5 * (pair_probs + pair_probs.transpose(-1, -2))

        bsj_mask = (self._circ_dist_matrix(L, device) >= L / 2).float()
        bsj_strength = (pair_probs * bsj_mask.unsqueeze(0)).sum(dim=(1, 2)) / bsj_mask.sum().clamp(min=1)

        struct_feat = pair_repr.mean(dim=(1, 2))
        class_input = torch.cat([seq_emb, struct_feat, bsj_strength.unsqueeze(-1)], dim=-1)

        return {
            'pathway_logits': self.pathway_head(class_input),
            'immunogenicity_logits': self.immunogenicity_head(class_input).squeeze(-1),
            'pair_probs': pair_probs,
            'bsj_strength': bsj_strength,
        }


# ═══════════════════════════════════════════════════════════════════
# 4. Dataset + Training
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
            'circrna_id': row.get('circrna_id', f'sample_{idx}'),
            'pathway': torch.tensor(PATHWAY_MAP.get(row.get('pathway', 'unknown'), 0), dtype=torch.long),
            'immunogenicity': torch.tensor(float(row.get('immunogenicity', 0)), dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        'sequences': [b['sequence'] for b in batch],
        'circrna_ids': [b['circrna_id'] for b in batch],
        'pathway': torch.stack([b['pathway'] for b in batch]),
        'immunogenicity': torch.stack([b['immunogenicity'] for b in batch]),
    }


# ═══════════════════════════════════════════════════════════════════
# 5. RhoFold+ 3D structure (if available)
# ═══════════════════════════════════════════════════════════════════

def run_rhofold(fasta_path, output_dir):
    """Run RhoFold+ on FASTA file for 3D structure prediction."""
    try:
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'rhofold', 'predict',
             '--input_fasta', str(fasta_path),
             '--output_dir', str(output_dir)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"  RhoFold+ succeeded: {output_dir}")
            return True
        else:
            print(f"  RhoFold+ failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  RhoFold+ not available: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=str(DATA_PATH))
    parser.add_argument('--backbone', type=str, default='rna-fm',
                        choices=['rna-fm', 'rinalmo', 'esm2', 'mock'])
    parser.add_argument('--rinalmo-model', type=str, default='giga',
                        choices=['giga', 'mega', 'micro'],
                        help='RiNALMo model size: giga (650M), mega (150M), micro (35M)')
    parser.add_argument('--rinalmo-weights', type=str,
                        default='/root/autodl-tmp/RiNALMo/weights/rinalmo_giga_pretrained.pt',
                        help='Path to RiNALMo pretrained weights')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-seq-len', type=int, default=200)
    parser.add_argument('--c-z', type=int, default=64)
    parser.add_argument('--n-pf-blocks', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rhofold', action='store_true', help='Also run RhoFold+ 3D prediction')
    parser.add_argument('--n-rhofold', type=int, default=5, help='Number of sequences for RhoFold+')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("TorusFold: circRNA Analysis Pipeline")
    print("=" * 70)
    print(f"  Backbone: {args.backbone}")
    print(f"  Device:   {args.device}")
    print(f"  RhoFold+: {args.rhofold}")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ─── Data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(args.data)
    df = df[df['sequence'].str.len().between(20, args.max_seq_len)].reset_index(drop=True)
    print(f"  {len(df)} samples (length 20-{args.max_seq_len})")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.seed,
                                          stratify=df['pathway'])
    train_loader = DataLoader(CircRNADataset(train_df), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CircRNADataset(test_df), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # ─── Write FASTA (for RhoFold+ etc.) ─────────────────────────
    print("\n[2/6] Writing FASTA...")
    fasta_dir = PROJECT_ROOT / 'output' / 'fasta'
    fasta_dir.mkdir(parents=True, exist_ok=True)

    # All sequences
    fasta_path = fasta_dir / 'circrna_all.fasta'
    write_fasta(df['sequence'].tolist(), df.get('circrna_id', [f'circ_{i}' for i in range(len(df))]).tolist(), fasta_path)
    print(f"  FASTA: {fasta_path} ({len(df)} sequences)")

    # ─── Backbone ─────────────────────────────────────────────────
    print(f"\n[3/6] Loading backbone ({args.backbone})...")
    if args.backbone == 'rna-fm':
        backbone = RNAFMBackbone(freeze=True)
    elif args.backbone == 'rinalmo':
        backbone = RiNALMoBackbone(model_name=args.rinalmo_model, freeze=True,
                                    weights_path=args.rinalmo_weights)
    elif args.backbone == 'esm2':
        import esm
        backbone = type('ESM2Backbone', (nn.Module,), {
            '__init__': lambda self: None,  # simplified
        })()
        model_esm, alphabet_esm = esm.pretrained.esm2_t30_150M_UR50D()
        for p in model_esm.parameters():
            p.requires_grad = False
        backbone.model = model_esm
        backbone.alphabet = alphabet_esm
        backbone.d_model = model_esm.embed_dim
        backbone.repr_layer = model_esm.num_layers

        def esm_encode(self, sequences, device):
            self.model = self.model.to(device)
            seqs_t = [s.upper().replace('U', 'T') for s in sequences]
            _, _, tokens = self.alphabet.get_batch_converter()([(f"s{i}", s) for i, s in enumerate(seqs_t)])
            tokens = tokens.to(device)
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.repr_layer])
                emb = results["representations"][self.repr_layer]
                mask = (tokens[:, 1:-1] != self.alphabet.padding_idx).float().unsqueeze(-1)
                return (emb[:, 1:-1, :] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        backbone.encode = esm_encode.__get__(backbone)
        print(f"  ESM2 loaded: d_model={backbone.d_model}")
    else:
        backbone = MockBackbone(d_model=128)

    d_model = backbone.d_model

    # ─── Model ────────────────────────────────────────────────────
    print("\n[4/6] Creating CircPairformer model...")
    model = CircRNAClassifier(
        d_model=d_model, c_z=args.c_z,
        n_pairformer_blocks=args.n_pf_blocks,
        n_pathways=N_PATHWAYS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ─── Train ────────────────────────────────────────────────────
    print("\n[5/6] Training...")
    best_f1 = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        total_loss = 0
        for batch in train_loader:
            pw_target = batch['pathway'].to(device)
            imm_target = batch['immunogenicity'].to(device)

            seq_emb = backbone.encode(batch['sequences'], device)
            out = model(seq_emb, device)

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
                seq_emb = backbone.encode(batch['sequences'], device)
                out = model(seq_emb, device)
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
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e} "
              f"loss={total_loss/len(train_loader):.4f} | "
              f"PW={pw_acc:.3f} F1={pw_f1:.3f} | IMM={imm_auc:.3f}")

        history.append({'epoch': epoch+1, 'pw_acc': pw_acc, 'pw_f1': pw_f1, 'imm_auc': imm_auc})

        if pw_f1 > best_f1:
            best_f1 = pw_f1
            save_path = PROJECT_ROOT / 'models' / 'circrna_best.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'pw_f1': pw_f1, 'pw_acc': pw_acc,
                         'backbone': args.backbone, 'd_model': d_model}, save_path)

    # Final report
    print(f"\n  Best Pathway F1: {best_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(all_pw_true, all_pw_pred,
                                target_names=PATHWAY_NAMES, zero_division=0))

    # ─── RhoFold+ 3D Structure ────────────────────────────────────
    if args.rhofold:
        print(f"\n[6/6] RhoFold+ 3D structure prediction...")
        # Select a few sequences for RhoFold+
        sample_df = test_df.head(args.n_rhofold)
        sample_fasta = fasta_dir / 'circrna_rhofold.fasta'
        write_fasta(sample_df['sequence'].tolist(),
                     sample_df.get('circrna_id', [f'rhofold_{i}' for i in range(len(sample_df))]).tolist(),
                     sample_fasta)
        rhofold_out = PROJECT_ROOT / 'output' / 'rhofold'
        rhofold_out.mkdir(parents=True, exist_ok=True)
        run_rhofold(sample_fasta, rhofold_out)

    # ─── ViennaRNA comparison ─────────────────────────────────────
    print("\n[ViennaRNA] Circular vs Linear structure:")
    try:
        import RNA
        for i in range(min(3, len(test_df))):
            seq = test_df.iloc[i]['sequence'].upper().replace('T', 'U')[:150]
            md_l = RNA.md()
            fc_l = RNA.fold_compound(seq, md_l)
            _, mfe_l = fc_l.mfe()
            md_c = RNA.md()
            md_c.circ = True
            fc_c = RNA.fold_compound(seq, md_c)
            _, mfe_c = fc_c.mfe()
            print(f"  Seq {i} (len={len(seq)}): linear={mfe_l:.1f}, circular={mfe_c:.1f}, "
                  f"diff={mfe_c-mfe_l:+.1f} kcal/mol")
    except ImportError:
        print("  ViennaRNA not available")

    # ─── Save results ─────────────────────────────────────────────
    results = {
        'timestamp': datetime.now().isoformat(),
        'backbone': args.backbone,
        'd_model': d_model,
        'n_params': n_params,
        'pathway_accuracy': float(pw_acc),
        'pathway_f1_macro': float(pw_f1),
        'immunogenicity_auc': float(imm_auc),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'history': history,
    }
    results_path = PROJECT_ROOT / 'models' / 'circrna_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
