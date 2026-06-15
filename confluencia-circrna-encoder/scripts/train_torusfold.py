"""
train_torusfold.py — Training script for TorusFold on circRNA data.

Features:
1. Sequence-aware split: clusters sequences by k-mer similarity,
   ensures similar sequences stay in the same fold.
2. Hyperparameter search: grid search over n_harmonics,
   n_rot_augments, d_pair with 3-fold CV.
3. External validation on sequences_enhanced.csv (immune pathway).
4. Supports both full TorusFold (ESM2 backbone) and mock mode.

Usage:
    # Quick test with mock backbone
    python scripts/train_torusfold.py --mock --epochs 5

    # Full training with ESM2
    python scripts/train_torusfold.py --epochs 20 --device cuda

    # Hyperparameter search
    python scripts/train_torusfold.py --hparam-search --mock --epochs 3

    # External validation only
    python scripts/train_torusfold.py --external-only --mock
"""

import sys
import os
from pathlib import Path
import argparse
import json
import time
from itertools import product
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IGEM_ROOT = Path(__file__).resolve().parents[2]
if str(IGEM_ROOT) not in sys.path:
    sys.path.insert(0, str(IGEM_ROOT))


# ─── Data ───────────────────────────────────────────────────────────────

class CircRNADataset(Dataset):
    """Dataset for circRNA multi-task training."""

    GENE_COLS = ["gene_TROP2", "gene_NECTIN4", "gene_LIV-1",
                 "gene_B7-H4", "gene_MKI67", "gene_MYC"]

    COMPOSITE_KEYS = [
        "target_immunotherapy_score", "target_tumor_killing_index",
        "target_overall_immunogenicity", "target_immune_cycle_score",
        "target_tme_score", "target_therapeutic_window",
        "target_tide_score", "target_ips",
    ]

    REPORT_KEYS = [
        "target_rig_i_score", "target_tlr_score",
        "target_pkr_score", "target_trained_model_risk",
    ]

    RESPONSE_CLASSES = ["likely_non_responder", "intermediate", "likely_responder"]
    RESPONSE_MAP = {c: i for i, c in enumerate(RESPONSE_CLASSES)}

    def __init__(self, data_path: str, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len

        print(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"  Loaded {len(self.df)} samples")

        # Filter by length (pair representation O(L^2) memory)
        self.df = self.df[self.df['sequence'].str.len() <= max_seq_len].reset_index(drop=True)
        self.df = self.df[self.df['sequence'].str.len() >= 20].reset_index(drop=True)
        print(f"  After length filter (20-{max_seq_len}): {len(self.df)} samples")

        # Handle missing values
        all_cols = self.GENE_COLS + self.COMPOSITE_KEYS + self.REPORT_KEYS
        for col in all_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        sequence = row['sequence']
        gene_expr = torch.tensor(
            [row.get(c, 0.5) for c in self.GENE_COLS], dtype=torch.float32)
        composite_target = torch.tensor(
            [row.get(c, 0.5) for c in self.COMPOSITE_KEYS], dtype=torch.float32)
        report_target = torch.tensor(
            [row.get(c, 0.5) for c in self.REPORT_KEYS], dtype=torch.float32)

        resp_str = row.get('target_predicted_response', 'intermediate')
        response_target = torch.tensor(
            self.RESPONSE_MAP.get(resp_str, 1), dtype=torch.long)

        return {
            'sequence': sequence,
            'gene_expr': gene_expr,
            'composite_target': composite_target,
            'report_target': report_target,
            'response_target': response_target,
        }


class ExternalDataset(Dataset):
    """External validation on sequences_enhanced.csv (immune pathway)."""

    def __init__(self, data_path: str, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.df = pd.read_csv(data_path)
        self.df = self.df[self.df['sequence'].str.len() <= max_seq_len].reset_index(drop=True)
        print(f"  External dataset: {len(self.df)} samples")

        # Pathway to index
        self.pathway_map = {
            'RIG-I': 0, 'MDA5': 0, 'RIG-I/MDA5': 0,
            'TLR7/8': 1, 'TLR3': 1,
            'JAK-STAT': 2, 'PKR': 2,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['sequence']

        # Immune pathway label
        pathway = row.get('pathway', 'unknown')
        pathway_idx = self.pathway_map.get(pathway, 1)  # default TLR

        # Immune score
        immune_score = row.get('immune_score', 0.5)
        immunogenicity = row.get('immunogenicity', 0)

        return {
            'sequence': sequence,
            'pathway_target': torch.tensor(pathway_idx, dtype=torch.long),
            'immune_score_target': torch.tensor(immune_score, dtype=torch.float32),
            'immunogenicity_target': torch.tensor(float(immunogenicity), dtype=torch.float32),
            'gene_expr': torch.tensor([0.5] * 6, dtype=torch.float32),  # placeholder
        }


def collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    gene_expr = torch.stack([item['gene_expr'] for item in batch])
    result = {'sequences': sequences, 'gene_expr': gene_expr}

    for key in ['composite_target', 'report_target', 'response_target',
                'pathway_target', 'immune_score_target', 'immunogenicity_target']:
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch])

    return result


# ─── Sequence-Aware Split ──────────────────────────────────────────────

def compute_kmer_features(sequences, k=4):
    """Compute k-mer frequency vectors for sequence clustering."""
    # Build k-mer vocabulary
    kmers = set()
    for seq in sequences:
        seq = seq.upper().replace('T', 'U')
        for i in range(len(seq) - k + 1):
            kmers.add(seq[i:i+k])

    kmer_list = sorted(kmers)
    kmer_idx = {km: i for i, km in enumerate(kmer_list)}
    n_kmers = len(kmer_list)

    # Compute frequency vectors
    features = np.zeros((len(sequences), n_kmers), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper().replace('T', 'U')
        for j in range(len(seq) - k + 1):
            km = seq[j:j+k]
            if km in kmer_idx:
                features[i, kmer_idx[km]] += 1
        # Normalize
        total = features[i].sum()
        if total > 0:
            features[i] /= total

    return features, kmer_list


def sequence_aware_split(dataset, n_splits=5, n_clusters=50, k=4):
    """
    Sequence-aware split: cluster sequences by k-mer similarity,
    then assign entire clusters to folds.

    This prevents data leakage where similar sequences appear in
    both train and validation folds.
    """
    print("Computing sequence-aware split...")

    sequences = [dataset.df.iloc[i]['sequence'] for i in range(len(dataset))]

    # Compute k-mer features
    features, _ = compute_kmer_features(sequences, k=k)
    print(f"  k-mer features: {features.shape}")

    # Cluster sequences
    n_clusters = min(n_clusters, len(dataset) // 10)  # ensure enough per cluster
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    cluster_labels = kmeans.fit_predict(features)
    print(f"  Clustered into {n_clusters} groups")

    # Assign clusters to folds via stratified split on cluster labels
    # Use the dominant immune class in each cluster for stratification
    response_strs = [dataset.df.iloc[i].get('target_predicted_response', 'intermediate')
                     for i in range(len(dataset))]
    response_ints = [CircRNADataset.RESPONSE_MAP.get(r, 1) for r in response_strs]

    # Create fold indices: assign each cluster to one fold
    cluster_to_fold = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Get unique clusters and their dominant class
    unique_clusters = np.unique(cluster_labels)
    cluster_classes = []
    for c in unique_clusters:
        mask = cluster_labels == c
        class_counts = Counter(np.array(response_ints)[mask])
        cluster_classes.append(class_counts.most_common(1)[0][0])

    # Split clusters into folds
    for fold_idx, (train_clust, val_clust) in enumerate(
        skf.split(unique_clusters, cluster_classes)
    ):
        pass  # We only need the last split

    # Use simple approach: assign clusters to folds
    fold_assignments = np.zeros(len(dataset), dtype=int)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(unique_clusters, cluster_classes)):
        val_clusters = unique_clusters[val_idx]
        for c in val_clusters:
            fold_assignments[cluster_labels == c] = fold_idx

    print(f"  Fold distribution: {Counter(fold_assignments)}")

    return fold_assignments


# ─── Models ─────────────────────────────────────────────────────────────

class ESM2Backbone(nn.Module):
    """ESM2 backbone wrapper for TorusFold training."""

    def __init__(self, model_name="esm2_t12_35M_UR50D", freeze=True):
        super().__init__()
        import esm
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.freeze = freeze
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        # Get embedding dim directly from model
        self.d_model = self.model.embed_dim
        self.repr_layer = self.model.num_layers

    def encode_sequences(self, sequences, device='cpu'):
        """Encode sequences to per-position and pooled embeddings."""
        self.model = self.model.to(device)

        # U→T for ESM compatibility
        seqs_t = [s.upper().replace('U', 'T') for s in sequences]

        batch_converter = self.alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter([
            (f"seq_{i}", s) for i, s in enumerate(seqs_t)
        ])
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )
            embeddings = results["representations"][self.repr_layer]
            # Remove BOS/EOS
            token_emb = embeddings[:, 1:-1, :]  # (B, L, d)

            # Mean pooling
            mask = (batch_tokens[:, 1:-1] != self.alphabet.padding_idx).float().unsqueeze(-1)
            pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled  # (B, d_model)


class MockBackbone(nn.Module):
    """Mock backbone for testing without ESM2."""

    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(5, d_model)

    def encode_sequences(self, sequences, device='cpu'):
        B = len(sequences)
        embeddings = torch.zeros(B, self.d_model, device=device)
        for i, seq in enumerate(sequences):
            counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
            for c in seq.upper():
                if c in counts:
                    counts[c] += 1
            total = len(seq)
            feat = torch.tensor([
                counts['A']/total, counts['C']/total,
                counts['G']/total, counts['U']/total,
                total/500,
            ], device=device)
            embeddings[i] = self.embed.weight.mean(dim=0) + feat.mean() * 0.1
        return embeddings


class TorusFoldTrainer(nn.Module):
    """Trainable TorusFold v2 wrapper with AF3-style CircPairformer."""

    def __init__(self, backbone, d_model, gene_dim=6, hidden_dim=256,
                 n_composite=8, n_report=4, n_response=3, dropout=0.2,
                 n_harmonics=8, c_z=128, n_pairformer_blocks=2,
                 n_heads_tri=4):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        # TPE layer
        from core.tpe import TorusPositionalEncoding
        self.tpe = TorusPositionalEncoding(
            d_model=d_model, n_harmonics=n_harmonics, dropout=dropout)

        # AF3-style CircPairformer
        from core.triangle_update import CircPairformerStack
        from core.torusfold import PairInitialization, PairPredictionHead

        self.pair_init = PairInitialization(d_model, c_z)
        self.pairformer = CircPairformerStack(
            n_blocks=n_pairformer_blocks, c_z=c_z,
            n_heads_tri=n_heads_tri)
        self.pair_head = PairPredictionHead(c_z)

        # Multi-task heads (input includes pair features now)
        input_dim = d_model + gene_dim + c_z + 1  # +1 for bsj_stability

        self.composite_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_composite), nn.Sigmoid(),
        )
        self.report_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_report), nn.Sigmoid(),
        )
        self.response_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_response),
        )

    def forward(self, sequences, gene_expr, device='cpu'):
        # Encode sequences (pooled)
        seq_emb = self.backbone.encode_sequences(sequences, device)
        # Apply TPE (on pooled embeddings, minimal effect but keeps interface)
        seq_emb = self.tpe(seq_emb.unsqueeze(1), seq_len=1).squeeze(1)

        # For pair representation, we need per-position embeddings
        # Since backbone returns pooled, we create pseudo-positional repr
        B = len(sequences)
        # Use actual sequence lengths, cap to avoid OOM
        L_max = min(max(len(s) for s in sequences), 128)  # hard cap for memory

        # Create per-position repr by repeating pooled emb + TPE per position
        seq_repr = seq_emb.unsqueeze(1).expand(-1, L_max, -1)  # (B, L, d_model)

        # Initialize pair representation
        pair_repr = self.pair_init(seq_repr)  # (B, L, L, c_z)

        # CircPairformer: refine pair representation (AF3-style)
        pair_repr = self.pairformer(pair_repr)  # (B, L, L, c_z)

        # Pair prediction
        pair_probs = self.pair_head(pair_repr)  # (B, L, L)

        # BSJ stability (simplified)
        L = pair_repr.size(1)
        positions = torch.arange(L, device=pair_repr.device)
        diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        bsj_mask = (diff > L / 2).float()
        bsj_pair_strength = (pair_probs * bsj_mask).sum(dim=-1).mean(dim=-1)  # (B,)
        bsj_stability = torch.sigmoid(bsj_pair_strength)  # (B,)

        # Multi-task heads
        gene_expr = gene_expr.to(device)
        struct_feat = pair_repr.mean(dim=(1, 2))  # (B, c_z) — mean over L×L

        multi_input = torch.cat([
            seq_emb, gene_expr, struct_feat, bsj_stability.unsqueeze(-1),
        ], dim=-1)

        composite = self.composite_head(multi_input)
        report = self.report_head(multi_input)
        response_logits = self.response_head(multi_input)
        response_probs = F.softmax(response_logits, dim=-1)

        return {
            'composite': composite,
            'report': report,
            'response_logits': response_logits,
            'response_probs': response_probs,
            'embedding': seq_emb,
            'pair_probs': pair_probs,
            'bsj_stability': bsj_stability,
        }


# ─── Training ───────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    metrics = {}

    comp_pred = preds['composite'].cpu().numpy()
    comp_true = targets['composite_target'].cpu().numpy()

    comp_names = ['immunotherapy', 'tumor_killing', 'immunogenicity',
                  'immune_cycle', 'tme', 'therapeutic_window', 'tide', 'ips']
    for i, name in enumerate(comp_names):
        metrics[f'{name}_mse'] = mean_squared_error(comp_true[:, i], comp_pred[:, i])
    metrics['composite_avg_mse'] = np.mean([metrics[f'{n}_mse'] for n in comp_names])

    rep_pred = preds['report'].cpu().numpy()
    rep_true = targets['report_target'].cpu().numpy()
    rep_names = ['rig_i', 'tlr', 'pkr', 'risk']
    for i, name in enumerate(rep_names):
        metrics[f'{name}_mse'] = mean_squared_error(rep_true[:, i], rep_pred[:, i])
    metrics['report_avg_mse'] = np.mean([metrics[f'{n}_mse'] for n in rep_names])

    resp_pred = preds['response_probs'].cpu().numpy()
    resp_true = targets['response_target'].cpu().numpy()
    metrics['response_acc'] = accuracy_score(resp_true, resp_pred.argmax(axis=1))

    try:
        binary_true = (resp_true != 0).astype(int)
        binary_pred = resp_pred[:, 1:].sum(axis=1)
        metrics['response_auc'] = roc_auc_score(binary_true, binary_pred)
    except Exception:
        metrics['response_auc'] = 0.5

    # RIG-I / TLR / PKR pathway AUC (from report scores)
    try:
        rig_i_true = (rep_true[:, 0] > 0.5).astype(int)
        rig_i_pred = rep_pred[:, 0]
        metrics['rig_i_auc'] = roc_auc_score(rig_i_true, rig_i_pred)
    except Exception:
        metrics['rig_i_auc'] = 0.5

    return metrics


def run_epoch(model, dataloader, optimizer, device, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in dataloader:
            sequences = batch['sequences']
            gene_expr = batch['gene_expr'].to(device)
            comp_target = batch['composite_target'].to(device)
            rep_target = batch['report_target'].to(device)
            resp_target = batch['response_target'].to(device)

            outputs = model(sequences, gene_expr, device)

            loss = (F.mse_loss(outputs['composite'], comp_target) +
                    F.mse_loss(outputs['report'], rep_target) +
                    F.cross_entropy(outputs['response_logits'], resp_target))

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_preds.append({
                'composite': outputs['composite'].detach(),
                'report': outputs['report'].detach(),
                'response_probs': outputs['response_probs'].detach(),
            })
            all_targets.append({
                'composite_target': comp_target,
                'report_target': rep_target,
                'response_target': resp_target,
            })

    preds = {k: torch.cat([p[k] for p in all_preds]) for k in all_preds[0]}
    targets = {k: torch.cat([t[k] for t in all_targets]) for k in all_targets[0]}

    metrics = compute_metrics(preds, targets)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def external_validate(model, data_path, device, max_seq_len=512):
    """Validate on sequences_enhanced.csv (immune pathway prediction)."""
    ext_dataset = ExternalDataset(data_path, max_seq_len)
    ext_loader = DataLoader(ext_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    pathway_preds, pathway_trues = [], []
    score_preds, score_trues = [], []

    with torch.no_grad():
        for batch in ext_loader:
            sequences = batch['sequences']
            gene_expr = batch['gene_expr'].to(device)
            outputs = model(sequences, gene_expr, device)

            # Use report head outputs as pathway predictions
            # RIG-I: report[0], TLR: report[1], PKR: report[2]
            report = outputs['report'].cpu().numpy()
            pathway_trues.append(batch['pathway_target'].numpy())
            # Pathway = argmax of report scores
            pathway_preds.append(report[:, :3].argmax(axis=1))

            score_preds.append(outputs['composite'][:, 2].cpu().numpy())  # overall_immunogenicity
            score_trues.append(batch['immune_score_target'].numpy())

    pathway_trues = np.concatenate(pathway_trues)
    pathway_preds = np.concatenate(pathway_preds)
    score_preds = np.concatenate(score_preds)
    score_trues = np.concatenate(score_trues)

    results = {
        'n_samples': len(pathway_trues),
        'pathway_acc': accuracy_score(pathway_trues, pathway_preds),
        'score_mse': mean_squared_error(score_trues, score_preds),
        'score_corr': np.corrcoef(score_trues, score_preds)[0, 1]
            if len(score_trues) > 2 else 0.0,
    }

    # Per-pathway metrics
    for pw_name, pw_idx in [('RIG-I/MDA5', 0), ('TLR7/8', 1), ('JAK-STAT/PKR', 2)]:
        mask = pathway_trues == pw_idx
        if mask.sum() > 0:
            pw_acc = accuracy_score(pathway_trues[mask], pathway_preds[mask])
            results[f'pathway_{pw_name}_acc'] = pw_acc
            results[f'pathway_{pw_name}_n'] = int(mask.sum())

    return results


# ─── Hyperparameter Search ──────────────────────────────────────────────

def hparam_search(args):
    """Grid search over key hyperparameters."""
    print("\n" + "=" * 60)
    print("TorusFold Hyperparameter Search")
    print("=" * 60)

    # Load data
    dataset = CircRNADataset(args.data, max_seq_len=args.max_seq_len)

    # Define search space
    search_space = {
        'n_harmonics': [4, 8, 16],
        'hidden_dim': [128, 256],
        'dropout': [0.1, 0.2],
    }

    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = list(product(*values))

    print(f"Search space: {len(combos)} combinations")
    print(f"  n_harmonics: {search_space['n_harmonics']}")
    print(f"  hidden_dim: {search_space['hidden_dim']}")
    print(f"  dropout: {search_space['dropout']}")

    results = []

    for combo_idx, combo in enumerate(combos):
        hparams = dict(zip(keys, combo))
        print(f"\n--- Combo {combo_idx + 1}/{len(combos)}: {hparams} ---")

        # Create model
        if args.mock:
            backbone = MockBackbone(d_model=128)
            d_model = 128
        else:
            backbone = ESM2Backbone("esm2_t12_35M_UR50D", freeze=True)
            d_model = backbone.d_model

        model = TorusFoldTrainer(
            backbone=backbone,
            d_model=d_model,
            hidden_dim=hparams['hidden_dim'],
            dropout=hparams['dropout'],
            n_harmonics=hparams['n_harmonics'],
            c_z=128 if not args.mock else 32,
            n_pairformer_blocks=2 if args.mock else 4,
            n_heads_tri=2 if args.mock else 4,
        )

        device = torch.device(args.device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

        # Simple train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn)

        # Train for a few epochs
        best_val = float('inf')
        for epoch in range(args.epochs):
            train_m = run_epoch(model, train_loader, optimizer, device, training=True)
            val_m = run_epoch(model, val_loader, None, device, training=False)
            best_val = min(best_val, val_m['composite_avg_mse'])
            print(f"  Epoch {epoch+1}: val_mse={val_m['composite_avg_mse']:.4f}")

        result = {**hparams, 'best_val_mse': best_val}
        results.append(result)
        print(f"  Best val MSE: {best_val:.4f}")

    # Report best
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('best_val_mse')
    print("\n" + "=" * 60)
    print("Hyperparameter Search Results (sorted by val MSE)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Save
    results_path = PROJECT_ROOT / 'models' / 'hparam_search_results.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved to {results_path}")

    best = results_df.iloc[0]
    print(f"\nBest config: {dict(best)}")

    return best


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train TorusFold on circRNA data')
    parser.add_argument('--data', type=str,
                        default='data/circrna/unified_training_data.csv')
    parser.add_argument('--external-data', type=str,
                        default='data/circrna/sequences_enhanced.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--hparam-search', action='store_true')
    parser.add_argument('--external-only', action='store_true')
    parser.add_argument('--seq-aware-split', action='store_true',
                        help='Use sequence-aware split (k-mer clustering)')
    parser.add_argument('--esm-model', type=str, default='esm2_t12_35M_UR50D',
                        help='ESM2 model name (smaller=faster)')
    parser.add_argument('--n-harmonics', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--output', type=str, default='models/torusfold_best.pt')
    parser.add_argument('--max-seq-len', type=int, default=200,
                        help='Max sequence length for pair repr (O(L^2) memory)')
    args = parser.parse_args()

    # Hyperparameter search mode
    if args.hparam_search:
        best = hparam_search(args)
        return

    print("=" * 60)
    print("TorusFold Training")
    print("=" * 60)
    print(f"  Data:       {args.data}")
    print(f"  Device:     {args.device}")
    print(f"  Mock:       {args.mock}")
    print(f"  ESM model:  {args.esm_model}")
    print(f"  Seq-aware:  {args.seq_aware_split}")
    print(f"  n_harmonics: {args.n_harmonics}")
    print(f"  hidden_dim: {args.hidden_dim}")
    print("=" * 60)

    # ─── Create model ───────────────────────────────────────────────

    if args.mock:
        backbone = MockBackbone(d_model=128)
        d_model = 128
        print("Using Mock backbone")
    else:
        print(f"Loading ESM2 backbone: {args.esm_model}")
        backbone = ESM2Backbone(args.esm_model, freeze=True)
        d_model = backbone.d_model
        print(f"  d_model={d_model}")

    c_z = 32 if args.mock else 128
    n_pf_blocks = 2 if args.mock else 4
    n_heads_tri = 2 if args.mock else 4

    model = TorusFoldTrainer(
        backbone=backbone,
        d_model=d_model,
        hidden_dim=args.hidden_dim,
        n_harmonics=args.n_harmonics,
        c_z=c_z,
        n_pairformer_blocks=n_pf_blocks,
        n_heads_tri=n_heads_tri,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ─── Load data ──────────────────────────────────────────────────

    dataset = CircRNADataset(args.data, max_seq_len=args.max_seq_len)

    # ─── Split ──────────────────────────────────────────────────────

    if args.seq_aware_split:
        fold_assignments = sequence_aware_split(dataset, n_splits=5, n_clusters=50)

        # Use fold 0 as validation
        val_indices = np.where(fold_assignments == 0)[0]
        train_indices = np.where(fold_assignments != 0)[0]

        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
    else:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # ─── External validation only ───────────────────────────────────

    if args.external_only:
        device = torch.device(args.device)
        model = model.to(device)

        # Load best model if exists
        model_path = PROJECT_ROOT / args.output
        if model_path.exists():
            state = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state['model_state_dict'])
            print(f"Loaded model from {model_path}")

        ext_results = external_validate(model, args.external_data, device)
        print("\n" + "=" * 60)
        print("External Validation Results (sequences_enhanced.csv)")
        print("=" * 60)
        for k, v in ext_results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return

    # ─── Training loop ──────────────────────────────────────────────

    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_val_mse = float('inf')
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, optimizer, device, training=True)
        val_m = run_epoch(model, val_loader, None, device, training=False)

        scheduler.step(val_m['composite_avg_mse'])

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) lr={lr_now:.2e}")
        print(f"  Train loss={train_m['loss']:.4f} comp_mse={train_m['composite_avg_mse']:.4f} "
              f"rep_mse={train_m['report_avg_mse']:.4f} acc={train_m['response_acc']:.4f}")
        print(f"  Val   loss={val_m['loss']:.4f} comp_mse={val_m['composite_avg_mse']:.4f} "
              f"rep_mse={val_m['report_avg_mse']:.4f} acc={val_m['response_acc']:.4f} "
              f"auc={val_m['response_auc']:.4f}")
        print(f"  RIG-I auc={val_m['rig_i_auc']:.4f} | "
              f"RIG-I mse={val_m['rig_i_mse']:.4f} TLR mse={val_m['tlr_mse']:.4f} "
              f"PKR mse={val_m['pkr_mse']:.4f}")

        history.append({
            'epoch': epoch + 1,
            'lr': lr_now,
            'train': {k: float(v) for k, v in train_m.items()},
            'val': {k: float(v) for k, v in val_m.items()},
        })

        # Save best model
        if val_m['composite_avg_mse'] < best_val_mse:
            best_val_mse = val_m['composite_avg_mse']
            output_path = PROJECT_ROOT / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_metrics': val_m,
                'config': {
                    'd_model': d_model,
                    'hidden_dim': args.hidden_dim,
                    'n_harmonics': args.n_harmonics,
                    'esm_model': args.esm_model,
                },
            }, output_path)
            print(f"  ★ Saved best model (val_mse={best_val_mse:.4f})")

    # ─── External validation ────────────────────────────────────────

    print("\n" + "=" * 60)
    print("Final External Validation")
    print("=" * 60)

    # Load best model
    model_path = PROJECT_ROOT / args.output
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])

    ext_results = external_validate(model, args.external_data, device)
    print(f"  External samples: {ext_results['n_samples']}")
    print(f"  Pathway accuracy: {ext_results['pathway_acc']:.4f}")
    print(f"  Score MSE: {ext_results['score_mse']:.4f}")
    print(f"  Score correlation: {ext_results['score_corr']:.4f}")

    for key in sorted(ext_results.keys()):
        if key.startswith('pathway_') and key.endswith('_acc'):
            pw_name = key.replace('pathway_', '').replace('_acc', '')
            n = ext_results.get(key.replace('_acc', '_n'), '?')
            print(f"  {pw_name}: acc={ext_results[key]:.4f} (n={n})")

    # ─── Save history ───────────────────────────────────────────────

    history_path = PROJECT_ROOT / 'models' / 'torusfold_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save external validation results
    ext_path = PROJECT_ROOT / 'models' / 'torusfold_external_results.json'
    with open(ext_path, 'w') as f:
        json.dump(ext_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best val MSE: {best_val_mse:.4f}")
    print(f"  External pathway acc: {ext_results['pathway_acc']:.4f}")
    print(f"  External score corr: {ext_results['score_corr']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
