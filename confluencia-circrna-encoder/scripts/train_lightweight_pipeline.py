"""
train_lightweight_pipeline.py — Complete training pipeline for circRNA.

Stage 1: Pretrain (feature extraction + pseudo-labels)
Stage 2: Finetune (with real labels if available)

Usage:
    python train_lightweight_pipeline.py --mode full
    python train_lightweight_pipeline.py --mode pretrain
    python train_lightweight_pipeline.py --mode finetune --labels data/circrna/unified_training_data.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import joblib

# Project paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class CircRNAFeatureExtractor:
    """Extract features from circRNA sequences."""

    NUCS = ['A', 'U', 'G', 'C']

    def __init__(self):
        self.feature_names = []

    def extract(self, sequence: str) -> np.ndarray:
        """Extract all features from a sequence."""
        seq = sequence.upper().replace('T', 'U')
        length = len(seq)

        features = []
        self.feature_names = []

        counts = Counter(seq)

        # 1. Nucleotide frequencies (4)
        for nuc in self.NUCS:
            features.append(counts.get(nuc, 0) / max(length, 1))
            self.feature_names.append(f'{nuc}_freq')

        # 2. GC content
        gc = (counts.get('G', 0) + counts.get('C', 0)) / max(length, 1)
        features.append(gc)
        self.feature_names.append('gc_content')

        # 3. AU content
        au = (counts.get('A', 0) + counts.get('U', 0)) / max(length, 1)
        features.append(au)
        self.feature_names.append('au_content')

        # 4. Purine (AG) content
        purine = (counts.get('A', 0) + counts.get('G', 0)) / max(length, 1)
        features.append(purine)
        self.feature_names.append('purine_content')

        # 5. Entropy
        probs = [counts.get(n, 0) / max(length, 1) for n in self.NUCS]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
        features.append(entropy)
        self.feature_names.append('entropy')

        # 6. Length
        features.append(min(length / 1000.0, 2.0))
        self.feature_names.append('length_normalized')

        # 7. Log length
        features.append(np.log1p(length))
        self.feature_names.append('log_length')

        # 8. Di-nucleotide frequencies (16)
        dinucs = ['AA', 'AU', 'AG', 'AC', 'UA', 'UU', 'UG', 'UC',
                  'GA', 'GU', 'GG', 'GC', 'CA', 'CU', 'CG', 'CC']
        for dinuc in dinucs:
            count = sum(1 for k in range(len(seq)-1) if seq[k:k+2] == dinuc)
            features.append(count / max(length-1, 1))
            self.feature_names.append(f'dinuc_{dinuc}')

        # 9. Important tri-nucleotides (16)
        trinucs = ['AUU', 'AGU', 'ACU', 'UAU', 'UGU', 'UCU', 'GAU', 'GUU',
                   'UUU', 'AAA', 'GGG', 'CCC', 'AUG', 'UAG', 'GAC', 'CAG']
        for trinuc in trinucs:
            count = sum(1 for k in range(len(seq)-2) if seq[k:k+3] == trinuc)
            features.append(count / max(length-2, 1))
            self.feature_names.append(f'trinuc_{trinuc}')

        # 10. Repeat content
        max_repeat = 0
        for nuc in self.NUCS:
            count = 0
            max_c = 0
            for c in seq:
                if c == nuc:
                    count += 1
                    max_c = max(max_c, count)
                else:
                    count = 0
            max_repeat = max(max_repeat, max_c)
        features.append(max_repeat / max(length, 1))
        self.feature_names.append('max_repeat_ratio')

        # 11. Complexity (unique 4-mers)
        if length >= 4:
            unique_4mers = len(set(seq[i:i+4] for i in range(length-3)))
            features.append(unique_4mers / max(length-3, 1))
        else:
            features.append(0)
        self.feature_names.append('complexity')

        # 12. Transition frequencies (AU→G, etc.)
        transitions = 0
        for k in range(len(seq)-1):
            if seq[k] in ['A', 'U'] and seq[k+1] in ['G', 'C']:
                transitions += 1
        features.append(transitions / max(length-1, 1))
        self.feature_names.append('au_to_gc_transition')

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        return self.feature_names


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    """Parse FASTA file."""
    sequences = []

    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        current_id = None
        current_seq = ""

        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append((current_id, current_seq))
                current_id = line[1:].split('|')[0].strip()
                current_seq = ""
            else:
                current_seq += line.upper()

        if current_id and current_seq:
            sequences.append((current_id, current_seq))

    return sequences


def generate_pseudo_labels(features: np.ndarray) -> np.ndarray:
    """Generate pseudo-labels based on features."""
    gc = features[:, 4]      # GC content
    entropy = features[:, 8]  # Entropy
    complexity = features[:, -2]  # Complexity
    au_to_gc = features[:, -1]  # Transitions

    # Composite score
    score = (
        gc * 0.25 +
        (entropy / 2.0) * 0.20 +
        complexity * 0.20 +
        au_to_gc * 0.15 +
        np.random.uniform(-0.10, 0.10, len(features))
    )

    labels = (score > 0.45).astype(int)
    return labels, score


def pretrain(
    fasta_path: str,
    max_sequences: int,
    output_dir: Path,
    seed: int,
) -> Dict:
    """Stage 1: Pretrain with pseudo-labels."""
    print("\n" + "=" * 70)
    print("STAGE 1: Pretraining (Feature Extraction + Pseudo-Labels)")
    print("=" * 70)

    # Load sequences
    print(f"\n[1] Loading sequences from {fasta_path}...")
    sequences = parse_fasta(fasta_path)
    print(f"    Loaded {len(sequences)} sequences")

    if len(sequences) > max_sequences:
        indices = np.random.choice(len(sequences), max_sequences, replace=False)
        sequences = [sequences[i] for i in indices]
        print(f"    Sampled {len(sequences)} sequences")

    # Extract features
    print("\n[2] Extracting features...")
    extractor = CircRNAFeatureExtractor()

    features_list = []
    seq_ids = []

    for i, (seq_id, seq) in enumerate(sequences):
        if i % 10000 == 0:
            print(f"    Progress: {i}/{len(sequences)}")
        features_list.append(extractor.extract(seq))
        seq_ids.append(seq_id)

    features = np.array(features_list)
    print(f"    Feature matrix: {features.shape}")
    print(f"    Features: {len(extractor.get_feature_names())} dimensions")

    # Generate pseudo-labels
    print("\n[3] Generating pseudo-labels...")
    labels, scores = generate_pseudo_labels(features)
    print(f"    Labels: 0={int((labels==0).sum())}, 1={int((labels==1).sum())}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train XGBoost
    print("\n[4] Training XGBoost...")
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1,
        )
        xgb_model.fit(X_train_scaled, y_train)

        y_pred_xgb = xgb_model.predict(X_val_scaled)
        y_prob_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]

        xgb_metrics = {
            'accuracy': accuracy_score(y_val, y_pred_xgb),
            'f1': f1_score(y_val, y_pred_xgb),
            'auc': roc_auc_score(y_val, y_prob_xgb),
            'precision': precision_score(y_val, y_pred_xgb),
            'recall': recall_score(y_val, y_pred_xgb),
        }
        print(f"    Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"    F1: {xgb_metrics['f1']:.4f}")
        print(f"    AUC: {xgb_metrics['auc']:.4f}")

        joblib.dump(xgb_model, output_dir / "pretrain_xgb.joblib")
    except ImportError:
        print("    XGBoost not available, skipping")
        xgb_model = None
        xgb_metrics = {}

    # Train Random Forest
    print("\n[5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_val_scaled)
    y_prob_rf = rf_model.predict_proba(X_val_scaled)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(y_val, y_pred_rf),
        'f1': f1_score(y_val, y_pred_rf),
        'auc': roc_auc_score(y_val, y_prob_rf),
        'precision': precision_score(y_val, y_pred_rf),
        'recall': recall_score(y_val, y_pred_rf),
    }
    print(f"    Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"    F1: {rf_metrics['f1']:.4f}")
    print(f"    AUC: {rf_metrics['auc']:.4f}")

    joblib.dump(rf_model, output_dir / "pretrain_rf.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")

    # Save feature info
    feature_info = {
        'feature_names': extractor.get_feature_names(),
        'n_features': features.shape[1],
        'n_sequences': len(sequences),
        'pretrain_xgb_metrics': xgb_metrics,
        'pretrain_rf_metrics': rf_metrics,
    }

    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"\n✓ Pretraining complete!")
    print(f"  Models saved: pretrain_xgb.joblib, pretrain_rf.joblib")

    return {
        'features': features,
        'labels': labels,
        'scores': scores,
        'seq_ids': seq_ids,
        'scaler': scaler,
        'xgb_model': xgb_model,
        'rf_model': rf_model,
    }


def finetune(
    labels_path: str,
    pretrain_data: Dict,
    output_dir: Path,
    seed: int,
) -> Dict:
    """Stage 2: Finetune with real labels."""
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-Tuning (Real Labels)")
    print("=" * 70)

    # Load labeled data
    print(f"\n[1] Loading labeled data from {labels_path}...")
    labels_path = Path(labels_path)
    if not labels_path.exists():
        labels_path = _PROJECT_ROOT / labels_path

    if not labels_path.exists():
        print("⚠ No labeled data found, skipping fine-tuning")
        return {}

    df = pd.read_csv(labels_path)
    print(f"    Loaded {len(df)} samples")
    print(f"    Columns: {list(df.columns)[:10]}...")

    # Extract features from labeled sequences
    print("\n[2] Extracting features from labeled data...")
    extractor = CircRNAFeatureExtractor()

    features_list = []
    valid_indices = []

    for i, row in df.iterrows():
        seq = str(row.get('sequence', row.get('full_sequence', '')))
        if len(seq) > 50:
            features_list.append(extractor.extract(seq))
            valid_indices.append(i)

    features = np.array(features_list)
    print(f"    Valid samples: {len(features)}")

    # Get labels
    if 'orig_immunogenicity' in df.columns:
        labels = df['orig_immunogenicity'].iloc[valid_indices].values
    elif 'pseudo_immunogenicity' in df.columns:
        labels = df['pseudo_immunogenicity'].iloc[valid_indices].values
    elif 'immunogenicity' in df.columns:
        labels = df['immunogenicity'].iloc[valid_indices].values
    else:
        print("⚠ No label column found, using pseudo-labels")
        labels, _ = generate_pseudo_labels(features)

    print(f"    Labels: 0={int((labels==0).sum())}, 1={int((labels==1).sum())}")

    # Use pretrain scaler
    scaler = pretrain_data.get('scaler')
    if scaler:
        features_scaled = scaler.transform(features)
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=seed
    )

    # Fine-tune XGBoost
    print("\n[3] Fine-tuning XGBoost...")
    try:
        from xgboost import XGBClassifier

        # Use pretrained model as base or train new
        xgb_finetune = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1,
        )
        xgb_finetune.fit(X_train, y_train)

        y_pred = xgb_finetune.predict(X_test)
        y_prob = xgb_finetune.predict_proba(X_test)[:, 1]

        xgb_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
        }
        print(f"    Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"    F1: {xgb_metrics['f1']:.4f}")
        print(f"    AUC: {xgb_metrics['auc']:.4f}")

        joblib.dump(xgb_finetune, output_dir / "finetune_xgb.joblib")
    except ImportError:
        xgb_finetune = None
        xgb_metrics = {}

    # Fine-tune Random Forest
    print("\n[4] Fine-tuning Random Forest...")
    rf_finetune = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    rf_finetune.fit(X_train, y_train)

    y_pred_rf = rf_finetune.predict(X_test)
    y_prob_rf = rf_finetune.predict_proba(X_test)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'auc': roc_auc_score(y_test, y_prob_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
    }
    print(f"    Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"    F1: {rf_metrics['f1']:.4f}")
    print(f"    AUC: {rf_metrics['auc']:.4f}")

    joblib.dump(rf_finetune, output_dir / "finetune_rf.joblib")
    joblib.dump(scaler, output_dir / "finetune_scaler.joblib")

    # Save finetune config
    finetune_config = {
        'n_samples': len(features),
        'n_features': features.shape[1],
        'xgb_metrics': xgb_metrics,
        'rf_metrics': rf_metrics,
    }
    with open(output_dir / "finetune_config.json", 'w') as f:
        json.dump(finetune_config, f, indent=2)

    print(f"\n✓ Fine-tuning complete!")
    print(f"  Models saved: finetune_xgb.joblib, finetune_rf.joblib")

    return {
        'xgb_model': xgb_finetune,
        'rf_model': rf_finetune,
        'scaler': scaler,
        'metrics': {'xgb': xgb_metrics, 'rf': rf_metrics},
    }


def main():
    parser = argparse.ArgumentParser(description="Complete circRNA training pipeline")
    parser.add_argument("--mode", choices=["full", "pretrain", "finetune"], default="full")
    parser.add_argument("--fasta", default="data/circrna/human_hg19_circRNAs_putative_spliced_sequence.fa")
    parser.add_argument("--labels", default="data/circrna/unified_training_data.csv")
    parser.add_argument("--output-dir", default="confluencia-circrna-encoder/data/models")
    parser.add_argument("--max-sequences", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("circRNA Lightweight Training Pipeline")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"No large backbone, no overfitting")
    print("=" * 70)

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve fasta path
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        fasta_path = _PROJECT_ROOT / args.fasta

    if not fasta_path.exists():
        print(f"⚠ FASTA file not found: {args.fasta}")
        return

    pretrain_data = {}

    # Stage 1: Pretrain
    if args.mode in ["full", "pretrain"]:
        pretrain_data = pretrain(
            str(fasta_path),
            args.max_sequences,
            output_dir,
            args.seed,
        )

    # Stage 2: Finetune
    if args.mode in ["full", "finetune"]:
        finetune_data = finetune(
            args.labels,
            pretrain_data,
            output_dir,
            args.seed,
        )

    # Summary
    print("\n" + "=" * 70)
    print("Training Pipeline Complete!")
    print("=" * 70)
    print(f"Models saved to: {output_dir}")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")

    print("\nTo use models:")
    print("  import joblib")
    print("  model = joblib.load('finetune_xgb.joblib')")
    print("  scaler = joblib.load('scaler.joblib')")


if __name__ == "__main__":
    main()