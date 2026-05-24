"""
circrna_cli.py — Command-line interface for circRNA prediction.

Usage:
    PYTHONPATH=/root/autodl-tmp/confluencia:$PYTHONPATH \
      python -m confluencia_circrna_encoder.tools.circrna_cli predict \
        --model confluencia-circrna-encoder/data/models/finetune_xgb.joblib \
        --sequence "AUCCAAAAGCGGGGUAUUUG" \
        --output json

Commands:
    train        Train model on circRNA data
    predict      Predict immunogenicity for circRNA sequence
    batch        Batch prediction from FASTA file
    optimize     Optimize sequence for target immunogenicity
    simulate     Run immune response simulation
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import joblib

# Add project paths - handle both module and direct script execution
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]

# Add both possible paths
for p in [_PROJECT_ROOT, _SCRIPT_DIR.parent, _SCRIPT_DIR.parent / "core"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Try relative imports first, fall back to absolute
try:
    from core.moe import train_moe_model
    from core.features import CircRNAFeatureExtractor
    from core.innate_immune import quick_predict
    from core.dose_tox import quick_dose_predict
    from core.admet import quick_admet
    from core.generative import generate_optimized_sequence
    from core.immune_abm import simulate_circrna_response
except ImportError:
    from confluencia_circrna_encoder.core.moe import train_moe_model
    from confluencia_circrna_encoder.core.features import CircRNAFeatureExtractor
    from confluencia_circrna_encoder.core.innate_immune import quick_predict
    from confluencia_circrna_encoder.core.dose_tox import quick_dose_predict
    from confluencia_circrna_encoder.core.admet import quick_admet
    from confluencia_circrna_encoder.core.generative import generate_optimized_sequence
    from confluencia_circrna_encoder.core.immune_abm import simulate_circrna_response


def predict_single(
    sequence: str,
    model_path: str,
    output_format: str = "json",
    detailed: bool = False,
) -> Dict:
    """Predict immunogenicity for single sequence."""

    # Load model
    model = joblib.load(model_path)

    # Load scaler if available
    scaler_path = Path(model_path).parent / "finetune_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None

    # Extract features
    extractor = CircRNAFeatureExtractor()
    features = extractor.extract(sequence)

    if scaler:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]

    # Probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = max(proba)
    else:
        confidence = 0.8  # Default

    result = {
        'sequence': sequence[:50] + '...' if len(sequence) > 50 else sequence,
        'length': len(sequence),
        'immunogenicity': float(prediction),
        'confidence': float(confidence),
        'level': "High" if prediction > 0.6 else ("Medium" if prediction > 0.4 else "Low"),
    }

    # Add detailed analysis
    if detailed:
        immune = quick_predict(sequence)
        dose = quick_dose_predict(sequence, dose=100)
        admet = quick_admet(sequence)

        result['detailed'] = {
            'innate_immune': {
                'rig_i': immune['rig_i']['score'],
                'tlr': immune['tlr']['score'],
                'pkr': immune['pkr']['score'],
            },
            'dose_response': {
                'efficacy': dose['efficacy_score'],
                'toxicity': dose['toxicity_score'],
                'therapeutic_window': dose['therapeutic_window'],
            },
            'admet': {
                'pass': admet['pass'],
                'recommendation': admet['recommendation'],
            },
        }

    return result


def predict_batch(
    fasta_path: str,
    model_path: str,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Batch prediction from FASTA file."""
    import gzip

    # Load model and scaler
    model = joblib.load(model_path)
    scaler_path = Path(model_path).parent / "finetune_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    # Parse FASTA
    sequences = []
    opener = gzip.open if fasta_path.endswith('.gz') else open
    with opener(fasta_path, 'rt') as f:
        current_id = None
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append((current_id, current_seq))
                current_id = line[1:].split('|')[0]
                current_seq = ""
            else:
                current_seq += line.upper()
        if current_id and current_seq:
            sequences.append((current_id, current_seq))

    # Batch predict
    extractor = CircRNAFeatureExtractor()

    results = []
    for seq_id, seq in sequences:
        features = extractor.extract(seq).reshape(1, -1)
        if scaler:
            features = scaler.transform(features)
        pred = model.predict(features)[0]
        results.append({
            'circrna_id': seq_id,
            'length': len(seq),
            'immunogenicity': float(pred),
            'level': "High" if pred > 0.6 else ("Medium" if pred > 0.4 else "Low"),
        })

    # Save output
    if output_path:
        import pandas as pd
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Saved {len(results)} predictions to {output_path}")

    return results


def optimize_sequence(
    sequence: str,
    target: float = 0.6,
    iterations: int = 50,
) -> Dict:
    """Optimize sequence for target immunogenicity."""

    opt_seq, score = generate_optimized_sequence(sequence, target, iterations)

    return {
        'original_sequence': sequence[:50] + '...',
        'optimized_sequence': opt_seq[:50] + '...',
        'original_length': len(sequence),
        'optimized_length': len(opt_seq),
        'target_score': target,
        'achieved_score': score,
        'improvement': score - target,
    }


def train_moe(
    labels_path: str,
    output_dir: str = "confluencia-circrna-encoder/data/models",
    n_sequences: int = 10000,
) -> Dict:
    """Train MOE (Mixture of Experts) model."""
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOE Model Training")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading data from {labels_path}...")
    df = pd.read_csv(labels_path)
    print(f"    Loaded {len(df)} samples")

    # Get sequences and labels
    sequences = df['sequence'].tolist()
    labels = df['orig_immunogenicity'].values

    if len(sequences) > n_sequences:
        import random
        random.seed(42)
        indices = random.sample(range(len(sequences)), n_sequences)
        sequences = [sequences[i] for i in indices]
        labels = labels[indices]
        print(f"    Sampled {len(sequences)} sequences")

    # Train MOE
    print(f"\n[2] Training MOE on {len(sequences)} sequences...")
    model = train_moe_model(sequences, labels)

    # Save model
    model_path = output_path / "moe_model.joblib"
    joblib.dump(model, model_path)

    print(f"\n✓ MOE model saved to: {model_path}")

    # Test prediction
    test_seq = sequences[0]
    pred = model.predict([test_seq])
    print(f"    Test prediction: {pred[0]:.4f}")

    return {
        'n_sequences': len(sequences),
        'model_path': str(model_path),
        'test_prediction': float(pred[0]),
    }


def train_model(
    fasta_path: str,
    labels_path: Optional[str] = None,
    output_dir: str = "confluencia-circrna-encoder/data/models",
    mode: str = "full",
    max_sequences: int = 50000,
) -> Dict:
    """Train circRNA prediction model."""
    import pandas as pd
    import gzip
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("circRNA Model Training")
    print("=" * 60)

    # Load sequences
    print(f"\n[1] Loading sequences from {fasta_path}...")
    sequences = []
    opener = gzip.open if fasta_path.endswith('.gz') else open
    with opener(fasta_path, 'rt') as f:
        current_id = None
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    sequences.append((current_id, current_seq))
                current_id = line[1:].split('|')[0]
                current_seq = ""
            else:
                current_seq += line.upper()
        if current_id and current_seq:
            sequences.append((current_id, current_seq))

    print(f"    Loaded {len(sequences)} sequences")

    if len(sequences) > max_sequences:
        import random
        random.seed(42)
        sequences = random.sample(sequences, max_sequences)
        print(f"    Sampled {len(sequences)} sequences")

    # Extract features
    print("\n[2] Extracting features...")
    extractor = CircRNAFeatureExtractor()

    features_list = []
    for i, (seq_id, seq) in enumerate(sequences):
        if i % 5000 == 0:
            print(f"    Progress: {i}/{len(sequences)}")
        features_list.append(extractor.extract(seq))

    features = np.array(features_list)
    print(f"    Feature matrix: {features.shape}")

    # Get labels
    if labels_path and Path(labels_path).exists():
        print(f"\n[3] Loading labels from {labels_path}...")
        df = pd.read_csv(labels_path)
        labels = df['orig_immunogenicity'].values[:len(features)]
        print(f"    Labels: 0={int((labels==0).sum())}, 1={int((labels==1).sum())}")
    else:
        print("\n[3] Generating pseudo-labels...")
        gc = features[:, 4]  # GC column
        entropy = features[:, 8]  # Entropy
        scores = gc * 0.4 + (entropy / 2) * 0.3 + np.random.uniform(-0.1, 0.1, len(features))
        labels = (scores > 0.45).astype(int)
        print(f"    Pseudo-labels: 0={int((labels==0).sum())}, 1={int((labels==1).sum())}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    print("\n[4] Training XGBoost...")
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train_scaled, y_train)

        y_pred = xgb_model.predict(X_test_scaled)
        y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]

        xgb_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob),
        }
        print(f"    Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"    F1: {xgb_metrics['f1']:.4f}")
        print(f"    AUC: {xgb_metrics['auc']:.4f}")

        joblib.dump(xgb_model, output_path / "circrna_xgb.joblib")
    except ImportError:
        print("    XGBoost not available")
        xgb_metrics = {}

    # Train Random Forest
    print("\n[5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    y_pred_rf = rf_model.predict(X_test_scaled)
    y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'auc': roc_auc_score(y_test, y_prob_rf),
    }
    print(f"    Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"    F1: {rf_metrics['f1']:.4f}")
    print(f"    AUC: {rf_metrics['auc']:.4f}")

    joblib.dump(rf_model, output_path / "circrna_rf.joblib")
    joblib.dump(scaler, output_path / "scaler.joblib")

    print(f"\n✓ Training complete!")
    print(f"  Models saved to: {output_path}")

    return {
        'n_sequences': len(sequences),
        'xgb_metrics': xgb_metrics,
        'rf_metrics': rf_metrics,
        'output_dir': str(output_path),
    }


def simulate_response(
    sequence: str,
    dose: float = 100.0,
    steps: int = 100,
) -> Dict:
    """Simulate immune response."""

    result = simulate_circrna_response(sequence, n_steps=steps)

    return {
        'sequence_length': len(sequence),
        'dose': dose,
        'simulation_steps': result['total_steps'],
        'final_tumor_count': result['final_tumor_count'],
        'tumor_kill_rate': result['tumor_kill_rate'],
        'peak_cytokines': result['peak_cytokines'],
    }


def main():
    parser = argparse.ArgumentParser(
        description="circRNA prediction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train XGBoost/RF model")
    train_parser.add_argument("--fasta", required=True, help="FASTA file with sequences")
    train_parser.add_argument("--labels", help="Labels CSV file (optional)")
    train_parser.add_argument("--output-dir", default="confluencia-circrna-encoder/data/models")
    train_parser.add_argument("--max-sequences", type=int, default=50000)

    # Train MOE command
    moe_parser = subparsers.add_parser("train-moe", help="Train MOE model")
    moe_parser.add_argument("--labels", required=True, help="Labels CSV file")
    moe_parser.add_argument("--output-dir", default="confluencia-circrna-encoder/data/models")
    moe_parser.add_argument("--n-sequences", type=int, default=10000, help="Number of sequences to use")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict immunogenicity")
    predict_parser.add_argument("--model", required=True, help="Model path")
    predict_parser.add_argument("--sequence", required=True, help="circRNA sequence")
    predict_parser.add_argument("--output", choices=["json", "text"], default="json")
    predict_parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch prediction")
    batch_parser.add_argument("--model", required=True, help="Model path")
    batch_parser.add_argument("--fasta", required=True, help="FASTA file")
    batch_parser.add_argument("--output", help="Output CSV path")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize sequence")
    optimize_parser.add_argument("--sequence", required=True, help="Starting sequence")
    optimize_parser.add_argument("--target", type=float, default=0.6, help="Target score")
    optimize_parser.add_argument("--iterations", type=int, default=50)

    # Simulate command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate immune response")
    simulate_parser.add_argument("--sequence", required=True)
    simulate_parser.add_argument("--dose", type=float, default=100.0)
    simulate_parser.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()

    if args.command == "train":
        result = train_model(
            args.fasta,
            args.labels,
            args.output_dir,
            args.max_sequences,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "train-moe":
        result = train_moe(
            args.labels,
            args.output_dir,
            args.n_sequences,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "predict":
        result = predict_single(args.sequence, args.model, args.output, args.detailed)
        if args.output == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Immunogenicity: {result['immunogenicity']:.4f}")
            print(f"Level: {result['level']}")

    elif args.command == "batch":
        results = predict_batch(args.fasta, args.model, args.output)
        if not args.output:
            print(json.dumps(results[:10], indent=2))
            print(f"... {len(results)} total predictions")

    elif args.command == "optimize":
        result = optimize_sequence(args.sequence, args.target, args.iterations)
        print(json.dumps(result, indent=2))

    elif args.command == "simulate":
        result = simulate_response(args.sequence, args.dose, args.steps)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()