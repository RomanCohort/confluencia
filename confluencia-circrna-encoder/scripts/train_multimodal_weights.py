"""
train_multimodal_weights.py - Train multi-modal prediction weights.

Usage:
    python train_multimodal_weights.py \
        --data data/circrna/unified_training_data.csv \
        --output models/multimodal_weights.json \
        --n-samples 5000 \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train multi-modal weights")
    parser.add_argument("--data", required=True, help="Training data CSV")
    parser.add_argument("--output", default="models/multimodal_weights.json", help="Output weights JSON")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")

    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Modal Weight Training")
    print("=" * 60)

    # Import weight fitting module
    try:
        from core.multimodal_weight_fitting import fit_weights_from_csv, save_weights
    except ImportError:
        from confluencia_circrna_encoder.core.multimodal_weight_fitting import fit_weights_from_csv, save_weights

    # Fit weights
    weights = fit_weights_from_csv(
        args.data,
        n_samples=args.n_samples,
        n_folds=args.n_folds,
    )

    # Save weights
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_weights(weights, str(output_path))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"\nFinal weights:")
    print(f"  sequence: {weights.sequence:.4f}")
    print(f"  innate:   {weights.innate:.4f}")
    print(f"  dose:     {weights.dose:.4f}")
    print(f"  ctm:      {weights.ctm:.4f}")
    print(f"  admet:    {weights.admet:.4f}")
    print(f"  gene:     {weights.gene:.4f}")


if __name__ == "__main__":
    main()