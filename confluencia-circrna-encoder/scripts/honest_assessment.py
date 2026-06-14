"""
honest_assessment.py — Honest assessment of TorusFold's real utility.

PROBLEM: The current training data (unified_training_data.csv) has
fundamental issues that no architecture can fix:

1. 8 "composite" targets are perfectly collinear (r=1.0)
   → effectively just 1 variable with different scalings
2. 96% (50K/52K) are pseudo-labels with zero variance
   → rig_i_score std=0.0, tlr_score std=0.0 in pseudo data
3. Gene expression features have ZERO correlation with targets
4. Only 20 experimental validation points (from literature)

WHAT CAN ACTUALLY WORK with the data we have:
- Pathway classification (RIG-I/TLR/PKR/JAK-STAT/MDA5/NF-kB/cGAS-STING)
  → 3000 labeled samples in sequences_enhanced.csv, 7 classes
- Immunogenicity binary classification (high/low/medium)
  → 3000 labeled samples
- Modification-aware prediction
  → 20 experimental points showing m6A/pseudoU/triphosphate effects

RECOMMENDED PIVOT:
  Focus on what the data actually supports, not what the architecture
  hypothetically could do with perfect data.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_data_issues():
    """Print honest data quality assessment."""
    import pandas as pd
    import numpy as np

    print("=" * 70)
    print("HONEST DATA QUALITY ASSESSMENT")
    print("=" * 70)

    df = pd.read_csv(
        "D:/IGEM集成方案/data/circrna/unified_training_data.csv")

    # Issue 1: Collinear targets
    targets = [c for c in df.columns if c.startswith('target_')
               and c != 'target_predicted_response']
    corr = df[targets].corr()

    print("\n1. TARGET COLLINEARITY")
    print("   8 'independent' targets are actually 1 variable:")
    high_corr_pairs = 0
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            if abs(corr.iloc[i, j]) > 0.95:
                high_corr_pairs += 1
    print(f"   Pairs with |r| > 0.95: {high_corr_pairs} / {len(targets)*(len(targets)-1)//2}")
    print(f"   immunotherapy <-> immunogenicity: r={corr.loc['target_immunotherapy_score', 'target_overall_immunogenicity']:.4f}")
    print(f"   immunotherapy <-> tide:           r={corr.loc['target_immunotherapy_score', 'target_tide_score']:.4f}")

    # Issue 2: Pseudo labels
    print("\n2. PSEUDO-LABEL DOMINANCE")
    real = df[df['source'] == 'real']
    pseudo = df[df['source'] == 'pseudo']
    print(f"   Real: {len(real)} ({100*len(real)/len(df):.1f}%)")
    print(f"   Pseudo: {len(pseudo)} ({100*len(pseudo)/len(df):.1f}%)")
    print(f"   Pseudo rig_i std: {pseudo['target_rig_i_score'].std():.6f} (effectively 0)")
    print(f"   Real rig_i std:   {real['target_rig_i_score'].std():.6f}")

    # Issue 3: Gene expression useless
    print("\n3. GENE EXPRESSION FEATURES")
    gene_cols = [c for c in df.columns if c.startswith('gene_')]
    for g in gene_cols:
        try:
            c = df[g].corr(df['target_overall_immunogenicity'])
            print(f"   {g}: r={c:.6f}")
        except:
            print(f"   {g}: NaN (constant?)")

    # What CAN work
    print("\n" + "=" * 70)
    print("WHAT CAN ACTUALLY WORK")
    print("=" * 70)

    df2 = pd.read_csv(
        "D:/IGEM集成方案/data/circrna/sequences_enhanced.csv")
    print(f"\n1. Pathway Classification (7-class)")
    print(f"   Data: sequences_enhanced.csv ({len(df2)} samples)")
    print(f"   Classes: {df2['pathway'].nunique()} balanced classes")
    print(f"   Baseline accuracy: {1/df2['pathway'].nunique():.2f}")
    print(f"   Target: >70% accuracy (reasonable for this data)")

    print(f"\n2. Immunogenicity Binary Classification")
    print(f"   Data: {len(df2)} samples, {(df2['immunogenicity']==1).sum()} positive")
    print(f"   This is a REAL prediction task with real variance")

    print(f"\n3. Modification Effect Prediction")
    print(f"   Data: 20 experimental points from literature")
    print(f"   Key: m6A → low immunogenicity, Unmodified → high")
    print(f"   This is the most clinically relevant but data-starved")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
    Current TorusFold architecture (CircPairformer + Diffusion) is
    technically sound but applied to fundamentally flawed data.

    Pivot to what works:

    A) PATHWAY CLASSIFICATION (highest ROI)
       - Use sequences_enhanced.csv (3K samples, 7 pathways)
       - Simplified model: ESM2 backbone → MLP classifier
       - Target: 7-class pathway prediction (RIG-I, TLR, PKR, etc.)
       - This is a REAL task with REAL labels

    B) IMMUNOGENICITY PREDICTION (medium ROI)
       - Binary: immunogenic (1800) vs non-immunogenic (1200)
       - Use same ESM2 backbone → sigmoid output
       - Validate on 20 experimental points

    C) MODIFICATION-AWARE PREDICTION (future, needs data)
       - Generate synthetic circRNA sequences with specific modifications
       - Predict IFN-α response based on modification type
       - Requires new experimental data collection

    The CircPairformer + TPE innovations are still valid — they should
    be the backbone for task (A) and (B), but the multi-task head with
    8 collinear composite targets should be replaced with the tasks
    that the data actually supports.
    """)


if __name__ == "__main__":
    analyze_data_issues()
