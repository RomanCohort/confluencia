"""
Merge all circRNA training data into unified format.

Combines:
- Real labeled data (2,000 sequences)
- Pseudo-labeled circBase (50,000 sequences)

Output: unified training CSV for encoder training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


DEFAULT_GENE_COLS = ["TROP2", "NECTIN4", "LIV-1", "B7-H4", "MKI67", "MYC"]


def main():
    parser = argparse.ArgumentParser(description="Merge circRNA training data")
    parser.add_argument("--output", default="data/circrna/unified_training_data.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("Merging circRNA Training Data")
    print("=" * 60)

    # 1. Load real labeled data
    print("\n[1] Loading real labeled data...")
    sequences_df = pd.read_csv(_PROJECT_ROOT / "data/circrna/sequences.csv")
    labels_df = pd.read_csv(_PROJECT_ROOT / "data/circrna/labels.csv")

    real_df = sequences_df.merge(labels_df, on="circrna_id", how="inner")
    print(f"    Real labeled: {len(real_df)} sequences")

    # 2. Load pseudo-labeled data
    print("\n[2] Loading pseudo-labeled data...")
    pseudo_df = pd.read_csv(_PROJECT_ROOT / "data/circrna/circbase_pseudo_labels_full.csv")
    print(f"    Pseudo labeled: {len(pseudo_df)} sequences")

    # 3. Load gene expression defaults
    print("\n[3] Loading gene expression defaults...")
    survival_df = pd.read_csv(_PROJECT_ROOT / "data/gene_signature/cache/combined_raw_with_survival.csv")

    gene_medians = {}
    for g in DEFAULT_GENE_COLS:
        if g in survival_df.columns:
            gene_medians[g] = survival_df[g].median()
        else:
            gene_medians[g] = 5.0  # Default

    print(f"    Gene medians: {gene_medians}")

    # 4. Build unified training format
    print("\n[4] Building unified format...")

    # Process real data
    real_records = []
    for idx, row in real_df.iterrows():
        record = {
            'circrna_id': row['circrna_id'],
            'sequence': row['sequence'],
            'seq_length': len(row['sequence']),
            'source': 'real',
            'host_gene': row.get('host_gene_name', 'unknown'),
        }

        # Gene expression
        for g in DEFAULT_GENE_COLS:
            record[f'gene_{g}'] = gene_medians.get(g, 5.0)

        # Targets from real labels
        imm_score = row.get('immune_score', 0.5)
        record['target_immunotherapy_score'] = imm_score
        record['target_tumor_killing_index'] = imm_score * 0.8
        record['target_overall_immunogenicity'] = imm_score
        record['target_immune_cycle_score'] = imm_score * 0.6
        record['target_tme_score'] = imm_score * 0.5
        record['target_therapeutic_window'] = imm_score * 0.7
        record['target_tide_score'] = 1.0 - imm_score
        record['target_ips'] = imm_score * 10.0

        # Report scores
        seq = row['sequence']
        gc = sum(1 for c in seq.upper() if c in "GC") / len(seq)
        u_ratio = sum(1 for c in seq.upper() if c == "U") / len(seq)
        record['target_rig_i_score'] = min(0.3 + gc * 0.3, 1.0)
        record['target_tlr_score'] = min(u_ratio * 2.0 + 0.1, 1.0)
        record['target_pkr_score'] = min(gc * 0.5 + 0.2, 1.0) * 0.85
        record['target_trained_model_risk'] = 1.0 - imm_score

        # Response class
        imm_class = row.get('immunogenicity_class', 'Medium')
        if imm_class == 'High':
            record['target_predicted_response'] = 'likely_responder'
        elif imm_class == 'Low':
            record['target_predicted_response'] = 'likely_non_responder'
        else:
            record['target_predicted_response'] = 'intermediate'

        record['orig_immunogenicity'] = row.get('immunogenicity', 0)
        record['orig_immune_score'] = imm_score
        record['orig_class'] = imm_class

        real_records.append(record)

    # Process pseudo data
    pseudo_records = []
    for idx, row in pseudo_df.iterrows():
        pseudo_score = row['pseudo_immuno_score']
        pseudo_label = row['pseudo_immunogenicity']

        record = {
            'circrna_id': row['circrna_id'],
            'sequence': row['sequence'],
            'seq_length': len(row['sequence']),
            'source': 'pseudo',
            'host_gene': row.get('gene', 'unknown'),
        }

        # Gene expression
        for g in DEFAULT_GENE_COLS:
            record[f'gene_{g}'] = gene_medians.get(g, 5.0)

        # Targets from pseudo labels
        record['target_immunotherapy_score'] = pseudo_score
        record['target_tumor_killing_index'] = pseudo_score * 0.75
        record['target_overall_immunogenicity'] = pseudo_score
        record['target_immune_cycle_score'] = pseudo_score * 0.55
        record['target_tme_score'] = pseudo_score * 0.45
        record['target_therapeutic_window'] = pseudo_score * 0.65
        record['target_tide_score'] = 1.0 - pseudo_score
        record['target_ips'] = pseudo_score * 10.0

        # Report scores
        gc = row.get('gc_content', 0.5)
        entropy = row.get('entropy', 1.5)
        record['target_rig_i_score'] = min(0.3 + gc * 0.25, 1.0)
        record['target_tlr_score'] = min(entropy / 2.0, 1.0)
        record['target_pkr_score'] = min(gc * 0.4 + 0.2, 1.0) * 0.85
        record['target_trained_model_risk'] = 1.0 - pseudo_score

        # Response class
        pseudo_class = row.get('pseudo_class', 'Medium')
        if pseudo_class == 'High':
            record['target_predicted_response'] = 'likely_responder'
        elif pseudo_class == 'Low':
            record['target_predicted_response'] = 'likely_non_responder'
        else:
            record['target_predicted_response'] = 'intermediate'

        record['orig_immunogenicity'] = pseudo_label
        record['orig_immune_score'] = pseudo_score
        record['orig_class'] = pseudo_class

        pseudo_records.append(record)

    # Merge
    all_records = real_records + pseudo_records
    unified_df = pd.DataFrame(all_records)

    # Shuffle
    unified_df = unified_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path = _PROJECT_ROOT / args.output

    unified_df.to_csv(output_path, index=False)

    # Statistics
    print(f"\n[5] Statistics:")
    print(f"    Total: {len(unified_df)}")
    print(f"    Real: {len(real_records)}")
    print(f"    Pseudo: {len(pseudo_records)}")

    print(f"\n    By source:")
    print(f"    {unified_df['source'].value_counts().to_dict()}")

    print(f"\n    By class:")
    print(f"    {unified_df['orig_class'].value_counts().to_dict()}")

    print(f"\n    By immunogenicity:")
    print(f"    0={int((unified_df['orig_immunogenicity']==0).sum())}")
    print(f"    1={int((unified_df['orig_immunogenicity']==1).sum())}")

    print(f"\n✓ Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()