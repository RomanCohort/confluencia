"""
Generate synthetic validation benchmark for circRNA encoder.

Creates 100 sequences with known properties:
- High immunogenicity (GC > 0.55, GU-rich, dsRNA potential)
- Low immunogenicity (GC < 0.35, A-rich, minimal structure)
- Medium immunogenicity (balanced composition)
- Known IRES motifs (EMCV, HCV-like)
- BSJ flanking regions with Alu-like elements

Usage:
    python scripts/generate_validation_benchmark.py \
        --output data/validation/synthetic_benchmark.csv
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


@dataclass
class BenchmarkSequence:
    """A synthetic benchmark sequence with known properties."""
    sequence: str
    label: str  # high/medium/low
    gc_content: float
    expected_rig_i: float
    expected_tlr: float
    expected_pkr: float
    has_ires: bool
    ires_type: str
    has_bsj_alu: bool
    description: str


# Literature-based IRES consensus sequences (simplified)
IRES_SEQUENCES = {
    # EMCV IRES domain 2-3 stem-loop (simplified)
    "emcv": "GCGGCCGCUUGGGCCCUGGGGGCGGUCCGGGCGGCCAGGAACCGGGCGCAG",
    # HCV IRES domain II stem-loop (simplified)
    "hcv": "GGCGGAGGAAUUUCCGGUGCGGGGAGCGCCUGUGGUGGCGGGCCCGGGCG",
    # Cellular c-myc IRES (simplified stem-loop)
    "cmyc": "GGGGCGGGGCGGGGCGGGGCGGGGCGGGGCUUUUUCCG",
    # VEGF IRES (AU-rich with stem)
    "vegf": "GCGCGCGCGCAGAGAGAUGUUUUUUUUUUGCGCGCGCGC",
}

# Alu consensus fragments (simplified for detection testing)
ALU_FRAGMENTS = [
    "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCAC",
    "GAGGTCAGGAGATCGAGACCATCCTGGCTAACATGGTGAAACCCCGTCTCTACTAAAAATACAA",
    "AAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTCGGGAGGCTGAGGCAGGA",
]


def generate_high_immunogenicity_seq(length: int = 500) -> str:
    """
    Generate high immunogenicity sequence.

    Features:
    - GC content > 0.55
    - GU-rich motifs (RIG-I activation)
    - dsRNA potential via GC pairing
    - CCUCC motifs for RIG-I
    """
    # GC-rich backbone
    gc_rich = "".join(random.choices("GCGC", k=int(length * 0.6)))

    # Add RIG-I motifs
    motifs = ["CCUCC", "GCUCC", "UCUCC"]
    for motif in motifs:
        pos = random.randint(0, len(gc_rich) - 5)
        gc_rich = gc_rich[:pos] + motif + gc_rich[pos+5:]

    # Add GU-rich regions for blunt end potential
    gu_region = "GUGUGUGU" * 10
    gc_rich = gc_rich[:len(gc_rich)//2] + gu_region + gc_rich[len(gc_rich)//2:]

    # Trim to length
    return gc_rich[:length]


def generate_low_immunogenicity_seq(length: int = 500) -> str:
    """
    Generate low immunogenicity sequence.

    Features:
    - GC content < 0.35
    - A/U-rich (minimal structure)
    - No RIG-I motifs
    - Poly-A tracts (overhang indicators)
    """
    # A/U-rich backbone
    au_rich = "".join(random.choices("AUAU", k=int(length * 0.7)))

    # Add poly-A tracts
    poly_a = "AAAAA" * 5
    positions = [100, 200, 300, 400]
    for pos in positions:
        if pos < len(au_rich):
            au_rich = au_rich[:pos] + poly_a + au_rich[pos+25:]

    return au_rich[:length]


def generate_medium_immunogenicity_seq(length: int = 500) -> str:
    """
    Generate medium immunogenicity sequence.

    Features:
    - GC content ~ 0.45
    - Mixed composition
    - Some GU motifs
    """
    # Balanced backbone
    mixed = "".join(random.choices("ACGU", k=length))

    # Add some GU motifs
    gu_region = "GUGU" * 5
    pos = length // 2
    mixed = mixed[:pos] + gu_region + mixed[pos+20:]

    return mixed


def generate_ires_sequence(ires_type: str = "emcv") -> str:
    """
    Generate sequence with known IRES motif.

    Args:
        ires_type: Type of IRES (emcv, hcv, cmyc, vegf)
    """
    ires_core = IRES_SEQUENCES.get(ires_type, IRES_SEQUENCES["emcv"])

    # Flank with moderate GC content
    flank_5 = generate_medium_immunogenicity_seq(100)
    flank_3 = generate_medium_immunogenicity_seq(100)

    return flank_5 + ires_core + flank_3


def generate_bsj_sequence(length: int = 500) -> str:
    """
    Generate sequence with BSJ flanking Alu-like elements.

    Simulates circRNA with back-splice junction and
    flanking intron complementarity.
    """
    # Core exon
    exon = generate_medium_immunogenicity_seq(int(length * 0.6))

    # 5' flanking Alu
    alu_5 = random.choice(ALU_FRAGMENTS)[:50]

    # 3' flanking Alu (reverse complement-like)
    alu_3 = random.choice(ALU_FRAGMENTS)[:50]

    return alu_5 + exon + alu_3


def compute_gc_content(seq: str) -> float:
    """Compute GC content."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def estimate_rig_i_score(seq: str) -> float:
    """Estimate RIG-I score based on features."""
    gc = compute_gc_content(seq)
    gu_count = seq.count("GU") + seq.count("UG")
    gu_score = min(gu_count / (len(seq) / 10), 1.0)

    motif_count = sum(seq.count(m) for m in ["CCUCC", "GCUCC", "UCUCC", "ACUCC"])
    motif_score = min(motif_count * 0.1, 1.0)

    return 0.4 * gc + 0.3 * gu_score + 0.3 * motif_score


def estimate_tlr_score(seq: str) -> float:
    """Estimate TLR7/8 score based on features."""
    u_count = seq.count("U")
    u_ratio = u_count / len(seq) if seq else 0

    au_count = seq.count("AU") + seq.count("UA")
    au_score = min(au_count / (len(seq) / 5), 1.0)

    return 0.6 * min(u_ratio * 2, 1.0) + 0.4 * au_score


def estimate_pkr_score(seq: str) -> float:
    """Estimate PKR score based on features."""
    gc = compute_gc_content(seq)

    # dsRNA potential from GC pairing
    gc_pairs = seq.count("GC") + seq.count("CG")
    pair_score = min(gc_pairs / (len(seq) / 10), 1.0)

    return 0.5 * gc + 0.5 * pair_score


def generate_benchmark(n_per_category: int = 20) -> List[BenchmarkSequence]:
    """Generate complete benchmark dataset."""
    sequences = []

    # High immunogenicity
    for i in range(n_per_category):
        seq = generate_high_immunogenicity_seq(random.randint(400, 600))
        sequences.append(BenchmarkSequence(
            sequence=seq,
            label="high",
            gc_content=compute_gc_content(seq),
            expected_rig_i=estimate_rig_i_score(seq),
            expected_tlr=estimate_tlr_score(seq),
            expected_pkr=estimate_pkr_score(seq),
            has_ires=False,
            ires_type="none",
            has_bsj_alu=False,
            description=f"High immunogenicity sequence {i+1}",
        ))

    # Low immunogenicity
    for i in range(n_per_category):
        seq = generate_low_immunogenicity_seq(random.randint(400, 600))
        sequences.append(BenchmarkSequence(
            sequence=seq,
            label="low",
            gc_content=compute_gc_content(seq),
            expected_rig_i=estimate_rig_i_score(seq),
            expected_tlr=estimate_tlr_score(seq),
            expected_pkr=estimate_pkr_score(seq),
            has_ires=False,
            ires_type="none",
            has_bsj_alu=False,
            description=f"Low immunogenicity sequence {i+1}",
        ))

    # Medium immunogenicity
    for i in range(n_per_category):
        seq = generate_medium_immunogenicity_seq(random.randint(400, 600))
        sequences.append(BenchmarkSequence(
            sequence=seq,
            label="medium",
            gc_content=compute_gc_content(seq),
            expected_rig_i=estimate_rig_i_score(seq),
            expected_tlr=estimate_tlr_score(seq),
            expected_pkr=estimate_pkr_score(seq),
            has_ires=False,
            ires_type="none",
            has_bsj_alu=False,
            description=f"Medium immunogenicity sequence {i+1}",
        ))

    # IRES-containing sequences
    for i, ires_type in enumerate(["emcv", "hcv", "cmyc", "vegf"]):
        for j in range(5):
            seq = generate_ires_sequence(ires_type)
            sequences.append(BenchmarkSequence(
                sequence=seq,
                label="medium",
                gc_content=compute_gc_content(seq),
                expected_rig_i=estimate_rig_i_score(seq),
                expected_tlr=estimate_tlr_score(seq),
                expected_pkr=estimate_pkr_score(seq),
                has_ires=True,
                ires_type=ires_type,
                has_bsj_alu=False,
                description=f"IRES ({ires_type}) sequence {j+1}",
            ))

    # BSJ sequences
    for i in range(n_per_category):
        seq = generate_bsj_sequence(random.randint(400, 600))
        sequences.append(BenchmarkSequence(
            sequence=seq,
            label="medium",
            gc_content=compute_gc_content(seq),
            expected_rig_i=estimate_rig_i_score(seq),
            expected_tlr=estimate_tlr_score(seq),
            expected_pkr=estimate_pkr_score(seq),
            has_ires=False,
            ires_type="none",
            has_bsj_alu=True,
            description=f"BSJ (Alu-flanked) sequence {i+1}",
        ))

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Generate validation benchmark")
    parser.add_argument("--output", type=str, default="data/validation/synthetic_benchmark.csv",
                        help="Output CSV path")
    parser.add_argument("--n-per-category", type=int, default=20,
                        help="Number of sequences per category")
    args = parser.parse_args()

    print("Generating validation benchmark...")

    sequences = generate_benchmark(args.n_per_category)

    # Convert to DataFrame
    records = []
    for i, s in enumerate(sequences):
        records.append({
            "sample_id": i,
            "sequence": s.sequence,
            "seq_length": len(s.sequence),
            "label": s.label,
            "gc_content": s.gc_content,
            "expected_rig_i": s.expected_rig_i,
            "expected_tlr": s.expected_tlr,
            "expected_pkr": s.expected_pkr,
            "has_ires": s.has_ires,
            "ires_type": s.ires_type,
            "has_bsj_alu": s.has_bsj_alu,
            "description": s.description,
        })

    df = pd.DataFrame(records)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} sequences")
    print(f"Label distribution:")
    print(df["label"].value_counts())
    print(f"\nIRES sequences: {df['has_ires'].sum()}")
    print(f"BSJ sequences: {df['has_bsj_alu'].sum()}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
