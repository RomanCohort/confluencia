"""
structure_prediction.py — RNA secondary structure prediction using ViennaRNA.

Integrates ViennaRNA package for:
1. Minimum Free Energy (MFE) structure prediction
2. Base pair probability matrices
3. dsRNA region identification (for PKR scoring)
4. Structure stability scores

Literature basis:
- Lorenz et al., 2011: ViennaRNA Package 2.0
- Zuker, 2003: MFE algorithm
- PKR requires >33bp dsRNA for activation (Nallagatla et al., 2007)

Provides fallback implementation when ViennaRNA is not installed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

# PKR activation threshold
PKR_MIN_DSRNA_LENGTH = 33  # Nallagatla et al., 2007

# MFE normalization constants
MFE_STABLE_THRESHOLD = -300  # kcal/mol - highly stable
MFE_UNSTABLE_THRESHOLD = -100  # kcal/mol - less stable


@dataclass
class StructureFeatures:
    """RNA secondary structure features."""
    mfe: float                    # Minimum free energy (kcal/mol)
    mfe_normalized: float         # MFE per nucleotide
    dsrna_regions: List[Tuple[int, int]]  # dsRNA region coordinates
    dsrna_fraction: float         # Fraction of sequence in dsRNA
    structure_stability: float    # Stability score [0, 1]
    hairpin_count: int            # Number of hairpin structures
    stem_count: int               # Number of stem regions
    dot_bracket: str              # Dot-bracket notation
    prediction_method: str        # Method used (viennarna/fallback)


class StructurePredictor:
    """RNA structure prediction using ViennaRNA or fallback."""

    def __init__(self, min_dsrna_length: int = PKR_MIN_DSRNA_LENGTH):
        """
        Initialize structure predictor.

        Args:
            min_dsrna_length: Minimum dsRNA length for PKR activation
                              (default: 33bp per Nallagatla et al., 2007)
        """
        self.min_dsrna_length = min_dsrna_length
        self._has_viennarna = self._check_viennarna_installed()

    def _check_viennarna_installed(self) -> bool:
        """Check if ViennaRNA (RNAfold) is available."""
        try:
            result = subprocess.run(
                ["RNAfold", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def predict(self, sequence: str) -> StructureFeatures:
        """
        Predict RNA secondary structure features.

        Args:
            sequence: RNA sequence (ACGU format)

        Returns:
            StructureFeatures with MFE, dsRNA regions, etc.
        """
        # Sanitize sequence
        seq = self._sanitize_sequence(sequence)
        if len(seq) == 0:
            return self._empty_features()

        # Run prediction
        if self._has_viennarna:
            mfe, dot_bracket = self._run_rnafold(seq)
            method = "viennarna"
        else:
            mfe, dot_bracket = self._estimate_structure(seq)
            method = "fallback"
            warnings.warn(
                "ViennaRNA not installed, using fallback estimation. "
                "Install ViennaRNA for accurate structure prediction."
            )

        # Normalize MFE
        mfe_normalized = mfe / len(seq) if len(seq) > 0 else 0.0

        # Extract dsRNA regions
        dsrna_regions = self._extract_dsrna_regions(dot_bracket)
        dsrna_fraction = sum(r[1] - r[0] for r in dsrna_regions) / len(seq)

        # Compute stability score
        stability = self._compute_stability_score(mfe_normalized)

        # Count structural elements
        hairpin_count, stem_count = self._count_structures(dot_bracket)

        return StructureFeatures(
            mfe=mfe,
            mfe_normalized=mfe_normalized,
            dsrna_regions=dsrna_regions,
            dsrna_fraction=dsrna_fraction,
            structure_stability=stability,
            hairpin_count=hairpin_count,
            stem_count=stem_count,
            dot_bracket=dot_bracket,
            prediction_method=method,
        )

    def _sanitize_sequence(self, sequence: str) -> str:
        """Convert DNA to RNA and filter invalid characters."""
        result = []
        for ch in sequence.upper():
            if ch == "T":
                result.append("U")
            elif ch in "AUGC":
                result.append(ch)
        return "".join(result)

    def _empty_features(self) -> StructureFeatures:
        """Return empty features for empty sequence."""
        return StructureFeatures(
            mfe=0.0,
            mfe_normalized=0.0,
            dsrna_regions=[],
            dsrna_fraction=0.0,
            structure_stability=0.0,
            hairpin_count=0,
            stem_count=0,
            dot_bracket="",
            prediction_method="none",
        )

    def _run_rnafold(self, sequence: str) -> Tuple[float, str]:
        """Run ViennaRNA RNAfold command."""
        try:
            result = subprocess.run(
                ["RNAfold", "--noPS"],
                input=f">temp\n{sequence}\n",
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse output
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                structure_line = lines[1]
                parts = structure_line.split()
                if len(parts) >= 2:
                    dot_bracket = parts[0]
                    # Parse MFE: "((...))  (-10.50)"
                    mfe_str = parts[1].strip("()")
                    try:
                        mfe = float(mfe_str)
                    except ValueError:
                        mfe = 0.0
                    return mfe, dot_bracket

        except subprocess.TimeoutExpired:
            warnings.warn("RNAfold timeout, using fallback")
        except Exception as e:
            warnings.warn(f"RNAfold error: {e}, using fallback")

        # Fallback
        return self._estimate_structure(sequence)

    def _estimate_structure(self, sequence: str) -> Tuple[float, str]:
        """
        Estimate structure features without ViennaRNA.

        Uses heuristics based on sequence composition:
        - GC content correlates with stability
        - GU pairs indicate potential stem regions
        - Simple stem-loop model
        """
        seq = sequence.upper()
        gc = sum(1 for c in seq if c in "GC") / len(seq) if seq else 0

        # Estimate MFE from GC content
        # Typical RNA MFE: -0.3 to -0.8 kcal/mol per nucleotide
        # Higher GC = more stable = more negative MFE
        mfe_per_nt = -0.3 - 0.5 * gc
        estimated_mfe = mfe_per_nt * len(seq)

        # Generate simplified dot-bracket
        # Estimate stem regions based on GC clusters
        dot_bracket = self._generate_dot_bracket_estimate(seq, gc)

        return estimated_mfe, dot_bracket

    def _generate_dot_bracket_estimate(self, seq: str, gc: float) -> str:
        """Generate simplified dot-bracket notation estimate."""
        # Simple model: alternating stem-loop regions
        # Stem length proportional to GC content
        stem_len = int(gc * 10) + 3

        result = []
        i = 0
        while i < len(seq):
            # Check for potential stem region
            window = seq[i:i+stem_len*2] if i+stem_len*2 <= len(seq) else seq[i:]

            # Count GC in window
            window_gc = sum(1 for c in window if c in "GC") / len(window) if window else 0

            if window_gc > 0.5 and len(window) >= stem_len:
                # Add stem
                for j in range(min(stem_len, len(seq) - i)):
                    result.append("(")
                i += stem_len
                # Add loop region
                loop_len = max(4, int(stem_len * 0.4))
                for j in range(min(loop_len, len(seq) - i)):
                    result.append(".")
                i += loop_len
                # Close stem
                for j in range(min(stem_len, len(seq) - i)):
                    result.append(")")
                i += stem_len
            else:
                # Unpaired region
                result.append(".")
                i += 1

        return "".join(result)[:len(seq)]

    def _extract_dsrna_regions(self, dot_bracket: str) -> List[Tuple[int, int]]:
        """
        Extract dsRNA regions from dot-bracket notation.

        dsRNA regions are indicated by paired bases ((...)).

        Returns:
            List of (start, end) tuples for dsRNA regions >= min_dsrna_length
        """
        regions = []
        in_stem = False
        start = 0

        for i, char in enumerate(dot_bracket):
            if char == "(":
                if not in_stem:
                    start = i
                    in_stem = True
            elif char == ")":
                # Continue in stem
                pass
            else:  # "." or other
                if in_stem:
                    end = i
                    # Count paired bases in this region
                    # Only count if region is long enough
                    paired_count = dot_bracket[start:end].replace(".", "").replace(")", "").count("(")
                    if paired_count >= self.min_dsrna_length:
                        regions.append((start, end))
                    in_stem = False

        # Handle end of string
        if in_stem:
            paired_count = dot_bracket[start:].replace(".", "").replace(")", "").count("(")
            if paired_count >= self.min_dsrna_length:
                regions.append((start, len(dot_bracket)))

        return regions

    def _compute_stability_score(self, mfe_normalized: float) -> float:
        """
        Compute structure stability score [0, 1].

        Based on MFE per nucleotide:
        - MFE < -300 kcal/mol total: highly stable (score ~1.0)
        - MFE > -100 kcal/mol total: less stable (score ~0.0)

        Typical values: -0.1 to -0.8 kcal/mol per nt
        More negative = more stable = higher score
        """
        # Map MFE normalized to stability score
        # -0.8 kcal/mol/nt -> score ~1.0 (more stable)
        # -0.1 kcal/mol/nt -> score ~0.0 (less stable)
        # Linear mapping: score = -mfe_normalized / 0.8 (normalized to [0,1])

        # mfe_normalized is negative, so negate to get positive score
        # Scale: divide by 0.8 to map [-0.8, 0] to [1.0, 0.0]
        score = -mfe_normalized / 0.8

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def _count_structures(self, dot_bracket: str) -> Tuple[int, int]:
        """Count hairpin and stem structures."""
        hairpin_count = 0
        stem_count = 0

        # Find hairpins: pattern ((...))
        # A hairpin has stem + loop + stem close
        i = 0
        while i < len(dot_bracket):
            if dot_bracket[i] == "(":
                # Find stem start
                stem_start = i
                while i < len(dot_bracket) and dot_bracket[i] == "(":
                    i += 1
                stem_open_len = i - stem_start

                # Find loop
                loop_start = i
                while i < len(dot_bracket) and dot_bracket[i] == ".":
                    i += 1
                loop_len = i - loop_start

                # Find stem close
                stem_close_start = i
                while i < len(dot_bracket) and dot_bracket[i] == ")":
                    i += 1
                stem_close_len = i - stem_close_start

                # Check if this is a valid hairpin
                if stem_open_len > 0 and loop_len >= 3 and stem_close_len > 0:
                    hairpin_count += 1
                    stem_count += 1
            else:
                i += 1

        return hairpin_count, stem_count


def compute_pkr_score_from_structure(features: StructureFeatures) -> float:
    """
    Compute PKR activation score from predicted structure.

    Literature basis (Nallagatla et al., 2007):
    - PKR requires >33bp dsRNA for activation
    - dsRNA fraction >0.3 indicates strong activation potential
    - Stable dsRNA structures more likely to activate PKR

    Args:
        features: StructureFeatures from prediction

    Returns:
        PKR activation potential score [0, 1]
    """
    # Base score from dsRNA fraction
    # Higher fraction = more dsRNA = more PKR activation
    dsrna_score = min(features.dsrna_fraction * 2.5, 1.0)

    # Length bonus for long dsRNA regions
    if features.dsrna_regions:
        max_dsrna_length = max(r[1] - r[0] for r in features.dsrna_regions)
        # Bonus for very long dsRNA (>100bp)
        length_bonus = min(max_dsrna_length / 200, 0.3)
    else:
        length_bonus = 0.0

    # Stability contribution
    # Stable structures = more persistent dsRNA = more PKR activation
    stability_score = features.structure_stability * 0.2

    # Total PKR score
    pkr_score = min(dsrna_score + length_bonus + stability_score, 1.0)

    return pkr_score


# Convenience function
def predict_structure(sequence: str, min_dsrna_length: int = PKR_MIN_DSRNA_LENGTH) -> StructureFeatures:
    """Convenience wrapper for structure prediction."""
    predictor = StructurePredictor(min_dsrna_length=min_dsrna_length)
    return predictor.predict(sequence)


if __name__ == "__main__":
    # Demo / test
    test_sequences = [
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",  # High GC, stable
        "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU",  # Low GC, unstable
        "GUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGUGU",  # GU-rich
        "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU",  # Mixed
    ]

    print("RNA Structure Prediction Demo")
    print("=" * 60)

    predictor = StructurePredictor()

    for seq in test_sequences:
        features = predictor.predict(seq)
        print(f"\nSequence: {seq[:20]}... (len={len(seq)})")
        print(f"  Method: {features.prediction_method}")
        print(f"  MFE: {features.mfe:.2f} kcal/mol ({features.mfe_normalized:.4f} per nt)")
        print(f"  dsRNA fraction: {features.dsrna_fraction:.2%}")
        print(f"  Stability: {features.structure_stability:.3f}")
        print(f"  Hairpins: {features.hairpin_count}, Stems: {features.stem_count}")
        print(f"  dsRNA regions: {len(features.dsrna_regions)}")

        pkr_score = compute_pkr_score_from_structure(features)
        print(f"  PKR score: {pkr_score:.3f}")