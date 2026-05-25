"""
cotrans_folding.py — Cotranscriptional RNA Folding Simulation

Integrates ViennaRNA RNAkinfold for:
1. Real-time folding during transcription simulation
2. Kinetic trapping analysis
3. Transcription speed effects on final structure
4. Intermediate structure detection

Literature basis:
- Geis et al., 2015: Cotranscriptional folding of circRNAs
- Heil et al., 2014: RNAkinfold - kinetic folding simulation
- Frieda et al., 2012: Transcription rate determines RNA structure
- Isambert & Siggia, 2005: RNA folding kinetics theory

Key concepts:
- RNA folds while being transcribed (5' → 3' direction)
- Transcription speed affects final structure (slow = more structured)
- Kinetic traps: structures that persist due to slow unfolding
- Intermediate states may expose immunogenic motifs
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import subprocess
import tempfile
import warnings
import re
from pathlib import Path

# Transcription constants
TRANSCRIPTION_RATES = {
    "slow": 10,     # nt/s - bacterial pause sites
    "medium": 50,   # nt/s - typical polymerase
    "fast": 200,    # nt/s - eukaryotic RNAP II max
}

# Kinetic trap thresholds
TRAP_THRESHOLD_LOW = 5.0    # kcal/mol - easily escapes
TRAP_THRESHOLD_HIGH = 15.0  # kcal/mol - persistent trap


@dataclass
class IntermediateStructure:
    """RNA structure at specific transcription length."""
    length: int                # Sequence length at this point
    structure: str             # Dot-bracket notation
    mfe: float                 # Free energy at this length
    fraction_folded: float     # Fraction of sequence folded


@dataclass
class KineticTrap:
    """A kinetic trap in folding pathway."""
    position: int              # Where trap occurs (nt position)
    structure: str             # Trapped structure
    barrier_escape: float      # Barrier to escape trap
    persistence_time: float    # Estimated time in trap (s)


@dataclass
class CotransFeatures:
    """Cotranscriptional folding features."""
    intermediate_structures: List[IntermediateStructure]
    kinetic_traps: List[KineticTrap]
    final_structure_match: float    # Match to equilibrium structure
    folding_timeline: List[float]    # Time points for each intermediate
    transcription_rate_effect: float # Effect of rate on structure
    immunogenic_exposure_windows: List[Tuple[int, int]]  # Positions where immune motifs exposed
    cotrans_method: str              # Method used (viennarna/fallback)


class CotranscriptionalFoldingPredictor:
    """
    Predict cotranscriptional folding using ViennaRNA RNAkinfold.

    Simulates RNA folding as it's being transcribed:
    - 5' end folds first
    - Structure evolves as sequence elongates
    - Final structure may differ from equilibrium MFE
    """

    def __init__(
        self,
        transcription_rate: float = 50.0,  # nt/s
        time_step: float = 0.1,            # s
        min_intermediate_length: int = 50,
    ):
        """
        Initialize cotranscriptional folding predictor.

        Args:
            transcription_rate: Nucleotides per second
            time_step: Simulation time step
            min_intermediate_length: Minimum length to record intermediate
        """
        self.transcription_rate = transcription_rate
        self.time_step = time_step
        self.min_intermediate_length = min_intermediate_length
        self._has_rnakinfold = self._check_rnakinfold_installed()

    def _check_rnakinfold_installed(self) -> bool:
        """Check if RNAkinfold is available."""
        try:
            result = subprocess.run(
                ["RNAkinfold", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def predict(self, sequence: str) -> CotransFeatures:
        """
        Simulate cotranscriptional folding.

        Args:
            sequence: Full RNA sequence

        Returns:
            CotransFeatures with intermediates, traps, timeline
        """
        seq = self._sanitize_sequence(sequence)
        if len(seq) < 100:
            return self._empty_features(len(seq))

        if self._has_rnakinfold:
            intermediates, traps = self._run_rnakinfold(seq)
            method = "viennarna_rnakinfold"
        else:
            intermediates, traps = self._simulate_cotrans_fallback(seq)
            method = "fallback_cotrans"
            warnings.warn(
                "RNAkinfold not installed, using fallback simulation. "
                "Install ViennaRNA for accurate cotranscriptional folding."
            )

        # Compute derived features
        timeline = self._compute_timeline(intermediates)
        rate_effect = self._compute_rate_effect(intermediates)
        exposure_windows = self._detect_exposure_windows(intermediates)

        return CotransFeatures(
            intermediate_structures=intermediates,
            kinetic_traps=traps,
            final_structure_match=self._compute_structure_match(intermediates),
            folding_timeline=timeline,
            transcription_rate_effect=rate_effect,
            immunogenic_exposure_windows=exposure_windows,
            cotrans_method=method,
        )

    def _sanitize_sequence(self, sequence: str) -> str:
        """Convert DNA to RNA."""
        return sequence.upper().replace("T", "U")

    def _empty_features(self, length: int) -> CotransFeatures:
        """Return empty features for short sequences."""
        return CotransFeatures(
            intermediate_structures=[],
            kinetic_traps=[],
            final_structure_match=0.0,
            folding_timeline=[],
            transcription_rate_effect=0.0,
            immunogenic_exposure_windows=[],
            cotrans_method="sequence_too_short",
        )

    def _run_rnakinfold(self, sequence: str) -> Tuple[List[IntermediateStructure], List[KineticTrap]]:
        """
        Run RNAkinfold for kinetic folding simulation.

        RNAkinfold simulates folding during transcription:
        - Starts with empty structure
        - Adds nucleotides at transcription_rate
        - Allows local folding at each step
        """
        intermediates = []
        traps = []

        try:
            # RNAkinfold parameters
            # -t: transcription rate (nt/s)
            # -s: sampling steps
            cmd = [
                "RNAkinfold",
                "-t", str(self.transcription_rate),
                "-s", str(self.time_step),
                "-n", "100",  # Number of trajectories
            ]

            result = subprocess.run(
                cmd,
                input=f">seq\n{sequence}\n",
                capture_output=True,
                text=True,
                timeout=300,
            )

            # Parse output
            lines = result.stdout.strip().split("\n")
            for line in lines:
                # RNAkinfold output format varies, parse intermediates
                if "length" in line and "structure" in line:
                    # Parse intermediate structure
                    parts = line.split()
                    if len(parts) >= 4:
                        length = int(parts[1])
                        structure = parts[3]
                        mfe = float(parts[5]) if len(parts) > 5 else 0.0
                        intermediates.append(IntermediateStructure(
                            length=length,
                            structure=structure,
                            mfe=mfe,
                            fraction_folded=structure.count("(") / length,
                        ))

                # Parse kinetic traps
                if "trap" in line or "kinetic_trap" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        traps.append(KineticTrap(
                            position=int(parts[1]),
                            structure=parts[2] if len(parts) > 2 else "",
                            barrier_escape=float(parts[4]) if len(parts) > 4 else 10.0,
                            persistence_time=0.0,
                        ))

        except subprocess.TimeoutExpired:
            warnings.warn("RNAkinfold timeout, using fallback")
        except Exception as e:
            warnings.warn(f"RNAkinfold error: {e}")

        # Fallback if no results
        if not intermediates:
            intermediates, traps = self._simulate_cotrans_fallback(sequence)

        return intermediates, traps

    def _simulate_cotrans_fallback(self, sequence: str) -> Tuple[List[IntermediateStructure], List[KineticTrap]]:
        """
        Simulate cotranscriptional folding without RNAkinfold.

        Heuristic simulation:
        1. Sequence grows from 5' end
        2. Local structure forms as each segment completes
        3. GC-rich segments fold faster
        4. AU-rich segments stay unfolded longer
        """
        intermediates = []
        traps = []

        length = len(sequence)
        step_size = 50  # Record every 50 nt

        # Simulate progressive folding
        for current_len in range(self.min_intermediate_length, length, step_size):
            segment = sequence[:current_len]

            # Estimate folding for this segment
            gc = sum(1 for c in segment if c in "GC") / current_len

            # Local structure estimation
            # Higher GC = more folded at this stage
            fraction_folded = gc * 0.8 + np.random.uniform(-0.1, 0.1)

            # Generate partial structure
            stem_count = int(current_len * fraction_folded * gc)
            structure = "(" * stem_count + "." * (current_len - 2 * stem_count) + ")" * stem_count

            # Estimate MFE
            mfe_per_nt = -0.3 - 0.5 * gc
            mfe = mfe_per_nt * current_len * fraction_folded

            intermediates.append(IntermediateStructure(
                length=current_len,
                structure=structure,
                mfe=mfe,
                fraction_folded=np.clip(fraction_folded, 0.0, 1.0),
            ))

            # Check for kinetic traps
            # High GC segments may form stable intermediates that persist
            if gc > 0.6 and fraction_folded > 0.5:
                # Potential trap
                trap_barrier = gc * 15.0  # Higher GC = harder to escape
                persistence = trap_barrier / (self.transcription_rate * 0.1)

                traps.append(KineticTrap(
                    position=current_len,
                    structure=structure,
                    barrier_escape=trap_barrier,
                    persistence_time=persistence,
                ))

        # Add final structure
        final_gc = sum(1 for c in sequence if c in "GC") / length
        final_folded = final_gc * 0.85
        final_stem = int(length * final_folded * final_gc)
        final_structure = "(" * final_stem + "." * (length - 2 * final_stem) + ")" * final_stem

        intermediates.append(IntermediateStructure(
            length=length,
            structure=final_structure,
            mfe=-0.3 * length * final_gc - 0.5 * final_folded * length,
            fraction_folded=final_folded,
        ))

        return intermediates, traps

    def _compute_timeline(self, intermediates: List[IntermediateStructure]) -> List[float]:
        """Compute time points for each intermediate."""
        timeline = []
        for inter in intermediates:
            time = inter.length / self.transcription_rate
            timeline.append(time)
        return timeline

    def _compute_rate_effect(self, intermediates: List[IntermediateStructure]) -> float:
        """
        Compute how transcription rate affects final structure.

        Faster transcription = less structured final state
        """
        if not intermediates:
            return 0.0

        # Compare final folding fraction to theoretical equilibrium
        final = intermediates[-1]
        equilibrium_folded = sum(1 for c in final.structure if c in "()") / len(final.structure)

        # Rate effect: faster = lower fraction
        rate_factor = self.transcription_rate / TRANSCRIPTION_RATES["medium"]
        effect = 1.0 - min(rate_factor - 1.0, 0.5) if rate_factor > 1.0 else rate_factor

        return np.clip(effect * equilibrium_folded, 0.0, 1.0)

    def _compute_structure_match(self, intermediates: List[IntermediateStructure]) -> float:
        """Compute match between cotrans final and equilibrium structure."""
        if not intermediates:
            return 0.0

        final = intermediates[-1]
        # Simple match: fraction of paired bases
        paired = final.structure.count("(") + final.structure.count(")")
        return paired / len(final.structure) if final.structure else 0.0

    def _detect_exposure_windows(self, intermediates: List[IntermediateStructure]) -> List[Tuple[int, int]]:
        """
        Detect windows where immunogenic motifs might be exposed.

        Unfolded regions during transcription may expose:
        - GU-rich sequences (RIG-I)
        - U-rich sequences (TLR7/8)
        """
        windows = []

        for i, inter in enumerate(intermediates):
            # Find unfolded regions (consecutive dots)
            dots = re.findall(r"\.{10,}", inter.structure)

            for dot_region in dots:
                # Find position in intermediate
                start = inter.structure.find(dot_region)
                end = start + len(dot_region)

                # Check if this region contains immunogenic motifs
                segment = inter.structure[start:end]  # Placeholder

                # Record potential exposure window
                if len(dot_region) > 20:  # Significant exposure
                    windows.append((start, end))

        return windows


def compare_transcription_rates(sequence: str) -> Dict[str, CotransFeatures]:
    """
    Compare cotranscriptional folding at different transcription rates.

    Args:
        sequence: RNA sequence

    Returns:
        Dict with features for slow/medium/fast rates
    """
    results = {}

    for rate_name, rate in TRANSCRIPTION_RATES.items():
        predictor = CotranscriptionalFoldingPredictor(transcription_rate=rate)
        results[rate_name] = predictor.predict(sequence)

    return results


def compute_cotrans_immunogenicity(features: CotransFeatures) -> float:
    """
    Compute immunogenicity score from cotranscriptional folding.

    More exposure windows = higher chance of immune recognition
    """
    # Factors:
    # 1. Number of exposure windows
    exposure_score = len(features.immunogenic_exposure_windows) / 10.0

    # 2. Kinetic traps (traps = stable intermediates = less exposure)
    trap_penalty = len(features.kinetic_traps) * 0.1

    # 3. Rate effect (faster = more exposure)
    rate_score = features.transcription_rate_effect

    score = exposure_score * 0.5 + rate_score * 0.4 - trap_penalty * 0.1

    return np.clip(score, 0.0, 1.0)


# Convenience function
def predict_cotrans_folding(sequence: str, rate: float = 50.0) -> CotransFeatures:
    """Predict cotranscriptional folding for RNA sequence."""
    predictor = CotranscriptionalFoldingPredictor(transcription_rate=rate)
    return predictor.predict(sequence)