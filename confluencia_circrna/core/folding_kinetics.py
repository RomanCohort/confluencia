"""
folding_kinetics.py — RNA Folding Kinetics Prediction

Integrates ViennaRNA kinetics tools for:
1. Suboptimal structure landscape (RNAsubopt)
2. Energy barrier analysis (barriers)
3. Folding rate estimation
4. Cotranscriptional folding potential
5. Structure stability dynamics

Literature basis:
- Isambert & Siggia, 2005: RNA folding kinetics and rate constants
- Geis et al., 2015: Cotranscriptional folding of circRNAs
- Lorenz et al., 2011: ViennaRNA Package 2.0 kinetics tools
- Flamm et al., 2002: barriers and folding pathways

Provides fallback estimation when ViennaRNA kinetics tools unavailable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import subprocess
import tempfile
import warnings
import re
from pathlib import Path

# Physical constants
RT_37C = 0.616  # kcal/mol at 37°C (310K)
K0_FOLDING = 1e6  # Base folding rate constant (s^-1)

# Kinetics thresholds
BARRIER_STABLE = 5.0    # kcal/mol - stable structure
BARRIER_UNSTABLE = 15.0 # kcal/mol - easily refolds
SUBOPTIMAL_DELTA = 10.0 # kcal/mol - energy window for suboptimal sampling


@dataclass
class SuboptimalStructure:
    """A suboptimal RNA structure."""
    mfe: float                # Free energy (kcal/mol)
    dot_bracket: str          # Structure representation
    energy_delta: float       # Energy difference from optimal
    probability: float        # Estimated Boltzmann probability


@dataclass
class KineticsFeatures:
    """RNA folding kinetics features."""
    folding_rate: float                     # Estimated folding rate (s^-1)
    barrier_height: float                   # Energy barrier to native (kcal/mol)
    metastable_count: int                   # Number of metastable states
    landscape_complexity: float             # Complexity score [0, 1]
    cotrans_folding_score: float            # Cotranscriptional potential [0, 1]
    stability_dynamic: float                # Dynamic stability score [0, 1]
    suboptimal_structures: List[SuboptimalStructure]  # Top suboptimal structures
    native_structure: Optional[SuboptimalStructure]   # Native (MFE) structure
    kinetics_method: str                    # Method used (viennarna/fallback)


class FoldingKineticsPredictor:
    """
    RNA folding kinetics prediction using ViennaRNA tools.

    Tools used:
    - RNAsubopt: Sample suboptimal structures
    - barriers: Energy landscape and folding pathways (if available)
    - Rate estimation: Arrhenius-type kinetics model
    """

    def __init__(
        self,
        suboptimal_energy_window: float = SUBOPTIMAL_DELTA,
        max_suboptimal: int = 100,
    ):
        """
        Initialize kinetics predictor.

        Args:
            suboptimal_energy_window: Energy range for suboptimal sampling
            max_suboptimal: Maximum number of suboptimal structures
        """
        self.energy_window = suboptimal_energy_window
        self.max_suboptimal = max_suboptimal
        self._has_viennarna = self._check_viennarna_installed()
        self._has_barriers = self._check_barriers_installed()

    def _check_viennarna_installed(self) -> bool:
        """Check if ViennaRNA (RNAsubopt) is available."""
        try:
            result = subprocess.run(
                ["RNAsubopt", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_barriers_installed(self) -> bool:
        """Check if barriers tool is available."""
        try:
            result = subprocess.run(
                ["barriers", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def predict(self, sequence: str) -> KineticsFeatures:
        """
        Predict folding kinetics features for RNA sequence.

        Args:
            sequence: RNA sequence (ACGU format)

        Returns:
            KineticsFeatures with folding rates, barriers, suboptimal structures
        """
        # Sanitize sequence
        seq = self._sanitize_sequence(sequence)
        if len(seq) < 50:
            return self._empty_features(len(seq))

        if self._has_viennarna:
            # Use ViennaRNA kinetics tools
            suboptimal = self._run_rnasubopt(seq)
            barrier = self._run_barriers(seq) if self._has_barriers else None
            kinetics_method = "viennarna_kinetics"
        else:
            # Fallback estimation
            suboptimal = self._estimate_suboptimal(seq)
            barrier = self._estimate_barrier(seq)
            kinetics_method = "fallback_kinetics"
            warnings.warn(
                "ViennaRNA kinetics tools not installed. "
                "Install ViennaRNA for accurate kinetics prediction."
            )

        # Compute kinetics parameters
        folding_rate = self._compute_folding_rate(seq, barrier)
        landscape = self._compute_landscape_complexity(suboptimal)
        cotrans_score = self._compute_cotrans_score(seq, suboptimal)
        dynamic_stability = self._compute_dynamic_stability(seq, barrier, landscape)

        # Extract native structure (MFE)
        native = suboptimal[0] if suboptimal else None

        return KineticsFeatures(
            folding_rate=folding_rate,
            barrier_height=barrier if barrier else 0.0,
            metastable_count=len(suboptimal) - 1,
            landscape_complexity=landscape,
            cotrans_folding_score=cotrans_score,
            stability_dynamic=dynamic_stability,
            suboptimal_structures=suboptimal[:self.max_suboptimal],
            native_structure=native,
            kinetics_method=kinetics_method,
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

    def _empty_features(self, length: int) -> KineticsFeatures:
        """Return empty features for short sequences."""
        return KineticsFeatures(
            folding_rate=0.0,
            barrier_height=0.0,
            metastable_count=0,
            landscape_complexity=0.0,
            cotrans_folding_score=0.0,
            stability_dynamic=0.0,
            suboptimal_structures=[],
            native_structure=None,
            kinetics_method="sequence_too_short",
        )

    def _run_rnasubopt(self, sequence: str) -> List[SuboptimalStructure]:
        """
        Run RNAsubopt to sample suboptimal structures.

        RNAsubopt -e <energy> samples structures within energy window.
        Output format:
            .((...))  -10.5
            ((...))   -9.2
            ...
        """
        structures = []

        try:
            result = subprocess.run(
                ["RNAsubopt", "-e", str(self.energy_window), "-s"],
                input=f">seq\n{sequence}\n",
                capture_output=True,
                text=True,
                timeout=120,
            )

            lines = result.stdout.strip().split("\n")
            for line in lines:
                if line.startswith(".") or line.startswith("("):
                    parts = line.split()
                    if len(parts) >= 2:
                        dot_bracket = parts[0]
                        try:
                            mfe = float(parts[-1])
                            structures.append(SuboptimalStructure(
                                mfe=mfe,
                                dot_bracket=dot_bracket,
                                energy_delta=0.0,  # Will compute later
                                probability=0.0,   # Will compute later
                            ))
                        except ValueError:
                            continue

        except subprocess.TimeoutExpired:
            warnings.warn("RNAsubopt timeout, using fallback")
        except Exception as e:
            warnings.warn(f"RNAsubopt error: {e}")

        # Fallback if no structures found
        if not structures:
            structures = self._estimate_suboptimal(sequence)

        # Compute energy deltas and probabilities
        if structures:
            min_mfe = min(s.mfe for s in structures)
            for s in structures:
                s.energy_delta = s.mfe - min_mfe
                # Boltzmann probability: exp(-ΔG/RT) / Σexp(-ΔG/RT)
                s.probability = np.exp(-s.mfe / RT_37C)

            # Normalize probabilities
            total_prob = sum(s.probability for s in structures)
            if total_prob > 0:
                for s in structures:
                    s.probability /= total_prob

        # Sort by energy
        structures.sort(key=lambda s: s.mfe)

        return structures

    def _run_barriers(self, sequence: str) -> Optional[float]:
        """
        Run barriers tool for energy landscape analysis.

        barriers computes minimum energy barrier between structures.
        Returns the barrier height to native state.
        """
        try:
            # barriers requires treegraph output
            result = subprocess.run(
                ["barriers", "-b", "10"],
                input=f">seq\n{sequence}\n",
                capture_output=True,
                text=True,
                timeout=180,
            )

            # Parse output for barrier heights
            lines = result.stdout.strip().split("\n")
            barriers = []
            for line in lines:
                # barriers output format: "barrier: 5.3 kcal/mol"
                match = re.search(r"barrier[:\s]+(\d+\.?\d*)", line)
                if match:
                    barriers.append(float(match.group(1)))

            if barriers:
                return min(barriers)  # Return minimum barrier

        except subprocess.TimeoutExpired:
            warnings.warn("barriers timeout")
        except Exception as e:
            warnings.warn(f"barriers error: {e}")

        return None

    def _estimate_suboptimal(self, sequence: str) -> List[SuboptimalStructure]:
        """
        Estimate suboptimal structures without ViennaRNA.

        Uses sequence composition heuristics:
        - GC content determines stability variation
        - Stem-loop patterns indicate alternative structures
        """
        seq = sequence.upper()
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / length

        # Estimate MFE from GC (per nucleotide)
        mfe_per_nt = -0.3 - 0.5 * gc
        mfe = mfe_per_nt * length

        # Generate estimated native structure
        native_dot = self._generate_estimated_dot_bracket(seq, gc)

        structures = [SuboptimalStructure(
            mfe=mfe,
            dot_bracket=native_dot,
            energy_delta=0.0,
            probability=0.8,  # Native dominates
        )]

        # Add estimated metastable states
        # Higher GC = fewer metastable states
        metastable_count = int(3 - gc * 2)  # 1-3 metastable states

        for i in range(metastable_count):
            delta = 2.0 + i * 3.0  # 2, 5, 8 kcal/mol above native
            subopt_mfe = mfe + delta
            subopt_dot = self._perturb_structure(native_dot, i)
            structures.append(SuboptimalStructure(
                mfe=subopt_mfe,
                dot_bracket=subopt_dot,
                energy_delta=delta,
                probability=0.2 / metastable_count,
            ))

        return structures

    def _estimate_barrier(self, sequence: str) -> float:
        """
        Estimate energy barrier without barriers tool.

        Heuristic: barrier ~ GC content * length_factor
        Higher GC = higher barrier (more stable)
        """
        seq = sequence.upper()
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / length

        # Barrier estimate: 5-15 kcal/mol typical
        # Higher GC = higher barrier
        barrier = BARRIER_STABLE + gc * 10.0 * (length / 500)

        return min(barrier, BARRIER_UNSTABLE * 2)

    def _generate_estimated_dot_bracket(self, seq: str, gc: float) -> str:
        """Generate estimated dot-bracket from GC content."""
        length = len(seq)
        # Simple stem-loop model
        stem_length = int(length * gc * 0.3)  # GC forms stems
        loop_length = length - 2 * stem_length

        if stem_length > 0:
            return "(" * stem_length + "." * loop_length + ")" * stem_length
        else:
            return "." * length

    def _perturb_structure(self, dot_bracket: str, perturbation: int) -> str:
        """Perturb structure for metastable estimation."""
        # Simple perturbation: reduce some stems
        bracket_list = list(dot_bracket)
        stems_to_reduce = perturbation + 1

        # Find stem pairs and reduce
        open_count = bracket_list.count("(")
        if open_count > stems_to_reduce:
            for i, ch in enumerate(bracket_list):
                if ch == "(" and stems_to_reduce > 0:
                    bracket_list[i] = "."
                    stems_to_reduce -= 1

        # Match closing brackets
        open_remaining = bracket_list.count("(")
        close_count = bracket_list.count(")")
        if close_count > open_remaining:
            excess = close_count - open_remaining
            for i in range(len(bracket_list) - 1, -1, -1):
                if bracket_list[i] == ")" and excess > 0:
                    bracket_list[i] = "."
                    excess -= 1

        return "".join(bracket_list)

    def _compute_folding_rate(self, seq: str, barrier: Optional[float]) -> float:
        """
        Compute folding rate using Arrhenius-type equation.

        k = k0 * exp(-ΔG_barrier / RT)

        Typical RNA folding rates: 10^-3 to 10^6 s^-1
        """
        if barrier is None:
            # Estimate barrier if not computed
            barrier = self._estimate_barrier(seq)

        # Arrhenius equation
        rate = K0_FOLDING * np.exp(-barrier / RT_37C)

        # Clamp to realistic range
        return np.clip(rate, 1e-3, 1e6)

    def _compute_landscape_complexity(self, structures: List[SuboptimalStructure]) -> float:
        """
        Compute structure landscape complexity score.

        Higher complexity = more metastable states = more functional potential
        """
        if not structures:
            return 0.0

        # Factors:
        # 1. Number of distinct structures
        n_structures = len(structures)

        # 2. Energy spread (how diverse are structures)
        if len(structures) > 1:
            energy_spread = max(s.mfe for s in structures) - min(s.mfe for s in structures)
            spread_factor = energy_spread / 20.0  # Normalize to typical max
        else:
            spread_factor = 0.0

        # 3. Probability distribution entropy
        probs = [s.probability for s in structures]
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)

        # Combine into complexity score [0, 1]
        complexity = (
            (n_structures - 1) / 10.0 * 0.3 +
            spread_factor * 0.3 +
            entropy / 3.0 * 0.4
        )

        return np.clip(complexity, 0.0, 1.0)

    def _compute_cotrans_score(self, seq: str, structures: List[SuboptimalStructure]) -> float:
        """
        Compute cotranscriptional folding potential.

        Higher score = more likely to fold during transcription
        (shorter stems, accessible loops)
        """
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / length

        # Cotranscriptional factors:
        # 1. Shorter stems (can fold during transcription)
        # 2. Lower GC (less kinetic trapping)
        # 3. Faster folding rate

        native = structures[0] if structures else None
        if native:
            stem_count = native.dot_bracket.count("(")
            avg_stem_length = stem_count / max(length * gc, 1)
            stem_factor = 1.0 - min(avg_stem_length * 10, 1.0)  # Shorter stems better
        else:
            stem_factor = 0.5

        # GC factor: lower GC = easier cotrans folding
        gc_factor = 1.0 - gc

        # Combine
        cotrans_score = stem_factor * 0.5 + gc_factor * 0.5

        return np.clip(cotrans_score, 0.0, 1.0)

    def _compute_dynamic_stability(self, seq: str, barrier: Optional[float], complexity: float) -> float:
        """
        Compute dynamic stability score.

        Higher stability = high barrier + low complexity
        (Structure unlikely to refold)
        """
        if barrier is None:
            barrier = self._estimate_barrier(seq)

        # High barrier = more stable
        barrier_factor = barrier / BARRIER_UNSTABLE

        # Low complexity = more stable (fewer alternative states)
        complexity_factor = 1.0 - complexity

        stability = barrier_factor * 0.6 + complexity_factor * 0.4

        return np.clip(stability, 0.0, 1.0)


def compute_kinetics_score(features: KineticsFeatures) -> Dict[str, float]:
    """
    Compute kinetics-related scores for immune sensing integration.

    Returns dict compatible with pipeline scoring.
    """
    scores = {
        "folding_rate_score": np.clip(features.folding_rate / 1e3, 0.0, 1.0),
        "barrier_score": np.clip(features.barrier_height / 15.0, 0.0, 1.0),
        "landscape_score": features.landscape_complexity,
        "cotrans_score": features.cotrans_folding_score,
        "dynamic_stability": features.stability_dynamic,
        "metastable_count": features.metastable_count,
    }

    # Immune relevance: metastable structures may expose immune sites
    # Higher complexity = more chance of exposing immunogenic motifs
    scores["immune_exposure_potential"] = features.landscape_complexity * 0.7

    return scores


# Convenience function
def predict_folding_kinetics(sequence: str) -> KineticsFeatures:
    """Predict folding kinetics for RNA sequence."""
    predictor = FoldingKineticsPredictor()
    return predictor.predict(sequence)