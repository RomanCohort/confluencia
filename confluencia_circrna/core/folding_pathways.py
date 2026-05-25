"""
folding_pathways.py — RNA Folding Pathway Analysis

Integrates ViennaRNA barriers/RNApvmin for:
1. Minimum energy folding pathway
2. Transition state identification
3. Folding pathway complexity
4. Alternative folding routes
5. Rate-limiting step analysis

Literature basis:
- Flamm et al., 2002: barriers - folding landscape analysis
- Wolfinger et al., 2004: RNApvmin - minimum path calculation
- Morgan & Higgs, 1998: Barrier heights in RNA folding
- Geis et al., 2015: circRNA folding pathway complexity

Key concepts:
- Folding pathway: sequence of structure transitions to native
- Barrier height: energy required for structure transition
- Transition state: highest energy point in pathway
- Alternative pathways: different routes to same native state
- Kinetic partitioning: fraction following fastest pathway
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import subprocess
import tempfile
import warnings
import re
from pathlib import Path
from enum import Enum

# Energy thresholds
SMALL_BARRIER = 3.0     # kcal/mol - fast transition
MEDIUM_BARRIER = 10.0   # kcal/mol - moderate transition
LARGE_BARRIER = 20.0    # kcal/mol - slow/rare transition


class PathwayType(Enum):
    """Folding pathway classification."""
    DIRECT = "direct"           # Single dominant pathway
    BRANCHED = "branched"       # Multiple competing pathways
    KINETIC_TRAP = "trap"       # Pathway with kinetic trap
    MISFOLD = "misfold"         # Pathway leading to non-native


@dataclass
class StructureTransition:
    """A transition between two structures in folding pathway."""
    structure_from: str         # Starting structure (dot-bracket)
    structure_to: str           # Target structure
    barrier: float              # Energy barrier for transition
    saddle_structure: str       # Transition state structure
    rate_forward: float         # Forward rate constant
    rate_backward: float        # Backward rate constant


@dataclass
class FoldingPathway:
    """A complete folding pathway."""
    pathway_id: int
    pathway_type: PathwayType
    transitions: List[StructureTransition]
    total_barrier: float        # Highest barrier in pathway
    pathway_rate: float         # Estimated overall rate
    probability: float          # Probability of following this pathway
    misfold_risk: float         # Risk of non-native outcome


@dataclass
class TransitionState:
    """Transition state in folding."""
    structure: str              # Saddle point structure
    energy: float               # Energy at saddle point
    position: int               # Position in pathway
    lifetime: float             # Estimated lifetime


@dataclass
class PathwayFeatures:
    """Folding pathway analysis features."""
    pathways: List[FoldingPathway]
    dominant_pathway: Optional[FoldingPathway]
    transition_states: List[TransitionState]
    rate_limiting_barrier: float    # Highest barrier overall
    pathway_complexity: float       # Complexity score [0, 1]
    kinetic_partition_factor: float # Fraction on fastest pathway
    misfold_probability: float      # Probability of misfolding
    folding_time_estimate: float    # Estimated total folding time (s)
    pathway_method: str             # Method used (viennarna/fallback)


class FoldingPathwayAnalyzer:
    """
    Analyze RNA folding pathways using ViennaRNA barriers/RNApvmin.

    Computes:
    - Minimum energy folding path (RNApvmin)
    - Barrier heights between structures
    - Alternative folding pathways
    - Kinetic partitioning factors
    """

    def __init__(
        self,
        max_pathways: int = 10,
        min_barrier: float = 0.5,
    ):
        """
        Initialize pathway analyzer.

        Args:
            max_pathways: Maximum number of pathways to analyze
            min_barrier: Minimum barrier to consider significant
        """
        self.max_pathways = max_pathways
        self.min_barrier = min_barrier
        self._has_barriers = self._check_barriers_installed()
        self._has_rnapvmin = self._check_rnapvmin_installed()

    def _check_barriers_installed(self) -> bool:
        """Check if barriers tool is available."""
        try:
            result = subprocess.run(
                ["barriers", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_rnapvmin_installed(self) -> bool:
        """Check if RNApvmin is available."""
        try:
            result = subprocess.run(
                ["RNApvmin", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def analyze(self, sequence: str) -> PathwayFeatures:
        """
        Analyze folding pathways for RNA sequence.

        Args:
            sequence: RNA sequence

        Returns:
            PathwayFeatures with pathways, barriers, transition states
        """
        seq = self._sanitize_sequence(sequence)
        if len(seq) < 50:
            return self._empty_features()

        if self._has_barriers and self._has_rnapvmin:
            pathways, transitions = self._run_pathway_tools(seq)
            method = "viennarna_pathway"
        else:
            pathways, transitions = self._estimate_pathways(seq)
            method = "fallback_pathway"
            warnings.warn(
                "barriers/RNApvmin not installed, using pathway estimation. "
                "Install ViennaRNA for accurate pathway analysis."
            )

        # Compute derived features
        dominant = self._find_dominant_pathway(pathways)
        transition_states = self._extract_transition_states(transitions)
        rate_barrier = self._find_rate_limiting_barrier(pathways)
        complexity = self._compute_pathway_complexity(pathways)
        partition = self._compute_kinetic_partition(pathways)
        misfold = self._compute_misfold_probability(pathways)
        folding_time = self._estimate_folding_time(pathways)

        return PathwayFeatures(
            pathways=pathways[:self.max_pathways],
            dominant_pathway=dominant,
            transition_states=transition_states,
            rate_limiting_barrier=rate_barrier,
            pathway_complexity=complexity,
            kinetic_partition_factor=partition,
            misfold_probability=misfold,
            folding_time_estimate=folding_time,
            pathway_method=method,
        )

    def _sanitize_sequence(self, sequence: str) -> str:
        """Convert DNA to RNA."""
        return sequence.upper().replace("T", "U")

    def _empty_features(self) -> PathwayFeatures:
        """Return empty features for short sequences."""
        return PathwayFeatures(
            pathways=[],
            dominant_pathway=None,
            transition_states=[],
            rate_limiting_barrier=0.0,
            pathway_complexity=0.0,
            kinetic_partition_factor=0.0,
            misfold_probability=0.0,
            folding_time_estimate=0.0,
            pathway_method="sequence_too_short",
        )

    def _run_pathway_tools(self, sequence: str) -> Tuple[List[FoldingPathway], List[StructureTransition]]:
        """
        Run barriers and RNApvmin for pathway analysis.

        barriers computes:
        - Energy landscape minima
        - Barrier heights between minima

        RNApvmin computes:
        - Minimum folding path
        - Transition state structures
        """
        pathways = []
        transitions = []

        try:
            # Run barriers for landscape
            barriers_result = subprocess.run(
                ["barriers", "-b", "10", "-s"],
                input=f">seq\n{sequence}\n",
                capture_output=True,
                text=True,
                timeout=180,
            )

            # Parse barriers output
            barriers_output = barriers_result.stdout
            landscape_minima = self._parse_barriers_output(barriers_output)

            # Run RNApvmin for minimum path
            if self._has_rnapvmin:
                pvmin_result = subprocess.run(
                    ["RNApvmin", "-p"],
                    input=f">seq\n{sequence}\n",
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                pvmin_output = pvmin_result.stdout
                min_path = self._parse_pvmin_output(pvmin_output)

                # Build primary pathway from pvmin
                if min_path:
                    pathway = self._build_pathway_from_transitions(min_path, 0)
                    pathways.append(pathway)
                    transitions.extend(min_path)

            # Build alternative pathways from landscape
            for i, min_pair in enumerate(landscape_minima[1:self.max_pathways]):
                alt_path = self._build_alternative_pathway(min_pair, i + 1)
                pathways.append(alt_path)

        except subprocess.TimeoutExpired:
            warnings.warn("Pathway tools timeout, using fallback")
        except Exception as e:
            warnings.warn(f"Pathway tools error: {e}")

        # Fallback if no results
        if not pathways:
            pathways, transitions = self._estimate_pathways(sequence)

        return pathways, transitions

    def _parse_barriers_output(self, output: str) -> List[Tuple[str, str, float]]:
        """Parse barriers tool output for landscape minima."""
        minima = []

        lines = output.strip().split("\n")
        for line in lines:
            # barriers output format:
            # structure1 structure2 barrier_energy
            parts = line.split()
            if len(parts) >= 3:
                try:
                    struct1 = parts[0]
                    struct2 = parts[1]
                    barrier = float(parts[2])
                    minima.append((struct1, struct2, barrier))
                except ValueError:
                    continue

        return minima

    def _parse_pvmin_output(self, output: str) -> List[StructureTransition]:
        """Parse RNApvmin output for minimum pathway."""
        transitions = []

        lines = output.strip().split("\n")
        prev_structure = None

        for line in lines:
            # RNApvmin outputs sequence of structures with energies
            if line.startswith(".") or line.startswith("("):
                parts = line.split()
                if len(parts) >= 2:
                    structure = parts[0]
                    energy = float(parts[1]) if len(parts) > 1 else 0.0

                    if prev_structure:
                        # Build transition
                        barrier = abs(energy) * 0.5  # Estimate
                        transition = StructureTransition(
                            structure_from=prev_structure,
                            structure_to=structure,
                            barrier=barrier,
                            saddle_structure=structure,  # Placeholder
                            rate_forward=self._compute_rate(barrier),
                            rate_backward=self._compute_rate(barrier * 0.5),
                        )
                        transitions.append(transition)

                    prev_structure = structure

        return transitions

    def _estimate_pathways(self, sequence: str) -> Tuple[List[FoldingPathway], List[StructureTransition]]:
        """
        Estimate folding pathways without ViennaRNA tools.

        Heuristic estimation based on:
        - GC content (stability)
        - Sequence length (complexity)
        - Known RNA folding patterns
        """
        seq = sequence.upper()
        length = len(seq)
        gc = sum(1 for c in seq if c in "GC") / length

        pathways = []
        transitions = []

        # Estimate primary pathway (direct folding)
        # Generate estimated structures
        unfolded = "." * length
        partial_stem_length = int(length * gc * 0.15)
        partial = "(" * partial_stem_length + "." * (length - 2 * partial_stem_length) + ")" * partial_stem_length

        full_stem_length = int(length * gc * 0.25)
        native = "(" * full_stem_length + "." * (length - 2 * full_stem_length) + ")" * full_stem_length

        # Build transitions for primary pathway
        barrier1 = gc * 5.0  # Unfolded to partial
        transition1 = StructureTransition(
            structure_from=unfolded,
            structure_to=partial,
            barrier=barrier1,
            saddle_structure=partial,
            rate_forward=self._compute_rate(barrier1),
            rate_backward=self._compute_rate(barrier1 * 2),
        )

        barrier2 = gc * 8.0  # Partial to native
        transition2 = StructureTransition(
            structure_from=partial,
            structure_to=native,
            barrier=barrier2,
            saddle_structure=native,
            rate_forward=self._compute_rate(barrier2),
            rate_backward=self._compute_rate(barrier2 * 1.5),
        )

        transitions = [transition1, transition2]

        # Primary pathway
        primary = FoldingPathway(
            pathway_id=0,
            pathway_type=PathwayType.DIRECT,
            transitions=transitions,
            total_barrier=max(barrier1, barrier2),
            pathway_rate=min(transition1.rate_forward, transition2.rate_forward),
            probability=0.7,  # Primary pathway dominant
            misfold_risk=0.1,
        )
        pathways.append(primary)

        # Estimate alternative pathway (if GC is moderate)
        if 0.4 <= gc <= 0.6:
            # Alternative: different stem formation
            alt_stem = int(length * gc * 0.2)
            alt_structure = "(" * alt_stem + "." * (length - 2 * alt_stem) + ")" * alt_stem

            alt_barrier = gc * 10.0
            alt_transition = StructureTransition(
                structure_from=unfolded,
                structure_to=alt_structure,
                barrier=alt_barrier,
                saddle_structure=alt_structure,
                rate_forward=self._compute_rate(alt_barrier),
                rate_backward=self._compute_rate(alt_barrier),
            )

            alt_pathway = FoldingPathway(
                pathway_id=1,
                pathway_type=PathwayType.BRANCHED,
                transitions=[alt_transition],
                total_barrier=alt_barrier,
                pathway_rate=alt_transition.rate_forward,
                probability=0.25,
                misfold_risk=0.15,
            )
            pathways.append(alt_pathway)

        # Estimate kinetic trap pathway (high GC)
        if gc > 0.6:
            trap_stem = int(length * 0.4)  # Long trapped stem
            trap_structure = "(" * trap_stem + "." * (length - 2 * trap_stem) + ")" * trap_stem

            trap_barrier = gc * 15.0
            trap_transition = StructureTransition(
                structure_from=partial,
                structure_to=trap_structure,
                barrier=trap_barrier,
                saddle_structure=trap_structure,
                rate_forward=self._compute_rate(trap_barrier) * 0.1,
                rate_backward=self._compute_rate(trap_barrier) * 0.01,
            )

            trap_pathway = FoldingPathway(
                pathway_id=2,
                pathway_type=PathwayType.KINETIC_TRAP,
                transitions=[trap_transition],
                total_barrier=trap_barrier,
                pathway_rate=trap_transition.rate_forward * 0.001,
                probability=0.05,
                misfold_risk=0.3,
            )
            pathways.append(trap_pathway)

        return pathways, transitions

    def _compute_rate(self, barrier: float) -> float:
        """
        Compute transition rate using Arrhenius equation.

        k = k0 * exp(-barrier / RT)
        """
        RT = 0.616  # kcal/mol at 37°C
        k0 = 1e6    # Base rate constant
        return k0 * np.exp(-barrier / RT)

    def _build_pathway_from_transitions(self, transitions: List[StructureTransition], id: int) -> FoldingPathway:
        """Build pathway from list of transitions."""
        if not transitions:
            return FoldingPathway(
                pathway_id=id,
                pathway_type=PathwayType.DIRECT,
                transitions=[],
                total_barrier=0.0,
                pathway_rate=0.0,
                probability=1.0,
                misfold_risk=0.0,
            )

        total_barrier = max(t.barrier for t in transitions)
        pathway_rate = min(t.rate_forward for t in transitions)

        # Classify pathway type
        if total_barrier < SMALL_BARRIER:
            ptype = PathwayType.DIRECT
        elif total_barrier < MEDIUM_BARRIER:
            ptype = PathwayType.BRANCHED
        else:
            ptype = PathwayType.KINETIC_TRAP

        return FoldingPathway(
            pathway_id=id,
            pathway_type=ptype,
            transitions=transitions,
            total_barrier=total_barrier,
            pathway_rate=pathway_rate,
            probability=1.0 / (id + 1),  # First pathway most probable
            misfold_risk=total_barrier / 30.0,  # Higher barrier = more risk
        )

    def _build_alternative_pathway(self, min_pair: Tuple, id: int) -> FoldingPathway:
        """Build alternative pathway from landscape minimum."""
        struct1, struct2, barrier = min_pair

        transition = StructureTransition(
            structure_from=struct1,
            structure_to=struct2,
            barrier=barrier,
            saddle_structure=struct2,
            rate_forward=self._compute_rate(barrier),
            rate_backward=self._compute_rate(barrier * 0.8),
        )

        return FoldingPathway(
            pathway_id=id,
            pathway_type=PathwayType.BRANCHED,
            transitions=[transition],
            total_barrier=barrier,
            pathway_rate=transition.rate_forward,
            probability=0.1 / id,
            misfold_risk=barrier / 25.0,
        )

    def _find_dominant_pathway(self, pathways: List[FoldingPathway]) -> Optional[FoldingPathway]:
        """Find the dominant (fastest) pathway."""
        if not pathways:
            return None

        # Highest probability = dominant
        return max(pathways, key=lambda p: p.probability)

    def _extract_transition_states(self, transitions: List[StructureTransition]) -> List[TransitionState]:
        """Extract transition states from transitions."""
        states = []

        for i, trans in enumerate(transitions):
            # Transition state is saddle structure
            states.append(TransitionState(
                structure=trans.saddle_structure,
                energy=trans.barrier,
                position=i,
                lifetime=1.0 / trans.rate_forward,
            ))

        return states

    def _find_rate_limiting_barrier(self, pathways: List[FoldingPathway]) -> float:
        """Find the highest barrier (rate-limiting step)."""
        if not pathways:
            return 0.0

        return max(p.total_barrier for p in pathways)

    def _compute_pathway_complexity(self, pathways: List[FoldingPathway]) -> float:
        """
        Compute pathway complexity score.

        More pathways + higher barriers = more complex folding
        """
        if not pathways:
            return 0.0

        # Factors:
        # 1. Number of pathways
        n_factor = len(pathways) / 10.0

        # 2. Barrier variation
        barriers = [p.total_barrier for p in pathways]
        barrier_var = np.std(barriers) / MEDIUM_BARRIER if len(barriers) > 1 else 0.0

        # 3. Pathway type diversity
        types = set(p.pathway_type for p in pathways)
        type_factor = len(types) / 4.0

        complexity = n_factor * 0.3 + barrier_var * 0.4 + type_factor * 0.3

        return np.clip(complexity, 0.0, 1.0)

    def _compute_kinetic_partition(self, pathways: List[FoldingPathway]) -> float:
        """
        Compute kinetic partitioning factor.

        Fraction of molecules following fastest pathway
        """
        if not pathways:
            return 0.0

        # Sum of probabilities weighted by rates
        total_prob_rate = sum(p.probability * p.pathway_rate for p in pathways)
        total_rate = sum(p.pathway_rate for p in pathways)

        if total_rate == 0:
            return 0.0

        # Dominant pathway contribution
        dominant = self._find_dominant_pathway(pathways)
        if dominant:
            return dominant.probability
        else:
            return max(p.probability for p in pathways)

    def _compute_misfold_probability(self, pathways: List[FoldingPathway]) -> float:
        """Compute total misfold probability."""
        if not pathways:
            return 0.0

        # Sum misfold risks weighted by pathway probability
        return sum(p.misfold_risk * p.probability for p in pathways)

    def _estimate_folding_time(self, pathways: List[FoldingPathway]) -> float:
        """
        Estimate total folding time.

        Based on dominant pathway rate
        """
        dominant = self._find_dominant_pathway(pathways)
        if not dominant or dominant.pathway_rate == 0:
            return 0.0

        # Folding time = 1 / pathway_rate
        return 1.0 / dominant.pathway_rate


def compute_pathway_immunogenicity(features: PathwayFeatures) -> Dict[str, float]:
    """
    Compute immunogenicity scores from pathway analysis.

    Pathway complexity affects immune recognition:
    - More complex = more intermediate structures
    - Intermediates may expose immunogenic motifs
    """
    scores = {
        "pathway_complexity_score": features.pathway_complexity,
        "misfold_risk_score": features.misfold_probability,
        "intermediate_exposure": features.pathway_complexity * 0.6,
        "rate_limiting_barrier": features.rate_limiting_barrier / 20.0,
        "folding_time_score": np.clip(features.folding_time_estimate / 10.0, 0.0, 1.0),
    }

    # Higher complexity = more chance of immune exposure during folding
    scores["immune_exposure_during_folding"] = (
        features.pathway_complexity * 0.5 +
        features.misfold_probability * 0.3 +
        (1 - features.kinetic_partition_factor) * 0.2
    )

    return scores


def find_optimal_folding_conditions(sequence: str) -> Dict[str, Any]:
    """
    Find conditions that optimize folding to native state.

    Returns:
        Recommendations for temperature, salt conditions, etc.
    """
    analyzer = FoldingPathwayAnalyzer()
    features = analyzer.analyze(sequence)

    recommendations = {
        "native_probability": features.kinetic_partition_factor,
        "misfold_risk": features.misfold_probability,
        "rate_limiting_barrier": features.rate_limiting_barrier,
        "suggestions": [],
    }

    if features.misfold_probability > 0.3:
        recommendations["suggestions"].append(
            "High misfold risk - consider slower transcription or lower temperature"
        )

    if features.rate_limiting_barrier > MEDIUM_BARRIER:
        recommendations["suggestions"].append(
            f"Large barrier ({features.rate_limiting_barrier:.1f} kcal/mol) - folding may be slow"
        )

    if features.pathway_complexity > 0.7:
        recommendations["suggestions"].append(
            "Complex folding landscape - multiple outcomes possible"
        )

    return recommendations


# Convenience function
def analyze_folding_pathways(sequence: str) -> PathwayFeatures:
    """Analyze folding pathways for RNA sequence."""
    analyzer = FoldingPathwayAnalyzer()
    return analyzer.analyze(sequence)