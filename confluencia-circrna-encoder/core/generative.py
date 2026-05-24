"""
generative.py — circRNA sequence generation and optimization.

Adapted from drug 2.0's generative.py for circRNA context.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class GenerativeConfig:
    """Configuration for circRNA generation."""

    # Sequence parameters
    min_length: int = 100
    max_length: int = 500
    default_length: int = 250

    # Composition constraints
    gc_min: float = 0.35
    gc_max: float = 0.65

    # Optimization targets
    target_immunogenicity: float = 0.6
    target_stability: float = 0.7

    # Generation parameters
    n_sequences: int = 100
    temperature: float = 1.0


class CircRNAGenerator:
    """
    Generate and optimize circRNA sequences.

    Methods:
    - Random generation with constraints
    - Template-based generation
    - Optimization via mutations
    """

    NUCS = ['A', 'U', 'G', 'C']

    def __init__(self, config: Optional[GenerativeConfig] = None):
        self.config = config or GenerativeConfig()

    def generate_random(self, length: int = None, gc_target: float = None) -> str:
        """
        Generate random circRNA sequence with constraints.

        Args:
            length: Sequence length
            gc_target: Target GC content

        Returns:
            circRNA sequence string
        """
        length = length or self.config.default_length

        if gc_target is None:
            gc_target = np.random.uniform(self.config.gc_min, self.config.gc_max)

        # Calculate required G and C count
        n_gc = int(length * gc_target)
        n_au = length - n_gc

        # Create sequence
        gc_bases = random.choices(['G', 'C'], k=n_gc)
        au_bases = random.choices(['A', 'U'], k=n_au)

        sequence = gc_bases + au_bases
        random.shuffle(sequence)

        return ''.join(sequence)

    def generate_batch(self, n: int = None) -> List[str]:
        """Generate batch of random sequences."""
        n = n or self.config.n_sequences

        sequences = []
        for _ in range(n):
            length = np.random.randint(self.config.min_length, self.config.max_length)
            gc_target = np.random.uniform(self.config.gc_min, self.config.gc_max)
            sequences.append(self.generate_random(length, gc_target))

        return sequences

    def generate_from_template(self, template: str, mutations: int = 5) -> str:
        """
        Generate sequence from template with mutations.

        Args:
            template: Template sequence
            mutations: Number of mutations to introduce

        Returns:
            Mutated sequence
        """
        seq = list(template)

        for _ in range(mutations):
            pos = random.randint(0, len(seq) - 1)
            original = seq[pos]

            # Mutate to different nucleotide
            choices = [n for n in self.NUCS if n != original]
            seq[pos] = random.choice(choices)

        return ''.join(seq)

    def optimize_sequence(
        self,
        sequence: str,
        target_immunogenicity: float = None,
        n_iterations: int = 50,
    ) -> Tuple[str, float]:
        """
        Optimize sequence for target immunogenicity.

        Args:
            sequence: Initial sequence
            target_immunogenicity: Target score
            n_iterations: Optimization iterations

        Returns:
            Optimized sequence, final score
        """
        from .innate_immune import quick_predict

        target = target_immunogenicity or self.config.target_immunogenicity

        best_seq = sequence
        best_score = quick_predict(sequence)['overall_score']

        for i in range(n_iterations):
            # Generate mutant
            mutant = self.generate_from_template(best_seq, mutations=1)

            # Evaluate
            score = quick_predict(mutant)['overall_score']

            # Accept if closer to target
            if abs(score - target) < abs(best_score - target):
                best_seq = mutant
                best_score = score

            if abs(best_score - target) < 0.05:
                break

        return best_seq, best_score

    def optimize_batch(
        self,
        sequences: List[str],
        target: float = None,
    ) -> List[Tuple[str, float]]:
        """Optimize batch of sequences."""
        target = target or self.config.target_immunogenicity

        optimized = []
        for seq in sequences:
            opt_seq, score = self.optimize_sequence(seq, target)
            optimized.append((opt_seq, score))

        return optimized

    def generate_high_immunogenic(self, length: int = 250) -> str:
        """Generate sequence optimized for high immunogenicity."""
        # High GC, moderate entropy
        n_gc = int(length * 0.55)
        n_au = length - n_gc

        # Use specific patterns for higher immunogenicity
        gc_bases = ['G', 'C'] * (n_gc // 2) + random.choices(['G', 'C'], k=n_gc % 2)
        au_bases = ['A', 'U'] * (n_au // 2) + random.choices(['A', 'U'], k=n_au % 2)

        sequence = gc_bases + au_bases
        random.shuffle(sequence)

        # Add some GU motifs (TLR7 activators)
        seq_list = list(sequence)
        for i in range(len(seq_list) - 1):
            if random.random() < 0.1:
                seq_list[i:i+2] = ['G', 'U']

        return ''.join(seq_list)

    def generate_low_immunogenic(self, length: int = 250) -> str:
        """Generate sequence optimized for low immunogenicity."""
        # Low GC, high A, minimal structure
        n_gc = int(length * 0.35)
        n_a = int(length * 0.4)
        n_u = length - n_gc - n_a

        sequence = (
            random.choices(['A'], k=n_a) +
            random.choices(['U'], k=n_u) +
            random.choices(['G', 'C'], k=n_gc)
        )
        random.shuffle(sequence)

        return ''.join(sequence)


class SequenceDesigner:
    """Design circRNA sequences with specific properties."""

    def __init__(self):
        self.generator = CircRNAGenerator()

    def design_for_target(
        self,
        target_gene: str,
        immunogenicity_target: float = 0.6,
        length: int = 250,
    ) -> Dict:
        """
        Design circRNA for specific target gene expression.

        Args:
            target_gene: Target gene name
            immunogenicity_target: Target immunogenicity
            length: Sequence length

        Returns:
            Designed sequence with metadata
        """
        # Generate base sequence
        base_seq = self.generator.generate_random(length)

        # Optimize for target
        opt_seq, final_score = self.generator.optimize_sequence(
            base_seq, immunogenicity_target
        )

        from .innate_immune import quick_predict

        immune_pred = quick_predict(opt_seq)

        return {
            'sequence': opt_seq,
            'length': length,
            'target_gene': target_gene,
            'gc_content': sum(1 for c in opt_seq if c in 'GC') / length,
            'immunogenicity_score': final_score,
            'rig_i_activation': immune_pred['rig_i']['score'],
            'tlr_activation': immune_pred['tlr']['score'],
            'pkr_activation': immune_pred['pkr']['score'],
            'immune_level': immune_pred['overall_level'],
        }

    def design_therapeutic_window(
        self,
        min_window: float = 0.3,
        n_attempts: int = 10,
    ) -> Dict:
        """Design circRNA with good therapeutic window."""
        from .dose_tox import quick_dose_predict

        best_seq = None
        best_window = 0

        for _ in range(n_attempts):
            seq = self.generator.generate_random(250)
            response = quick_dose_predict(seq, dose=100)

            if response['therapeutic_window'] > best_window:
                best_window = response['therapeutic_window']
                best_seq = seq

            if best_window >= min_window:
                break

        if best_seq:
            return {
                'sequence': best_seq,
                'therapeutic_window': best_window,
                'efficacy': quick_dose_predict(best_seq, 100)['efficacy_score'],
                'toxicity': quick_dose_predict(best_seq, 100)['toxicity_score'],
            }

        return None


def generate_optimized_sequence(target_score: float = 0.6) -> Tuple[str, float]:
    """Quick generation of optimized sequence."""
    generator = CircRNAGenerator()
    base_seq = generator.generate_random(250)
    return generator.optimize_sequence(base_seq, target_score)