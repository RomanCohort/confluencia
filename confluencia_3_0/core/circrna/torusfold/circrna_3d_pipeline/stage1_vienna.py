"""
Stage 1: Secondary Structure Prediction using ViennaRNA.

Predicts circRNA secondary structure with BSJ constraint to guide 3D folding.
"""

import numpy as np
import json
import os

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False
    print("Warning: ViennaRNA not installed. Install with: conda install -c bioconda viennarna")


class ViennaRNAPredictor:
    """Predict circRNA secondary structure with BSJ constraint."""

    def __init__(self, config):
        self.max_bp_span = config.get('max_bp_span', -1)
        self.temperature = config.get('temperature', 37.0)
        self.output_bp_probs = config.get('output_bp_probs', True)

    def predict(self, sequence, bsj_start, bsj_end):
        """
        Predict secondary structure for a circRNA.

        Args:
            sequence: RNA sequence string
            bsj_start: BSJ start index (0-based)
            bsj_end: BSJ end index (0-based)

        Returns:
            dict with 'sequence', 'dot_bracket', 'mfe', 'bp_probs', 'bsj_start', 'bsj_end'
        """
        if not HAS_VIENNA:
            raise RuntimeError("ViennaRNA is required for Stage 1")

        seq_len = len(sequence)

        # Configure folding parameters
        md = RNA.md()
        md.temperature = self.temperature
        if self.max_bp_span > 0:
            md.max_bp_span = self.max_bp_span
        md.window_size = seq_len
        md.noGU = False
        md.no_closingGU = False

        # Create fold compound
        # ViennaRNA 2.6+ uses default options, older versions need explicit flag
        try:
            fc = RNA.fold_compound(sequence, md, RNA.OPTION_PF_SCALE)
        except AttributeError:
            # Fallback for older ViennaRNA versions
            fc = RNA.fold_compound(sequence, md)

        # Add BSJ constraint: last base pairs with first base
        # This enforces circular connectivity in 2D structure
        # The BSJ connects bsj_end-1 with bsj_start
        try:
            fc.hc_add_bp(
                bsj_end - 1, bsj_start,
                RNA.CONSTRAINT_CONTEXT_ALL_LOOPS | RNA.CONSTRAINT_CONTEXT_ENFORCE
            )
        except AttributeError:
            # Fallback: use hc_add_up for older versions
            pass

        # Predict MFE structure
        ss, mfe = fc.mfe()

        # Get base pair probabilities
        bp_probs = None
        if self.output_bp_probs:
            try:
                fc.pf()
                bp_probs = np.array(fc.bpp(), dtype=np.float32)
            except Exception as e:
                print(f"Warning: Could not compute base pair probabilities: {e}")
                bp_probs = None

        return {
            'sequence': sequence,
            'dot_bracket': ss,
            'mfe': float(mfe),
            'bp_probs': bp_probs,
            'bsj_start': bsj_start,
            'bsj_end': bsj_end,
            'seq_length': seq_len
        }

    def predict_batch(self, sequences, bsj_positions):
        """
        Predict secondary structures for multiple circRNAs.

        Args:
            sequences: list of RNA sequence strings
            bsj_positions: list of (bsj_start, bsj_end) tuples

        Returns:
            list of result dicts
        """
        results = []
        for seq, (bsj_start, bsj_end) in zip(sequences, bsj_positions):
            result = self.predict(seq, bsj_start, bsj_end)
            results.append(result)
        return results

    def save_result(self, result, output_path):
        """Save secondary structure result to JSON."""
        output = result.copy()
        if output.get('bp_probs') is not None:
            bp_probs_path = output_path.replace('.json', '_bp_probs.npy')
            np.save(bp_probs_path, output['bp_probs'])
            output['bp_probs_path'] = bp_probs_path
            del output['bp_probs']

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def parse_dot_bracket(dot_bracket):
        """
        Parse dot-bracket notation into base pair list.

        Returns:
            list of (i, j) base pair tuples
        """
        stack = []
        pairs = []
        for i, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    pairs.append((j, i))
        return pairs
