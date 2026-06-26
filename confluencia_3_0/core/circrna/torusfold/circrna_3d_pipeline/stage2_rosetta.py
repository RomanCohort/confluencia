"""
Stage 2: 3D Structure Prediction using RoseTTAFold2NA.

Predicts initial 3D coordinates for linear RNA sequence.
Note: RoseTTAFold2NA predicts LINEAR RNA, not circRNA.
Cyclization is done in Stage 3.
"""

import os
import subprocess
import numpy as np
import json
import tempfile

# RoseTTAFold2NA is typically installed separately
# This module provides an interface to call it


class RoseTTAFold2NAPredictor:
    """Interface for RoseTTAFold2NA 3D structure prediction."""

    def __init__(self, config):
        self.model_path = config.get('model_path', 'models/rosettafold2na/')
        self.num_samples = config.get('num_samples', 5)
        self.batch_size = config.get('batch_size', 1)
        self.device = config.get('device', 'cuda:0')
        self.max_seq_length = config.get('max_seq_length', 500)

        self.rosetta_script = None
        self._find_rosetta()

    def _find_rosetta(self):
        """Find RoseTTAFold2NA installation."""
        # Check common installation paths
        possible_paths = [
            os.path.join(os.environ.get('HOME', ''), 'RoseTTAFold2NA'),
            '/opt/RoseTTAFold2NA',
            '/usr/local/RoseTTAFold2NA',
            os.path.expanduser('~/software/RoseTTAFold2NA'),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                self.rosetta_script = os.path.join(path, 'run_infer.py')
                if os.path.exists(self.rosetta_script):
                    print(f"Found RoseTTAFold2NA at: {path}")
                    return

        print("Warning: RoseTTAFold2NA not found in common paths.")
        print("Please set model_path in config or set ROSETTAFOLD2NA_HOME environment variable.")

    def predict(self, sequence, dot_bracket=None, bp_probs=None, output_dir=None):
        """
        Predict 3D structure for a linear RNA sequence.

        Args:
            sequence: RNA sequence string
            dot_bracket: Secondary structure (optional, helps accuracy)
            bp_probs: Base pair probability matrix (optional)
            output_dir: Directory to save outputs

        Returns:
            list of dicts with 'pdb_path', 'confidence', 'distance_matrix'
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        os.makedirs(output_dir, exist_ok=True)

        # Prepare input files
        fasta_path = os.path.join(output_dir, 'input.fasta')
        with open(fasta_path, 'w') as f:
            f.write(f">rna\n{sequence}\n")

        # Run RoseTTAFold2NA
        results = []
        try:
            results = self._run_rosetta(
                fasta_path=fasta_path,
                output_dir=output_dir,
                dot_bracket=dot_bracket,
                bp_probs=bp_probs
            )
        except Exception as e:
            print(f"RoseTTAFold2NA failed: {e}")
            # Fallback: generate dummy structure for testing
            results = self._generate_dummy_structure(sequence, output_dir)

        return results

    def _run_rosetta(self, fasta_path, output_dir, dot_bracket=None, bp_probs=None):
        """
        Execute RoseTTAFold2NA prediction.

        This is a placeholder that should be adapted to the actual
        RoseTTAFold2NA interface when available.
        """
        if self.rosetta_script is None or not os.path.exists(self.rosetta_script):
            raise FileNotFoundError(
                "RoseTTAFold2NA not installed. "
                "Please install from: https://github.com/baker-laboratory/RoseTTAFold2NA"
            )

        # Save bp_probs if provided
        bp_probs_path = None
        if bp_probs is not None:
            bp_probs_path = os.path.join(output_dir, 'bp_probs.npy')
            np.save(bp_probs_path, bp_probs)

        # Build command
        cmd = [
            'python', self.rosetta_script,
            '--fasta', fasta_path,
            '--output_dir', output_dir,
            '--num_samples', str(self.num_samples),
            '--device', self.device,
        ]

        if dot_bracket:
            ss_path = os.path.join(output_dir, 'ss.txt')
            with open(ss_path, 'w') as f:
                f.write(dot_bracket)
            cmd.extend(['--ss', ss_path])

        if bp_probs_path:
            cmd.extend(['--bp_probs', bp_probs_path])

        # Run prediction
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"RoseTTAFold2NA failed: {result.stderr}")

        # Parse outputs
        results = []
        for i in range(self.num_samples):
            pdb_path = os.path.join(output_dir, f'prediction_{i}.pdb')
            if os.path.exists(pdb_path):
                # Load confidence scores
                confidence_path = os.path.join(output_dir, f'confidence_{i}.npy')
                confidence = np.load(confidence_path) if os.path.exists(confidence_path) else 0.7

                results.append({
                    'pdb_path': pdb_path,
                    'confidence': float(np.mean(confidence)) if isinstance(confidence, np.ndarray) else confidence,
                    'sample_id': i
                })

        return results

    def _generate_dummy_structure(self, sequence, output_dir):
        """
        Generate a dummy linear RNA structure for testing.

        Creates a simple extended conformation.
        """
        results = []

        for sample_id in range(self.num_samples):
            pdb_path = os.path.join(output_dir, f'linear_dummy_{sample_id}.pdb')

            # Generate simple extended structure
            coords = self._generate_extended_coords(len(sequence))

            # Write PDB
            self._write_pdb(pdb_path, sequence, coords)

            results.append({
                'pdb_path': pdb_path,
                'confidence': 0.5,  # Low confidence for dummy structures
                'sample_id': sample_id
            })

        return results

    def _generate_extended_coords(self, seq_length):
        """Generate extended linear RNA coordinates."""
        coords = []
        for i in range(seq_length):
            # Simple helical parameters
            angle = i * 33.0  # ~33 degrees per nucleotide in A-form RNA
            rise = 2.8  # Angstroms per nucleotide

            x = 10.0 * np.cos(np.radians(angle))
            y = 10.0 * np.sin(np.radians(angle))
            z = i * rise

            # Add some variation
            x += np.random.randn() * 0.5
            y += np.random.randn() * 0.5
            z += np.random.randn() * 0.2

            coords.append([x, y, z])

        return np.array(coords)

    def _write_pdb(self, pdb_path, sequence, coords):
        """Write a simple PDB file with C3' atoms."""
        with open(pdb_path, 'w') as f:
            for i, (seq_char, coord) in enumerate(zip(sequence, coords)):
                atom_name = "C3'"
                res_name = f"  {seq_char}  "  # Single letter RNA code
                x, y, z = coord

                line = (
                    f"ATOM  {i+1:5d} {atom_name:4s} {res_name} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
                )
                f.write(line)
            f.write("END\n")

    def predict_batch(self, sequences, ss_results, output_dir):
        """
        Predict 3D structures for multiple sequences.

        Args:
            sequences: list of RNA sequence strings
            ss_results: list of secondary structure results from Stage 1
            output_dir: base output directory

        Returns:
            list of list of results
        """
        all_results = []
        for i, (seq, ss) in enumerate(zip(sequences, ss_results)):
            seq_dir = os.path.join(output_dir, f'seq_{i}')
            result = self.predict(
                sequence=seq,
                dot_bracket=ss.get('dot_bracket'),
                bp_probs=ss.get('bp_probs'),
                output_dir=seq_dir
            )
            all_results.append(result)
        return all_results
