"""
Stage 2: 3D Structure Prediction using trRosettaRNA2.

trRosettaRNA2 predicts inter-nucleotide distances and orientations,
which are then converted to 3D coordinates via restrained folding.

Key advantages over RoseTTAFold2NA:
- 3-5x faster (no MSA generation required for most sequences)
- RNA-specific optimization
- Distance/orientation predictions can directly guide cyclization

Reference: https://yanglab.qd.sdu.edu.cn/trRosettaRNA2
"""

import os
import subprocess
import numpy as np
import json
import tempfile
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class trRosettaRNA2Predictor:
    """
    Interface for trRosettaRNA2 3D structure prediction.

    Outputs distance and orientation restraints that can be used
    for restrained folding with OpenMM or Rosetta.
    """

    def __init__(self, config):
        """
        Initialize trRosettaRNA2 predictor.

        Args:
            config: dict with keys:
                - model_path: path to trRosettaRNA2 installation
                - num_samples: number of conformations to generate
                - device: 'cuda:0' or 'cpu'
                - max_seq_length: maximum sequence length
                - use_msa: whether to generate MSA (optional, improves accuracy)
                - use_gpu: whether to use GPU acceleration
        """
        self.model_path = config.get('model_path', 'models/trRosettaRNA2/')
        self.num_samples = config.get('num_samples', 5)
        self.device = config.get('device', 'cuda:0')
        self.max_seq_length = config.get('max_seq_length', 500)
        self.use_msa = config.get('use_msa', False)
        self.use_gpu = config.get('use_gpu', True)

        self.trrosetta_home = None
        self._find_trrosetta()

    def _find_trrosetta(self):
        """Find trRosettaRNA2 installation."""
        possible_paths = [
            os.environ.get('TRROSETTARNA2_HOME', ''),
            self.model_path,
            '/opt/trRosettaRNA2',
            os.path.expanduser('~/trRosettaRNA2'),
            os.path.expanduser('~/software/trRosettaRNA2'),
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                self.trrosetta_home = path
                predict_script = os.path.join(path, 'predict.py')
                if os.path.exists(predict_script):
                    print(f"Found trRosettaRNA2 at: {path}")
                    return

        print("Warning: trRosettaRNA2 not found in common paths.")
        print("Please install from: https://yanglab.qd.sdu.edu.cn/trRosettaRNA2")
        print("Or set TRROSETTARNA2_HOME environment variable.")

    def predict(
        self,
        sequence: str,
        dot_bracket: Optional[str] = None,
        bp_probs: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Predict 3D structure restraints for a linear RNA sequence.

        Args:
            sequence: RNA sequence string
            dot_bracket: Secondary structure (optional, improves accuracy)
            bp_probs: Base pair probability matrix (optional)
            output_dir: Directory to save outputs

        Returns:
            list of dicts with 'pdb_path', 'confidence', 'restraints', 'distance_matrix'
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()

        # Try trRosettaRNA2 if installed
        if self.trrosetta_home:
            try:
                results = self._run_trrosetta(
                    sequence=sequence,
                    output_dir=output_dir,
                    dot_bracket=dot_bracket,
                    bp_probs=bp_probs
                )
                elapsed = time.time() - start_time
                print(f"trRosettaRNA2 completed in {elapsed:.1f}s")
                return results
            except Exception as e:
                print(f"trRosettaRNA2 failed: {e}")
                print("Falling back to dummy structure generation")

        # Fallback: generate dummy structure for testing
        return self._generate_dummy_structure(sequence, output_dir)

    def _run_trrosetta(
        self,
        sequence: str,
        output_dir: str,
        dot_bracket: Optional[str] = None,
        bp_probs: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Execute trRosettaRNA2 prediction.

        trRosettaRNA2 workflow:
        1. Predict distance/orientation restraints from sequence
        2. Convert restraints to 3D coordinates using restrained folding
        3. Generate multiple conformations (num_samples)
        """
        predict_script = os.path.join(self.trrosetta_home, 'predict.py')
        if not os.path.exists(predict_script):
            raise FileNotFoundError(
                f"trRosettaRNA2 predict.py not found at {predict_script}"
            )

        # Prepare input FASTA
        fasta_path = os.path.join(output_dir, 'input.fasta')
        with open(fasta_path, 'w') as f:
            f.write(f">rna\n{sequence}\n")

        # Save secondary structure if provided
        ss_path = None
        if dot_bracket:
            ss_path = os.path.join(output_dir, 'secondary_structure.txt')
            with open(ss_path, 'w') as f:
                f.write(dot_bracket)

        # Save bp_probs if provided
        bp_probs_path = None
        if bp_probs is not None:
            bp_probs_path = os.path.join(output_dir, 'bp_probs.npy')
            np.save(bp_probs_path, bp_probs)

        # Build command
        cmd = [
            'python', predict_script,
            '--input', fasta_path,
            '--output_dir', output_dir,
            '--num_samples', str(self.num_samples),
            '--device', self.device,
        ]

        if self.use_msa:
            cmd.append('--use_msa')

        if ss_path:
            cmd.extend(['--secondary_structure', ss_path])

        if bp_probs_path:
            cmd.extend(['--bp_probs', bp_probs_path])

        # Run prediction
        print(f"Running trRosettaRNA2: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(f"trRosettaRNA2 failed: {result.stderr}")

        # Parse outputs
        results = []

        # trRosettaRNA2 outputs:
        # - restraints.npy: distance/orientation predictions
        # - pdbs/*.pdb: predicted structures
        # - confidence.json: per-sample confidence scores

        restraints_path = os.path.join(output_dir, 'restraints.npy')
        if os.path.exists(restraints_path):
            restraints = np.load(restraints_path, allow_pickle=True).item()
        else:
            restraints = self._generate_default_restraints(len(sequence))

        pdb_dir = os.path.join(output_dir, 'pdbs')
        if os.path.exists(pdb_dir):
            pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
            for i, pdb_file in enumerate(pdb_files):
                pdb_path = os.path.join(pdb_dir, pdb_file)

                # Load confidence
                confidence_path = os.path.join(output_dir, f'confidence_{i}.json')
                if os.path.exists(confidence_path):
                    with open(confidence_path) as f:
                        confidence_data = json.load(f)
                    confidence = confidence_data.get('mean_plddt', 0.7)
                else:
                    confidence = 0.7

                results.append({
                    'pdb_path': pdb_path,
                    'confidence': float(confidence),
                    'sample_id': i,
                    'restraints': restraints,
                    'distance_matrix': restraints.get('distance', None),
                    'orientation_matrix': restraints.get('orientation', None),
                })

        # If no PDBs generated, create from restraints
        if not results:
            results = self._build_structures_from_restraints(
                sequence, restraints, output_dir
            )

        return results

    def _build_structures_from_restraints(
        self,
        sequence: str,
        restraints: Dict,
        output_dir: str
    ) -> List[Dict]:
        """
        Build 3D structures from distance/orientation restraints.

        Uses restrained folding approach:
        1. Generate initial extended structure
        2. Apply distance restraints
        3. Optimize geometry

        This is a simplified version - full implementation would use
        OpenMM or Rosetta for restrained folding.
        """
        results = []

        # Extract distance predictions
        distance_matrix = restraints.get('distance', None)
        if distance_matrix is None:
            distance_matrix = self._estimate_distances(len(sequence))

        for sample_id in range(self.num_samples):
            pdb_path = os.path.join(output_dir, f'linear_{sample_id}.pdb')

            # Build structure guided by distance predictions
            coords = self._guided_folding(distance_matrix, sequence)

            # Write PDB
            self._write_pdb(pdb_path, sequence, coords)

            results.append({
                'pdb_path': pdb_path,
                'confidence': 0.6 + 0.2 * np.random.random(),  # Placeholder
                'sample_id': sample_id,
                'restraints': restraints,
                'distance_matrix': distance_matrix,
            })

        return results

    def _guided_folding(
        self,
        distance_matrix: np.ndarray,
        sequence: str
    ) -> np.ndarray:
        """
        Generate 3D coordinates guided by distance predictions.

        Simplified implementation - uses gradient descent to satisfy
        distance constraints.
        """
        n = len(sequence)
        coords = self._generate_extended_coords(n)

        if distance_matrix is None:
            return coords

        # Gradient descent to satisfy distance constraints
        # (Full implementation would use OpenMM or Rosetta)
        coords = self._optimize_distances(coords, distance_matrix)

        return coords

    def _optimize_distances(
        self,
        coords: np.ndarray,
        target_distances: np.ndarray,
        iterations: int = 100
    ) -> np.ndarray:
        """
        Optimize coordinates to satisfy distance constraints.
        """
        n = len(coords)

        for _ in range(iterations):
            for i in range(n):
                for j in range(i + 1, n):
                    if target_distances[i, j] > 0:
                        current_dist = np.linalg.norm(coords[i] - coords[j])
                        target_dist = target_distances[i, j]

                        if current_dist > 0:
                            # Adjust positions
                            direction = (coords[j] - coords[i]) / current_dist
                            adjustment = (target_dist - current_dist) * 0.1

                            coords[i] -= direction * adjustment * 0.5
                            coords[j] += direction * adjustment * 0.5

        return coords

    def _estimate_distances(self, length: int) -> np.ndarray:
        """
        Estimate inter-nucleotide distances for extended structure.

        Typical RNA geometry:
        - Adjacent nucleotides: ~6 Å (along backbone)
        - Base pairs: ~10 Å
        - Non-interacting: >15 Å
        """
        distances = np.zeros((length, length))

        for i in range(length):
            for j in range(i + 1, length):
                # Simple helical model
                rise_per_nt = 2.8  # Å
                distances[i, j] = abs(j - i) * rise_per_nt
                distances[j, i] = distances[i, j]

        return distances

    def _generate_default_restraints(self, length: int) -> Dict:
        """Generate default restraints based on RNA geometry."""
        return {
            'distance': self._estimate_distances(length),
            'orientation': np.zeros((length, length, 3)),  # Placeholder
        }

    def _generate_extended_coords(self, seq_length: int) -> np.ndarray:
        """Generate extended linear RNA coordinates."""
        coords = []
        for i in range(seq_length):
            # A-form RNA helical parameters
            angle = i * 33.0  # degrees per nucleotide
            rise = 2.8  # Å per nucleotide

            x = 10.0 * np.cos(np.radians(angle))
            y = 10.0 * np.sin(np.radians(angle))
            z = i * rise

            # Add small variations
            x += np.random.randn() * 0.3
            y += np.random.randn() * 0.3
            z += np.random.randn() * 0.1

            coords.append([x, y, z])

        return np.array(coords)

    def _write_pdb(self, pdb_path: str, sequence: str, coords: np.ndarray):
        """Write a simple PDB file with C3' atoms."""
        with open(pdb_path, 'w') as f:
            for i, (seq_char, coord) in enumerate(zip(sequence, coords)):
                atom_name = "C3'"
                res_name = f"  {seq_char}  "
                x, y, z = coord

                line = (
                    f"ATOM  {i+1:5d} {atom_name:4s} {res_name} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           C\n"
                )
                f.write(line)
            f.write("END\n")

    def _generate_dummy_structure(
        self,
        sequence: str,
        output_dir: str
    ) -> List[Dict]:
        """
        Generate dummy linear RNA structure for testing.

        Used when trRosettaRNA2 is not installed.
        """
        results = []

        for sample_id in range(self.num_samples):
            pdb_path = os.path.join(output_dir, f'linear_dummy_{sample_id}.pdb')

            coords = self._generate_extended_coords(len(sequence))
            self._write_pdb(pdb_path, sequence, coords)

            # Generate estimated restraints
            restraints = self._generate_default_restraints(len(sequence))

            results.append({
                'pdb_path': pdb_path,
                'confidence': 0.5,  # Low confidence for dummy
                'sample_id': sample_id,
                'restraints': restraints,
                'distance_matrix': restraints['distance'],
            })

        return results

    def predict_batch(
        self,
        sequences: List[str],
        ss_results: List[Dict],
        output_dir: str
    ) -> List[List[Dict]]:
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


def convert_trrosetta_restraints_to_openmm(
    restraints: Dict,
    sequence: str,
    bsj_start: int,
    bsj_end: int
) -> List[Tuple]:
    """
    Convert trRosettaRNA2 restraints to OpenMM CustomBondForce format.

    Args:
        restraints: dict with 'distance' and 'orientation' matrices
        sequence: RNA sequence
        bsj_start: BSJ start index
        bsj_end: BSJ end index

    Returns:
        list of (atom1, atom2, target_distance, weight) tuples
    """
    distance_matrix = restraints.get('distance', None)
    if distance_matrix is None:
        return []

    n = len(sequence)
    openmm_restraints = []

    # High-confidence distance predictions → OpenMM restraints
    for i in range(n):
        for j in range(i + 1, n):
            target_dist = distance_matrix[i, j]

            # Weight based on confidence (shorter distances = higher confidence)
            weight = 50.0 if target_dist < 10 else 10.0

            # Skip BSJ region (handled separately in Stage 3)
            if (i == bsj_start and j == bsj_end) or (i == bsj_end and j == bsj_start):
                continue

            openmm_restraints.append((i, j, target_dist, weight))

    return openmm_restraints


# ============================================================
# Testing
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='trRosettaRNA2 structure prediction')
    parser.add_argument('--sequence', help='RNA sequence to predict')
    parser.add_argument('--fasta', help='Input FASTA file')
    parser.add_argument('--output', default='trrosetta_output/', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--test', action='store_true', help='Run test mode')

    args = parser.parse_args()

    config = {
        'model_path': 'models/trRosettaRNA2/',
        'num_samples': args.num_samples,
        'device': args.device,
        'max_seq_length': 500,
    }

    predictor = trRosettaRNA2Predictor(config)

    if args.test:
        # Test with dummy sequence
        test_seq = "ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"
        print(f"Testing with sequence: {test_seq}")
        results = predictor.predict(test_seq, output_dir=args.output)
        print(f"Generated {len(results)} structures")
        for r in results:
            print(f"  Sample {r['sample_id']}: confidence={r['confidence']:.2f}")

    elif args.sequence:
        results = predictor.predict(args.sequence, output_dir=args.output)
        print(f"Generated {len(results)} structures")

    elif args.fasta:
        # Load FASTA
        sequences = []
        with open(args.fasta) as f:
            current_seq = ''
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ''
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)

        print(f"Loaded {len(sequences)} sequences")

        # Process all
        ss_results = [{'dot_bracket': None, 'bp_probs': None} for _ in sequences]
        results = predictor.predict_batch(sequences, ss_results, args.output)

        total = sum(len(r) for r in results)
        print(f"Generated {total} structures total")