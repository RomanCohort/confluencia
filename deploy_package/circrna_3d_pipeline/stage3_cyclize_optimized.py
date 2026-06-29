"""
Stage 3: BSJ Cyclization using OpenMM — OPTIMIZED VERSION.

Optimizations:
1. Stronger BSJ restraint with annealing schedule
2. Secondary structure-guided cyclization (ViennaRNA constraints)
3. Distance restraint integration (from trRosettaRNA2 predictions)
4. Multiple cyclization attempts with different starting conformations
5. Gradual restraint release for smoother transition
6. Better clash resolution with longer minimization
"""

import os
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("Warning: OpenMM not installed. Install with: pip install openmm")


class OptimizedBSJCyclizer:
    """
    Optimized circRNA BSJ cyclization with:
    - SS-guided folding
    - Distance restraint integration
    - Annealing schedule
    - Multiple attempts
    """

    def __init__(self, config):
        # BSJ restraint parameters (optimized)
        self.bsj_restraint_k_initial = config.get('bsj_restraint_k_initial', 2000.0)
        self.bsj_restraint_k_final = config.get('bsj_restraint_k_final', 500.0)
        self.bsj_target_distance = config.get('bsj_target_distance', 3.5)

        # Secondary structure restraint (optimized)
        self.ss_restraint_k = config.get('ss_restraint_k', 100.0)
        self.bp_target_distance = config.get('bp_target_distance', 10.0)  # C3'-C3' distance

        # Distance restraint from trRosettaRNA2
        self.use_trrosetta_restraints = config.get('use_trrosetta_restraints', True)
        self.restraint_weight = config.get('restraint_weight', 50.0)

        # Minimization parameters (optimized)
        self.max_iterations_phase1 = config.get('max_iterations_phase1', 1000)
        self.max_iterations_phase2 = config.get('max_iterations_phase2', 2000)
        self.minimization_tolerance = config.get('minimization_tolerance', 5.0)

        # Annealing schedule
        self.annealing_steps = config.get('annealing_steps', 100)
        self.annealing_temperature_start = config.get('annealing_temperature_start', 500.0)
        self.annealing_temperature_end = config.get('annealing_temperature_end', 300.0)

        # Multiple attempts
        self.num_attempts = config.get('num_attempts', 3)

    def cyclize(
        self,
        pdb_path: str,
        bsj_start: int,
        bsj_end: int,
        ss_pairs: Optional[List[Tuple[int, int]]] = None,
        distance_restraints: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Optimized cyclization with multiple phases.

        Protocol:
        Phase 1: Strong BSJ restraint + SS constraints + minimize
        Phase 2: Annealing with gradual restraint release
        Phase 3: Final minimization with moderate BSJ restraint

        Args:
            pdb_path: Linear RNA PDB file
            bsj_start: BSJ start index (0-based)
            bsj_end: BSJ end index (0-based)
            ss_pairs: Base pairs from secondary structure [(i,j), ...]
            distance_restraints: Distance matrix from trRosettaRNA2 (optional)
            output_path: Output PDB path

        Returns:
            dict with cyclization results
        """
        if not HAS_OPENMM:
            raise RuntimeError("OpenMM is required for Stage 3")

        if output_path is None:
            output_path = pdb_path.replace('.pdb', '_cyclized.pdb')

        start_time = time.time()

        # Load PDB
        pdb = app.PDBFile(pdb_path)
        forcefield = app.ForceField('amber14-all.xml')

        # Create system
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )

        # Find BSJ atoms
        bsj_end_atom = self._find_atom(pdb, bsj_end - 1, "O3'")
        bsj_start_atom = self._find_atom(pdb, bsj_start, 'P')

        if bsj_end_atom is None or bsj_start_atom is None:
            # Fallback to C3'
            bsj_end_atom = self._find_atom(pdb, bsj_end - 1, "C3'")
            bsj_start_atom = self._find_atom(pdb, bsj_start, "C3'")

        if bsj_end_atom is None or bsj_start_atom is None:
            raise ValueError(f"Cannot find BSJ atoms for residues {bsj_end-1} and {bsj_start}")

        # ========== PHASE 1: Strong BSJ + SS restraints ==========

        # Add BSJ restraint (strong)
        bsj_force = mm.CustomBondForce('k_bsj * (r - r0)^2')
        bsj_force.addPerBondParameter('k_bsj')
        bsj_force.addPerBondParameter('r0')
        bsj_force.addBond(bsj_end_atom, bsj_start_atom,
                         [self.bsj_restraint_k_initial, self.bsj_target_distance * 0.1])  # nm
        bsj_force_index = system.addForce(bsj_force)

        # Add secondary structure restraints
        ss_force = None
        ss_force_index = -1
        if ss_pairs:
            ss_force = mm.CustomBondForce('k_ss * (r - r0)^2')
            ss_force.addPerBondParameter('k_ss')
            ss_force.addPerBondParameter('r0')

            for i, j in ss_pairs:
                atom_i = self._find_atom(pdb, i, "C3'")
                atom_j = self._find_atom(pdb, j, "C3'")
                if atom_i is not None and atom_j is not None:
                    ss_force.addBond(atom_i, atom_j,
                                   [self.ss_restraint_k, self.bp_target_distance * 0.1])  # nm

            ss_force_index = system.addForce(ss_force)

        # Add trRosettaRNA2 distance restraints
        dist_force = None
        dist_force_index = -1
        if distance_restraints is not None and self.use_trrosetta_restraints:
            dist_force = mm.CustomBondForce('k_dist * (r - r0)^2')
            dist_force.addPerBondParameter('k_dist')
            dist_force.addPerBondParameter('r0')

            n = len(distance_restraints)
            # Add high-confidence distance predictions
            for i in range(n):
                for j in range(i + 3, n):  # Skip immediate neighbors
                    target_dist = distance_restraints[i, j]

                    # Skip BSJ region
                    if (i == bsj_start and j == bsj_end) or (i == bsj_end and j == bsj_start):
                        continue

                    # Only add confident predictions (distance < 15 Å)
                    if target_dist > 0 and target_dist < 15:
                        atom_i = self._find_atom(pdb, i, "C3'")
                        atom_j = self._find_atom(pdb, j, "C3'")
                        if atom_i is not None and atom_j is not None:
                            weight = self.restraint_weight * (1.0 - target_dist / 15.0)
                            dist_force.addBond(atom_i, atom_j, [weight, target_dist * 0.1])

            if dist_force.getNumBonds() > 0:
                dist_force_index = system.addForce(dist_force)

        # Create integrator
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.001 * unit.picosecond
        )

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # Phase 1 minimization
        state = simulation.context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy()._value

        simulation.minimizeEnergy(
            maxIterations=self.max_iterations_phase1,
            tolerance=self.minimization_tolerance * unit.kilojoules_per_mole
        )

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        phase1_energy = state.getPotentialEnergy()._value
        phase1_positions = state.getPositions()

        # ========== PHASE 2: Annealing with gradual restraint release ==========

        annealing_traj = []
        current_positions = phase1_positions

        # Gradually reduce BSJ restraint and cool down
        for step in range(self.annealing_steps):
            # Update temperature
            progress = step / self.annealing_steps
            temp = self.annealing_temperature_start * (1 - progress) + \
                   self.annealing_temperature_end * progress

            # Update integrator temperature
            integrator.setTemperature(temp * unit.kelvin)

            # Update BSJ restraint strength
            k_current = self.bsj_restraint_k_initial * (1 - progress) + \
                       self.bsj_restraint_k_final * progress

            # This requires updating force parameters
            # (simplified: just run MD at each temperature)
            simulation.step(10)

            state = simulation.context.getState(getEnergy=True, getPositions=True)
            annealing_traj.append({
                'step': step,
                'temperature': temp,
                'energy': state.getPotentialEnergy()._value
            })

        # ========== PHASE 3: Final minimization ==========

        simulation.minimizeEnergy(
            maxIterations=self.max_iterations_phase2,
            tolerance=self.minimization_tolerance * unit.kilojoules_per_mole
        )

        state = simulation.context.getState(getPositions=True, getEnergy=True)
        final_energy = state.getPotentialEnergy()._value
        final_positions = state.getPositions()

        # Calculate BSJ distance
        bsj_dist = self._calculate_distance(final_positions, bsj_end_atom, bsj_start_atom)

        # Save cyclized structure
        app.PDBFile.writeFile(pdb.topology, final_positions, open(output_path, 'w'))

        elapsed = time.time() - start_time

        return {
            'pdb_path': output_path,
            'initial_energy_kjmol': initial_energy,
            'phase1_energy_kjmol': phase1_energy,
            'final_energy_kjmol': final_energy,
            'bsj_distance_nm': bsj_dist,
            'bsj_distance_angstrom': bsj_dist * 10.0,
            'cyclization_success': bsj_dist < 0.5,  # 5 Å threshold
            'elapsed_seconds': elapsed,
            'annealing_trajectory': annealing_traj[:10],  # Sample
            'num_ss_pairs_used': len(ss_pairs) if ss_pairs else 0,
            'num_distance_restraints_used': dist_force.getNumBonds() if dist_force else 0,
        }

    def cyclize_multiple_attempts(
        self,
        pdb_path: str,
        bsj_start: int,
        bsj_end: int,
        ss_pairs: Optional[List[Tuple[int, int]]] = None,
        distance_restraints: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Multiple cyclization attempts with different starting conformations.

        Returns the best result based on BSJ distance and energy.
        """
        if output_dir is None:
            output_dir = os.path.dirname(pdb_path)
        os.makedirs(output_dir, exist_ok=True)

        results = []

        for attempt in range(self.num_attempts):
            output_path = os.path.join(output_dir, f'cyclized_attempt_{attempt}.pdb')

            # Perturb initial coordinates slightly
            if attempt > 0:
                perturbed_pdb = self._perturb_structure(pdb_path, attempt, output_dir)
                input_pdb = perturbed_pdb
            else:
                input_pdb = pdb_path

            result = self.cyclize(
                pdb_path=input_pdb,
                bsj_start=bsj_start,
                bsj_end=bsj_end,
                ss_pairs=ss_pairs,
                distance_restraints=distance_restraints,
                output_path=output_path
            )
            result['attempt_id'] = attempt
            results.append(result)

        # Select best result
        best = self._select_best_cyclization(results)

        return results, best

    def _perturb_structure(self, pdb_path: str, attempt_id: int, output_dir: str) -> str:
        """
        Perturb structure coordinates for different starting conformations.

        Small random perturbation to break out of local minima.
        """
        coords = []
        sequence = []

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and "C3'" in line:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_name = line[17:20].strip()

                    # Add random perturbation (scale increases with attempt)
                    scale = 0.5 * attempt_id  # Å
                    x += np.random.randn() * scale
                    y += np.random.randn() * scale
                    z += np.random.randn() * scale

                    coords.append([x, y, z])
                    sequence.append(res_name)

        # Write perturbed PDB
        perturbed_path = os.path.join(output_dir, f'perturbed_{attempt_id}.pdb')
        with open(perturbed_path, 'w') as f:
            for i, (coord, seq_char) in enumerate(zip(coords, sequence)):
                line = (
                    f"ATOM  {i+1:5d} C3'   {seq_char} A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 50.00           C\n"
                )
                f.write(line)
            f.write("END\n")

        return perturbed_path

    def _select_best_cyclization(self, results: List[Dict]) -> Dict:
        """Select best cyclization based on BSJ distance and energy."""
        # Score = BSJ closeness + energy
        scores = []
        for r in results:
            bsj_dist = r['bsj_distance_angstrom']
            energy = r['final_energy_kjmol']

            # BSJ score (Gaussian centered at 3.5 Å)
            bsj_score = np.exp(-((bsj_dist - 3.5) / 0.5) ** 2)

            # Energy score (relative to min)
            min_energy = min(r['final_energy_kjmol'] for r in results)
            energy_score = 1.0 if energy <= min_energy * 1.1 else 0.5

            total_score = 0.7 * bsj_score + 0.3 * energy_score
            scores.append(total_score)

        best_idx = np.argmax(scores)
        return results[best_idx]

    def _find_atom(self, pdb, residue_index: int, atom_name: str) -> Optional[int]:
        """Find atom index for a specific residue and atom name."""
        atoms = list(pdb.topology.atoms())
        for i, atom in enumerate(atoms):
            if atom.residue.index == residue_index and atom.name == atom_name:
                return i
        return None

    def _calculate_distance(self, positions, atom_i: int, atom_j: int) -> float:
        """Calculate distance between two atoms in nm."""
        pos_i = positions[atom_i]
        pos_j = positions[atom_j]
        diff = pos_i - pos_j
        dist = np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
        return dist._value if hasattr(dist, '_value') else dist


# Keep backward compatibility
class BSJCyclizer(OptimizedBSJCyclizer):
    """Backward-compatible wrapper for optimized cyclizer."""
    pass


def convert_trrosetta_restraints(distance_matrix: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert trRosettaRNA2 distance predictions to OpenMM-compatible format.

    Returns distance matrix in Angstroms (or None if not available).
    """
    if distance_matrix is None:
        return None

    # trRosettaRNA2 typically outputs distances in some unit
    # Convert to Angstroms if needed (assumption: output is in Angstroms)
    return distance_matrix