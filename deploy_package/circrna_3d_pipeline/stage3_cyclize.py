"""
Stage 3: BSJ Cyclization using OpenMM.

Connects BSJ ends to form circular topology and resolves steric clashes.
"""

import os
import numpy as np
import json

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("Warning: OpenMM not installed. Install with: pip install openmm")


class BSJCyclizer:
    """Cyclize linear RNA by connecting BSJ ends."""

    def __init__(self, config):
        self.bsj_restraint_k = config.get('bsj_restraint_k', 1000.0)
        self.bsj_target_distance = config.get('bsj_target_distance', 3.5)
        self.ss_restraint_k = config.get('ss_restraint_k', 50.0)
        self.max_iterations = config.get('max_iterations', 500)
        self.minimization_tolerance = config.get('minimization_tolerance', 10.0)

    def cyclize(self, pdb_path, bsj_start, bsj_end, ss_pairs=None, output_path=None):
        """
        Cyclize a linear RNA structure by connecting BSJ ends.

        Args:
            pdb_path: Path to linear PDB structure
            bsj_start: BSJ start index (0-based)
            bsj_end: BSJ end index (0-based)
            ss_pairs: List of (i, j) base pairs from secondary structure
            output_path: Path to save cyclized PDB

        Returns:
            dict with 'pdb_path', 'energy', 'bsj_distance'
        """
        if not HAS_OPENMM:
            raise RuntimeError("OpenMM is required for Stage 3")

        if output_path is None:
            output_path = pdb_path.replace('.pdb', '_cyclized.pdb')

        # Load PDB
        pdb = app.PDBFile(pdb_path)

        # Create force field
        forcefield = app.ForceField('amber14-all.xml')

        # Create system (no solvent for minimization)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )

        # Find BSJ atoms
        # BSJ connects: last nucleotide O3' to first nucleotide P
        bsj_end_atom = self._find_atom(pdb, bsj_end - 1, 'O3\'')
        bsj_start_atom = self._find_atom(pdb, bsj_start, 'P')

        if bsj_end_atom is None or bsj_start_atom is None:
            # Fallback: use C3' atoms
            bsj_end_atom = self._find_atom(pdb, bsj_end - 1, 'C3\'')
            bsj_start_atom = self._find_atom(pdb, bsj_start, 'C3\'')

        # Add BSJ distance restraint
        bsj_force = mm.CustomBondForce('k*(r - r0)^2')
        bsj_force.addPerBondParameter('k')
        bsj_force.addPerBondParameter('r0')
        bsj_force.addBond(bsj_end_atom, bsj_start_atom,
                         [self.bsj_restraint_k, self.bsj_target_distance * 0.1])  # nm
        system.addForce(bsj_force)

        # Add secondary structure restraints
        if ss_pairs:
            ss_force = mm.CustomBondForce('k*(r - r0)^2')
            ss_force.addPerBondParameter('k')
            ss_force.addPerBondParameter('r0')

            for i, j in ss_pairs:
                atom_i = self._find_atom(pdb, i, 'C3\'')
                atom_j = self._find_atom(pdb, j, 'C3\'')
                if atom_i is not None and atom_j is not None:
                    ss_force.addBond(atom_i, atom_j,
                                   [self.ss_restraint_k, 1.0])  # 1.0 nm typical base pair distance

            system.addForce(ss_force)

        # Create integrator and simulation
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.001 * unit.picosecond  # 1 fs for stability
        )

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True)
        initial_energy = state.getPotentialEnergy()

        # Minimize energy
        simulation.minimizeEnergy(
            maxIterations=self.max_iterations,
            tolerance=self.minimization_tolerance * unit.kilojoules_per_mole
        )

        # Get final state
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        final_energy = state.getPotentialEnergy()

        # Calculate BSJ distance
        positions = state.getPositions()
        bsj_dist = self._calculate_distance(positions, bsj_end_atom, bsj_start_atom)

        # Save cyclized structure
        app.PDBFile.writeFile(pdb.topology, positions, open(output_path, 'w'))

        return {
            'pdb_path': output_path,
            'initial_energy_kjmol': initial_energy._value,
            'final_energy_kjmol': final_energy._value,
            'bsj_distance_nm': bsj_dist,
            'bsj_distance_angstrom': bsj_dist * 10.0
        }

    def _find_atom(self, pdb, residue_index, atom_name):
        """Find atom index for a specific residue and atom name."""
        atoms = list(pdb.topology.atoms())
        for i, atom in enumerate(atoms):
            if atom.residue.index == residue_index and atom.name == atom_name:
                return i
        return None

    def _calculate_distance(self, positions, atom_i, atom_j):
        """Calculate distance between two atoms in nm."""
        pos_i = positions[atom_i]
        pos_j = positions[atom_j]
        diff = pos_i - pos_j
        return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

    def cyclize_batch(self, linear_results, ss_results, output_dir):
        """
        Cyclize multiple linear structures.

        Args:
            linear_results: list of results from Stage 2
            ss_results: list of secondary structure results from Stage 1
            output_dir: output directory

        Returns:
            list of cyclization results
        """
        all_results = []
        for i, (linear, ss) in enumerate(zip(linear_results, ss_results)):
            for sample in linear:
                pdb_path = sample['pdb_path']
                output_path = os.path.join(output_dir, f'cyclized_{i}_{sample["sample_id"]}.pdb')

                ss_pairs = None
                if ss.get('dot_bracket'):
                    from .stage1_vienna import ViennaRNAPredictor
                    ss_pairs = ViennaRNAPredictor.parse_dot_bracket(ss['dot_bracket'])

                result = self.cyclize(
                    pdb_path=pdb_path,
                    bsj_start=ss['bsj_start'],
                    bsj_end=ss['bsj_end'],
                    ss_pairs=ss_pairs,
                    output_path=output_path
                )
                result['sample_id'] = sample['sample_id']
                result['seq_id'] = i
                all_results.append(result)

        return all_results
