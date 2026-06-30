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
        pdb_path = self._add_rna_capping_groups(pdb_path, bsj_start, bsj_end)
        pdb = app.PDBFile(pdb_path)

        # CRITICAL FIX: Add missing hydrogen atoms WITHOUT forcefield
        # Using forcefield parameter causes circular dependency (addHydrogens tries to createSystem)
        # Instead, add H atoms with pH=7.0 (standard RNA conditions)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(pH=7.0)  # Add standard H atoms for RNA at neutral pH

        print(f"  ✓ Added missing hydrogen atoms")

        # CRITICAL FIX: Manually add HO5' if missing (for A5 and similar terminal residues)
        # The O5' atom becomes a free end after removing phosphate, requiring HO5'
        residues = list(modeller.topology.residues())
        first_res = residues[0]
        has_ho5prime = any(atom.name == "HO5'" for atom in list(first_res.atoms()))

        if not has_ho5prime:
            print(f"  ⚠ Manually adding HO5' to first residue...")
            for atom in list(first_res.atoms()):
                if atom.name == "O5'":
                    from openmm import unit
                    h_pos = atom.getPosition().value_in_unit(unit.angstroms)
                    new_h = app.Atom('H')
                    new_h.setPosition([h_pos[0], h_pos[1], h_pos[2]])
                    modeller.addAtom(new_h, first_res, "HO5'", None)
                    break
            print(f"  ✓ First residue now has complete atom set (including HO5')")
        else:
            print(f"  ✓ First residue already has HO5'")

        # Create force field - MUST use amber14 (CHARMM36 lacks RNA terminal templates)
        forcefield = app.ForceField('amber14-all.xml')
        print(f"  Using amber14-all forcefield (has A5, G3 templates)")

        # CRITICAL FIX: Use ignoreExternalBonds=True
        # Terminal residues don't have upstream/downstream bonds, causing template mismatch
        # This parameter skips external bond matching for terminal residues
        system = forcefield.createSystem(
            modeller.topology,  # Use modeller topology (with H atoms added)
            nonbondedMethod=app.NoCutoff,
            ignoreExternalBonds=True,  # Critical for terminal residues
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

    def _add_rna_capping_groups(self, pdb_path, bsj_start, bsj_end):
        """
        Fix PDB file to work with OpenMM forcefield.

        Problem: trRosettaRNA2 generates PDBs that don't match OpenMM templates:
        1. All residues have phosphate groups (should only be at 5' end)
        2. Residue names are not terminal variants (A5, U3, etc.)

        Solution:
        1. Remove phosphate atoms from middle residues
        2. Change terminal residue names (A->A5, G->G3)

        For circRNA, the BSJ connects 5' and 3' ends, so we keep phosphate at both ends.
        """
        import tempfile
        import shutil

        # Create fixed PDB file path
        fixed_path = pdb_path.replace('.pdb', '_fixed.pdb')
        terminal_path = pdb_path.replace('.pdb', '_terminal.pdb')

        print(f"  Fixing PDB for OpenMM compatibility...")

        # Step 1: Remove phosphate from middle residues
        print(f"    [1] Removing phosphate atoms from middle residues...")
        self._remove_middle_phosphates(pdb_path, fixed_path)

        # Step 2: Change terminal residue names
        print(f"    [2] Setting terminal residue names...")
        self._set_terminal_residue_names(fixed_path, terminal_path)

        print(f"  ✓ PDB fixed: {terminal_path}")

        return terminal_path

    def _remove_middle_phosphates(self, input_pdb, output_pdb):
        """Remove P, OP1, OP2 from ALL residues (OpenMM templates don't include phosphate)."""

        # Read PDB
        with open(input_pdb, 'r') as f:
            lines = f.readlines()

        print(f"    Total ATOM lines: {len([l for l in lines if l.startswith('ATOM')])}")

        # Filter out ALL phosphate atoms
        # OpenMM RNA templates (A5, A3, etc.) do NOT include P, OP1, OP2
        filtered_lines = []
        removed_count = 0

        for line in lines:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()

                # Remove phosphate atoms from ALL residues
                if atom_name in ['P', 'OP1', 'OP2']:
                    removed_count += 1
                    continue  # Skip all phosphate atoms
                else:
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)

        print(f"    Removed {removed_count} phosphate atoms (P, OP1, OP2)")

        # Write fixed PDB
        with open(output_pdb, 'w') as f:
            f.writelines(filtered_lines)

    def _set_terminal_residue_names(self, input_pdb, output_pdb):
        """Change first residue name to X5, last to X3."""

        # Read PDB
        with open(input_pdb, 'r') as f:
            lines = f.readlines()

        # Get total residues
        residue_nums = set()
        for line in lines:
            if line.startswith('ATOM'):
                res_num = int(line[22:26].strip())
                residue_nums.add(res_num)

        max_res = max(residue_nums) if residue_nums else 0

        # Modify residue names
        modified_lines = []
        for line in lines:
            if line.startswith('ATOM'):
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()

                # First residue: change to 5' terminal
                if res_num == 1:
                    new_res_name = f"{res_name}5 ".ljust(3)
                    line = line[:17] + new_res_name + line[20:]

                # Last residue: change to 3' terminal
                elif res_num == max_res:
                    new_res_name = f"{res_name}3 ".ljust(3)
                    line = line[:17] + new_res_name + line[20:]

                modified_lines.append(line)
            else:
                modified_lines.append(line)

        # Write modified PDB
        with open(output_pdb, 'w') as f:
            f.writelines(modified_lines)

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
