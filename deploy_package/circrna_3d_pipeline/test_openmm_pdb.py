#!/usr/bin/env python3
"""
Test if OpenMM can load fixed RNA PDB file.

Usage:
  python test_openmm_pdb.py model_1_fixed.pdb
"""

import sys

try:
    import openmm.app as app
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("ERROR: OpenMM not installed!")
    sys.exit(1)


def test_openmm_pdb(pdb_path):
    """Test if OpenMM can load and create system from PDB."""
    print(f"Testing OpenMM with: {pdb_path}")

    # Load PDB
    print("  Loading PDB file...")
    pdb = app.PDBFile(pdb_path)
    print(f"  ✓ PDB loaded: {pdb.topology.getNumAtoms()} atoms, {pdb.topology.getNumResidues()} residues")

    # Create forcefield
    print("  Creating forcefield...")
    forcefield = app.ForceField('amber14-all.xml')
    print("  ✓ Forcefield created")

    # Create system
    print("  Creating system...")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )
    print(f"  ✓ System created: {system.getNumParticles()} particles")

    print("\n✓✓✓ SUCCESS! OpenMM can process this PDB file.")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_openmm_pdb.py <pdb_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    success = test_openmm_pdb(pdb_path)
    sys.exit(0 if success else 1)