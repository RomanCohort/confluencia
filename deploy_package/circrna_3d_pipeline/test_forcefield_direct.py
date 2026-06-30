#!/usr/bin/env python3
"""
Test OpenMM forcefield directly with PDB file.

Shows what residue names OpenMM sees and tries to match with forcefield templates.

Usage:
  python test_forcefield_direct.py input.pdb
"""

import sys

try:
    import openmm.app as app
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("ERROR: OpenMM not installed!")
    sys.exit(1)


def test_forcefield_direct(pdb_path):
    """Test OpenMM forcefield matching directly."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    # Show what residue names OpenMM sees
    residues = list(pdb.topology.residues())
    print(f"\nOpenMM parsed residue names:")
    print(f"  Residue 0: '{residues[0].name}'")
    print(f"  Residue 106: '{residues[-1].name}'")

    # Check middle residue
    print(f"  Residue 1: '{residues[1].name}'")

    # Test forcefield
    print("\nTesting amber14-all.xml forcefield...")
    forcefield = app.ForceField('amber14-all.xml')

    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        print(f"\n✓✓✓ SUCCESS!")
        print(f"  System created with {system.getNumParticles()} particles")
        return True

    except Exception as e:
        print(f"\n✗✗✗ ERROR!")
        print(f"  {str(e)[:200]}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_forcefield_direct.py input.pdb")
        sys.exit(1)

    pdb_path = sys.argv[1]
    success = test_forcefield_direct(pdb_path)
    sys.exit(0 if success else 1)