#!/usr/bin/env python3
"""
Debug script to check residue names after Modeller.addHydrogens().

Shows what happens to residue names when Modeller processes the topology.

Usage:
  python debug_modeller_residues.py input.pdb
"""

import sys

try:
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def debug_modeller_residues(pdb_path):
    """Debug residue names after Modeller."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    print("\nOriginal residue names (from PDBFile):")
    for i, res in enumerate(pdb.topology.residues()):
        if i < 5:
            atom_count = len(list(res.atoms()))
            print(f"  Residue {i}: name='{res.name}', atoms={atom_count}")

    print("\nAfter Modeller.addHydrogens():")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens()

    for i, res in enumerate(modeller.topology.residues()):
        if i < 5:
            atom_count = len(list(res.atoms()))
            print(f"  Residue {i}: name='{res.name}', atoms={atom_count}")

    # Check if residue names changed
    print("\nResidue name changes:")
    orig_residues = list(pdb.topology.residues())
    new_residues = list(modeller.topology.residues())

    for i in range(min(5, len(orig_residues))):
        old_name = orig_residues[i].name
        new_name = new_residues[i].name
        if old_name != new_name:
            print(f"  Residue {i}: '{old_name}' -> '{new_name}' (CHANGED!)")
        else:
            print(f"  Residue {i}: '{old_name}' (unchanged)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_modeller_residues.py input.pdb", file=sys.stderr)
        sys.exit(1)

    debug_modeller_residues(sys.argv[1])