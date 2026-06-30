#!/usr/bin/env python3
"""
Check atoms in first residue to see if phosphate group is present.

Usage:
  python check_first_residue_atoms.py input.pdb
"""

import sys

try:
    import openmm.app.internal.pdbstructure as ps
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_first_residue_atoms(pdb_path):
    """Check atoms in first residue."""

    print(f"Loading PDB: {pdb_path}")

    with open(pdb_path, 'r') as f:
        pdb = ps.PdbStructure(f)

    first_res = pdb.models[0].chains[0].residues[0]
    atoms = list(first_res.iter_atoms())

    print(f"\nFirst residue: {first_res.name}")
    print(f"Total atoms: {len(atoms)}")
    print(f"\nAtom list:")

    for atom in atoms:
        print(f"  {atom.name}")

    # Check for phosphate atoms
    phosphate_atoms = ['P', 'OP1', 'OP2']
    has_phosphate = [atom.name for atom in atoms if atom.name in phosphate_atoms]

    print(f"\nPhosphate atoms found: {has_phosphate}")
    if has_phosphate:
        print(f"  ✓ First residue HAS phosphate group!")
    else:
        print(f"  ✗ First residue MISSING phosphate group!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_first_residue_atoms.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_first_residue_atoms(sys.argv[1])