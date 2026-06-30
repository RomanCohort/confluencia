#!/usr/bin/env python3
"""
Directly inspect how OpenMM parses residue names from PDB.

Uses pdbstructure.PdbStructure to read raw residue names without
any replacement or modification.

Usage:
  python inspect_raw_residue_names.py input.pdb
"""

import sys

try:
    import openmm.app.internal.pdbstructure as ps
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def inspect_raw_residue_names(pdb_path):
    """Inspect raw residue names directly from PDB structure."""

    print(f"Loading PDB: {pdb_path}")

    with open(pdb_path, 'r') as f:
        pdb = ps.PdbStructure(f)

    # Check first few residues
    chain = pdb.models[0].chains[0]
    residues = list(chain.iter_residues())

    print("\nFirst 5 residues (raw parsing):")
    for i, res in enumerate(residues[:5]):
        atom_count = len(list(res.iter_atoms()))
        print(f"  Residue {i}: name='{res.name}', atoms={atom_count}")

    # Check last residue
    if len(residues) > 0:
        last_res = residues[-1]
        print(f"\nLast residue: name='{last_res.name}'")

    # Show what columns contain the residue name
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                cols_17_20 = line[17:20]
                cols_18_20 = line[18:21]
                print(f"\nSample ATOM line:")
                print(f"  Full line: {line.rstrip()}")
                print(f"  Columns 17-20: '{cols_17_20}'")
                print(f"  Columns 18-20: '{cols_18_20}'")
                break

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_raw_residue_names.py input.pdb", file=sys.stderr)
        sys.exit(1)

    inspect_raw_residue_names(sys.argv[1])