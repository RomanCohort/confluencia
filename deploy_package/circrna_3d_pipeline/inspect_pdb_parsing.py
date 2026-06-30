#!/usr/bin/env python3
"""
Inspect how OpenMM parses a PDB file.

Shows residue names, chain IDs, and atom names as OpenMM sees them.

Usage:
  python inspect_pdb_parsing.py input.pdb
"""

import sys

try:
    import openmm.app as app
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("ERROR: OpenMM not installed!")
    sys.exit(1)


def inspect_pdb(pdb_path):
    """Inspect how OpenMM parses PDB file."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    print(f"\nPDB Summary:")
    print(f"  Total atoms: {pdb.topology.getNumAtoms()}")
    print(f"  Total residues: {pdb.topology.getNumResidues()}")
    print(f"  Total chains: {pdb.topology.getNumChains()}")

    # Show first few residues
    print(f"\nFirst 5 residues:")
    for i, res in enumerate(pdb.topology.residues()):
        if i < 5:
            atom_count = len(list(res.atoms()))
            print(f"  Residue {i}: name='{res.name}', chain='{res.chain.id}', atoms={atom_count}")

        if i == 0:
            print(f"\n  First residue atoms:")
            for atom in res.atoms():
                print(f"    {atom.name}")

    # Show last few residues
    residues = list(pdb.topology.residues())
    last_res = residues[-1]
    last_atom_count = len(list(last_res.atoms()))
    print(f"\nLast residue:")
    print(f"  Residue {len(residues)-1}: name='{last_res.name}', chain='{last_res.chain.id}', atoms={last_atom_count}")
    print(f"  Last residue atoms:")
    for atom in last_res.atoms():
        print(f"    {atom.name}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_pdb_parsing.py input.pdb")
        sys.exit(1)

    pdb_path = sys.argv[1]
    inspect_pdb(pdb_path)