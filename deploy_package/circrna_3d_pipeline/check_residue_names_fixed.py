#!/usr/bin/env python3
"""
Check residue names in fixed PDB file.

Verifies that only first and last residues have terminal names (A5, G3).

Usage:
  python check_residue_names_fixed.py input.pdb
"""

import sys

try:
    import openmm.app.internal.pdbstructure as ps
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_residue_names(pdb_path):
    """Check residue names in PDB."""

    print(f"Loading PDB: {pdb_path}")

    with open(pdb_path, 'r') as f:
        pdb = ps.PdbStructure(f)

    chain = pdb.models[0].chains[0]
    residues = list(chain.iter_residues())

    print(f"\nTotal residues: {len(residues)}")
    print(f"\n前10个残基名称:")
    for i, res in enumerate(residues):
        if i < 10:
            print(f"  Residue {i}: name='{res.name}'")

    # Check last residue
    print(f"\n最后一个残基:")
    print(f"  Residue {len(residues)-1}: name='{residues[-1].name}'")

    # Verify only first and last have terminal suffixes
    print(f"\n验证终端残基:")
    terminal_suffixes = ['5', '3']
    first_ok = residues[0].name.endswith('5') if len(residues) > 0 else False
    last_ok = residues[-1].name.endswith('3') if len(residues) > 0 else False

    print(f"  First residue terminal (ends with '5'): {first_ok}")
    print(f"  Last residue terminal (ends with '3'): {last_ok}")

    # Check middle residues
    middle_ok = True
    for i in range(1, len(residues)-1):
        if residues[i].name.endswith('5') or residues[i].name.endswith('3'):
            print(f"  ✗ Residue {i} '{residues[i].name}' has terminal suffix (should not!)")
            middle_ok = False

    if middle_ok and len(residues) > 2:
        print(f"  ✓ All middle residues have standard names (A, U, G, C)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_residue_names_fixed.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_residue_names(sys.argv[1])