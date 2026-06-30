#!/usr/bin/env python3
"""
Compare atoms before and after addHydrogens().

Shows what atoms addHydrogens(pH=7.0) actually adds.

Usage:
  python check_addhydrogens_effect.py input.pdb
"""

import sys

try:
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_addhydrogens_effect(pdb_path):
    """Check what addHydrogens adds."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    # Check original first residue
    print(f"\n原始PDB第一个残基原子:")
    first_res_orig = list(pdb.topology.residues())[0]
    orig_atoms = [atom.name for atom in first_res_orig.atoms()]
    for atom in orig_atoms:
        print(f"  {atom}")
    print(f"Total: {len(orig_atoms)} atoms")

    # Apply addHydrogens
    print(f"\naddHydrogens(pH=7.0)后:")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(pH=7.0)

    first_res_mod = list(modeller.topology.residues())[0]
    mod_atoms = [atom.name for atom in first_res_mod.atoms()]
    for atom in mod_atoms:
        print(f"  {atom}")
    print(f"Total: {len(mod_atoms)} atoms")

    # Find added atoms
    added = [a for a in mod_atoms if a not in orig_atoms]
    removed = [a for a in orig_atoms if a not in mod_atoms]

    print(f"\n原子变化:")
    print(f"  Added: {added}")
    print(f"  Removed: {removed}")

    # Check HO5' specifically
    print(f"\nHO5'检查:")
    has_ho5_orig = "HO5'" in orig_atoms
    has_ho5_mod = "HO5'" in mod_atoms
    print(f"  Before: {has_ho5_orig}")
    print(f"  After: {has_ho5_mod}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_addhydrogens_effect.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_addhydrogens_effect(sys.argv[1])