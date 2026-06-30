#!/usr/bin/env python3
"""
Check second residue (U) atoms vs amber14 U template.

Usage:
  python check_second_residue.py input.pdb
"""

import sys

try:
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_second_residue(pdb_path):
    """Check second residue atoms."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    # Check second residue original
    print(f"\n第二个残基原始原子:")
    residue_1_orig = list(pdb.topology.residues())[1]
    atoms_orig = [atom.name for atom in residue_1_orig.atoms()]
    for atom in atoms_orig:
        print(f"  {atom}")
    print(f"Total: {len(atoms_orig)} atoms")

    # Apply addHydrogens
    print(f"\naddHydrogens(pH=7.0)后:")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(pH=7.0)

    residue_1_mod = list(modeller.topology.residues())[1]
    atoms_mod = [atom.name for atom in residue_1_mod.atoms()]
    for atom in atoms_mod:
        print(f"  {atom}")
    print(f"Total: {len(atoms_mod)} atoms")

    added = [a for a in atoms_mod if a not in atoms_orig]
    print(f"\n添加的原子: {added}")

    # Check amber14 U template
    print(f"\namber14 U模板原子:")
    ff = app.ForceField('amber14-all.xml')
    if 'U' in ff._templates:
        u_template = ff._templates['U']
        print(f"U template atoms ({len(u_template.atoms)}):")
        u_atoms = [atom.name for atom in u_template.atoms]
        for atom in u_atoms:
            print(f"  {atom}")

        # Find missing atoms
        missing = [a for a in u_atoms if a not in atoms_mod]
        if missing:
            print(f"\n缺失的原子: {missing}")
        else:
            print(f"\n✓ 第二个残基匹配U模板（无缺失原子）")
    else:
        print("U template not found!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_second_residue.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_second_residue(sys.argv[1])