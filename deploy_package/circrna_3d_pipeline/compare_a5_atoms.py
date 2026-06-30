#!/usr/bin/env python3
"""
Compare atoms between our PDB and amber14 A5 template.

Usage:
  python compare_a5_atoms.py
"""

import openmm.app as app

# Load amber14 forcefield
ff = app.ForceField('amber14-all.xml')

# Get A5 template
a5_template = ff._templates['A5']

print(f"A5 template atoms ({len(a5_template.atoms)}):")
a5_atoms = []
for atom in a5_template.atoms:
    a5_atoms.append(atom.name)
    print(f"  {atom.name}")

print(f"\nOur first residue atoms (33):")
our_atoms = [
    'P', 'OP2', 'OP1', 'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'',
    'C1\'', 'C2\'', 'O2\'', 'N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6',
    'N7', 'C8', 'N9', 'H5\'', 'H5\'\'', 'H4\'', 'H3\'', 'H1\'', 'H2\'',
    'HO2\'', 'H2', 'H61', 'H62', 'H8'
]

for atom in our_atoms:
    print(f"  {atom}")

print(f"\nComparison:")
print(f"  A5 template: {len(a5_atoms)} atoms")
print(f"  Our residue: {len(our_atoms)} atoms")

# Find missing atoms in our residue
missing = [a for a in a5_atoms if a not in our_atoms]
print(f"\nMissing atoms (in A5 but not in our PDB):")
for atom in missing:
    print(f"  {atom}")

# Find extra atoms in our residue
extra = [a for a in our_atoms if a not in a5_atoms]
print(f"\nExtra atoms (in our PDB but not in A5):")
for atom in extra:
    print(f"  {atom}")