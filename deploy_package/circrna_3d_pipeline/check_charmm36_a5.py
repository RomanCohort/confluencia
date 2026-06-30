#!/usr/bin/env python3
"""
Check CHARMM36 and amber14 A5 template atoms.

Usage:
  python check_charmm36_a5.py
"""

import openmm.app as app

print("="*70)
print("Checking CHARMM36 forcefield:")
print("="*70)

try:
    ff_charmm = app.ForceField('charmm36.xml')
    if 'A5' in ff_charmm._templates:
        a5 = ff_charmm._templates['A5']
        print(f"\nCHARMM36 A5 template atoms ({len(a5.atoms)}):")
        for i, atom in enumerate(a5.atoms, 1):
            print(f"  {i:2d}. {atom.name}")
    else:
        print("\nA5 template NOT found in CHARMM36")
except Exception as e:
    print(f"\nCHARMM36 error: {e}")

print("\n" + "="*70)
print("Checking amber14-all forcefield:")
print("="*70)

try:
    ff_amber = app.ForceField('amber14-all.xml')
    if 'A5' in ff_amber._templates:
        a5 = ff_amber._templates['A5']
        print(f"\namber14 A5 template atoms ({len(a5.atoms)}):")
        for i, atom in enumerate(a5.atoms, 1):
            print(f"  {i:2d}. {atom.name}")
    else:
        print("\nA5 template NOT found in amber14")
except Exception as e:
    print(f"\namber14 error: {e}")