#!/usr/bin/env python3
"""
Test OpenMM with ignoreExternalBonds=True to handle terminal residues.

OpenMM requires external bonds to match residue templates.
For 5' terminal RNA, there's no upstream O3' bond, causing template mismatch.

Solution: Use ignoreExternalBonds=True to skip external bond matching.

Usage:
  python test_ignore_external_bonds.py input.pdb
"""

import sys

try:
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def test_ignore_external_bonds(pdb_path):
    """Test forcefield with ignoreExternalBonds=True."""

    print(f"Loading PDB: {pdb_path}")
    pdb = app.PDBFile(pdb_path)

    print("\nStandard forcefield matching (ignoreExternalBonds=False):")
    forcefield = app.ForceField('amber14-all.xml')
    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            ignoreExternalBonds=False
        )
        print(f"  ✓ SUCCESS! System: {system.getNumParticles()} particles")
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:200]}")

    print("\nWith ignoreExternalBonds=True:")
    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            ignoreExternalBonds=True
        )
        print(f"  ✓✓✓ SUCCESS! System: {system.getNumParticles()} particles")
        print(f"  This bypasses external bond matching for terminal residues!")
        return True
    except Exception as e:
        print(f"  ✗✗✗ ERROR: {str(e)[:200]}")
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_ignore_external_bonds.py input.pdb", file=sys.stderr)
        sys.exit(1)

    success = test_ignore_external_bonds(sys.argv[1])
    sys.exit(0 if success else 1)