#!/usr/bin/env python3
"""
Test different residue name formats to see what OpenMM accepts.

Tests various formats in columns 17-20 to find which one OpenMM
will recognize as 'A5' instead of 'A'.

Usage:
  python test_residue_formats.py
"""

import tempfile
import os

try:
    import openmm.app as app
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("ERROR: OpenMM not installed!")
    exit(1)


def test_residue_formats():
    """Test different residue name formats."""

    # Different formats to test
    formats = [
        ('A5',  'A5 in cols 18-19 (2 chars)'),
        ('A5 ', 'A5 in cols 18-20 (left-aligned, 3 chars)'),
        (' A5', 'A5 in cols 17-19 (right-aligned)'),
        ('5A ', '5A in cols 18-20 (digit first)'),
    ]

    # Simple PDB content for testing
    pdb_template = """ATOM      1  P   {res} A   1      0.0    0.0    0.0  1.00  0.00           P
ATOM      2  OP1 {res} A   1      1.0    0.0    0.0  1.00  0.00           O
ATOM      3  OP2 {res} A   1      0.0    1.0    0.0  1.00  0.00           O
END
"""

    print("Testing different residue name formats:")
    print("="*60)

    for res_format, desc in formats:
        pdb_content = pdb_template.format(res=res_format)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name

        try:
            # Parse with OpenMM
            pdb = app.PDBFile(temp_path)
            res_name = pdb.topology.residues()[0].name

            # Show actual column content
            with open(temp_path, 'r') as f:
                line = f.readline()
                cols_17_20 = line[17:20]

            print(f"\nFormat: '{res_format}' ({desc})")
            print(f"  Columns 17-20 in file: '{cols_17_20}'")
            print(f"  OpenMM reads as: '{res_name}'")

            # Check if it matches expected
            if res_name == 'A5':
                print(f"  ✓ SUCCESS! OpenMM recognized as 'A5'")
            else:
                print(f"  ✗ Failed. OpenMM read as '{res_name}' instead of 'A5'")

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    test_residue_formats()