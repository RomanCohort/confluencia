#!/usr/bin/env python3
"""
Direct check of PDB file columns.

Shows the exact content of columns 17-20 (residue name) as stored in the file.

Usage:
  python check_pdb_columns.py input.pdb
"""

import sys


def check_pdb_columns(pdb_path):
    """Check residue name columns in PDB file."""

    print(f"Checking PDB file: {pdb_path}")
    print("\nFirst 10 ATOM lines (columns 17-20 = residue name):")

    with open(pdb_path, 'r') as f:
        count = 0
        for line in f:
            if line.startswith('ATOM'):
                # PDB format: columns 18-20 are residue name (1-indexed)
                # In Python (0-indexed): columns 17-19
                res_name_raw = line[17:20]

                # Show full line for first few
                if count < 3:
                    print(f"\nLine {count+1}:")
                    print(f"  Full: {line.rstrip()}")
                    print(f"  Columns 17-20 (residue name): '{res_name_raw}'")
                    print(f"  Stripped: '{res_name_raw.strip()}'")
                else:
                    print(f"Line {count+1}: residue name = '{res_name_raw}' (stripped: '{res_name_raw.strip()}')")

                count += 1
                if count >= 10:
                    break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_pdb_columns.py input.pdb")
        sys.exit(1)

    pdb_path = sys.argv[1]
    check_pdb_columns(pdb_path)