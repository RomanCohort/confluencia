#!/usr/bin/env python3
"""
Fix RNA PDB for OpenMM by correctly formatting terminal residue names.

CRITICAL: OpenMM PDB parser checks column alignment strictly:
- Residue name: columns 18-20 (3 chars, can be 1-3 chars with spaces)
- Chain ID: column 22

Format for 'A5' (5' terminal adenine):
  Columns 18-20: 'A5 ' (left-aligned, NOT right-aligned!)
  Column 21: ' ' (space)
  Column 22: chain ID

WRONG (causes 'Misaligned residue name' error):
  Columns 18-20: ' A5' (right-aligned)
  Column 21: 'A' (chain ID in wrong position!)

Usage:
  python fix_rna_terminals_correct.py input.pdb output.pdb
"""

import sys


def fix_rna_terminals_correct(input_pdb, output_pdb):
    """Fix terminal residue names with correct column alignment."""

    # Read PDB
    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    # Get total residues
    residue_nums = set()
    for line in lines:
        if line.startswith('ATOM'):
            res_num = int(line[22:26].strip())
            residue_nums.add(res_num)

    max_res = max(residue_nums) if residue_nums else 0
    print(f"Total residues: {max_res}")

    # Modify residue names
    modified_lines = []
    first_res_name = None
    last_res_name = None

    for line in lines:
        if line.startswith('ATOM'):
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()  # Get original residue name

            # First residue: change to 5' terminal
            if res_num == 1:
                if first_res_name is None:
                    first_res_name = res_name
                    print(f"First residue: {res_name} -> {res_name}5 (5' terminal)")
                # Left-aligned: 'A5 ' in columns 18-20
                new_res_name = f"{res_name}5 ".ljust(3)
                line = line[:17] + new_res_name + line[20:]

            # Last residue: change to 3' terminal
            elif res_num == max_res:
                if last_res_name is None:
                    last_res_name = res_name
                    print(f"Last residue: {res_name} -> {res_name}3 (3' terminal)")
                # Left-aligned: 'U3 ' in columns 18-20
                new_res_name = f"{res_name}3 ".ljust(3)
                line = line[:17] + new_res_name + line[20:]

            modified_lines.append(line)
        else:
            modified_lines.append(line)

    # Write modified PDB
    with open(output_pdb, 'w') as f:
        f.writelines(modified_lines)

    print(f"\nModified PDB written to: {output_pdb}")
    print(f"Format: Left-aligned residue names in columns 18-20")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fix_rna_terminals_correct.py input.pdb output.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]

    fix_rna_terminals_correct(input_pdb, output_pdb)