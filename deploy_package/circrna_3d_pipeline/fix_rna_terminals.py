#!/usr/bin/env python3
"""
Fix RNA PDB residue names for OpenMM terminal templates.

Problem: OpenMM amber14 forcefield has different templates for:
- Middle residues: A, U, G, C (expect P, OP1, OP2, and O3' connected to next residue)
- 5' terminal: A5, U5, G5, C5 (expect P, OP1, OP2, but NO upstream O3')
- 3' terminal: A3, U3, G3, C3 (expect NO P, OP1, OP2, and NO downstream O3')

Solution:
- Change first residue name to X5 (5' terminal)
- Change last residue name to X3 (3' terminal)
- Remove phosphate from middle residues (already done by fix_rna_pdb_for_openmm.py)

Usage:
  python fix_rna_terminals.py input.pdb output.pdb
"""

import sys


def fix_rna_terminals(input_pdb, output_pdb):
    """Change first and last residue names to terminal templates."""

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
            res_name = line[17:20].strip()

            # First residue: change to 5' terminal
            if res_num == 1:
                if first_res_name is None:
                    first_res_name = res_name
                    print(f"First residue: {res_name} → {res_name}5")
                line = line[:17] + f"{res_name}5".ljust(3) + line[20:]

            # Last residue: change to 3' terminal
            elif res_num == max_res:
                if last_res_name is None:
                    last_res_name = res_name
                    print(f"Last residue: {res_name} → {res_name}3")
                line = line[:17] + f"{res_name}3".ljust(3) + line[20:]

            modified_lines.append(line)
        else:
            modified_lines.append(line)

    # Write modified PDB
    with open(output_pdb, 'w') as f:
        f.writelines(modified_lines)

    print(f"\nModified PDB written to: {output_pdb}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fix_rna_terminals.py input.pdb output.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]

    fix_rna_terminals(input_pdb, output_pdb)