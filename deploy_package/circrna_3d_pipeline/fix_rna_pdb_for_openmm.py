#!/usr/bin/env python3
"""
Fix RNA PDB file for OpenMM.

Problem: trRosettaRNA2 generates complete RNA structure with phosphate groups
(P, OP1, OP2) for every residue. OpenMM forcefield expects:
- Only 5' end residue to have phosphate group
- Middle residues without phosphate group
- 3' end residue without phosphate group (for linear RNA)

For circRNA, both 5' and 3' ends will be connected at BSJ, so we keep
phosphate at residue 1 and last residue.

Usage:
  python fix_rna_pdb_for_openmm.py input.pdb output.pdb
"""

import sys
import os


def fix_rna_pdb(input_pdb, output_pdb):
    """Remove phosphate atoms from middle residues."""

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

    # Filter out P, OP1, OP2 from middle residues
    filtered_lines = []
    removed_count = 0

    for line in lines:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            res_num = int(line[22:26].strip())

            # Keep phosphate atoms only for first and last residue
            if atom_name in ['P', 'OP1', 'OP2']:
                if res_num == 1 or res_num == max_res:
                    filtered_lines.append(line)
                else:
                    # Skip phosphate atoms for middle residues
                    removed_count += 1
                    continue
            else:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    # Write fixed PDB
    with open(output_pdb, 'w') as f:
        f.writelines(filtered_lines)

    print(f"Fixed PDB written to: {output_pdb}")
    print(f"Removed {removed_count} phosphate atoms from middle residues")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python fix_rna_pdb_for_openmm.py input.pdb output.pdb")
        sys.exit(1)

    input_pdb = sys.argv[1]
    output_pdb = sys.argv[2]

    fix_rna_pdb(input_pdb, output_pdb)