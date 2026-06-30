#!/usr/bin/env python3
"""
Check if amber14 forcefield has A5/G3 terminal RNA templates.

Also checks residue name replacement rules in OpenMM.

Usage:
  python check_amber14_terminal_templates.py
"""

import sys

try:
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_terminal_templates():
    """Check if amber14 has terminal RNA templates."""

    print("Loading amber14-all.xml forcefield...")
    ff = app.ForceField('amber14-all.xml')

    # Check for terminal RNA templates
    terminal_templates = ['A5', 'U5', 'G5', 'C5', 'A3', 'U3', 'G3', 'C3']
    middle_templates = ['A', 'U', 'G', 'C']

    print("\nChecking for RNA terminal templates:")
    for tmpl in terminal_templates:
        if tmpl in ff._templates:
            template = ff._templates[tmpl]
            print(f"  ✓ {tmpl} exists ({len(template.atoms)} atoms)")
        else:
            print(f"  ✗ {tmpl} NOT found")

    print("\nChecking for middle RNA templates:")
    for tmpl in middle_templates:
        if tmpl in ff._templates:
            template = ff._templates[tmpl]
            print(f"  ✓ {tmpl} exists ({len(template.atoms)} atoms)")

    # Check residue name replacements
    print(f"\nResidue name replacements:")
    print(f"  Total: {len(app.PDBFile._residueNameReplacements)}")

    if len(app.PDBFile._residueNameReplacements) > 0:
        print("  Sample replacements:")
        for i, (old, new) in enumerate(app.PDBFile._residueNameReplacements.items()):
            if i < 10:
                print(f"    '{old}' -> '{new}'")


if __name__ == '__main__':
    check_terminal_templates()