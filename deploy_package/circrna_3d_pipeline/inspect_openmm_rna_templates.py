#!/usr/bin/env python3
"""
Inspect OpenMM forcefield RNA templates.

Shows what RNA residue templates are available and what atoms they expect.

Usage:
  python inspect_openmm_rna_templates.py
"""

import sys

try:
    import openmm.app as app
    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False
    print("ERROR: OpenMM not installed!")
    sys.exit(1)


def inspect_forcefield_templates():
    """Inspect RNA templates in amber14 forcefield."""

    # Load forcefield
    print("Loading amber14-all.xml forcefield...")
    forcefield = app.ForceField('amber14-all.xml')

    # Find all RNA-related templates
    print("\nAvailable RNA residue templates:")
    rna_templates = []
    for template_name in forcefield._templates:
        # RNA templates typically start with R (RA, RC, RG, RU)
        if template_name.startswith('R') or template_name in ['A', 'U', 'G', 'C']:
            rna_templates.append(template_name)
            print(f"  - {template_name}")

    # Show detailed atom list for each RNA template
    for template_name in sorted(rna_templates):
        template = forcefield._templates[template_name]
        print(f"\n{template_name} template atoms:")
        print(f"  Total atoms: {len(template.atoms)}")
        for i, atom in enumerate(template.atoms, 1):
            print(f"  {i:2d}. {atom.name:6s} (type: {atom.type})")

    # Check for terminal residue templates
    print("\n\nLooking for terminal residue templates:")
    for template_name in forcefield._templates:
        if '5' in template_name or '3' in template_name or 'term' in template_name.lower():
            print(f"  - {template_name}")

    return forcefield


if __name__ == '__main__':
    inspect_forcefield_templates()