#!/usr/bin/env python3
"""
Check which platform OpenMM actually uses in working conditions.

Uses the same workflow as stage3_cyclize.py (addHydrogens, ignoreExternalBonds).

Usage:
  python check_platform_working.py input.pdb
"""

import sys

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_platform_working(pdb_path):
    """Check platform in actual working conditions."""

    print(f"Loading PDB: {pdb_path}")

    # Load PDB
    pdb = app.PDBFile(pdb_path)

    # Add hydrogens (same as stage3)
    print(f"Adding hydrogens...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(pH=7.0)

    # Create forcefield (same as stage3)
    print(f"Creating amber14 forcefield...")
    forcefield = app.ForceField('amber14-all.xml')

    # Create system (same as stage3)
    print(f"Creating system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        ignoreExternalBonds=True,
        constraints=app.HBonds
    )

    # Create integrator
    integrator = mm.LangevinIntegrator(
        300*unit.kelvin,
        1/unit.picosecond,
        0.001*unit.picosecond
    )

    # Create simulation WITHOUT specifying platform
    print(f"\nCreating simulation (auto-select platform)...")
    simulation = app.Simulation(modeller.topology, system, integrator)

    # Check platform
    platform = simulation.context.getPlatform()
    platform_name = platform.getName()

    print(f"\n" + "="*70)
    print(f"RESULT:")
    print(f"="*70)
    print(f"Platform used: {platform_name}")

    if platform_name == 'CUDA':
        print(f"\n✓✓✓ SUCCESS! Using CUDA (GPU acceleration)")
        print(f"GPU is being utilized for:")
        print(f"  - Energy minimization")
        print(f"  - MD simulation")
        print(f"  - Faster calculations (10-100x speedup)")
    elif platform_name == 'OpenCL':
        print(f"\n✓ Using OpenCL (GPU acceleration, may be slower)")
    elif platform_name == 'CPU':
        print(f"\n✗ WARNING: Using CPU (no GPU acceleration)")
        print(f"\nTo force GPU usage, modify stage3_cyclize.py:")
        print(f"  platform = mm.Platform.getPlatformByName('CUDA')")
        print(f"  simulation = app.Simulation(modeller.topology, system, integrator, platform)")
    else:
        print(f"\n✗ Using {platform_name} (extremely slow)")

    # Set positions to complete the test
    simulation.context.setPositions(modeller.positions)
    print(f"\n✓ Simulation context created successfully")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_platform_working.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_platform_working(sys.argv[1])