#!/usr/bin/env python3
"""
Check which OpenMM platform is actually used when creating Simulation.

Tests with a real PDB file to see if CUDA is automatically selected.

Usage:
  python check_actual_platform.py input.pdb
"""

import sys

try:
    import openmm as mm
    import openmm.app as app
    from openmm import unit
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_actual_platform(pdb_path):
    """Check which platform OpenMM actually uses."""

    print(f"Loading PDB: {pdb_path}")

    # Load small portion of PDB for quick test
    pdb = app.PDBFile(pdb_path)

    print(f"\nCreating simulation WITHOUT specifying platform...")
    print(f"(OpenMM should auto-select best platform)")

    # Create minimal system for testing
    forcefield = app.ForceField('amber14-all.xml')

    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            ignoreExternalBonds=True
        )

        integrator = mm.LangevinIntegrator(
            300*unit.kelvin,
            1/unit.picosecond,
            0.001*unit.picosecond
        )

        # Create simulation (no platform specified)
        simulation = app.Simulation(pdb.topology, system, integrator)

        # Check platform
        platform = simulation.context.getPlatform()
        platform_name = platform.getName()
        platform_index = platform.getIndex()

        print(f"\nActual platform used:")
        print(f"  Name: {platform_name}")
        print(f"  Index: {platform_index}")

        # Interpret result
        print(f"\nResult:")
        if platform_name == 'CUDA':
            print(f"  ✓✓✓ Using CUDA (GPU acceleration)!")
            print(f"  GPU is being utilized for calculations!")
        elif platform_name == 'OpenCL':
            print(f"  ✓ Using OpenCL (GPU acceleration, may be slower than CUDA)")
        elif platform_name == 'CPU':
            print(f"  ✗ Using CPU (no GPU acceleration)")
            print(f"  To enable GPU, explicitly specify CUDA platform:")
            print(f"  platform = mm.Platform.getPlatformByName('CUDA')")
            print(f"  simulation = app.Simulation(topology, system, integrator, platform)")
        else:
            print(f"  ✗ Using Reference platform (extremely slow, debugging only)")

    except Exception as e:
        print(f"\nError creating system: {e}")
        print(f"This may be due to template matching (expected at this stage)")
        print(f"But platform selection logic is the same for actual runs")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_actual_platform.py input.pdb", file=sys.stderr)
        sys.exit(1)

    check_actual_platform(sys.argv[1])