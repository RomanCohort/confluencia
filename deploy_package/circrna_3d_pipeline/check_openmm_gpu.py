#!/usr/bin/env python3
"""
Check OpenMM GPU availability and current platform usage.

Usage:
  python check_openmm_gpu.py
"""

import sys

try:
    import openmm as mm
    import openmm.app as app
except ImportError:
    print("ERROR: OpenMM not installed!", file=sys.stderr)
    sys.exit(1)


def check_openmm_gpu():
    """Check OpenMM GPU availability."""

    print("="*70)
    print("OpenMM GPU Status Check")
    print("="*70)

    # Check available platforms
    print("\nAvailable OpenMM platforms:")
    platforms = []
    for i in range(mm.Platform.getNumPlatforms()):
        platform = mm.Platform.getPlatform(i)
        platforms.append(platform.getName())
        print(f"  {i+1}. {platform.getName()}")

    # Check CUDA availability
    print("\nCUDA platform status:")
    try:
        cuda_platform = mm.Platform.getPlatformByName('CUDA')
        print(f"  ✓ CUDA platform is available!")
        print(f"  Platform name: {cuda_platform.getName()}")
        print(f"  Default device: {cuda_platform.getPropertyDefaultValue('CudaDevice')}")

        # Check CUDA compiler
        cuda_compiler = cuda_platform.getPropertyDefaultValue('CudaCompiler')
        print(f"  CUDA compiler: {cuda_compiler}")

        return True
    except Exception as e:
        print(f"  ✗ CUDA platform NOT available")
        print(f"  Error: {e}")
        return False

    # Check OpenCL availability
    print("\nOpenCL platform status:")
    try:
        opencl_platform = mm.Platform.getPlatformByName('OpenCL')
        print(f"  ✓ OpenCL platform is available!")
        print(f"  Platform name: {opencl_platform.getName()}")
    except Exception as e:
        print(f"  ✗ OpenCL platform NOT available")
        print(f"  Error: {e}")

    # Recommendation
    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)

    if 'CUDA' in platforms:
        print("\n✓ CUDA detected! OpenMM should automatically use GPU.")
        print("  If not, explicitly specify platform when creating Simulation:")
        print("  platform = mm.Platform.getPlatformByName('CUDA')")
        print("  simulation = app.Simulation(topology, system, integrator, platform)")
    elif 'OpenCL' in platforms:
        print("\n⚠ OpenCL detected! Can use GPU (may be slower than CUDA).")
        print("  Explicitly specify OpenCL platform:")
        print("  platform = mm.Platform.getPlatformByName('OpenCL')")
    else:
        print("\n✗ No GPU platform detected! Will use CPU only.")
        print("  Install CUDA toolkit and OpenMM with CUDA support:")
        print("  conda install -c conda-forge openmm cudatoolkit")


if __name__ == '__main__':
    check_openmm_gpu()