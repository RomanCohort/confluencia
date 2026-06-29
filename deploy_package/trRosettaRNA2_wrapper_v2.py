#!/usr/bin/env python3
"""
Wrapper script for trRosettaRNA2 predict.py
Forwards arguments to trRNA2/predict.py with proper module handling

The real predict.py uses relative imports (from .utils import *)
which fail when called as a standalone script.
This wrapper adds trRNA2 to sys.path and imports as a module.
"""

import sys
import os
from pathlib import Path

# Add trRNA2 directory to Python path
trrna2_dir = Path(__file__).parent / 'trRNA2'
sys.path.insert(0, str(trrna2_dir.parent))

# Import and run the real predict module
try:
    # Import trRNA2 as a package
    import trRNA2

    # Now we can call it properly
    # The real predict.py should be executed via python -m trRNA2.predict
    import subprocess
    cmd = [sys.executable, '-m', 'trRNA2.predict'] + sys.argv[1:]

    # Run in trRNA2 parent directory (trRosettaRNA2)
    result = subprocess.run(cmd, cwd=str(trrna2_dir.parent))
    sys.exit(result.returncode)

except ImportError as e:
    print(f"ERROR: Cannot import trRNA2 module: {e}")
    print(f"trRNA2 directory: {trrna2_dir}")
    print(f"sys.path: {sys.path}")

    # Fallback: try running as module directly
    import subprocess
    cmd = [sys.executable, '-m', 'trRNA2.predict'] + sys.argv[1:]
    print(f"Trying: {cmd}")
    result = subprocess.run(cmd, cwd=str(trrna2_dir.parent))
    sys.exit(result.returncode)