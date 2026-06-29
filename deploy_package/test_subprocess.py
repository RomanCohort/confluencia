#!/usr/bin/env python3
"""
Test subprocess.run behavior with absolute paths.
"""

import os
import sys
import subprocess

# Test the exact same code as stage2_trrosetta.py
wrapper_script = "/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2/predict.py"

print("="*70)
print(f"Testing subprocess.run with wrapper_script:")
print(f"  wrapper_script: {wrapper_script}")
print(f"  File exists: {os.path.exists(wrapper_script)}")
print("="*70)

# Build cmd exactly as in stage2_trrosetta.py
cmd = [
    sys.executable, wrapper_script,
    '-h'  # Just test help to see if it can be executed
]

print(f"\nCommand: {' '.join(cmd)}")
print(f"cmd[0] (sys.executable): {cmd[0]}")
print(f"cmd[1] (wrapper_script): {cmd[1]}")

# Try to run
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    print(f"\n✓ subprocess.run succeeded!")
    print(f"Return code: {result.returncode}")
    if result.returncode == 0:
        print("✓ predict.py -h executed successfully!")
except Exception as e:
    print(f"\n✗ subprocess.run failed: {e}")