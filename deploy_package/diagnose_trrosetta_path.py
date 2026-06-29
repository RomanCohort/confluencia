#!/usr/bin/env python3
"""
Diagnostic script to test trRosettaRNA2 path resolution.
"""

import os
import sys

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'circrna_3d_pipeline'))

from stage2_trrosetta import trRosettaRNA2Predictor

# Test config
config = {
    'model_path': './trRosettaRNA2/',
    'weights_path': './trRosettaRNA2/weights/params/',
    'num_samples': 10,
    'device': 'cuda:0',
    'use_gpu': True
}

print("=" * 70)
print("Testing trRosettaRNA2 path resolution")
print("=" * 70)

# Create predictor
predictor = trRosettaRNA2Predictor(config)

print(f"\nConfig model_path: {config['model_path']}")
print(f"Predictor.trrosetta_home: {predictor.trrosetta_home}")
print(f"Expected: {os.path.abspath('./trRosettaRNA2/')}")

# Test wrapper script path
wrapper_script = os.path.join(predictor.trrosetta_home, 'predict.py')
print(f"\nWrapper script path: {wrapper_script}")
print(f"Wrapper script exists: {os.path.exists(wrapper_script)}")

# Check actual trRosettaRNA2 location
print(f"\nActual trRosettaRNA2 location:")
print(f"  ./trRosettaRNA2/predict.py: {os.path.exists('./trRosettaRNA2/predict.py')}")
print(f"  ../trRosettaRNA2/predict.py: {os.path.exists('../trRosettaRNA2/predict.py')}")

# List all trRosettaRNA2 directories
print(f"\nAll trRosettaRNA2 directories in current path:")
for root, dirs, files in os.walk('.'):
    if 'trRosettaRNA2' in dirs:
        tr_path = os.path.join(root, 'trRosettaRNA2')
        predict_py = os.path.join(tr_path, 'predict.py')
        print(f"  {tr_path}/predict.py: {os.path.exists(predict_py)}")

print("\n" + "=" * 70)