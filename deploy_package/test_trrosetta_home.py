#!/usr/bin/env python3
"""
Test script to verify trRosettaRNA2 path resolution.
"""

import os, sys
sys.path.insert(0, 'circrna_3d_pipeline')

# Import and test
from stage2_trrosetta import trRosettaRNA2Predictor

config = {
    'model_path': './trRosettaRNA2/',
    'num_samples': 10,
    'device': 'cuda:0',
    'use_gpu': True
}

predictor = trRosettaRNA2Predictor(config)
print("="*70)
print(f"self.trrosetta_home after __init__: {predictor.trrosetta_home}")
print(f"wrapper_script would be: {os.path.join(predictor.trrosetta_home, 'predict.py')}")
print("="*70)

# Check actual file
actual_file = os.path.join(predictor.trrosetta_home, 'predict.py')
print(f"File exists: {os.path.exists(actual_file)}")
print(f"Actual location: /root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2/predict.py")
print(f"Match: {actual_file == '/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2/predict.py'}")