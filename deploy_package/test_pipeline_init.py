#!/usr/bin/env python3
"""
Test pipeline initialization and check stage2.trrosetta_home.
"""

import sys, os
sys.path.insert(0, 'circrna_3d_pipeline')
import yaml
from pipeline import CircRNA3DPipeline

# Initialize pipeline
print("Initializing pipeline with config_quality.yaml...")
pipeline = CircRNA3DPipeline('config_quality.yaml')

# Check stage2
print("="*70)
print(f"Pipeline.stage2.trrosetta_home: {pipeline.stage2.trrosetta_home}")
print(f"Expected: /root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2")
print(f"Match: {pipeline.stage2.trrosetta_home == '/root/autodl-tmp/confluencia/confluencia/deploy_package/trRosettaRNA2'}")

# Test wrapper_script
import os
wrapper_script = os.path.join(pipeline.stage2.trrosetta_home, 'predict.py')
print(f"\nwrapper_script: {wrapper_script}")
print(f"wrapper_script exists: {os.path.exists(wrapper_script)}")
print("="*70)