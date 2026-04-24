import os
import sys
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '..'))
# Prefer workspace root so imports mirror build environment
sys.path.insert(0, workspace_root)

try:
    m = importlib.import_module('confluencia_shared')
    print('OK:', m.__file__)
except Exception as e:
    print('ERROR:', type(e).__name__, e)
    raise
