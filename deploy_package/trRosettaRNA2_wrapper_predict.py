#!/usr/bin/env python3
"""
Wrapper script for trRosettaRNA2 predict.py
Forwards arguments to trRNA2/predict.py

This wrapper is needed because trRosettaRNA2 official repo
does not have predict.py in top-level directory.
Real predict.py is in trRNA2/predict.py
"""

import sys
import os
import subprocess
from pathlib import Path

# 真正的predict.py位置
trrna2_dir = Path(__file__).parent / 'trRNA2'
real_predict = trrna2_dir / 'predict.py'

if not real_predict.exists():
    print(f"ERROR: predict.py not found at {real_predict}")
    print(f"trRosettaRNA2 directory structure:")
    for item in Path(__file__).parent.iterdir():
        print(f"  {item}")
    sys.exit(1)

# 转发所有参数到真正的predict.py
cmd = [sys.executable, str(real_predict)] + sys.argv[1:]

# 在trRNA2目录中执行（保持相对路径正确）
result = subprocess.run(cmd, cwd=str(trrna2_dir))
sys.exit(result.returncode)